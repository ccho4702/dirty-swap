"""Check a single draft intervention's cache invariants on an allocated device.

This is an integration check, not a QA benchmark. Resume replays an unfinished
check; a completed result is verified and returned without loading either model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from dirty_swapping.backend import TransformersBackend
from dirty_swapping.config import load_spec, runtime_spec
from dirty_swapping.core import atomic_json, digest, file_digest, now, source_fingerprint
from dirty_swapping.draft import draft_model_config, generate_draft
from dirty_swapping.engine import intervene
from dirty_swapping.protocol import SwapPlan
from dirty_swapping.runner import runtime_identity, selected_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/draft-swap-dev4.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    spec = (
        json.loads((args.resume / "plan.json").read_text())["config"]
        if args.resume
        else runtime_spec(load_spec(args.config), work, device=args.device)
    )
    if (
        spec["generation"]["alternative_selection"] != "draft_swap"
        or spec["generation"].get("branch_policy") != "reasoning_step"
    ):
        parser.error("this check requires draft_swap with reasoning_step branching")
    row = selected_rows(work, spec, ["gsm8k"], 1, "development")[0]
    row = {key: value for key, value in row.items() if key != "gold"}
    identity = {
        "config": spec,
        "row_sha256": digest(row),
        "source": source_fingerprint(),
        "script_sha256": file_digest(Path(__file__)),
        "runtime": runtime_identity(),
    }
    if args.resume:
        run = args.resume.resolve()
        if json.loads((run / "plan.json").read_text()) != identity:
            raise ValueError("verification source, data, configuration or runtime changed")
        if (run / "result.json").exists():
            saved = json.loads((run / "result.json").read_text())
            if saved["sha256"] != digest(saved["payload"]):
                raise ValueError("corrupt verification checkpoint")
            print(json.dumps(saved["payload"], indent=2))
            return
    else:
        name = args.run_name or time.strftime("draft-cache-check-%Y%m%dT%H%M%SZ", time.gmtime())
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
            raise ValueError("invalid run name")
        run = work / "outputs" / name
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "plan.json", identity)
    started = time.monotonic()

    def progress(stage, label):
        elapsed = time.monotonic() - started
        print(
            f"{label}; {stage}/5; elapsed={elapsed:.1f}s ETA={elapsed / stage * (5 - stage):.1f}s"
            if stage
            else f"{label}; elapsed=0s ETA pending",
            flush=True,
        )

    atomic_json(run / "status.json", {"state": "running", "at": now()})
    try:
        progress(0, "load frozen models")
        backend = TransformersBackend(spec["model"])
        drafter = TransformersBackend(draft_model_config(spec))
        progress(1, "models ready")
        state = backend.prefill(row)
        backend.advance(state, spec["generation"]["branch_after_reasoning_tokens"])
        while (
            len(state.ids) - state.prompt_length <= spec["generation"]["branch_scan_end"]
            and not backend.finished(state)
            and backend.reasoning_end(state) is None
            and backend.reasoning_step_boundary(state) is None
        ):
            backend.advance(state, 1)
        if (
            len(state.ids) - state.prompt_length > spec["generation"]["branch_scan_end"]
            or backend.finished(state)
            or backend.reasoning_end(state) is not None
            or backend.reasoning_step_boundary(state) is None
        ):
            raise ValueError("development example has no eligible boundary")
        draft = generate_draft(
            drafter,
            backend,
            row["question"],
            spec["generation"]["draft"],
            spec["generation"]["rollout_tokens"],
        )
        if draft["prefix"] is None:
            raise ValueError(f"no usable draft: {draft['reason']}")
        progress(2, "draft ready")
        plan = SwapPlan(
            len(state.ids), draft["rollout_tokens"], spec["generation"]["delay_tokens"], 2
        )
        ordinary = intervene(backend, backend.fork(state), plan, enabled=False)
        edited = intervene(
            backend,
            backend.fork(state),
            plan,
            enabled=True,
            selection="guided_swap",
            alternative_prefix=draft["prefix"],
        )
        assert edited.swapped, edited.reason
        assert ordinary.state.ids == edited.state.ids, "existing token IDs changed"
        assert backend.torch.equal(ordinary.state.next_logits, edited.state.next_logits), (
            "stale logits changed"
        )
        end = plan.branch_position + plan.rollout_tokens
        retained = []
        for before, after in zip(
            ordinary.state.cache.layers, edited.state.cache.layers, strict=True
        ):
            assert backend.torch.equal(
                before.keys[:, :, : plan.branch_position], after.keys[:, :, : plan.branch_position]
            )
            assert backend.torch.equal(
                before.values[:, :, : plan.branch_position],
                after.values[:, :, : plan.branch_position],
            )
            assert backend.torch.equal(before.keys[:, :, end:], after.keys[:, :, end:])
            assert backend.torch.equal(before.values[:, :, end:], after.values[:, :, end:])
            assert not (
                backend.torch.equal(
                    before.keys[:, :, plan.branch_position : end],
                    after.keys[:, :, plan.branch_position : end],
                )
                and backend.torch.equal(
                    before.values[:, :, plan.branch_position : end],
                    after.values[:, :, plan.branch_position : end],
                )
            )
            retained.append((after.keys[:, :, end:].clone(), after.values[:, :, end:].clone()))
        progress(3, "all layer prefix/suffix checks passed")
        old_ids = list(edited.state.ids)
        backend.advance(ordinary.state, 1)
        backend.advance(edited.state, 1)
        assert ordinary.state.ids[-1] == edited.state.ids[-1], (
            "first future token must use stale logits"
        )
        prompt_ids = backend.torch.tensor([state.ids[: state.prompt_length]], device=backend.device)
        native = backend.model.generate(
            input_ids=prompt_ids,
            attention_mask=backend.torch.ones_like(prompt_ids),
            do_sample=False,
            max_new_tokens=len(ordinary.state.ids) - state.prompt_length,
            pad_token_id=backend.tokenizer.pad_token_id or min(backend.eos_ids),
        )
        assert native[0].tolist() == ordinary.state.ids, "plain generate differs across branch"
        backend.advance(edited.state, 8)
        assert edited.state.ids[: len(old_ids)] == old_ids
        for layer, (keys, values) in zip(edited.state.cache.layers, retained, strict=True):
            assert backend.torch.equal(layer.keys[:, :, end : plan.swap_after_position], keys)
            assert backend.torch.equal(layer.values[:, :, end : plan.swap_after_position], values)
        progress(4, "retained suffix also unchanged after future generation")
        assert all(
            not model.training
            and all(not p.requires_grad and p.grad is None for p in model.parameters())
            for model in (backend.model, drafter.model)
        )
        result = {
            "passed": True,
            "id": row["id"],
            "layers": len(retained),
            "rollout_tokens": plan.rollout_tokens,
            "delay_tokens": plan.delay_tokens,
            "device": str(backend.device),
            "models_frozen": True,
            "existing_ids_preserved": True,
            "prefix_kv_preserved": True,
            "suffix_kv_preserved": True,
            "all_layer_spans_changed": True,
            "stale_first_future_token_preserved": True,
            "native_generate_matches_baseline": True,
            "at": now(),
        }
        atomic_json(run / "result.json", {"payload": result, "sha256": digest(result)})
        atomic_json(run / "status.json", {"state": "complete", "at": now()})
        progress(5, "verification complete")
        print(json.dumps(result, indent=2))
    except BaseException as error:
        atomic_json(
            run / "status.json",
            {
                "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "at": now(),
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()

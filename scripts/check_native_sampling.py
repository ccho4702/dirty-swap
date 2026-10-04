"""Single-case native sampling diagnostic; this does not perform a KV intervention.

The config selects one already-inspected development case. An interrupted run is
replayed from its prompt/seed; completed checks are verified without loading a model.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import time
from pathlib import Path

from dirty_swapping.backend import DecodeState, TransformersBackend
from dirty_swapping.config import load_spec, runtime_spec
from dirty_swapping.core import atomic_json, digest, file_digest, now, source_fingerprint
from dirty_swapping.math_scoring import math_correct, natural_answer
from dirty_swapping.prompt import instruction
from dirty_swapping.runner import _graceful_interrupts, runtime_identity, selected_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/native-sampling-diagnostic.json")
    )
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--run-name", default="native-sampling-diagnostic")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    saved = json.loads((args.resume / "plan.json").read_text()) if args.resume else None
    settings = saved["settings"] if saved else json.loads(args.config.read_text())
    spec = (
        saved["config"] if saved else runtime_spec(load_spec(work / settings["base_config"]), work)
    )
    sampling = settings["sampling"]
    if set(sampling) != {"temperature", "top_p", "top_k"}:
        raise ValueError("sampling requires temperature, top_p, and top_k")
    if type(sampling["top_k"]) is not int or sampling["top_k"] < 1:
        raise ValueError("top_k must be a positive integer")
    for key in ("temperature", "top_p"):
        if (
            type(sampling[key]) not in (int, float)
            or not math.isfinite(sampling[key])
            or sampling[key] <= 0
        ):
            raise ValueError(f"invalid {key}")
    if sampling["top_p"] > 1:
        raise ValueError("top_p must not exceed one")
    row = selected_rows(work, spec, [settings["dataset"]], 1, settings["split"])[0]
    identity = {
        "settings": settings,
        "config": spec,
        "row_sha256": digest(row),
        "source": source_fingerprint(),
        "script_sha256": file_digest(Path(__file__)),
        "lock_sha256": file_digest(work / "uv.lock"),
        "runtime": runtime_identity(),
    }
    if saved is not None:
        if saved != identity:
            raise ValueError("diagnostic inputs, source, configuration, or runtime changed")
        run = args.resume.resolve()
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
            raise ValueError("invalid run name")
        run = work / "outputs" / args.run_name
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "plan.json", identity)
    log_dir = work / "logs" / run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    with (
        (run / ".lock").open("a") as lock,
        (log_dir / "run.log").open("a", buffering=1) as log_file,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def log(message):
            line = f"{now()} {message}"
            print(line, flush=True)
            log_file.write(line + "\n")

        log("resume/start single-case native sampling diagnostic")
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            if digest(result["payload"]) != result["sha256"]:
                raise ValueError("corrupt diagnostic checkpoint")
            atomic_json(run / "status.json", {"state": "complete", "at": now()})
            log(f"verified completed result: correct={result['payload']['correct']}")
            return
        atomic_json(run / "status.json", {"state": "running", "at": now()})
        try:
            with _graceful_interrupts():
                from transformers import StoppingCriteria, StoppingCriteriaList

                log("loading frozen model; ETA pending")
                backend = TransformersBackend(spec["model"])
                seed = int(digest([spec["generation"]["seed"], row["id"]])[:8], 16)
                backend.torch.manual_seed(seed)
                backend.begin_measurement()
                backend.synchronize()
                started = time.monotonic()
                ids = backend.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction(row) + row["question"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=spec["model"].get("enable_thinking", True),
                )
                if not ids or len(ids) > spec["generation"]["max_input_tokens"]:
                    raise ValueError("prompt exceeds configured limit")
                cap = spec["generation"]["max_new_tokens"]

                class Progress(StoppingCriteria):
                    last = started

                    def __call__(self, input_ids, scores, **kwargs):
                        current = time.monotonic()
                        if current - self.last >= spec["execution"]["progress_interval_seconds"]:
                            n = input_ids.shape[-1] - len(ids)
                            elapsed = current - started
                            log(
                                f"{n}/{cap} generated; elapsed={elapsed:.0f}s ETA(cap)={elapsed / max(n, 1) * (cap - n):.0f}s"
                            )
                            self.last = current
                        return False

                inputs = backend.torch.tensor([ids], device=backend.device)
                log(f"generate {row['id']}; elapsed=0s ETA pending")
                with backend.torch.inference_mode():
                    output = backend.model.generate(
                        input_ids=inputs,
                        attention_mask=backend.torch.ones_like(inputs),
                        do_sample=True,
                        **sampling,
                        max_new_tokens=cap,
                        pad_token_id=backend.tokenizer.pad_token_id or min(backend.eos_ids),
                        stopping_criteria=StoppingCriteriaList([Progress()]),
                    )
                backend.synchronize()
                latency = time.monotonic() - started
                state = DecodeState(output[0].tolist(), len(ids), None, None)
                answer = backend.decode_answer(state)
                prediction = natural_answer(answer, row) if answer else None
                result = {
                    "id": row["id"],
                    "seed": seed,
                    "sampling": sampling,
                    "correct": bool(
                        math_correct(row["gold"], prediction)
                        if row["answer_format"] == "math"
                        else prediction == row["gold"]
                    ),
                    "prediction": prediction,
                    "answer_extracted": prediction is not None,
                    "thinking_complete": backend.reasoning_end(state) is not None,
                    "eos_ended": backend.finished(state),
                    "generated_tokens": len(state.ids) - len(ids),
                    "latency_s": latency,
                    "peak_cuda_memory_mib": backend.peak_memory_mib(),
                    "generated_text": backend.tokenizer.decode(
                        state.ids[len(ids) :], skip_special_tokens=False
                    ),
                    "diagnostic_only": True,
                    "kv_swap": False,
                    "at": now(),
                }
                atomic_json(run / "result.json", {"payload": result, "sha256": digest(result)})
                atomic_json(run / "status.json", {"state": "complete", "at": now()})
                log(
                    f"complete: correct={result['correct']}; tokens={result['generated_tokens']}; latency={latency:.1f}s"
                )
        except BaseException as error:
            log(f"stopped: {type(error).__name__}: {error}")
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

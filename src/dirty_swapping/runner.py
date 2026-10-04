"""Paired, resumable evaluation with one atomic result per example and arm."""

from __future__ import annotations

import importlib.metadata
import fcntl
import json
import platform
import re
import signal
import statistics
import subprocess
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .backend import TransformersBackend
from .config import validate_spec
from .core import atomic_json, atomic_text, digest, file_digest, now, source_fingerprint
from .data import load_prepared
from .engine import intervene
from .math_scoring import math_correct, natural_answer
from .protocol import SwapPlan

ARMS = ("baseline", "swap")


def runtime_identity() -> dict:
    import torch

    packages = {}
    for package in ("torch", "transformers", "huggingface-hub", "math-verify", "pyarrow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    driver = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu_driver_versions": driver,
        "visible_gpus": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }


def selected_rows(
    work_dir: Path, spec: dict, datasets: list[str], limit: int | None, split: str = "evaluation"
) -> list[dict]:
    if split not in ("development", "evaluation"):
        raise ValueError("unknown split")
    rows = []
    for task in datasets:
        prepared = [
            row
            for row in load_prepared(work_dir, spec, task)
            if row.get("split", "evaluation") == split
        ]
        requested = spec.get("execution", {}).get("case_ids", {}).get(task)
        if requested is not None:
            lookup = {row["id"]: row for row in prepared}
            missing = set(requested) - set(lookup)
            if missing:
                raise ValueError(f"requested IDs absent from {task}/{split}: {sorted(missing)}")
            prepared = [lookup[case_id] for case_id in requested]
        if limit is not None:
            prepared = prepared[:limit]
        rows.extend(prepared)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty evaluation cohort or duplicate IDs")
    return rows


def create_run(
    work_dir: Path,
    spec: dict,
    datasets: list[str],
    *,
    limit: int | None = None,
    run_name: str | None = None,
    split: str = "evaluation",
) -> Path:
    validate_spec(spec)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("select distinct datasets")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows = selected_rows(work_dir, spec, datasets, limit, split)
    name = run_name or time.strftime("run-%Y%m%dT%H%M%SZ-", time.gmtime()) + uuid.uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
        raise ValueError("run name must be a simple directory name")
    run = work_dir / "outputs" / name
    run.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": now(),
        "config": spec,
        "datasets": datasets,
        "limit": limit,
        "split": split,
        "ids": [row["id"] for row in rows],
        "row_hashes": {row["id"]: digest(row) for row in rows},
        "source": source_fingerprint(),
        "runtime": runtime_identity(),
        "lock_sha256": (
            file_digest(Path(__file__).resolve().parents[2] / "uv.lock")
            if (Path(__file__).resolve().parents[2] / "uv.lock").exists()
            else None
        ),
    }
    atomic_json(run / "plan.json", plan)
    atomic_json(run / "status.json", {"state": "prepared", "at": now()})
    return run


def _result_path(run: Path, arm: str, row: dict) -> Path:
    return run / "cases" / arm / (digest(row["id"]) + ".json")


def _verified_result(path: Path, *, arm: str, row: dict) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("sha256") != digest(value.get("payload")):
        raise ValueError(f"corrupt case checkpoint: {path}")
    result = value["payload"]
    if result["id"] != row["id"] or result["task"] != row["task"] or result["arm"] != arm:
        raise ValueError("case checkpoint identity mismatch")
    return result


@contextmanager
def _heartbeat(label: str, log, interval: int, progress=None):
    done = threading.Event()
    started = time.monotonic()

    def announce():
        while not done.wait(interval):
            elapsed = time.monotonic() - started
            if progress is not None:
                completed, target = progress()
                if completed > 0:
                    eta = max(0, target - completed) * elapsed / completed
                    log(
                        f"{label}; {completed}/{target} forward tokens (cap); elapsed={elapsed:.0f}s ETA(cap)={eta:.0f}s"
                    )
                    continue
            log(f"{label}; elapsed={elapsed:.0f}s; ETA pending first completion")

    thread = threading.Thread(target=announce, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


def run_case(
    backend: TransformersBackend, row: dict, spec: dict, arm: str, *, drafter=None
) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    gen = spec["generation"]
    case_seed = int(digest([gen["seed"], row["id"]])[:8], 16)
    if hasattr(backend, "torch"):
        backend.torch.manual_seed(case_seed)
    if hasattr(backend, "begin_measurement"):
        backend.begin_measurement()
    if drafter is not None and hasattr(drafter, "begin_measurement"):
        drafter.begin_measurement()
    backend.synchronize()
    started = time.monotonic()
    state = backend.prefill(row)
    if state.prompt_length > gen["max_input_tokens"]:
        raise ValueError("prompt exceeds configured input cap")
    branch_after = gen["branch_after_reasoning_tokens"]
    policy = gen.get("branch_policy", "fixed")
    branch_position = state.prompt_length + branch_after if policy == "fixed" else None
    swap_event = {
        "swapped": False,
        "reason": "reasoning_ended_before_branch" if policy == "fixed" else f"no_{policy}",
    }
    branched = False
    scanned = 0
    while len(state.ids) - state.prompt_length < gen["max_new_tokens"]:
        if backend.finished(state):
            break
        generated = len(state.ids) - state.prompt_length
        trigger = None
        if not branched and backend.reasoning_end(state) is None:
            if policy == "fixed" and generated == branch_after:
                trigger = {"policy": "fixed"}
            elif (
                policy == "numeric_ambiguity"
                and branch_after <= generated <= gen["branch_scan_end"]
            ):
                scanned += 1
                trigger = backend.numeric_ambiguity(state, gen["min_top2_ratio"])
            elif policy == "reasoning_step" and branch_after <= generated <= gen["branch_scan_end"]:
                scanned += 1
                trigger = backend.reasoning_step_boundary(state)
        if trigger is not None:
            branched = True
            branch_position = len(state.ids)
            probe_mode = gen["alternative_selection"] == "probe_preference"
            draft_mode = gen["alternative_selection"] == "draft_swap"
            draft = None
            actual_rollout = gen["rollout_tokens"]
            alternative_prefix = gen.get("alternative_prefix")
            if arm == "swap" and draft_mode:
                if drafter is None:
                    raise ValueError("draft swap requires its configured draft backend")
                from .draft import generate_draft

                draft = generate_draft(
                    drafter, backend, row["question"], gen["draft"], actual_rollout
                )
                if draft["prefix"] is None:
                    swap_event = {
                        "swapped": False,
                        "reason": draft["reason"],
                        "draft": draft,
                        "draft_s": draft["seconds"],
                    }
                    backend.advance(state, 1)
                    continue
                actual_rollout = draft["rollout_tokens"]
                alternative_prefix = draft["prefix"]
            probe_budget = gen["probe"]["tokens"] if probe_mode and arm == "swap" else 0
            if (
                generated + actual_rollout + gen["delay_tokens"] + probe_budget
                <= gen["max_new_tokens"]
            ):
                plan = SwapPlan(
                    branch_position=branch_position,
                    rollout_tokens=actual_rollout,
                    delay_tokens=gen["delay_tokens"],
                    candidate_count=gen["candidate_count"],
                )
                judge = None
                if probe_mode and arm == "swap":
                    from .preference import ProbePreferenceJudge

                    judge = ProbePreferenceJudge(backend, gen["probe"])
                event = intervene(
                    backend,
                    state,
                    plan,
                    enabled=arm == "swap",
                    selection=(
                        "guided_swap" if arm == "swap" else "second_highest_first_token_probability"
                    )
                    if draft_mode
                    else gen["alternative_selection"],
                    probe_config=gen.get("probe"),
                    question=row["question"],
                    judge=judge,
                    alternative_prefix=alternative_prefix,
                )
                state = event.state
                swap_event = {
                    "swapped": event.swapped,
                    "reason": "draft_swapped" if draft_mode and event.swapped else event.reason,
                    "main_token": event.main_token,
                    "alternative_token": event.alternative_token,
                    "alternative_rollout_s": event.alternative_rollout_s,
                    "swap_copy_s": event.swap_copy_s,
                    "candidate_count_evaluated": event.candidate_count_evaluated,
                    "probe_s": event.probe_s,
                    "judge_s": event.judge_s,
                    "judgment": event.judgment,
                    "trigger": trigger,
                    "candidate_generation": gen["alternative_selection"],
                    "rollout_tokens": actual_rollout,
                    "draft": draft,
                    "draft_s": draft["seconds"] if draft is not None else 0.0,
                    "alternative_text": (
                        backend.tokenizer.decode(event.alternative_ids, skip_special_tokens=False)
                        if event.alternative_ids is not None
                        else None
                    ),
                }
                continue
            swap_event = {"swapped": False, "reason": "insufficient_output_budget"}
        backend.advance(state, 1)
    backend.synchronize()
    latency = time.monotonic() - started
    answer_text = backend.decode_answer(state)
    prediction = natural_answer(answer_text, row) if answer_text else None
    correct = (
        math_correct(row["gold"], prediction)
        if row["answer_format"] == "math"
        else prediction == row["gold"]
    )
    return {
        "id": row["id"],
        "task": row["task"],
        "arm": arm,
        "seed": case_seed,
        "prediction": prediction,
        "correct": bool(correct),
        "answer_extracted": prediction is not None,
        "thinking_complete": backend.reasoning_end(state) is not None,
        "eos_ended": backend.finished(state),
        "answer_text": answer_text,
        "generated_text": backend.tokenizer.decode(
            state.ids[state.prompt_length :], skip_special_tokens=False
        ),
        "prompt_tokens": state.prompt_length,
        "generated_tokens": len(state.ids) - state.prompt_length,
        "latency_s": latency,
        "branch_position": branch_position,
        "branch_scan_tokens": scanned,
        "peak_cuda_memory_mib": (
            backend.peak_memory_mib() if hasattr(backend, "peak_memory_mib") else None
        ),
        "draft_peak_cuda_memory_mib": (
            drafter.peak_memory_mib()
            if drafter is not None and hasattr(drafter, "peak_memory_mib")
            else None
        ),
        "swap": swap_event,
        "finished_at": now(),
    }


def execute(run: Path, work_dir: Path, *, backend_factory=TransformersBackend) -> dict:
    """Execute or resume a run; a file lock prevents two controllers racing."""
    with _graceful_interrupts(), (run / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_locked(run, work_dir, backend_factory=backend_factory)


@contextmanager
def _graceful_interrupts():
    """Ignore duplicate forwarded signals until atomic status cleanup is done."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {item: signal.getsignal(item) for item in signals}
    received = False

    def interrupt(signum, frame):
        nonlocal received
        if received:
            return
        received = True
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")

    for item in signals:
        signal.signal(item, interrupt)
    try:
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def _execute_locked(run: Path, work_dir: Path, *, backend_factory) -> dict:
    run = run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    if plan["source"] != source_fingerprint() or plan["runtime"] != runtime_identity():
        raise ValueError("source or runtime changed; resume rejected")
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    if plan["lock_sha256"] != (file_digest(lock) if lock.exists() else None):
        raise ValueError("dependency lock changed; resume rejected")
    spec = plan["config"]
    rows = selected_rows(work_dir, spec, plan["datasets"], plan["limit"], plan["split"])
    if [row["id"] for row in rows] != plan["ids"] or {
        row["id"]: digest(row) for row in rows
    } != plan["row_hashes"]:
        raise ValueError("prepared cohort changed; resume rejected")

    log_path = work_dir / "logs" / run.name / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:

        def log(message):
            line = f"{now()} {message}"
            print(line, flush=True)
            log_file.write(line + "\n")

        completed = 0
        total = len(rows) * len(ARMS)
        for row in rows:
            for arm in ARMS:
                if _verified_result(_result_path(run, arm, row), arm=arm, row=row) is not None:
                    completed += 1
        log(f"resume/start {completed}/{total} completed")
        if completed == total:
            result = report(run)
            atomic_json(
                run / "status.json",
                {"state": "complete", "at": now(), "completed": completed, "total": total},
            )
            return result
        atomic_json(
            run / "status.json",
            {"state": "running", "at": now(), "completed": completed, "total": total},
        )
        try:
            with _heartbeat("loading model", log, spec["execution"]["progress_interval_seconds"]):
                backend = backend_factory(spec["model"])
                drafter = None
                if spec["generation"]["alternative_selection"] == "draft_swap":
                    from .draft import draft_model_config

                    drafter = backend_factory(draft_model_config(spec))
            started = time.monotonic()
            pending_at_start = total - completed
            for row in rows:
                for arm in ARMS:
                    path = _result_path(run, arm, row)
                    if _verified_result(path, arm=arm, row=row) is not None:
                        continue
                    log(f"case {completed + 1}/{total} {row['id']} {arm}")
                    before_steps = getattr(backend, "forward_steps", 0)
                    before_draft_steps = getattr(drafter, "forward_steps", 0)
                    target_steps = (
                        min(row["prompt_tokens"], spec["generation"]["max_input_tokens"])
                        + spec["generation"]["max_new_tokens"]
                        + (spec["generation"]["candidate_count"] - 1 if arm == "swap" else 0)
                        * spec["generation"]["rollout_tokens"]
                    )
                    if (
                        arm == "swap"
                        and spec["generation"]["alternative_selection"] == "probe_preference"
                    ):
                        probe = spec["generation"]["probe"]
                        target_steps += probe["tokens"] + 2 * (
                            probe["judge_input_cap"] + probe["judge_reasoning_tokens"] + 16
                        )
                    if drafter is not None and arm == "swap":
                        draft_config = spec["generation"]["draft"]
                        target_steps += (
                            draft_config["max_input_tokens"] + draft_config["max_new_tokens"]
                        )

                    def progress():
                        return (
                            getattr(backend, "forward_steps", 0)
                            - before_steps
                            + getattr(drafter, "forward_steps", 0)
                            - before_draft_steps,
                            target_steps,
                        )

                    with _heartbeat(
                        f"processing {row['id']} {arm}",
                        log,
                        spec["execution"]["progress_interval_seconds"],
                        progress,
                    ):
                        result = (
                            run_case(backend, row, spec, arm, drafter=drafter)
                            if drafter is not None
                            else run_case(backend, row, spec, arm)
                        )
                    atomic_json(path, {"payload": result, "sha256": digest(result)})
                    completed += 1
                    newly_completed = pending_at_start - (total - completed)
                    eta = (time.monotonic() - started) / newly_completed * (total - completed)
                    log(
                        f"complete {completed}/{total}; ETA={eta:.0f}s; latency={result['latency_s']:.1f}s"
                    )
                    atomic_json(
                        run / "status.json",
                        {"state": "running", "at": now(), "completed": completed, "total": total},
                    )
        except BaseException as exc:
            log(f"run stopped after {completed}/{total}: {type(exc).__name__}: {exc}")
            atomic_json(
                run / "status.json",
                {
                    "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "at": now(),
                    "completed": completed,
                    "total": total,
                    "error": str(exc),
                },
            )
            raise
        result = report(run)
        atomic_json(
            run / "status.json",
            {"state": "complete", "at": now(), "completed": completed, "total": total},
        )
        return result


def report(run: Path) -> dict:
    plan = json.loads((run / "plan.json").read_text())
    all_results = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in (run / "cases" / arm).glob("*.json"):
            value = json.loads(path.read_text())
            if value.get("sha256") != digest(value.get("payload")):
                raise ValueError(f"corrupt result: {path}")
            row = value["payload"]
            if row["id"] in all_results[arm]:
                raise ValueError("duplicate result ID")
            all_results[arm][row["id"]] = row
    tasks = plan["datasets"]
    if any(set(all_results[arm]) - set(plan["ids"]) for arm in ARMS):
        raise ValueError("result outside frozen cohort")
    summary = {
        "at": now(),
        "complete": all(len(all_results[arm]) == len(plan["ids"]) for arm in ARMS),
        "by_task": {},
        "common_completed": 0,
    }
    common = set(all_results["baseline"]) & set(all_results["swap"])
    summary["common_completed"] = len(common)
    for task in tasks:
        ids = [
            case_id
            for case_id in plan["ids"]
            if case_id in common and all_results["baseline"][case_id]["task"] == task
        ]
        arm_stats = {}
        for arm in ARMS:
            cases = [all_results[arm][case_id] for case_id in ids]
            latencies = sorted(row["latency_s"] for row in cases)
            arm_stats[arm] = {
                "correct": sum(row["correct"] for row in cases),
                "accuracy": sum(row["correct"] for row in cases) / len(cases) if cases else None,
                "answer_extracted": sum(
                    row.get("answer_extracted", row["prediction"] is not None) for row in cases
                ),
                "thinking_complete": sum(row.get("thinking_complete", False) for row in cases),
                "eos_ended": sum(row.get("eos_ended", False) for row in cases),
                "swap_reasons": dict(Counter(row["swap"]["reason"] for row in cases)),
                "mean_latency_s": statistics.mean(latencies) if cases else None,
                "median_latency_s": statistics.median(latencies) if cases else None,
                "p95_latency_s": (
                    latencies[max(0, (95 * len(cases) + 99) // 100 - 1)] if cases else None
                ),
                "sum_latency_s": sum(latencies),
                "max_peak_cuda_memory_mib": max(
                    (
                        row["peak_cuda_memory_mib"]
                        for row in cases
                        if row.get("peak_cuda_memory_mib") is not None
                    ),
                    default=None,
                ),
                "mean_alternative_rollout_s": (
                    statistics.mean(row["swap"].get("alternative_rollout_s", 0) for row in cases)
                    if cases
                    else None
                ),
                "mean_swap_copy_s": (
                    statistics.mean(row["swap"].get("swap_copy_s", 0) for row in cases)
                    if cases
                    else None
                ),
                "mean_probe_s": statistics.mean(row["swap"].get("probe_s", 0) for row in cases)
                if cases
                else None,
                "mean_judge_s": statistics.mean(row["swap"].get("judge_s", 0) for row in cases)
                if cases
                else None,
                "mean_draft_s": statistics.mean(row["swap"].get("draft_s", 0) for row in cases)
                if cases
                else None,
                "swap_rate": (
                    sum(row["swap"]["swapped"] for row in cases) / len(cases) if cases else None
                ),
            }
        summary["by_task"][task] = {
            "n": len(ids),
            "arms": arm_stats,
            "delta_accuracy_pp": (
                100 * (arm_stats["swap"]["correct"] - arm_stats["baseline"]["correct"]) / len(ids)
                if ids
                else None
            ),
            "wins": sum(
                all_results["swap"][case_id]["correct"]
                and not all_results["baseline"][case_id]["correct"]
                for case_id in ids
            ),
            "losses": sum(
                all_results["baseline"][case_id]["correct"]
                and not all_results["swap"][case_id]["correct"]
                for case_id in ids
            ),
        }
    summary["default_suite_complete"] = (
        summary["complete"]
        and plan["limit"] is None
        and plan["split"] == "evaluation"
        and not plan["config"]["execution"].get("case_ids")
        and set(tasks) == set(plan["config"]["data"]["default_datasets"])
    )
    if summary["complete"]:
        summary["selected_macro_accuracy"] = {
            arm: sum(summary["by_task"][task]["arms"][arm]["accuracy"] for task in tasks)
            / len(tasks)
            for arm in ARMS
        }
    summary["primary_macro_accuracy"] = (
        summary.get("selected_macro_accuracy") if summary["default_suite_complete"] else None
    )
    atomic_json(run / "report.json", summary)
    lines = [
        "# Dirty Swapping report",
        "",
        f"Completed: {summary['complete']}",
        f"Default suite complete: {summary['default_suite_complete']}",
        "",
        "| Dataset | Paired n | Baseline accuracy | Swap accuracy | Delta pp | Baseline answers | Swap answers | Baseline mean s | Swap mean s | Swap rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["by_task"].items():
        baseline = item["arms"]["baseline"]["accuracy"]
        swap = item["arms"]["swap"]["accuracy"]
        lines.append(
            f"| {task} | {item['n']} | {baseline:.3f} | {swap:.3f} | {item['delta_accuracy_pp']:+.2f} | "
            f"{item['arms']['baseline']['answer_extracted']} | {item['arms']['swap']['answer_extracted']} | "
            f"{item['arms']['baseline']['mean_latency_s']:.2f} | {item['arms']['swap']['mean_latency_s']:.2f} | "
            f"{item['arms']['swap']['swap_rate']:.2f} |"
            if item["n"]
            else f"| {task} | 0 | pending | pending | pending | pending | pending | pending | pending | pending |"
        )
    atomic_text(run / "report.md", "\n".join(lines) + "\n")
    return summary

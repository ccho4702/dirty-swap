"""Small public scorer; organizer should pin a stronger math verifier."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def normalize_answer(value: object, task: str) -> str:
    answer = str(value).strip()
    if task in {"aime25", "hmmt25", "gsm8k"}:
        if task == "gsm8k" and "####" in answer:
            answer = answer.rsplit("####", 1)[1].strip()
        answer = answer.replace(",", "").replace("$", "")
        if re.fullmatch(r"[+-]?\d+(?:\.0+)?", answer):
            return str(int(answer.split(".", 1)[0]))
        return answer.casefold()
    if task == "gpqa":
        match = re.fullmatch(r"(?:\(?)([A-Da-d])(?:\)?)", answer)
        return match.group(1).upper() if match else answer.casefold()
    raise ValueError(f"unsupported task: {task}")


def score(gold_rows: list[dict], predictions: list[dict]) -> dict:
    if not gold_rows:
        raise ValueError("gold file is empty")
    gold_by_id = {row["id"]: row for row in gold_rows}
    pred_by_id = {row["id"]: row for row in predictions}
    if len(gold_by_id) != len(gold_rows) or len(pred_by_id) != len(predictions):
        raise ValueError("duplicate ids")
    if set(gold_by_id) != set(pred_by_id):
        raise ValueError("prediction ids must exactly match gold ids")

    by_task: dict[str, dict] = {}
    latencies: list[float] = []
    swaps = 0
    for case_id, gold in gold_by_id.items():
        prediction = pred_by_id[case_id]
        task = gold["task"]
        latency = prediction["latency_s"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError(f"{case_id}: latency_s must be a finite nonnegative number")
        if not isinstance(prediction.get("swap_applied"), bool):
            raise ValueError(f"{case_id}: swap_applied must be a boolean")
        correct = normalize_answer(prediction["answer"], task) == normalize_answer(
            gold["answer"], task
        )
        bucket = by_task.setdefault(task, {"correct": 0, "total": 0})
        bucket["correct"] += int(correct)
        bucket["total"] += 1
        latencies.append(float(latency))
        swaps += int(prediction["swap_applied"])
    for bucket in by_task.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
    ordered = sorted(latencies)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "by_task": by_task,
        "macro_accuracy": statistics.mean(item["accuracy"] for item in by_task.values()),
        "micro_accuracy": sum(item["correct"] for item in by_task.values()) / len(gold_rows),
        "mean_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": ordered[p95_index],
        "swap_rate": swaps / len(gold_rows),
    }

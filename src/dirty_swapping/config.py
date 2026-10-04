"""Load and validate the single packaged experiment configuration."""

from __future__ import annotations

import copy
import json
import re
import math
from importlib.resources import files
from pathlib import Path


def load_spec(path: Path | None = None) -> dict:
    source = (
        Path(path).read_text()
        if path
        else files("dirty_swapping").joinpath("default.json").read_text()
    )
    return validate_spec(json.loads(source))


def runtime_spec(
    spec: dict, work_dir: Path, *, device: str | None = None, model_cache: Path | None = None
) -> dict:
    result = copy.deepcopy(spec)
    result["model"]["cache_dir"] = str((model_cache or work_dir / "models").expanduser().resolve())
    if device is not None:
        result["model"]["device"] = device
        if device == "cpu":
            result["model"]["dtype"] = "float32"
    return validate_spec(result)


def validate_spec(spec: dict) -> dict:
    if spec.get("protocol") != "dirty-swapping-v1":
        raise ValueError("unsupported protocol")
    model, gen, data = spec["model"], spec["generation"], spec["data"]
    policy = gen.get("branch_policy", "fixed")
    if policy not in ("fixed", "numeric_ambiguity", "reasoning_step"):
        raise ValueError("unsupported branch policy")
    for name, value in (
        ("model.threads", model["threads"]),
        ("model.prefill_chunk_tokens", model["prefill_chunk_tokens"]),
        ("model.max_context_tokens", model["max_context_tokens"]),
        ("generation.max_input_tokens", gen["max_input_tokens"]),
        ("generation.max_new_tokens", gen["max_new_tokens"]),
        ("generation.candidate_count", gen["candidate_count"]),
        ("generation.rollout_tokens", gen["rollout_tokens"]),
        ("generation.branch_after_reasoning_tokens", gen["branch_after_reasoning_tokens"]),
        ("data.prompt_limit", data["prompt_limit"]),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        gen["candidate_count"] < 2
        or type(gen["max_swaps_per_trace"]) is not int
        or gen["max_swaps_per_trace"] != 1
    ):
        raise ValueError("v1 requires at least two candidates and exactly one swap maximum")
    if type(gen["delay_tokens"]) is not int or gen["delay_tokens"] < 0:
        raise ValueError("delay_tokens must be a nonnegative integer")
    if gen["alternative_selection"] not in (
        "second_highest_first_token_probability",
        "probe_preference",
        "guided_swap",
    ):
        raise ValueError("unsupported alternative selector")
    if gen["alternative_selection"] == "guided_swap":
        if gen["candidate_count"] != 2:
            raise ValueError("guided swap requires exactly one alternative")
        if (
            not isinstance(gen.get("alternative_prefix"), str)
            or not gen["alternative_prefix"].strip()
        ):
            raise ValueError("guided swap requires a nonempty alternative_prefix")
    elif "alternative_prefix" in gen:
        raise ValueError("alternative_prefix requires guided_swap")
    probe_budget = 0
    if gen["alternative_selection"] == "probe_preference":
        if gen["candidate_count"] != 2:
            raise ValueError("the experimental probe selector compares top-2 only")
        probe = gen["probe"]
        for key in ("tokens", "history_tokens", "judge_input_cap", "judge_reasoning_tokens"):
            if type(probe[key]) is not int or probe[key] < 1:
                raise ValueError(f"probe.{key} must be a positive integer")
        if probe["tokens"] < 2:
            raise ValueError("probe must include tokens after the stale first logits")
        if (
            type(probe["min_margin"]) not in (int, float)
            or not math.isfinite(probe["min_margin"])
            or not 0 < probe["min_margin"] < 1
        ):
            raise ValueError("probe.min_margin must be between zero and one")
        if (
            probe["judge_input_cap"] + probe["judge_reasoning_tokens"] + 16
            > model["max_context_tokens"]
        ):
            raise ValueError("judge prompt and reasoning exceed model context")
        probe_budget = probe["tokens"]
    branch_last = gen["branch_after_reasoning_tokens"]
    if policy in ("numeric_ambiguity", "reasoning_step"):
        branch_last = gen["branch_scan_end"]
        if type(branch_last) is not int or branch_last < gen["branch_after_reasoning_tokens"]:
            raise ValueError("branch_scan_end must be at least the first scan token")
    if policy == "numeric_ambiguity":
        ratio = gen["min_top2_ratio"]
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("min_top2_ratio must be between zero and one")
    if data["prompt_limit"] > gen["max_input_tokens"]:
        raise ValueError("prompt_limit exceeds max_input_tokens")
    if gen["max_input_tokens"] + gen["max_new_tokens"] > model["max_context_tokens"]:
        raise ValueError("input and output caps exceed model context")
    if (
        branch_last + gen["rollout_tokens"] + gen["delay_tokens"] + probe_budget
        > gen["max_new_tokens"]
    ):
        raise ValueError("swap schedule exceeds max_new_tokens")
    if model["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("unsupported model dtype")
    if (
        type(model["gpu_memory_fraction"]) not in (int, float)
        or not 0 < model["gpu_memory_fraction"] <= 1
    ):
        raise ValueError("gpu_memory_fraction must be in (0, 1]")
    if model.get("enable_thinking") is not True:
        raise ValueError("Qwen thinking mode must be enabled")
    if not re.fullmatch(r"cpu|cuda:\d+", model["device"]):
        raise ValueError("device must be cpu or cuda:<index>")
    if not model["name"] or not model["revision"] or type(model["local_files_only"]) is not bool:
        raise ValueError("model identity or local_files_only is invalid")
    if type(gen["seed"]) is not int or gen["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        type(spec["execution"]["progress_interval_seconds"]) is not int
        or spec["execution"]["progress_interval_seconds"] < 1
    ):
        raise ValueError("progress_interval_seconds must be positive")
    if not data["default_datasets"] or len(set(data["default_datasets"])) != len(
        data["default_datasets"]
    ):
        raise ValueError("default_datasets must be unique and nonempty")
    from .data import sources

    if set(data["default_datasets"]) - set(sources()):
        raise ValueError("unknown default dataset")
    cohorts = spec["execution"].get("case_ids", {})
    if not isinstance(cohorts, dict) or set(cohorts) - set(sources()):
        raise ValueError("case_ids must map supported datasets to IDs")
    for task, ids in cohorts.items():
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"case_ids.{task} must contain unique nonempty IDs")
    return spec

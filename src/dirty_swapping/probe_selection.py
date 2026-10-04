"""Compare actual edited-cache futures without rewriting the existing suffix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .cache import swap_kv_segment


@dataclass
class ProbeSelection:
    state: Any
    swapped: bool
    reason: str
    probe_seconds: float
    copy_seconds: float
    judgment: dict


def choose_probe(
    backend, main, alternative_cache, plan, question: str, config: dict, judge
) -> ProbeSelection:
    """Commit the selected trial's future tokens verbatim, including no-swap."""
    backend.synchronize()
    started = time.monotonic()
    original_trial = backend.fork(main)
    backend.advance(original_trial, config["tokens"])
    alternative_trial = backend.fork(main)
    backend.synchronize()
    copy_started = time.monotonic()
    swap_kv_segment(
        backend.cache(alternative_trial),
        alternative_cache,
        plan.branch_position,
        plan.rollout_tokens,
    )
    backend.synchronize()
    copy_seconds = time.monotonic() - copy_started
    backend.advance(alternative_trial, config["tokens"])
    backend.synchronize()
    probe_seconds = time.monotonic() - started
    end = backend.sequence_length(main)
    original_text = backend.tokenizer.decode(original_trial.ids[end:], skip_special_tokens=False)
    alternative_text = backend.tokenizer.decode(
        alternative_trial.ids[end:], skip_special_tokens=False
    )
    trace = {
        "original_probe": original_text,
        "alternative_probe": alternative_text,
        "original_probe_tokens": len(original_trial.ids[end:]),
        "alternative_probe_tokens": len(alternative_trial.ids[end:]),
    }
    if original_trial.ids[end:] == alternative_trial.ids[end:]:
        return ProbeSelection(
            original_trial, False, "identical_probes", probe_seconds, copy_seconds, trace
        )
    history_start = max(main.prompt_length, end - config["history_tokens"])
    history = backend.tokenizer.decode(main.ids[history_start:end], skip_special_tokens=False)
    judgment = {
        **judge.compare(question, history, original_text, alternative_text),
        **trace,
        "history": history,
    }
    if judgment["accepted"]:
        return ProbeSelection(
            alternative_trial, True, "probe_selected", probe_seconds, copy_seconds, judgment
        )
    return ProbeSelection(
        original_trial, False, "probe_rejected", probe_seconds, copy_seconds, judgment
    )

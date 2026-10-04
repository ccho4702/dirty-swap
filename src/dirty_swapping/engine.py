"""One-branch intervention driver; a model adapter supplies token generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from .cache import swap_kv_segment
from .protocol import SwapPlan


class GenerationAdapter(Protocol):
    """Implement with the same frozen model and decoding rules for both arms."""

    def fork(self, state: Any) -> Any: ...
    def sequence_length(self, state: Any) -> int: ...
    def top_tokens(self, state: Any, count: int) -> list[int]: ...
    def rollout(self, state: Any, first_token: int, total_tokens: int) -> Any: ...
    def guided_rollout(self, state: Any, prefix: str, total_tokens: int) -> Any: ...
    def advance(self, state: Any, token_count: int) -> Any: ...
    def cache(self, state: Any) -> Any: ...
    def reasoning_end(self, state: Any) -> int | None: ...
    def finished(self, state: Any) -> bool: ...
    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class InterventionResult:
    state: Any
    swapped: bool
    reason: str
    main_token: int
    alternative_token: int | None
    alternative_rollout_s: float = 0.0
    swap_copy_s: float = 0.0
    candidate_count_evaluated: int = 1
    probe_s: float = 0.0
    judge_s: float = 0.0
    judgment: dict | None = None
    alternative_ids: list[int] | None = None


def intervene(
    adapter: GenerationAdapter,
    prefix_state: Any,
    plan: SwapPlan,
    *,
    enabled: bool,
    selection: str = "second_highest_first_token_probability",
    probe_config: dict | None = None,
    question: str = "",
    judge=None,
    alternative_prefix: str | None = None,
) -> InterventionResult:
    """Commit top-1 rollout, then optionally transplant top-2 KV after a delay.

    The returned state's token history remains the original top-1 path. The
    alternative cache is computed from an independent copy of the same prefix.
    The caller must time prefill, this function, and final decoding together.
    """
    if adapter.sequence_length(prefix_state) != plan.branch_position:
        raise ValueError("prefix length must equal absolute branch position")
    if selection not in (
        "second_highest_first_token_probability",
        "probe_preference",
        "guided_swap",
    ):
        raise ValueError("unknown alternative selector")
    if selection == "guided_swap" and (not alternative_prefix or plan.candidate_count != 2):
        raise ValueError("guided swap requires one alternative and its prefix")
    if enabled and selection == "probe_preference" and (probe_config is None or judge is None):
        raise ValueError("probe selection requires configuration and a judge")
    expected = plan.candidate_count if enabled else 1
    top = adapter.top_tokens(prefix_state, expected)
    if len(top) < expected or len(set(top)) != len(top):
        raise ValueError("adapter returned too few distinct candidate tokens")

    main_state = adapter.fork(prefix_state) if enabled else prefix_state
    main = adapter.rollout(main_state, top[0], plan.rollout_tokens)
    if adapter.sequence_length(main) != plan.branch_position + plan.rollout_tokens:
        return InterventionResult(main, False, "main_ended_early", top[0], None)
    if adapter.finished(main):
        return InterventionResult(main, False, "main_ended_in_rollout", top[0], None)
    if adapter.reasoning_end(main) is not None:
        return InterventionResult(main, False, "reasoning_ended_in_rollout", top[0], None)

    alternative = None
    short_alternative = False
    selected_alternative_left_reasoning = False
    alternative_rollout_s = 0.0
    alternative_token = top[1] if enabled else None
    if enabled:
        adapter.synchronize()
        started = time.monotonic()
        for index, token in enumerate(top[1:expected], 1):
            if selection == "guided_swap":
                candidate = adapter.guided_rollout(
                    adapter.fork(prefix_state), alternative_prefix, plan.rollout_tokens
                )
                alternative_token = candidate.ids[plan.branch_position]
            else:
                candidate = adapter.rollout(adapter.fork(prefix_state), token, plan.rollout_tokens)
            if adapter.sequence_length(candidate) != adapter.sequence_length(main):
                short_alternative = True
            if index == 1:
                alternative = candidate
                selected_alternative_left_reasoning = (
                    adapter.finished(candidate) or adapter.reasoning_end(candidate) is not None
                )
        adapter.synchronize()
        alternative_rollout_s = time.monotonic() - started

    main = adapter.advance(main, plan.delay_tokens)
    if adapter.sequence_length(main) != plan.swap_after_position:
        return InterventionResult(
            main,
            False,
            "generation_ended_in_delay",
            top[0],
            alternative_token,
            alternative_rollout_s,
            0.0,
            expected,
        )
    if not enabled:
        return InterventionResult(main, False, "baseline", top[0], None)
    if short_alternative:
        return InterventionResult(
            main,
            False,
            "alternative_ended_early",
            top[0],
            alternative_token,
            alternative_rollout_s,
            0.0,
            expected,
        )
    if selected_alternative_left_reasoning:
        return InterventionResult(
            main,
            False,
            "alternative_left_reasoning",
            top[0],
            alternative_token,
            alternative_rollout_s,
            0.0,
            expected,
        )
    reasoning_end = adapter.reasoning_end(main)
    if reasoning_end is not None and plan.swap_after_position >= reasoning_end:
        return InterventionResult(
            main,
            False,
            "reasoning_ended",
            top[0],
            alternative_token,
            alternative_rollout_s,
            0.0,
            expected,
        )

    if selection == "probe_preference":
        from .probe_selection import choose_probe

        chosen = choose_probe(
            adapter, main, adapter.cache(alternative), plan, question, probe_config, judge
        )
        return InterventionResult(
            chosen.state,
            chosen.swapped,
            chosen.reason,
            top[0],
            alternative_token,
            alternative_rollout_s,
            chosen.copy_seconds,
            expected,
            probe_s=chosen.probe_seconds,
            judge_s=chosen.judgment.get("seconds", 0.0),
            judgment=chosen.judgment,
        )

    adapter.synchronize()
    copy_started = time.monotonic()
    swap_kv_segment(
        adapter.cache(main), adapter.cache(alternative), plan.branch_position, plan.rollout_tokens
    )
    adapter.synchronize()
    return InterventionResult(
        main,
        True,
        "guided_swapped" if selection == "guided_swap" else "swapped",
        top[0],
        alternative_token,
        alternative_rollout_s,
        time.monotonic() - copy_started,
        expected,
        alternative_ids=(
            alternative.ids[plan.branch_position :] if selection == "guided_swap" else None
        ),
    )

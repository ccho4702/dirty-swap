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


def intervene(
    adapter: GenerationAdapter,
    prefix_state: Any,
    plan: SwapPlan,
    *,
    enabled: bool,
) -> InterventionResult:
    """Commit top-1 rollout, then optionally transplant top-2 KV after a delay.

    The returned state's token history remains the original top-1 path. The
    alternative cache is computed from an independent copy of the same prefix.
    The caller must time prefill, this function, and final decoding together.
    """
    if adapter.sequence_length(prefix_state) != plan.branch_position:
        raise ValueError("prefix length must equal absolute branch position")
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
    if enabled:
        adapter.synchronize()
        started = time.monotonic()
        for index, token in enumerate(top[1:expected], 1):
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
            top[1] if enabled else None,
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
            top[1],
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
            top[1],
            alternative_rollout_s,
            0.0,
            expected,
        )
    reasoning_end = adapter.reasoning_end(main)
    if reasoning_end is not None and plan.swap_after_position >= reasoning_end:
        return InterventionResult(
            main, False, "reasoning_ended", top[0], top[1], alternative_rollout_s, 0.0, expected
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
        "swapped",
        top[0],
        top[1],
        alternative_rollout_s,
        time.monotonic() - copy_started,
        expected,
    )

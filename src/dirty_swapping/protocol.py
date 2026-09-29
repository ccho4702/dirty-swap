"""Model-independent intervention schedule for a single QA trace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwapPlan:
    branch_position: int  # absolute token index, including prompt tokens
    rollout_tokens: int
    delay_tokens: int
    candidate_count: int

    def __post_init__(self) -> None:
        if self.branch_position < 0 or self.rollout_tokens < 1:
            raise ValueError("invalid branch position or rollout length")
        if self.delay_tokens < 0 or self.candidate_count < 2:
            raise ValueError("invalid delay or candidate count")

    @property
    def swap_after_position(self) -> int:
        return self.branch_position + self.rollout_tokens + self.delay_tokens

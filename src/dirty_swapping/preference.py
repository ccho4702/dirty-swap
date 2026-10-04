"""Frozen, order-checked self-judgment for math continuation probes.

Conditional A/B/T probabilities are a preference heuristic, not calibrated
correctness probabilities. The scorer never receives a reference answer.
"""

from __future__ import annotations

import time

from .backend import DecodeState


def preference_decision(forward: dict, reverse: dict, margin: float) -> dict:
    """Accept the alternative only if both option orders beat original and tie."""
    forward_margin = forward["B"] - max(forward["A"], forward["T"])
    reverse_margin = reverse["A"] - max(reverse["B"], reverse["T"])
    minimum = min(forward_margin, reverse_margin)
    return {
        "accepted": minimum >= margin,
        "min_margin": minimum,
        "forward": forward,
        "reverse": reverse,
    }


class ProbePreferenceJudge:
    def __init__(self, backend, config: dict):
        self.backend = backend
        self.config = config
        labels = [
            backend.tokenizer.encode(" " + label, add_special_tokens=False) for label in "ABT"
        ]
        if any(len(tokens) != 1 for tokens in labels):
            raise ValueError("probe judge requires single-token A/B/T labels")
        self.label_ids = [tokens[0] for tokens in labels]

    def _score(self, question: str, history: str, left: str, right: str) -> dict:
        backend, torch = self.backend, self.backend.torch
        prompt = f"""Compare these two next math steps. Check the calculations in A and B first;
do not restate the problem. Earlier reasoning may be wrong. Prefer mathematical
validity and progress, not agreement or style. Decide A, B, or T for a tie.

Problem: {question}
Earlier reasoning: {history}
Option A: {left}
Option B: {right}
"""
        ids = backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids += backend.tokenizer.encode("Check A's computation:", add_special_tokens=False)
        if len(ids) > self.config["judge_input_cap"]:
            return {"valid": False, "reason": "judge_input_cap"}
        with torch.inference_mode():
            output = backend.model(
                input_ids=torch.tensor([ids], device=backend.device),
                use_cache=True,
                logits_to_keep=1,
            )
        backend.forward_steps += len(ids)
        state = DecodeState(
            list(ids), len(ids), output.past_key_values, output.logits[0, -1].detach()
        )
        for _ in range(self.config["judge_reasoning_tokens"]):
            if backend.reasoning_end(state) is not None or backend.finished(state):
                break
            backend.advance(state, 1)
        if backend.finished(state) and backend.reasoning_end(state) is None:
            return {"valid": False, "reason": "judge_ended_before_decision"}
        thinking = backend.tokenizer.decode(state.ids[len(ids) :], skip_special_tokens=False)
        closed = backend.reasoning_end(state) is not None
        tail = ("" if closed else "</think>\n") + "Final decision:"
        for token in backend.tokenizer.encode(tail, add_special_tokens=False):
            backend.step(state, token)
        with torch.inference_mode():
            scores = state.next_logits.float()[self.label_ids].softmax(-1).tolist()
        return {
            "valid": True,
            "probabilities": dict(zip("ABT", scores)),
            "thinking": thinking,
            "thinking_closed": closed,
            "input_tokens": len(ids),
        }

    def compare(self, question: str, history: str, original: str, alternative: str) -> dict:
        self.backend.synchronize()
        started = time.monotonic()
        forward = self._score(question, history, original, alternative)
        reverse = (
            self._score(question, history, alternative, original) if forward["valid"] else None
        )
        self.backend.synchronize()
        seconds = time.monotonic() - started
        if not forward["valid"] or not reverse["valid"]:
            bad = forward if not forward["valid"] else reverse
            return {
                "accepted": False,
                "reason": bad["reason"],
                "seconds": seconds,
                "forward_detail": forward,
                "reverse_detail": reverse,
            }
        decision = preference_decision(
            forward["probabilities"], reverse["probabilities"], self.config["min_margin"]
        )
        return {
            **decision,
            "reason": "preferred" if decision["accepted"] else "no_clear_preference",
            "seconds": seconds,
            "forward_detail": forward,
            "reverse_detail": reverse,
        }

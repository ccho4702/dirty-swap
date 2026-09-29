"""Explicit boxed-answer extraction and version-locked Math-Verify scoring."""

from functools import lru_cache
import re


def boxed_answer(text):
    """Last box only; an unfinished/empty last box is an incomplete answer."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] not in "{}":
            continue
        backslashes, previous = 0, index - 1
        while previous >= 0 and text[previous] == "\\":
            backslashes += 1
            previous -= 1
        if backslashes % 2:
            continue
        depth += 1 if text[index] == "{" else -1
        if depth == 0:
            return text[start:index].strip() or None
    return None


@lru_cache(maxsize=4096)
def parsed_answer(answer):
    from math_verify import LatexExtractionConfig, parse

    return parse(
        "\\boxed{" + answer + "}",
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        fallback_mode="first_match",
        extraction_mode="first_match",
        parsing_timeout=5,
    )


def math_correct(gold, prediction):
    if prediction is None:
        return False
    from math_verify import verify

    return bool(
        verify(
            parsed_answer(gold),
            parsed_answer(prediction),
            float_rounding=6,
            numeric_precision=15,
            strict=True,
            timeout_seconds=5,
        )
    )


def natural_answer(text, row):
    from .core import extract_choice

    return (
        boxed_answer(text)
        if row.get("answer_format") == "math"
        else extract_choice(text, row["choices"])
    )

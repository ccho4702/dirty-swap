"""Shared QA prompt for baseline and swap arms."""


def instruction(row: dict) -> str:
    ending = (
        "Please reason step by step, and put your final answer within \\boxed{}."
        if row.get("answer_format") == "math"
        else "Please reason step by step. End your final response with 'Final answer: (LETTER)'."
    )
    return ending + "\n\nProblem:\n"

"""A short independent draft is re-encoded by the main model at its original prefix."""

from __future__ import annotations

import time

from .backend import PromptLimitError


def draft_model_config(spec: dict) -> dict:
    config = {**spec["model"], **spec["generation"]["draft"]["model"]}
    config["enable_thinking"] = False
    return config


def download_draft_model(spec: dict) -> None:
    if spec["generation"]["alternative_selection"] != "draft_swap":
        return
    from huggingface_hub import snapshot_download

    config = draft_model_config(spec)
    print(f"Download/verify draft {config['name']} @ {config['revision']}", flush=True)
    snapshot_download(
        config["name"],
        revision=config["revision"],
        cache_dir=config["cache_dir"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
        max_workers=spec["generation"]["draft"]["download_workers"],
        token=False,
    )


def generate_draft(drafter, main, question: str, config: dict, span_cap: int) -> dict:
    """Consume only the question; completion and span checks are independent of answer type."""
    drafter.synchronize()
    started = time.monotonic()
    result = {"prefix": None, "reason": "draft_incomplete"}
    try:
        state = drafter.prefill_text(
            config["prompt"] + "\n\nProblem:\n" + question,
            input_cap=config["max_input_tokens"],
        )
    except PromptLimitError:
        result["reason"] = "draft_input_limit"
    else:
        drafter.advance(state, config["max_new_tokens"])
        text = drafter.tokenizer.decode(state.ids[state.prompt_length :], skip_special_tokens=True)
        result.update(
            text=text,
            generated_tokens=len(state.ids) - state.prompt_length,
            prompt_tokens=state.prompt_length,
        )
        if drafter.finished(state) and text.strip():
            lines = text.strip().splitlines()
            while lines:
                prefix = "\n" + "\n".join(lines)
                tokens = main.tokenizer.encode(prefix, add_special_tokens=False)
                if len(tokens) <= span_cap:
                    break
                lines.pop(0)
            else:
                prefix, tokens = "", []
            if not tokens:
                result["reason"] = "draft_span_limit"
            elif any(token in main.eos_ids or token == main.think_end_id for token in tokens):
                result["reason"] = "draft_special_token"
            else:
                result.update(
                    prefix=prefix,
                    rollout_tokens=len(tokens),
                    truncated=prefix.strip() != text.strip(),
                    reason="draft_ready",
                )
        elif drafter.finished(state):
            result["reason"] = "draft_empty"
    drafter.synchronize()
    result["seconds"] = time.monotonic() - started
    return result

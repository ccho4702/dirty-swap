"""Frozen Transformers model with an explicit, token-aligned DynamicCache."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from .prompt import instruction


@dataclass
class DecodeState:
    ids: list[int]
    prompt_length: int
    cache: Any
    next_logits: Any


class PromptLimitError(ValueError):
    """A prompt was rejected before allocating a model cache."""


class TransformersBackend:
    def __init__(self, config: dict, *, model: Any = None, tokenizer: Any = None):
        import torch

        self.torch = torch
        self.config = config
        self.device = torch.device(config["device"])
        torch.set_num_threads(config["threads"])
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; use --device cpu or an allocated GPU")
            if self.device.type == "cuda":
                torch.cuda.set_per_process_memory_fraction(
                    config["gpu_memory_fraction"], self.device
                )
            options = {
                "revision": config["revision"],
                "cache_dir": config["cache_dir"],
                "local_files_only": config["local_files_only"],
                "trust_remote_code": False,
                "token": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(config["name"], **options)
            model = AutoModelForCausalLM.from_pretrained(
                config["name"],
                dtype=getattr(torch, config["dtype"]),
                attn_implementation="sdpa",
                **options,
            )
            model = model.to(self.device)
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.forward_steps = 0
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if len(end_ids) != 1:
            raise ValueError("tokenizer must encode </think> as one token")
        self.think_end_id = int(end_ids[0])
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def begin_measurement(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_mib(self) -> float | None:
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device) / 2**20
        return None

    def prefill(self, row: dict) -> DecodeState:
        return self.prefill_text(instruction(row) + row["question"])

    def prefill_text(self, text: str, *, input_cap: int | None = None) -> DecodeState:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config.get("enable_thinking", True),
        )
        if not ids or len(ids) > self.config["max_context_tokens"]:
            raise PromptLimitError("empty prompt or prompt exceeds context")
        if input_cap is not None and len(ids) > input_cap:
            raise PromptLimitError("prompt exceeds input cap")
        cache = None
        output = None
        with self.torch.inference_mode():
            for start in range(0, len(ids), self.config.get("prefill_chunk_tokens", 256)):
                chunk = ids[start : start + self.config.get("prefill_chunk_tokens", 256)]
                output = self.model(
                    input_ids=self.torch.tensor([chunk], device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                self.forward_steps += len(chunk)
        if not getattr(cache, "layers", None):
            raise ValueError("model must return a layer-based DynamicCache")
        return DecodeState(list(ids), len(ids), cache, output.logits[0, -1].detach())

    def fork(self, state: DecodeState) -> DecodeState:
        return DecodeState(
            list(state.ids),
            state.prompt_length,
            copy.deepcopy(state.cache),
            state.next_logits.clone(),
        )

    def sequence_length(self, state: DecodeState) -> int:
        return len(state.ids)

    def top_tokens(self, state: DecodeState, count: int) -> list[int]:
        # topk can choose a different first token from argmax on tied logits.
        # The main path must always use the same token as ordinary greedy decoding.
        main = int(self.torch.argmax(state.next_logits).item())
        if count == 1:
            return [main]
        ranked = self.torch.topk(state.next_logits.float(), count).indices.tolist()
        return [main] + [int(token) for token in ranked if int(token) != main][: count - 1]

    def numeric_ambiguity(self, state: DecodeState, min_ratio: float) -> dict | None:
        """Find a concrete numeric top-1/top-2 fork without treating confidence as truth."""
        tokens = self.top_tokens(state, 2)
        values = state.next_logits.float()[tokens]
        texts = [
            self.tokenizer.decode([token], skip_special_tokens=False).strip() for token in tokens
        ]
        if (
            any(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text) is None for text in texts)
            or texts[0] == texts[1]
        ):
            return None
        ratio = float((values[1] - values[0]).exp().item())
        if ratio < min_ratio:
            return None
        return {
            "policy": "numeric_ambiguity",
            "top_tokens": tokens,
            "top_texts": texts,
            "top2_ratio": ratio,
        }

    def step(self, state: DecodeState, token: int) -> DecodeState:
        if len(state.ids) + 1 > self.config["max_context_tokens"]:
            raise ValueError("generation exceeds context")
        with self.torch.inference_mode():
            output = self.model(
                input_ids=self.torch.tensor([[token]], device=self.device),
                past_key_values=state.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        state.cache = output.past_key_values
        state.ids.append(int(token))
        state.next_logits = output.logits[0, -1].detach()
        self.forward_steps += 1
        return state

    def reasoning_step_boundary(self, state: DecodeState) -> dict | None:
        tail = self.tokenizer.decode(state.ids[-16:], skip_special_tokens=False)
        return {"policy": "reasoning_step"} if re.search(r"\n\s*$", tail) else None

    def soft_rollout(
        self, state: DecodeState, total_tokens: int, config: dict, seed: int
    ) -> DecodeState:
        """Build a donor from randomized embedding mixtures; IDs are diagnostic proxies."""
        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(seed)
        embeddings = self.model.get_input_embeddings()
        for _ in range(total_tokens):
            if self.finished(state) or self.reasoning_end(state) is not None:
                break
            if len(state.ids) + 1 > self.config["max_context_tokens"]:
                raise ValueError("generation exceeds context")
            with torch.inference_mode():
                scores, ids = torch.topk(
                    state.next_logits.float() / config["temperature"],
                    min(config["top_k"], state.next_logits.numel()),
                )
                # Preserve greedy tie-breaking, including the top_k=1 identity control.
                if config["top_k"] == 1:
                    ids = torch.argmax(state.next_logits).reshape(1)
                probabilities = scores.softmax(-1)
                remove = probabilities.cumsum(-1) - probabilities >= config["top_p"]
                scores = scores.masked_fill(remove, -torch.inf)
                uniform = torch.rand(scores.shape, device=self.device, generator=generator)
                uniform = uniform.clamp(
                    torch.finfo(torch.float32).tiny, 1 - torch.finfo(torch.float32).eps
                )
                noise = -torch.log(-torch.log(uniform))
                weights = ((scores + noise) / config["soft_temperature"]).softmax(-1)
                proxy = int(ids[weights.argmax()].item())
                if proxy in self.eos_ids or proxy == self.think_end_id:
                    self.step(state, proxy)
                    break
                mixed = (embeddings(ids).float() * weights[:, None]).sum(0)
                output = self.model(
                    inputs_embeds=mixed.to(embeddings.weight.dtype).reshape(1, 1, -1),
                    past_key_values=state.cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            state.cache = output.past_key_values
            state.ids.append(proxy)
            state.next_logits = output.logits[0, -1].detach()
            self.forward_steps += 1
        return state

    def guided_rollout(self, state: DecodeState, prefix: str, total_tokens: int) -> DecodeState:
        """Teacher-force a short alternative step, then finish the equal-length span."""
        tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        if not tokens or len(tokens) > total_tokens:
            raise ValueError("alternative prefix must fit inside the replacement span")
        if any(token in self.eos_ids or token == self.think_end_id for token in tokens):
            raise ValueError("alternative prefix must stay inside reasoning")
        for token in tokens:
            self.step(state, token)
        return self.advance(state, total_tokens - len(tokens))

    def advance(self, state: DecodeState, token_count: int) -> DecodeState:
        for _ in range(token_count):
            if state.ids[-1] in self.eos_ids:
                break
            token = int(self.torch.argmax(state.next_logits).item())
            self.step(state, token)
        return state

    def rollout(self, state: DecodeState, first_token: int, total_tokens: int) -> DecodeState:
        self.step(state, first_token)
        return self.advance(state, total_tokens - 1)

    def cache(self, state: DecodeState) -> Any:
        return state.cache

    def reasoning_end(self, state: DecodeState) -> int | None:
        try:
            return state.ids.index(self.think_end_id, state.prompt_length)
        except ValueError:
            return None

    def finished(self, state: DecodeState) -> bool:
        return bool(state.ids and state.ids[-1] in self.eos_ids)

    def decode_answer(self, state: DecodeState) -> str:
        end = self.reasoning_end(state)
        if end is None:
            return (
                self.tokenizer.decode(state.ids[state.prompt_length :], skip_special_tokens=True)
                if self.finished(state)
                else ""
            )
        return self.tokenizer.decode(state.ids[end + 1 :], skip_special_tokens=True)

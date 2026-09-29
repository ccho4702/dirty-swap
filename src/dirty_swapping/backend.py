"""Frozen Transformers model with an explicit, token-aligned DynamicCache."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .prompt import instruction


@dataclass
class DecodeState:
    ids: list[int]
    prompt_length: int
    cache: Any
    next_logits: Any


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
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction(row) + row["question"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if not ids or len(ids) > self.config["max_context_tokens"]:
            raise ValueError("empty prompt or prompt exceeds context")
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
        return [
            int(token)
            for token in self.torch.topk(state.next_logits.float(), count).indices.tolist()
        ]

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

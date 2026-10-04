import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dirty_swapping.backend import TransformersBackend
from dirty_swapping.config import load_spec, validate_spec
from dirty_swapping.engine import intervene
from dirty_swapping.protocol import SwapPlan


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [98] if text == "</think>" else [1]

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]


class SoftTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=64,
                eos_token_id=99,
            )
        )
        self.backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 64},
            model=model,
            tokenizer=Tokenizer(),
        )
        self.backend.eos_ids = {998}
        self.backend.think_end_id = 999
        self.prefix = self.backend.prefill({"question": "Q", "answer_format": "math"})
        self.soft = {"top_k": 8, "top_p": 0.95, "temperature": 0.6, "soft_temperature": 0.5}

    def test_single_component_is_exact_greedy_cache(self):
        hard = self.backend.advance(self.backend.fork(self.prefix), 4)
        soft = self.backend.soft_rollout(
            self.backend.fork(self.prefix), 4, {**self.soft, "top_k": 1}, 18
        )
        self.assertEqual(soft.ids, hard.ids)
        for a, b in zip(soft.cache.layers, hard.cache.layers):
            self.assertTrue(torch.equal(a.keys, b.keys))
            self.assertTrue(torch.equal(a.values, b.values))

    def test_private_rng_replays_without_changing_global_rng(self):
        rng = torch.get_rng_state().clone()
        a = self.backend.soft_rollout(self.backend.fork(self.prefix), 4, self.soft, 18)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.rand(100)
        b = self.backend.soft_rollout(self.backend.fork(self.prefix), 4, self.soft, 18)
        self.assertEqual(a.ids, b.ids)
        for x, y in zip(a.cache.layers, b.cache.layers):
            self.assertTrue(torch.equal(x.keys, y.keys))
            self.assertTrue(torch.equal(x.values, y.values))
        self.assertEqual(self.prefix.ids, [1, 2, 3])

    def test_transplant_preserves_existing_ids_prefix_and_suffix(self):
        plan = SwapPlan(3, 4, 2, 2)
        baseline = intervene(self.backend, self.backend.fork(self.prefix), plan, enabled=False)
        swap = intervene(
            self.backend,
            self.backend.fork(self.prefix),
            plan,
            enabled=True,
            selection="soft_swap",
            soft_config=self.soft,
            seed=18,
        )
        self.assertTrue(swap.swapped)
        self.assertEqual(baseline.state.ids, swap.state.ids)
        changed = False
        for a, b in zip(baseline.state.cache.layers, swap.state.cache.layers):
            for key in ("keys", "values"):
                x, y = getattr(a, key), getattr(b, key)
                self.assertTrue(torch.equal(x[:, :, :3], y[:, :, :3]))
                self.assertTrue(torch.equal(x[:, :, 7:], y[:, :, 7:]))
                changed |= not torch.equal(x[:, :, 3:7], y[:, :, 3:7])
        self.assertTrue(changed)
        self.assertTrue(torch.equal(baseline.state.next_logits, swap.state.next_logits))
        self.assertTrue(
            all(not p.requires_grad and p.grad is None for p in self.backend.model.parameters())
        )

    def test_dominant_end_token_stops_without_padding(self):
        self.backend.eos_ids = {99}
        self.prefix.next_logits = self.prefix.next_logits.clone()
        self.prefix.next_logits.fill_(-torch.inf)
        self.prefix.next_logits[99] = 0
        state = self.backend.soft_rollout(self.backend.fork(self.prefix), 4, self.soft, 18)
        self.assertEqual(state.ids, [1, 2, 3, 99])
        self.assertTrue(self.backend.finished(state))

    def test_invalid_configuration_rejected(self):
        spec = load_spec()
        spec["generation"].update(
            alternative_selection="soft_swap", candidate_count=2, soft=self.soft
        )
        validate_spec(spec)
        for key, value in [
            ("top_k", 0),
            ("top_k", True),
            ("temperature", 0),
            ("top_p", 1.1),
            ("soft_temperature", float("nan")),
        ]:
            bad = copy.deepcopy(spec)
            bad["generation"]["soft"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_spec(bad)


if __name__ == "__main__":
    unittest.main()

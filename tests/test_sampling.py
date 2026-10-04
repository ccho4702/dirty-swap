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
        return [98]

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]


class SamplingTests(unittest.TestCase):
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
                pad_token_id=0,
            )
        )
        self.backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 64},
            model=model,
            tokenizer=Tokenizer(),
        )
        self.settings = {"temperature": 0.6, "top_k": 12, "top_p": 0.95}
        self.prefix = self.backend.prefill({"question": "Q", "answer_format": "math"})

    def test_matches_native_hf_sampling(self):
        self.backend.configure_sampling(self.prefix, self.settings, 42)
        self.backend.advance(self.prefix, 20)
        torch.manual_seed(42)
        native = self.backend.model.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            do_sample=True,
            **self.settings,
            max_new_tokens=20,
        )
        self.assertEqual(native[0].tolist(), self.prefix.ids)

    def test_branch_and_zero_strength_match_uninterrupted_trace(self):
        self.backend.eos_ids = {998}
        self.backend.think_end_id = 999
        self.backend.configure_sampling(self.prefix, self.settings, 42)
        plain = self.backend.fork(self.prefix)
        self.backend.advance(plain, 18)
        self.backend.advance(self.prefix, 3)
        self.assertNotEqual(plain.ids[6], self.backend.top_tokens(self.prefix, 2)[0])
        original_rng = self.prefix.generator.get_state().clone()
        plan = SwapPlan(6, 4, 2, 2)
        ordinary = intervene(self.backend, self.backend.fork(self.prefix), plan, enabled=False)
        zero = intervene(
            self.backend,
            self.backend.fork(self.prefix),
            plan,
            enabled=True,
            swap_strength=0,
        )
        self.assertFalse(zero.swapped)
        self.assertEqual(zero.reason, "zero_strength")
        self.assertEqual(ordinary.state.ids, zero.state.ids)
        for a, b in zip(ordinary.state.cache.layers, zero.state.cache.layers):
            self.assertTrue(torch.equal(a.keys, b.keys))
            self.assertTrue(torch.equal(a.values, b.values))
        for result in (ordinary, zero):
            self.backend.advance(result.state, 9)
            self.assertEqual(result.state.ids, plain.ids)
            self.assertTrue(
                torch.equal(result.state.generator.get_state(), plain.generator.get_state())
            )
        self.assertTrue(torch.equal(original_rng, self.prefix.generator.get_state()))

    def test_forks_and_donor_do_not_consume_main_or_global_rng(self):
        self.backend.eos_ids = {998}
        self.backend.think_end_id = 999
        global_rng = torch.get_rng_state().clone()
        self.backend.configure_sampling(self.prefix, self.settings, 42)
        a, b = self.backend.fork(self.prefix), self.backend.fork(self.prefix)
        self.backend.advance(a, 8)
        self.backend.soft_rollout(
            self.backend.fork(self.prefix),
            4,
            {"top_k": 8, "top_p": 0.95, "temperature": 0.6, "soft_temperature": 0.5},
            19,
        )
        self.backend.advance(b, 8)
        self.assertEqual(a.ids, b.ids)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertNotEqual(a.ids, self.prefix.ids)

    def test_invalid_sampling_configuration(self):
        spec = load_spec()
        spec["generation"]["sampling"] = self.settings
        validate_spec(spec)
        for key, value in (
            ("top_k", 0),
            ("top_k", True),
            ("top_p", 1.1),
            ("temperature", float("nan")),
            ("temperature", 0),
        ):
            bad = copy.deepcopy(spec)
            bad["generation"]["sampling"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_spec(bad)
        for settings in (None, {}, {"top_k": 3}, {**self.settings, "unexpected": 1}):
            bad = copy.deepcopy(spec)
            bad["generation"]["sampling"] = settings
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_spec(bad)


if __name__ == "__main__":
    unittest.main()

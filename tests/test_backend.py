import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dirty_swapping.backend import TransformersBackend
from dirty_swapping.cache import swap_kv_segment
from dirty_swapping.config import load_spec, validate_spec
from dirty_swapping.engine import intervene
from dirty_swapping.protocol import SwapPlan
from dirty_swapping.runner import run_case


class TinyTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [98] if text == "</think>" else [0]

    def apply_chat_template(self, *args, **kwargs):
        return [1, 1, 1]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class TinyCausalModel:
    generation_config = SimpleNamespace(eos_token_id=99)

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def __call__(self, *, input_ids, past_key_values=None, use_cache=True, logits_to_keep=None):
        previous = None if past_key_values is None else past_key_values.layers[0].keys
        saw_alternative = previous is not None and bool(torch.any(previous == 8))
        values = input_ids.to(torch.float32).reshape(1, 1, -1, 1)
        keys = values if previous is None else torch.cat((previous, values), dim=2)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=keys.clone())])
        logits = torch.zeros((1, input_ids.shape[1], 100), dtype=torch.float32)
        for rank, token in enumerate(([6, 7, 8, 9] if saw_alternative else [7, 8, 9, 10])):
            logits[0, -1, token] = 10 - rank
        return SimpleNamespace(past_key_values=cache, logits=logits)


class BackendTests(unittest.TestCase):
    def test_guided_swap_changes_only_old_span_and_logs_actual_token(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        prefix = backend.prefill({"question": "Q", "answer_format": "math"})
        plan = SwapPlan(3, 2, 2, 2)
        baseline = intervene(backend, backend.fork(prefix), plan, enabled=False)
        with patch.object(backend.tokenizer, "encode", return_value=[9]):
            swapped = intervene(
                backend,
                backend.fork(prefix),
                plan,
                enabled=True,
                selection="guided_swap",
                alternative_prefix="Compute directly.",
            )
        self.assertTrue(swapped.swapped)
        self.assertEqual(swapped.reason, "guided_swapped")
        self.assertEqual(swapped.alternative_token, 9)
        self.assertEqual(swapped.state.ids, baseline.state.ids)
        layer = swapped.state.cache.layers[0]
        self.assertEqual(layer.keys.flatten().tolist(), [1, 1, 1, 9, 7, 7, 7])
        self.assertEqual(prefix.ids, [1, 1, 1])

    def test_guided_prefix_validated_before_any_cache_mutation(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        prefix = backend.prefill({"question": "Q", "answer_format": "math"})
        for tokens in ([], [8, 8, 8], [98], [99]):
            with (
                self.subTest(tokens=tokens),
                patch.object(backend.tokenizer, "encode", return_value=tokens),
            ):
                with self.assertRaises(ValueError):
                    backend.guided_rollout(prefix, "invalid", 2)
                self.assertEqual(prefix.ids, [1, 1, 1])
        for tail, expected in (("Step\n", True), ("Step\n\n", True), ("123", False)):
            with patch.object(backend.tokenizer, "decode", return_value=tail):
                self.assertEqual(backend.reasoning_step_boundary(prefix) is not None, expected)

    def test_guided_configuration_requires_prefix_and_fitting_scan(self):
        spec = load_spec()
        spec["generation"].update(
            alternative_selection="guided_swap",
            alternative_prefix="Compute directly.",
            candidate_count=2,
            branch_policy="reasoning_step",
            branch_scan_end=512,
        )
        validate_spec(spec)
        spec["generation"]["alternative_prefix"] = " "
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_spec(spec)
        spec["generation"]["alternative_prefix"] = "Compute directly."
        spec["generation"]["branch_scan_end"] = 40000
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_spec(spec)

    def test_tied_logits_preserve_the_plain_greedy_main_token(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        prefix = backend.prefill({"question": "Q", "answer_format": "math"})
        prefix.next_logits = torch.ones_like(prefix.next_logits)
        self.assertEqual(backend.top_tokens(prefix, 2)[0], 0)
        pure = backend.fork(prefix)
        backend.advance(pure, 8)
        baseline = intervene(
            backend,
            backend.fork(prefix),
            SwapPlan(3, 2, 2, 2),
            enabled=False,
        )
        backend.advance(baseline.state, 4)
        self.assertEqual(baseline.main_token, 0)
        self.assertEqual(baseline.state.ids, pure.ids)

    def test_numeric_trigger_uses_distinct_numeric_tokens_and_ratio(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        state = backend.prefill({"question": "Q", "answer_format": "math"})
        self.assertEqual(backend.numeric_ambiguity(state, 0.1)["top_tokens"], [7, 8])
        self.assertIsNone(backend.numeric_ambiguity(state, 0.5))
        with patch.object(backend.tokenizer, "decode", return_value="word"):
            self.assertIsNone(backend.numeric_ambiguity(state, 0.1))
        with patch.object(backend.tokenizer, "decode", side_effect=["7", " 7"]):
            self.assertIsNone(backend.numeric_ambiguity(state, 0.1))

    def test_baseline_matches_plain_autoregression(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        prefix = backend.prefill({"question": "Q", "answer_format": "choice"})
        pure = backend.fork(prefix)
        backend.advance(pure, 8)
        baseline = intervene(
            backend,
            backend.fork(prefix),
            SwapPlan(branch_position=3, rollout_tokens=2, delay_tokens=2, candidate_count=4),
            enabled=False,
        )
        backend.advance(baseline.state, 4)
        self.assertEqual(baseline.reason, "baseline")
        self.assertEqual(baseline.state.ids, pure.ids)
        self.assertTrue(torch.equal(baseline.state.cache.layers[0].keys, pure.cache.layers[0].keys))

    def test_real_qwen3_dynamic_cache_preserves_suffix(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        config = Qwen3Config(
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
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 64},
            model=Qwen3ForCausalLM(config),
            tokenizer=TinyTokenizer(),
        )
        self.assertFalse(backend.model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in backend.model.parameters())
        )
        weights_before = {
            name: parameter.detach().clone() for name, parameter in backend.model.named_parameters()
        }
        prefix = backend.prefill({"question": "Q", "answer_format": "choice"})
        main, alternative = backend.fork(prefix), backend.fork(prefix)
        for token in (7, 7, 7):
            backend.step(main, token)
        for token in (8, 8):
            backend.step(alternative, token)
        before = [(layer.keys.clone(), layer.values.clone()) for layer in main.cache.layers]
        swap_kv_segment(main.cache, alternative.cache, 3, 2)
        changed = False
        for layer, (keys, values) in zip(main.cache.layers, before):
            self.assertTrue(torch.equal(layer.keys[:, :, :3, :], keys[:, :, :3, :]))
            self.assertTrue(torch.equal(layer.values[:, :, :3, :], values[:, :, :3, :]))
            self.assertTrue(torch.equal(layer.keys[:, :, 5:, :], keys[:, :, 5:, :]))
            self.assertTrue(torch.equal(layer.values[:, :, 5:, :], values[:, :, 5:, :]))
            changed |= not torch.equal(layer.keys[:, :, 3:5, :], keys[:, :, 3:5, :])
        self.assertTrue(changed)
        backend.step(main, 7)
        self.assertEqual(main.cache.get_seq_length(), 7)
        for name, parameter in backend.model.named_parameters():
            self.assertTrue(torch.equal(parameter, weights_before[name]))
            self.assertIsNone(parameter.grad)

    def test_edit_changes_later_generation_without_suffix_rewrite(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        row = {"question": "Q", "answer_format": "choice"}
        prefix = backend.prefill(row)
        plan = SwapPlan(branch_position=3, rollout_tokens=2, delay_tokens=2, candidate_count=4)
        baseline = intervene(backend, prefix, plan, enabled=False)
        swapped = intervene(backend, backend.prefill(row), plan, enabled=True)
        self.assertEqual(baseline.state.ids, swapped.state.ids)
        self.assertEqual(
            baseline.state.cache.layers[0].keys.flatten()[:3].tolist(),
            swapped.state.cache.layers[0].keys.flatten()[:3].tolist(),
        )
        self.assertEqual(
            baseline.state.cache.layers[0].keys.flatten()[5:].tolist(),
            swapped.state.cache.layers[0].keys.flatten()[5:].tolist(),
        )
        self.assertEqual(swapped.state.cache.layers[0].keys.flatten()[3:5].tolist(), [8, 7])
        backend.advance(baseline.state, 2)
        backend.advance(swapped.state, 2)
        self.assertEqual(baseline.state.ids[-2:], [7, 7])
        self.assertEqual(swapped.state.ids[-2:], [7, 6])

    def test_runner_uses_transformers_adapter_contract(self):
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32},
            model=TinyCausalModel(),
            tokenizer=TinyTokenizer(),
        )
        spec = load_spec()
        spec["generation"].update(
            max_input_tokens=8,
            max_new_tokens=8,
            branch_after_reasoning_tokens=2,
            rollout_tokens=2,
            delay_tokens=2,
        )
        row = {
            "id": "tiny",
            "task": "gpqa",
            "question": "Q",
            "answer_format": "choice",
            "gold": "B",
            "choices": ["A", "B", "C", "D"],
        }
        baseline = run_case(backend, row, spec, "baseline")
        swapped = run_case(backend, row, spec, "swap")
        self.assertEqual(baseline["generated_tokens"], 8)
        self.assertEqual(swapped["generated_tokens"], 8)
        self.assertTrue(swapped["swap"]["swapped"])


if __name__ == "__main__":
    unittest.main()

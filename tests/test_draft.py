import copy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dirty_swapping.backend import PromptLimitError, TransformersBackend
from dirty_swapping.config import load_spec, runtime_spec, validate_spec
from dirty_swapping.draft import generate_draft, draft_model_config
from pathlib import Path


class Characters:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) for token in ids if token != 999)


class DraftBackend:
    tokenizer = Characters()

    def __init__(self, body, ended=True):
        self.body, self.ended = body, ended
        self.prompts = []
        self.calls = 0

    def synchronize(self):
        pass

    def prefill_text(self, text, input_cap):
        self.prompts.append(text)
        if len(text) > input_cap:
            raise PromptLimitError("too long")
        return SimpleNamespace(ids=[0], prompt_length=1)

    def advance(self, state, tokens):
        self.calls += 1
        state.ids.extend(self.tokenizer.encode(self.body) + ([999] if self.ended else []))

    def finished(self, state):
        return self.ended


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.main = SimpleNamespace(tokenizer=Characters(), eos_ids={999}, think_end_id=998)
        self.config = {"prompt": "Solve briefly.", "max_input_tokens": 128, "max_new_tokens": 64}

    def test_tail_keeps_complete_answer_and_uses_question_only(self):
        drafter = DraftBackend("Initial explanation too long for the span.\n\\boxed{2}")
        result = generate_draft(drafter, self.main, "Q", self.config, 16)
        self.assertEqual(result["reason"], "draft_ready")
        self.assertEqual(result["prefix"], "\n\\boxed{2}")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rollout_tokens"], len(result["prefix"]))
        self.assertEqual(drafter.prompts, ["Solve briefly.\n\nProblem:\nQ"])

    def test_missing_incomplete_and_oversized_answer_reject(self):
        for body, ended, cap, reason in (
            ("\\boxed{2}", False, 32, "draft_incomplete"),
            ("\\boxed{", True, 32, "draft_incomplete"),
            ("no box", True, 32, "draft_incomplete"),
            ("\\boxed{123456789}", True, 4, "draft_span_limit"),
        ):
            with self.subTest(body=body, ended=ended):
                result = generate_draft(DraftBackend(body, ended), self.main, "Q", self.config, cap)
                self.assertIsNone(result["prefix"])
                self.assertEqual(result["reason"], reason)

    def test_input_limit_and_special_tokens_fail_closed(self):
        config = {**self.config, "max_input_tokens": 1}
        drafter = DraftBackend("\\boxed{2}")
        result = generate_draft(drafter, self.main, "Q", config, 32)
        self.assertEqual(result["reason"], "draft_input_limit")
        self.assertEqual(drafter.calls, 0)
        with patch.object(self.main.tokenizer, "encode", return_value=[998]):
            result = generate_draft(drafter, self.main, "Q", self.config, 32)
        self.assertEqual(result["reason"], "draft_special_token")

    def test_backend_rejects_input_before_forward_and_honors_thinking_mode(self):
        model, tokenizer = MagicMock(), MagicMock()
        model.eval.return_value = model
        model.generation_config.eos_token_id = 99
        tokenizer.encode.return_value = [98]
        tokenizer.apply_chat_template.return_value = [1] * 10
        backend = TransformersBackend(
            {"device": "cpu", "threads": 1, "max_context_tokens": 32, "enable_thinking": False},
            model=model,
            tokenizer=tokenizer,
        )
        with self.assertRaises(PromptLimitError):
            backend.prefill_text("Q", input_cap=4)
        model.assert_not_called()
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_configuration_and_cpu_override(self):
        spec = load_spec()
        spec["generation"].update(alternative_selection="draft_swap", candidate_count=2)
        spec["generation"]["draft"] = {
            "model": {"name": "aux", "revision": "f" * 40},
            "prompt": "Solve briefly.",
            "max_input_tokens": 128,
            "max_new_tokens": 64,
            "download_workers": 2,
        }
        validate_spec(spec)
        resolved = runtime_spec(spec, Path("."), device="cpu")
        child = draft_model_config(resolved)
        self.assertEqual(child["device"], "cpu")
        self.assertEqual(child["dtype"], "float32")
        self.assertFalse(child["enable_thinking"])
        for key, value in (("max_new_tokens", 100000), ("max_input_tokens", 0), ("prompt", "")):
            bad = copy.deepcopy(spec)
            bad["generation"]["draft"][key] = value
            with self.assertRaises(ValueError):
                validate_spec(bad)


if __name__ == "__main__":
    unittest.main()

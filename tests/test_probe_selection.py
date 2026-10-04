import copy
import unittest
from types import SimpleNamespace

import torch

from dirty_swapping.config import load_spec, validate_spec
from dirty_swapping.preference import ProbePreferenceJudge, preference_decision
from dirty_swapping.probe_selection import choose_probe
from dirty_swapping.protocol import SwapPlan


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))

    def encode(self, text, add_special_tokens=False):
        return [ord(text.strip()[-1])]

    def apply_chat_template(self, *args, **kwargs):
        return [1] * 10


class Adapter:
    tokenizer = Tokenizer()

    def __init__(self, identical=False):
        self.identical = identical

    def fork(self, state):
        return copy.deepcopy(state)

    def cache(self, state):
        return state.cache

    def sequence_length(self, state):
        return len(state.ids)

    def synchronize(self):
        pass

    def advance(self, state, count):
        changed = bool(torch.any(state.cache.layers[0].keys == 8))
        token = 9 if changed and not self.identical else 7
        state.ids.extend([token] * count)
        value = torch.full((1, 1, count, 1), token, dtype=torch.float32)
        layer = state.cache.layers[0]
        layer.keys = torch.cat((layer.keys, value), dim=2)
        layer.values = torch.cat((layer.values, value), dim=2)


class Judge:
    def __init__(self, accepted):
        self.accepted = accepted
        self.calls = []

    def compare(self, *args):
        self.calls.append(args)
        return {"accepted": self.accepted, "seconds": 0.0}


def state(ids):
    tensor = torch.tensor(ids, dtype=torch.float32).reshape(1, 1, -1, 1)
    return SimpleNamespace(
        ids=list(ids),
        prompt_length=3,
        cache=SimpleNamespace(layers=[SimpleNamespace(keys=tensor, values=tensor.clone())]),
    )


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.main = state([1, 2, 3, 7, 7, 7, 7])
        self.alternative = state([1, 2, 3, 8, 8])
        self.plan = SwapPlan(3, 2, 2, 2)

    def test_accept_commits_probe_without_changing_existing_suffix(self):
        before = self.main.cache.layers[0].keys.clone()
        judge = Judge(True)
        selected = choose_probe(
            Adapter(),
            self.main,
            self.alternative.cache,
            self.plan,
            "Question only",
            {"tokens": 3, "history_tokens": 3},
            judge,
        )
        self.assertTrue(selected.swapped)
        self.assertEqual(selected.state.ids, self.main.ids + [9, 9, 9])
        self.assertTrue(torch.equal(self.main.cache.layers[0].keys, before))
        self.assertTrue(
            torch.equal(selected.state.cache.layers[0].keys[:, :, 5:7, :], before[:, :, 5:7, :])
        )
        self.assertEqual(selected.state.cache.layers[0].keys.flatten()[3:5].tolist(), [8, 8])
        self.assertEqual(judge.calls[0][0], "Question only")

    def test_reject_returns_unedited_probe(self):
        selected = choose_probe(
            Adapter(),
            self.main,
            self.alternative.cache,
            self.plan,
            "Q",
            {"tokens": 3, "history_tokens": 3},
            Judge(False),
        )
        self.assertFalse(selected.swapped)
        self.assertEqual(selected.state.ids, self.main.ids + [7, 7, 7])
        self.assertEqual(selected.state.cache.layers[0].keys.flatten()[3:5].tolist(), [7, 7])

    def test_identical_probes_skip_judge(self):
        judge = Judge(True)
        selected = choose_probe(
            Adapter(identical=True),
            self.main,
            self.alternative.cache,
            self.plan,
            "Q",
            {"tokens": 3, "history_tokens": 3},
            judge,
        )
        self.assertEqual(selected.reason, "identical_probes")
        self.assertFalse(selected.swapped)
        self.assertEqual(judge.calls, [])

    def test_order_bias_and_ties_reject(self):
        b = {"A": 0.1, "B": 0.8, "T": 0.1}
        a = {"A": 0.8, "B": 0.1, "T": 0.1}
        self.assertTrue(preference_decision(b, a, 0.15)["accepted"])
        self.assertFalse(preference_decision(b, b, 0.15)["accepted"])
        self.assertFalse(preference_decision(a, a, 0.15)["accepted"])
        tie = {"A": 0.1, "B": 0.1, "T": 0.8}
        self.assertFalse(preference_decision(tie, tie, 0.15)["accepted"])

    def test_overlong_judge_prompt_refuses_before_model_call(self):
        backend = SimpleNamespace(tokenizer=Tokenizer(), torch=torch, synchronize=lambda: None)
        judge = ProbePreferenceJudge(backend, {"judge_input_cap": 1, "min_margin": 0.15})
        result = judge.compare("Q", "history", "original", "alternative")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "judge_input_cap")

    def test_probe_config_limits(self):
        spec = load_spec("configs/probe-preference.json")
        for key, value in (("tokens", 1), ("min_margin", float("nan")), ("judge_input_cap", 9000)):
            bad = copy.deepcopy(spec)
            bad["generation"]["probe"][key] = value
            with self.assertRaises(ValueError):
                validate_spec(bad)
        bad = copy.deepcopy(spec)
        bad["generation"]["max_new_tokens"] = 450
        with self.assertRaises(ValueError):
            validate_spec(bad)

    def test_numeric_scan_config_limits(self):
        spec = load_spec("configs/probe-numeric.json")
        for change in ({"min_top2_ratio": 0}, {"branch_scan_end": 10}, {"branch_scan_end": 4096}):
            bad = copy.deepcopy(spec)
            bad["generation"].update(change)
            with self.assertRaises(ValueError):
                validate_spec(bad)


if __name__ == "__main__":
    unittest.main()

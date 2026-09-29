import unittest
from types import SimpleNamespace

import torch

from dirty_swapping.engine import intervene
from dirty_swapping.protocol import SwapPlan


class FakeAdapter:
    def __init__(self, reasoning_end=None):
        self.end = reasoning_end
        self.rollout_count = 0

    def fork(self, state):
        return SimpleNamespace(
            tokens=list(state.tokens),
            cache=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        keys=state.cache.layers[0].keys.clone(),
                        values=state.cache.layers[0].values.clone(),
                    )
                ]
            ),
        )

    def sequence_length(self, state):
        return len(state.tokens)

    def top_tokens(self, state, count):
        return [7, 8, 9, 10][:count]

    def rollout(self, state, first_token, total_tokens):
        self.rollout_count += 1
        return self._append(state, [first_token] * total_tokens)

    def advance(self, state, token_count):
        return self._append(state, [7] * token_count)

    def cache(self, state):
        return state.cache

    def synchronize(self):
        pass

    def reasoning_end(self, state):
        return self.end if self.end is not None and len(state.tokens) > self.end else None

    def finished(self, state):
        return False

    def _append(self, state, tokens):
        state.tokens.extend(tokens)
        new = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1, 1)
        layer = state.cache.layers[0]
        layer.keys = torch.cat((layer.keys, new), dim=2)
        layer.values = torch.cat((layer.values, new), dim=2)
        return state


class EngineTests(unittest.TestCase):
    def setUp(self):
        tensor = torch.zeros((1, 1, 3, 1))
        self.prefix = SimpleNamespace(
            tokens=[1, 2, 3],
            cache=SimpleNamespace(
                layers=[SimpleNamespace(keys=tensor.clone(), values=tensor.clone())]
            ),
        )
        self.plan = SwapPlan(branch_position=3, rollout_tokens=2, delay_tokens=2, candidate_count=4)

    def test_swaps_old_kv_but_keeps_text_and_suffix(self):
        adapter = FakeAdapter()
        result = intervene(adapter, self.prefix, self.plan, enabled=True)
        self.assertTrue(result.swapped)
        self.assertEqual(adapter.rollout_count, 4)
        self.assertEqual(result.state.tokens, [1, 2, 3, 7, 7, 7, 7])
        self.assertEqual(
            result.state.cache.layers[0].keys.flatten().tolist(), [0, 0, 0, 8, 8, 7, 7]
        )

    def test_baseline_and_ended_reasoning(self):
        baseline_adapter = FakeAdapter()
        baseline = intervene(baseline_adapter, self.prefix, self.plan, enabled=False)
        tensor = torch.zeros((1, 1, 3, 1))
        fresh_prefix = SimpleNamespace(
            tokens=[1, 2, 3],
            cache=SimpleNamespace(
                layers=[SimpleNamespace(keys=tensor.clone(), values=tensor.clone())]
            ),
        )
        ended = intervene(FakeAdapter(reasoning_end=6), fresh_prefix, self.plan, enabled=True)
        self.assertFalse(baseline.swapped)
        self.assertEqual(baseline_adapter.rollout_count, 1)
        self.assertFalse(ended.swapped)
        self.assertEqual(ended.reason, "reasoning_ended")
        self.assertEqual(ended.state.cache.layers[0].keys.flatten().tolist(), [0, 0, 0, 7, 7, 7, 7])


if __name__ == "__main__":
    unittest.main()

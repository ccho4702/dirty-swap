import unittest
import copy
from types import SimpleNamespace

import torch

from dirty_swapping.cache import swap_kv_segment
from dirty_swapping.config import load_spec, validate_spec


def make_cache(value, length=8):
    tensor = torch.full((1, 2, length, 3), value, dtype=torch.float32)
    return SimpleNamespace(
        layers=[SimpleNamespace(keys=tensor.clone(), values=tensor.clone()) for _ in range(2)]
    )


class CacheTests(unittest.TestCase):
    def test_residual_strength_and_donor_preservation(self):
        for dtype in (torch.float32, torch.bfloat16):
            main, alternative = make_cache(1), make_cache(9, 5)
            for cache in (main, alternative):
                for layer in cache.layers:
                    layer.keys = layer.keys.to(dtype)
                    layer.values = layer.values.to(dtype)
            saved = copy.deepcopy(alternative)
            swap_kv_segment(main, alternative, 2, 2, strength=0.25)
            for layer, donor, original in zip(main.layers, alternative.layers, saved.layers):
                for name in ("keys", "values"):
                    tensor = getattr(layer, name)
                    self.assertEqual(tensor.dtype, dtype)
                    self.assertTrue(torch.all(tensor[:, :, :2] == 1))
                    self.assertTrue(torch.all(tensor[:, :, 2:4] == 3))
                    self.assertTrue(torch.all(tensor[:, :, 4:] == 1))
                    self.assertTrue(torch.equal(getattr(donor, name), getattr(original, name)))

    def test_zero_strength_is_bitwise_identity(self):
        main, alternative = make_cache(1), make_cache(9)
        swap_kv_segment(main, alternative, 2, 2, strength=0)
        for layer in main.layers:
            self.assertTrue(torch.all(layer.keys == 1))
            self.assertTrue(torch.all(layer.values == 1))

    def test_strength_rejects_invalid_values_before_mutation(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            main = make_cache(1)
            with self.subTest(value=value), self.assertRaises(ValueError):
                swap_kv_segment(main, make_cache(9), 2, 2, strength=value)
            self.assertTrue(torch.all(main.layers[0].keys == 1))
            spec = load_spec()
            spec["generation"]["swap_strength"] = value
            with self.assertRaises(ValueError):
                validate_spec(spec)

    def test_only_selected_segment_changes(self):
        main, alternative = make_cache(1), make_cache(9, 5)
        swap_kv_segment(main, alternative, 2, 2)
        for layer in main.layers:
            for tensor in (layer.keys, layer.values):
                self.assertTrue(torch.all(tensor[:, :, :2, :] == 1))
                self.assertTrue(torch.all(tensor[:, :, 2:4, :] == 9))
                self.assertTrue(torch.all(tensor[:, :, 4:, :] == 1))

    def test_invalid_later_layer_does_not_partially_write(self):
        main, alternative = make_cache(1), make_cache(9)
        alternative.layers[1].keys = torch.zeros((1, 3, 8, 3))
        with self.assertRaises(ValueError):
            swap_kv_segment(main, alternative, 2, 2)
        self.assertTrue(torch.all(main.layers[0].keys == 1))

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            swap_kv_segment(make_cache(1), make_cache(9), 7, 2)

    def test_rejects_sliding_cache(self):
        main, alternative = make_cache(1), make_cache(9)
        alternative.layers[0].is_sliding = True
        with self.assertRaises(ValueError):
            swap_kv_segment(main, alternative, 2, 2)
        self.assertTrue(torch.all(main.layers[0].keys == 1))


if __name__ == "__main__":
    unittest.main()

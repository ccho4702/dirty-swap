import unittest
from types import SimpleNamespace

import torch

from dirty_swapping.cache import swap_kv_segment


def make_cache(value, length=8):
    tensor = torch.full((1, 2, length, 3), value, dtype=torch.float32)
    return SimpleNamespace(
        layers=[SimpleNamespace(keys=tensor.clone(), values=tensor.clone()) for _ in range(2)]
    )


class CacheTests(unittest.TestCase):
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

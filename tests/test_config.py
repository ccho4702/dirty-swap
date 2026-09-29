import copy
import unittest

from dirty_swapping.config import load_spec, validate_spec


class ConfigTests(unittest.TestCase):
    def test_default_and_smoke_are_valid(self):
        self.assertEqual(load_spec()["protocol"], "dirty-swapping-v1")
        self.assertEqual(load_spec("configs/smoke.json")["generation"]["max_new_tokens"], 64)

    def test_invalid_caps_and_selector_fail(self):
        original = load_spec()
        for change in (
            {"max_new_tokens": original["model"]["max_context_tokens"] + 1},
            {"max_new_tokens": 100},
            {"candidate_count": 1},
            {"alternative_selection": "gold_answer"},
        ):
            spec = copy.deepcopy(original)
            spec["generation"].update(change)
            with self.assertRaises(ValueError):
                validate_spec(spec)


if __name__ == "__main__":
    unittest.main()

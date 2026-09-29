import unittest

from dirty_swapping.scoring import score
from dirty_swapping.math_scoring import boxed_answer, math_correct


class ScoringTests(unittest.TestCase):
    def test_official_math_answer_extraction(self):
        self.assertEqual(boxed_answer("Therefore \\boxed{4}."), "4")
        self.assertTrue(math_correct("4", "4"))
        self.assertFalse(math_correct("4", "5"))

    def test_numeric_and_choice(self):
        gold = [
            {"id": "a", "task": "gsm8k", "answer": "work #### 1,234"},
            {"id": "b", "task": "gpqa", "answer": "B"},
        ]
        pred = [
            {"id": "a", "answer": "1234", "latency_s": 2, "swap_applied": True},
            {"id": "b", "answer": "C", "latency_s": 4, "swap_applied": False},
        ]
        result = score(gold, pred)
        self.assertEqual(result["micro_accuracy"], 0.5)
        self.assertEqual(result["mean_latency_s"], 3)

    def test_missing_and_duplicate_ids_fail(self):
        gold = [{"id": "a", "task": "aime25", "answer": "1"}]
        with self.assertRaises(ValueError):
            score(gold, [])
        with self.assertRaises(ValueError):
            score(gold, [{"id": "a"}, {"id": "a"}])


if __name__ == "__main__":
    unittest.main()

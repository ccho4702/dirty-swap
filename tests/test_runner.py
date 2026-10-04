import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dirty_swapping.config import load_spec, runtime_spec
from dirty_swapping.runner import create_run, execute, report, run_case, selected_rows


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "generated"


class FakeBackend:
    def __init__(self, config):
        self.tokenizer = FakeTokenizer()

    def synchronize(self):
        pass

    def prefill(self, row):
        tensor = torch.zeros((1, 1, 3, 1))
        return SimpleNamespace(
            ids=[1, 2, 3],
            prompt_length=3,
            cache=SimpleNamespace(
                layers=[SimpleNamespace(keys=tensor.clone(), values=tensor.clone())]
            ),
        )

    def fork(self, state):
        layer = state.cache.layers[0]
        return SimpleNamespace(
            ids=list(state.ids),
            prompt_length=state.prompt_length,
            cache=SimpleNamespace(
                layers=[SimpleNamespace(keys=layer.keys.clone(), values=layer.values.clone())]
            ),
        )

    def sequence_length(self, state):
        return len(state.ids)

    def top_tokens(self, state, count):
        return [7, 8, 9, 10][:count]

    def step(self, state, token):
        state.ids.append(token)
        layer = state.cache.layers[0]
        value = torch.tensor([token], dtype=torch.float32).reshape(1, 1, 1, 1)
        layer.keys = torch.cat((layer.keys, value), dim=2)
        layer.values = torch.cat((layer.values, value), dim=2)
        return state

    def rollout(self, state, first_token, total_tokens):
        self.step(state, first_token)
        return self.advance(state, total_tokens - 1)

    def advance(self, state, token_count):
        for _ in range(token_count):
            self.step(state, 7)
        return state

    def cache(self, state):
        return state.cache

    def reasoning_end(self, state):
        return None

    def finished(self, state):
        return len(state.ids) - state.prompt_length >= 7

    def decode_answer(self, state):
        return "Final answer: (B)"


class InterruptAfterFirstCase(FakeBackend):
    def __init__(self, config):
        super().__init__(config)
        self.prefills = 0

    def prefill(self, row):
        self.prefills += 1
        if self.prefills == 2:
            raise KeyboardInterrupt("test interruption")
        return super().prefill(row)


class NoAnswerBackend(FakeBackend):
    def decode_answer(self, state):
        return ""


class NumericBackend(FakeBackend):
    def __init__(self, config, eligible=True):
        super().__init__(config)
        self.eligible = eligible
        self.scans = 0

    def numeric_ambiguity(self, state, min_ratio):
        self.scans += 1
        return {"policy": "numeric_ambiguity", "top2_ratio": 0.5} if self.eligible else None

    def decode_answer(self, state):
        return "\\boxed{4}"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec()
        self.spec["generation"].update(
            max_input_tokens=8,
            max_new_tokens=8,
            branch_after_reasoning_tokens=2,
            rollout_tokens=2,
            delay_tokens=2,
        )
        self.spec["data"]["prompt_limit"] = 8
        self.spec["model"]["max_context_tokens"] = 16
        self.row = {
            "id": "gpqa-test",
            "task": "gpqa",
            "question": "Q",
            "gold": "B",
            "choices": ["A", "B", "C", "D"],
            "answer_format": "choice",
            "prompt_tokens": 3,
        }

    def test_case_runs_both_arms(self):
        backend = FakeBackend({})
        baseline = run_case(backend, self.row, self.spec, "baseline")
        swapped = run_case(backend, self.row, self.spec, "swap")
        self.assertTrue(baseline["correct"])
        self.assertTrue(swapped["correct"])
        self.assertTrue(baseline["answer_extracted"])
        self.assertTrue(swapped["answer_extracted"])
        self.assertFalse(baseline["swap"]["swapped"])
        self.assertTrue(swapped["swap"]["swapped"])
        self.assertEqual(swapped["swap"]["candidate_count_evaluated"], 4)

    def test_missing_final_answer_is_incorrect(self):
        result = run_case(NoAnswerBackend({}), self.row, self.spec, "baseline")
        self.assertFalse(result["answer_extracted"])
        self.assertIsNone(result["prediction"])
        self.assertFalse(result["correct"])

    def test_numeric_scan_has_one_event_and_no_trigger_fallback(self):
        self.spec["generation"].update(
            branch_policy="numeric_ambiguity", branch_scan_end=2, min_top2_ratio=0.1
        )
        row = {**self.row, "answer_format": "math", "gold": "4", "choices": []}
        backend = NumericBackend({})
        result = run_case(backend, row, self.spec, "swap")
        self.assertTrue(result["swap"]["swapped"])
        self.assertEqual(result["branch_position"], 5)
        self.assertEqual(backend.scans, 1)
        backend = NumericBackend({}, eligible=False)
        result = run_case(backend, row, self.spec, "swap")
        self.assertFalse(result["swap"]["swapped"])
        self.assertEqual(result["swap"]["reason"], "no_numeric_ambiguity")
        self.assertIsNone(result["branch_position"])
        self.assertEqual(backend.scans, 1)

    def test_explicit_cohort_preserves_order_and_rejects_missing_split_ids(self):
        rows = [{**self.row, "id": "a"}, {**self.row, "id": "b"}]
        self.spec["execution"]["case_ids"] = {"gpqa": ["b", "a"]}
        with patch("dirty_swapping.runner.load_prepared", return_value=rows):
            selected = selected_rows(Path("."), self.spec, ["gpqa"], None)
            self.assertEqual([row["id"] for row in selected], ["b", "a"])
            self.spec["execution"]["case_ids"] = {"gpqa": ["missing"]}
            with self.assertRaises(ValueError):
                selected_rows(Path("."), self.spec, ["gpqa"], None)

    def test_explicit_cohort_never_claims_full_default_suite(self):
        tasks = self.spec["data"]["default_datasets"]
        self.spec["execution"]["case_ids"] = {task: [task + "-one"] for task in tasks}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            spec = runtime_spec(self.spec, work, device="cpu")

            def prepared(work_dir, config, task):
                return [{**self.row, "task": task, "id": task + "-one"}]

            with patch("dirty_swapping.runner.load_prepared", side_effect=prepared):
                run = create_run(work, spec, tasks, run_name="explicit-subset")
                result = execute(run, work, backend_factory=FakeBackend)
                self.assertTrue(result["complete"])
                self.assertFalse(result["default_suite_complete"])
                self.assertIsNone(result["primary_macro_accuracy"])

    def test_run_resume_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            spec = runtime_spec(self.spec, work, device="cpu")
            with patch("dirty_swapping.runner.load_prepared", return_value=[self.row]):
                run = create_run(work, spec, ["gpqa"], limit=1, run_name="test-run")
                first = execute(run, work, backend_factory=FakeBackend)
                self.assertTrue(first["complete"])
                self.assertEqual(first["common_completed"], 1)
                self.assertEqual(first["by_task"]["gpqa"]["arms"]["swap"]["answer_extracted"], 1)
                second = execute(run, work, backend_factory=FakeBackend)
                self.assertTrue(second["complete"])
                self.assertEqual(report(run)["by_task"]["gpqa"]["arms"]["swap"]["correct"], 1)
                path = next((run / "cases" / "swap").glob("*.json"))
                saved = json.loads(path.read_text())
                saved["payload"]["correct"] = False
                path.write_text(json.dumps(saved))
                with self.assertRaises(ValueError):
                    execute(run, work, backend_factory=FakeBackend)

    def test_interruption_reuses_completed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            spec = runtime_spec(self.spec, work, device="cpu")
            with patch("dirty_swapping.runner.load_prepared", return_value=[self.row]):
                run = create_run(work, spec, ["gpqa"], limit=1, run_name="interrupted")
                with self.assertRaises(KeyboardInterrupt):
                    execute(run, work, backend_factory=InterruptAfterFirstCase)
                baseline = next((run / "cases" / "baseline").glob("*.json"))
                original = baseline.read_bytes()
                self.assertEqual(
                    json.loads((run / "status.json").read_text())["state"], "interrupted"
                )
                partial = report(run)
                self.assertFalse(partial["complete"])
                self.assertIsNone(partial["primary_macro_accuracy"])
                result = execute(run, work, backend_factory=FakeBackend)
                self.assertTrue(result["complete"])
                self.assertEqual(baseline.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

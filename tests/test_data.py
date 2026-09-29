import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from dirty_swapping.data import raw_rows, sources


class DatasetTests(unittest.TestCase):
    def test_default_sources_have_expected_tasks(self):
        self.assertTrue({"aime25", "hmmt25", "gsm8k", "gpqa"} <= set(sources()))

    def test_gsm8k_split_and_final_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs" / "gsm8k" / "test-revision"
            root.mkdir(parents=True)
            (root / "train.jsonl").write_text(
                json.dumps({"question": "train Q", "answer": "work #### 1,234"}) + "\n"
            )
            (root / "test.jsonl").write_text(
                json.dumps({"question": "test Q", "answer": "work #### 5"}) + "\n"
            )
            manifest = {
                "gsm8k": {
                    "revision": "test-revision",
                    "files": {"train.jsonl": "a", "test.jsonl": "b"},
                }
            }
            with (
                patch("dirty_swapping.data.sources", return_value=manifest),
                patch(
                    "dirty_swapping.data.file_digest",
                    side_effect=lambda path: "a" if path.name == "train.jsonl" else "b",
                ),
            ):
                rows = list(raw_rows(directory, "gsm8k", 18))
            self.assertEqual([row["split"] for row in rows], ["development", "evaluation"])
            self.assertEqual([row["gold"] for row in rows], ["1234", "5"])

    def test_matharena_parquet_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs" / "aime25" / "test-revision" / "data"
            root.mkdir(parents=True)
            pq.write_table(pa.table({"problem_idx": [1], "problem": ["2+2?"],
                                     "answer": [4], "problem_type": ["algebra"]}),
                           root / "train-00000-of-00001.parquet")
            manifest = {"aime25": {"revision": "test-revision", "files": {
                "data/train-00000-of-00001.parquet": "a"}}}
            with patch("dirty_swapping.data.sources", return_value=manifest), patch("dirty_swapping.data.file_digest", return_value="a"):
                rows = list(raw_rows(directory, "aime25", 18))
            self.assertEqual(rows[0]["gold"], "4")
            self.assertEqual(rows[0]["id"], "aime25-1")

    def test_math500_jsonl_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs" / "math500" / "test-revision"
            root.mkdir(parents=True)
            (root / "test.jsonl").write_text(
                json.dumps({
                    "unique_id": "test/algebra/1.json",
                    "problem": "What is 2 + 2?",
                    "answer": "4",
                    "solution": "\\boxed{4}",
                    "subject": "Algebra",
                    "level": 1,
                }) + "\n"
            )
            manifest = {"math500": {"revision": "test-revision", "files": {"test.jsonl": "a"}}}
            with patch("dirty_swapping.data.sources", return_value=manifest), patch(
                "dirty_swapping.data.file_digest", return_value="a"
            ):
                rows = list(raw_rows(directory, "math500", 18))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "math500-test/algebra/1.json")
            self.assertEqual(rows[0]["gold"], "4")
            self.assertEqual(rows[0]["answer_format"], "math")

    def test_gpqa_main_adapter_and_deterministic_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs" / "gpqa" / "test-revision"
            root.mkdir(parents=True)
            main = ("Record ID,Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,"
                    "Incorrect Answer 3,High-level domain\nrecord-1,Which?,right,wrong1,wrong2,wrong3,science\n")
            diamond = "Record ID\nrecord-1\n"
            with zipfile.ZipFile(root / "dataset.zip", "w") as archive:
                archive.writestr("dataset/gpqa_main.csv", main)
                archive.writestr("dataset/gpqa_diamond.csv", diamond)
            manifest = {"gpqa": {"revision": "test-revision", "files": {"dataset.zip": "a"},
                                 "public_archive_password": "public"}}
            with patch("dirty_swapping.data.sources", return_value=manifest), patch("dirty_swapping.data.file_digest", return_value="a"):
                first = list(raw_rows(directory, "gpqa", 18))[0]
                second = list(raw_rows(directory, "gpqa", 18))[0]
            self.assertEqual(first, second)
            self.assertTrue(first["diamond"])
            self.assertIn(f"({first['gold']}) right", first["question"])


if __name__ == "__main__":
    unittest.main()

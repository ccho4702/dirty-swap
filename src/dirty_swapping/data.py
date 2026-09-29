"""Pinned downloads and deterministic, answer-safe benchmark adapters."""

from __future__ import annotations

import csv
import io
import json
from importlib.resources import files
from pathlib import Path
import random
import time
import urllib.error
import urllib.request
import zipfile

from .core import atomic_json, digest, file_digest
from .prompt import instruction


def sources():
    return json.loads(files("dirty_swapping").joinpath("datasets.json").read_text())


def download_file(url, path, expected_sha, log=print):
    """Resume this task's partial download; only publish hash-verified bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_digest(path) != expected_sha:
            raise ValueError(f"Existing input failed SHA256 verification: {path}")
        log(f"Verified {path.name}")
        return path
    partial = path.with_name(path.name + ".part")
    if partial.exists() and file_digest(partial) == expected_sha:
        partial.replace(path)
        log(f"Published verified partial {path.name}")
        return path
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                total = offset + int(response.headers.get("Content-Length", 0))
                started = last = time.monotonic()
                count = offset
                with partial.open("ab" if append else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last >= 5:
                            elapsed = time.monotonic() - started
                            eta = (
                                (total - count) * elapsed / max(1, count - offset)
                                if total
                                else None
                            )
                            log(
                                f"Download {path.name}: {count / 1024**2:.1f} MiB; ETA {eta:.0f}s"
                                if eta is not None
                                else f"Download {path.name}: {count / 1024**2:.1f} MiB"
                            )
                            last = time.monotonic()
            if file_digest(partial) != expected_sha:
                partial.unlink()
                raise ValueError(f"Download failed SHA256 verification: {url}")
            partial.replace(path)
            log(f"Downloaded and verified {path.name}")
            return path
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Download did not complete")


def download_datasets(work_dir, benchmarks):
    manifest = sources()
    for task in benchmarks:
        source = manifest[task]
        root = Path(work_dir) / "inputs" / task / source["revision"]
        for name, sha in source["files"].items():
            url = source.get("urls", {}).get(name) or source["url_prefix"] + name
            download_file(url, root / name, sha)
        atomic_json(root / "source.json", source)


def download_models(spec):
    from huggingface_hub import snapshot_download

    for cfg in (spec["model"],):
        print(f"Download/verify model {cfg['name']} @ {cfg['revision']}", flush=True)
        snapshot_download(
            cfg["name"],
            revision=cfg["revision"],
            cache_dir=cfg["cache_dir"],
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
            token=False,
        )


def choice_row(task, ident, question, options, gold_index, **metadata):
    letters = list("ABCDEFGHIJ"[: len(options)])
    if not 2 <= len(options) <= 10 or not 0 <= gold_index < len(options):
        raise ValueError("Invalid answer options")
    return {
        "id": f"{task}-{ident}",
        "task": task,
        "question": question + "\n\n" + "\n".join(f"({a}) {b}" for a, b in zip(letters, options)),
        "choices": letters,
        "gold": letters[gold_index],
        "answer_format": "choice",
        **metadata,
    }


def raw_rows(work_dir, task, seed):
    source = sources()[task]
    root = Path(work_dir) / "inputs" / task / source["revision"]
    for name, sha in source["files"].items():
        if not (root / name).exists() or file_digest(root / name) != sha:
            raise ValueError(f"Missing or corrupt {task} input; run dirty-swapping download first")
    if task in ("aime25", "hmmt25"):
        import pyarrow.parquet as pq

        for row in pq.read_table(root / "data/train-00000-of-00001.parquet").to_pylist():
            yield {
                "id": f"{task}-{row['problem_idx']}",
                "task": task,
                "question": row["problem"],
                "gold": str(row["answer"]),
                "choices": [],
                "answer_format": "math",
                "problem_type": row["problem_type"],
            }
    elif task == "math500":
        with (root / "test.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                yield {
                    "id": f"math500-{row['unique_id']}",
                    "task": task,
                    "question": row["problem"],
                    "gold": row["answer"],
                    "choices": [],
                    "answer_format": "math",
                    "subject": row["subject"],
                    "level": row["level"],
                }
    elif task == "supergpqa":
        with (root / "SuperGPQA-all.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                index = "ABCDEFGHIJ".index(row["answer_letter"])
                if row["options"][index].strip() != row["answer"].strip():
                    raise ValueError("SuperGPQA answer/option mismatch")
                yield choice_row(
                    task,
                    row["uuid"],
                    row["question"],
                    row["options"],
                    index,
                    discipline=row["discipline"],
                    difficulty=row["difficulty"],
                )
    elif task == "longbench_v2":
        for row in json.loads((root / "data.json").read_text()):
            yield choice_row(
                task,
                row["_id"],
                row["question"],
                [row["choice_" + a] for a in "ABCD"],
                "ABCD".index(row["answer"]),
                context=row["context"],
                domain=row["domain"],
                difficulty=row["difficulty"],
                length=row["length"],
            )
    elif task == "gpqa":
        with zipfile.ZipFile(root / "dataset.zip") as archive:

            def read(name):
                data = archive.read(
                    "dataset/" + name, pwd=source["public_archive_password"].encode()
                )
                return list(csv.DictReader(io.StringIO(data.decode())))

            diamond = {r["Record ID"] for r in read("gpqa_diamond.csv")}
            for row in read("gpqa_main.csv"):
                options = [row["Correct Answer"]] + [
                    row[f"Incorrect Answer {i}"] for i in range(1, 4)
                ]
                order = list(range(4))
                random.Random(f"{seed}:{row['Record ID']}").shuffle(order)
                yield choice_row(
                    task,
                    row["Record ID"],
                    row["Question"],
                    [options[i] for i in order],
                    order.index(0),
                    diamond=row["Record ID"] in diamond,
                    domain=row["High-level domain"],
                )
    elif task == "gsm8k":
        for split in ("train", "test"):
            with (root / f"{split}.jsonl").open() as stream:
                for index, line in enumerate(stream):
                    item = json.loads(line)
                    yield {
                        "id": f"gsm8k-{split}-{index}",
                        "task": "gsm8k",
                        "question": item["question"],
                        "gold": item["answer"].rsplit("####", 1)[1].strip().replace(",", ""),
                        "choices": [],
                        "answer_format": "math",
                        "split": "development" if split == "train" else "evaluation",
                    }


def normalize(row, tokenizer, prompt_limit):
    row = dict(row)

    def length(value):
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction(value) + value["question"]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    if row["task"] == "longbench_v2":
        context = tokenizer.encode(row.pop("context"), add_special_tokens=False)
        suffix = "\n\nQuestion:\n" + row["question"]
        reserve = length({**row, "question": "Context:\n" + suffix}) + 32
        keep = min(len(context), prompt_limit - reserve)
        if keep <= 0:
            raise ValueError("Question/options alone exceed prompt limit")
        while True:
            retained = (
                context
                if keep == len(context)
                else context[: keep // 2] + context[-(keep - keep // 2) :]
            )
            row["question"] = (
                "Context:\n" + tokenizer.decode(retained, skip_special_tokens=True) + suffix
            )
            n = length(row)
            if n <= prompt_limit:
                break
            keep -= n - prompt_limit + 16
            if keep <= 0:
                raise ValueError("Cannot fit complete question/options")
        row.update(
            original_context_tokens=len(context),
            retained_context_tokens=keep,
            context_truncated=keep < len(context),
        )
    row["prompt_tokens"] = length(row)
    if row["prompt_tokens"] > prompt_limit:
        raise ValueError(
            f"{row['id']} exceeds prompt limit; refusing to truncate its question/options"
        )
    return row


def preparation_identity(spec, task):
    return {
        "source": sources()[task],
        "model": {k: spec["model"][k] for k in ("name", "revision", "enable_thinking")},
        "data": spec["data"],
        "adapter_sha256": file_digest(Path(__file__)),
    }


def prepared_path(work_dir, spec, task):
    return Path(work_dir) / "intermediates" / task / digest(preparation_identity(spec, task))[:16]


def prepare(work_dir, spec, benchmarks):
    from transformers import AutoTokenizer

    cfg = spec["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["name"],
        token=False,
        trust_remote_code=False,
        **{k: cfg[k] for k in ("revision", "cache_dir", "local_files_only")},
    )
    for task in benchmarks:
        out = prepared_path(work_dir, spec, task)
        if (out / "validation.json").exists():
            load_prepared(work_dir, spec, task)
            print(f"Verified prepared {task}", flush=True)
            continue
        identity = preparation_identity(spec, task)
        rows = list(raw_rows(work_dir, task, spec["data"]["shuffle_seed"]))
        expected = {
            "aime25": 30,
            "hmmt25": 30,
            "gpqa": 448,
            "supergpqa": 26529,
            "longbench_v2": 503,
            "gsm8k": 8792,
            "math500": 500,
        }[task]
        if len(rows) != expected or len({r["id"] for r in rows}) != expected:
            raise ValueError("Unexpected dataset cardinality or duplicate IDs")
        completed = []
        started = last = time.monotonic()
        for i, row in enumerate(rows):
            cache = out / "rows" / (digest(row["id"]) + ".json")
            if cache.exists():
                saved = json.loads(cache.read_text())
                if saved["raw_sha256"] != digest(row) or saved["row_sha256"] != digest(
                    saved["row"]
                ):
                    raise ValueError("Corrupt normalized row checkpoint")
                normalized = saved["row"]
            else:
                normalized = normalize(row, tokenizer, spec["data"]["prompt_limit"])
                atomic_json(
                    cache,
                    {
                        "raw_sha256": digest(row),
                        "row": normalized,
                        "row_sha256": digest(normalized),
                    },
                )
            completed.append(normalized)
            if i == 0 or time.monotonic() - last >= 10 or i + 1 == len(rows):
                elapsed = time.monotonic() - started
                print(
                    f"Prepare {task}: {i + 1}/{len(rows)}; ETA {elapsed / (i + 1) * (len(rows) - i - 1):.0f}s",
                    flush=True,
                )
                last = time.monotonic()
        random.Random(f"{spec['data']['shuffle_seed']}:{task}").shuffle(completed)
        atomic_json(out / "rows.json", completed)
        atomic_json(
            out / "validation.json",
            {
                "identity": identity,
                "rows_sha256": digest(completed),
                "n": len(completed),
                "max_prompt_tokens": max(r["prompt_tokens"] for r in completed),
                "truncated": sum(r.get("context_truncated", False) for r in completed),
            },
        )


def load_prepared(work_dir, spec, task):
    root = prepared_path(work_dir, spec, task)
    if not (root / "validation.json").exists():
        raise ValueError(f"{task} is not prepared for this configuration; run dirty-swapping setup")
    meta = json.loads((root / "validation.json").read_text())
    rows = json.loads((root / "rows.json").read_text())
    if meta["identity"] != preparation_identity(spec, task) or digest(rows) != meta["rows_sha256"]:
        raise ValueError("Prepared data integrity check failed")
    return rows

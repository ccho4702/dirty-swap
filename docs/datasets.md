# Dataset catalog and preprocessing

The dataset manifest at [`src/dirty_swapping/datasets.json`](../src/dirty_swapping/datasets.json) pins source revisions and SHA256 hashes. `dirty-swapping download` saves verified raw files under `inputs/`; `prepare` writes normalized, hash-checked rows under `intermediates/`. Neither directory is committed. Source licenses and access terms remain with the publishers.

## Supported pool

| CLI key | Source and raw format | Prepared rows / split | Answer and role |
| --- | --- | --- | --- |
| `aime25` | [MathArena AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025), Parquet | 30 evaluation | Math answer; default suite |
| `hmmt25` | [MathArena HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025), Parquet | 30 evaluation | Math answer; default suite |
| `gsm8k` | [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k), JSONL | 7,473 development + 1,319 evaluation | Final value after `####`; default suite |
| `gpqa` | [GPQA Main](https://github.com/idavidrein/gpqa), public ZIP/CSV | 448 evaluation | Deterministically shuffled A–D choice; default suite |
| `math500` | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), JSONL | 500 evaluation | Math answer; additional math track |
| `supergpqa` | [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA), JSONL | 26,529 evaluation | Choice answer; larger-scale additional track |
| `longbench_v2` | [LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2), JSON | 503 evaluation | Choice answer; capped-context additional track |

The four default datasets are AIME25, HMMT25, GSM8K **test**, and GPQA Main. `math500`, `supergpqa`, and `longbench_v2` have working download and preparation adapters but are reported separately from the default four-dataset macro score. Use `--datasets` to select a subset. `gsm8k --split development` selects its train questions; the other datasets are evaluation-only in this implementation. Public test questions remain public even if an organizer hides labels in a competition system.

## Normalized row contract

Every prepared row has `id`, `task`, `question`, `gold`, `choices`, `answer_format`, and `prompt_tokens`. Optional metadata such as subject, difficulty, or truncation counts stays with the row. Only the formatted **question and visible options** enter the model prompt, prefixed by the shared instruction in [`prompt.py`](../src/dirty_swapping/prompt.py). `gold`, GSM8K worked solutions, MATH-500 `solution`, and other answer metadata never enter the prompt or alternative selection.

- AIME25/HMMT25 read `problem`, `answer`, `problem_idx`, and `problem_type` from Parquet; `answer` becomes math gold.
- GSM8K reads `question` and `answer` from the two JSONL splits. The answer portion after the final `####` becomes gold; the preceding worked solution is discarded for inference.
- GPQA Main reads `gpqa_main.csv` from the authors' public archive. Options are shuffled by the configured seed and stable record ID, and the gold letter is updated to match. The archive's public password is recorded in the manifest.
- MATH-500 reads `problem`, `answer`, `unique_id`, `subject`, and `level` from its pinned `test.jsonl`. `solution` is intentionally omitted from the normalized prompt.
- SuperGPQA checks that `answer_letter` agrees with the indexed option text before producing a choice row.
- LongBench v2 keeps the entire question and choices. If its context does not fit, it retains the context head and tail, records original/retained token counts, and marks `context_truncated=true`.

Preparation checks source hashes, expected row counts, unique IDs, and tokenized prompt lengths. With the current default **4,608-token input cap**, every AIME25, HMMT25, GSM8K, GPQA, MATH-500, and SuperGPQA question fits without truncating its question or options. SuperGPQA's longest observed prepared prompt was 4,108 tokens. **All 503 LongBench v2 contexts were truncated** at this cap; these results must be called a capped-context evaluation, not the official full-context score. A malformed or over-cap question fails preparation rather than being silently shortened.

For reproducible comparisons, keep the same prepared row IDs, option order, prompt, model/tokenizer revision, and input/output caps in both arms. Ground-truth answers are used only after generation for scoring. The exact answer extraction and comparison rules are in [the evaluation protocol](competition.md).

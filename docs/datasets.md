# Datasets

## Default evaluation suite

| Key | Source | Task and answer | Suggested split policy |
| --- | --- | --- | --- |
| `aime25` | [MathArena AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | Competition math; integer answer | Final evaluation; tune on other math sets |
| `hmmt25` | [MathArena HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | Competition math; numeric or symbolic answer | Final evaluation; no same-question tuning |
| `gsm8k` | [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Multi-step arithmetic; final value after `####` | Use train for development, test for final evaluation |
| `gpqa` | [GPQA authors' repository](https://github.com/idavidrein/gpqa) | GPQA Main, graduate-level scientific multiple choice; A–D after deterministic option shuffle | Final evaluation; honor original access terms |

These are the four datasets in the proposal. The included `src/dirty_swapping/datasets.json` pins the AIME/HMMT/GPQA revisions from [lost-tokens](https://github.com/ccho4702/lost-tokens/blob/main/src/lost_tokens/datasets.json). GSM8K uses the original public JSONL files with exact SHA256 checks; its `master` URLs may move, but changed bytes will fail verification. Prompt template, option order, and normalized row IDs are deterministic. Do not commit downloaded source data or answer-bearing hidden evaluation files. Public benchmark questions are visible online, so a private competition may need newly collected or organizer-held questions for a genuinely hidden final set.

## Recommended additions

| Dataset | Why add it | Role |
| --- | --- | --- |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Broader math difficulty and symbolic answers | Development or separate final track |
| [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) | Larger scientific and academic QA coverage; adapter included | Scalability track |
| [LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | Long-context reasoning, where stale suffix effects may differ; adapter included with documented input cap | Optional long-context track |

Do not silently truncate long-context examples. The included LongBench v2 adapter keeps context head and tail to fit the configured input cap and records truncation metadata; report this as a capped-context evaluation. The official runner uses a pinned `math-verify` version for math and explicit option matching for multiple choice. The separate toy JSONL scorer only handles simple numeric and choice answers.

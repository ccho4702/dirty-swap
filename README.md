# Dirty Swapping — Group 18 IDL baseline

**처음 시작한다면, 레포 루트에서 이 명령 하나를 실행하세요:**

```bash
./start.sh
```

설치나 다운로드 없이 이 레포의 자료 지도와 첫 실행 명령을 보여줍니다. 자세한 설명은 [한국어 시작 가이드](docs/START_HERE.ko.md), 에이전트용 안내는 [온보딩 SKILL.md](skills/onboard-dirty-swapping/SKILL.md)에 있습니다.

This repository provides a minimal, runnable reference for the [Group 18 proposal](docs/proposal.pdf): **improve final-answer accuracy on text QA by replacing one past KV-cache segment with a precomputed counterfactual segment**. The model's weights, already generated token IDs, and all KV positions after that segment remain unchanged. Baseline and swap runs use the same frozen Qwen3-4B-Thinking-2507 model and greedy main path. Accuracy may or may not improve; this repository defines a reproducible starting point for a competition-style project.

The implementation reuses and adapts dataset downloading, normalization, answer parsing, and atomic-output patterns from [lost-tokens](https://github.com/ccho4702/lost-tokens) commit `ca80edc24aa4f4b86e9b162581a4d3be0bf26cf9`. The KV intervention and paired runner live in `src/dirty_swapping/`. See [method](docs/method.md) and [source attribution](docs/implementation.md).

## What runs

`dirty-swapping setup` downloads hash-checked datasets and the pinned model, then prepares deterministic rows. `dirty-swapping run` evaluates **baseline** and **swap** on the same IDs. Each completed case is written atomically; `resume` reuses those cases with the saved resolved configuration and rejects changed source, prepared data, dependency lock, or runtime. `report` computes paired metrics. CPU tests use a tiny causal model to verify that a swapped KV span changes later generation while suffix KV stays intact.

The first implementation uses one intervention per trace: at reasoning token 256, generate four 32-token continuations from the same prefix. The top-1 continuation is committed. After 128 more main-path tokens, replace its 32 KV positions with the top-2 continuation's KV, across all attention layers. If reasoning or generation ends too early, record a skip reason. The first token after the edit uses logits computed before the edit; the following token can reflect the edited cache. No suffix forward pass is repeated.

## Environment

Use Linux, Python 3.11–3.13, `uv >= 0.8`, and a project-local `.venv`. GPU evaluation needs an allocated NVIDIA GPU and compatible driver. A 4B model plus candidate caches can need substantial memory; begin with one short example. Model and dataset downloads require internet access. Downloaded data, weights, logs, and outputs stay in the repository's ignored `inputs/`, `models/`, `intermediates/`, `outputs/`, and `logs/` directories by default.

```bash
cd dirty-swapping
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v

# Small end-to-end Qwen run; first setup downloads model and GSM8K.
uv run --frozen dirty-swapping setup --datasets gsm8k
uv run --frozen dirty-swapping run --datasets gsm8k --limit 1 \
  --config configs/smoke.json --run-name gsm8k-smoke
uv run --frozen dirty-swapping report --run outputs/gsm8k-smoke
```

The smoke configuration has a 64-token output cap and is **not** a benchmark setting. The default benchmark configuration permits 32,768 generated tokens so a long `<think>` trace is not automatically marked wrong at 4,096 tokens; this can take substantial time. To run the proposed default suite, omit `--datasets` and `--config` in `setup` and `run`; use a unique run name and no `--limit`. If the process stops, run `uv run --frozen dirty-swapping resume --run outputs/<run-name>`. Run `uv run --frozen dirty-swapping config > my-config.json` to inspect or edit settings; pass `--config my-config.json` to `setup` and `run`. Device selection is `--device cuda:0` or `--device cpu`. `--model-cache` can reuse an existing model cache. The model revision and dataset file hashes are pinned.

This setup was verified on an RTX 3090: `uv sync --frozen`, tests, lint, wheel build, hash-checked preparation of all seven supported datasets, and paired GPU smoke examples completed. On one example from **each of the seven datasets**, baseline exactly matched ordinary autoregressive decoding for all 256 generated tokens; GSM8K also matched Hugging Face `model.generate`. The swap path ran on all seven without OOM; LongBench's long input reached about 11 GiB peak allocation. A short GSM8K question was answered correctly by both arms. A short MATH-500 question was correct in both arms with a 4,096-token cap, but at 2,048 tokens swap had not yet produced a final answer while baseline had. These small checks establish functioning paths and a real fixed-budget failure case, **not a full-dataset accuracy-preservation result**. See [verification details](docs/implementation.md). GPU kernels can introduce small nondeterminism even with a fixed seed.

## Data

The **default suite** is AIME 2025, HMMT February 2025, GSM8K test, and GPQA Main. GSM8K train is a disjoint development split; run it with `--datasets gsm8k --split development`. AIME/HMMT/GPQA are evaluation-only in this baseline. **MATH-500, SuperGPQA, and LongBench v2** are supported additional tracks and are reported separately from the default macro score. Dataset source URLs, revisions, SHA256 hashes, raw formats, and preprocessing rules are in the [data catalog](docs/datasets.md). The default input cap is 4,608 tokens; all 503 LongBench v2 contexts are shortened under that cap while their questions and options remain intact. Weights and source data are not redistributed.

## Metrics and competition use

The primary metric is **macro final-answer accuracy** over all four default datasets, with correct/total and accuracy for each dataset. Subset runs report a selected-dataset macro score but no primary score. Paired baseline-vs-swap accuracy difference (percentage points), wins/losses, mean/median/p95 per-example latency, alternative-rollout time, cache-copy time, peak GPU memory, and swap rate are reported. Latency includes tokenization, prefill, all rollouts, the swap, and final generation. Reports use only IDs completed by both arms; partial reports are labeled incomplete. Run outputs include resolved configuration, source hashes, runtime versions, per-example text, checkpoints, logs, status, and `report.json`/`report.md`.

The `score` command validates simple JSONL development submissions, for example:

```bash
uv run --frozen dirty-swapping score \
  --gold examples/dev_gold.jsonl \
  --predictions examples/dev_predictions.jsonl
```

The official runner uses `math-verify` for math and explicit final-choice extraction for multiple choice. A hosted competition should keep final labels server-side and run participant code on controlled hardware to measure latency; self-reported `latency_s` is not an official speed measurement. See the [evaluation protocol](docs/competition.md) for answer extraction, incomplete outputs, paired cohorts, timing boundaries, and reporting rules.

The **research theme** is reuse of otherwise discarded tokens or continuations. The current one-swap method is a starting baseline, not a limit on experiments. Participants may vary settings and develop new reuse strategies; compare each variant with plain inference under the same model and evaluation conditions. The current v1 code fixes top-2 selection, one swap per trace, and full-attention `DynamicCache`, so broader strategies require code changes.

For the current limited-compute experiment order and stopping criteria, see [the research plan](docs/RESEARCH_PLAN.ko.md). The first pilot uses `configs/pilot-short-delay.json` on the GSM8K development split.

## Layout

| Path | Purpose |
| --- | --- |
| `src/dirty_swapping/default.json` | Single default experiment configuration |
| `configs/smoke.json` | Small end-to-end check |
| `src/dirty_swapping/datasets.json` | Dataset provenance and SHA256s |
| `src/dirty_swapping/backend.py`, `engine.py`, `cache.py` | Model decoding and one KV edit |
| `src/dirty_swapping/data.py`, `runner.py` | Data preparation and resumable paired evaluation |
| `tests/`, `examples/` | CPU checks and synthetic submission example |
| `inputs/`, `intermediates/`, `outputs/`, `logs/`, `models/` | Ignored generated artifacts |
| `temp/` | Ignored recovery snapshots and scratch work |

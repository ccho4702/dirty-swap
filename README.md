# Dirty Swapping — Group 18 IDL baseline

처음 시작한다면 [한국어 시작 가이드](docs/START_HERE.ko.md)를 보세요. 에이전트가 새 참가자를 안내할 때 사용할 [레포 내부 온보딩 스킬](skills/onboard-dirty-swapping/SKILL.md)도 있습니다.

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

The smoke configuration has a 64-token output cap and is **not** a benchmark setting. To run the proposed default suite, omit `--datasets` and `--config` in `setup` and `run`; use a unique run name and no `--limit`. If the process stops, run `uv run --frozen dirty-swapping resume --run outputs/<run-name>`. Run `uv run --frozen dirty-swapping config > my-config.json` to inspect or edit settings; pass `--config my-config.json` to `setup` and `run`. Device selection is `--device cuda:0` or `--device cpu`. `--model-cache` can reuse an existing model cache. The model revision and dataset file hashes are pinned.

This setup was verified on an RTX 3090: `uv sync --frozen`, tests, lint, wheel build, hash-checked preparation of all four default datasets, and paired GPU smoke examples completed. On one example from each default dataset, baseline exactly matched ordinary autoregressive decoding for all 256 generated tokens; GSM8K also exactly matched Hugging Face `model.generate`. Baseline time averaged 1.01× the ordinary decoding path across these four examples. With the proposal's 4-candidate, 32-token rollout, 128-token delay settings and a 512-token cap, swap added 3.70 s on average, matching the 3.72 s spent preparing alternatives; peak GPU allocation was under 8 GiB. A separate short GSM8K question was answered correctly by both arms. These small checks establish functioning paths, **not a full-dataset accuracy result**. See [verification details](docs/implementation.md). GPU kernels can introduce small nondeterminism even with a fixed seed.

## Data

The **default suite** is AIME 2025, HMMT February 2025, GSM8K test, and GPQA Main. GSM8K train is a disjoint development split; run it with `--datasets gsm8k --split development`. AIME/HMMT/GPQA are evaluation-only in this baseline. Dataset source URLs, revisions, and SHA256 hashes are in `src/dirty_swapping/datasets.json`. Weights and source data are not redistributed. Recommended additional tracks are MATH-500, SuperGPQA, and LongBench v2; see [dataset notes](docs/datasets.md). Only SuperGPQA and LongBench v2 currently have adapters; MATH-500 is a recommendation, not an implemented CLI dataset.

## Metrics and competition use

The primary metric is **macro final-answer accuracy** over all four default datasets, with correct/total and accuracy for each dataset. Subset runs report a selected-dataset macro score but no primary score. Paired baseline-vs-swap accuracy difference (percentage points), wins/losses, mean/median/p95 per-example latency, alternative-rollout time, cache-copy time, peak GPU memory, and swap rate are reported. Latency includes tokenization, prefill, all rollouts, the swap, and final generation. Reports use only IDs completed by both arms; partial reports are labeled incomplete. Run outputs include resolved configuration, source hashes, runtime versions, per-example text, checkpoints, logs, status, and `report.json`/`report.md`.

The `score` command validates simple JSONL development submissions, for example:

```bash
uv run --frozen dirty-swapping score \
  --gold examples/dev_gold.jsonl \
  --predictions examples/dev_predictions.jsonl
```

The official runner uses `math-verify` for math and explicit final-choice extraction for multiple choice. A hosted competition should keep final labels server-side and run participant code on controlled hardware to measure latency; self-reported `latency_s` is not an official speed measurement. See [competition protocol](docs/competition.md).

The **research theme** is reuse of otherwise discarded tokens or continuations. The current one-swap method is a starting baseline, not a limit on experiments. Participants may vary settings and develop new reuse strategies; compare each variant with plain inference under the same model and evaluation conditions. The current v1 code fixes top-2 selection, one swap per trace, and full-attention `DynamicCache`, so broader strategies require code changes.

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

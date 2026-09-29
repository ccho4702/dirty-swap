# Implementation and verification status

## Reused upstream foundation

`core.py`, `data.py`, `math_scoring.py`, and `datasets.json` were adapted from [lost-tokens](https://github.com/ccho4702/lost-tokens) commit `ca80edc24aa4f4b86e9b162581a4d3be0bf26cf9`. They provide atomic JSON writes, source fingerprints, SHA256-checked downloads, deterministic QA rows and GPQA choice order, and pinned `math-verify` scoring. The original is locally available at `../temp/upstream/lost-tokens` in this workspace, but that scratch copy is not part of the repository.

## Dirty Swapping implementation

- `backend.py` runs a frozen causal LM with an explicit `DynamicCache`, independent cache forks, top-k candidate tokens, and greedy continuation.
- `engine.py` commits the top-1 rollout, computes the other configured rollouts, waits the configured number of tokens, and applies one top-2 KV replacement if reasoning is still open.
- `cache.py` checks every layer before writing and copies only the equal-length selected segment in inference mode.
- `runner.py` evaluates paired examples, writes each completed case atomically, resumes at the case boundary, records provenance and latency, and produces reports.
- `cli.py` exposes `config`, `download`, `prepare`, `setup`, `run`, `resume`, `status`, `report`, and a separate toy `score` command.

## Verification completed here

On September 30, 2026 (Asia/Seoul), `uv sync --frozen` created a project-local environment. `uv run --frozen python -m unittest discover -s tests -v` passed the CPU suite, including a two-layer real Qwen3 `DynamicCache` test, plain-vs-baseline parity regression test, and official `math-verify` checks. `uv run --frozen ruff check src tests`, `uv build`, and a CLI invocation from another working directory passed.

The pinned Qwen3-4B-Thinking-2507 model downloaded and loaded on an RTX 3090 (24 GB, driver 565.57.01). The four default datasets passed their SHA256 checks and preparation: AIME25 30 rows, HMMT25 30, GSM8K 8,792 (train plus test), GPQA Main 448. One paired 64-token GPU smoke example per dataset completed, and each swap arm recorded one replacement. Peak CUDA allocation in these smoke cases was below 8 GiB. The completed-run `resume` command returned without regenerating cases.

A separate synthetic text QA check asked `What is 2 + 2?` with a 1,024-token cap. The baseline returned `\boxed{4}` after 865 tokens; the swap arm returned `\boxed{4}` after 709 tokens and recorded a swap. Both passed `math-verify`. A 512-token GSM8K check and the 64-token suite ended inside `<think>`, so their zero accuracy is a decoding-cap artifact, **not a benchmark result**. No full-dataset score, accuracy improvement claim, or multi-GPU scaling claim is made. Repeat the smoke test on each competition host before running a full evaluation.

The proposal's default intervention schedule was then checked on the same synthetic `2 + 2` question with a 1,024-token cap: four candidates, 32-token rollouts, a 128-token delay, and a branch at reasoning token 256. Baseline and swap both answered `4` correctly; the swap arm recorded one replacement. The baseline generated 865 tokens in 32.9 s and the swap arm 948 tokens in 39.8 s, with peak allocation below 8 GiB. These single-example timings are smoke evidence only.

## Plain inference and performance checks

The pinned 4B model was also run on one evaluation example from each default dataset with a 256-token cap. Ordinary greedy decoding and this repository's baseline generated **identical token IDs at every position** in all four examples. The mean per-example time was 10.30 s for ordinary decoding and 10.42 s for baseline (1.01×); peak GPU allocation matched for each pair. On GSM8K, Hugging Face `model.generate(do_sample=False)` also matched all 256 token IDs. A separate 128-token check with an explicit all-ones attention mask matched as well. These measurements indicate no large baseline runtime regression on the sampled workload; four examples are too few for a stable speed estimate.

The four tasks were then run as paired baseline/swap examples with the proposal's default intervention schedule and a 512-token cap. All four swaps occurred without errors or OOM. Mean per-example time was 19.95 s baseline and 23.65 s swap. The 3.70 s difference closely matched the 3.72 s mean alternative-rollout time; the KV copy averaged roughly 0.002 s. Maximum observed swap allocation was 8,177 MiB on the 24,576 MiB RTX 3090. All eight 512-token outputs remained in reasoning, so this check does not measure answer accuracy.

For an answer-completing benchmark check, a short public GSM8K question was run with a 1,536-token cap and the same default intervention schedule. Baseline answered `13` correctly after 1,012 tokens in 38.82 s; swap answered `13` correctly after 993 tokens in 41.25 s and recorded one replacement. This is one selected question, not evidence of unchanged dataset-level accuracy.

## Run recovery boundary

Completed examples are atomic checkpoints. If execution stops inside an example, `resume` reruns that example from its deterministic prompt; it does not serialize a live GPU KV cache. This avoids replaying already completed examples and preserves results and logs. The baseline uses deterministic greedy decoding, so per-example replay needs no sampler RNG state. Resume uses the frozen resolved configuration stored in `plan.json`; editing an external config file afterward does not alter that run. Changed source, prepared data hashes, dependency lock, or runtime versions reject resume. Start a new run for a changed experiment.

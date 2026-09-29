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

## Expanded seven-dataset audit

There is **no training stage** in this reference method: no loss, optimizer, or weight update is implemented. The backend calls `eval()`, sets every parameter's `requires_grad=False`, and uses inference mode for model forwards. A small real Qwen3 test confirmed that parameter tensors were unchanged and gradients remained absent after decoding and KV replacement. This verifies frozen inference, not training convergence.

All seven supported source datasets were downloaded at their pinned revisions and hashes, then prepared under the 4,608-token input cap. The audit found the expected 30 AIME25, 30 HMMT25, 8,792 GSM8K (7,473 development / 1,319 evaluation), 448 GPQA, 500 MATH-500, 26,529 SuperGPQA, and 503 LongBench v2 rows. Every row had a unique ID, valid answer format and in-range prompt; math gold parsing and choice-letter checks found **zero issues**. One SuperGPQA prompt measured 4,108 tokens, so the former 4,096-token cap failed preparation; raising the cap to 4,608 allowed all its questions/options without truncation. LongBench v2 retained head/tail context and marked **503/503** rows truncated. See [datasets.md](datasets.md) and the local ignored `outputs/data-schema-audit-20260930.json` artifact.

One 64-token paired GPU smoke case per dataset completed, with one recorded swap in each swap arm. This proves seven input and cache-edit paths execute but yields no final answers at that cap. On one case per dataset with a 256-token cap, ordinary greedy decoding and this repository's baseline generated **exactly matching token IDs** throughout; peak memory and latency were comparable. A longer 512-token test of the three additional tracks also completed its default 4-candidate/32-token-rollout/128-token-delay swap on each case. LongBench's swap peak was about 11.0 GiB at a 4,577-token prompt. None of those 512-token outputs finished reasoning, so they are not accuracy evidence.

The first sampled AIME25 problem did not finish `<think>` at either 4,096 or 8,192 generated tokens in plain baseline mode. Both arms of the 4,096-token run were saved as incorrect due to the cap; the exploratory 8,192-token run was stopped after its baseline because continuing the capped comparison would not answer the accuracy question. The default benchmark output cap was therefore raised to **32,768 tokens**, matching the long-reasoning intent of the upstream setup. This longer cap is a budget, not a claim that every trace finishes or that a full run is cheap.

A short public MATH-500 arithmetic question exposed a real budget trade-off. With a 2,048-token cap, baseline answered `-50` correctly after 1,831 tokens, while swap reached the cap with no extracted answer and was incorrect. With a 4,096-token cap, baseline again answered `-50` after 1,831 tokens and swap answered `-50` after 2,113 tokens; their measured times were about 70.5 s and 85.1 s. The intervention can therefore delay an answer enough to **lower accuracy under a fixed output budget**, even when the answer is recovered under a longer budget. No dataset-level quality-preservation or improvement claim has been established.

A frozen full-cohort plan was prepared locally at `outputs/full-seven-prepared-20260930`: **29,359 evaluation questions and 58,718 paired arm-cases** across seven datasets, using the 32,768-token default output cap. It has `prepared` status; **the full inference has not been launched**. For scale, a single SuperGPQA pair at the insufficient 512-token cap took about 44 s. Multiplying that one-case timing by 26,529 SuperGPQA questions alone gives roughly 13.5 GPU-days, and a 32K answer-completing cap could cost far more. This is only a rough lower-bound planning signal, not a runtime prediction. The repository can resume a run at completed-example boundaries, but a full seven-dataset accuracy claim needs an explicit compute allocation and completed cohort.

## Run recovery boundary

Completed examples are atomic checkpoints. If execution stops inside an example, `resume` reruns that example from its deterministic prompt; it does not serialize a live GPU KV cache. This avoids replaying already completed examples and preserves results and logs. The baseline uses deterministic greedy decoding, so per-example replay needs no sampler RNG state. Resume uses the frozen resolved configuration stored in `plan.json`; editing an external config file afterward does not alter that run. Changed source, prepared data hashes, dependency lock, or runtime versions reject resume. Start a new run for a changed experiment.

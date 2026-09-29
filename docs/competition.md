# Evaluation protocol

This repository's reference experiment is training-free: Qwen3-4B-Thinking-2507 runs with frozen weights, and Dirty Swapping reuses KV from an unchosen continuation. The broader research theme permits different reuse methods and settings. If a variant changes the model, prompt, selection rule, or training procedure, disclose the change and compare it with **plain inference using that same resulting setup**. Do not attribute a better model's accuracy to KV reuse.

## Cohort and decoding

The default comparison uses AIME25, HMMT25, GSM8K **test**, and GPQA Main. MATH-500, SuperGPQA, and LongBench v2 are separate additional tracks. Use the same stable question IDs in both arms, with identical model weights, tokenizer, prompt, option order, seed schedule, device, precision, and input/output token caps. The provided baseline follows ordinary greedy autoregressive decoding without alternative rollout or cache edit; the swap arm commits the same top-1 main path before its intervention. The current method makes at most one past KV replacement while leaving already generated text and later KV entries unchanged.

`--limit` is a deterministic subset **per dataset** for development or smoke testing. It is not a full-dataset score. GSM8K train is development data; the other supported datasets are evaluation-only in this repository. Do not use evaluation gold labels to choose branch positions, alternatives, or stopping rules. See [the data catalog](datasets.md) for formats and truncation policy.

## Final-answer scoring

The model is prompted through the same Qwen chat template with thinking enabled. Prefer the answer after the first generated `</think>` token. If generation ends with EOS without that boundary, parse the full generated body. A capped, still-open thinking trace has **no final answer** and is scored incorrect; it is not removed from a completed evaluation cohort.

| Row format | Required final response | Extraction and correctness |
| --- | --- | --- |
| `math` (AIME25, HMMT25, GSM8K, MATH-500) | `\boxed{ANSWER}` | Extract the **last complete, nonempty box**. Compare with gold using pinned `math-verify` (`strict=True`, 6-digit float rounding, 15-digit numeric precision, 5 s verification timeout). Missing/malformed answer is incorrect. |
| `choice` (GPQA, SuperGPQA, LongBench v2) | `Final answer: (LETTER)` | Extract the **last explicitly marked** choice letter and require it to be among the row's options. Compare the letter exactly with gold. An option mentioned only in reasoning is not a prediction. |

The reference implementation of these rules is [`math_scoring.py`](../src/dirty_swapping/math_scoring.py) and [`core.py`](../src/dirty_swapping/core.py). Audit a sample of model outputs and equivalence judgments before presenting a new dataset's score. The separate `dirty-swapping score` JSONL command checks a simple public submission format; the paired model runner above is the benchmark path.

## The two main outcomes

1. **QA quality:** report correct/total and accuracy for every dataset, the paired `swap − baseline` difference in percentage points, and paired wins/losses. The four-dataset **macro accuracy** is the reference suite's aggregate only when all four evaluation splits complete with no `--limit`. Report micro accuracy separately; larger datasets must not silently dominate the macro score. Additional tracks get their own scores.
2. **Inference cost:** measure each example from prompt tokenization/prefill through final-token generation, **including all candidate rollouts and KV copy**. Model loading and answer scoring are outside this per-example timer and should be reported separately if comparing total compute budgets. Report mean, median, p95, summed per-example time, and the ratio or percentage increase relative to matching plain inference. Include alternative-rollout time, copy time, peak GPU memory, and swap/skip counts.

The provided `baseline` is token-identical to plain greedy inference in the sampled verification cases, but its branch bookkeeping can add a small timing cost. For a strict absolute time comparison, also measure ordinary `model.generate` under the same conditions. Show both quality and latency; this protocol does not invent a combined score or claim that a slower method is better merely because one small sample improved.

## Completeness, provenance, and reporting

Full-evaluation accuracy uses all selected question IDs as the denominator. A failed or timed-out case must be reported; do not quietly replace it with an easier case. The runner writes each completed arm/case atomically and resumes without rerunning finished cases. Its interim paired report uses IDs completed by both arms and marks the run incomplete. **Do not publish a time-censored or partial paired subset as the official full-dataset result**, because completion speed can bias the cohort.

Keep a unique run name, resolved configuration, source/model/data revisions and hashes, dependency lock, hardware and driver versions, seeds, timestamps, per-example outputs, logs, status, and reports. Do not overwrite an existing run. The reference outputs are under `outputs/<run-name>/`; `report.json` has per-task accuracy, timing, memory, and swap rate, while `report.md` gives a compact table. Record known GPU nondeterminism and repeat the same cohort when estimating an effect rather than relying on one seed or one question.

LongBench v2 is a **4,608-token capped-context** evaluation in the default configuration: all 503 contexts were shortened while their questions and options were preserved. Label that result separately from any full-context LongBench score. The 64-token smoke configuration often ends before a final answer and tests execution only. An AIME25 sample stayed inside `<think>` at both 4,096 and 8,192 tokens; use the default 32,768-token output budget, or label a smaller budget explicitly. A MATH-500 sample was baseline-correct but swap-incomplete at 2,048 tokens, then correct in both arms at 4,096: **a capped, missing answer is an actual error under that budget**, even if a longer run later recovers it. Report answer-extraction and thinking-completion counts beside accuracy.

For a hosted Kaggle-style challenge, keep final gold labels server-side and run submitted code on controlled hardware for the official time measurement. A participant's self-reported `latency_s` may help local debugging but must not determine an official speed ranking.

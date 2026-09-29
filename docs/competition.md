# Competition protocol

## Goal and evaluation

The shipped reference setup uses frozen Qwen3-4B-Thinking-2507 and a one-swap Dirty Swapping method. The broader research theme is reuse of otherwise discarded tokens or continuations. Participants may change settings or implement new reuse methods under that theme. Report every changed model, prompt, selection rule, cache policy, and inference budget so a gain is attributable to the method being studied.

**Final-answer accuracy and end-to-end inference time relative to plain inference are the two main outcomes.** For the reference four-dataset suite, report per-dataset correct/total and macro accuracy across AIME25, HMMT25, GSM8K, and GPQA Main. A submission missing a default dataset is incomplete for that suite. Report micro accuracy separately so larger datasets do not silently dominate. Show both quality and cost rather than collapsing them into an arbitrary single score.

Accuracy versus latency is the research question, so also publish paired accuracy change against baseline (percentage points), mean/median/p95 latency, summed per-example inference time, alternative-rollout time, swap rate, and counts of skipped interventions. Provide per-example outcomes and aggregate paired wins/losses. State hardware, precision, model and dataset revisions, seeds, input/output caps, and number of complete examples. Partial or time-censored runs must be labeled as such.

## Fair execution

- Use the same model weights, tokenizer, prompt, maximum input/output lengths, seed schedule, and main-path decoding rule for baseline and intervention.
- The reference baseline keeps model weights frozen. If a variant changes weights or uses training, disclose that change and compare against plain inference using the same resulting model. Never use final-evaluation answers to select a branch or alternative.
- Start timing before tokenization/prefill; stop after final answer. Include candidate rollouts, cache copies, and model load separately if load time is excluded from per-example latency.
- Use the same allocated GPU class and comparable load conditions. Run paired questions; record complete-case selection and timeouts.
- Emit one unique run ID, resolved config, code revision, dataset hashes, dependency lock hash, hardware info, timestamps, logs, predictions, and status. Never overwrite a run. Checkpoint atomically and resume without duplicating completed cases.
- Verify the intervention on a real model: selected KV changed, prefix/suffix KV and token IDs stayed byte-identical, one swap occurred at most, and the final decode consumed the edited cache.

## Local development format

Labeled development JSONL rows contain `id`, `task`, `question`, and `answer`. Prediction JSONL rows contain `id`, extracted `answer`, numeric `latency_s`, and Boolean `swap_applied`. IDs must match exactly, with no duplicates. See `examples/`. Run:

```bash
uv run dirty-swapping score --gold examples/dev_gold.jsonl --predictions examples/dev_predictions.jsonl
```

The included `score` command checks the format and simple answer types for development. The model runner uses explicit answer extraction and the pinned `math-verify` dependency for its official local results. Audit representative extracted answers before opening a leaderboard. For a hosted Kaggle-style challenge, publish prediction submission fields and a separate executable entry point; run submitted code in an organizer-controlled environment to measure latency. Self-reported `latency_s` must not determine an official speed ranking. Keep final gold labels out of the public repository and score server-side.

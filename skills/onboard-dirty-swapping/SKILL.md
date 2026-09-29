---
name: onboard-dirty-swapping
description: Orient a new contributor to this Dirty Swapping repository, its setup, dataset choices, paired QA-and-latency evaluation, and configurable discarded-token reuse experiments. Use for repo onboarding or experiment planning here.
---

# Onboard Dirty Swapping contributors

Use [the Korean start guide](../../docs/START_HERE.ko.md) for the participant-facing explanation. Check the current [README](../../README.md), [default config](../../src/dirty_swapping/default.json), [dataset manifest](../../src/dirty_swapping/datasets.json), and [competition rules](../../docs/competition.md) when commands, dataset support, or scoring details matter. Keep one source of truth for details that can change.

Explain the research umbrella as reuse of information from unchosen tokens or continuations. Describe the shipped v1 method precisely: one past, equal-length KV replacement; generated text and later KV remain as they were. State that the baseline has the ordinary main-path tokens and that alternative preparation belongs in swap latency.

Guide setup through project-local `uv sync --frozen`, tests, dataset/model `setup`, a short paired `run`, `report`, and `resume`. The 64-token smoke config checks execution and may end before a final answer. Do not describe smoke accuracy as a benchmark result.

Distinguish dataset status: AIME25, HMMT25, GSM8K, and GPQA Main are the default suite; MATH-500, SuperGPQA, and LongBench v2 are supported additional tracks. Explain development versus evaluation splits, LongBench's capped context, and the normalized row/answer formats. Keep reference answers and worked solutions out of prompts and intervention selection.

Present **final-answer accuracy** and **end-to-end inference time versus matching plain inference** as the two central outcomes. Pair the same examples and decoding conditions; show per-dataset accuracy, paired change, latency distribution, alternative cost, swap/skip rate, and memory. If a participant changes the model or other setup, require a corresponding plain baseline under that same setup before attributing gains to discarded-token reuse.

Welcome experiments under the broad reuse idea without implying that every variant is implemented. v1 config can change candidate count, rollout length, branch offset, delay, token caps, and supported model path. The present implementation fixes top-2 alternative selection, at most one swap, and full-attention `DynamicCache`; new selectors, multiple edits, and other cache layouts require code and verification changes.

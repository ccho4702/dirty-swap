---
name: onboard-dirty-swapping
description: Orient a new contributor to this Dirty Swapping repository, its setup, dataset choices, paired QA-and-latency evaluation, and configurable discarded-token reuse experiments. Use for repo onboarding or experiment planning here.
---

# Onboard Dirty Swapping contributors

For someone opening the repository for the first time, point to `./start.sh`: it prints the document map without installing anything. Use [the Korean start guide](../../docs/START_HERE.ko.md) for the participant-facing explanation. Check the current [README](../../README.md), [default config](../../src/dirty_swapping/default.json), [dataset manifest](../../src/dirty_swapping/datasets.json), and [competition rules](../../docs/competition.md) when commands, dataset support, or scoring details matter. Keep one source of truth for details that can change.

Explain the research umbrella as reuse of information from unchosen tokens or continuations. Describe the shipped v1 method precisely: one past, equal-length KV replacement; generated text and later KV remain as they were. State that the baseline has the ordinary main-path tokens and that alternative preparation belongs in swap latency.

Guide setup through project-local `uv sync --frozen`, tests, dataset/model `setup`, a short paired `run`, `report`, and `resume`. The 64-token smoke config checks execution and may end before a final answer. Do not describe smoke accuracy as a benchmark result.

Distinguish dataset status: AIME25, HMMT25, GSM8K, and GPQA Main are the default suite; MATH-500, SuperGPQA, and LongBench v2 are supported additional tracks. Explain development versus evaluation splits, LongBench's capped context, and the normalized row/answer formats. Keep reference answers and worked solutions out of prompts and intervention selection.

Present **final-answer accuracy** and **end-to-end inference time versus matching plain inference** as the two central outcomes. Pair the same examples and decoding conditions; show per-dataset accuracy, paired change, latency distribution, alternative cost, swap/skip rate, and memory. If a participant changes the model or other setup, require a corresponding plain baseline under that same setup before attributing gains to discarded-token reuse.

Use [the general methodology](../../docs/GENERAL_METHOD.ko.md) as the current research direction. Apply one candidate-generation/cache-edit/future-selection policy and one grounding, factual-consistency, reasoning-validity, and progress criterion across datasets. Dataset names and math/choice formats must not select different intervention rules. Only input/output adapters and final scoring use answer format. The `probe_preference` policy now runs across QA types; `draft_swap` does not require boxed math answers. Recommend `general-probe.json` and its common smoke configuration. Native alternatives from the same frozen main model are the primary path. Mathematical guide prompts and numeric-trigger configurations are historical exploratory controls. Do not describe them as a general method or extend them with problem-specific rules. A learned general QA value function is planned, not implemented; do not imply a calibrated correctness probability or verified cross-domain gain. Research pushes target `changho`; promote to shared `main` only on an explicit release instruction.

Read [the completed results](../../docs/RESULTS.ko.md) before reporting quality. Development gains did not improve aggregate accuracy on either separate 8-question cohort, and individual regressions remain. The 36-layer GPU cache check and native-generate comparison passed. Report the explicit 2,048-token budget and the separate 4,096-token recovery check. The retained [public summary](../../outputs/research-summary-20261004/summary.json) includes posthoc draft-only controls without reference answers.

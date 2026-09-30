# Dirty Swapping repository

- Follow `docs/method.md` for the one-swap invariant: replace an equal-length past KV span across every attention layer and preserve all downstream text and KV exactly.
- Keep model, dataset, and runtime defaults in `src/dirty_swapping/default.json`; source checksums are in `src/dirty_swapping/datasets.json`.
- Use a project-local `uv` environment. Generated data goes into ignored `inputs/`, `intermediates/`, `outputs/`, `logs/`, and `temp/` as appropriate.
- Save a pre-edit snapshot under `temp/snapshots/` before changing source. Run the unit tests after changes; run a real-model GPU smoke test for GPU integration changes when an explicitly available GPU is allocated.
- Do not publish final gold labels or report toy scorer numbers as benchmark results.
- For onboarding or explaining experiment choices, start with the read-only `./start.sh` map, then use `skills/onboard-dirty-swapping/SKILL.md` and its linked `docs/START_HERE.ko.md` guide.

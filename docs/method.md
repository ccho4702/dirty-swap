# Method contract

## One intervention

1. Prefill the question and begin ordinary deterministic decoding with frozen model weights. Only consider a branch position within the reasoning span delimited by `<think>` and `</think>`.
2. At the configured fixed reasoning-token offset (default 256), inspect next-token probabilities under one shared prefix. Select the top `k` distinct token IDs. The top-1 token defines the main path; the top-2 token is the fixed initial alternative. Do not inspect the gold answer.
3. Fork the prefix cache independently for each candidate. Generate a 32-token rollout from each forced first token under the same deterministic decoding rule and position IDs. The main path commits the top-1 rollout exactly.
4. Generate 128 more tokens on the main path. If the rollout or delay ends the reasoning span, skip replacement and record why.
5. At absolute positions `[branch_position, branch_position + rollout_tokens)`, copy the alternative keys and values into every full-attention layer of the main cache. Preserve all token IDs, prefix KV before the branch, and downstream KV after the segment. The alternative cache may be shorter than the main cache; its selected segment must be the same length.
6. Continue generation from the modified main cache to a final answer. No suffix recomputation or second swap occurs.

The next-token logits at the swap point were computed before the edit. Therefore the first token sampled after the swap still follows those stale logits. Processing that token uses the edited cache, so subsequent logits can change. This one-token delay preserves every already computed suffix KV entry.

For a conceptual `A B C D`, replacing the KV corresponding to `B C` with those of alternative `E F` yields text `A B C D` but a cache containing `A E F D`. The `D` cache still reflects the original `B C`. This deliberate mismatch is the intervention, and coherence failures are an expected outcome to measure.

## Baseline and invariants

The baseline uses the same model, prompt, top-1 main-path rollout, decoding limit, seed, and final-answer parser. It does not compute unused alternatives or edit KV. Compare the same question IDs in both arms. An alternative rollout's first token is forced; subsequent tokens follow the same deterministic policy. If the model stops early, record a skipped intervention rather than padding a shorter candidate into a cache segment.

`swap_kv_segment` validates every layer before writing. It assumes the Transformers `DynamicCache` full-attention tensor layout `[batch, kv_heads, seq_len, head_dim]`. Sliding-window and static caches are rejected; quantized or heterogeneous caches with incompatible tensors also fail validation. A real-model smoke test should assert suffix preservation because cache implementations can differ by model and library version.

## Planned ablations

Start with `k=4`, rollout 32, delay 128, one fixed branch. Then vary one factor at a time: candidate count, rollout length, delay, branch position, and alternative selector. A verifier-based selector must be trained or tuned on development data only. Compare swap versus no-swap while retaining alternative rollout computation to isolate the cache edit effect from its compute overhead; also report the standard no-rollout baseline.

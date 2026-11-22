# DSpark speculative decoding: design notes

Plan for running Inkling 2 Mini's bundled drafter on top of the GGUF model. Not implemented yet.

Sources:
- `thinkingmachines/Inkling-2-Mini`: `config.json` → `mtp_config`, and `mtp.safetensors` (636 MB, BF16), already downloaded.
- vmlx-swift `310c583` (bench/engines/vmlx-swift): `Libraries/MLXLLM/Models/Inkling2DSpark.swift` (drafter), `Inkling2DSparkSession.swift` (round bookkeeping), `Inkling2Verify.swift`, `Inkling2Loop.swift` (`Inkling2Capture`, `Inkling2ConvRecorder`). Their header cites SGLang PR #51926 (`models/inkling2_mtp.py`, `dflash.py`, `dspark.py`) as the reference.
- Ollama 0.40.17 runs the same drafter (`Inkling2ForConditionalGenerationMTP`, tensors `draft.*`).

## Why

Measured with Ollama's official tag (bench/results/SUMMARY.md): drafting takes its decode from 21 to 29.4 tok/s at ~512 context and 36.8 at ~4.5K. 50–65% of drafted tokens accepted, 1.6–3.9 committed per verify. Our base decode is ~2x Ollama's, so the same multiplier lands on a faster base.

## The drafter (mtp_config)

- 3 layers, d = 2304, 24 q / 8 kv heads × 128, dense SwiGLU MLP (9216), RMSNorm eps 1e-6.
- RoPE, neox half-split, θ = 1e6, at absolute positions. Layers 0 and 2 see a 1024-key window of context; layer 1 sees all of it.
- q_norm / k_norm per head; separate `k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx` for the context keys.
- `block_size` 8: one round drafts 7 tokens. `markov_rank` 256, `markov_head_type` rnn_hidden.
- `target_layer_ids` [5, 23, 41, 48]: target features it reads, in layer-visit numbering (visit = layer for the prelude; `layer + pass*12` in the loop; `layer + 24` for the coda; 48 = the final normed hidden).
  - In our slot numbering: visit 5 = input of layer 5; visit 23 = input of layer 11 on pass 2 (slot 29); visit 41 = input of layer 17 on pass 3 (slot 47); 48 = the final normed hidden that feeds lm_head.
  - "Input" = the residual stream entering the layer (our `a.h` at layer start), captured before the layer runs.
- `fuse_weight` [4, 4]: per drafter layer (rows 0-2) and for the Markov head (row 3), softmax weights over the 4 features.

## One round

Committed length P; the anchor is the token at position P (sampled, not yet in the target's KV).

1. **Draft**: block input `[embed_norm(embed(anchor)), mask_embed × 7]` at positions P..P+7, through the 3 layers.
   - Attention is bidirectional within the block. Keys = context KV for committed positions (windowed on layers 0 and 2) plus the 8 block keys. No mask.
   - Context KV per drafter layer: `c = rmsnorm(Σ_t softmax(fuse_weight[layer])_t · feature_t) · dflare_norm`, then `k = rope(k_norm_ctx(k_proj_ctx(c)))`, `v = v_proj_ctx(c)`. Computed once per committed position and appended, like a KV cache.
   - Block layers: pre-norm attention and MLP residuals, then the final `norm`. Keep rows 1..7: h [7, d].
2. **Markov walk** (7 sequential steps):
   - `base = lm_head(h)` [7, V], using the target's own output head.
   - `f_rnn = rmsnorm(Σ softmax(fuse_weight[3]) · features of the last committed position) · markov_head.dflare_norm`.
   - State s (rank 256) starts at 0; prev = anchor. Per step: `p = rnn_proj([s, markov_w1[prev], h_i, f_rnn])` (5120 → 768, with bias); `g = σ(p[0:256])`; `s = g·s + (1-g)·tanh(p[256:512])`; `latent = tanh(p[512:768])`.
   - Logits `v = softcap30(base_i + W_fold · latent + b_fold)`, with `W_fold = W_U · hidden_out.weight` [V, 256] and `b_fold = W_U · hidden_out.bias` folded at load. Then `d_i = argmax(v[:200058])`.
   - The reference rounds to BF16 at specific points (see `walk`). Acceptance is the check that matters here, not bit-exactness.
3. **Verify**: the target runs `[anchor, d1..d7]` causally at positions P..P+7 (our batched path with nt = 8, logits at every row). Accept the n leading positions where the target's argmax equals the next draft, and take the target's token after the last accepted one as the bonus (next anchor). Greedy output is exactly the target's.
4. **Commit** `keep = n + 1` rows:
   - Target KV: rows past P+keep are just ignored (pos = P+keep), so nothing to do.
   - Target conv states (4 per slot × 48 slots) must rewind: `state = last 3 of [old_state, inputs[0..keep]]`. Record each conv's pre-verify state and its 8 input rows during verify.
   - Draft context KV: append rows for the kept positions only, computed from those rows' captured features.
   - f_rnn source: the features of row keep-1.

## Engine work (in order)

1. Load `mtp.safetensors` with `gguf::safetensors` next to the GGUF. Convert the big matrices to Q8_0 at load: about 300 MB instead of 636. Keep norms, `fuse_weight` and biases in f32. Build W_fold/b_fold from our Q8_0 lm_head (dequantized on the CPU once), stored as Q8_0 [V, 256]: about 55 MB, read 7× per round.
2. Feature capture in `encode`: copy `a.h` at slots 5 / 29 / 47 plus the final normed hidden, for each row, into a [nt, 4, d] buffer.
3. Verify path:
   - nt = 8 batch with all-row logits (exists: `prefill_each`).
   - Conv-input recording and a rewind kernel.
   - Pos rollback.
   - Make small batches cheap. 8 tokens × top-8 can touch up to 64 experts per layer, so the MoE path at nt = 8 decides most of the verify cost. The current batched MoE uses 32-token GEMM tiles; a matvec-style multi-token expert kernel (grouped by expert) is likely better here.
4. Drafter forward in Metal:
   - RoPE.
   - Bidirectional block attention over context + block. Reuse the split-K decode attention with 8 queries.
   - Dense projections at nt = 8 through the Q8_0 GEMM.
   - The 7-step walk. One dispatch chain per step: rnn_proj matvec, gate math, W_fold matvec, add base, softcap, argmax. It stays on the GPU end to end.
5. Round loop on the GPU, without CPU round trips. Acceptance (compare argmaxes) and the bonus token can be a tiny kernel. The CPU only reads back the accepted count to advance pos. Pipelining across rounds is harder, because the next round's positions depend on n.

## Risks and checks before building all of it

- **Acceptance with GGUF features.** The drafter was trained on the original model's hidden states and tokens. Our IQ4_NL features differ by a few % at deep layers, and our greedy tokens differ where the two models disagree. Estimate: per-token acceptance about 0.75 → 0.70, so about 15% less speedup than with the original weights.
  - Measure before engine work: dump our features and greedy tokens for a few prompts, run the drafter in MLX (vmlx-swift or Python), count accepted drafts.
  - If acceptance is poor: distill the drafter (3 layers) on hidden states and tokens from our engine.
- **Verify cost.** Ollama's verify steps cost ~2x a decode step. Ours must stay below about (1 + accepted tokens per round)× a decode step to win.
- **Draft cost per round.** About 1 lm_head pass over 7 rows (490 MB), 7 W_fold reads (Q8_0, about 385 MB) and 3 small layers (about 300 MB): roughly 1.2 GB of reads, about 30% of a target step. W_fold could be shrunk further (it only needs to rank candidates near the top of `base`).

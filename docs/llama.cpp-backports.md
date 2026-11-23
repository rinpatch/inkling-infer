# Possible llama.cpp improvements for Inkling 2

Findings from building inkling-infer, for later. Nothing here has been tried in llama.cpp yet, and nothing goes upstream without review.
Checked against llama.cpp at the commit used for the benchmarks in bench/results/SUMMARY.md.

## 1. Flash attention never runs for Inkling 2 (biggest)

`src/llama-graph.cpp:3202`:

```cpp
const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
```

`src/models/inkling2.cpp` always passes `kq_b`, the hidden-state-conditioned relative-position bias (with the per-query sink folded in, lines ~356-372). So `-fa 1` has no effect: every attention layer materializes KQ in f32 (n_kv x n_tokens x n_head), adds a gathered bias tensor of the same size, runs a separate softmax, then KQV. Decode at depth also misses the split-K flash-decoding kernel.

Where it shows: pp4096 222 tok/s (llama.cpp) vs 315 (ours, query-tiled attention with the bias fused); decode at 4K context drops more than ours.

### 1a. Small patch: fold the bias into a per-head mask

ggml's flash attention accepts a per-head mask: `ggml_flash_attn_ext_impl` only requires `q->ne[2] % mask->ne[2] == 0` (ggml/src/ggml.c ~5444), so `mask->ne[2] = n_head` is legal. Build `mask_with_bias = kq_mask + kq_b` (F16, contiguous, padded like the mask) and call `build_attn` with `kq_b = nullptr`.

- Still materializes an n_kv x n_tokens x n_head tensor, but drops the f32 KQ, the add, the separate softmax pass and the unfused KQV.
- Precision: the bias is rounded to f16. Check with a multi-position KL against the current output (method: crates/inkling-metal/tests/quality.rs).
- Measure pp512 / pp4096 / tg128 at depth 0 and 4096, with `-fa 0` vs `-fa 1`.

### 1b. Larger change: relative bias inside the flash-attention kernel

What inkling-infer does: per (token, head), project r onto the bias table once (`rel_table`, 1024 floats for extent 1024), then the attention kernel adds `table[pos_q - pos_k]` (0 past the extent) to each score (`attn_tiled_*` and `attn_split_*` in crates/inkling-metal/kernels/layer.metal). No n_kv-sized bias tensor exists. Upstream this would be a flash-attention variant taking a (table, extent) pair, in every backend.

## 2. Tokenizer: `ignore_merges` missing

See docs/llama.cpp-tokenizer-bug.md. A correctness fix, not speed.

## Looked at, probably not worth upstreaming

- Concurrent dispatch of independent kernels: ggml-metal already tracks memory ranges and runs independent nodes concurrently.
- Model-specific fusions that help us (K/V conv + clip/norm written straight into the cache, fused message gates, one depth-mix kernel, fused SwiGLU-softcap in the decode MoE): real wins in a dedicated engine, but one-model fused ops are a hard sell for ggml.
- GPU argmax with back-to-back decode steps: llama-bench `tg` does not sample, so it would not change llama.cpp's numbers.
- The Inkling "DSpark" drafter: llama.cpp's `src/models/dspark.cpp` is DeepSeek-V4's DSpark, unrelated. Inkling's drafter ships only as `mtp.safetensors` in thinkingmachines/Inkling-2-Mini (no GGUF exists); supporting it would be a new model plus speculative-decoding wiring.

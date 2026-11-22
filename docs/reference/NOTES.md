# Where transformers (T) and llama.cpp (L) differ in form

These are equivalent unless marked otherwise. Fixtures come from L, so our engine follows L's tensor layouts and T's semantics.

- **Sinks.** T appends one sink logit per (query, head) and softmaxes. L subtracts the sink from the bias table and uses a fixed zero sink: `exp(s−t)/(1+Σexp(s−t))`. The math is identical, but the fp rounding differs slightly.
- **Relative bias.** Both project `r` onto `proj`/`attn_rel_bias` ([16 × extent]) and then gather by distance. L pads the table with one zero column, so distances ≥ extent hit 0. T clamps the index and then masks.
- **Conv states.** T keeps four states per layer (`conv_idx` 0..3 = k, v, post_attn, post_ffn). L packs them into one recurrent state at channel offsets 0, 1024, 2048, 2048+2304. We may use either layout.
- **Static conv tap order.** T: `static_weight[0]` multiplies the current token. L feeds `ssm_conv`, which expects the oldest token first, so the GGUF stores the static taps flipped (the last tap is the current token). Dynamic taps index by backward lag in both. Verified: `Kcur-5` (full layer, dynamic taps) matches the fixture at 0.8% rel L2, the same as the SWA layers.
- **Block list update.** T appends the layer input to `block_states` inside the layer (local copy) and again in the model loop. L pushes it once in `build_layer`. Same result.
- **Router.** T: `top_k(sigmoid(logits) + bias)`, weights from the unbiased scores, normalized with eps 1e-20, times 8. L: `expert_router_f32`, `expert_weights_norm_eps = 1e-20`, `expert_weights_scale = 8`. Same.
- **SITU = softcapped SwiGLU.** L comment: `beta·tanh(gate/beta)·sigmoid(gate) · lb·tanh(up/lb)` with beta 4 and lb 25. Same as T `SoftcappedSwiGLU`.
- **Precision.** T keeps the router, convs, attention-residual queries, loop gates and message queries in fp32. The original checkpoint is NVFP4 (experts) / MXFP8 (last 4 layers) / BF16 (rest). The GGUF is a requantization, so logits will not match the original checkpoint, only llama.cpp on the same GGUF.
- **MTP head.** `mtp.safetensors` and `mtp_config` hold a 3-layer speculative drafter. It isn't in the GGUF, so it's out of scope for v1.
- **Tokenizer: llama.cpp is wrong, HF is the reference.** llama.cpp's `inkling2` pre-tokenizer doesn't apply `tokenizer.json`'s `ignore_merges`, so it splits vocabulary words (`kwargs` → `kwar g s`). Details, repro and the one-line fix are in `docs/llama.cpp-tokenizer-bug.md`. Our tokenizer matches HF exactly. The forward-pass fixtures still use llama.cpp's token IDs (`tokens.txt`), which is fine because they test the network, not the tokenizer. Tokenizer tests use `tokens_hf.txt`.

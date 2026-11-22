# Where transformers (T) and llama.cpp (L) differ in form

These are equivalent unless marked otherwise. Fixtures come from L, so our engine follows L's tensor layouts and T's semantics.

- **Sinks.** T appends one sink logit per (query, head) and softmaxes. L subtracts the sink from the bias table and uses a fixed zero sink: `exp(s−t)/(1+Σexp(s−t))`. The math is identical, but the fp rounding differs slightly.
- **Relative bias.** Both project `r` onto `proj`/`attn_rel_bias` ([16 × extent]) and then gather by distance. L pads the table with one zero column, so distances ≥ extent hit 0. T clamps the index and then masks.
- **Conv states.** T keeps four states per layer (`conv_idx` 0..3 = k, v, post_attn, post_ffn). L packs them into one recurrent state at channel offsets 0, 1024, 2048, 2048+2304. We may use either layout.
- **Static conv tap order.** T: `static_weight[0]` multiplies the current token. L feeds `ssm_conv`, which expects the oldest token first. The converter must have flipped the static taps, and dynamic taps index by lag in both. **To verify** with the `Kcur` fixture on a full layer (layer 5).
- **Block list update.** T appends the layer input to `block_states` inside the layer (local copy) and again in the model loop. L pushes it once in `build_layer`. Same result.
- **Router.** T: `top_k(sigmoid(logits) + bias)`, weights from the unbiased scores, normalized with eps 1e-20, times 8. L: `expert_router_f32`, `expert_weights_norm_eps = 1e-20`, `expert_weights_scale = 8`. Same.
- **SITU = softcapped SwiGLU.** L comment: `beta·tanh(gate/beta)·sigmoid(gate) · lb·tanh(up/lb)` with beta 4 and lb 25. Same as T `SoftcappedSwiGLU`.
- **Precision.** T keeps the router, convs, attention-residual queries, loop gates and message queries in fp32. The original checkpoint is NVFP4 (experts) / MXFP8 (last 4 layers) / BF16 (rest). The GGUF is a requantization, so logits will not match the original checkpoint, only llama.cpp on the same GGUF.
- **MTP head.** `mtp.safetensors` and `mtp_config` hold a 3-layer speculative drafter. It isn't in the GGUF, so it's out of scope for v1.

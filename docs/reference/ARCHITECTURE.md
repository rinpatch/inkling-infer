# Inkling 2 Mini (`inkling2_loop`): what the engine must compute

Sources, in priority order:
- **T**: `transformers/inkling2_loop/modeling_inkling2_loop.py` (the official reference)
- **L**: `~/sw/llama.cpp/src/models/inkling2.cpp` + `inkling2-loop.cpp`, plus `build_moe_ffn` in `src/llama-graph.cpp`
- Background: `web/*.md` (blog, model card, Goodfire appendix with the loop equations), `attention-residuals-2603.15031.pdf`

Where **T** and **L** differ in form but agree in math, the difference is listed in `NOTES.md`.

## Dimensions (from `gguf-IQ4_NL-dump.txt` and `hf/config.json`)

| name | value | notes |
|---|---|---|
| `d` (n_embd) | 2304 | |
| layers (physical) | 24 | 4 attention-residual blocks of 6 |
| full-attention layers | 5, 11, 17, 23 | every other layer uses SWA (`local_layer_ids`) |
| heads / kv heads / head_dim | 24 / 8 / 128 | GQA ratio 3; the same for SWA and full layers |
| `d_rel` | 16 | relative-position states per head |
| `rel_extent` | 1024 for full layers, 1024 (= window) for SWA layers | bias is 0 beyond it |
| sliding window | 1024 | |
| short-conv kernel | 4 | state = 3 past tokens |
| experts / used | 128 / 8 | no shared expert |
| expert ffn | 1152 | `gate_up` is [2·1152 × 2304]; gate rows first, then up |
| vocab padded / real | 201024 / 200058 | slice logits to 200058 |
| loop | core = layers [6, 18), 3 passes, gate rank 64 | fixed (`varying = false`) |
| cache slots | 24 + 2·12 = 48 | one KV + conv state per layer application |
| eps | 1e-6 everywhere | unweighted norms also use 1e-6 |
| qk clip | 22.6274 (= √512) | rel clip 8.0 |
| log scale | `τ(p) = 1 + 0.1·ln(max(1, (p+1)/8192))` | full layers only, on q and on the rel states |
| softcaps | final logits 30; MLP gate 4; MLP up 25 | |
| route scale | 8.0 | `expert_weights_scale` |

Tensor types in the IQ4_NL file: **F32** (norms, conv taps, queries, router, loop gates, rel tables), **Q8_0** (all attention projections, `token_embd`, `output`), **IQ4_NL** (`ffn_gate_up_exps`, `ffn_down_exps`). There are no other types.

## Notation

- `rms(x) = x / sqrt(mean(x²) + 1e-6)` (unweighted). `RMSNorm_w(x) = w ⊙ rms(x)`.
- `clip_c(x) = x · min(1, c / (‖x‖₂ + 1e-6))`, applied per head vector.
- `softcap_c(x) = c·tanh(x/c)`.
- `mix(B, x, Q)`: depth attention. Stack `S = rms([B₀ … B_{n-1}, x])` (n+1 vectors). For each query row `q ∈ Q`: `out_q = Σ_j softmax_j(q·S_j / √d) · S_j`. **The values are the normalized states**, not the raw ones. (T `BlockAttentionResidual`, L `mix`.)

## Forward pass for one token stream

```
h   = RMSNorm_{token_embd_norm}(embed[tok])          # T NormedEmbedding
B   = []                                             # block list ("block_states")
for il in 0..6:            h = layer(il, slot=il, h, B)                 # prelude
prelude_h, prelude_B = h, B
for il in 6..18:           h = layer(il, slot=il, h, B)                 # pass 1
for pass in 2..=3:
    bank = B[len(prelude_B)+1 :]  (= [output of block 2])  ;  bank_out = h   # output of block 3
    h, B = prelude_h, prelude_B
    for il in 6..18:
        msgs = mix(bank, bank_out, loop_msg_query[il])      # 4 messages: q, k, v, mlp
        h = layer(il, slot = 24 + (pass-2)·12 + (il-6), h, B, msgs)
for il in 18..24:          h = layer(il, slot=il, h, B)                 # coda
h = RMSNorm_{output_norm}( mix(B, h, output_res_query) )
logits = softcap_30( output · h )[:200058]
```

The block list `B` is updated **inside** the layer, at block boundaries (`il % 6 == 0`): the layer **input** is appended to `B`, and the residual stream restarts from the attention output. In T the model loop appends `hidden_states` (the layer input) after the layer runs. The effect is the same.

## One layer: `layer(il, slot, h, B, msgs=None)`

```
x_in = h
# attention inputs
if il == 0:            a_q = a_k = a_v = h
elif SWA:              a_q = a_k = a_v = mix(B, h, attn_res_query[il])          # 1 query
else (full):           a_q, a_k, a_v = mix(B, h, attn_res_query[il])            # 3 queries
if msgs:               a_s = gate_s(a_s, msgs[s]) for s in q,k,v    # SWA loop layers: gates apply to the shared input, then q/k/v diverge
norms: SWA  -> n_s = RMSNorm_{attn_norm}(a_s)       (one weight shared by q,k,v)
       full -> n_q = RMSNorm_{attn_q_inp_norm}(a_q), n_k = RMSNorm_{k_inp}(a_k), n_v = RMSNorm_{v_inp}(a_v)

# projections (Q8_0 matvecs)
q = clip_c(attn_q_clip ⊙ (Wq n_q))            per head, c = 22.63  ; full: q *= τ(pos)
k = clip_c(conv_k(Wk n_k))                     per kv head
v = rms(conv_v(Wv n_v))                        per kv head, no weight
r = clip_8(attn_rel_clip ⊙ (Wrel n_q))         [24 heads × 16] ; full: r *= τ(pos)
sink = Wsink n_q                               [24]

# attention per head h (kv head h // 3), over cached keys j with distance δ = pos_q − pos_j ≥ 0
logit_j = (q·k_j)/√128 + bias_j ,  bias_j = (δ < extent) ? Σ_i r_i · attn_rel_bias[i, δ] : 0
SWA layers additionally mask δ ≥ 1024
p = softmax over {logit_j} ∪ {sink_h}  ; the sink has no value, it only absorbs mass
o_h = Σ_j p_j v_j
o = σ(Wgate n_q) ⊙ concat(o_h)                  # gate input is n_q (the query-side normed input)
attn = conv_post_attn(Wo o)

# residual bookkeeping
if il % 6 == 0:   B.append(x_in);  res = attn          # new block starts from the attention output
else:             res = x_in + attn

# MoE
m = mix(B, res, ffn_res_query[il])
if msgs: m = gate_mlp(m, msgs[mlp])
n = RMSNorm_{ffn_norm}(m)
s = σ(W_router n)                        (F32 router, 128 scores)
top = top8(s + exp_probs_b)              # the bias only affects selection
w_e = 8 · s_e / (Σ_top s + 1e-20)
y = Σ_{e∈top} w_e · Down_e( softcap_4(g)·σ(g) · softcap_25(u) ),  [g;u] = GateUp_e n
return res + conv_post_ffn(y)
```

### Short convolutions (state per slot, 4 streams: k, v, post_attn, post_ffn)
- Static: `y_t = (x_t + Σ_{l=0..3} w[l]·x_{t−l}) / √5` per channel. **Tap index = backward lag in T** (`static_weight[0]` multiplies the current token). The GGUF stores ggml `ssm_conv` order (`[n_conv, ch]`, the oldest token first). Check this order once against a fixture.
- Dynamic (k and v of the full layers): the taps are `static[l, c] + dyn[t, head(c), l]` with `dyn = W_conv_proj · n_s` (`[8 heads × 4]`), and the scale is `1/√9`. The kv head for channel c is `c // 128`.
- The state holds the last 3 **pre-conv** inputs (the projection outputs for k and v, `Wo·o` for post_attn, `y` for post_ffn).

### Message gate (loop passes 2–3, four per loop layer: q, k, v, mlp)
```
z = W_down [rms(x); rms(m)]       (2304·2 → 64, F32)
gate(x, m) = (1 − tanh(W_forget z)) ⊙ x + tanh(W_gain z) ⊙ m
```
In the GGUF: `loop_{q,k,v,ffn}_ctrl_{down,forget,gain}`, `loop_msg_query` [4 × 2304].

## KV cache
- 48 slots. Each SWA slot keeps a ring of the 1024 most recent K/V (8 heads × 128 each). Each full slot keeps everything.
- Positions are absolute. The relative bias needs `δ`, so the cache must record each entry's position, or store entries densely by position.
- The bias, the log scale τ and the masks are all derived from positions. There is no RoPE.

## Decode roofline (bytes read per token, IQ4_NL file)

| part | per layer application | × | total |
|---|---|---|---|
| attention projections Q8_0 (q, gate, o: 3×7.08M; k, v: 2×2.36M; rel 0.88M; sink, conv_proj) ≈ 27.0M elems × 1.0625 B | 28.7 MB | 48 | 1.38 GB |
| router F32 2304×128 | 1.2 MB | 48 | 0.06 GB |
| experts IQ4_NL, 8 × (5.31M + 2.65M) elems × 0.5625 B | 35.8 MB | 48 | 1.72 GB |
| loop gates F32 (4 × (295k + 2×147k)) | 9.4 MB | 24 | 0.23 GB |
| lm_head Q8_0 2304×201024 | | 1 | 0.49 GB |
| **total** | | | **≈ 3.9 GB / token** |

At 273 GB/s this caps decode at **≈ 70 tok/s** at short context. KV reads grow on top: (24 SWA slots, bounded) + (8 full slots, growing). Shared weights are re-read on every pass because they don't fit in the 24 MB SLC. Storing the F32 router and loop gates as f16 would save about 0.15 GB per token, but it's a numerics change and has to pass the fixtures.

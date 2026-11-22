# Results summary

M4 Pro (20-core GPU, 48 GB, ~273 GB/s), macOS 27.0, AC power, quiet machine (checked with top, memory_pressure, wired memory).
Decode roofline ≈ 68 tok/s (≈4.0 GB of weights read per token, see docs/reference/ARCHITECTURE.md).

## Head-to-head, 2026-11-22 (`bench/run_all.sh`)

| engine | weights | pp512 | tg128 | pp4096 | tg128 @ 4096 ctx |
|---|---|---|---|---|---|
| ours `9c30f6b` (Metal) | IQ4_NL GGUF, 16 GB | 280.4 ± 2.7 | **38.6 ± 0.2** | 191.1 ± 21.3 | 20.0 ± 0.2 |
| llama.cpp `f7b65a6b5` (Metal) | same GGUF | 272.0 ± 51.9 | 26.1 ± 0.5 | 221.7 ± 0.8 | 25.3 ± 0.6 |
| mlx-vlm (Neywa-Labs fork, mlx 0.32.4) | VisualInference MLX 4-bit, 14.8 GiB | **411.8** | 29.5 | **352.7** | **25.5** |

- tok/s, mean ± stddev (mlx-vlm: median of 3 after warmup, its own reported rates; prompts cut from fixtures/prompts/long.txt).
- llama.cpp pp512 has one slow repeat (an earlier clean run of c66103334 measured 302.3 ± 0.4).
- ours tg128 at empty context: 57% of the roofline.
- Where we lose:
  - **decode at 4K context**: attn_flash gives each head one threadgroup, so at T = 1 only 24 threadgroups walk 4096 keys each, serially. It needs flash-decoding style split-K over the keys.
  - **long prefill**: f32 GEMM tiles, and attention within a 512-token ubatch is per query token with no tiling. mlx-vlm is about 1.5x ahead at pp512 and 1.8x at pp4096.

## Earlier runs
| date | engine | backend | pp512 | tg128 | tg128 @4K |
|---|---|---|---|---|---|
| 2026-11-22 | llama.cpp c66103334 | Metal | 302.3 ± 0.4 | 27.5 ± 0.1 | 26.1 ± 0.2 |
| 2026-11-22 | llama.cpp c66103334 | CPU, 10 threads | 91.3 ± 0.7 | 22.8 ± 4.1 (tg64) | – |

## Osaurus (vmlx-swift `310c583`, the Inkling JANGH merge), 2026-11-22

`OsaurusAI/Inkling-2-Mini-JANGH2` (10.2 GB: 2.34-bpw Hadamard-codebook experts, affine 8-bit trunk), run with vmlx-swift's RunBench on the same 512- and 4096-token prompts (fixtures/prompts/long.txt, HF tokenization), greedy, 128 tokens. Decode is measured *after* the prompt (at ~512 and ~4096 context).

| engine | weights | pp512 | decode @512 | pp4096 | decode @4096 |
|---|---|---|---|---|---|
| Osaurus | JANGH2 | 267 | 48.3 | 333 | 45.7 |
| ours `b402c50` | IQ4_NL GGUF | ~307 | 40.7 (@0) | ~213 | 34.2 |

Setup notes: RunBench's generic loader tries the VLM factory first, which can't decode `jangtq2`. The LLM factory has the Inkling JANGH preparation but refuses symlinked files, so the HF snapshot has to be copied as regular files (`cp -R -L -c`, an APFS clone) to `~/models/Inkling-2-Mini-JANGH2`. Most of the decode gap is format: about 3.1 vs 4.0 GB read per token.

## Ollama 0.40.17 (MLX runner, mlx 0.32.4), native NVFP4, 2026-11-22

`thinkingmachines/Inkling-2-Mini`, imported from the local HF snapshot with `ollama create` (FROM <snapshot dir>; "preserving source quantization": NVFP4 experts, MXFP8 last 4 layers, BF16 trunk; 19 GB). `ollama ps` reports the `mlx` runner. Measured with `bench/ollama_bench.py` from Ollama's own timing fields: median of 3 after a warmup, raw prompts cut from fixtures/prompts/long.txt, greedy, 128 tokens. Decode is measured after the prompt.

| engine | weights | pp (524 tok) | decode @524 | pp (4567 tok) | decode @4567 |
|---|---|---|---|---|---|
| Ollama | NVFP4 (official) | 292.0 | 20.9 | 303.7 | 21.0 |
| ours `082f75c` | IQ4_NL GGUF | ~325 (pp512) | 40.4 (@0) | ~315 (pp4096) | 34.2 (@4096) |

The MLX runner ignores `cache_prompt: false` and reuses a shared prompt prefix, so the script puts a random tag at the start of every prompt. Without it, prefill measures 9k–84k tok/s. Its decode is flat with context (about 21 tok/s), and BF16 attention weights make it read about 1.2 GB more per token than the Q8_0 attention in the GGUF (16 vs 8.5 bits).

### Official tag `inkling-2-mini:26b-a5b-nvfp4` (with the DSpark drafter), default server settings

`ollama create` from the HF snapshot drops `mtp.safetensors`, so the rows above ran without speculative decoding. The official tag (20.2 GB) carries the drafter as `draft.*` tensors, and the runner logs `Loaded draft model ... Inkling2ForConditionalGenerationMTP`.

| engine | pp (524 tok) | decode @524 | pp (4567 tok) | decode @4567 |
|---|---|---|---|---|
| Ollama official, drafting | 291.7 | 29.4 | 304.2 | 36.8 |
| Ollama import, no drafter | 292.0 | 20.9 | 303.7 | 21.0 |
| ours `082f75c`, no drafting | ~325 | 40.4 (@0) | ~315 | 34.2 (@4096) |

Runner stats on this greedy text: 50–65% of drafted tokens accepted, 1.6–3.9 tokens committed per verification step, up to 7 drafted. Gains depend on the text: more predictable continuations accept more drafts.

## Decode: concurrent dispatch groups + GPU-side greedy (5b01504 vs 0f01b77), 2026-11-23

`bench/ab.sh 0f01b77 2`, interleaved rounds, same machine state:

| test | base round 1 / 2 | new round 1 / 2 |
|---|---|---|
| pp512 | 329.3 / 333.4 | 339.9 / 311.3 |
| tg128 @ 0 | 40.47 / 40.48 | **43.43 / 42.16** |
| pp4096 | 324.2 / 315.1 | 328.1 / 307.3 |
| tg128 @ 4096 | 34.26 / 34.10 | **36.42 / 35.94** |

Decode +4-7% at empty context and +5-6% at 4K (64% / 54% of the roofline). The concurrent groups alone measured +1-3% in an earlier A/B; the rest is decode_greedy (argmax on the GPU, two command buffers in flight). Prefill moves both ways between rounds (thermal noise in round 2 for both builds).

## vs llama.cpp master and the open Inkling performance PRs, 2026-11-23

Same session, AC power, `llama-bench -ngl 99 -r 3` (`-d 4096` for decode at depth) on the same IQ4_NL GGUF; ours via `inkling bench`. Builds in worktrees under ~/sw/llama-prs, each at its PR head (19-27 commits behind master).

| engine | pp512 | pp4096 | tg128 | tg128 @ 4K |
|---|---|---|---|---|
| llama.cpp master `94f3af3d0`, `-fa 0` | 313.1 | 262.9 | 26.6 | 23.9 |
| llama.cpp master, `-fa 1` | 305.1 | 234.4 | 26.4 | 24.9 |
| + #32808 Metal indexed bias in FLASH_ATTN_EXT (on #32740) `167ad27b9` | **378.4** | **343.4** | 27.2 | 26.1 |
| + #32752 build the mask in chunks `56a4ac9ee` | 306.8 | 239.1 | 26.9 | 25.8 |
| ours `b4b99b7` | 326.6 | 319.9 | **42.9** | **37.2** |

- Master can't use flash attention for Inkling at all: the relative bias blocks it (docs/llama.cpp-backports.md). #32740/#32808 move the bias inside the flash-attention kernel, which makes llama.cpp's prefill faster than ours (+16% at 512, +7% at 4096). Its decode moves only +2-5%.
- #32752 is about compute-buffer memory; speed is neutral at these sizes.
- #32752 conflicts with #32808 in src/models/inkling2.cpp (both rewrite the mask/bias path), so they were not combined.
- Decode: ours is 1.58x the best llama.cpp build at short context and 1.43x at 4K.

### DSpark speculative decoding: llama.cpp #32218 vs ours, same chat prompt (LRU-cache code, 256 tokens, greedy)

llama.cpp through llama-server (`llama-speculative` can't create the DSpark draft context) with a drafter converted by the PR's own `convert_hf_to_gguf.py --dspark --outtype q8_0` from the native mtp.safetensors (unsloth's dspark GGUF has a padded markov_w1 that the PR rejects); bench/llama_dspark.sh. Ours: `inkling run --chat --draft`, INKLING_DRAFTS = drafts per round.

| drafts per round | llama.cpp tok/s (accepted / drafted) | ours tok/s (accepted drafts / round) |
|---|---|---|
| none | 27.13 | 40.8 |
| 1 | 25.08 (106 / 148) | 39.1 (0.67) |
| 2 | 25.67 (133 / 241) | 37.8 (1.00) |
| 3 | 25.08 (146 / 322) | 32.8 (1.11) |
| 4 | 23.80 (151 / 408) | – |
| 7 | 19.64 (153 / 694) | 19.3 (1.22) |

Speculation loses in both engines on this machine: verifying extra tokens of this MoE model costs about as much as it saves (docs/dspark.md). #32773 reports +68% on an RX 9070 XT, a GPU with far more compute per byte of bandwidth. llama.cpp accepts slightly more drafts (72% vs 67% at 1 draft).

## Prefill experiments, 2026-11-23

- Token-parallel short convolutions (committed): pp512 +7% / +1%, pp4096 −2% / +3% in a noisy two-round A/B; decode unchanged.
- Relative bias computed per score inside attn_tiled instead of the rel_table pass (reverted): pp4096 288 / 276 vs 338 / 313 (−12–15%), pp512 −1–3%. Sixteen extra multiply-adds and a device load per score stall the MMA loop more than the 25 MB/layer table costs. llama.cpp's #32808 gathers precomputed bias values in its kernel instead.

## Prefill: MoE tiling and occupancy (37b5137 vs 334f3cc), 2026-11-23

`bench/ab.sh 334f3cc 2`:

| test | base round 1 / 2 | new round 1 / 2 |
|---|---|---|
| pp512 | 340.9 / 341.0 | **440.7** / 357.2 |
| tg128 @ 0 | 43.26 / 43.17 | 43.37 / 41.06 |
| pp4096 | 331.2 / 303.0 | **405.6** / 356.4 |
| tg128 @ 4096 | 37.27 / 36.54 | 37.16 / 36.72 |

Round 1 is clean: prefill +29% at 512 and +22% at 4096, decode unchanged. In round 2 the machine throttled (the new build's decode, which runs code these commits did not touch, read 41.1 against 43.2), so its prefill gains are understated.

Where it came from, measured by running each kernel family twice (INKLING_DOUBLE) in a 512-token batch: two-thirds of prefill was MoE GEMMs, at ~3.5 TFLOP/s.
- 32- and 16-pair tiles from a work list (0c856a6): routing is uneven, so one-size 32-pair tiles ran 63% full. pp512 351 → 388.
- Bias table only for distances the batch reaches (3f6486c): pp512 → 394.
- MoE GEMM results written straight from the accumulators (6044abe): the f32 epilogue tiles had sized threadgroup memory at 16-20 KB, one MoE threadgroup per core. pp512 → 442.
- Token-parallel short convolutions (76004b0): a few percent.

Tried and dropped: relative bias computed inside attn_tiled (pp4096 −12-15%); per-token message-gate kernels for batches (pp512 351 → 302); 64-pair tiles for busy experts (slower, 413 vs 443, and produced wrong results, not debugged).

Updated standing against llama.cpp's best build (#32808, flash-attention bias, measured earlier today): pp512 378.4 vs ours ~441, pp4096 343.4 vs ~406, tg128 27.2 vs 43.4, tg128 @ 4K 26.1 vs 37.2.

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

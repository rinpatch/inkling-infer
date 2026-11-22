# Results summary

Decode roofline ≈ 70 tok/s (3.9 GB/token at 273 GB/s). "% roof" = tg / 70.

| date | engine | weights | backend | pp512 tok/s | tg128 tok/s | % roof | tg128 @4K | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-22 | llama.cpp c66103334 | IQ4_NL GGUF (16 GB) | Metal, -ngl 99 | 302.3 ± 0.4 | 27.5 ± 0.1 | 39% | 26.1 ± 0.2 | MLX weight download running in background (network only) |
| 2026-11-22 | llama.cpp c66103334 | IQ4_NL GGUF | CPU, 10 threads | 91.3 ± 0.7 | 22.8 ± 4.1 | 33% | – | tg64, 3 reps |

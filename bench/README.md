# Benchmarks

Machine: Mac with M4 Pro (10P + 4E CPU, 20-core GPU, 48 GB, ~273 GB/s), macOS 27.0, on AC power.

## Method
- Before every run, check `top`, `memory_pressure` and wired memory. Nothing else may hold a model resident (a stray `llama-server` caused the first run to fail with Metal OOM), and Spotlight must be excluded from `~/.cache/huggingface` and this repo.
- Use the same GGUF for llama.cpp and our engine (`Inkling-2-Mini-IQ4_NL.gguf`). MLX engines use their own 4-bit or original weights, so every result row also lists bits per weight and weight GiB, and quality gets compared through perplexity on the same text.
- pp512 / tg128 at empty context, then tg128 at 4096 depth. llama-bench defaults: 5 reps, mean ± stddev.
- Report decode as a percentage of the roofline: ≈3.9 GB read per token (docs/reference/ARCHITECTURE.md), so ≈70 tok/s at 273 GB/s.

## Engines
| engine | weights | how |
|---|---|---|
| llama.cpp `c66103334` | IQ4_NL GGUF (ggml-org) | `bench/run_llama.sh` |
| mlx-vlm (Neywa-Labs fork) | `VisualInference/Inkling-2-Mini-MLX-4bit` | published M4 Pro numbers: tg 31.1 / 27.6 / 26.5 at 28 / 1K / 4K prompt |
| inkling2-mini-mlx-fast `3558b31` | official NVFP4/MXFP8 checkpoint | `bench/engines/inkling2-mini-mlx-fast` |
| razorback16/inkling-infer `87d0e22` | official checkpoint | `bench/engines/razorback-inkling-infer` |
| realugbun llama.cpp inkling2-speed fork | IQ4_NL GGUF | to try |
| ours | IQ4_NL GGUF | `inkling bench` (Phase 5) |

## Results
See `results/*.json` (raw) and the table in `results/SUMMARY.md`.

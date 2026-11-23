# inkling-infer

A Rust + Metal inference engine for Inkling 2 Mini (`inkling2_loop` GGUF), benchmarked against llama.cpp and the MLX engines.
Plan: `~/.claude/plans/let-s-try-writing-a-functional-planet.md`. Model spec: `docs/reference/ARCHITECTURE.md` (read it before touching the forward pass); T vs L differences in `docs/reference/NOTES.md`.

## Paths
- Model: `~/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf`
- llama.cpp (oracle + baseline): `~/sw/llama.cpp`, tools in `build/bin` (`llama-bench`, `llama-eval-callback`, `llama-perplexity`, `llama-tokenize`)
- Python helpers run with `uv run --with numpy python -I <script>`. No global pip installs.

## Rules
- Correctness before speed. A change lands only if `cargo nextest run` passes, including the fixture tests and the 64-token greedy match with llama.cpp.
- The CPU reference path stays simple, f32 and unoptimized. It is the oracle for the Metal kernels.
- No silent fallbacks: if a Metal path is missing, return an error instead of quietly running on the CPU.
- Benchmarks: release build with LTO, plugged in, nothing else on the GPU. Report tok/s together with the % of the decode roofline (≈3.9 GB/token, 273 GB/s), and record the llama.cpp commit and macOS version.
- Commit at the end of each phase. The commit message explains the logic in detail: what the design is and why, the invariants, and how it was verified.

## Agent benchmark (SWE-bench)
`bench/swe/README.md`: ours vs llama-server vs Ollama on 20 SWE-bench Verified tasks via mini-swe-agent (containers on `nanakusa`). One command, `bench/swe/night.sh`; run it only when the user asks.

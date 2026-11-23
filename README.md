# inkling-infer

An experimental inference engine, written to see how far **Inkling 2 Mini** can be pushed on an **M4 Pro** laptop (20-core GPU, 48 GB, ~273 GB/s). It's Rust plus hand-written Metal kernels for exactly one model: the `inkling2_loop` GGUF at IQ4_NL.

## Current standing

| engine | weights | pp512 | pp4096 | decode | decode @ 4K |
|---|---|---|---|---|---|
| **ours** | IQ4_NL GGUF | **~441** | **~406** | 48.2 | 43.5 |
| llama.cpp #32808 `167ad27b9` | same GGUF | 378.4 | 343.4 | 27.2 | 26.1 |
| llama.cpp master `94f3af3d0` | same GGUF | 313.1 | 262.9 | 26.6 | 23.9 |
| mlx-vlm (Neywa-Labs fork `fd6404f`, mlx 0.32.4) | MLX 4-bit | 411.8 | 352.7 | 29.5 | 25.5 |
| Ollama 0.40.17, with the DSpark drafter | NVFP4 | 291.7 | 304.2 | 29.4 | 36.8 |
| Osaurus (vmlx-swift `310c583`) | JANGH2, 2.34-bit experts | 267 | 333 | **48.3** | **45.7** |

Measured in tok/s on the M4 Pro. A decode token reads ≈4.0 GB of weights, so the decode roofline is ≈68 tok/s. Details and history are in [bench/results/SUMMARY.md](bench/results/SUMMARY.md).

Our decode is 71% of the roofline. Osaurus reaches the same decode speed by reading less: about 3.1 GB per token against our 4.0.

## The model

Inkling 2 Mini is a 26B-total / ~5B-active MoE with an unusual shape. The full spec is in [docs/reference/ARCHITECTURE.md](docs/reference/ARCHITECTURE.md).

- **Looped.** Of 24 layers, the middle 12 run three times per token, so each token makes 48 layer visits.
- **Attention residuals.** Each layer's input is an attention-weighted mix of earlier block states, not a plain residual sum.
- **Attention.** 4 global layers; the rest use a 1,024-token sliding window. There's no RoPE: position comes from a relative bias computed from the hidden state.
- **Short convolutions** on K, V and the layer outputs.
- **MoE.** 128 experts, 8 active, softcapped SwiGLU.

## What's different from existing engines

General engines run Inkling as a graph of generic ops. This one is written for its shape alone:

- **Fused kernels across op boundaries.** Convs go straight into the KV cache, all loop messages for a pass come from one kernel, and depth mixing is fused with its norms. A decode token is about a thousand dispatches, and the small ones are tuned for latency, since the weight kernels already run at the bandwidth limit.
- **MoE prefill tiled to the routing.** Expert GEMMs take tiles sized to how many tokens each expert actually received.

Speculative decoding with the bundled DSpark drafter is implemented but breaks even at best: on this GPU, verifying extra tokens costs about as much as it saves ([docs/dspark.md](docs/dspark.md)).

## Layout

```
crates/gguf           GGUF reader
crates/inkling        model definition, tokenizer, f32 CPU reference (the oracle)
crates/inkling-metal  Metal runtime and kernels
crates/inkling-cli    the `inkling` binary: run, bench, serve
docs/                 model spec and design notes
bench/                comparison scripts, results, SWE-bench agent benchmark
fixtures/             llama.cpp activation dumps, prompts, chat-template cases
```

## Usage

```sh
cargo build --release
M=~/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf

target/release/inkling run   -m $M -p "Write a haiku about caches" --chat -n 128
target/release/inkling bench -m $M --pp 512 --tg 128 [-d 4096]
target/release/inkling serve -m $M --port 8080 --ctx 65536
```

`serve` is an OpenAI-compatible chat completions endpoint with tool calls and prefix reuse between requests. `--draft <dir>` turns on DSpark, using the native `thinkingmachines/Inkling-2-Mini` checkpoint.

## Testing

```sh
cargo nextest run --release
```

The tests compare the Metal path against llama.cpp activation dumps, the CPU reference, and a 64-token greedy match with llama.cpp. They skip if the model or the fixtures are missing.

## Benchmarks

[bench/README.md](bench/README.md) covers the head-to-head runs. [bench/swe/README.md](bench/swe/README.md) covers an agent benchmark: 20 SWE-bench Verified tasks through mini-swe-agent, against llama-server and Ollama.

#!/bin/sh
# llama.cpp baseline for the IQ4_NL GGUF. Run on a quiet machine (see bench/README.md).
set -eu
LLAMA=${LLAMA:-$(cd "$(dirname "$0")/.." && pwd)/../llama.cpp}
M=${INKLING_MODEL:-$(ls "$HOME"/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf)}
B=$LLAMA/build/bin/llama-bench
C=$(git -C "$LLAMA" rev-parse --short HEAD)
R=$(cd "$(dirname "$0")" && pwd)/results
mkdir -p "$R"

# pp512 / tg128 at empty context, then decode at 4K depth (KV reads start to matter), then CPU
"$B" -m "$M" -ngl 99 -p 512 -n 128 -r 5 -o json > "$R/llama.cpp-$C-metal.json"
"$B" -m "$M" -ngl 99 -p 0 -n 128 -d 4096 -r 3 -o json > "$R/llama.cpp-$C-metal-d4096.json"
"$B" -m "$M" -ngl 0 -t 10 -p 512 -n 64 -r 3 -o json > "$R/llama.cpp-$C-cpu.json"

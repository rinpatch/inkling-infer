#!/bin/sh
# Head-to-head on a quiet machine: ours, llama.cpp, mlx-vlm, razorback/inkling-infer.
# Tests: pp512, tg128 at empty context, pp4096, tg128 after 4096 tokens of context.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
R=$here/results
mkdir -p "$R"
M=${INKLING_MODEL:-$(ls "$HOME"/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf)}
LLAMA=${LLAMA:-$HOME/sw/llama.cpp}
C=$(git -C "$LLAMA" rev-parse --short HEAD)
OURS=$(git -C "$repo" rev-parse --short HEAD)
INK=$repo/target/release/inkling

echo "== ours ($OURS)"
(cd "$repo" && cargo build --release -q -p inkling-cli)
"$INK" bench -m "$M" --pp 512 --tg 128 -r 5 --json "$R/ours-$OURS-d0.json"
"$INK" bench -m "$M" --pp 4096 --tg 128 -d 4096 -r 3 --json "$R/ours-$OURS-d4096.json"

echo "== llama.cpp ($C)"
B=$LLAMA/build/bin/llama-bench
"$B" -m "$M" -ngl 99 -p 512,4096 -n 128 -r 3 -o json > "$R/llama.cpp-$C-pp.json"
"$B" -m "$M" -ngl 99 -p 0 -n 128 -d 4096 -r 3 -o json > "$R/llama.cpp-$C-d4096.json"

echo "== mlx-vlm (VisualInference 4-bit)"
"$here/engines/mlx-vlm-venv/bin/python" -I "$here/mlx_vlm_bench.py" VisualInference/Inkling-2-Mini-MLX-4bit \
  "$repo/fixtures/prompts/long.txt" "$R/mlx-vlm-4bit.json"

echo "== razorback inkling-infer (official checkpoint, custom kernels)"
RB=$here/engines/razorback-inkling-infer
(cd "$RB" && .venv/bin/python decode_benchmark.py --kernel | tee "$R/razorback-decode-d4096.txt")
(cd "$RB" && .venv/bin/python prefill_benchmark.py | tee "$R/razorback-pp4096.txt")

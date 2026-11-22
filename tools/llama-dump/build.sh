#!/bin/sh
# Builds llama-dump against the llama.cpp checkout used as the oracle.
set -eu
LLAMA=${LLAMA:-$HOME/sw/llama.cpp}
here=$(cd "$(dirname "$0")" && pwd)
clang++ -std=c++17 -O2 \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/common" \
  "$here/dump.cpp" -o "$here/llama-dump" \
  -L"$LLAMA/build/bin" -lllama -lllama-common -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin"
echo "built $here/llama-dump (llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD))"

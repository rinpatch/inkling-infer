#!/bin/sh
# Interleaved A/B of the working tree against a base commit, on one quiet machine.
# usage: bench/ab.sh <base-commit> [rounds=2]
# The baseline is built in its own worktree and target dir: the binary bakes in the absolute path
# of its metallib, so sharing a target dir would make both binaries load the newest kernels.
set -eu
base=$1; rounds=${2:-2}
repo=$(cd "$(dirname "$0")/.." && pwd)
tmp=${TMPDIR:-/tmp}/inkling-ab
M=${INKLING_MODEL:-$(ls "$HOME"/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf)}
mkdir -p "$tmp"
git -C "$repo" worktree remove --force "$tmp/base-wt" 2>/dev/null || true
git -C "$repo" worktree add -q "$tmp/base-wt" "$base"
(cd "$tmp/base-wt" && CARGO_TARGET_DIR="$tmp/base-target" cargo build --release -q -p inkling-cli)
(cd "$repo" && CARGO_TARGET_DIR="$tmp/new-target" cargo build --release -q -p inkling-cli)
git -C "$repo" worktree remove --force "$tmp/base-wt"
for r in $(seq "$rounds"); do
  for b in base new; do
    echo "## round $r $b"
    "$tmp/$b-target/release/inkling" bench -m "$M" --pp 512 --tg 128 -r 3 2>&1 | grep -v 'last token'
    "$tmp/$b-target/release/inkling" bench -m "$M" --pp 4096 --tg 128 -d 4096 -r 2 2>&1 | grep -v 'last token'
  done
done

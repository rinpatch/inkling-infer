#!/bin/sh
# Scores a run's preds.json with the SWE-bench harness on nanakusa; copies the report back.
# usage: bench/swe/score.sh <label>
set -eu
label=$1
repo=$(cd "$(dirname "$0")/../.." && pwd)
host=${SCORE_HOST:-nanakusa}
scp -q "$repo/bench/swe/runs/$label/preds.json" "$host:swe-eval/$label.json"
ssh "$host" "cd ~/swe-eval && .venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified --predictions_path $label.json \
  --max_workers 2 --run_id $label" | tail -12
ssh "$host" "cat ~/swe-eval/*.$label.json" > "$repo/bench/swe/runs/$label/report.json"

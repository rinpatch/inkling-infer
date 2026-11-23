#!/bin/sh
# mini-swe-agent on bench/swe/instances.txt against one OpenAI-compatible server.
# usage: bench/swe/run.sh <label> <api_base> [litellm model name]
# Containers run on $DOCKER_HOST (default ssh://nanakusa) so the Mac only holds the model.
set -eu
label=$1; api_base=$2; model=${3:-openai/inkling-2-mini}
repo=$(cd "$(dirname "$0")/../.." && pwd)
filter="^($(paste -sd'|' "$repo/bench/swe/instances.txt"))\$"
export DOCKER_HOST=${DOCKER_HOST:-ssh://nanakusa}
export MSWEA_COST_TRACKING=ignore_errors
exec "$repo/bench/engines/swe/.venv/bin/mini-extra" swebench --subset verified --split test \
  --filter "$filter" -w 1 -o "$repo/bench/swe/runs/$label" \
  -c swebench.yaml -c "$repo/bench/swe/local.yaml" \
  -c "model.model_name=$model" -c "model.model_kwargs.api_base=$api_base"

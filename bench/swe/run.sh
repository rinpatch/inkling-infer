#!/bin/sh
# mini-swe-agent on bench/swe/instances.txt against one OpenAI-compatible server.
# usage: bench/swe/run.sh <label> <api_base> [litellm model name]
# Containers run on $DOCKER_HOST (a remote Docker host) so the Mac only holds the model.
set -eu
label=$1; api_base=$2; model=${3:-openai/inkling-2-mini}
repo=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$repo/bench/swe/local.env" ] && . "$repo/bench/swe/local.env"
filter="^($(paste -sd'|' "$repo/bench/swe/instances.txt"))\$"
: "${DOCKER_HOST:?set DOCKER_HOST, e.g. in bench/swe/local.env}"
export DOCKER_HOST
export MSWEA_COST_TRACKING=ignore_errors
exec "$repo/bench/engines/swe/.venv/bin/mini-extra" swebench --subset verified --split test \
  --filter "$filter" -w 1 -o "$repo/bench/swe/runs/$label" \
  -c swebench.yaml -c "$repo/bench/swe/local.yaml" \
  -c "model.model_name=$model" -c "model.model_kwargs.api_base=$api_base"

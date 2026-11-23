#!/bin/bash
# Runs the SWE-bench comparison end to end, one engine at a time (see bench/swe/README.md):
# start the server, run mini-swe-agent on the 20 instances, stop the server, score on nanakusa.
# usage: bench/swe/night.sh [inkling] [llama] [ollama]   (default: all three, in that order)
# Re-running resumes: mini-swe-agent skips instances that already have a trajectory.
set -u
repo=$(cd "$(dirname "$0")/../.." && pwd)
runs=$repo/bench/swe/runs
M=$(ls "$HOME"/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf)
CTX=131072
LLAMA_SERVER=${LLAMA_SERVER:-$HOME/sw/llama-prs/fa/build/bin/llama-server}
OLLAMA_TAG=${OLLAMA_TAG:-inkling-2-mini:26b-a5b-nvfp4}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_up() {
  for _ in $(seq 900); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# one tool-calling request: the engine must answer with a bash tool call
smoke() {
  curl -sf "$1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$2\", \"temperature\": 0, \"max_tokens\": 2000,
    \"messages\": [{\"role\": \"user\", \"content\": \"Use the bash tool to list /tmp.\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Run a bash command\",
      \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}}]
  }" | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; assert m.get("tool_calls"), m; print("smoke ok:", m["tool_calls"][0]["function"])'
}

for port in 8080 8081 11434; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is in use; stop whatever serves it first (an Ollama app, an old server)" >&2
    exit 1
  fi
done

caffeinate -dimsu -w $$ &
engines=("$@")
[ ${#engines[@]} -eq 0 ] && engines=(inkling llama ollama)

for e in "${engines[@]}"; do
  mkdir -p "$runs/$e"
  case $e in
    inkling)
      "$repo/target/release/inkling" serve -m "$M" --port 8080 --ctx $CTX \
        --log "$runs/$e/requests.jsonl" >>"$runs/$e/server.log" 2>&1 &
      pid=$!; base=http://127.0.0.1:8080/v1; model=inkling-2-mini; health=http://127.0.0.1:8080/health ;;
    llama)
      "$LLAMA_SERVER" -m "$M" -c $CTX -np 1 -fa on -ngl 99 --jinja --port 8081 \
        >>"$runs/$e/server.log" 2>&1 &
      pid=$!; base=http://127.0.0.1:8081/v1; model=inkling-2-mini; health=http://127.0.0.1:8081/health ;;
    ollama)
      OLLAMA_CONTEXT_LENGTH=$CTX ollama serve >>"$runs/$e/server.log" 2>&1 &
      pid=$!; base=http://127.0.0.1:11434/v1; model=$OLLAMA_TAG; health=http://127.0.0.1:11434/api/version ;;
    *) echo "unknown engine $e" >&2; exit 1 ;;
  esac
  log "$e: starting server (pid $pid)"
  if ! wait_up "$health"; then
    log "$e: server did not come up; see $runs/$e/server.log"
    kill $pid 2>/dev/null; continue
  fi
  if ! smoke "$base" "$model" >>"$runs/$e/smoke.log" 2>&1; then
    log "$e: smoke test failed; see $runs/$e/smoke.log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; continue
  fi
  log "$e: running agent"
  "$repo/bench/swe/run.sh" "$e" "$base" "openai/$model" >>"$runs/$e/agent.log" 2>&1
  log "$e: agent done (exit $?)"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  log "$e: scoring"
  "$repo/bench/swe/score.sh" "$e" >>"$runs/$e/score.log" 2>&1 || log "$e: scoring failed; see $runs/$e/score.log"
done
python3 "$repo/bench/swe/summarize.py" "${engines[@]}" | tee "$runs/summary.md"

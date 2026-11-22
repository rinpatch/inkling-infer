#!/bin/bash
# llama.cpp (PR #32218) decode speed with and without DSpark, through llama-server
M=$(ls ~/.cache/huggingface/hub/models--ggml-org--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-IQ4_NL.gguf)
DS=$1; LOG=${2:-/tmp}
P='<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Write a Python class implementing an LRU cache with get and put methods, with type hints and docstrings, then show a short usage example.<|end_message|><|message_model|>'
BODY=$(python3 -I -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 256, "temperature": 0, "cache_prompt": False}))' "$P")
for nmax in 0 1 2 3 4 7; do
  extra=()
  [ $nmax -gt 0 ] && extra=(-md "$DS" --spec-type draft-dspark --spec-draft-n-max $nmax -ngld 99)
  ~/sw/llama-prs/dspark/build/bin/llama-server -m "$M" -ngl 99 -fa on -c 4096 --port 8099 "${extra[@]}" > "$LOG/srv_$nmax.log" 2>&1 &
  pid=$!
  for i in $(seq 180); do curl -s localhost:8099/health | grep -q ok && break; sleep 1; done
  for r in 1 2; do res=$(curl -s --max-time 300 localhost:8099/completion -d "$BODY"); done
  echo "n_max=$nmax: $(echo "$res" | python3 -I -c 'import json,sys
try:
    t=json.load(sys.stdin)["timings"]; print("%.2f tok/s, %d tokens, drafted=%s accepted=%s" % (t["predicted_per_second"], t["predicted_n"], t.get("draft_n"), t.get("draft_n_accepted")))
except Exception as e: print("failed", e)')"
  kill $pid; wait $pid 2>/dev/null
done

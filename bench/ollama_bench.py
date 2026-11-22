# Ollama baseline: prompt processing and decode speed from Ollama's own timing fields.
# usage: python3 -I bench/ollama_bench.py <model-tag> <prompt.txt> [out.json]
# The prompt text is cut to about 512 and 4096 tokens (by Ollama's own count, reported back);
# raw mode skips the chat template; decode generates 128 tokens greedily. Each size runs once as
# warmup (model load, caches) and then 3 measured times; medians are reported.
import json
import random
import statistics
import sys
import urllib.request

tag, text_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
text = open(text_path).read()
URL = "http://127.0.0.1:11434/api/generate"


def generate(prompt):
    body = {
        "model": tag,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        # keep_alive keeps the model resident. The MLX runner ignores cache_prompt, so callers
        # make every prompt differ from its first token to force a full prefill.
        "keep_alive": "10m",
        "options": {"num_predict": 128, "temperature": 0, "num_ctx": 8192, "cache_prompt": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


rows = []
# ~3.43 characters per token on this text by Ollama's count
for target, chars in ((512, 1740), (4096, 14000)):
    pp, tg, counts = [], [], []
    for rep in range(4):
        r = generate(f"[run {target}-{rep}-{random.random():.6f}] " + text[:chars])
        if rep:
            pp.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
            tg.append(r["eval_count"] / (r["eval_duration"] / 1e9))
        counts.append(r["prompt_eval_count"])
    row = dict(engine=f"ollama {tag}", target=target, prompt_tokens=counts[-1], pp_tps=statistics.median(pp),
               tg_tps=statistics.median(tg), pp_samples=pp, tg_samples=tg)
    print(json.dumps(row), flush=True)
    rows.append(row)
if out_path:
    json.dump(rows, open(out_path, "w"), indent=1)

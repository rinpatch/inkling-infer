# mlx-vlm baseline: prompt processing and decode speed as reported by mlx-vlm itself.
# usage: bench/engines/mlx-vlm-venv/bin/python -I bench/mlx_vlm_bench.py <model> <prompt.txt> [out.json]
# Prompts are cut to 512 and 4096 tokens of the given text; decode generates 128 tokens greedily.
import json
import statistics
import sys

from mlx_vlm import load, stream_generate

model_path, text_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
model, processor = load(model_path)
tok = getattr(processor, "tokenizer", processor)
ids = tok.encode(open(text_path).read(), add_special_tokens=False)

rows = []
for n in (512, 4096):
    prompt = tok.decode(ids[:n])
    pp, tg, actual = [], [], 0
    for rep in range(4):  # rep 0 is warmup
        last = None
        for r in stream_generate(model, processor, prompt, max_tokens=128, temperature=0.0):
            last = r
        if rep:
            pp.append(last.prompt_tps)
            tg.append(last.generation_tps)
        actual = last.prompt_tokens
    row = dict(engine="mlx-vlm", prompt_tokens=actual, pp_tps=statistics.median(pp), tg128_tps=statistics.median(tg),
               pp_samples=pp, tg_samples=tg, peak_gb=last.peak_memory)
    print(json.dumps(row), flush=True)
    rows.append(row)
if out_path:
    json.dump(rows, open(out_path, "w"), indent=1)

# SWE-bench agent benchmark: ours vs llama.cpp vs Ollama

Real-world comparison: the same agent (mini-swe-agent 2.4.6, default `swebench.yaml` config with
OpenAI tool calls) solves the same 20 SWE-bench Verified tasks through each engine's
OpenAI-compatible server. It measures end-to-end behaviour under long, growing contexts with
prefix reuse (contexts reach ~40K tokens; ~98% of prompt tokens are cache hits on our server),
plus pass rate.

Run it only when the user says so: it keeps the GPU busy for many hours.

## Layout

- Model servers run on this Mac (M4 Pro, 48 GB). One engine at a time; nothing else on the GPU.
- Agent containers and scoring run on a separate x86-64 Linux Docker host: the Mac cannot hold
  the model and a Docker VM at once. Put `DOCKER_HOST` (e.g. `ssh://<host>`), `SCORE_HOST` and
  `LLAMA_SERVER` in `bench/swe/local.env` (not in git); the scripts source it.
- `instances.txt`: 20 instances, SWE-bench Verified difficulty "<15 min fix", seed 0.
  Pull their images (`swebench/sweb.eval.x86_64.*`) on the Docker host beforehand.
- `local.yaml`: overlay on mini-swe-agent's `swebench.yaml` (step limit 100, no cost limit, cost
  tracking off, API base set by `run.sh`). Sampling and thinking effort are each engine's and the
  model's defaults (temperature 1.0, template effort 0.9).
- Agent venv: `bench/engines/swe/.venv` (mini-swe-agent, datasets; not in git). Recreate with
  `uv venv -p 3.12 bench/engines/swe/.venv && uv pip install -p bench/engines/swe/.venv/bin/python mini-swe-agent datasets`.
- Scoring venv: `~/swe-eval/.venv` on the scoring host (swebench 5.0.2). Use dataset
  `SWE-bench/SWE-bench_Verified` (the princeton-nlp name lacks the `image` field 5.x needs).

## Engines

| label | server | weights | context |
|---|---|---|---|
| `inkling` | `target/release/inkling serve` (this repo) | IQ4_NL GGUF | 131072 |
| `llama` | llama.cpp #32808 build, `$LLAMA_SERVER -c 131072 -np 1 -fa on --jinja` | same GGUF | 131072 |
| `ollama` | `OLLAMA_CONTEXT_LENGTH=131072 ollama serve`, tag `inkling-2-mini:26b-a5b-nvfp4` | official NVFP4 (+ DSpark drafter) | 131072 |

`-np 1` matters: llama-server otherwise splits the context across parallel slots. Ollama's
default context is small, hence the environment variable. Ollama's weights differ (NVFP4, not the
GGUF), so its pass rate also reflects a different quantization.

## Running

Before starting:
1. Mac plugged in (`pmset -g batt` says AC Power), no other GPU jobs (`pgrep -fl "inkling|llama|ollama|nextest"`),
   no Ollama app running (ports 8080, 8081, 11434 free; `night.sh` checks).
2. `cargo build --release -p inkling-cli` (the server must be built from the commit being measured).
3. `docker ps` works against `$DOCKER_HOST`, and `docker images | grep -c sweb.eval` there prints 20.

Then, in the background (it takes roughly 4-6 h for ours, longer for the other two):

```sh
bench/swe/night.sh                 # all three: inkling, llama, ollama
bench/swe/night.sh llama ollama    # a subset; re-running resumes unfinished instances
```

For each engine it starts the server, waits for health, sends one tool-calling smoke request
(fails fast if the engine cannot produce tool calls), runs `run.sh`, stops the server, and runs
`score.sh` on the scoring host. Everything lands in `bench/swe/runs/<label>/`: `preds.json`,
`report.json` (harness result), `server.log`, `agent.log`, `score.log`, `smoke.log`, our
server's per-request `requests.jsonl`, and one trajectory directory per instance (git-ignored).
At the end it writes `runs/summary.md` with `summarize.py`.

Manual pieces, if needed:
- `bench/swe/run.sh <label> <api_base> <litellm model>`: agent only, against a running server.
- `bench/swe/score.sh <label>`: scoring only.
- `python3 bench/swe/summarize.py inkling llama ollama`: the comparison tables.

Monitoring: `tail -f bench/swe/runs/<label>/agent.log`; our server prints one line per request
(prompt / cached tokens, prefill and decode speed) in `server.log`.

## After the run

Record in `bench/results/SUMMARY.md`: the summary tables, commits (ours, llama.cpp `167ad27b9`),
Ollama version (`ollama --version`), macOS version, and anything odd in the logs (smoke failures,
context overflows, crashed servers). Commit `bench/swe/runs/*/{preds.json,report.json,*.log,requests.jsonl}`
and `runs/summary.md`.

## Pilot (for scale)

`runs/pilot-inkling`: psf__requests-5414 on our server at 48K context, resolved. 56 model calls,
17.5K generated tokens, contexts up to 38.9K tokens, 98.4% of prompt tokens served from the prefix
cache. 26 min wall, but ~20 min of that overlapped a GPU test run; undisturbed decode was ~32 tok/s
at 30-38K context, so expect ~10-15 min per instance.

## Known issues

- zsh: `"$n:latest"` applies zsh's `:l` modifier; write `"${n}:latest"` in loops.
- Our server serves one sequence at a time (requests queue); keep `-w 1`.
- Prefill of a short suffix at ~37K context runs only ~50-250 tok/s (e.g. 218 tokens in 4.2 s);
  not yet profiled.

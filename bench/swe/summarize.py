"""Compares mini-swe-agent runs in bench/swe/runs/<label>/ (trajectories + report.json from score.sh).

usage: python3 bench/swe/summarize.py <label> [<label> ...]

Per run: instances resolved (SWE-bench harness), submitted, model calls, generated and prompt
tokens, the share of prompt tokens served from the prefix cache (as the server reports it in
usage.prompt_tokens_details.cached_tokens; Ollama does not report it), and agent wall time
(first to last model reply, so it includes command execution on the Docker host).
"""

import glob
import json
import os
import sys

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load(label):
    d = os.path.join(RUNS, label)
    resolved = None
    rp = os.path.join(d, "report.json")
    if os.path.exists(rp):
        resolved = set(json.load(open(rp)).get("resolved_ids", []))
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*", "*.traj.json"))):
        t = json.load(open(f))
        iid = t["instance_id"]
        calls = [m for m in t["messages"] if m.get("role") == "assistant" and "extra" in m]
        gen = prompt = cached = 0
        cached_known = False
        for m in calls:
            u = (m["extra"].get("response") or {}).get("usage") or {}
            gen += u.get("completion_tokens") or 0
            prompt += u.get("prompt_tokens") or 0
            det = u.get("prompt_tokens_details") or {}
            if det.get("cached_tokens") is not None:
                cached_known = True
                cached += det["cached_tokens"]
        ts = [m["extra"]["timestamp"] for m in calls if m["extra"].get("timestamp")]
        rows.append(
            {
                "id": iid,
                "status": t["info"].get("exit_status"),
                "resolved": None if resolved is None else iid in resolved,
                "calls": len(calls),
                "gen": gen,
                "prompt": prompt,
                "cached": cached if cached_known else None,
                "wall_s": (ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
                "max_ctx": max(
                    [((m["extra"].get("response") or {}).get("usage") or {}).get("prompt_tokens") or 0 for m in calls]
                    or [0]
                ),
            }
        )
    return rows


def main():
    labels = sys.argv[1:]
    if not labels:
        sys.exit(__doc__)
    print("| run | resolved | submitted | calls | gen tok | prompt tok | cached | max ctx | wall h | gen tok / wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    for lab in labels:
        rows = load(lab)
        per[lab] = {r["id"]: r for r in rows}
        n = len(rows)
        res = sum(1 for r in rows if r["resolved"]) if rows and rows[0]["resolved"] is not None else None
        sub = sum(1 for r in rows if r["status"] == "Submitted")
        calls = sum(r["calls"] for r in rows)
        gen = sum(r["gen"] for r in rows)
        prompt = sum(r["prompt"] for r in rows)
        cached = None if any(r["cached"] is None for r in rows) else sum(r["cached"] for r in rows)
        wall = sum(r["wall_s"] for r in rows)
        print(
            f"| {lab} | {'?' if res is None else f'{res}/{n}'} | {sub}/{n} | {calls} | {gen} | {prompt} | "
            f"{'n/a' if cached is None else f'{100 * cached / max(prompt, 1):.1f}%'} | "
            f"{max((r['max_ctx'] for r in rows), default=0)} | {wall / 3600:.2f} | {gen / max(wall, 1):.1f} |"
        )
    ids = sorted({i for p in per.values() for i in p})
    print()
    print("| instance | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for i in ids:
        cells = []
        for lab in labels:
            r = per[lab].get(i)
            if r is None:
                cells.append("-")
                continue
            mark = {True: "resolved", False: "fail", None: r["status"]}[r["resolved"]]
            cells.append(f"{mark}, {r['calls']} calls, {r['wall_s'] / 60:.0f} min")
        print(f"| {i} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

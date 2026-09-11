#!/usr/bin/env python3
"""Concurrency scaling of PLD (n-gram->MTP chain) vs production MTP, on heretic Q4_K_S.

Decision question: the owner wants to move production to `--spec-type ngram-mod,draft-mtp` and possibly
raise `-np` from 2 to 4. Does the n-gram win survive concurrency, and does per-stream TG collapse (the
owner's disqualifier)? Server is relaunched with matching -np for each level (KV split changes with -np).

configs : draft-mtp (production) vs ngram+mtp (candidate)
np      : 1, 2, 4  (server -np set to match; N identical concurrent streams)
prompts : code-edit (high overlap, realistic agent work) and copy (max overlap, ceiling)
each stream forced to 512 tokens, temp 0. Reports per-stream decode mean/min, aggregate tok/s, accept.
Production stopped for the run, restored in finally.
"""
import os, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/benchmarks/core19"))
import pld_test as P  # reuse BIN/args/prompts/call/gpu/stop/restore
import json

OUT = P.OUT
CONFIGS = ("draft-mtp", "ngram+mtp")
NPS = tuple(int(x) for x in os.environ.get("NPS", "1,2,4").split(","))
_MERGE = os.environ.get("MERGE_JSON")  # keep prior grid entries across split runs
PROMPTS = {"code": P.PROMPTS["code"], "copy": P.PROMPTS["copy"]}


def launch(cfg, np):
    subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", "llama-server .*--port 8085"], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    time.sleep(2)
    args = [a for a in P.BASE_ARGS]
    i = args.index("-np"); args[i + 1] = str(np)          # override -np
    args = args + P.SPEC[cfg]
    log = open(os.path.join(OUT, f"conc_{cfg}_np{np}.log"), "ab")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for k in range(150):
        try:
            if urllib.request.urlopen(P.BASE + "/health", timeout=4).status == 200:
                return proc, k * 2
        except Exception:
            pass
        if proc.poll() is not None:
            return None, None
        time.sleep(2)
    return None, None


def run_conc(prompt, n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.time()
        rows = list(ex.map(lambda _: P.call(prompt), range(n)))
        wall = time.time() - t0
    dec = [r["decode_tok_s"] for r in rows]
    accs = [r["accept"] for r in rows if r["accept"] is not None]
    toks = sum(r["completion_tokens"] for r in rows)
    return {"per_stream_decode_mean": round(statistics.mean(dec), 2),
            "per_stream_decode_min": round(min(dec), 2),
            "aggregate_tok_s": round(toks / wall, 2),
            "accept_mean": round(statistics.mean(accs), 3) if accs else None,
            "wall_s": round(wall, 1)}


def main():
    res = {"started": time.strftime("%F %T"), "model": os.path.basename(P.MODEL), "gen_tokens": P.GEN, "grid": {}}
    if _MERGE and os.path.exists(os.path.join(OUT, "pld_conc_results.json")):
        try:
            res["grid"] = json.load(open(os.path.join(OUT, "pld_conc_results.json"))).get("grid", {})
        except Exception:
            pass
    save = lambda: open(os.path.join(OUT, "pld_conc_results.json"), "w").write(json.dumps(res, indent=2) + "\n")
    print("stopping production...", flush=True)
    if not P.stop_production():
        print("ABORT: GPU not freed"); return
    print(f"  GPU0 {P.gpu0_mib()} MiB", flush=True)
    for np in NPS:
        for cfg in CONFIGS:
            key = f"np{np}/{cfg}"
            proc, load_s = launch(cfg, np)
            if not proc:
                print(f"  {key}: server FAILED"); res["grid"][key] = {"error": "start failed"}; save(); continue
            P.call(PROMPTS["code"], gen=32)  # warm
            res["grid"][key] = {"load_s": load_s, "gpu0_mib": P.gpu0_mib()}
            streams = int(os.environ["STREAMS"]) if os.environ.get("STREAMS") else np
            for pk, prompt in PROMPTS.items():
                r = run_conc(prompt, streams)
                res["grid"][key][pk] = r
                print(f"  {key:<18} {pk:<4}: per-stream {r['per_stream_decode_mean']:>7} tok/s "
                      f"(min {r['per_stream_decode_min']}) | aggregate {r['aggregate_tok_s']:>7} | accept {r['accept_mean']}", flush=True)
                save()
            subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    res["finished"] = time.strftime("%F %T"); save()
    print("\n==== per-stream decode scaling (code-edit) ====", flush=True)
    for cfg in CONFIGS:
        row = " ".join(f"np{np}={res['grid'].get(f'np{np}/{cfg}',{}).get('code',{}).get('per_stream_decode_mean','?')}" for np in NPS)
        agg = " ".join(f"np{np}={res['grid'].get(f'np{np}/{cfg}',{}).get('code',{}).get('aggregate_tok_s','?')}" for np in NPS)
        print(f"  {cfg:<10} per-stream {row}", flush=True)
        print(f"  {cfg:<10} aggregate  {agg}", flush=True)
    print("PLD CONC DONE ->", os.path.join(OUT, "pld_conc_results.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\nrestoring production...", flush=True)
        ok = P.restore_production()
        print("production healthy on 8080:", ok, "| GPU0", P.gpu0_mib(), "MiB", flush=True)

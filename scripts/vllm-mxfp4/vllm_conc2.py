#!/usr/bin/env python3
"""Concurrency ladder matching scripts/harness/conc_test.py methodology.

Why this exists: vllm_conc.py used an 8k context and 512 generated tokens, so prefill
dominated the wall clock and "aggregate" was an END-TO-END figure that plateaued at
N=2. The historical vLLM numbers in docs/02 (1,634 @ n=80 on 35B-A3B; 2,485 @ n=80 on
Gemma-4-12B) come from conc_test.py, which uses a ~70-token prompt and MAXTOK=800 --
prefill negligible, so total_tok/wall measures DECODE concurrency. Same metric name,
different quantity. This file reproduces the historical method so the numbers compare.

Counts content AND reasoning: with a reasoning parser the thinking text lands in its own
field, and counting only `content` yields bogus low rows (the trap conc_test.py documents).

Usage: vllm_conc2.py <tag> [port] [max_n]
"""
import os
import json, statistics, sys, threading, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "conc2"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
MAX_N = int(sys.argv[3]) if len(sys.argv) > 3 else 96
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen3.8-PARO"
OUTDIR = os.path.expanduser("~/benchmarks")
MAXTOK = 800

# verbatim from scripts/harness/conc_test.py
PROMPT = ("Write a comprehensive, well-commented Python module implementing: an LRU cache with "
          "TTL expiry, a thread-safe bounded queue, a retry decorator with exponential backoff "
          "and jitter, and a binary search tree with insert/delete/traverse. Include type hints, "
          "docstrings, and a demonstration section exercising every component.")


def one(idx, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK, "temperature": 0, "seed": 42}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        wall = time.time() - t0
        u = d["usage"]
        tok = u["completion_tokens"]          # includes reasoning tokens
        m = d["choices"][0]["message"]
        txt = (m.get("content") or "") + (m.get("reasoning") or "") + (m.get("reasoning_content") or "")
        tail = txt[-300:].split()
        results[idx] = {"ok": True, "wall": wall, "tok": tok,
                        "tok_s": tok / wall if wall else 0,
                        "prompt_tokens": u["prompt_tokens"],
                        "tail_distinct": round(len(set(tail)) / max(1, len(tail)), 2)}
    except Exception as e:
        results[idx] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"}


def gpu_power():
    import subprocess
    try:
        o = subprocess.run(["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=15).stdout
        for l in o.splitlines():
            if "GPU[0]" in l and "Average Graphics Package Power" in l:
                return float(l.rsplit(":", 1)[1].strip())
    except Exception:
        pass
    return None


def cpu_snapshot():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            out[int(f[0][3:])] = (sum(int(x) for x in f[1:]), int(f[4]))
    return out


def cpu_busy(a, b):
    res = {}
    for c in a:
        dt = b[c][0] - a[c][0]
        di = b[c][1] - a[c][1]
        res[c] = 100.0 * (dt - di) / dt if dt > 0 else 0.0
    return res


def run(n, rows):
    results = {}
    threads = [threading.Thread(target=one, args=(i, results)) for i in range(n)]
    c0 = cpu_snapshot()
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(min(15, 4 + n * 0.1))
    pw = gpu_power()
    for t in threads:
        t.join()
    wall = time.time() - t0
    busy = cpu_busy(c0, cpu_snapshot())
    ok = [r for r in results.values() if r.get("ok")]
    bad = [r for r in results.values() if not r.get("ok")]
    if not ok:
        print(f"  n={n}: ALL FAILED -> {bad[0].get('err') if bad else '?'}", flush=True)
        rows.append({"n": n, "all_failed": True, "err": bad[0].get("err") if bad else None})
        return False
    per = [r["tok_s"] for r in ok]
    total = sum(r["tok"] for r in ok)
    row = {"n": n, "ok": len(ok), "failed": len(bad),
           "per_stream_mean": round(statistics.mean(per), 2),
           "per_stream_min": round(min(per), 2), "per_stream_max": round(max(per), 2),
           "aggregate": round(total / wall, 1), "wall_s": round(wall, 1),
           "prompt_tokens": ok[0]["prompt_tokens"],
           "tail_distinct_min": min(r["tail_distinct"] for r in ok),
           "gpu_power_w": pw,
           "cpu_phys_max": round(max(busy.get(c, 0) for c in range(12)), 1),
           "cpu_sib_max": round(max(busy.get(c, 0) for c in range(12, 24)), 1),
           "cpu_mean_all": round(sum(busy.values()) / 24, 1)}
    rows.append(row)
    print(f"  n={n:<3} per-stream {row['per_stream_mean']:6.2f} "
          f"(min {row['per_stream_min']:5.2f}/max {row['per_stream_max']:6.2f})  "
          f"AGGREGATE {row['aggregate']:7.1f}  wall {row['wall_s']:5.1f}s  "
          f"tail {row['tail_distinct_min']:.2f}  {pw}W  "
          f"cpu phys {row['cpu_phys_max']:.0f}% sib {row['cpu_sib_max']:.0f}% mean {row['cpu_mean_all']:.0f}%  "
          f"ok={len(ok)}/{n}", flush=True)
    return True


def main():
    print(f"=== {TAG} : port {PORT} : conc_test.py method (70-tok prompt, MAXTOK={MAXTOK}) ===", flush=True)
    warm = {}
    one(0, warm)
    print(f"  warmup: prompt_tokens={warm[0].get('prompt_tokens')} "
          f"tok={warm[0].get('tok')} tok_s={warm[0].get('tok_s', 0):.1f}", flush=True)
    rows = []
    ladder = [n for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96) if n <= MAX_N]
    for n in ladder:
        if not run(n, rows):
            print("  stopping ladder after total failure", flush=True)
            break
        time.sleep(4)
    p = f"{OUTDIR}/vllm_conc2_{TAG}.json"
    json.dump({"tag": TAG, "maxtok": MAXTOK, "method": "conc_test.py", "rows": rows},
              open(p, "w"), indent=2)
    print(f"\nwrote {p}", flush=True)
    best = max((r for r in rows if not r.get("all_failed")), key=lambda r: r["aggregate"], default=None)
    if best:
        print(f"PEAK AGGREGATE {best['aggregate']} tok/s @ n={best['n']}", flush=True)


if __name__ == "__main__":
    main()

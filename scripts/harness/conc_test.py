#!/usr/bin/env python3
"""Concurrency ladder at the production config: 1, 2, 3 simultaneous streams.

Counts content AND reasoning_content -- with --reasoning-format deepseek the reasoning
is split into its own field, and counting only `content` produces bogus low rows
(same trap drafter_ladder.py documents).

Prior behaviour on the OLD build (the owner): n=2 halved per-stream decode so aggregate was
flat, and n=3 collapsed to 3-7 tok/s. The question is whether the new build changes that
-- if aggregate scales, one server can host both agents.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

PORT = 8080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MAXTOK = 800

PROMPT = ("Write a comprehensive, well-commented Python module implementing: an LRU cache with "
          "TTL expiry, a thread-safe bounded queue, a retry decorator with exponential backoff "
          "and jitter, and a binary search tree with insert/delete/traverse. Include type hints, "
          "docstrings, and a demonstration section exercising every component.")


def one(idx, results):
    body = {"model": "x",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK, "temperature": 0, "seed": 42}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        wall = time.time() - t0
        u = d["usage"]
        tok = u["completion_tokens"]
        results[idx] = {"ok": True, "wall": wall, "tok": tok, "tok_s": tok / wall if wall else 0}
    except Exception as e:
        results[idx] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"}


def run(n):
    results = {}
    threads = [threading.Thread(target=one, args=(i, results)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results.values() if r.get("ok")]
    if not ok:
        print(f"  n={n}: ALL FAILED -> {list(results.values())[0].get('err')}")
        return
    per = [r["tok_s"] for r in ok]
    total_tok = sum(r["tok"] for r in ok)
    print(f"  n={n}  per-stream: {statistics.mean(per):6.2f} tok/s "
          f"(min {min(per):5.2f} / max {max(per):5.2f})   "
          f"AGGREGATE: {total_tok / wall:6.2f} tok/s   wall {wall:5.1f}s   ok={len(ok)}/{n}")


print("warmup...")
warm = {}
one(0, warm)
print("concurrency ladder (-np 3, n_max 4, dflash ON):")
for n in (1, 2, 3):
    run(n)
    time.sleep(3)

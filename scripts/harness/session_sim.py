#!/usr/bin/env python3
"""Engine-floor harness: measure a stack the way an agent actually loads it.

WHY THIS EXISTS INSTEAD OF llama-bench
--------------------------------------
The ROCm depth page fault on gfx1201 is reproducible in `llama-bench -d` and has NEVER
reproduced through the real server -- a full live session including a large code generation
ran clean. The fault is confined to the bench's synthetic depth priming, and its mechanism
(quantised KV dequantised into a scratch buffer, failing only on the *repeat* rep) is a
benchmark access pattern, not an inference one. Benchmarks here have crashed at depths a
Hermes conversation handles without complaint.

So depth is reached the way a conversation reaches it: incrementally, turn by turn, with the
prefix cache doing its job. One giant prompt is a different memory-access pattern and a
different test.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No speculative decoding. Not MTP, not DFlash. We are establishing the RAW ENGINE FLOOR, and
spec is a layer added afterward once a stack is chosen -- it differs per stack (MTP is
built in, DFlash needs a separate build) and folding it in makes engines incomparable.
Turn it off at the server, not here.

MEASUREMENT RULES ENCODED HERE
------------------------------
- >=128 generated tokens per turn, because a shorter completion cannot reveal a
  degeneration loop.
- All generated text saved, for coherence.py.
- TTFT timed client-side so llama.cpp and vLLM are measured identically; server-reported
  timings recorded additionally when offered.
- Token counts come from `usage`/`timings`, never from counting SSE chunks.

Works against any OpenAI-compatible endpoint, so llama.cpp and vLLM score on one ruler.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

MODEL_NAME = "m"   # vLLM requires it; llama.cpp ignores it

FILLER = ("The subsystem records each measurement in a rolling buffer and compares it against "
          "the calibration baseline established during the previous maintenance window. ")

TASKS = [
    "Explain how a B-tree index works in a relational database. Continuous prose.",
    "Describe the tradeoffs between write-ahead logging and shadow paging.",
    "Walk through how a query planner chooses between a nested loop and a hash join.",
    "Explain multi-version concurrency control and what anomalies it does not prevent.",
    "Describe how a log-structured merge tree handles compaction, and its write amplification.",
    "Explain the CAP theorem precisely, including what it does not say.",
    "Describe how vector clocks detect causality violations in a distributed store.",
    "Explain how a bloom filter trades space for false positives, with the math.",
]


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(base, text):
    """llama.cpp only. Returns None on vLLM, which has no /tokenize."""
    try:
        with post(base + "/tokenize", {"content": text}, 120) as r:
            return len(json.loads(r.read()).get("tokens") or [])
    except Exception:
        return None


def chat(base, messages, max_tokens, timeout=2400):
    """One turn. Returns metrics plus the generated text."""
    body = {"model": MODEL_NAME, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    chunks = []
    timings = {}
    usage = {}
    try:
        with post(base + "/v1/chat/completions", body, timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("timings"):
                    timings = d["timings"]
                if d.get("usage"):
                    usage = d["usage"]
                for ch in (d.get("choices") or []):
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        chunks.append(piece)
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %s: %s" % (e.code, e.read()[:300].decode("utf-8", "ignore"))}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}

    wall = time.time() - t0
    text = "".join(chunks)
    gen = (timings.get("predicted_n")
           or usage.get("completion_tokens")
           or len(text.split()))
    prm = timings.get("prompt_n") or usage.get("prompt_tokens")
    return {
        "ttft_s": round(ttft, 2) if ttft else None,
        "wall_s": round(wall, 2),
        "prompt_n": prm,
        "gen_n": gen,
        # server-reported where available; otherwise derived from wall clock
        "pp_tok_s": round(timings["prompt_per_second"], 1) if timings.get("prompt_per_second") else None,
        "tg_tok_s": (round(timings["predicted_per_second"], 2) if timings.get("predicted_per_second")
                     else (round(gen / max(wall - (ttft or 0), 1e-6), 2) if gen else None)),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080", help="server base URL")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--seed-tokens", type=int, default=36000,
                    help="opening context, standing in for the agent's system prefix")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="per-turn generation; must stay >=128 for the coherence rule")
    ap.add_argument("--out", default=os.path.expanduser("~/benchmarks/engine_matrix/results_floor"))
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    if args.max_tokens < 128:
        raise SystemExit("--max-tokens must be >=128: shorter completions cannot show a loop")

    os.makedirs(args.out, exist_ok=True)
    base = args.base.rstrip("/")

    # Build the opening context. Sized by /tokenize where available, never by a
    # chars-per-token guess -- that error was made three times in this project.
    salt = "FLOOR-%s-%d. " % (args.label, int(time.time()))
    seed = salt
    i = 0
    while True:
        seed += FILLER + "Index %d deviates by %d units. " % (i, (i * 7) % 13)
        i += 1
        if i % 200 == 0:
            n = tokenize(base, seed)
            if n is None:                       # vLLM: fall back to a conservative estimate
                if len(seed.split()) * 1.35 >= args.seed_tokens:
                    break
            elif n >= args.seed_tokens:
                break
        if i > 200000:
            break
    measured_seed = tokenize(base, seed)
    print("seed context: %s tokens (%s)" %
          (measured_seed if measured_seed else "~%d est" % int(len(seed.split()) * 1.35),
           "/tokenize" if measured_seed else "estimated"))
    print()

    messages = [{"role": "user", "content": seed + "\n\nAcknowledge in one sentence."}]
    rows = []

    hdr = "%-5s %9s %9s %8s %9s %9s  %s" % ("turn", "depth", "ttft_s", "gen", "pp_tok/s", "tg_tok/s", "status")
    print(hdr)
    print("-" * len(hdr))

    for t in range(args.turns):
        want = 160 if t == 0 else args.max_tokens
        r = chat(base, messages, want)

        if "error" in r:
            depth = rows[-1]["depth_after"] if rows else measured_seed
            print("%-5d %9s %9s %8s %9s %9s  FAILED: %s" %
                  (t, depth, "-", "-", "-", "-", r["error"]))
            rows.append({"turn": t, "status": "FAILED", "error": r["error"],
                         "depth_before": depth})
            break

        depth_after = (r["prompt_n"] or 0) + (r["gen_n"] or 0)
        print("%-5d %9s %9s %8s %9s %9s  ok" %
              (t, r["prompt_n"], r["ttft_s"], r["gen_n"],
               r["pp_tok_s"] if r["pp_tok_s"] else "-", r["tg_tok_s"]))

        fn = os.path.join(args.out, "out_%s_turn%02d.txt" % (args.label, t))
        with open(fn, "w", encoding="utf-8") as f:
            f.write(r["text"])

        rows.append({"turn": t, "status": "ok", "depth_before": r["prompt_n"],
                     "depth_after": depth_after, "ttft_s": r["ttft_s"],
                     "gen_n": r["gen_n"], "pp_tok_s": r["pp_tok_s"],
                     "tg_tok_s": r["tg_tok_s"], "wall_s": r["wall_s"]})

        messages.append({"role": "assistant", "content": r["text"]})
        messages.append({"role": "user", "content": TASKS[(t + 1) % len(TASKS)]})

    with open(os.path.join(args.out, "floor_%s.json" % args.label), "w") as f:
        json.dump({"args": vars(args), "seed_tokens": measured_seed, "rows": rows}, f, indent=2)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("tg_tok_s")]
    print()
    if ok:
        deepest = max(r["depth_after"] for r in ok)
        print("turns completed : %d of %d" % (len(ok), args.turns))
        print("deepest reached : %d tokens" % deepest)
        print("decode  first/last : %.2f / %.2f tok/s" % (ok[0]["tg_tok_s"], ok[-1]["tg_tok_s"]))
        pp = [r for r in ok if r.get("pp_tok_s")]
        if pp:
            print("prefill first/last : %.1f / %.1f tok/s" % (pp[0]["pp_tok_s"], pp[-1]["pp_tok_s"]))
    failed = [r for r in rows if r.get("status") == "FAILED"]
    if failed:
        print("FAILED at depth ~%s: %s" % (failed[0]["depth_before"], failed[0]["error"]))
    print()
    print("next: python3 coherence.py \"%s/out_%s_*.txt\"" % (args.out, args.label))


if __name__ == "__main__":
    main()

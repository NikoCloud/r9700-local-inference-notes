#!/usr/bin/env python3
"""Prefill/decode by depth against an OpenAI-compatible endpoint.

Same measurement contract as engine_ab.py so the numbers compare like-for-like with the
llama.cpp production line: forced 512-token generation, temperature 0, one stream,
prefill = prompt_tokens / TTFT, decode = completion_tokens / (total - TTFT).

Usage: vllm_ab.py <tag> [port] [depths...]
"""
import os
import json, random, string, sys, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
DEPTHS = [int(x) for x in sys.argv[3:]] or [2000, 32000, 60000]
GEN = 512
URL = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = "Qwen3.8-PARO"
OUTDIR = os.path.expanduser("~/benchmarks")

_tpw = [1.0]


def words(nw, seed):
    r = random.Random(seed)
    return " ".join(
        "".join(r.choices(string.ascii_lowercase, k=r.randint(3, 9))) for _ in range(nw)
    )


def call(prompt, max_tokens, stream=True):
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    total = time.time() - t0
    return ttft, total, usage, ntok


def calibrate():
    p = words(400, 1)
    _, _, usage, _ = call(p, 1)
    _tpw[0] = usage["prompt_tokens"] / 400.0
    print(f"  calibrated {_tpw[0]:.2f} tok/word", flush=True)


def prompt_for(depth, seed):
    return words(max(8, int(depth / _tpw[0])), seed)


def main():
    print(f"=== {TAG} : port {PORT} : depths {DEPTHS} ===", flush=True)
    calibrate()
    call(prompt_for(3000, 99), 8)  # warm
    out = []
    for d in DEPTHS:
        best = None
        for rep in range(2):
            ttft, total, usage, ntok = call(prompt_for(d, 1000 + rep), GEN)
            if ttft is None or usage is None:
                print(f"  depth {d}: no tokens returned", flush=True)
                continue
            pt = usage["prompt_tokens"]
            ct = usage["completion_tokens"]
            pre = pt / ttft
            dec = ct / (total - ttft) if total > ttft else 0.0
            row = {
                "depth": d,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "ttft_s": round(ttft, 3),
                "total_s": round(total, 3),
                "prefill_tok_s": round(pre, 1),
                "decode_tok_s": round(dec, 2),
            }
            if best is None or row["decode_tok_s"] > best["decode_tok_s"]:
                best = row
        if best:
            out.append(best)
            print(
                f"  depth {best['depth']:>6} | prompt {best['prompt_tokens']:>6} "
                f"| prefill {best['prefill_tok_s']:>7.1f} tok/s "
                f"| decode {best['decode_tok_s']:>6.2f} tok/s",
                flush=True,
            )
    path = f"{OUTDIR}/vllm_ab_{TAG}.json"
    import os

    os.makedirs(os.path.expanduser("~/benchmarks"), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"tag": TAG, "port": PORT, "gen": GEN, "rows": out}, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

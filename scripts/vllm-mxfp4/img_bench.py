#!/usr/bin/env python3
"""Vision wallclock A/B: llama.cpp production vs the MXFP4 vLLM serve.

Same images, same prompt, same request shape (OpenAI vision: base64 data URI).
Reports per-image wallclock, TTFT (image encode + prefill), image/prompt tokens,
decode rate, and a degeneration guard on the output.

Usage: img_bench.py <tag> <port> [image_dir]
"""
import base64, json, os, sys, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "img"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
IMGDIR = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/benchmarks/imgtest")
MODEL = os.environ.get("BENCH_MODEL", "gpt-3.5-turbo")
OUTDIR = os.path.expanduser("~/benchmarks")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MAXTOK = 700

TASK = ("This is one page of a webtoon, read top to bottom. Transcribe every line of dialogue "
        "in order, attributing each to a speaker where you can tell. Then give a two-sentence "
        "summary of what happens on the page.")


def data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    ext = "jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64," + base64.b64encode(b).decode()


def run_one(path):
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri(path)}},
        {"type": "text", "text": TASK},
    ]}]
    body = json.dumps({"model": MODEL, "messages": msg, "max_tokens": MAXTOK,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; text = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                d = json.loads(p)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    delta = ch[0].get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning")
                             or delta.get("reasoning_content") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        text.append(piece)
    except Exception as e:
        return {"image": os.path.basename(path), "error": repr(e)[:200]}
    wall = time.time() - t0
    out = "".join(text)
    tail = out[-300:].split()
    ct = usage["completion_tokens"] if usage else 0
    pt = usage["prompt_tokens"] if usage else 0
    return {
        "image": os.path.basename(path),
        "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": round(ttft, 2) if ttft else None,
        "wall_s": round(wall, 2),
        "decode_tok_s": round(ct / (wall - ttft), 2) if (ttft and wall > ttft) else 0.0,
        "prefill_tok_s": round(pt / ttft, 1) if ttft else 0.0,
        "tail_distinct": round(len(set(tail)) / max(1, len(tail)), 2),
        "chars": len(out),
        "sample": out[:180].replace("\n", " "),
    }


def main():
    imgs = sorted(os.path.join(IMGDIR, f) for f in os.listdir(IMGDIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"=== {TAG} : port {PORT} : {len(imgs)} images ===", flush=True)
    rows = []
    t_all = time.time()
    for p in imgs:
        r = run_one(p)
        rows.append(r)
        if "error" in r:
            print(f"  {r['image']:<16} ERROR {r['error']}", flush=True)
        else:
            print(f"  {r['image']:<16} prompt {r['prompt_tokens']:>6} tok | ttft {r['ttft_s']:>6.2f}s "
                  f"| wall {r['wall_s']:>6.2f}s | prefill {r['prefill_tok_s']:>7.1f} "
                  f"| decode {r['decode_tok_s']:>6.2f} | tail {r['tail_distinct']:.2f}", flush=True)
    total = time.time() - t_all
    ok = [r for r in rows if "error" not in r]
    print(f"\n  TOTAL WALLCLOCK {total:.1f}s for {len(ok)}/{len(imgs)} images "
          f"({total/max(1,len(ok)):.1f}s each)", flush=True)
    if ok:
        print(f"  mean prompt tokens {sum(r['prompt_tokens'] for r in ok)/len(ok):.0f} | "
              f"mean ttft {sum(r['ttft_s'] for r in ok)/len(ok):.2f}s | "
              f"mean decode {sum(r['decode_tok_s'] for r in ok)/len(ok):.2f} tok/s", flush=True)
    path = f"{OUTDIR}/img_bench_{TAG}.json"
    json.dump({"tag": TAG, "port": PORT, "total_wall_s": round(total, 1), "rows": rows},
              open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

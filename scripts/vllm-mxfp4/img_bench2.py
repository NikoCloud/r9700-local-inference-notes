#!/usr/bin/env python3
"""Vision benchmark, non-thinking: PP, TG, concurrency and draft acceptance in one run.

Thinking is disabled via chat_template_kwargs {"enable_thinking": false} -- the template
defaults to enable_thinking=true / reasoning_effort=medium, which is this model's WORST
configuration and burns the output budget on preamble (it truncated the earlier OCR run).

Two passes over the same images:
  sequential  -- one at a time: clean per-image PP (prompt tokens / TTFT, incl. ViT encode)
                 and TG, uncontended.
  concurrent  -- all at once: per-stream and aggregate TG, TTFT spread, fairness.

Draft acceptance is read from the server's own SpecDecoding metrics lines, sampled around
each pass, so it reflects image workloads rather than text.

Usage: img_bench2.py <tag> <port> [image_dir] [container_name]
"""
import base64, json, os, subprocess, sys, threading, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "img2"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
IMGDIR = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/benchmarks/imgtest")
CONTAINER = sys.argv[4] if len(sys.argv) > 4 else "vllmparo"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-heretic-PARO")
OUTDIR = os.path.expanduser("~/benchmarks")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MAXTOK = int(os.environ.get("MAXTOK", "600"))

TASK = ("Describe this webtoon page: the setting, the characters present, what happens panel "
        "by panel, and the overall mood. Be specific and concrete.")


def data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    ext = "jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64," + base64.b64encode(b).decode()


def call(path):
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri(path)}},
        {"type": "text", "text": TASK},
    ]}]
    body = json.dumps({
        "model": MODEL, "messages": msg, "max_tokens": MAXTOK, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; txt = []; think = 0
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
                    rp = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if rp:
                        think += len(rp)
                    piece = delta.get("content") or rp
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        txt.append(piece)
    except Exception as e:
        return {"image": os.path.basename(path), "error": repr(e)[:200]}
    wall = time.time() - t0
    out = "".join(txt)
    tail = out[-300:].split()
    pt = usage["prompt_tokens"] if usage else 0
    ct = usage["completion_tokens"] if usage else 0
    return {
        "image": os.path.basename(path), "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": round(ttft, 2) if ttft else None, "wall_s": round(wall, 2),
        "pp_tok_s": round(pt / ttft, 1) if ttft else 0.0,
        "tg_tok_s": round(ct / (wall - ttft), 2) if (ttft and wall > ttft) else 0.0,
        "reasoning_chars": think,
        "tail_distinct": round(len(set(tail)) / max(1, len(tail)), 2),
        "chars": len(out), "sample": out[:200].replace("\n", " "),
    }


def spec_lines():
    try:
        o = subprocess.run(["docker", "logs", "--tail", "4000", CONTAINER],
                           capture_output=True, text=True, timeout=60)
        return [l for l in (o.stdout + o.stderr).splitlines() if "SpecDecoding metrics" in l]
    except Exception:
        return []


def parse_spec(lines):
    out = []
    for l in lines:
        try:
            seg = l.split("SpecDecoding metrics:", 1)[1]
            d = {}
            for part in seg.split(","):
                if ":" in part:
                    k, v = part.rsplit(":", 1)
                    k = k.strip().lower().replace(" ", "_")
                    v = v.strip().rstrip("%").split()[0]
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            if d:
                out.append(d)
        except Exception:
            pass
    return out


def summarise_spec(before, after):
    new = parse_spec(after[len(before):]) if len(after) > len(before) else []
    if not new:
        return None
    def avg(k):
        vals = [d[k] for d in new if k in d]
        return round(sum(vals) / len(vals), 2) if vals else None
    return {"windows": len(new),
            "mean_acceptance_length": avg("mean_acceptance_length"),
            "avg_draft_acceptance_rate_pct": avg("avg_draft_acceptance_rate")}


def main():
    imgs = sorted(os.path.join(IMGDIR, f) for f in os.listdir(IMGDIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"=== {TAG} : port {PORT} : {len(imgs)} images : NON-THINKING : max_tokens={MAXTOK} ===", flush=True)
    res = {"tag": TAG, "model": MODEL, "images": len(imgs)}

    # ---- sequential
    print("\n-- sequential (uncontended PP / TG) --", flush=True)
    s0 = spec_lines(); t0 = time.time(); seq = []
    for p in imgs:
        r = call(p); seq.append(r)
        if "error" in r:
            print(f"  {r['image']:<16} ERROR {r['error']}", flush=True)
        else:
            print(f"  {r['image']:<16} prompt {r['prompt_tokens']:>5} | ttft {r['ttft_s']:>5.2f}s "
                  f"| PP {r['pp_tok_s']:>7.1f} | TG {r['tg_tok_s']:>6.2f} | out {r['completion_tokens']:>4} "
                  f"| think {r['reasoning_chars']:>4}ch | tail {r['tail_distinct']:.2f}", flush=True)
    seq_wall = time.time() - t0
    res["sequential"] = {"rows": seq, "wall_s": round(seq_wall, 1),
                         "spec": summarise_spec(s0, spec_lines())}
    ok = [r for r in seq if "error" not in r]
    if ok:
        print(f"  wall {seq_wall:.1f}s | mean PP {sum(r['pp_tok_s'] for r in ok)/len(ok):.1f} "
              f"| mean TG {sum(r['tg_tok_s'] for r in ok)/len(ok):.2f} "
              f"| mean think {sum(r['reasoning_chars'] for r in ok)/len(ok):.0f}ch", flush=True)

    time.sleep(5)

    # ---- concurrent
    print(f"\n-- concurrent (all {len(imgs)} at once) --", flush=True)
    s1 = spec_lines(); out = [None] * len(imgs); t1 = time.time()
    def worker(i):
        out[i] = call(imgs[i])
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(len(imgs))]
    for t in ths: t.start()
    for t in ths: t.join()
    conc_wall = time.time() - t1
    cok = [r for r in out if r and "error" not in r]
    for r in cok:
        print(f"  {r['image']:<16} prompt {r['prompt_tokens']:>5} | ttft {r['ttft_s']:>5.2f}s "
              f"| PP {r['pp_tok_s']:>7.1f} | TG {r['tg_tok_s']:>6.2f} | out {r['completion_tokens']:>4}", flush=True)
    agg = sum(r["completion_tokens"] for r in cok) / conc_wall if cok else 0
    res["concurrent"] = {"rows": out, "wall_s": round(conc_wall, 1),
                         "aggregate_tg_tok_s": round(agg, 1),
                         "spec": summarise_spec(s1, spec_lines())}
    if cok:
        ttfts = sorted(r["ttft_s"] for r in cok)
        print(f"  wall {conc_wall:.1f}s | AGGREGATE TG {agg:.1f} tok/s "
              f"| per-stream mean {sum(r['tg_tok_s'] for r in cok)/len(cok):.2f} "
              f"| ttft min {ttfts[0]:.2f}s max {ttfts[-1]:.2f}s", flush=True)
        print(f"  speedup vs sequential: {seq_wall/conc_wall:.2f}x", flush=True)

    for k in ("sequential", "concurrent"):
        sp = res[k].get("spec")
        if sp:
            print(f"  [{k}] draft acceptance: mean_len {sp['mean_acceptance_length']} "
                  f"| rate {sp['avg_draft_acceptance_rate_pct']}% over {sp['windows']} windows", flush=True)

    path = f"{OUTDIR}/img_bench2_{TAG}.json"
    json.dump(res, open(path, "w"), indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()

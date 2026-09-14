import base64, json, os, sys, time, urllib.request, concurrent.futures as cf

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
IMGDIR = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/benchmarks/ai_renders_ch1")
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 60
MAXTOK = int(os.environ.get("MAXTOK", "600"))
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-PARO")
TASK = ("Describe this image: the setting, the characters present (appearance, clothing, "
        "expression), what is happening, and the overall style/mood. Be specific and concrete. 3-5 sentences.")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def data_uri(p):
    with open(p, "rb") as f:
        b = f.read()
    ext = "jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64," + base64.b64encode(b).decode()

def describe(name):
    p = os.path.join(IMGDIR, name)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri(p)}},
            {"type": "text", "text": TASK}]}],
        "max_tokens": MAXTOK, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; txt = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0].get("delta") or {}
                c = ch.get("content")
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    txt.append(c)
        wall = time.time() - t0
        pt = (usage or {}).get("prompt_tokens", 0)
        ct = (usage or {}).get("completion_tokens", 0)
        if not ct and txt:
            ct = max(1, len(" ".join(txt).split()))
        pp = pt / ttft if (ttft and ttft > 0.02) else 0.0
        tg = ct / max(1e-9, wall - (ttft or 0)) if (wall - (ttft or 0)) > 0.02 else 0.0
        return {"name": name, "ok": True, "ttft_s": round(ttft, 2) if ttft else None,
                "wall_s": round(wall, 2), "prompt_tokens": pt, "completion_tokens": ct,
                "pp_tok_s": round(pp, 1), "tg_tok_s": round(tg, 2),
                "desc": " ".join(txt)[:300]}
    except Exception as e:
        return {"name": name, "ok": False, "err": f"{type(e).__name__}: {str(e)[:150]}",
                "wall_s": round(time.time() - t0, 2)}

def main():
    imgs = sorted(f for f in os.listdir(IMGDIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"=== swarm : {len(imgs)} images : {WORKERS} workers : port {PORT} : MAXTOK={MAXTOK} ===", flush=True)
    t0 = time.time()
    results = [None] * len(imgs)
    done = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(describe, n): i for i, n in enumerate(imgs)}
        for f in cf.as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            done += 1
            r = results[i]
            if r.get("ok"):
                print(f"[{done:3d}/{len(imgs)}] {r['name'][:22]:22s} ttft {str(r['ttft_s']):>6s}s wall {r['wall_s']:6.1f}s pt {r['prompt_tokens']:5d} ct {r['completion_tokens']:4d} PP {r['pp_tok_s']:7.1f} TG {r['tg_tok_s']:6.2f}", flush=True)
            else:
                print(f"[{done:3d}/{len(imgs)}] {r['name'][:22]:22s} ERROR {r['err']}", flush=True)
    wall = time.time() - t0
    ok = [r for r in results if r and r.get("ok")]
    tot_ct = sum(r["completion_tokens"] for r in ok)
    agg = tot_ct / wall if wall else 0
    import statistics
    pps = [r["pp_tok_s"] for r in ok if r["pp_tok_s"]]
    tgs = [r["tg_tok_s"] for r in ok if r["tg_tok_s"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"]]
    walls = [r["wall_s"] for r in ok]
    out = {
        "tag": f"swarm_{len(imgs)}img_w{WORKERS}", "n_images": len(imgs), "workers": WORKERS,
        "maxtok": MAXTOK, "wall_s": round(wall, 1), "ok": len(ok), "failed": len(imgs) - len(ok),
        "total_completion_tokens": tot_ct, "aggregate_tok_s": round(agg, 1),
        "per_stream": {"wall_mean": round(statistics.mean(walls), 2), "wall_med": round(statistics.median(walls), 2),
                        "wall_p95": round(sorted(walls)[int(0.95 * len(walls))], 2), "wall_max": max(walls),
                        "ttft_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
                        "ttft_max": max(ttfts) if ttfts else None,
                        "pp_mean": round(statistics.mean(pps), 1) if pps else None,
                        "pp_med": round(statistics.median(pps), 1) if pps else None,
                        "pp_max": round(max(pps), 1) if pps else None,
                        "tg_mean": round(statistics.mean(tgs), 2) if tgs else None,
                        "tg_med": round(statistics.median(tgs), 2) if tgs else None,
                        "tg_min": round(min(tgs), 2) if tgs else None},
        "rows": results,
    }
    p = os.path.expanduser(f"~/benchmarks/swarm_{len(imgs)}img_w{WORKERS}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n=== SUMMARY ===")
    print(f"wall (makespan): {wall:.1f}s   ok {len(ok)}/{len(imgs)}")
    print(f"total completion tokens: {tot_ct}")
    print(f"AGGREGATE: {agg:.1f} tok/s")
    print(f"per-stream: wall mean {out['per_stream']['wall_mean']}s med {out['per_stream']['wall_med']}s p95 {out['per_stream']['wall_p95']}s max {out['per_stream']['wall_max']}s")
    print(f"           ttft mean {out['per_stream']['ttft_mean']}s max {out['per_stream']['ttft_max']}s")
    print(f"           PP mean {out['per_stream']['pp_mean']} med {out['per_stream']['pp_med']} max {out['per_stream']['pp_max']}")
    print(f"           TG mean {out['per_stream']['tg_mean']} med {out['per_stream']['tg_med']} min {out['per_stream']['tg_min']}")
    print(f"wrote {p}")
    print("SWARM_DONE")

main()

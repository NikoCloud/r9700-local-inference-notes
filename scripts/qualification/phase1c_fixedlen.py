#!/usr/bin/env python3
"""Phase 1c -- fixed-length re-measure of the two gates: TG without MTP, and true batching.

Why this pass exists: in phase1b every request stopped at the model's own EOS. On the filler
summaries Turbo stops after ~90-175 tokens while heretic writes 460-720, so per-stream TG was
averaged over very different windows and aggregate tok/s (tokens / wall) mostly measured
verbosity. Here every request is forced to exactly GEN tokens with `ignore_eos`, so all three
models are timed over identical decode windows. PP is not re-measured: it comes from TTFT and
does not depend on output length (phase1b has it at five depths).

Launch = production line minus the two spec flags (phase1b's patch, imported).
Depth TG at -np 2 (production shape); batching ladder at -np 4 (design goal).
Production is restored in `finally`.
"""
import datetime, json, os, random, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

E = os.path.expanduser("~/benchmarks/engine_matrix")
sys.path.insert(0, E)
import bench_sweep as bs          # noqa: E402
import phase1_turbo_ab as p1      # noqa: E402
import phase1b_nospec as b        # noqa: E402  (patches p1.argv to drop the spec flags)

OUT = p1.OUT
RES = os.path.join(OUT, "results_fixedlen.json")
URL = p1.BASE + "/v1/chat/completions"
GEN = 768
DEPTHS = [2500, 42000, 140000]
log = p1.log


def save(d):
    open(RES, "w").write(json.dumps(d, indent=2) + "\n")


def call_fixed(prompt_tokens, gen=GEN, timeout=1800):
    """bench_sweep.call, but generation is forced to exactly `gen` tokens (ignore_eos)."""
    body = {"model": "x",
            "messages": [{"role": "user", "content": bs.build_prompt(prompt_tokens)}],
            "max_tokens": gen, "ignore_eos": True,
            "temperature": 0, "seed": 42,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    ttft = None
    usage = {}
    tail = []
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    ch = json.loads(p)
                except Exception:
                    continue
                if ch.get("usage"):
                    usage = ch["usage"]
                c = ch.get("choices") or []
                if c:
                    dl = c[0].get("delta") or {}
                    txt = dl.get("content") or dl.get("reasoning_content")
                    if txt:
                        if ttft is None:
                            ttft = time.time() - t0
                        tail.append(txt)
                        if len(tail) > 400:
                            del tail[:200]
        wall = time.time() - t0
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        dec = (wall - ttft) if ttft is not None else None
        return {"ok": True, "prompt_tokens": pt, "completion_tokens": ct, "forced_length_honored": ct == gen,
                "ttft_s": round(ttft, 3) if ttft is not None else None, "wall_s": round(wall, 3),
                "prefill_tok_s": round(pt / ttft, 2) if (pt and ttft) else None,
                "decode_tok_s": round((ct - 1) / dec, 2) if (ct and dec and dec > 0) else None,
                "error": None}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                "wall_s": round(time.time() - t0, 3)}


def conc_fixed(n, prompt_tokens=2000):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        rows = list(ex.map(lambda _: call_fixed(prompt_tokens), range(n)))
    wall = time.time() - t0
    ok = [r for r in rows if r["ok"]]
    dec = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    return {"n": n, "gen_tokens": GEN, "ok": "%d/%d" % (len(ok), n),
            "forced_length_honored": all(r.get("forced_length_honored") for r in ok) if ok else None,
            "per_stream_decode_mean": round(statistics.mean(dec), 2) if dec else None,
            "per_stream_decode_min": round(min(dec), 2) if dec else None,
            "aggregate_completion_tok_s": round(sum(r["completion_tokens"] or 0 for r in ok) / wall, 2) if wall else None,
            "wall_s": round(wall, 1), "errors": [r["error"] for r in rows if not r["ok"]][:2]}


def main():
    d = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
         "why": "phase1b outputs stopped at model EOS; Turbo answers ~5x shorter -> TG/aggregate length-confounded. Forced length here.",
         "launch": "production line minus --spec-type draft-mtp --spec-draft-n-max 2",
         "gen_tokens": GEN, "depth_targets": DEPTHS, "models": {}}
    save(d)
    if not b.stop_production(d):
        log("ABORT: VRAM not freed")
        return

    for name, model in p1.MODELS:
        m = {"file": model}
        d["models"][name] = m
        log("================ %s (fixed %d tokens, no spec) ================" % (name, GEN))

        lf = os.path.join(OUT, "%s_fixed_np2.log" % name)
        m["np2_load_s"] = p1.launch(model, 2, lf)
        if m["np2_load_s"]:
            call_fixed(200, 32)
            m["tg_fixed"] = []
            for dt in DEPTHS:
                r = call_fixed(dt)
                r["target"] = dt
                m["tg_fixed"].append(r)
                log("tg  depth %6d: pt=%s TG=%s ct=%s honored=%s ttft=%s %s" % (
                    dt, r.get("prompt_tokens"), r.get("decode_tok_s"), r.get("completion_tokens"),
                    r.get("forced_length_honored"), r.get("ttft_s"), r.get("error") or ""))
                save(d)
        else:
            m["np2_load_s"] = "START_FAIL"
        save(d)

        lf = os.path.join(OUT, "%s_fixed_np4.log" % name)
        m["np4_load_s"] = p1.launch(model, 4, lf)
        if m["np4_load_s"]:
            call_fixed(200, 32)
            m["conc_fixed"] = []
            for n in (1, 2, 3, 4):
                r = conc_fixed(n)
                m["conc_fixed"].append(r)
                log("conc fixed n=%d: per-stream %s (min %s) agg %s ok %s honored=%s" % (
                    n, r["per_stream_decode_mean"], r["per_stream_decode_min"], r["aggregate_completion_tok_s"],
                    r["ok"], r["forced_length_honored"]))
                save(d)
        else:
            m["np4_load_s"] = "START_FAIL"
        save(d)

    log("================ SUMMARY (fixed length) ================")
    for name, m in d["models"].items():
        tg = " | ".join("%s:%s" % (r.get("prompt_tokens"), r.get("decode_tok_s")) for r in m.get("tg_fixed", []))
        cc = " ".join("n%d=%s/%s" % (r["n"], r["per_stream_decode_mean"], r["aggregate_completion_tok_s"]) for r in m.get("conc_fixed", []))
        log("%-18s TG(pt:tok/s) %s" % (name, tg))
        log("%-18s conc(per/agg) %s" % ("", cc))
    save(d)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("CRASH: %r" % e)
        raise
    finally:
        up = p1.restore_production()
        log("production qwen38.service restored: health after %ss" % up)
        try:
            d = json.load(open(RES))
            d["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
            d["production_restored_s"] = up
            d["production_cmdline"] = p1.sh("pgrep -af '[l]lama-server'").stdout.strip()
            save(d)
        except Exception:
            pass
        log("PHASE1C DONE")

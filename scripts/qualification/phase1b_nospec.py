#!/usr/bin/env python3
"""Phase 1b -- no-speculation depth ladder + true-batching ladder (the owner's criteria, 2026-09-10).

Criteria (the owner, verbatim intent):
  PP        = responsiveness; higher wins, usually decides, unless a disqualifier fires
  TG no-MTP = >20 usable, 30+ dream, <18 DISQUALIFIER
  batching  = TG collapse under concurrency is a DISQUALIFIER (current has true batching)
  MTP       = bonus; >50% acceptance (DavidAU's caution); measured in phase1_turbo_ab.py
  depth     = PP and TG degrade with depth on this family -> measure low AND high

Launch = production line MINUS --spec-type/--spec-draft-n-max. Same ruler as Sessions 6/7
(bench_sweep.call: unique prompts, temp 0, seed 42, thinking off, TTFT = prefill).
Plumbing (launch/stop/VRAM/restore) is reused from phase1_turbo_ab.py so both passes
behave identically. Production is restored in `finally`.
"""
import datetime, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

E = os.path.expanduser("~/benchmarks/engine_matrix")
sys.path.insert(0, E)
import bench_sweep as bs          # noqa: E402
import phase1_turbo_ab as p1      # noqa: E402

bs.THINKING_MODE = "off"
_prod_argv = p1.argv


def argv_nospec(model, np_):
    a = _prod_argv(model, np_)
    i = a.index("--spec-type")
    assert a[i:i + 4][2] == "--spec-draft-n-max", a[i:i + 4]
    del a[i:i + 4]
    return a


p1.argv = argv_nospec             # p1.launch looks argv up at call time

OUT = p1.OUT
RES = os.path.join(OUT, "results_nospec.json")
URL = p1.BASE + "/v1/chat/completions"
DEPTHS = [2500, 16000, 42000, 140000, 200000]   # bench_sweep targets; actual ~0.92x
GEN = 1536
log = p1.log


def save(d):
    open(RES, "w").write(json.dumps(d, indent=2) + "\n")


def distinct_ratio(text):
    w = (text or "").split()
    return round(len(set(w)) / len(w), 2) if len(w) >= 20 else None


def slim(r):
    d = {k: r.get(k) for k in ("ok", "prompt_tokens", "ttft_s", "prefill_tok_s",
                               "completion_tokens", "decode_tok_s", "error")}
    d["tail_distinct_word_ratio"] = distinct_ratio(r.get("text_tail"))
    d["text_tail"] = (r.get("text_tail") or "")[-160:]
    return d


def conc(model, n, prompt_tokens, max_tokens):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        rows = list(ex.map(lambda _: bs.call(URL, model, prompt_tokens, max_tokens, timeout=1800), range(n)))
    wall = time.time() - t0
    ok = [r for r in rows if r["ok"]]
    dec = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    tt = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)
    return {"n": n, "prompt_tokens_target": prompt_tokens, "ok": "%d/%d" % (len(ok), n),
            "prompt_tokens_each": [r["prompt_tokens"] for r in ok],
            "per_stream_decode_mean": round(statistics.mean(dec), 2) if dec else None,
            "per_stream_decode_min": round(min(dec), 2) if dec else None,
            "aggregate_completion_tok_s": round(sum(r["completion_tokens"] or 0 for r in ok) / wall, 2) if wall else None,
            "ttft_p50": tt[len(tt) // 2] if tt else None,
            "wall_s": round(wall, 1),
            "errors": [r["error"] for r in rows if not r["ok"]][:2],
            "tail_distinct_word_ratio": distinct_ratio(ok[0]["text_tail"]) if ok else None}


def stop_production(d):
    import urllib.request
    for _ in range(120):
        try:
            slots = json.load(urllib.request.urlopen(p1.PROD_BASE + "/slots", timeout=5))
            if not any(s.get("is_processing") for s in slots):
                break
            log("a slot is processing; waiting for the turn to finish")
        except Exception:
            break
        time.sleep(10)
    live = p1.sh("pgrep -af '[l]lama-server'").stdout.strip()
    rpath = os.path.join(E, "restore", "llamaserver_cmd_production_mtp_n2_%s.txt" % time.strftime("%Y%m%d_%H%M"))
    open(rpath, "w").write(live + "\n")
    p1.sh(["systemctl", "--user", "stop", "qwen38.service"])
    freed = p1.wait_gone(120)
    log("restore line -> %s ; production stopped; VRAM freed=%s (%s GiB)" % (rpath, freed, p1.gpu0_vram_gib()))
    d["restore_line"] = rpath
    return freed


def main():
    d = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
         "launch": "production line minus --spec-type draft-mtp --spec-draft-n-max 2",
         "power": p1.sh("rocm-smi --showmaxpower | grep -i max").stdout.strip().splitlines(),
         "depth_targets": DEPTHS, "gen_tokens": GEN, "models": {}}
    save(d)
    if not stop_production(d):
        log("ABORT: VRAM not freed")
        return

    for name, model in p1.MODELS:
        m = {"file": model}
        d["models"][name] = m
        log("================ %s (no spec) ================" % name)

        # --- depth ladder, single stream, np=2 (production shape) ---
        lf = os.path.join(OUT, "%s_nospec_np2.log" % name)
        m["np2_load_s"] = p1.launch(model, 2, lf)
        if m["np2_load_s"]:
            m["np2_vram_gib"] = p1.gpu0_vram_gib()
            m["np2_sanity"] = p1.sanity()
            bs.call(URL, model, 200, 32)
            m["depth"] = []
            for dt in DEPTHS:
                r = slim(bs.call(URL, model, dt, GEN))
                r["target"] = dt
                m["depth"].append(r)
                log("depth %6d: pt=%s PP=%s ttft=%ss TG=%s ct=%s distinct=%s %s" % (
                    dt, r["prompt_tokens"], r["prefill_tok_s"], r["ttft_s"], r["decode_tok_s"],
                    r["completion_tokens"], r["tail_distinct_word_ratio"], r["error"] or ""))
                save(d)
            m["np2_vram_gib_after_depth"] = p1.gpu0_vram_gib()
        else:
            m["np2_load_s"] = "START_FAIL"
        save(d)

        # --- true batching ladder, np=4 (design goal) ---
        lf = os.path.join(OUT, "%s_nospec_np4.log" % name)
        m["np4_load_s"] = p1.launch(model, 4, lf)
        if m["np4_load_s"]:
            m["np4_vram_gib"] = p1.gpu0_vram_gib()
            bs.call(URL, model, 200, 32)
            m["conc_low"] = []
            for n in (1, 2, 3, 4):
                r = conc(model, n, 2000, 800)
                m["conc_low"].append(r)
                log("conc low  n=%d: per-stream %s (min %s) agg %s ok %s" % (
                    n, r["per_stream_decode_mean"], r["per_stream_decode_min"], r["aggregate_completion_tok_s"], r["ok"]))
                save(d)
            m["conc_depth"] = []
            for n in (1, 2):
                r = conc(model, n, 36000, 800)
                m["conc_depth"].append(r)
                log("conc 33k  n=%d: per-stream %s (min %s) agg %s ttft_p50 %s ok %s" % (
                    n, r["per_stream_decode_mean"], r["per_stream_decode_min"], r["aggregate_completion_tok_s"],
                    r["ttft_p50"], r["ok"]))
                save(d)
            m["np4_vram_gib_after"] = p1.gpu0_vram_gib()
        else:
            m["np4_load_s"] = "START_FAIL"
        save(d)

    log("================ SUMMARY (no spec) ================")
    for name, m in d["models"].items():
        dp = " | ".join("%s:%s/%s" % (r.get("prompt_tokens"), r.get("prefill_tok_s"), r.get("decode_tok_s"))
                        for r in m.get("depth", []))
        cl = " ".join("n%d=%s/%s" % (r["n"], r["per_stream_decode_mean"], r["aggregate_completion_tok_s"])
                      for r in m.get("conc_low", []))
        log("%-18s depth(pt:PP/TG) %s" % (name, dp))
        log("%-18s conc low(per/agg) %s" % ("", cl))
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
        log("PHASE1B DONE")

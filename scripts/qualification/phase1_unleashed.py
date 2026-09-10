#!/usr/bin/env python3
"""Add outsourc-e Qwen3.8-27B Unleashed UD-Q4_K_M (stock proxy) to the Phase 1 battery.

the owner, 2026-09-10: after Turbo Q4_K_S won qualification, run the Unleashed build (last measured
before the llama.cpp build bump) through the SAME three passes and add it to the chart.
It is MTP-capable (qwen35.nextn_predict_layers = 1 in its header).

This runs in a later thermal window than the other three, so it starts with a drift check on
the live production server (heretic, MTP n2, -np 2 -- the exact phase1 config), with no swap:
  * MTP predictable 2048-tok gen   (phase1 heretic: 68.01 tok/s)
  * prefill @42k, thinking off     (phase1 heretic: 854.56 tok/s)

Then, reusing the existing scripts unchanged apart from model list and output paths:
  pass A  phase1_turbo_ab.main   (MTP on)        -> results_unleashed_mtp.json
  pass B  phase1b_nospec.main    (no spec)       -> results_unleashed_nospec.json
  pass C  phase1c_fixedlen.main  (forced 768)    -> results_unleashed_fixedlen.json
Production is restored in `finally`.
"""
import datetime, json, os, sys, time, urllib.request

E = os.path.expanduser("~/benchmarks/engine_matrix")
sys.path.insert(0, E)
import phase1_turbo_ab as p1          # noqa: E402
PROD_ARGV = p1.argv                   # capture BEFORE phase1b's import-time patch
import bench_sweep as bs              # noqa: E402
import phase1b_nospec as b            # noqa: E402  (patches p1.argv -> no spec)
import phase1c_fixedlen as c          # noqa: E402

NAME = "unleashed-ud-q4_k_m"
MODEL = "/mnt/models/llm/qwen3.8-27b-unleashed-outsourc-e/qwen3.8-27b-unleashed-outsourc-e-ud-q4_k_m.gguf"
OUT = p1.OUT
DRIFT = os.path.join(OUT, "results_unleashed_drift.json")
log = p1.log


def wait_idle():
    for _ in range(120):
        try:
            slots = json.load(urllib.request.urlopen(p1.BASE + "/slots", timeout=5))
            if not any(s.get("is_processing") for s in slots):
                return True
            log("a slot is processing; waiting for the turn to finish")
        except Exception:
            return False
        time.sleep(10)
    return False


def drift_check():
    log("=== drift check on live production (heretic, MTP n2, -np 2) ===")
    wait_idle()
    heretic = "/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf"
    p1.gen("Say hello in one sentence.")
    g = p1.gen(p1.PREDICTABLE)
    bs.THINKING_MODE = "off"
    pf = bs.call(p1.BASE + "/v1/chat/completions", heretic, 42000, 1536)
    d = {"at": datetime.datetime.now().isoformat(timespec="seconds"),
         "mtp_predictable_tok_s": g.get("tok_s"), "mtp_predictable_ref": 68.01,
         "prefill42k_tok_s": pf.get("prefill_tok_s"), "prefill42k_ref": 854.56,
         "prefill42k_prompt_tokens": pf.get("prompt_tokens")}
    for k, ref in (("mtp_predictable_tok_s", 68.01), ("prefill42k_tok_s", 854.56)):
        if isinstance(d[k], (int, float)):
            d[k.replace("_tok_s", "_delta_pct")] = round((d[k] / ref - 1) * 100, 2)
    open(DRIFT, "w").write(json.dumps(d, indent=2) + "\n")
    log("drift: MTP pred %s (ref 68.01, %s%%) | prefill42k %s (ref 854.56, %s%%)" % (
        d["mtp_predictable_tok_s"], d.get("mtp_predictable_delta_pct"),
        d["prefill42k_tok_s"], d.get("prefill42k_delta_pct")))


def main():
    p1.MODELS[:] = [(NAME, MODEL)]

    drift_check()

    # pass A: MTP on (phase1 stops production and saves the restore line itself)
    p1.argv = PROD_ARGV
    p1.RESULTS = os.path.join(OUT, "results_unleashed_mtp.json")
    log("=== pass A: MTP (phase1_turbo_ab) ===")
    p1.main()
    p1.stop_test_server()

    # passes B + C: production already stopped; don't re-save an (empty) restore line
    b.stop_production = lambda d: p1.wait_gone(120)
    p1.argv = b.argv_nospec

    b.RES = os.path.join(OUT, "results_unleashed_nospec.json")
    log("=== pass B: no spec depth + batching (phase1b_nospec) ===")
    b.main()
    p1.stop_test_server()

    c.RES = os.path.join(OUT, "results_unleashed_fixedlen.json")
    log("=== pass C: forced-length gates (phase1c_fixedlen) ===")
    c.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("CRASH: %r" % e)
        raise
    finally:
        up = p1.restore_production()
        log("production qwen38.service restored: health after %ss" % up)
        log("UNLEASHED DONE")

#!/usr/bin/env python3
"""Core-19 in-run speed: heretic vs Turbo from the per-arm llama-server logs.

Per request (llama.cpp task id): prefill tokens/ms, decode tokens/ms, MTP accepted/generated,
and depth = n_tokens at release. Aggregated by depth bin (token-weighted, not mean of rates),
plus a same-task comparison for the task(s) Turbo has finished or is running.
"""
import datetime, glob, json, os, re, sys

C19 = os.path.expanduser("~/benchmarks/core19")
JOBS = os.path.expanduser("~/projects/terminal-bench-mini/jobs")
ARMS = {"heretic": "full-heretic", "turbo": "full-turbo"}
BINS = [(0, 10_000), (10_000, 20_000), (20_000, 40_000), (40_000, 60_000), (60_000, 90_000), (90_000, 10**9)]

TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.\d+ ")  # minutes.seconds.ms.us since server start
TASK = re.compile(r"\| task (\d+) \|")
PE = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
EV = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens")
DA = re.compile(r"draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated")
REL = re.compile(r"stop processing: n_tokens = (\d+)")


def server_start(arm_dir):
    """Wall-clock launch time: 'server healthy ... after ~Ns' line in campaign.log minus N."""
    pat = re.compile(r"^\[(\S+ \S+)\] %s: server healthy .* after ~(\d+)s" % re.escape(arm_dir))
    t = None
    for line in open(os.path.join(C19, "campaign.log")):
        m = pat.match(line)
        if m:
            t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") - datetime.timedelta(seconds=int(m.group(2)))
    return t


def parse(arm_dir):
    start = server_start(arm_dir)
    reqs, cur = [], {}
    for line in open(os.path.join(C19, arm_dir, "llama-server.log"), errors="ignore"):
        mt, mk = TS.match(line), TASK.search(line)
        if not (mt and mk):
            continue
        tid = int(mk.group(1))
        r = cur.setdefault(tid, {"task": tid})
        secs = int(mt.group(1)) * 60 + int(mt.group(2)) + int(mt.group(3)) / 1000
        if (m := PE.search(line)):
            r["pp_ms"], r["pp_tok"] = float(m.group(1)), int(m.group(2))
        elif (m := EV.search(line)):
            r["tg_ms"], r["tg_tok"] = float(m.group(1)), int(m.group(2))
        elif (m := DA.search(line)):
            r["acc"], r["gen"] = int(m.group(1)), int(m.group(2))
        elif (m := REL.search(line)):
            r["depth"] = int(m.group(1))
            r["end"] = start + datetime.timedelta(seconds=secs) if start else None
            if "tg_ms" in r:
                reqs.append(r)
            cur.pop(tid, None)
    return reqs


def agg(rows):
    pp_ms = sum(r.get("pp_ms", 0) for r in rows if r.get("pp_tok", 0) >= 256)
    pp_tok = sum(r.get("pp_tok", 0) for r in rows if r.get("pp_tok", 0) >= 256)
    tg_ms, tg_tok = sum(r["tg_ms"] for r in rows), sum(r["tg_tok"] for r in rows)
    acc, gen = sum(r.get("acc", 0) for r in rows), sum(r.get("gen", 0) for r in rows)
    return {"n": len(rows),
            "pp": round(pp_tok / pp_ms * 1000, 1) if pp_ms else None,
            "tg": round(tg_tok / tg_ms * 1000, 1) if tg_ms else None,
            "acc": round(acc / gen, 3) if gen else None,
            "gen_tok": tg_tok}


def trials(model_key):
    """(task_name, start, end) for every trial of this model's full jobs (attempt 1 and 2)."""
    out = []
    for j in glob.glob(os.path.join(JOBS, "*core19-full*%s*" % model_key)):
        for rp in glob.glob(os.path.join(j, "*", "result.json")):
            d = json.load(open(rp))
            ae = d.get("agent_execution") or {}
            s, e = ae.get("started_at") or d.get("started_at"), ae.get("finished_at") or d.get("finished_at")
            if s and e:
                f = lambda x: datetime.datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
                out.append((os.path.basename(os.path.dirname(rp)).split("__")[0], f(s), f(e),
                            ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")))
    return out


def main():
    data = {k: parse(v) for k, v in ARMS.items()}
    print("requests parsed:", {k: len(v) for k, v in data.items()})
    print("\n=== by context depth (token-weighted; PP only for prompts >= 256 new tokens) ===")
    print("%-12s | %-34s | %-34s" % ("depth", "heretic  n / PP / TG / MTP acc", "turbo  n / PP / TG / MTP acc"))
    for lo, hi in BINS:
        cells = []
        for k in ARMS:
            a = agg([r for r in data[k] if lo <= r.get("depth", 0) < hi])
            cells.append("%4d / %6s / %5s / %5s" % (a["n"], a["pp"], a["tg"], a["acc"]) if a["n"] else "-")
        print("%-12s | %-34s | %-34s" % ("%dk-%s" % (lo // 1000, "%dk" % (hi // 1000) if hi < 10**9 else "+"), *cells))
    print("\n=== whole arm so far ===")
    for k in ARMS:
        a = agg(data[k])
        mx = max((r.get("depth", 0) for r in data[k]), default=0)
        print("%-8s %s  max depth %d" % (k, a, mx))

    print("\n=== same task, both arms (requests matched to trial agent window) ===")
    tt = {"heretic": trials("heretic-trohrbaugh"), "turbo": trials("turbo-fable")}
    turbo_names = {t[0] for t in tt["turbo"]}
    last_turbo = max((r["end"] for r in data["turbo"] if r.get("end")), default=None)
    for k in ARMS:
        for name, s, e, rew in sorted(tt[k], key=lambda x: x[1]):
            if k == "heretic" and name not in turbo_names and name != "break-filter-js-from-html":
                continue
            rows = [r for r in data[k] if r.get("end") and s <= r["end"] <= e]
            a = agg(rows)
            print("%-8s %-28s reward=%s  %s min  %s  depth max %s" % (
                k, name, rew, round((e - s).total_seconds() / 60, 1), a, max((r["depth"] for r in rows), default=None)))
    if last_turbo:
        running = [r for r in data["turbo"] if r.get("end") and not any(s <= r["end"] <= e for _, s, e, _ in tt["turbo"])]
        print("turbo    (current, unscored task) %s  depth max %s" % (agg(running), max((r["depth"] for r in running), default=None)))


if __name__ == "__main__":
    main()

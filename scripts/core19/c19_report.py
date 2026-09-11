#!/usr/bin/env python3
"""Core-19 A/B report: heretic vs Turbo (box) + MiniCPM5 (laptop, optional input) from Harbor job directories.

Per task, per arm: attempt-1 and attempt-2 outcome (reward / exception), agent minutes, steps, input/cached/output
tokens. Arm metrics: pass@1 (raw, and excluding harness/infra errors where the agent never ran), pass@2,
agent minutes and output tokens in total and per solved task (attempt 1), medians.

usage: c19_report.py <out_dir> [--laptop-json <file>]   (run on the box)
Writes <out_dir>/core19_report.json and <out_dir>/core19_report.md
"""
import datetime, glob, json, os, statistics, sys

JOBS = os.path.expanduser("~/projects/terminal-bench-mini/jobs")
ARMS = {
    "heretic": {"match": "*core19-full*heretic-trohrbaugh*", "smoke": "*core19-smoke*heretic-trohrbaugh*"},
    "turbo": {"match": "*core19-full*turbo-fable*"},
}
INFRA = {"RuntimeError"}  # harness/container errors before agent execution; verified per trial below


def ts(x):
    return datetime.datetime.fromisoformat(x.replace("Z", "+00:00")) if x else None


def trial(rp):
    d = json.load(open(rp)); T = os.path.dirname(rp)
    ae = d.get("agent_execution") or {}
    ex = (d.get("exception_info") or {})
    ar = d.get("agent_result") or {}
    try:
        steps = len(json.load(open(os.path.join(T, "agent", "trajectory.json"))).get("steps") or [])
    except Exception:
        steps = None
    s, e = ts(ae.get("started_at")), ts(ae.get("finished_at"))
    exc = ex.get("exception_type")
    infra = bool(exc in INFRA and not ae.get("started_at"))
    return {"trial": os.path.basename(T), "reward": ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception": exc, "exception_message": str(ex.get("exception_message") or "")[:200], "infra_error": infra,
            "agent_minutes": round((e - s).total_seconds() / 60, 2) if s and e else None, "steps": steps,
            "input_tokens": ar.get("n_input_tokens"), "cached_tokens": ar.get("n_cache_tokens"), "output_tokens": ar.get("n_output_tokens"),
            "started": ae.get("started_at")}


def newest_job(pattern, exclude_attempt2=True):
    js = [j for j in glob.glob(os.path.join(JOBS, pattern)) if os.path.isdir(j)]
    if exclude_attempt2:
        js = [j for j in js if not j.endswith("-attempt2")]
    return max(js, key=os.path.getmtime) if js else None


def load_arm(name, cfg):
    j1 = newest_job(cfg["match"])
    arm = {"job": os.path.basename(j1) if j1 else None, "tasks": {}}
    if not j1:
        return arm
    j2 = j1 + "-attempt2" if os.path.isdir(j1 + "-attempt2") else None
    arm["job_attempt2"] = os.path.basename(j2) if j2 else None
    for rp in glob.glob(os.path.join(j1, "*", "result.json")):
        t = os.path.basename(os.path.dirname(rp)).split("__")[0]
        arm["tasks"].setdefault(t, {})["a1"] = trial(rp)
    if j2:
        for rp in glob.glob(os.path.join(j2, "*", "result.json")):
            t = os.path.basename(os.path.dirname(rp)).split("__")[0]
            arm["tasks"].setdefault(t, {})["a2"] = trial(rp)
    if cfg.get("smoke"):  # tasks the runner reused from the matching smoke job
        sj = newest_job(cfg["smoke"])
        if sj:
            for rp in glob.glob(os.path.join(sj, "*", "result.json")):
                t = os.path.basename(os.path.dirname(rp)).split("__")[0]
                if t not in arm["tasks"] or "a1" not in arm["tasks"][t]:
                    arm["tasks"].setdefault(t, {})["a1"] = dict(trial(rp), reused_from_smoke=os.path.basename(sj))
    return arm


def summarize(arm, n_tasks=19):
    a1 = {t: v["a1"] for t, v in arm["tasks"].items() if "a1" in v}
    solved1 = [t for t, x in a1.items() if x["reward"] == 1.0]
    infra = [t for t, x in a1.items() if x["infra_error"]]
    solved2 = [t for t, v in arm["tasks"].items() if (v.get("a1", {}).get("reward") == 1.0 or v.get("a2", {}).get("reward") == 1.0)]
    mins = [x["agent_minutes"] for x in a1.values() if x["agent_minutes"] is not None]
    outs = [x["output_tokens"] for x in a1.values() if x["output_tokens"] is not None]
    s_mins = [a1[t]["agent_minutes"] for t in solved1 if a1[t]["agent_minutes"] is not None]
    s_outs = [a1[t]["output_tokens"] for t in solved1 if a1[t]["output_tokens"] is not None]
    return {"scored_a1": len(a1), "pass1": len(solved1), "pass1_of": n_tasks, "infra_errors_a1": infra,
            "pass1_excluding_infra": f"{len(solved1)}/{n_tasks - len(infra)}", "pass2": len(solved2),
            "agent_minutes_total_a1": round(sum(mins), 1), "output_tokens_total_a1": sum(outs),
            "minutes_per_solved_a1": round(sum(mins) / len(solved1), 1) if solved1 else None,
            "output_tokens_per_solved_a1": round(sum(outs) / len(solved1)) if solved1 else None,
            "median_minutes_solved_a1": round(statistics.median(s_mins), 1) if s_mins else None,
            "median_output_tokens_solved_a1": round(statistics.median(s_outs)) if s_outs else None}


def cell(x):
    if not x:
        return "–"
    if x.get("infra_error"):
        return "INFRA"
    r = "PASS" if x["reward"] == 1.0 else ("TIMEOUT" if x.get("exception") == "AgentTimeoutError" else "fail")
    out = f"{x['output_tokens'] / 1000:.1f}k" if x.get("output_tokens") else "?"
    return f"{r} · {x['agent_minutes']} min · {out}" + (" (smoke)" if x.get("reused_from_smoke") else "")


def main():
    out_dir = os.path.expanduser(sys.argv[1]); os.makedirs(out_dir, exist_ok=True)
    rep = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "arms": {}}
    for name, cfg in ARMS.items():
        arm = load_arm(name, cfg); arm["summary"] = summarize(arm); rep["arms"][name] = arm
    if "--laptop-json" in sys.argv:
        lap = json.load(open(sys.argv[sys.argv.index("--laptop-json") + 1]))
        lap["summary"] = summarize(lap); rep["arms"]["minicpm5_laptop"] = lap
    json.dump(rep, open(os.path.join(out_dir, "core19_report.json"), "w"), indent=2)
    names = list(rep["arms"])
    tasks = sorted({t for a in rep["arms"].values() for t in a["tasks"]})
    md = ["| task | " + " | ".join(f"{n} a1 | {n} a2" for n in names) + " |", "|---|" + "---|---|" * len(names)]
    for t in tasks:
        md.append(f"| {t} | " + " | ".join(f"{cell(rep['arms'][n]['tasks'].get(t, {}).get('a1'))} | {cell(rep['arms'][n]['tasks'].get(t, {}).get('a2'))}" for n in names) + " |")
    md += ["", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in rep["arms"][names[0]]["summary"]:
        md.append(f"| {k} | " + " | ".join(str(rep["arms"][n]["summary"].get(k)) for n in names) + " |")
    open(os.path.join(out_dir, "core19_report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

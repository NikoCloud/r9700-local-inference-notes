#!/usr/bin/env python3
"""Extract reasoning traces from the tee'd sanity / loop-A/B terminal logs into structured files.

Output (per source): <out>/<model>/<label>.md  (prompt, stats, checks, reasoning, answer)
                      <out>/traces.jsonl       (one record per model call)
The logs contain ANSI colour codes and harness banners; markers used:
  --- <label>: request sent ... ---   [reasoning]   [answer]
  --- <label> stats: {...} ---   /   --- <label>: LOOP DETECTED ...   /   === resumed at task 2
"""
import ast, json, os, re, sys

sys.path.insert(0, os.path.expanduser("~/benchmarks/core19/manual-turbo"))
import sanity_live as S  # prompts

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SENT = re.compile(r"^--- (.+?): request sent.*---\s*$", re.M)


def model_for(source, label):
    if "loop_ab" in source:
        return {"A": "turbo_kv-q8_262k", "B": "turbo_kv-f16_131k", "C": "heretic_kv-q8_262k"}[label[0]]
    return "heretic_kv-q8_262k" if "manual-heretic" in source else "turbo_kv-q8_262k"


def prompt_for(label):
    if "task1" in label or "_run" in label or " run " in label:
        return S.POEM_PROMPT
    if "task2" in label:
        return S.BUG_PROMPT
    if "task3" in label:
        return S.SIM_PROMPT
    return None


def parse(source):
    text = ANSI.sub("", open(source, errors="ignore").read())
    marks = list(SENT.finditer(text))
    out = []
    for i, m in enumerate(marks):
        label = m.group(1).strip()
        seg = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        # where this call's stream ends
        ends = [seg.find(s) for s in (f"--- {label} stats:", f"--- {label}: LOOP DETECTED", f"--- {label}: finished, grading",
                                      "=== resumed at task 2", "\n  CONDITION ", "\n  TASK ")]
        ends = [e for e in ends if e >= 0]
        stream = seg[: min(ends)] if ends else seg
        aborted = not any(seg.find(s) >= 0 for s in (f"--- {label} stats:",))
        # reasoning / answer segments (a stream can switch more than once)
        parts = re.split(r"\n?\[(reasoning|answer)\]\n", stream)
        reasoning, answer = [], []
        for kind, body in zip(parts[1::2], parts[2::2]):
            (reasoning if kind == "reasoning" else answer).append(body)
        stats = None
        sm = re.search(r"--- %s stats: (\{.*?\}) ---" % re.escape(label), seg)
        if sm:
            try:
                stats = ast.literal_eval(sm.group(1))
            except Exception:
                stats = {"raw": sm.group(1)[:500]}
        loop = re.search(r"--- %s: LOOP DETECTED -> cancelled: (\{.*?\}) ---" % re.escape(label), seg)
        checks = re.findall(r"^\s+(PASS|FAIL)\s+(.+?)\s{2,}(.*)$", seg, re.M)
        out.append({"source": source, "model": model_for(source, label), "label": label,
                    "prompt": prompt_for(label), "aborted_no_stats": aborted,
                    "loop": ast.literal_eval(loop.group(1)) if loop else None, "stats": stats,
                    "checks": [{"ok": ok == "PASS", "check": c.strip(), "detail": d.strip()} for ok, c, d in checks],
                    "reasoning": "".join(reasoning).strip(), "answer": "".join(answer).strip()})
    return out


def main():
    out_dir = os.path.expanduser(sys.argv[1])
    sources = [os.path.expanduser(s) for s in sys.argv[2:]]
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    for src in sources:
        if os.path.exists(src):
            recs += parse(src)
    with open(os.path.join(out_dir, "traces.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    for r in recs:
        d = os.path.join(out_dir, r["model"]); os.makedirs(d, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", r["label"]).strip("_")
        with open(os.path.join(d, name + ".md"), "w") as f:
            f.write(f"# {r['label']}  ({r['model']})\n\nsource log: `{r['source']}`\n\n")
            f.write("## Stats\n\n```\n" + json.dumps(r["stats"], indent=2) + "\n```\n\n")
            if r["loop"]:
                f.write("**Loop detected, cancelled:** `" + json.dumps(r["loop"]) + "`\n\n")
            if r["aborted_no_stats"] and not r["loop"]:
                f.write("**Stream ended without stats (operator abort).**\n\n")
            if r["checks"]:
                f.write("## Checks\n\n" + "\n".join(f"- {'PASS' if c['ok'] else 'FAIL'} {c['check']} {c['detail']}" for c in r["checks"]) + "\n\n")
            f.write("## Prompt\n\n```\n" + (r["prompt"] or "?") + "\n```\n\n")
            f.write("## Reasoning\n\n```text\n" + r["reasoning"] + "\n```\n\n## Answer\n\n" + r["answer"] + "\n")
    print(f"{len(recs)} traces -> {out_dir}")
    for r in recs:
        print(f"  {r['model']:<20} {r['label']:<28} reasoning {len(r['reasoning']):>7} chars  answer {len(r['answer']):>6} chars"
              f"  {'LOOP' if r['loop'] else ''}{'ABORTED' if r['aborted_no_stats'] and not r['loop'] else ''}"
              f"  stats_chars={(r['stats'] or {}).get('reasoning_chars')}")


if __name__ == "__main__":
    main()

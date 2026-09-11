#!/usr/bin/env python3
"""Reasoning-trace quirk detectors, pass 2 (heuristics; definitions still to be ratified).

#1 Reversal away from correct
   poem : grade every complete draft in the reasoning with the final-answer checker; flag drafts that
          outscore the final answer or pass a check the final answer failed.
   bugs : where each planted bug is first DIAGNOSED (mention outside copied source code), whether it
          reaches the answer, dismissal phrases after a mention, and how much reasoning follows the
          point where all five were diagnosed.
#2 Self-verification accuracy
   2a per-line word counts the model states, matched to the line they describe, vs the real count.
      Formats: `<line text> (n)`, `<line text> -> n words`, and positional enumerations
      `lineK: w(1) w2 w3 ... =n` matched to line K of the stanza just above.
      Diffs beyond +-4 are treated as "not a per-line count" (running totals), reported separately.
   2b stated TOTALS ("total ... N", "= N words") paired with the nearest preceding complete draft;
      restated rules ("140-160", "between 140 and 160") and running/sub totals are excluded.

usage: analyze_traces.py <traces.jsonl> [...]  -> report on stdout, quirks_report.json next to the first file
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.expanduser("~/benchmarks/core19/manual-turbo"))
import sanity_live as S  # noqa: E402

S.check = lambda name, ok, detail="": {"check": name, "ok": bool(ok), "detail": str(detail)}
S.print = lambda *a, **k: None
FIRSTS = ["Wind", "Tick", "Rust", "Chime", "Stop"]
TITLE = "# The Clockmaker's Last Apprentice"
WORD = re.compile(r"[A-Za-z']+")


def words(s):
    return WORD.findall(s)


def is_poem(r):
    return "task1" in r["label"] or " run " in r["label"]


def is_bugs(r):
    return "task2" in r["label"]


# ------------------------------------------------------------------------------------------------ stanza/draft extraction
HEADER = re.compile(r"^\s*(\*\*)?(stanza\s*\d|draft|version|attempt|let me|final|poem|title|#)", re.I)
SEPARATOR = re.compile(r"^\s*(word\s*count|count|total|line\s*\d+\s*[:=]|grand|running|check|[A-Za-z']+\(\d+\))", re.I)
TRAIL_ANNOT = re.compile(r"\s*(?:\((?:[A-Za-z' ]{1,20}|\d{1,3}(?: words?)?)\)|->\s*\d+ words?|[-–=]\s*\d{1,2})\s*$")


def clean_line(line):
    prev = None
    while prev != line:
        prev, line = line, TRAIL_ANNOT.sub("", line).strip()
    return line.strip("\"'`* ")


def stanza_groups(text):
    """(start, end, [4 cleaned lines]) for 4-line groups; blank lines, headers and count/annotation lines end a group."""
    pos = 0; cur = []; start = None; out = []
    def flush(end):
        if len(cur) == 4:
            out.append((start, end, list(cur)))
    for raw in text.splitlines(True):
        line = raw.strip()
        core = clean_line(line)  # trailing "(8)" / "(rhyme)" annotations are normal draft lines, not recount views
        recount_view = bool(re.match(r"^\d+[\s.)]", core) or re.search(r"\(\d+\)|[A-Za-z,.;:!?]\d+\b|\[rhyme", core))
        if not line or HEADER.match(line) or SEPARATOR.match(line) or (line.endswith(":") and len(line) < 60) or "=" in line or recount_view:
            flush(pos); cur = []; start = None
        else:
            if start is None:
                start = pos
            cur.append(clean_line(line))
            if len(cur) > 4:
                cur = cur[1:]; start = pos  # sliding: keep the last 4 prose lines (annotation-free stanza above)
        pos += len(raw)
    flush(pos)
    return out


def complete_drafts(text, groups=None):
    groups = [g for g in (groups or stanza_groups(text)) if words(g[2][0]) and words(g[2][0])[0] in FIRSTS]
    drafts = []; i = 0
    while i < len(groups):
        if words(groups[i][2][0])[0] != "Wind":
            i += 1; continue
        run = [groups[i]]; j = i + 1
        while j < len(groups) and len(run) < 5:
            if words(groups[j][2][0])[0] == FIRSTS[len(run)] and groups[j][0] - run[-1][1] < 2500:
                run.append(groups[j]); j += 1
            elif words(groups[j][2][0])[0] == FIRSTS[len(run) - 1]:  # a revised version of the same stanza: take the newest
                run[-1] = groups[j]; j += 1
            else:
                break
        if len(run) == 5:
            body = "\n\n".join("\n".join(g[2]) for g in run)
            drafts.append({"start": run[0][0], "end": run[-1][1], "poem": TITLE + "\n\n" + body})
        i = j if len(run) > 1 else i + 1
    return drafts


def grade(poem):
    rs = S.grade_poem(poem)
    return sum(r["ok"] for r in rs), [r["check"] for r in rs if not r["ok"]]


def body_words(poem):
    return len(words("\n".join(poem.strip().splitlines()[1:])))


# ------------------------------------------------------------------------------------------------ #2a per-line counts
ANNOT_PAREN = re.compile(r"^\s*(?:Line \d+:\s*|\d+[.)]\s*)?[\"'`]?([A-Z][^\n]{8,}?)[\"'`]?\s*[-–:]*\s*\((\d{1,2})(?: words?)?\)\s*$", re.M)
ANNOT_ARROW = re.compile(r"^\s*[\"'`]?([A-Z][^\n]{8,}?)[\"'`]?\s*(?:->|=|:|—|–)\s*(\d{1,2}) words?\b", re.M)
ENUM = re.compile(r"(?:^|\n)\s*(?:word count\s*)?line\s*(\d)\s*:\s*([^\n]*?)=\s*(\d{1,2})\b", re.I)


LABEL = re.compile(r"^\s*(?:word\s*count|count|words?|line\s*\d+|stanza\s*\d+|total)(?:\s+line\s*\d+)?\s*[:=]?\s*", re.I)
ENUM_NUM = re.compile(r"\(\d+\)|(?<=[A-Za-z.,;:!?'\"])\d+")


def strip_labels(line):
    prev = None
    while prev != line:
        prev, line = line, LABEL.sub("", line, count=1)
    return ENUM_NUM.sub("", line).strip()


def per_line_counts(text):
    rows = []  # (format, stated, real, snippet)
    for rx, fmt in ((ANNOT_PAREN, "text (n)"), (ANNOT_ARROW, "text -> n")):
        for m in rx.finditer(text):
            line = strip_labels(clean_line(m.group(1)))
            if "rhyme" in line.lower() or len(words(line)) < 3 or len(words(line)) > 14:  # >14 = stanza-level enumeration
                continue
            rows.append((fmt, int(m.group(2)), len(words(line)), line[:60]))
    groups = stanza_groups(text)
    for m in ENUM.finditer(text):
        k = int(m.group(1))
        above = [g for g in groups if g[1] <= m.start()]
        if not above or not 1 <= k <= 4:
            continue
        line = above[-1][2][k - 1]
        enumerated = [w for w in re.split(r"\s+", re.sub(r"\(\d+\)|\d+", " ", m.group(2))) if WORD.search(w)]
        rows.append(("enum lineK: w1 w2 =n", int(m.group(3)), len(words(line)), f"{line[:40]} | enum {len(enumerated)} tokens"))
    per_line = [r for r in rows if abs(r[1] - r[2]) <= 4]
    return per_line, len(rows) - len(per_line)


# ------------------------------------------------------------------------------------------------ #2b totals
TOTAL = re.compile(r"(?:grand\s+total|total(?:\s+body)?(?:\s+words?)?|word\s*count)[^0-9\n]{0,12}(\d{3})\b|=\s*(\d{3})\s*words", re.I)
RULE = re.compile(r"140\s*(?:-|–|to|and)\s*160|between 140|within 140|140\s*[-–]\s*160|range", re.I)
RUNNING = re.compile(r"running|so far|subtotal|cumulative|stanza \d total", re.I)


def total_claims(text, drafts):
    out = []
    for m in TOTAL.finditer(text):
        v = int(m.group(1) or m.group(2))
        ctx = text[max(0, m.start() - 70): m.end() + 25]
        if not 100 <= v <= 250 or RULE.search(ctx) or RUNNING.search(ctx):
            continue
        prev = [d for d in drafts if d["end"] <= m.start() and m.start() - d["end"] < 8000]
        if prev:
            actual = body_words(prev[-1]["poem"])
            out.append({"stated": v, "actual": actual, "diff": v - actual, "at_pct": round(100 * m.start() / max(1, len(text)), 1),
                        "context": ctx.replace("\n", " ")})
    return out


# ------------------------------------------------------------------------------------------------ analyses
def analyze_poem(rec):
    r = rec["reasoning"]; n = max(1, len(r))
    drafts = complete_drafts(r)
    o = {"label": rec["label"], "model": rec["model"], "reasoning_chars": len(r), "complete_drafts": len(drafts), "drafts": []}
    for d in drafts:
        s, failed = grade(d["poem"])
        o["drafts"].append({"at_pct": round(100 * d["start"] / n, 1), "score": s, "failed": failed, "words": body_words(d["poem"])})
    if rec["answer"]:
        fs, ff = grade(rec["answer"])
        o.update(final_score=fs, final_failed=ff, final_words=body_words(rec["answer"]),
                 reversal_flag=any(d["score"] > fs for d in o["drafts"]),
                 lost_checks=sorted({c for c in ff for d in o["drafts"] if c not in d["failed"]}))
    else:
        o.update(final_score=None, final_failed=None, final_words=None, reversal_flag=None, lost_checks=None)
    pl, excluded = per_line_counts(r)
    o["per_line"] = {"n": len(pl), "exact": sum(1 for x in pl if x[1] == x[2]),
                     "bias_mean": round(sum(x[1] - x[2] for x in pl) / len(pl), 2) if pl else None,
                     "diffs": dict(sorted(collections.Counter(x[1] - x[2] for x in pl).items())),
                     "by_format": dict(collections.Counter(x[0] for x in pl)), "excluded_not_per_line": excluded,
                     "examples_wrong": [x for x in pl if x[1] != x[2]][:4]}
    tc = total_claims(r, drafts)
    o["totals"] = {"n": len(tc), "wrong_ge3": [c for c in tc if abs(c["diff"]) >= 3]}
    return o


BUGS = {
    "mutable default result=[]": r"mutable default|default (list|argument).{0,30}(shared|persist|reuse)|persists? (across|between) calls|shared (across|between) calls",
    "touching intervals (< vs <=)": r"should be <=|use <=|`<=`|touching intervals?.{0,40}(not merge|won't merge|fail|bug)|\[1,\s*2\].{0,30}\[2,\s*3\].{0,40}(not|fail|separate)",
    "sort mutates caller list": r"(sort|\.sort\(\)).{0,40}(in[- ]place|mutat|modif)|modif(y|ies) the (caller|input|original)",
    "p=100 index out of range": r"(p\s*=\s*100|c\s*=\s*(len|f\s*\+\s*1)).{0,80}(IndexError|out of (range|bounds))|IndexError.{0,60}p\s*=\s*100",
    "empty input no ValueError": r"(empty).{0,80}(no|doesn't|does not|not raise|never raise|IndexError|missing).{0,30}(ValueError|check|raise)",
}
DISMISS = re.compile(r"(actually|wait|hmm|on second thought)[^.\n]{0,80}(fine|not a bug|okay|ok\b|correct|no issue|not an issue|works)", re.I)


def code_spans(text):
    """Character spans that look like copied code (the module the prompt contains)."""
    spans = []
    for m in re.finditer(r"(?:^|\n)(?:[ \t]*(?:def |return |if |for |s = |k = |f = |c = |\"\"\"|result|intervals)[^\n]*\n)+", text):
        spans.append((m.start(), m.end()))
    return spans


def analyze_bugs(rec):
    r, a = rec["reasoning"], rec["answer"]; n = max(1, len(r))
    spans = code_spans(r)
    in_code = lambda p: any(s <= p < e for s, e in spans)
    o = {"label": rec["label"], "model": rec["model"], "reasoning_chars": len(r), "bugs": {}}
    for name, pat in BUGS.items():
        ms = [m for m in re.finditer(pat, r, re.I | re.S) if not in_code(m.start())]
        dis = []
        for m in ms:
            w = r[m.end(): m.end() + 250]; dm = DISMISS.search(w)
            if dm:
                dis.append({"at_pct": round(100 * m.start() / n, 1), "text": (r[m.start():m.end()] + w[:dm.end()]).replace("\n", " ")[:180]})
        o["bugs"][name] = {"first_diagnosis_pct": round(100 * ms[0].start() / n, 1) if ms else None, "mentions": len(ms),
                           "in_answer": bool(re.search(pat, a, re.I | re.S)), "dismissals": dis[:3]}
    firsts = [b["first_diagnosis_pct"] for b in o["bugs"].values()]
    o["all_diagnosed_by_pct"] = max(firsts) if all(f is not None for f in firsts) else None
    return o


def main():
    recs = [json.loads(l) for p in sys.argv[1:] for l in open(os.path.expanduser(p)) if l.strip()]
    rep = {"poem": [analyze_poem(r) for r in recs if is_poem(r)], "bugs": [analyze_bugs(r) for r in recs if is_bugs(r)]}
    print("=== POEM: #1 reversal | #2a per-line count accuracy | #2b totals")
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for o in rep["poem"]:
        pl = o["per_line"]; fam = o["model"].split("_")[0]
        agg[fam][0] += pl["n"]; agg[fam][1] += pl["exact"]; agg[fam][2] += (pl["bias_mean"] or 0) * pl["n"]
        print(f"\n{o['model']:<20} {o['label']:<26} drafts {o['complete_drafts']:<2} "
              + " ".join(f"{d['score']}@{d['at_pct']}%" for d in o["drafts"][:8])
              + f" | final {o['final_score']} failed {o['final_failed']} | REVERSAL={o['reversal_flag']} lost={o['lost_checks']}")
        print(f"    per-line counts: {pl['exact']}/{pl['n']} exact, bias {pl['bias_mean']}, diffs {pl['diffs']}, formats {pl['by_format']}, excluded(running totals) {pl['excluded_not_per_line']}")
        for x in pl["examples_wrong"][:2]:
            print(f"        stated {x[1]} real {x[2]} [{x[0]}] {x[3]}")
        for c in o["totals"]["wrong_ge3"][:2]:
            print(f"    WRONG TOTAL: stated {c['stated']} real {c['actual']} @{c['at_pct']}%: ...{c['context'][-90:]}")
    print("\n=== per-line counting accuracy by model family")
    for fam, (nl, ex, b) in agg.items():
        print(f"    {fam:<8} {ex}/{nl} exact ({100*ex/max(1,nl):.0f}%), mean bias {b/max(1,nl):+.2f} words/line")
    print("\n=== BUG HUNT: #1 reversal (diagnoses outside copied code)")
    for o in rep["bugs"]:
        print(f"\n{o['model']:<20} {o['label']}  reasoning {o['reasoning_chars']} chars; all 5 diagnosed by {o['all_diagnosed_by_pct']}%")
        for k, b in o["bugs"].items():
            print(f"    {k:<30} first@{b['first_diagnosis_pct']}%  mentions {b['mentions']:<4} in_answer={b['in_answer']}  dismissals={len(b['dismissals'])}")
            for d in b["dismissals"][:2]:
                print(f"        @{d['at_pct']}%: {d['text'][:150]}")
    dst = os.path.join(os.path.dirname(os.path.expanduser(sys.argv[1])), "quirks_report.json")
    json.dump(rep, open(dst, "w"), indent=2); print("\nsaved", dst)


if __name__ == "__main__":
    main()

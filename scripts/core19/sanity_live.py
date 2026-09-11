#!/usr/bin/env python3
"""Live sanity check for the Core-19 server (127.0.0.1:8085): three tasks, streamed so a human can watch.

Request shape mirrors Core-19's Terminus-2 agent: no max_tokens, no temperature/top_p, no reasoning_effort,
server defaults only; earlier reasoning is NOT sent back in history (Terminus-2 drops it too).

  1. instruction following  - constrained poem, auto-checked (rhyme shown for eyeballing)
  2. bug hunt               - two functions with planted bugs, graded by hidden tests
  3. difficult build        - stdlib 2D elastic-collision sim; checker recomputes physics from the printed
                              states; failures are fed back for up to 3 fix turns
"""
import json, math, os, re, resource, subprocess, sys, tempfile, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8085/v1")
OUT = os.path.expanduser(os.environ.get("OUT", "~/benchmarks/core19/manual-turbo"))
GREY, WHITE, CYAN, GREEN, RED, YEL, RST = "\033[90m", "\033[97m", "\033[96m", "\033[92m", "\033[91m", "\033[93m", "\033[0m"
RESULTS = {"started": time.strftime("%F %T"), "base": BASE, "tasks": {}}


def banner(t):
    print(f"\n{CYAN}{'=' * 100}\n  {t}\n{'=' * 100}{RST}", flush=True)


def chat(messages, label):
    """Stream one completion. Returns (content, stats)."""
    body = {"model": "x", "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    print(f"\n{YEL}--- {label}: request sent ({len(messages)} messages) ---{RST}", flush=True)
    t0 = time.time(); ttft = None; mode = None
    content, reasoning, usage, timings, finish = [], [], {}, {}, None
    with urllib.request.urlopen(req, timeout=36000) as r:
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
            usage = ch.get("usage") or usage
            timings = ch.get("timings") or timings
            for c in ch.get("choices") or []:
                finish = c.get("finish_reason") or finish
                d = c.get("delta") or {}
                rc, cc = d.get("reasoning_content"), d.get("content")
                if rc:
                    if ttft is None: ttft = time.time() - t0
                    if mode != "r": print(f"\n{GREY}[reasoning]\n", end=""); mode = "r"
                    print(rc, end="", flush=True); reasoning.append(rc)
                if cc:
                    if ttft is None: ttft = time.time() - t0
                    if mode != "c": print(f"{RST}\n{WHITE}[answer]\n", end=""); mode = "c"
                    print(cc, end="", flush=True); content.append(cc)
    wall = time.time() - t0
    st = {"started": time.strftime("%F %T", time.localtime(t0)), "ended": time.strftime("%F %T"),
          "wall_s": round(wall, 1), "ttft_s": round(ttft or 0, 2), "finish": finish,
          "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
          "reasoning_chars": len("".join(reasoning)), "answer_chars": len("".join(content)),
          "pp_tok_s": round(timings.get("prompt_per_second") or 0, 1),
          "tg_tok_s": round(timings.get("predicted_per_second") or 0, 1)}
    if timings.get("draft_n"):
        st["mtp_accept"] = round(timings.get("draft_n_accepted", 0) / timings["draft_n"], 3)
    print(f"{RST}\n{YEL}--- {label} stats: {st} ---{RST}", flush=True)
    try:  # preserve full traces: one JSON record per model call
        with open(os.path.join(OUT, "traces.jsonl"), "a") as tf:
            tf.write(json.dumps({"label": label, "model": RESULTS.get("model"), "messages": messages, "stats": st,
                                 "reasoning": "".join(reasoning), "answer": "".join(content)}) + "\n")
    except Exception as e:
        print(f"{RED}   (trace save failed: {e}){RST}")
    return "".join(content), st


def last_block(text, lang="python"):
    blocks = re.findall(r"```(?:%s|py)?\s*\n(.*?)```" % lang, text, re.S)
    return blocks[-1] if blocks else None


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def run_py(code, timeout=90, stdin_extra=""):
    d = tempfile.mkdtemp(prefix="sanity_")
    f = os.path.join(d, "prog.py")
    open(f, "w").write(code + stdin_extra)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, text=True,
                           timeout=timeout, preexec_fn=limits)
        return p.returncode, p.stdout, p.stderr, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT after %ss" % timeout, timeout


def check(name, ok, detail=""):
    print(f"   {GREEN + 'PASS' if ok else RED + 'FAIL'}{RST}  {name}  {GREY}{detail}{RST}", flush=True)
    return {"check": name, "ok": bool(ok), "detail": str(detail)[:300]}


# ----------------------------------------------------------------------------------------------- task 1
POEM_PROMPT = """Write a short poem that follows ALL of these rules exactly.

Title line: `# The Clockmaker's Last Apprentice` (exactly this, first line).
Voice: first person, an old clockmaker speaking to their young apprentice on the clockmaker's final day at the bench.
Style: gentle and slightly archaic; use the words "thee" or "thou" at least 3 times in total.
Theme: some things cannot be repaired, only kept.
Form:
- exactly 5 stanzas of exactly 4 lines each, stanzas separated by one blank line
- rhyme scheme AABB in every stanza
- the first word of each stanza, in order, must be: Wind, Tick, Rust, Chime, Stop
- the poem body (everything after the title) must be between 140 and 160 words
- the word "time" (or "times") must not appear anywhere
- the very last line must end with a question mark
Output only the title line and the poem, nothing else."""


def grade_poem(text):
    rs = []
    lines = text.strip().splitlines()
    rs.append(check("title line exact", lines and lines[0].strip() == "# The Clockmaker's Last Apprentice", lines[0] if lines else ""))
    body = "\n".join(lines[1:]).strip()
    stanzas = [s for s in re.split(r"\n\s*\n", body) if s.strip()]
    rs.append(check("5 stanzas", len(stanzas) == 5, len(stanzas)))
    shape = [len([l for l in s.splitlines() if l.strip()]) for s in stanzas]
    rs.append(check("4 lines per stanza", shape == [4] * 5, shape))
    words = re.findall(r"[A-Za-z']+", body)
    rs.append(check("140-160 words", 140 <= len(words) <= 160, len(words)))
    firsts = [re.findall(r"[A-Za-z']+", s)[0] if re.findall(r"[A-Za-z']+", s) else "" for s in stanzas]
    rs.append(check("stanza first words Wind/Tick/Rust/Chime/Stop", firsts == ["Wind", "Tick", "Rust", "Chime", "Stop"], firsts))
    banned = re.findall(r"\btimes?\b", text, re.I)
    rs.append(check('no "time"/"times"', not banned, banned))
    tt = len(re.findall(r"\b(thee|thou)\b", body, re.I))
    rs.append(check("thee/thou >= 3", tt >= 3, tt))
    rs.append(check("last line ends with ?", body.rstrip().endswith("?"), body.rstrip()[-40:]))
    print(f"   {GREY}rhyme pairs (judge by ear):{RST}")
    for i, s in enumerate(stanzas, 1):
        ends = [re.findall(r"[A-Za-z']+", l)[-1].lower() if re.findall(r"[A-Za-z']+", l) else "" for l in s.splitlines() if l.strip()]
        print(f"   {GREY}  stanza {i}: {ends[0:2]}  {ends[2:4]}{RST}")
    return rs


# ----------------------------------------------------------------------------------------------- task 2
BUG_CODE = '''def merge_intervals(intervals, sort_input=True, result=[]):
    """Merge overlapping closed intervals [start, end].

    Intervals that merely touch, like [1, 2] and [2, 3], must also merge.
    The caller's list must not be modified. Returns a new list of [start, end] lists.
    """
    if sort_input:
        intervals.sort()
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def percentile(values, p):
    """Return the p-th percentile (0 <= p <= 100) of values, linear interpolation between closest ranks.

    Raises ValueError on an empty input.
    """
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    return s[f] + (s[c] - s[f]) * (k - f)
'''

BUG_PROMPT = f"""Here is a small Python module. Its docstrings are the specification. It contains bugs.

```python
{BUG_CODE}```

Deliverables:
1. A numbered list of every bug you find: which function, which line, and why it is wrong against the docstring.
2. The complete corrected module in ONE ```python code block (keep both function names and their parameters;
   you may change a default value if that is part of a fix).
Do not add tests or example usage to the code block."""

BUG_TESTS = r'''

def _norm(x): return [list(i) for i in x]
_results = []
def _t(name, fn):
    try:
        ok = bool(fn())
    except Exception as e:
        ok = False; name += " (raised %s: %s)" % (type(e).__name__, e)
    _results.append((name, ok))
_t("merge basic", lambda: _norm(merge_intervals([[1,3],[2,6],[8,10]])) == [[1,6],[8,10]])
_t("merge touching", lambda: _norm(merge_intervals([[1,2],[2,3]])) == [[1,3]])
_t("merge no state leak between calls", lambda: (merge_intervals([[1,2]]), _norm(merge_intervals([[5,6]])))[1] == [[5,6]])
def _nomut():
    a = [[5,6],[1,2]]; merge_intervals(a); return a == [[5,6],[1,2]]
_t("merge does not modify caller list", _nomut)
_t("merge contained", lambda: _norm(merge_intervals([[1,10],[2,3]])) == [[1,10]])
_t("merge empty", lambda: _norm(merge_intervals([])) == [])
_t("merge tuples input", lambda: _norm(merge_intervals([(1,4),(2,5)])) == [[1,5]])
_t("percentile p=100", lambda: percentile([1,2,3,4], 100) == 4)
_t("percentile p=0", lambda: percentile([1,2,3,4], 0) == 1)
_t("percentile p=50", lambda: abs(percentile([1,2,3,4], 50) - 2.5) < 1e-9)
_t("percentile single element", lambda: percentile([5], 73) == 5)
def _empty():
    try:
        percentile([], 50)
    except ValueError:
        return True
    return False
_t("percentile empty raises ValueError", _empty)
import json as _j; print("__TESTS__" + _j.dumps(_results))
'''


def grade_bugs(text):
    rs = []
    code = last_block(text)
    rs.append(check("corrected module in a python block", code is not None))
    if code is None:
        return rs
    rc, out, err, secs = run_py(code, timeout=30, stdin_extra=BUG_TESTS)
    m = re.search(r"__TESTS__(\[.*\])", out)
    if not m:
        rs.append(check("hidden tests ran", False, (err or out)[-300:]))
        return rs
    for name, ok in json.loads(m.group(1)):
        rs.append(check("hidden test: " + name, ok))
    return rs


# ----------------------------------------------------------------------------------------------- task 3
SIM_PROMPT = """Write a single-file Python 3 program: a 2D rigid-disk physics simulation using ONLY the standard library
(no numpy, no pygame, nothing to install).

Physics:
- 12 disks, all radius r = 0.05, masses between 1.0 and 3.0, inside a 1 x 1 box (0 <= x, y <= 1).
- Gravity g = 9.81 pulling toward y = 0.
- Perfectly elastic collisions (restitution 1, no friction, no rotation) between disks and with all four walls.
- Initial positions must not overlap; initial velocities random in [-1, 1]. Use random.seed(42).
- Simulate exactly 4.0 seconds of simulated time.

Requirements (a checker will verify them independently from your printed state, so they must be physically real):
1. Total mechanical energy E = sum(0.5*m*(vx^2+vy^2) + m*9.81*y) at the end must be within 0.5% of the initial value.
2. At the end, no pair of disks overlaps by more than 1% of the radius, and every disk is fully inside the box.
3. Runs in under 20 seconds on plain CPython.
4. The program's LAST line of output must be exactly one JSON object with these keys:
   "t_end" (float), "steps" (int), "collisions" (int),
   "initial_state" and "final_state": each a list of 12 entries [x, y, vx, vy, m].
Print nothing that could be confused with that final JSON line.

Return the complete program in ONE ```python code block."""


def grade_sim(text):
    rs = []
    code = last_block(text)
    rs.append(check("program in a python block", code is not None))
    if code is None:
        return rs, "Your reply did not contain a ```python code block with the program."
    if re.search(r"^\s*(import|from)\s+(numpy|scipy|pygame|pymunk)", code, re.M):
        rs.append(check("stdlib only", False))
        return rs, "The program imports a non-standard library. Use only the Python standard library."
    rc, out, err, secs = run_py(code, timeout=90)
    rs.append(check("exits 0", rc == 0, "rc=%s %s" % (rc, err[-200:] if rc else "")))
    rs.append(check("runtime < 20 s", secs < 20 and rc != -9, "%ss" % secs))
    lines = [l for l in out.strip().splitlines() if l.strip()]
    try:
        data = json.loads(lines[-1])
        s0, s1 = data["initial_state"], data["final_state"]
        assert len(s0) == 12 and len(s1) == 12 and all(len(e) == 5 for e in s0 + s1)
        rs.append(check("final JSON line parses with 12x5 states", True))
    except Exception as e:
        rs.append(check("final JSON line parses with 12x5 states", False, repr(e)[:150]))
        fb = f"The run failed the output contract.\nexit code: {rc}\nruntime: {secs}s\nstderr tail:\n{err[-1500:]}\nlast stdout lines:\n" + "\n".join(lines[-5:])[-1500:]
        return rs, fb
    E = lambda st: sum(0.5 * m * (vx * vx + vy * vy) + m * 9.81 * y for x, y, vx, vy, m in st)
    e0, e1 = E(s0), E(s1)
    drift = abs(e1 - e0) / abs(e0) * 100 if e0 else float("inf")
    rs.append(check("t_end == 4.0", abs(float(data.get("t_end", -1)) - 4.0) < 1e-6, data.get("t_end")))
    rs.append(check("masses in [1,3] and unchanged", all(1 <= a[4] <= 3 and abs(a[4] - b[4]) < 1e-12 for a, b in zip(s0, s1))))
    def overlaps(st):
        worst = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                d = math.hypot(st[i][0] - st[j][0], st[i][1] - st[j][1])
                worst = max(worst, (0.1 - d) / 0.05)
        return worst
    ov0, ov1 = overlaps(s0), overlaps(s1)
    margin = min(min(x - 0.05, 1 - x - 0.05, y - 0.05, 1 - y - 0.05) for x, y, *_ in s1)
    rs.append(check("initial disks non-overlapping", ov0 <= 0.0, "worst overlap %.4f r" % ov0))
    rs.append(check("energy drift < 0.5% (recomputed)", drift < 0.5, "E0=%.6f E1=%.6f drift=%.4f%%" % (e0, e1, drift)))
    rs.append(check("final overlap <= 1% r", ov1 <= 0.01, "worst %.4f r" % ov1))
    rs.append(check("all disks inside box", margin >= -1e-9, "min wall margin %.5f" % margin))
    rs.append(check("collisions counted > 0", int(data.get("collisions", 0)) > 0, data.get("collisions")))
    failed = [r for r in rs if not r["ok"]]
    if not failed:
        return rs, None
    fb = ("The checker ran your program and found problems:\n" + "\n".join("- %s: %s" % (r["check"], r["detail"]) for r in failed)
          + f"\n(runtime {secs}s, exit code {rc})\nFix the program and return the complete corrected program in ONE ```python code block.")
    return rs, fb


# ----------------------------------------------------------------------------------------------- main
def save():
    open(os.path.join(OUT, "sanity_results.json"), "w").write(json.dumps(RESULTS, indent=2) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    models = json.load(urllib.request.urlopen(BASE + "/models", timeout=10))
    RESULTS["model"] = (models.get("data") or [{}])[0].get("id")
    banner("SANITY CHECK  model: %s" % os.path.basename(RESULTS["model"] or "?"))

    start = int(os.environ.get("START_TASK", "1"))
    prev = os.path.join(OUT, "sanity_results.json")
    if start > 1 and os.path.exists(prev):
        RESULTS["tasks"].update(json.load(open(prev)).get("tasks", {}))
    if os.environ.get("TASK1_NOTE"):
        RESULTS["tasks"]["1_poem"] = {"aborted": os.environ["TASK1_NOTE"], "stats": [], "checks": [], "passed": 0, "total": 0}
        save()

    if start <= 1:
        banner("TASK 1/3  instruction following: constrained poem")
        text, st = chat([{"role": "user", "content": POEM_PROMPT}], "task1 turn1")
        print(f"\n{CYAN}--- task 1 checks ---{RST}")
        rs = grade_poem(text)
        RESULTS["tasks"]["1_poem"] = {"stats": [st], "checks": rs, "passed": sum(r["ok"] for r in rs), "total": len(rs)}; save()

    banner("TASK 2/3  bug hunt: planted bugs, hidden tests")
    text, st = chat([{"role": "user", "content": BUG_PROMPT}], "task2 turn1")
    print(f"\n{CYAN}--- task 2 checks ---{RST}")
    rs = grade_bugs(text)
    RESULTS["tasks"]["2_bugs"] = {"stats": [st], "checks": rs, "passed": sum(r["ok"] for r in rs), "total": len(rs)}; save()

    banner("TASK 3/3  difficult build: stdlib elastic-collision simulation (up to 3 fix turns)")
    msgs = [{"role": "user", "content": SIM_PROMPT}]
    t3 = {"stats": [], "turns": []}
    for turn in range(1, 5):
        text, st = chat(msgs, "task3 turn%d" % turn)
        t3["stats"].append(st)
        print(f"\n{CYAN}--- task 3 checks (turn {turn}) ---{RST}")
        rs, fb = grade_sim(text)
        t3["turns"].append({"turn": turn, "checks": rs, "passed": sum(r["ok"] for r in rs), "total": len(rs)})
        RESULTS["tasks"]["3_sim"] = t3; save()
        if fb is None:
            print(f"{GREEN}   task 3 solved on turn {turn}{RST}")
            break
        if turn == 4:
            print(f"{RED}   task 3 not solved after 4 turns{RST}")
            break
        print(f"\n{YEL}--- feeding checker output back to the model ---{RST}\n{GREY}{fb}{RST}")
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": fb}]

    banner("SUMMARY")
    for k, v in RESULTS["tasks"].items():
        if v.get("aborted"):
            print(f"  {k}: ABORTED - {v['aborted']}")
            continue
        if "turns" in v:
            last = v["turns"][-1]
            print(f"  {k}: turn {last['turn']} -> {last['passed']}/{last['total']} checks | turns used {len(v['turns'])}")
        else:
            print(f"  {k}: {v['passed']}/{v['total']} checks")
        for s in v["stats"]:
            print(f"     {GREY}{s}{RST}")
    RESULTS["finished"] = time.strftime("%F %T"); save()
    print(f"\n{CYAN}SANITY DONE{RST}  results: {OUT}/sanity_results.json")


if __name__ == "__main__":
    main()

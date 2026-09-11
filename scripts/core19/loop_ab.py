#!/usr/bin/env python3
"""Loop A/B: does Turbo's reasoning loop depend on KV-cache compression, or is it the fine-tune?

Same poem prompt as the sanity check, 3 runs per condition, no max_tokens, no sampler overrides
(server/GGUF defaults, exactly like Core-19). A run is stopped ONLY when a loop is detected:
some reasoning line (>= 15 chars) has appeared >= LOOP_REPEATS times. No token cap otherwise.

  A  Turbo    KV q8_0  ctx 262144   (the Core-19 launch line)
  B  Turbo    KV f16   ctx 131072   (David's examples: "NO cache compression of any kind"; 131k so f16 fits)
  C  Heretic  KV q8_0  ctx 262144   (control: same launch line as A)

Server is (re)launched per condition on 127.0.0.1:8085. At the end the Core-19 Turbo server is restored;
production (qwen38.service) is left alone (stopped).
"""
import collections, json, os, re, signal, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sanity_live as S  # noqa: E402  (prompt, poem grader, colours)

OUT = os.path.expanduser("~/benchmarks/core19/manual-turbo/loop_ab")
BIN = os.path.expanduser("~/projects/llama.cpp-master/build-vulkan/bin/llama-server")
MMPROJ = "/mnt/models/llm/qwen3.8-27b-instruct-unsloth/mmproj-BF16.gguf"
TURBO = "/mnt/models/llm/qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau/qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau-mtp-q4_k_s.gguf"
HERETIC = "/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf"
RUNS = int(os.environ.get("RUNS", "3"))
LOOP_REPEATS = int(os.environ.get("LOOP_REPEATS", "30"))
CONDITIONS = [
    ("A_turbo_q8_262k", TURBO, "262144", "q8_0"),
    ("B_turbo_f16_131k", TURBO, "131072", "f16"),
    ("C_heretic_q8_262k", HERETIC, "262144", "q8_0"),
]
BASE = "http://127.0.0.1:8085"


def argv(model, ctx, kv):
    return [BIN, "-m", model, "--mmproj", MMPROJ, "--image-min-tokens", "1024", "-dev", "Vulkan0", "-ngl", "99",
            "-c", ctx, "-ctk", kv, "-ctv", kv, "-fa", "on", "--ctx-checkpoints", "2", "-np", "2", "-kvu", "--jinja",
            "--reasoning-format", "deepseek", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "-cram", "12288",
            "--host", "127.0.0.1", "--port", "8085", "--log-timestamps", "--log-prefix"]


def gpu0_mib():
    for d in os.listdir("/sys/class/drm"):
        p = f"/sys/class/drm/{d}/device"
        try:
            if int(open(p + "/mem_info_vram_total").read()) > 32_000_000_000:
                return int(open(p + "/mem_info_vram_used").read()) // 1048576
        except Exception:
            pass
    return None


def stop_servers():
    subprocess.run(["tmux", "kill-session", "-t", "c19srv"], capture_output=True)
    subprocess.run(["pkill", "-TERM", "-f", "build-vulkan/bin/llama-server .*--port 8085"], capture_output=True)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", "build-vulkan/bin/llama-server .*--port 8085"], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", "build-vulkan/bin/llama-server .*--port 8085"], capture_output=True)
    time.sleep(3)
    return gpu0_mib()


def start_server(name, model, ctx, kv):
    log = open(os.path.join(OUT, f"{name}.server.log"), "ab")
    p = subprocess.Popen(argv(model, ctx, kv), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for i in range(120):
        try:
            if urllib.request.urlopen(BASE + "/health", timeout=4).status == 200:
                return p, i * 2, gpu0_mib()
        except Exception:
            pass
        if p.poll() is not None:
            return None, None, None
        time.sleep(2)
    return None, None, None


def run_poem(label):
    """Stream the poem task; cancel only on a detected loop. Returns result dict."""
    body = {"model": "x", "messages": [{"role": "user", "content": S.POEM_PROMPT}], "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"\n{S.YEL}--- {label}: request sent ---{S.RST}", flush=True)
    t0 = time.time(); mode = None
    reasoning, content, timings, usage = [], [], {}, {}
    counts = collections.Counter(); buf = ""; loop = None; chunks = 0
    r = urllib.request.urlopen(req, timeout=36000)
    try:
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
                d = c.get("delta") or {}
                rc, cc = d.get("reasoning_content"), d.get("content")
                if rc:
                    chunks += 1
                    if mode != "r": print(f"\n{S.GREY}[reasoning]\n", end=""); mode = "r"
                    print(rc, end="", flush=True); reasoning.append(rc)
                    buf += rc
                    while "\n" in buf:
                        ln, buf = buf.split("\n", 1)
                        ln = ln.strip()
                        if len(ln) >= 15:
                            counts[ln] += 1
                            if counts[ln] >= LOOP_REPEATS and loop is None:
                                loop = {"line": ln[:120], "repeats": counts[ln], "at_s": round(time.time() - t0, 1),
                                        "reasoning_chars": len("".join(reasoning))}
                if cc:
                    if mode != "c": print(f"{S.RST}\n{S.WHITE}[answer]\n", end=""); mode = "c"
                    print(cc, end="", flush=True); content.append(cc)
            if loop:
                break
    finally:
        r.close()  # closing the connection makes llama-server cancel the task
    wall = round(time.time() - t0, 1)
    res = {"label": label, "started": time.strftime("%F %T", time.localtime(t0)), "ended": time.strftime("%F %T"),
           "wall_s": wall, "loop": loop, "reasoning_chars": len("".join(reasoning)),
           "answer_chars": len("".join(content)), "completion_tokens": usage.get("completion_tokens"),
           "tg_tok_s": round(timings.get("predicted_per_second") or 0, 1),
           "mtp_accept": round(timings["draft_n_accepted"] / timings["draft_n"], 3) if timings.get("draft_n") else None}
    if loop:
        print(f"{S.RST}\n{S.RED}--- {label}: LOOP DETECTED -> cancelled: {loop} ---{S.RST}", flush=True)
    else:
        print(f"{S.RST}\n{S.CYAN}--- {label}: finished, grading ---{S.RST}", flush=True)
        rs = S.grade_poem("".join(content))
        res["checks_passed"] = sum(x["ok"] for x in rs); res["checks_total"] = len(rs)
        res["checks"] = rs
        res["answer"] = "".join(content)
    res["reasoning"] = "".join(reasoning)
    res["answer"] = "".join(content)
    print(f"{S.YEL}--- {label} stats: { {k: v for k, v in res.items() if k not in ('checks', 'answer', 'reasoning')} } ---{S.RST}", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"started": time.strftime("%F %T"), "runs_per_condition": RUNS, "loop_repeats": LOOP_REPEATS, "conditions": {}}
    save = lambda: open(os.path.join(OUT, "results.json"), "w").write(json.dumps(results, indent=2) + "\n")
    try:
        for name, model, ctx, kv in CONDITIONS:
            S.banner(f"CONDITION {name}: {os.path.basename(model)}  ctx {ctx}  KV {kv}")
            free = stop_servers()
            print(f"   GPU0 after stop: {free} MiB", flush=True)
            proc, load_s, mib = start_server(name, model, ctx, kv)
            if not proc:
                print(f"{S.RED}   server failed to start; see {name}.server.log{S.RST}")
                results["conditions"][name] = {"error": "server start failed"}; save(); continue
            print(f"   server healthy after ~{load_s}s, GPU0 {mib} MiB", flush=True)
            cond = {"model": os.path.basename(model), "ctx": ctx, "kv": kv, "gpu0_mib": mib, "runs": []}
            results["conditions"][name] = cond
            for i in range(1, RUNS + 1):
                cond["runs"].append(run_poem(f"{name} run {i}/{RUNS}")); save()
            os.killpg(proc.pid, signal.SIGTERM)
    finally:
        S.banner("SUMMARY")
        for name, c in results["conditions"].items():
            if "error" in c:
                print(f"  {name}: {c['error']}"); continue
            loops = sum(1 for r in c["runs"] if r["loop"])
            print(f"  {name}: loops {loops}/{len(c['runs'])}")
            for r in c["runs"]:
                if r["loop"]:
                    print(f"     LOOP after {r['loop']['at_s']}s / {r['completion_tokens'] or '?'} tok: {r['loop']['line'][:80]}")
                else:
                    print(f"     done in {r['wall_s']}s, {r.get('completion_tokens')} tok, poem checks {r.get('checks_passed')}/{r.get('checks_total')}, reasoning {r['reasoning_chars']} chars")
        results["finished"] = time.strftime("%F %T"); save()
        print(f"\n{S.CYAN}restoring the Core-19 Turbo server (q8_0, 262k) in tmux c19srv; production stays stopped{S.RST}")
        stop_servers()
        subprocess.run(["tmux", "new-session", "-d", "-s", "c19srv",
                        "bash " + os.path.expanduser("~/benchmarks/core19/manual-turbo/relaunch.sh")])
        print(f"{S.CYAN}LOOP AB DONE{S.RST}  results: {OUT}/results.json")


if __name__ == "__main__":
    main()

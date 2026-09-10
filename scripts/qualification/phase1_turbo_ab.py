#!/usr/bin/env python3
"""Phase 1 -- qualify DavidAU Turbo Heretic quants against current heretic (co-measured).

Handoff: agent-1 -> Claude, 2026-09-10_turbo-heretic-ab.md. The handoff is the consent for
these swaps. Every launch is the live production line with ONLY -m swapped (plus -np for
the VRAM ladder). The same harness is used for every model; nothing is tuned per model.

Per model, in one thermal window (heretic re-measured as the control, not borrowed from Session 7):
  np=1  VRAM
  np=2  VRAM; predictable + creative 2048-tok gens (nmax_sweep_mtp.py method) with
        per-request MTP acceptance; bench_sweep.py prefill @42k (Session 7 settings)
  np=3  VRAM; conc_test.py 1/2/3-stream ladder + acceptance under load
  np=4  VRAM + sanity (design goal, previously untested)
Disqualifier (handoff + DavidAU card): pooled MTP acceptance < 0.50.
Production (qwen38.service) is restored in `finally`, whatever happens.
"""
import datetime, json, os, re, shlex, subprocess, sys, time, urllib.request

E = os.path.expanduser("~/benchmarks/engine_matrix")
OUT = os.path.join(E, "results_turbo_ab")
os.makedirs(OUT, exist_ok=True)
BIN = os.path.expanduser("~/projects/llama.cpp-master/build-vulkan/bin/llama-server")
MMPROJ = "/mnt/models/llm/qwen3.8-27b-instruct-unsloth/mmproj-BF16.gguf"
TD = "/mnt/models/llm/qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau/"
TP = TD + "qwen3.8-27b-turbo-fable-cold-fusion-735-heretic-davidau-mtp-"
MODELS = [
    ("heretic-q4_k_s", "/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf"),
    ("turbo-mtp-q4_k_s", TP + "q4_k_s.gguf"),
    ("turbo-mtp-iq4_xs", TP + "iq4_xs.gguf"),
]
# Port split (the owner, 2026-09-10): production stays on 0.0.0.0:8080; the test server binds
# 127.0.0.1:8085 so crons/compaction/agents that target 8080 fail loudly during a benchmark
# instead of silently sharing the GPU with it.
PORT = 8085
BASE = "http://127.0.0.1:%d" % PORT
PROD_BASE = "http://127.0.0.1:8080"
SRV = "p1srv"
MAXTOK = 2048
RESULTS = os.path.join(OUT, "results.json")

PREDICTABLE = ("Write a comprehensive, well-commented Python module implementing: an LRU cache with "
 "TTL expiry, a thread-safe bounded queue, a retry decorator with exponential backoff and jitter, "
 "a simple dependency-injection container, and a binary search tree with insert/delete/traverse. "
 "Include type hints, docstrings, and a demonstration section exercising every component.")
CREATIVE = ("Write the opening chapter of a surrealist novel. A lighthouse keeper on a distant coast "
 "wakes one morning to discover the ocean has vanished overnight, replaced by an endless field of tall "
 "grey grass. Strange figures move through the grass at dusk. Develop the keeper's history, the "
 "texture of the place, and the first encounter. Begin in medias res, unsettling and dreamlike.")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)


def argv(model, np_):
    return [BIN, "-m", model, "--mmproj", MMPROJ, "--image-min-tokens", "1024",
            "-dev", "Vulkan0", "-ngl", "99", "-c", "262144", "-ctk", "q8_0", "-ctv", "q8_0",
            "-fa", "on", "--ctx-checkpoints", "2", "-np", str(np_), "-kvu", "--jinja",
            "--reasoning-format", "deepseek", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "-cram", "12288", "--host", "127.0.0.1", "--port", str(PORT),
            "--log-timestamps", "--log-prefix"]


def gpu0_vram_gib():
    # R9700 = the 32 GB card; pick it by size, not by cardN (numbering is inverted in tools)
    for d in sorted(os.listdir("/sys/class/drm")):
        p = "/sys/class/drm/%s/device" % d
        try:
            total = int(open(p + "/mem_info_vram_total").read())
            if total > 30 * 2**30:
                return round(int(open(p + "/mem_info_vram_used").read()) / 2**30, 2)
        except Exception:
            continue
    return None


def health(base=None):
    try:
        with urllib.request.urlopen((base or BASE) + "/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def llama_pids():
    return sh(["pgrep", "-x", "llama-server"]).stdout.split()


def wait_gone(limit=90):
    t0 = time.time()
    while time.time() - t0 < limit:
        if not llama_pids() and (gpu0_vram_gib() or 0) < 1.5:
            return True
        time.sleep(2)
    return False


def stop_test_server():
    sh(["tmux", "kill-session", "-t", SRV])
    if not wait_gone(60):
        for p in llama_pids():
            sh(["kill", p])
        wait_gone(30)


def launch(model, np_, logfile):
    stop_test_server()
    cmd = shlex.join(argv(model, np_)) + " > " + shlex.quote(logfile) + " 2>&1"
    sh(["tmux", "new-session", "-d", "-s", SRV, cmd])
    t0 = time.time()
    while time.time() - t0 < 900:
        if health():
            return round(time.time() - t0, 1)
        if not llama_pids() and time.time() - t0 > 20:
            break
        time.sleep(3)
    return None


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def sanity():
    body = {"model": "x", "max_tokens": 200, "temperature": 0,
            "messages": [{"role": "user", "content": "Name three primary colors. Answer in one short sentence."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        d = json.load(post(body, 300))
        c = d["choices"][0]
        return ((c["message"].get("content") or "").strip()[:120], c.get("finish_reason"))
    except Exception as e:
        return ("ERROR " + str(e)[:120], None)


def gen(prompt):
    """nmax_sweep_mtp.py method: stream, temp 0, seed 42, template-default thinking,
    count reasoning + content chunks. Also split them, for the thinking-length claim."""
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "temperature": 0, "seed": 42, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    tf = tl = None
    n = n_reason = n_content = 0
    usage = {}
    head = []
    with post(body) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if not ch:
                continue
            dl = ch[0].get("delta") or {}
            rc, cc = dl.get("reasoning_content"), dl.get("content")
            if rc or cc:
                now = time.time()
                tf = tf or now
                tl = now
                n += 1
                n_reason += 1 if rc else 0
                n_content += 1 if cc else 0
                if len(head) < 60 and cc:
                    head.append(cc)
    return {"tok": n, "reasoning_chunks": n_reason, "content_chunks": n_content,
            "completion_tokens": usage.get("completion_tokens"),
            "tok_s": round(n / (tl - tf), 2) if (tf and tl and tl > tf and n > 1) else None,
            "ttft": round(tf - t0, 2) if tf else None, "wall": round(time.time() - t0, 2),
            "content_head": "".join(head)[:160]}


ACC = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")


def acceptance(logfile, offset):
    try:
        with open(logfile, "rb") as f:
            f.seek(offset)
            txt = f.read().decode("utf-8", "ignore")
    except Exception:
        return {"pooled": None, "mean_len": None, "n": 0}
    rows = ACC.findall(txt)
    if not rows:
        return {"pooled": None, "mean_len": None, "n": 0}
    acc = sum(int(r[1]) for r in rows)
    gen_ = sum(int(r[2]) for r in rows)
    return {"pooled": round(acc / gen_, 3) if gen_ else None,
            "mean_len": round(sum(float(r[3]) for r in rows) / len(rows), 2), "n": len(rows)}


def fsize(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def save(data):
    open(RESULTS, "w").write(json.dumps(data, indent=2) + "\n")


def restore_production():
    stop_test_server()
    sh(["systemctl", "--user", "start", "qwen38.service"])
    t0 = time.time()
    while time.time() - t0 < 900:
        if health(PROD_BASE):
            return round(time.time() - t0, 1)
        time.sleep(5)
    return None


def main():
    data = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
            "build": sh([BIN, "--version"]).stderr.strip().splitlines()[-2:],
            "power": sh("rocm-smi --showmaxpower | grep -i max").stdout.strip().splitlines(),
            "harness": "phase1_turbo_ab.py; prod line with only -m/-np swapped; MTP n_max=2",
            "models": {}}

    # --- 0. wait for agent-1/agent-2 to be idle, save restore line, stop production ---
    for _ in range(120):
        try:
            slots = json.load(urllib.request.urlopen(PROD_BASE + "/slots", timeout=5))
            if not any(s.get("is_processing") for s in slots):
                break
            log("a slot is processing; waiting for the turn to finish")
        except Exception:
            break
        time.sleep(10)
    live = sh("pgrep -af '[l]lama-server'").stdout.strip()
    rpath = os.path.join(E, "restore", "llamaserver_cmd_production_mtp_n2_%s.txt" % time.strftime("%Y%m%d_%H%M"))
    open(rpath, "w").write(live + "\n")
    log("restore line saved -> " + rpath)
    sh(["systemctl", "--user", "stop", "qwen38.service"])
    freed = wait_gone(120)
    log("production stopped; VRAM freed=%s (%s GiB)" % (freed, gpu0_vram_gib()))
    data["restore_line"] = rpath
    save(data)
    if not freed:
        log("ABORT: VRAM not freed")
        return

    for name, model in MODELS:
        m = {"file": model, "size_bytes": fsize(model)}
        data["models"][name] = m
        log("================ %s ================" % name)

        # np=1: VRAM only
        lf = os.path.join(OUT, "%s_np1.log" % name)
        m["np1_load_s"] = launch(model, 1, lf)
        m["np1_vram_gib"] = gpu0_vram_gib() if m["np1_load_s"] else "START_FAIL"
        m["np1_sanity"] = sanity() if m["np1_load_s"] else None
        log("np1 load=%ss vram=%s sanity=%s" % (m["np1_load_s"], m["np1_vram_gib"], m["np1_sanity"]))
        save(data)
        if not m["np1_load_s"]:
            continue

        # np=2: production shape
        lf = os.path.join(OUT, "%s_np2.log" % name)
        m["np2_load_s"] = launch(model, 2, lf)
        if m["np2_load_s"]:
            m["np2_vram_gib_idle"] = gpu0_vram_gib()
            m["np2_sanity"] = sanity()
            gen("Say hello in one sentence.")
            for cls, prompt in (("predictable", PREDICTABLE), ("creative", CREATIVE)):
                off = fsize(lf)
                r = gen(prompt)
                time.sleep(1)
                r["accept"] = acceptance(lf, off)
                m["np2_" + cls] = r
                log("np2 %s: %s" % (cls, json.dumps(r)))
                save(data)
            off = fsize(lf)
            bs = sh(["python3", os.path.join(E, "bench_sweep.py"),
                     "--url", BASE + "/v1/chat/completions", "--model", model,
                     "--tag", name + "__np2__mtp2", "--outdir", OUT, "--thinking", "off",
                     "--large-prompt-tokens", "42000", "--large-gen-tokens", "1536",
                     "--sweep-levels", "1,2", "--sweep-budget-s", "240"])
            try:
                bj = json.load(open(os.path.join(OUT, name + "__np2__mtp2.json")))
                lp = bj["large_prompt_prefill_gen"]
                m["np2_prefill42k"] = {k: lp.get(k) for k in ("prompt_tokens", "ttft_s", "prefill_tok_s",
                                                              "completion_tokens", "decode_tok_s", "text_head")}
            except Exception as e:
                m["np2_prefill42k"] = {"error": str(e)[:200], "stderr": bs.stderr[-400:]}
            m["np2_bench_accept"] = acceptance(lf, off)
            m["np2_vram_gib_after"] = gpu0_vram_gib()
            log("np2 prefill42k: %s" % json.dumps(m["np2_prefill42k"]))
        else:
            m["np2_load_s"] = "START_FAIL"
        save(data)

        # np=3: concurrency ladder
        lf = os.path.join(OUT, "%s_np3.log" % name)
        m["np3_load_s"] = launch(model, 3, lf)
        if m["np3_load_s"]:
            m["np3_vram_gib_idle"] = gpu0_vram_gib()
            off = fsize(lf)
            ct = sh(["python3", os.path.join(E, "conc_test.py")])
            ladder = {}
            for mm in re.finditer(r"n=(\d+)\s+per-stream:\s+([0-9.]+).*?AGGREGATE:\s+([0-9.]+).*?ok=(\d+)/(\d+)", ct.stdout):
                ladder["n%s" % mm.group(1)] = {"per_stream": float(mm.group(2)), "aggregate": float(mm.group(3)),
                                               "ok": "%s/%s" % (mm.group(4), mm.group(5))}
            m["np3_conc"] = ladder or {"raw": ct.stdout[-600:], "stderr": ct.stderr[-300:]}
            m["np3_accept"] = acceptance(lf, off)
            m["np3_vram_gib_after"] = gpu0_vram_gib()
            log("np3 conc: %s accept=%s" % (json.dumps(m["np3_conc"]), m["np3_accept"]))
        else:
            m["np3_load_s"] = "START_FAIL"
        save(data)

        # np=4: VRAM + sanity only (design goal)
        lf = os.path.join(OUT, "%s_np4.log" % name)
        m["np4_load_s"] = launch(model, 4, lf)
        m["np4_vram_gib"] = gpu0_vram_gib() if m["np4_load_s"] else "START_FAIL"
        m["np4_sanity"] = sanity() if m["np4_load_s"] else None
        log("np4 load=%ss vram=%s sanity=%s" % (m["np4_load_s"], m["np4_vram_gib"], m["np4_sanity"]))

        pa = [(m.get("np2_" + c) or {}).get("accept", {}).get("pooled") for c in ("predictable", "creative")]
        pa = [x for x in pa if x is not None]
        m["disqualified_accept_lt_0.50"] = (min(pa) < 0.50) if pa else None
        save(data)

    # --- summary ---
    log("================ SUMMARY ================")
    for name, m in data["models"].items():
        p, c = m.get("np2_predictable") or {}, m.get("np2_creative") or {}
        pf = m.get("np2_prefill42k") or {}
        cc = m.get("np3_conc") or {}
        log("%-18s vram np1/2/3/4=%s/%s/%s/%s | pred %s acc %s len %s | creat %s acc %s len %s | "
            "prefill42k %s decode %s | conc n1 %s n2 %s/%s n3 %s/%s | DQ=%s" % (
                name, m.get("np1_vram_gib"), m.get("np2_vram_gib_after"), m.get("np3_vram_gib_after"), m.get("np4_vram_gib"),
                p.get("tok_s"), (p.get("accept") or {}).get("pooled"), (p.get("accept") or {}).get("mean_len"),
                c.get("tok_s"), (c.get("accept") or {}).get("pooled"), (c.get("accept") or {}).get("mean_len"),
                pf.get("prefill_tok_s"), pf.get("decode_tok_s"),
                (cc.get("n1") or {}).get("per_stream"), (cc.get("n2") or {}).get("per_stream"), (cc.get("n2") or {}).get("aggregate"),
                (cc.get("n3") or {}).get("per_stream"), (cc.get("n3") or {}).get("aggregate"),
                m.get("disqualified_accept_lt_0.50")))
    save(data)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("CRASH: %r" % e)
        raise
    finally:
        up = restore_production()
        log("production qwen38.service restored: health after %ss" % up)
        try:
            d = json.load(open(RESULTS))
            d["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
            d["production_restored_s"] = up
            d["production_cmdline"] = sh("pgrep -af '[l]lama-server'").stdout.strip()
            save(d)
        except Exception:
            pass
        log("PHASE1 DONE")

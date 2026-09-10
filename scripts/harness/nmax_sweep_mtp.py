#!/usr/bin/env python3
"""Full --spec-draft-n-max sweep at the intended production config.

Config under test: patched PR#27342 binary, heretic Q4_K_S + mmproj, -c 262144, q8_0 KV,
DFlash2 Q8_0 drafter pinned with -devd Vulkan0, and **-np 2 -kvu** (the multisession
setup the owner wants, not the -np 1 currently running).

Two changes from earlier sweeps:
  * max_tokens 2048 instead of 512 -- longer runs dilute startup noise, which matters
    because an identical config measured 81.86 and 77.74 on two runs (~5% variance).
  * n_max swept 1..8 rather than spot-checked. Upstream #27117 reports the optimum moves
    with concurrency (n_max 1 recovers acceptance at -np 16), so the -np 1 optimum of 4
    may not hold at -np 2.

Finishes with a concurrency probe at the winning n_max: two simultaneous requests, to see
whether acceptance degrades the way #27117 describes.
"""
from __future__ import annotations
import json, os, re, subprocess, threading, time, urllib.request

BIN = os.path.expanduser("~/projects/llama.cpp-master/build-vulkan/bin/llama-server")
TARGET = "/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf"
MMPROJ = "/mnt/models/llm/qwen3.8-27b-instruct-unsloth/mmproj-BF16.gguf"
DRAFT = "/mnt/models/llm/qwen3.8-27b-dflash2-incoai/qwen3.8-27b-dflash2-incoai-q8_0.gguf"
OUT = os.path.expanduser("~/benchmarks/engine_matrix/results_nmax_sweep_mtp")
os.makedirs(OUT, exist_ok=True)
NL = chr(10)
PORT = 8080
MAXTOK = 2048

COMMON = (" --mmproj " + MMPROJ + " --image-min-tokens 1024"
          " -dev Vulkan0 -ngl 99 -c 262144 -ctk q8_0 -ctv q8_0 -fa on --ctx-checkpoints 2"
          " -np 2 -kvu --jinja --reasoning-format deepseek"
          " --host 0.0.0.0 --port " + str(PORT) + " --log-timestamps --log-prefix")

PREDICTABLE = ("Write a comprehensive, well-commented Python module implementing: an LRU cache with "
 "TTL expiry, a thread-safe bounded queue, a retry decorator with exponential backoff and jitter, "
 "a simple dependency-injection container, and a binary search tree with insert/delete/traverse. "
 "Include type hints, docstrings, and a demonstration section exercising every component.")
CREATIVE = ("Write the opening chapter of a surrealist novel. A lighthouse keeper on a distant coast "
 "wakes one morning to discover the ocean has vanished overnight, replaced by an endless field of tall "
 "grey grass. Strange figures move through the grass at dusk. Develop the keeper's history, the "
 "texture of the place, and the first encounter. Begin in medias res, unsettling and dreamlike.")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


def health():
    try:
        urllib.request.urlopen("http://127.0.0.1:" + str(PORT) + "/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


def launch(n_max, log):
    sh("tmux kill-session -t prodrun")
    sh("tmux kill-session -t nmaxrun")
    time.sleep(7)
    spec = ("--spec-type draft-mtp "
            "--spec-draft-n-max " + str(n_max)) if n_max > 0 else ""
    cmd = BIN + " -m " + TARGET + COMMON + " " + spec
    sh("tmux new-session -d -s nmaxrun " + chr(34) + cmd + " > " + log + " 2>&1" + chr(34))
    for _ in range(70):
        if health():
            return True
        time.sleep(3)
    return False


def gen(prompt, box=None, idx=None):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "temperature": 0, "seed": 42, "stream": True}
    req = urllib.request.Request("http://127.0.0.1:" + str(PORT) + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    tf = tl = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
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
            ch = d.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    now = time.time()
                    if tf is None:
                        tf = now
                    tl = now
                    n += 1
    res = {"tok": n, "tok_s": round(n / (tl - tf), 2) if (tf and tl and tl > tf and n > 1) else None,
           "ttft": round(tf - t0, 2) if tf else None, "wall": round(time.time() - t0, 2)}
    if box is not None:
        box[idx] = res
    return res


def accept(log):
    o = sh("grep -a 'draft acceptance' " + log + " | tail -3").stdout.strip()
    a = re.findall(r"draft acceptance = ([0-9.]+)", o)
    l = re.findall(r"mean len = +([0-9.]+)", o)
    return (round(sum(map(float, a)) / len(a), 3) if a else "n/a",
            round(sum(map(float, l)) / len(l), 2) if l else "n/a")


rows = []
for n_max in [2, 3, 4]:
    label = "nospec" if n_max == 0 else ("n" + str(n_max))
    log = "/tmp/nmax_" + label + ".log"
    print("=== " + label + " ===", flush=True)
    if not launch(n_max, log):
        err = sh("grep -aiE 'error|failed|assert|memory' " + log + " | tail -4").stdout.strip()
        print("   START_FAIL :: " + err[:300], flush=True)
        rows.append({"n_max": n_max, "status": "START_FAIL"})
        continue
    gen("Say hello in one sentence.")
    row = {"n_max": n_max, "status": "ok"}
    for cls, p in (("predictable", PREDICTABLE), ("creative", CREATIVE)):
        r = gen(p)
        row[cls] = r["tok_s"]
        row[cls + "_tok"] = r["tok"]
    row["accept"], row["mean_len"] = accept(log) if n_max > 0 else ("n/a", "n/a")
    print("   " + json.dumps(row), flush=True)
    rows.append(row)
    open(os.path.join(OUT, "results.json"), "w").write(json.dumps(rows, indent=2) + NL)

ok = [r for r in rows if r.get("status") == "ok" and r.get("predictable")]
best = max(ok, key=lambda x: x["predictable"]) if ok else None
print("", flush=True)
print("=== SWEEP DONE. best predictable: " + json.dumps(best) + " ===", flush=True)

# concurrency probe at the winner -- #27117 says acceptance degrades with active slots
if best and best["n_max"] > 0:
    print("", flush=True)
    print("=== CONCURRENCY PROBE at n_max=" + str(best["n_max"]) + " (2 simultaneous) ===", flush=True)
    log = "/tmp/nmax_conc.log"
    if launch(best["n_max"], log):
        gen("Say hello in one sentence.")
        box = {}
        th = [threading.Thread(target=gen, args=(PREDICTABLE, box, 0)),
              threading.Thread(target=gen, args=(CREATIVE, box, 1))]
        t0 = time.time()
        for t in th:
            t.start()
            time.sleep(0.4)
        for t in th:
            t.join()
        a, l = accept(log)
        print("   A: " + json.dumps(box.get(0)), flush=True)
        print("   B: " + json.dumps(box.get(1)), flush=True)
        print("   both wall: " + str(round(time.time() - t0, 2)) +
              "  accept=" + str(a) + "  mean_len=" + str(l), flush=True)
        print("   (single-stream at this n_max was " + str(best["predictable"]) +
              " / " + str(best["creative"]) + ")", flush=True)
        rows.append({"concurrency_probe": {"A": box.get(0), "B": box.get(1),
                                           "accept": a, "mean_len": l}})
        open(os.path.join(OUT, "results.json"), "w").write(json.dumps(rows, indent=2) + NL)

print("", flush=True)
print("=== leaving server on the BEST config ===", flush=True)
if best:
    launch(best["n_max"], "/tmp/prod_dflash2_np2.log")
    cmd = BIN + " -m " + TARGET + COMMON + (
        " --spec-type draft-dflash -md " + DRAFT + " -devd Vulkan0 --spec-draft-n-max " +
        str(best["n_max"]))
    open(os.path.expanduser("~/benchmarks/engine_matrix/restore/llamaserver_cmd_dflash2_np2.txt"),
         "w").write(cmd + NL)
print("serving: " + str(health()), flush=True)
print("", flush=True)
for r in rows:
    print("   " + json.dumps(r), flush=True)

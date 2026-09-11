#!/usr/bin/env python3
"""Prompt-lookup / n-gram speculative decoding on heretic Q4_K_S: sanity gate, then a small matrix.

Question (from a Gemini chat, fact-checked): does llama.cpp's context n-gram lookup ("PLD") help on
high-overlap agent-style work, and can it chain in front of MTP as a fallback? On this card decode is
memory-bandwidth bound with lots of spare compute, so verifying a wrong draft should be nearly free.

SANITY GATE first: run `ngram-mod` ALONE on a copy-verbatim prompt (maximal overlap). If it doesn't
accept drafts there, it is broken/misconfigured and the matrix is pointless -> abort before the matrix.

MATRIX (only if the gate passes):
  configs : none | draft-mtp (production) | ngram-mod | ngram-mod,draft-mtp (n-gram first, MTP fallback)
  prompts : copy (max overlap) | code-edit (high overlap) | prose (~zero overlap)
  all forced to exactly GEN tokens (ignore_eos), temp 0, seed 42 -> decode is comparable AND, since
  speculation is lossless at greedy, every config must produce IDENTICAL text (a correctness check).
  np=1 for the full matrix; an np=2 spot-check on code-edit for draft-mtp vs ngram-mod,draft-mtp
  (the real question is whether a n-gram win survives concurrency).

Server = production line (systemd qwen38.service) minus the spec flags, bound to 127.0.0.1:8085.
Production is stopped for the run and restored in `finally`.
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BIN = os.path.expanduser("~/projects/llama.cpp-master/build-vulkan/bin/llama-server")
MODEL = "/mnt/models/llm/qwen3.8-27b-heretic-trohrbaugh/qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf"
MMPROJ = "/mnt/models/llm/qwen3.8-27b-instruct-unsloth/mmproj-BF16.gguf"
OUT = os.path.expanduser("~/benchmarks/core19/pld_test")
BASE = "http://127.0.0.1:8085"
GEN = 512
os.makedirs(OUT, exist_ok=True)

BASE_ARGS = [BIN, "-m", MODEL, "--mmproj", MMPROJ, "--image-min-tokens", "1024", "-dev", "Vulkan0",
             "-ngl", "99", "-c", "262144", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--ctx-checkpoints", "2",
             "-np", "2", "-kvu", "--jinja", "--reasoning-format", "deepseek", "-cram", "12288",
             "--host", "127.0.0.1", "--port", "8085", "--log-timestamps", "--log-prefix"]

SPEC = {
    "none": [],
    "draft-mtp": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "ngram-mod": ["--spec-type", "ngram-mod"],
    "ngram+mtp": ["--spec-type", "ngram-mod,draft-mtp", "--spec-draft-n-max", "2"],
}

# ------------------------------------------------------------------ prompts
SRC_TEXT = (
    "The maintenance log for pump station 7 records every inspection since commissioning. Each entry lists the "
    "inspector, the date, the measured discharge pressure in kilopascals, the vibration amplitude in millimetres "
    "per second, and any corrective action taken. Entries flagged critical require a follow-up within seventy-two "
    "hours, and the supervisor must countersign the closure. The seal replacement in March followed a slow rise in "
    "vibration that the trend analysis had predicted three weeks earlier. The bearing temperature stayed within "
    "tolerance throughout, which ruled out lubrication starvation as the cause. The revised schedule moves the next "
    "full teardown forward by one quarter to align with the shutdown window, reducing the number of separate "
    "isolation events and the associated permit overhead."
)
CODE_SRC = '''import json
from dataclasses import dataclass, field


@dataclass
class Account:
    """A ledger account with a running balance and an audit trail."""
    name: str
    balance: float = 0.0
    history: list = field(default_factory=list)

    def deposit(self, amount, memo=""):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount
        self.history.append({"op": "deposit", "amount": amount, "memo": memo, "balance": self.balance})
        return self.balance

    def withdraw(self, amount, memo=""):
        if amount <= 0:
            raise ValueError("withdraw must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        self.history.append({"op": "withdraw", "amount": amount, "memo": memo, "balance": self.balance})
        return self.balance

    def statement(self):
        lines = [f"Account: {self.name}"]
        for h in self.history:
            lines.append(f"  {h['op']:<9} {h['amount']:>10.2f}  -> {h['balance']:>10.2f}  {h['memo']}")
        lines.append(f"  {'final':<9} {'':>10}     {self.balance:>10.2f}")
        return "\\n".join(lines)

    def to_json(self):
        return json.dumps({"name": self.name, "balance": self.balance, "history": self.history}, indent=2)
'''

PROMPTS = {
    "copy": "Reproduce the following text exactly, word for word, with no commentary:\n\n" + SRC_TEXT,
    "code": ("Here is a Python module. Add a `transfer(self, other, amount, memo=\"\")` method to the `Account` "
             "class that withdraws from self and deposits into `other`, recording both. Change nothing else. "
             "Output the COMPLETE modified module in one code block.\n\n```python\n" + CODE_SRC + "```"),
    "prose": ("Write an original 400-word piece of vivid descriptive prose about the first hour after a snowfall "
              "in an empty mountain town. No lists, no headings, just flowing paragraphs."),
}
OVERLAP = {"copy": "max", "code": "high", "prose": "~zero"}


def gpu0_mib():
    for d in os.listdir("/sys/class/drm"):
        p = f"/sys/class/drm/{d}/device"
        try:
            if int(open(p + "/mem_info_vram_total").read()) > 32_000_000_000:
                return int(open(p + "/mem_info_vram_used").read()) // 1048576
        except Exception:
            pass
    return -1


def stop_production():
    subprocess.run(["systemctl", "--user", "stop", "qwen38.service"], check=False)
    for _ in range(60):
        if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode != 0 and gpu0_mib() < 1500:
            return True
        time.sleep(2)
    return False


def restore_production():
    subprocess.run(["tmux", "kill-session", "-t", "pldsrv"], capture_output=True)
    subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    time.sleep(4)
    subprocess.run(["pkill", "-KILL", "-f", "llama-server .*--port 8085"], capture_output=True)
    subprocess.run(["systemctl", "--user", "start", "qwen38.service"], check=False)
    for _ in range(90):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=4).status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def launch(cfg):
    subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", "llama-server .*--port 8085"], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    time.sleep(2)
    log = open(os.path.join(OUT, f"server_{cfg}.log"), "ab")
    p = subprocess.Popen(BASE_ARGS + SPEC[cfg], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for i in range(150):
        try:
            if urllib.request.urlopen(BASE + "/health", timeout=4).status == 200:
                return p, i * 2, gpu0_mib()
        except Exception:
            pass
        if p.poll() is not None:
            return None, None, None
        time.sleep(2)
    return None, None, None


def call(prompt, gen=GEN, timeout=1200):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": gen,
            "ignore_eos": True, "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = {}; timings = {}; text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            pl = line[5:].strip()
            if pl == "[DONE]":
                break
            try:
                ch = json.loads(pl)
            except Exception:
                continue
            usage = ch.get("usage") or usage
            timings = ch.get("timings") or timings
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                t = d.get("content") or d.get("reasoning_content")
                if t:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(t)
    wall = time.time() - t0
    ct = usage.get("completion_tokens")
    acc = round(timings["draft_n_accepted"] / timings["draft_n"], 3) if timings.get("draft_n") else None
    return {"decode_tok_s": round(timings.get("predicted_per_second") or 0, 2),
            "prefill_tok_s": round(timings.get("prompt_per_second") or 0, 1),
            "ttft_s": round(ttft, 3) if ttft else None, "completion_tokens": ct,
            "draft_n": timings.get("draft_n"), "draft_accepted": timings.get("draft_n_accepted"),
            "accept": acc, "wall_s": round(wall, 2), "text": "".join(text)}


def conc(prompt, n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.time()
        rows = list(ex.map(lambda _: call(prompt), range(n)))
        wall = time.time() - t0
    dec = [r["decode_tok_s"] for r in rows]
    accs = [r["accept"] for r in rows if r["accept"] is not None]
    return {"n": n, "per_stream_decode_mean": round(statistics.mean(dec), 2),
            "aggregate_tok_s": round(sum(r["completion_tokens"] for r in rows) / wall, 2),
            "accept_mean": round(statistics.mean(accs), 3) if accs else None}


def main():
    res = {"started": time.strftime("%F %T"), "model": os.path.basename(MODEL), "gen_tokens": GEN}
    save = lambda: open(os.path.join(OUT, "pld_results.json"), "w").write(json.dumps(res, indent=2) + "\n")

    print("stopping production...", flush=True)
    if not stop_production():
        print("ABORT: production did not free the GPU"); return
    print(f"  GPU0 now {gpu0_mib()} MiB", flush=True)

    # -------- SANITY GATE: ngram-mod alone on the copy task must accept drafts
    print("\n==== SANITY GATE: ngram-mod alone on a copy-verbatim prompt ====", flush=True)
    proc, load_s, mib = launch("ngram-mod")
    if not proc:
        print("ABORT: ngram-mod server failed to start"); return
    print(f"  server up ~{load_s}s, GPU0 {mib} MiB", flush=True)
    call(PROMPTS["copy"], gen=32)  # warm
    gate = call(PROMPTS["copy"])
    res["sanity_gate"] = {"config": "ngram-mod", "prompt": "copy", **{k: gate[k] for k in
                          ("decode_tok_s", "accept", "draft_n", "draft_accepted", "completion_tokens")}}
    save()
    print(f"  copy task: decode {gate['decode_tok_s']} tok/s, accept {gate['accept']} "
          f"({gate['draft_accepted']}/{gate['draft_n']} draft tokens)", flush=True)
    subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    if not gate["accept"] or gate["accept"] < 0.4:
        res["gate_passed"] = False
        print(f"\nGATE FAILED: n-gram acceptance {gate['accept']} on a copy task. PLD is not working; "
              "skipping the matrix.", flush=True)
        save(); return
    res["gate_passed"] = True
    print(f"\nGATE PASSED (accept {gate['accept']} >= 0.4).", flush=True)

    # quick contrast: no-spec baseline on the SAME copy prompt, so "faster than nothing" is concrete
    proc2, l2, m2 = launch("none")
    if proc2:
        call(PROMPTS["copy"], gen=32)
        base = call(PROMPTS["copy"])
        res["sanity_gate"]["nospec_copy_decode_tok_s"] = base["decode_tok_s"]
        res["sanity_gate"]["speedup_vs_nospec"] = round(gate["decode_tok_s"] / base["decode_tok_s"], 2)
        print(f"  no-spec copy decode {base['decode_tok_s']} tok/s -> n-gram is "
              f"{res['sanity_gate']['speedup_vs_nospec']}x faster on this copy task", flush=True)
        subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
    save()
    if os.environ.get("SANITY_ONLY"):
        print("\nSANITY_ONLY set -> stopping before the matrix.", flush=True)
        return
    print("Running the matrix.\n", flush=True)

    # -------- MATRIX (np=1), all four configs x three prompts
    res["matrix"] = {}
    for cfg in ("none", "draft-mtp", "ngram-mod", "ngram+mtp"):
        print(f"==== config {cfg} ====", flush=True)
        proc, load_s, mib = launch(cfg)
        if not proc:
            print(f"  server failed for {cfg}"); res["matrix"][cfg] = {"error": "start failed"}; save(); continue
        call(PROMPTS["copy"], gen=32)  # warm
        res["matrix"][cfg] = {"gpu0_mib": mib, "prompts": {}}
        for pk in ("copy", "code", "prose"):
            r = call(PROMPTS[pk])
            res["matrix"][cfg]["prompts"][pk] = r
            print(f"  {pk:<5} ({OVERLAP[pk]:>5} overlap): decode {r['decode_tok_s']:>6} tok/s  "
                  f"accept {r['accept']}  ttft {r['ttft_s']}s  ct {r['completion_tokens']}", flush=True)
            save()
        subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)

    # -------- output identity check (greedy speculation is lossless -> identical text)
    print("\n==== output identity (should be identical across configs at temp 0) ====", flush=True)
    res["identity"] = {}
    for pk in ("copy", "code", "prose"):
        texts = {c: res["matrix"][c]["prompts"][pk]["text"] for c in res["matrix"] if "prompts" in res["matrix"][c]}
        ref = texts.get("none")
        res["identity"][pk] = {c: (t == ref) for c, t in texts.items()}
        print(f"  {pk}: {res['identity'][pk]}", flush=True)
    save()

    if os.environ.get("SKIP_NP2"):
        res["finished"] = time.strftime("%F %T"); save()
        print("\nMATRIX DONE (np2 skipped) ->", os.path.join(OUT, "pld_results.json"), flush=True); return
    # -------- np=2 spot check on code-edit: does a n-gram win survive concurrency?
    print("\n==== np=2 spot check on code-edit ====", flush=True)
    res["np2_code"] = {}
    for cfg in ("draft-mtp", "ngram+mtp"):
        proc, load_s, mib = launch(cfg)
        if not proc:
            res["np2_code"][cfg] = {"error": "start failed"}; continue
        call(PROMPTS["code"], gen=32)
        res["np2_code"][cfg] = conc(PROMPTS["code"], 2)
        print(f"  {cfg}: {res['np2_code'][cfg]}", flush=True)
        subprocess.run(["pkill", "-TERM", "-f", "llama-server .*--port 8085"], capture_output=True)
        save()

    res["finished"] = time.strftime("%F %T"); save()
    print("\nPLD TEST DONE ->", os.path.join(OUT, "pld_results.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("CRASH:", repr(e))
        raise
    finally:
        print("\nrestoring production (qwen38.service)...", flush=True)
        ok = restore_production()
        print("production healthy on 8080:", ok, "| GPU0", gpu0_mib(), "MiB", flush=True)

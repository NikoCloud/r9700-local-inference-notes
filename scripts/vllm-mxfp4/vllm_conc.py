#!/usr/bin/env python3
"""Concurrency ladder for the radiance MXFP4-PARO serve.

Finds (a) where aggregate throughput plateaus and (b) where it degenerates.

Method notes:
  * Each stream gets a UNIQUE random preamble of ~2500 tokens. The server's attention
    block size is 1568 tokens, so >1 block of uniqueness defeats prefix caching --
    otherwise concurrent identical prompts all hit cache and prefill is fiction.
  * Bulk context is real prose, and the task is a real code edit, so draft acceptance
    stays realistic (random filler + ignore_eos makes the model loop, which inflates
    speculative decode ~5x).
  * Per-core CPU is sampled from /proc/stat deltas across each rung, to test whether
    the SMT siblings or a couple of physical cores bind at high concurrency.
  * Degeneration probe: N=8 at a depth whose N*depth exceeds the KV pool, to see
    preemption / queueing rather than graceful sharing.

Usage: vllm_conc.py <tag> [port]
"""
import os
import json, random, string, sys, threading, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "conc"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen3.8-PARO"
OUTDIR = os.path.expanduser("~/benchmarks")
GEN = 512

PARA = (
    "The scheduler maintains a queue of pending requests and admits them only when the "
    "block manager can guarantee enough physical pages for the sequence's projected growth. "
    "Each admitted sequence is assigned a slot, and the slot holds a reference to its own "
    "page table. When a sequence terminates the pages are returned to the free pool, "
    "although the page table itself is retained so that a follow-up turn on the same "
    "conversation can be resumed without a full re-prefill. Fragmentation is managed by "
    "keeping every block the same size, which trades a small amount of internal waste for "
    "the guarantee that any free block fits any request. "
)

CODE = '''
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda p: p[0])
    out = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
'''

TASK = ("Rewrite the merge_intervals function shown above so that it validates its input "
        "and raises ValueError on malformed intervals. Output the full function, then "
        "explain each validation you added.")

UNIQUE_TOKENS = 2500   # > 1 attention block (1568) so prefix caching cannot match


def unique_preamble(stream_id):
    r = random.Random((stream_id + 1) * 7919 + int(time.time()))
    n = int(UNIQUE_TOKENS / 3.4)  # random words are ~3.4 tok each
    return " ".join("".join(r.choices(string.ascii_lowercase, k=r.randint(3, 9))) for _ in range(n))


def prose_filler(target_tokens, tpw):
    need = max(20, int(target_tokens / tpw))
    unit = PARA.split()
    return " ".join((unit * (need // len(unit) + 1))[:need])


def build_prompt(stream_id, depth, tpw):
    body = depth - UNIQUE_TOKENS
    parts = [unique_preamble(stream_id)]
    if body > 0:
        parts.append(prose_filler(body, tpw))
    parts.append(CODE)
    parts.append(TASK)
    return "\n\n".join(parts)


def call(prompt):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": GEN, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; text = []
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                d = json.loads(p)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    delta = ch[0].get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        text.append(piece)
    except Exception as e:
        return {"error": repr(e)[:200]}
    total = time.time() - t0
    if ttft is None or not usage:
        return {"error": "no output"}
    ct = usage["completion_tokens"]
    tail = "".join(text)[-300:].split()
    return {
        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": ct,
        "ttft_s": round(ttft, 3), "total_s": round(total, 3),
        "decode_tok_s": round(ct / (total - ttft), 2) if total > ttft else 0.0,
        "tail_distinct": round(len(set(tail)) / max(1, len(tail)), 2),
    }


def cpu_snapshot():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            out[int(f[0][3:])] = (sum(int(x) for x in f[1:]), int(f[4]))
    return out


def cpu_busy(a, b):
    """Per-core busy % between two snapshots."""
    res = {}
    for c in a:
        dt = b[c][0] - a[c][0]
        di = b[c][1] - a[c][1]
        res[c] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0
    return res


def gpu_power():
    import subprocess
    try:
        o = subprocess.run(["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=15).stdout
        for l in o.splitlines():
            if "GPU[0]" in l and "Average Graphics Package Power" in l:
                return float(l.rsplit(":", 1)[1].strip())
    except Exception:
        pass
    return None


def rung(n, depth, tpw):
    prompts = [build_prompt(i, depth, tpw) for i in range(n)]
    results = [None] * n
    c0 = cpu_snapshot()
    t0 = time.time()

    def worker(i):
        results[i] = call(prompts[i])

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    time.sleep(min(20, 5 + n))          # sample power mid-flight
    pw = gpu_power()
    for t in ths:
        t.join()
    wall = time.time() - t0
    c1 = cpu_snapshot()
    busy = cpu_busy(c0, c1)

    ok = [r for r in results if r and "error" not in r]
    errs = [r for r in results if r and "error" in r]
    if not ok:
        print(f"  N={n:<2} depth={depth:<6} ALL FAILED: {errs[:1]}", flush=True)
        return {"n": n, "depth": depth, "errors": errs}

    per = sum(r["decode_tok_s"] for r in ok) / len(ok)
    agg = sum(r["completion_tokens"] for r in ok) / wall
    ttfts = sorted(r["ttft_s"] for r in ok)
    row = {
        "n": n, "depth": depth, "ok": len(ok), "failed": len(errs),
        "per_stream_decode": round(per, 2), "aggregate_decode": round(agg, 1),
        "ttft_med_s": ttfts[len(ttfts) // 2], "ttft_max_s": ttfts[-1],
        "wall_s": round(wall, 1),
        "prompt_tokens": ok[0]["prompt_tokens"],
        "tail_distinct_min": min(r["tail_distinct"] for r in ok),
        "gpu_power_w": pw,
        "cpu_physical_max": max(busy.get(c, 0) for c in range(12)),
        "cpu_sibling_max": max(busy.get(c, 0) for c in range(12, 24)),
        "cpu_total_busy": round(sum(busy.values()) / 24, 1),
    }
    print(
        f"  N={n:<2} depth={depth:<6} | per-stream {per:>6.2f} | aggregate {agg:>7.1f} "
        f"| ttft med {row['ttft_med_s']:>6.2f}s max {row['ttft_max_s']:>6.2f}s "
        f"| tail {row['tail_distinct_min']:.2f} | {pw}W "
        f"| cpu phys {row['cpu_physical_max']:.0f}% sib {row['cpu_sibling_max']:.0f}% "
        f"| fail {len(errs)}",
        flush=True,
    )
    return row


def main():
    print(f"=== {TAG} concurrency ladder : port {PORT} ===", flush=True)
    # calibrate on real prose (count the words we actually send)
    probe_words = PARA.split() * 3
    probe = " ".join(probe_words)
    r = call(probe + "\n\nReply with the single word: ok")
    tpw = r["prompt_tokens"] / len(probe_words)
    print(f"  calibrated {tpw:.2f} tok/word over {len(probe_words)} words", flush=True)

    rows = []
    print("\n-- plateau ladder @ ~8k depth (8 x 8k = 64k, fits the 103k pool) --", flush=True)
    for n in (1, 2, 4, 8):
        rows.append(rung(n, 8000, tpw))
        time.sleep(5)

    print("\n-- degeneration probe: N=8 at increasing depth --", flush=True)
    for depth in (16000, 32000):
        rows.append(rung(8, depth, tpw))
        time.sleep(5)

    p = f"{OUTDIR}/vllm_conc_{TAG}.json"
    json.dump({"tag": TAG, "gen": GEN, "rows": rows}, open(p, "w"), indent=2)
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prefill/decode by depth with REALISTIC generation tasks.

v1 used random-word filler plus ignore_eos, which made the model emit a repeating
loop -- trivially predictable, so speculative acceptance hit 88-100% and decode was
inflated ~5x. Unspeculated decode and prefill were unaffected (both are
content-independent), but the spec numbers were meaningless.

v2: context filler is real English/code text (so the model stays coherent), and the
generation task is a real instruction. Two tasks bracket the overlap range:
  prose     -- novel explanation, ~zero overlap with context: WORST case for a drafter
  code_edit -- rewrite a function present in context, high but realistic overlap
No ignore_eos; natural stopping, capped at 512.

Usage: vllm_ab2.py <tag> [port] [depths...]
"""
import os
import json, sys, time, urllib.request, urllib.error

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
DEPTHS = [int(x) for x in sys.argv[3:]] or [2000, 32000, 60000]
GEN = 512
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen3.8-PARO"
OUTDIR = os.path.expanduser("~/benchmarks")

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


def filler(target_tokens, tpw):
    """Real prose repeated to approximately target_tokens."""
    need_words = max(20, int(target_tokens / tpw))
    unit = PARA.split()
    reps = need_words // len(unit) + 1
    return " ".join((unit * reps)[:need_words])


def call(messages, max_tokens):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    return ttft, time.time() - t0, usage, "".join(text)


TASKS = {
    "prose": "Ignore the notes above. Explain, in your own words and in detail, why "
    "B-tree nodes are sized to match a disk page. Do not quote the notes.",
    "code_edit": "Rewrite the merge_intervals function shown above so that it validates "
    "its input and raises ValueError on malformed intervals. Output the full function.",
}


def main():
    print(f"=== {TAG} : port {PORT} : depths {DEPTHS} ===", flush=True)
    # calibrate tokens-per-word on the real prose
    probe = " ".join(PARA.split()[:200])
    _, _, usage, _ = call([{"role": "user", "content": probe}], 1)
    tpw = usage["prompt_tokens"] / 200.0
    print(f"  calibrated {tpw:.2f} tok/word (real prose)", flush=True)

    rows = []
    for depth in DEPTHS:
        ctx = filler(depth, tpw)
        for task, instr in TASKS.items():
            content = ctx + "\n\n" + CODE + "\n\n" + instr if task == "code_edit" else ctx + "\n\n" + instr
            ttft, total, usage, out = call([{"role": "user", "content": content}], GEN)
            if ttft is None or not usage:
                print(f"  {depth}/{task}: no output", flush=True)
                continue
            pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
            pre = pt / ttft
            dec = ct / (total - ttft) if total > ttft else 0.0
            # crude degeneration check: distinct/total words in the last 300 chars
            tail = out[-300:].split()
            ratio = len(set(tail)) / max(1, len(tail))
            rows.append(
                {
                    "depth": depth,
                    "task": task,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "prefill_tok_s": round(pre, 1),
                    "decode_tok_s": round(dec, 2),
                    "tail_distinct_ratio": round(ratio, 2),
                }
            )
            print(
                f"  {depth:>6} {task:<10} | prompt {pt:>6} | prefill {pre:>7.1f} "
                f"| decode {dec:>6.2f} | tail_distinct {ratio:.2f}",
                flush=True,
            )
    import os

    os.makedirs(os.path.expanduser("~/benchmarks"), exist_ok=True)
    p = f"{OUTDIR}/vllm_ab2_{TAG}.json"
    json.dump({"tag": TAG, "gen": GEN, "rows": rows}, open(p, "w"), indent=2)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()

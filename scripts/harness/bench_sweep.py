"""Engine/model-agnostic OpenAI-compatible-endpoint benchmark.

Measures, against any /v1/chat/completions server (llama-server or vLLM):
  1. single-shot large-prompt prefill + large-generation decode speed
  2. concurrency/batching scaling curve at a fixed moderate prompt size

Determinism: temperature=0, seed=42. Deliberately does NOT send min_tokens --
llama-server ignores it, but vLLM enforces it, which suppresses EOS and drives
greedy decoding into repetition loops once a prompt runs out of real content
(observed with Gemma, 2026-08-08). Throughput math uses the actual
completion_tokens from the response, so nothing needs forced length to be valid.
"""
import argparse, json, random, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Results land next to this script, NOT in /tmp: /tmp is cleared on reboot, and a
# 2026-08-18 reboot destroyed a full evening of Qwen3.8 run outputs. The numbers
# survived only because they had already been written into PP_FINDINGS.md and
# RECIPES.md. Override with --outdir for one-off throwaway runs.
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results_adhoc"

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Historians have long debated the exact causes of the transition, citing "
    "economic, social, and technological factors that interacted in complex "
    "ways across many decades of gradual change. Engineers building large "
    "distributed systems must reason carefully about failure modes, latency "
    "budgets, and the tradeoffs between consistency and availability. "
)

# Every request must carry a genuinely unique prompt. Identical prompts across
# "concurrent" requests let llama-server's/vLLM's automatic prefix-cache reuse
# (or serialize) the shared KV state, which silently turns a batching-scaling
# measurement into a cache-hit measurement -- caught via a 15x latency cliff
# at concurrency=4 in early smoke testing where all requests were byte-identical.
_WORDPOOL = FILLER.split() * 40
def build_prompt(target_tokens):
    # ~1.3 tokens/word heuristic; good enough for a prefill benchmark, not exact.
    words_needed = int(target_tokens / 1.3)
    pool = _WORDPOOL * (words_needed // len(_WORDPOOL) + 2)
    start = random.randint(0, len(pool) - words_needed - 1)
    body = " ".join(pool[start:start + words_needed])
    unique_tag = f"[request id {random.randint(0, 1_000_000_000)}] "
    return (f"{unique_tag}Here is a long passage. After reading it, write a "
            f"detailed, thorough summary covering every topic mentioned, as "
            f"long as you are able.\n\n{body}\n\nWrite the detailed summary now.")

# "off" matches the manhwa project's proven-working Gemma requests (harmless
# no-op for templates that don't reference the kwarg); "on" leaves thinking
# enabled -- needed for models whose whole point IS their thinking behavior
# (e.g. ThinkingCap, which is benchmarked by its authors with thinking on);
# "default" omits the kwarg entirely and lets the server's own template decide.
THINKING_MODE = "off"

def call(url, model, prompt_tokens, max_tokens, timeout=1800):
    """Streams the response so time-to-first-token (TTFT) can be measured
    directly -- TTFT *is* prefill time (plus a negligible first-token decode
    step), the standard way serving benchmarks isolate PP without needing two
    separate requests and an error-prone subtraction."""
    prompt = build_prompt(prompt_tokens)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # NOT min_tokens: llama-server has no such parameter and silently ignores
        # it (lets the model stop naturally on EOS), but vLLM correctly enforces
        # the OpenAI extended-API field -- meaning vLLM was the only engine ever
        # actually forced past natural completion. For a model/prompt combo with
        # less than max_tokens of genuine content (observed with Gemma, not Qwen,
        # on this filler-repetition prompt), that forcing suppresses EOS and
        # greedy (temp=0) decoding collapses into repetition. Throughput math
        # below already uses actual completion_tokens from the response, never
        # the requested max_tokens, so nothing needs forced length to stay valid.
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if THINKING_MODE == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif THINKING_MODE == "on":
        body["chat_template_kwargs"] = {"enable_thinking": True}
    # "default": omit the kwarg entirely
    t0 = time.time()
    ttft = None
    text_chunks = []
    usage = {}
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        if ttft is None:
                            ttft = time.time() - t0
                        text_chunks.append(content)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        wall = time.time() - t0
        text = "".join(text_chunks).strip()
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        decode_s = (wall - ttft) if ttft is not None else None
        # ct-1: TTFT already accounts for producing the first token
        prefill_tok_s = round(pt / ttft, 2) if (ttft and pt) else None
        decode_tok_s = round((ct - 1) / decode_s, 2) if (decode_s and ct and decode_s > 0) else None
        return {"ok": True, "wall_s": round(wall, 3), "ttft_s": round(ttft, 3) if ttft is not None else None,
                "prompt_tokens": pt, "completion_tokens": ct,
                "prefill_tok_s": prefill_tok_s, "decode_tok_s": decode_tok_s,
                "error": None,
                # Qwen3.6-family models are known to degenerate (repetition/garbage)
                # below ~128k unquantized KV -- capture head+tail so coherence can
                # be checked after the fact, not just raw throughput numbers.
                "text_head": text[:300], "text_tail": text[-300:] if len(text) > 300 else ""}
    except Exception as e:
        wall = time.time() - t0
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
        return {"ok": False, "wall_s": round(wall, 3), "ttft_s": None,
                "prompt_tokens": None, "completion_tokens": None,
                "prefill_tok_s": None, "decode_tok_s": None,
                "error": f"{type(e).__name__}: {str(e)[:200]} {detail}",
                "text_head": "", "text_tail": ""}

def single_shot(url, model, prompt_tokens, max_tokens):
    print(f"[single-shot] prompt~{prompt_tokens}tok max_tokens={max_tokens}", flush=True)
    r = call(url, model, prompt_tokens, max_tokens)
    if not r["ok"]:
        print(f"  FAILED: {r['error']}", flush=True)
        return r
    pt, ct, wall = r["prompt_tokens"], r["completion_tokens"], r["wall_s"]
    print(f"  wall={wall}s prompt_tok={pt} completion_tok={ct} "
          f"ttft={r.get('ttft_s')}s prefill_tok/s={r.get('prefill_tok_s')} "
          f"decode_tok/s={r.get('decode_tok_s')}", flush=True)
    print(f"  text_tail: {r.get('text_tail','')[:150]!r}", flush=True)
    return r

def sweep(url, model, prompt_tokens, max_tokens, levels, budget_s=600, decline_ratio=0.6):
    results = []
    best_total_tok_s = 0.0
    sweep_start = time.time()
    for n in levels:
        elapsed = time.time() - sweep_start
        if elapsed > budget_s and results:
            print(f"  sweep time budget ({budget_s}s) exceeded, stopping before n={n}", flush=True)
            break
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(call, url, model, prompt_tokens, max_tokens, timeout=600) for _ in range(n)]
            rows = [f.result() for f in as_completed(futs)]
        wall = time.time() - t0
        ok_rows = [r for r in rows if r["ok"]]
        fail = n - len(ok_rows)
        sum_p = sum(r["prompt_tokens"] or 0 for r in ok_rows)
        sum_c = sum(r["completion_tokens"] or 0 for r in ok_rows)
        lat = sorted(r["wall_s"] for r in ok_rows) if ok_rows else [0]
        ttfts = sorted(r["ttft_s"] for r in ok_rows if r.get("ttft_s") is not None)
        total_tok_s = round((sum_p + sum_c) / wall, 2) if wall else 0
        row = {
            "concurrency": n, "wall_s": round(wall, 3), "ok": len(ok_rows), "fail": fail,
            "sum_prompt_tokens": sum_p, "sum_completion_tokens": sum_c,
            "agg_prompt_tok_s": round(sum_p / wall, 2) if wall else None,
            "agg_completion_tok_s": round(sum_c / wall, 2) if wall else None,
            "agg_total_tok_s": total_tok_s,
            "latency_s": {"min": lat[0], "p50": lat[len(lat)//2], "mean": round(sum(lat)/len(lat), 3), "max": lat[-1]},
            "ttft_s": ({"min": ttfts[0], "p50": ttfts[len(ttfts)//2], "mean": round(sum(ttfts)/len(ttfts), 3), "max": ttfts[-1]}
                       if ttfts else None),
            "errors": [r["error"] for r in rows if not r["ok"]][:3],
            # one representative sample per level, to check coherence under load
            # without bloating the file with every response
            "sample_text_tail": (ok_rows[0]["text_tail"] if ok_rows else ""),
        }
        ttft_p50 = row["ttft_s"]["p50"] if row["ttft_s"] else None
        print(f"  n={n:3d} wall={row['wall_s']:7.2f}s ok={row['ok']} fail={row['fail']} "
              f"comp_tok/s={row['agg_completion_tok_s']} total_tok/s={row['agg_total_tok_s']} "
              f"ttft_p50={ttft_p50}s", flush=True)
        results.append(row)
        if fail == n:
            print("  all requests failed at this level, stopping sweep", flush=True)
            break
        # Stop once throughput has clearly peaked and declined -- the peak/peak-1
        # shape is what matters (matches the manhwa project's own sweep convention),
        # no need to keep climbing concurrency once decline is confirmed.
        if n > levels[0] and total_tok_s < best_total_tok_s * decline_ratio:
            print(f"  throughput declined to {round(total_tok_s/best_total_tok_s,2)}x peak, stopping sweep", flush=True)
            break
        best_total_tok_s = max(best_total_tok_s, total_tok_s)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="config identifier for output filename")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="where <tag>.json is written (default: "
                         f"{DEFAULT_OUTDIR} -- on disk, survives reboots)")
    ap.add_argument("--large-prompt-tokens", type=int, default=16000)
    ap.add_argument("--large-gen-tokens", type=int, default=1536)
    ap.add_argument("--sweep-prompt-tokens", type=int, default=2000)
    ap.add_argument("--sweep-gen-tokens", type=int, default=512)
    ap.add_argument("--sweep-levels", default="1,2,4,8,16,32")
    ap.add_argument("--sweep-budget-s", type=int, default=600)
    ap.add_argument("--thinking", choices=["off", "on", "default"], default="off",
                     help="off=force enable_thinking:false, on=force true, "
                          "default=omit the kwarg and let the server template decide")
    args = ap.parse_args()

    global THINKING_MODE
    THINKING_MODE = args.thinking

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.sweep_levels.split(",")]

    result = {"tag": args.tag, "url": args.url, "model": args.model}

    print(f"=== {args.tag} === large-context prefill+gen ===", flush=True)
    result["large_prompt_prefill_gen"] = single_shot(
        args.url, args.model, args.large_prompt_tokens, args.large_gen_tokens)

    print(f"=== {args.tag} === small-prompt gen (sanity) ===", flush=True)
    result["small_prompt_gen"] = single_shot(args.url, args.model, 200, args.large_gen_tokens)

    print(f"=== {args.tag} === concurrency/batching sweep ===", flush=True)
    result["batching_sweep"] = sweep(
        args.url, args.model, args.sweep_prompt_tokens, args.sweep_gen_tokens, levels,
        budget_s=args.sweep_budget_s)

    outfile = outdir / f"{args.tag}.json"
    outfile.write_text(json.dumps(result, indent=2))
    print(f"wrote {outfile}", flush=True)

if __name__ == "__main__":
    main()

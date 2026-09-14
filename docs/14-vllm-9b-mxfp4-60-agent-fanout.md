# 14 — Dropping the drafter doubled the 60-agent throughput; 134 vision tasks in 31.7 s

> **Date:** 2026-09-14. Qwen3.5-9B heretic (abliterated) build, MXFP4 W4A8 with trained ParoQuant
> rotations, served by vLLM on the **9070 XT (16 GB)** — the second card of the rig in
> [01](01-hardware-and-software.md). fp8 KV, `max-model-len` 65,536, chunk 4096, util 0.98.
> Text numbers at matched power cap, co-measured back-to-back on the same serve instance.
> The MTP-vs-no-spec comparison is the headline; everything else is what it took to get there.

**Short version:** at 60 concurrent streams, turning the MTP drafter **off** produces
**1,915.5 tok/s aggregate vs 862.1 with it on — 2.22×** — and a symmetric per-stream spread
(1.00× vs 5.3×). The crossover is at n ≈ 8–16: below it the drafter wins (single-stream
104.3 vs 77.9, +34%), above it the verification overhead and draft-acceptance variance create
stragglers that double the makespan. The drafter also costs ~2 GiB: the no-spec KV pool is
**273,881 tokens vs 110,649 (2.4×)**. The standing serve for 60-agent fan-out is therefore
no-spec, and a real workload — 134 image descriptions at 60 workers — finished in
**31.7 s wall, 134/134, zero failures**.

## 1. The build: rotations grafted onto an abliterated 9B, MTP head re-attached

The base is a heretic (abliterated) 9B — abliteration, not a finetune, which is what lets
stock-trained rotations transfer cleanly (the 27B precedent is
[13 §1](13-vllm-mxfp4-w4a8-rdna4.md)). Two grafts:

- **ParoQuant rotations** (z-lab's trained Givens rotations, `krot=8`): 200 modules —
  32× MLP down/gate/up, 24× linear-attention qkv/z/out, 8× full-attention q/k/v/o.
  Preflight 200/200, zero missing modules.
- **MTP head**: the abliterated checkpoint had dropped its 15 `mtp.*` tensors; they were
  grafted back from the corresponding Base shards (243.3M params — exactly the
  Base↔heretic tensor delta, confirming abliteration never touched the MTP block).

Build in **243 s**, worst pseudo round-trip **5.71e-07** (fp32 noise; the 27B was 8.47e-07).
Output: 1,575 tensors = 200×5 quantized + 575 passthrough + 15 MTP, **8.55 GB** on disk.
One config fix was needed: the rotation-donor `config.json` declared top-level
`dtype: float16`, which the R4D attention gate rejects — patched to `bfloat16`.

## 2. Four boot gates, each a hard error until fixed

| Gate | Error | Fix |
|---|---|---|
| dtype | R4D: `dtype not supported` (donor config `float16`) | built config → `bfloat16` (all three spots) |
| GQA | R4D: `attention does not support gqa=4` — the kernel is hard-tuned to gqa 6 (the 27B's 24q/4kv); the 9B is 16q/4kv | unified-attention backend instead of R4D |
| compile | `compile_sizes contains 8 which would be padded to 9` (MTP spec padding maps 8→9) | empty `compile_sizes` (what the 27B serve uses) |
| KV pool | `No available memory for the cache blocks` — the graph profiler's *estimated* CUDA-graph reservation (3.65 GiB) exceeded the actual (2.25 GiB) at util 0.92 | disable the estimate, let the profiler measure |

## 3. Tuning: the chunk-size cliff

Boot-by-boot on the same model (each row one boot, re-measured):

| cfg | levers moved | KV pool (fp8 tokens) | n=60 agg / per-stream | tail min |
|---|---|---:|---:|---:|
| v1 | util 0.96, cap 512, chunk 8192, max-len 32,768 | 69,206 | 649.7 / 28.8 | 0.71 |
| v2 | util 0.97, chunk 4096, max-len 65,536 | 93,579 | 742.0 / 31.6 | 0.69 |
| **v3 (final)** | **util 0.98, cap 288, chunk 4096** | **110,649** | **854.6 / 33.3** | **0.73** |
| v4 (rejected) | chunk 2048 | 103,028 | 821.7 / 31.6 | **0.56** |

**The cliff:** chunk 8192→4096 never moved tail-distinct; 4096→2048 collapsed it
0.70→0.56, reproducible bit-for-bit across two runs. The pool delta (110,649 → 103,028)
is inside boot-to-boot noise — the same v3 config booted at 101,199 then 110,649 — so
chunk 2048 bought *nothing* on pool and cost repetition. Mechanism (hypothesis, not
measured): the Gated-DeltaNet chunked scan is numerically sensitive to chunk boundaries,
which drifts token choices over long generations. Rule: **chunk ≥ 4096**; and treat any
same-config boot delta under ~10% as noise.

## 4. MTP vs no-spec: the knee at n ≈ 8–16

Co-measured on the same serve instance at the same power cap (350 W), final config,
800-token streams:

| n | MTP agg | no-spec agg | ratio | MTP per-stream min–max | no-spec min–max |
|--:|--:|--:|:--:|:--:|:--:|
| 1 | 104.3 | 77.9 | 0.75× | — | — |
| 8 | 628.3 | 510.7 | 0.81× | | |
| **16 (crossover)** | 651.8 | 884.2 | **1.36×** | | |
| 30 | 755.4 | 1,378.3 | 1.82× | 25–76 | 46–46 |
| **60** | **862.1** | **1,915.5** | **2.22×** | **14–76 (5.3×)** | **32–32 (1.00×)** |
| 64 (peak) | 888.6 | **1,988.7** | 2.24× | | |
| 90 | 900.2 | 1,517.4 | 1.69× | | |

**The finding:** MTP's per-step verification overhead plus draft-acceptance variance
(acceptance ~57% at mean length 2.1) creates **per-stream stragglers** — 5.3× spread at
n=60 — that blow up the *makespan*: **55.7 s vs 25.1 s** for the same total tokens at
n=60. No-spec is symmetric (1.00×). The drafter wins below the crossover
(single-stream +34%), and 2.22× is its ceiling above it. At n=90 no-spec falls back to
1.5× because the pool saturates (below, §6) — the 2.22× is an n=60 number.

**The KV cost:** the MTP drafter plus its int2 draft head costs ~2 GiB. No-spec pool is
**273,881 fp8 tokens vs 110,649 (2.4×)** — at 60 concurrent, the guaranteed per-agent
context floor rises from 1,844 to **4,564 tokens**.

Depth profile (single stream, no-spec): decode 76.0 / 74.2 / 71.6 tok/s at 8k / 16k / 28k
— roughly 2× the MTP serve's 50.3 / 31.3 / 18.6 at the same depths. Prefill: prose
~10–15k tok/s; **code edits run to 222,971 tok/s at 28k depth** (prompt-lookup/n-gram
caching, see [12](12-prompt-lookup-decoding.md)) — agent workloads are code-heavy, so the
effective prefill for real traffic is the fast one.

One artifact to know: the no-spec harness runs *identical* prompts at temperature 0, so all
streams converge on a common ending and `tail_distinct` collapses artifactually (0.54–0.68).
Re-run with diverse prompts at temperature 0.7: tails 0.74–0.95, coherent output, no
repetition pathology. ([MISTAKES](../MISTAKES.md) for the tail-distinct trap in general.)

## 5. TDP is not a perf lever here — and two power logs read the wrong card

Ladder peaks (aggregate tok/s, n=64) by power cap, co-measured same serve:

| Cap | n=64 | n=90 | 9070 XT true draw |
|--:|--:|--:|---|
| 231 W (floor) | 881.5 | 886.6 | pinned 227–231 W — cap enforced |
| 250 W | 881.7 | 889.6 | — |
| 350 W (operating) | 888.6 | 901.4 | — |
| 374 W (ceiling) | 893.3 | 902.5 | **natural median 283 W, max 382 W** |

Floor→ceiling moves the peak **+1.3% at n=64, +1.8% at n=90 — inside boot noise.** The
workload is bandwidth/latency-bound, not power-bound; the card naturally settles around
283 W under full load and never approaches the cap. A 350 W operating cap is a
**PSU-safety dial for the dual-card rig** (transients, and both cards loaded at once),
not a performance choice.

**Correction:** the first two per-cap power logs read `rocm-smi GPU[0]` — which on this
box is the **9700, not the 9070 XT** (rocm-smi's indices are inverted relative to the
power-tool's). Their power readings (a "331 W peak at 350" and a "250 W run that drew
329–356 W") were the *other card serving chat*, and are void. The throughput numbers from
those runs stand; the power column above comes from runs that sampled the correct
device. [MISTAKES](../MISTAKES.md) has the full entry.

## 6. Context at 60, and what the pool buys

No-spec pool 273,881 tokens, `max-model-len` 65,536/request:

- Even-share floor at 60 concurrent: **4,564 tokens/agent** (vs 1,844 with MTP).
- Per-token KV: 16 KB fp8 (8 full-attention layers × 4 KV heads × 256 dim × 2, fp8).
- **Prefix caching is the multiplier for fan-out:** a second 55,254-token prefill with a
  shared prefix ran **58,350 tok/s vs 3,860 cold — 15×**. A shared system prompt is paid
  once, not 60×.
- Single-agent verified at a 55,254-token prompt (32k depth): decode 16.9 tok/s, tail 0.84.

## 7. Vision: 16/16, and the 60-agent proof

Vision bench (16 rendered images, non-thinking mode, 933–1,579 prompt tokens each):

- Sequential: **16/16 ok**, mean prefill **3,820 tok/s** (image encoding included),
  mean decode **93.1 tok/s**, TTFT 0.11–0.40 s (first request 8.74 s = one-time kernel
  JIT), 0 reasoning characters.
- Concurrent 16: **439.0 tok/s aggregate, 5.30× wall-clock speedup**, worst TTFT 13.66 s.

**The swarm test — the actual 60-agent workload:** describe **134 rendered images** with
**60 concurrent workers** against the standing no-spec serve.

| | |
|---|---|
| Wall clock (makespan) | **31.7 s** |
| Success | **134/134, zero failures** |
| Aggregate | **644.8 tok/s** (20,424 completion tokens) |
| Slot utilization | **93%** (4.23 img/s vs 4.55 theoretical) |
| Per-stream wall | mean 13.24 s · median 13.36 · p95 21.56 · max 23.42 |
| TTFT | mean 2.41 s · max 8.84 s (first-wave queueing) |
| Per-stream prefill | mean 1,479.8 · median 1,145.2 · max 4,238.6 tok/s |
| Per-stream decode | mean 16.45 · median 13.57 · min 8.89 tok/s |
| Output length | median 150 tokens (range 111–229) |

134/60 = 2.23 waves, overlapped by the scheduler rather than run lockstep — the makespan
sits between 2 and 3 sequential waves. The aggregate (644.8) is well under the
synthetic-text n=60 figure (1,915.5) by construction: outputs are 150 tokens, not 800,
so fixed per-stream costs (vision prefill, TTFT) dominate the 13 s wall, and the vision
encoder burns GPU time that never appears in output-token counts. Descriptions were
specific and accurate (setting / characters / action / mood, 3–5 sentences as prompted).

**Decision:** the standing serve for fan-out is **no-spec** — this card's job is 60
concurrent agents, exactly where no-spec wins 2.22× and holds 2.4× the context. MTP stays
available for interactive low-concurrency use, where it wins +34% single-stream.

## 8. Rejected: 8-bpw heads and embeddings

Lowering the quantization of `lm_head`/embeddings was considered for the ~2 GiB pool
upside it would free. Rejected: it measurably hurts output quality in the 27B experience,
and the MXFP4 scheme leaves heads/embeddings full-precision by design (the quantizer's own
passthrough list includes `lm_head`). AMD's own reference quants exclude them. The
no-spec pool (273,881) is the final context headroom.

## Raw data

Data: [data/9b-paro-mxfp4](../data/9b-paro-mxfp4) — ladders, rungs, depth profiles,
vision bench, and the full 134-row swarm result with per-image descriptions.
Harnesses: [scripts/](../scripts/) (`vllm_conc2.py` concurrency ladder, `vllm_ab2.py`
depth profile, `img_bench2.py` vision, `swarm_desc.py` the 60-worker swarm,
`mtp_vs_ns.py` the comparison table).
Two numbers are from serve startup logs rather than committed JSON: the MTP-serve pool
(110,649 — that serve instance was stopped and its log rotated) and the no-spec pool
(273,881 — still live, re-verified 2026-09-14).
Measurement traps hit and corrected along the way (wrong-card power logs, tail-distinct
artifact, chunk-noise): [MISTAKES.md](../MISTAKES.md).

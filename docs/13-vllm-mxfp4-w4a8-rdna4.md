# 13 — MXFP4 W4A8 on RDNA4's fp8 WMMA: the first thing that actually uses them

> **Date:** 2026-09-12. One Radeon AI PRO R9700 (gfx1201), **TP=1**, undervolted with memory OC.
> §1–6 at the **250 W cap**; §6b repeats the key measurements at **330 W** and adds a matched
> llama.cpp reference taken the same hour. Model: `Launch80/Qwen3.8-27B-PARO-MXFP4` served through
> [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (image
> `stilldeadcode/vllm-radiance:0.9.3`, **vLLM 0.27.1**, ROCm 7.14, torch 2.11). Production
> (llama.cpp) was evicted for every run. Nothing here is a production recommendation — this was an
> academic qualification of the fp8 tensor path.

## Why this was worth a day

RDNA4 has native FP8 WMMA in the ISA, and as far as we knew **nothing used it**. Earlier attempts on
this box (the Hephaestus Mojo work) could *issue* the instruction but never got throughput out of it,
and the working assumption became that you needed an Instinct part — which is partly why the
kyuz0/Donato toolboxes present the R9700 as an Instinct card to get past library gates.

This checkpoint claims MXFP4 weights served W4A8 through that path. It checks out.

## 1. The fp8 WMMA claim is real, and native

Verified by reading the kernels, not by trusting the card:

```
20 x __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
 2 x __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
```

192 references to `gfx1201` across the tree. **No `HSA_OVERRIDE`, no gfx942 spoof** — the only
`gfx942` string in the repo is in `prune_rocm.sh`, which *deletes* hipBLASLt's gfx942 kernels to
shrink the image. So the silicon was never the gate; the **libraries** were. Hand-written HIP kernels
bypass hipBLASLt entirely, and the image separately carries an "AITER enablement for gfx12x
(upstream gates it to MI3xx / CDNA)" patch.

**Why MXFP4 rather than int4 is the point.** Their measured VALU ladder for the GEMM inner loop:

| VALU ops per tile-group | throughput |
|---|---|
| 16 (fp16 group-scale FMA + zero-point FMA) — int4 g128 | 180–188 TF/s |
| 8 | 200 TF/s |
| **0 — MXFP4's loop** | **225 TF/s** |

e2m1 values are exactly representable in e4m3 and the e8m0 scale is a power of two that folds at
weight-staging, so the inner loop can be *literally empty* of vector-ALU work. Int4 g128 structurally
cannot be. For scale, [01](01-hardware-and-software.md) puts FP16 at ~95.7 TFLOPS, so 225 TF/s is
**~2.35×** the FP16 figure. That is the number the earlier hand-kernel attempts never reached, and the
ladder says why: *any* dequant arithmetic left in the loop bleeds throughput.

## 2. KV availability — the actual tuning surface

Every row is one R9700, TP=1, MXFP4 weights resident at **18.07 GiB** (20.12 GiB with the drafter
loaded). This table is the headline finding: **on a 32 GB card a 27B leaves 4–8 GiB for KV, and every
feature you enable spends it.**

| # | mode | GPU_UTIL | MAXSEQS | drafter | KV avail | **KV tokens** | tokens/GiB | max conc. |
|---|---|--:|--:|---|--:|--:|--:|---|
| 1 | eager | 0.92 | 8 | — | 7.93 GiB | **225,280** | 28,410 | 6.88× @32k |
| 2 | compiled | 0.92 | 8 | — | 5.97 GiB | **169,301** | 28,359 | 2.58× @65k |
| 3 | compiled | 0.96 | 8 | DFlash2 | 5.13 GiB | **103,268** | 20,130 | 1.58× @65k |
| 4 | compiled | 0.96 | 96 | DFlash2 | 6.38 GiB | **58,709** | 9,202 | 3.58× @16k |
| 5 | compiled | 0.96 | 96 | — | 4.06 GiB | **80,956** | 19,940 | *failed, see §5* |
| 6 | compiled | 0.96 | 84 | — | 4.48 GiB | **89,630** | 20,006 | 5.47× @16k |

> **KV figures carry ~25% run-to-run variance.** Row 6's configuration, re-profiled later at 330 W
> (which cannot change memory), reported **5.74 GiB / 115,651 tokens** instead of 4.48 GiB / 89,630.
> vLLM's memory profiling is a measurement, not a constant. Treat every row as ±25% and re-read the
> startup log rather than trusting a remembered number.

What each step costs:

- **CUDA graphs cost ~56k tokens** (row 1 → 2, same util). vLLM says so itself: `--gpu-memory-utilization
  0.9200 is equivalent to 0.8901 without CUDA graph memory profiling ... increase to 0.9499 to
  maintain the same effective KV cache size`.
- **The DFlash2 drafter costs ~66k tokens** (row 2 → 3) *even after* raising util 0.92 → 0.96. It also
  costs 2.05 GiB of weights. Without it, row 3's config would not have OOM'd at 0.92 at all.
- **Raising MAXSEQS 8 → 96 halves tokens/GiB** (row 3 → 4: 20,130 → 9,202) *with more KV memory
  available*. This is the hybrid-architecture tax — see §5.
- **Per-request context caps at 65,536 on one card** (the launcher's own comment: "A TP=1 serve on one
  32 GB card needs MAXLEN <= 65536"), even though the pool can exceed 200k.

For comparison, production llama.cpp holds a **262,144-token** pool at `-np 4` in 28.0 GiB under `-kvu`
([12](12-prompt-lookup-decoding.md)). Only row 1 — eager, no drafter — gets close, and eager is not a
serving configuration. **Context capacity is where this loses, and it is the argument for the second
R9700.**

## 3. PP (prefill) — where this wins outright

Forced 512-token generation, temp 0, one stream, unique random prompts (no prefix-cache hits).
Config rows 2 and 3 above.

| depth | production llama.cpp | MXFP4 no-spec | MXFP4 + DFlash2 |
|---|--:|--:|--:|
| 2k | 1,045 | **2,702** | 3,135 |
| 32k | 843 | **2,910** | 2,818 |
| 60k | — | **4,760** | 4,494 |
| 128k | 443 | *unreachable at TP=1* | *unreachable* |

- **2.6× at 2k, 3.5× at 32k.** And PP *rises* with depth here (2,702 → 4,760) where llama.cpp
  collapses (1,045 → 443). That inversion is the single most useful property found.
- 4,760 tok/s × ~47 GFLOP/token ≈ **224 TFLOPS**, landing on their 225 TF/s zero-VALU figure almost
  exactly. Independent corroboration of §1 from a completely different measurement.
- Speculation does not help PP and slightly hurts it, as expected.

**GFLOP/token for this model is ~47, not the 54 in [01](01-hardware-and-software.md).** That doc's
figure assumes a dense 27B. From `config.json`: 64 layers × 3 × 5120 × 17408 MLP = 17.1B, 48
linear-attention layers ≈ 4.6B, 16 full-attention layers ≈ 1.2B → **~23.4B GEMM-active**. Embeddings
(1.27B) are a gather, the untied `lm_head` (1.27B) fires once per prefill, and the vision tower
(~0.5B) is unused on text. Cross-check: 23.4B × 4.25 bpw + 6.1 GB bf16 = 18.5 GB predicted vs 19.06 GB
on disk.

## 4. Decode — settled, not the constraint

| condition | tok/s |
|---|--:|
| no spec, 2k / 32k / 60k | 21.0 / 20.1 / 19.5 |
| DFlash2, prose (low overlap), 3.5k / 55k | 61.9 / 58.0 |
| DFlash2, code-edit, 3.5k / 55k | 86.5 / 72.9 |
| *production PLD chain, 2k / 32k / 128k* | *59.3 / 46.4 / 28.2* |

No-spec decode is **worse** than production's 34.4 — the model is 19.4 GB vs heretic Q4_K_S's ~15 GB,
and efficiency against the ~640 GB/s roofline is 64% here vs llama.cpp's 81%. But with the drafter
every realistic configuration clears the project's 18 tok/s usability floor and the 30 tok/s "good"
tier, and **holds 72.9 tok/s at 55k where production's PLD chain is down to 28.2 at 128k**. Decode
degrades gracefully with depth instead of falling off a cliff.

Draft acceptance on realistic tasks: **46.7–63.3%**, mean accepted length 3.3–4.2 of 6, per-position
decay 0.85 / 0.73 / 0.60 / 0.55 / 0.43. The code-edit task legitimately reaches 93.8% when quoting a
function back.

## 5. Four hard RDNA4 / architecture limits

1. **`CHUNK` (`--max-num-batched-tokens`) cannot exceed 8192.** Doubling it dies with
   `triton.runtime.errors.OutOfResources: shared memory, Required: 131072, Hardware limit: 65536` —
   gfx1201 has 64 KiB of LDS per workgroup. The image advertises the workaround it already applies
   ("bf16 / auto KV cache attention fit + tune — fits the RDNA4 64 KiB LDS at head_size 256"). So
   raising the prefill chunk is **permanently unavailable**, which matters because chunked-prefill
   admission is what makes TTFT explode at deep concurrency.
2. **The int2 draft head blows LDS above ~64 sequences.** Same error, traced to
   `radiance_drafthead.py::_apply_head_int2` (`RADIANCE_FAST_DRAFT`, **on by default**). It is also
   **hardcoded** `-e RADIANCE_FAST_DRAFT=1` in `paroquant/run_paroquant.sh`, so the host environment
   cannot turn it off without patching the launcher. Worth reporting upstream.
3. **One Mamba cache block per decode sequence**, and it is the resident-concurrency ceiling:
   `ValueError: max_num_seqs (96) exceeds available Mamba cache blocks (84). Each decode sequence
   requires one Mamba cache block, so CUDA graph capture cannot proceed.` The 48 linear-attention
   layers carry per-sequence recurrent state in a page pool forced to the same page size as attention
   ("Padding mamba page size by 0.13% to ensure that mamba page size and attention page size are
   exactly equal"), so **sequence slots and context tokens compete for one budget**. Pure-attention
   models have no such term — which is why the Gemma/Qwen3.6 models in
   [02](02-engines-llamacpp-vs-vllm.md) reached n=80–96 and this cannot.
4. **Automatic GPU detection is dangerous on this box.** `gpu-detect.sh` uses a VRAM floor of 8192
   MiB, which the 16 GB RX 9070 XT passes — default detection reports "2 x Radeon AI PRO R9700 (16304
   MiB each)", picks **TP=2 across a 32 GB and a 16 GB card**, sizes KV to the smaller one, and would
   take ComfyUI's GPU. Always pass `GPUS=0 TP=1`.

## 6. Concurrency

Method: `scripts/harness/conc_test.py`'s (~70-token prompt, `max_tokens=800`), so
`total_tokens / wall` measures **decode** concurrency and the numbers compare with
[02](02-engines-llamacpp-vs-vllm.md).

| n | no-spec per-stream / agg | DFlash2 per-stream / agg |
|--:|--:|--:|
| 1 | 20.8 / 20.8 | 62.8 / 62.8 |
| 4 | 19.9 / 79.6 | 54.5 / 211.1 |
| **8** | 18.5 / 147.7 | **43.5 / 220.3** |
| 16 | 16.9 / 270.5 | 31.8 / 267.0 |
| 24 | 15.0 / 211.6 | 25.6 / 292.8 |
| 48 | 11.2 / 267.1 | 16.7 / 293.4 |
| 64 | 9.7 / **331.4** | 13.7 / 286.7 |

- **n=8 with the drafter is the agent operating point: 43.5 tok/s per stream, 220 aggregate** — better
  per-stream *and* 1.6× the aggregate of production's PLD chain at `-np 4` (39.2 / 139.0), at twice the
  streams.
- **No-spec scales almost perfectly to n=16** and is *fair*: per-stream min and max agree to two
  decimals (16.91 / 16.93), and 16.91 × 16 = 270.6 matches the measured 270.5, so all 16 were truly
  resident. The speculative run at n=16 shows min 16.69 / max 51.24 — a fast minority and a queued
  majority.
- **Speculation is a low-concurrency optimization.** 3× at n=1, neutral by n=16, and it loses the
  ceiling at n=64 (287 vs 331).
- **Peak aggregate 331 tok/s**, versus 1,634–2,485 for the models in [02](02-engines-llamacpp-vs-vllm.md).
  The gap is *not* the engine: resident batch tops out near **18** because 19 GB of weights leave 4.5
  GiB of KV, where Gemma-4-12B leaves ~18 GB. Active parameters (~23B vs 3–12B) explain part; the
  KV/batch ceiling explains most.

**It is not a compute ceiling.** At 331 tok/s with ~18 resident, decode GEMM is
`2 × 23.4B × 331 ≈ 15 TFLOPS` against the 225 TF/s measured ceiling, and bandwidth is
`18.4 steps/s × 19.4 GB ≈ 357 GB/s` of ~640 — about 7% of compute and 56% of bandwidth. Board power,
however, sat at **245–267 W against the 250 W cap on all sixteen rungs measured**. The cap and GDN's
48-layer serial recurrent update (launch-bound, not FLOP-bound) are the remaining candidates.

**CPU is not the bottleneck either.** With `--cpuset-cpus=0-11` (one thread per physical core; `N` and
`N+12` are siblings on this 5900X, single NUMA node), SMT siblings stayed at **0–1%** on every rung
while one physical core sat at 94–100% and the 24-thread mean never exceeded 9%. The pegged core is
vLLM's spinning loop and does not scale with load, so disabling SMT would buy nothing — and would cost
compile time, since hipcc/clang is the one workload that benefits from SMT.

## 6b. The 330 W power cap, and a matched llama.cpp reference (2026-09-12, later)

§6 established the concurrency plateau was **not** compute- or bandwidth-bound while board power sat
pegged at the 250 W cap. Raising the R9700's cap to its 330 W ceiling — keeping the undervolt and
memory OC, changing only the limit — confirms it.

**Card indices are inverted between LACT and HIP. Check before touching a cap.**

| index | LACT / sysfs | HIP / rocm-smi |
|---|---|---|
| 0 | RX 9070 XT, 16304 MiB, `0x7550`, cap_max **374 W** | **R9700** |
| 1 | **R9700**, 32624 MiB, `0x7551`, cap_max **330 W** | RX 9070 XT |

`sudo lact cli set-power-cap --gpu-id 1 330` targets the R9700; `--gpu-id 0` would raise the
*16 GB card that runs ComfyUI*. `lact cli ... power-limit get` reads unprivileged; `set` goes through
the daemon.

**Effect on the MXFP4 serve** (SPEC=5, `MAXLEN=65536`, `MAXSEQS=8`, util 0.96 — KV identical at
5.13 GiB / 103,268 both times, so the config is byte-identical and the delta is purely power):

| metric | 250 W | 330 W | delta |
|---|--:|--:|--:|
| PP @2k | 3,134.8 | **3,542.2** | +13.0% |
| PP @32k | 2,818.2 | **3,238.3** | +14.9% |
| PP @60k | 4,493.7 | **5,128.1** | +14.1% |
| decode, prose shallow / deep | 61.9 / 58.0 | **66.2 / 62.0** | +7.0% |
| decode, code-edit shallow / deep | 86.5 / 72.9 | **92.7 / 78.1** | +7.1% |
| peak concurrent aggregate | 331.4 @ n=64 | **385.8 @ n=24** | +16% |

PP gains ~14%, decode ~7%, and the concurrency peak rises **and arrives at a third the stream count**
with near-perfect fairness (16.07 / 16.10 at n=24; `16.08 × 24 = 386` = the measured 385.8, so all 24
were resident). At 330 W, 5,128 tok/s × ~47 GFLOP/token ≈ **241 TFLOPS** — above the 225 TF/s measured
under a lower cap, which is consistent with that ceiling being a power ceiling too.

**Thermals and transients:** junction climbed monotonically **82 → 91 °C** across a short run without
reaching steady state, and board power sampled **up to 373 W against the 330 W cap** (+13%). This
matches [06](06-power-and-stability.md): the cap does not bound transients. Idle recovery was clean
(47 °C, 18 W). The hardlock class was attributed to TP=2 seesaw, absent here at TP=1.

### Matched comparison: llama.cpp production vs MXFP4 vLLM, both at 330 W, same harnesses

Production (`qwen38.service`, heretic Q4_K_S, `ngram-mod,draft-mtp`, `-np 4`, 262k) re-measured the
same hour so nothing is cross-dated or cross-harness.

| | llama.cpp production | MXFP4 vLLM TP=1 | |
|---|--:|--:|--:|
| PP @2k | 990.8 | **3,542.2** | **3.6×** |
| PP @32k | 1,013.1 | **3,238.3** | **3.2×** |
| PP @128k | 711.2 | *unreachable* | — |
| decode, prose shallow | 52.5 | **66.2** | +26% |
| decode, code-edit shallow | 65.3 | **92.7** | +42% |
| decode, prose deep | 44.8 @42k | **62.0 @55k** | +39% |
| decode, code-edit deep | 53.6 @42k | **78.1 @55k** | +46% |
| **peak concurrent aggregate** | **95.7 @ n=2** | **385.8 @ n=24** | **4.0×** |
| usable context | **262,144** | 103,268 (65k/request) | **llama.cpp** |

- **vLLM wins every speed axis at matched power**, and the deep-decode rows favour it despite vLLM
  being measured at a *deeper* context (55k vs 42k — the two harnesses calibrate tokens/word
  differently on the same prose).
- **llama.cpp saturates at n=2 and never exceeds ~96 tok/s aggregate**, with per-stream collapsing
  70.1 → 12.75 and fairness falling apart (min 5.73 / max 35.26 at n=16). vLLM holds 16.07/16.10 at
  n=24. This is the continuous-batching advantage, finally measured on equal footing.
- **llama.cpp is no longer dispatch-bound.** [02](02-engines-llamacpp-vs-vllm.md) found one core pinned
  at 100% on the builds of that time; here it used 15–30% of one core with a 2% 24-thread mean.
- **Context remains the only axis llama.cpp wins, and it wins it by 2.5×.**

### `SPEC=7` is not viable at TP=1

The drafter warns it was trained at `block_size=8` (7 speculative tokens) and that acceptance is capped
below its potential at `SPEC=5`. It cannot be honoured on one card — two extra speculative tokens cost
roughly **4 GiB** of draft buffers and graphs:

| config | KV available | needed for one max-len request | result |
|---|--:|--:|---|
| SPEC=5, `MAXLEN=65536`, util 0.96 | 5.13 GiB | — | serves |
| SPEC=7, `MAXLEN=65536`, util 0.96 | **1.15 GiB** | 3.57 GiB | refuses |
| SPEC=7, `MAXLEN=16384`, util 0.96 | **0.77 GiB** | 2.06 GiB | refuses |
| SPEC=7, `MAXLEN=16384`, util 0.98, `MAXSEQS=4` | **1.49 GiB** | 2.06 GiB | refuses — *"estimated maximum model length is 1648"* |

At the top of the documented utilisation range with only four sequence slots, SPEC=7 leaves room for a
**1,648-token** context. Upstream's own sweep preferred SPEC=5 anyway (5 beat 7 by 8–13% aggregate),
but that was at TP=2 — here the choice is made by memory, not by throughput. Another point for the
second card.

## 6c. Vision: the image-token budget, and a wallclock A/B on real pages

Test set: five webtoon pages from the manhwa project, **800 x 3,520-7,190 px** (aspect ratios to 1:9).
Identical images, identical prompt (transcribe dialogue in order, then summarise), `max_tokens=700`,
temp 0, one at a time. Both engines at the **330 W** cap.

### The budget is a configured ceiling on both engines, and they do NOT default the same

This is the confound that nearly invalidated the comparison. Native image tokens = pixels / 1024
(patch 16, merge 2 -> 32x32 px per token).

| engine | knob | default here | cap in tokens |
|---|---|---|--:|
| llama.cpp | `--image-min-tokens` / `--image-max-tokens` (both *"default: read from model"*) | production sets min=1024, leaves max at the model default | **~4,096** |
| vLLM | checkpoint `preprocessor_config.json` -> `size.longest_edge` (a **pixel-area** budget, not an edge length) | `16777216` px | **16,384** |

The llama.cpp ceiling is visible in the run itself — two pages came in at native and three were clipped:

| page | px | native tokens | llama.cpp prompt | verdict |
|---|--:|--:|--:|---|
| 2 | 800x5120 | 4,000 | 4,102 | native |
| 5 | 800x3520 | 2,750 | 2,852 | native |
| 1 | 800x7190 | 5,617 | 4,113 | **clipped ~4,011** |
| 3 | 800x7135 | 5,574 | 4,113 | **clipped ~4,011** |
| 4 | 800x6925 | 5,410 | 4,050 | **clipped ~3,948** |

Left at defaults, vLLM would have processed **40% more image tokens** on the tall pages — looking
slower while doing more work and (probably) reading more text. Any vision comparison between these two
engines is meaningless until the budgets are matched.

Upstream's launcher has no passthrough for this, so `MM_KWARGS` was added locally:
`MM_KWARGS='{"size": {"longest_edge": <tokens x 1024>, "shortest_edge": 65536}}'`.

### Results

| | llama.cpp (~4,096 cap) | vLLM **native** | vLLM **matched 4,096** |
|---|--:|--:|--:|
| total wallclock, 5 pages | 91.3 s | 81.0 s | **75.1 s** |
| per page | 18.3 s | 16.2 s | **15.0 s** |
| mean prompt tokens | 3,846 | 4,730 | 3,804 |
| **mean TTFT** (ViT encode + prefill) | 6.42 s | 2.31 s | **1.78 s** |
| prefill tok/s incl. encode | 446-734 | 1,940-2,372 | 2,058-2,373 |
| decode tok/s | **59.3** | 46.8 | 45.4 |

- **Matched, vLLM is 18% faster wallclock and 3.6x faster to first token**, despite losing decode by
  23%. Image encode plus prefill dominates this workload — decode is about a fifth of the time — so
  the fp8 prefill advantage decides it.
- **vLLM at full native resolution (81.0 s) still beats llama.cpp at the reduced 4,096 cap (91.3 s).**
  The quality-for-speed trade this workload used to require is gone: take native and finish sooner.
- Matched token counts confirm the normalisation held (vLLM 4,071/4,060/4,071/4,008/2,810 vs
  llama.cpp 4,113/4,102/4,113/4,050/2,852 — within ~1%).

**Quality was NOT measured.** All three arms hit the 700-token cap with a large share spent on
reasoning preamble, so the transcriptions were truncated. A real OCR comparison needs xhigh or
thinking-off (where this model is known to do its best work) and a much larger output budget.

## 6d. Two productions, mutually exclusive

The box now has **two production inference endpoints**, and they cannot run together — each needs all
of GPU0.

| | llama production | vLLM production |
|---|---|---|
| unit | `qwen38.service` | `qwen_vllm.service` (installed 2026-09-12) |
| port | **8080** | **8000** (vLLM default) |
| engine | llama.cpp Vulkan `434ddbb` | radiance vLLM 0.27.1, MXFP4 W4A8 |
| model | heretic Qwen3.8-27B Q4_K_S | **stock** Qwen3.8-27B PARO-MXFP4 |
| context | **262,144** | 65,536/request, 103k pool |
| strengths | context, decode, heretic behaviour | PP 3.2-3.6x, TTFT, vision wallclock, 4x concurrency |
| launch | `systemctl --user start qwen38.service` | `systemctl --user start qwen_vllm.service` |

`qwen_vllm.service` declares `Conflicts=qwen38.service` and is deliberately **static (no autostart)**,
so it cannot race llama production at boot. Config lives in `~/launch_vllm_prod.sh`. First start after
a config change recompiles Triton/inductor graphs and takes ~8-9 minutes; later starts reuse the cache.

**RESOLVED in §6e:** a heretic build of this format now exists and scores 18/19 on Core-19. The text
below described the state before that. This is stock
Qwen3.8-27B. A heretic or abliterated checkpoint in this format would need building — upstream ships
`paroquant/build_hybrid.py`, which produces the format from bf16 base weights plus z-lab's rotations
(400 modules in 91 s), so it is a real path rather than a wait for someone else to publish one.

## 6e. Building a heretic MXFP4 checkpoint, and Core-19 on it

The published checkpoint is stock Qwen3.8-27B, which made it unusable as a production worker
([12](12-prompt-lookup-decoding.md) explains why heretic is the line). So the format was rebuilt from
the heretic weights.

### The build

`paroquant/build_hybrid.py` takes bf16 base weights plus z-lab's **trained rotations** and quantizes
one-shot to MXFP4 — no optimizer, no calibration. Inputs: `trohrbaugh/Qwen3.8-27B-heretic-ara`
(a67ae100, 52 GiB bf16), `z-lab/Qwen3.8-27B-PARO` (rotations), `z-lab/paroquant @ 9ee635a` plus this
repo's `paroquant_radiance.patch` (which supplies the ROCm/HIP path — upstream's rotation extension is
CUDA-only).

**It streams.** `safe_open` materialises one tensor at a time and rotates it on the GPU, so the 55.6 GiB
CPU-resident requirement in `requant.sh`'s MEMORY note applies to the *optimizer*, not to this path. It
runs fine in 31 GB.

Pre-flight, checked before spending the download: every architecture parameter identical to z-lab's,
and **all 400 of z-lab's quantized modules present in the heretic base**, so `assert n_q == 400` could
not fire. Result:

```
z-lab quantized modules: 400 | mxfp4 scale rule: ocp
DONE: 400 modules, worst pseudo round-trip rel 8.47e-07, 62s
```

8.47e-07 is fp32 round-trip noise — the transform is exact. It says nothing about whether stock-trained
rotations suit heretic; see the verdict below for why they do.

**Why the rotation transfer was safe, discovered after the fact.** heretic-ara is not a finetune. It is
[Heretic](https://github.com/p-e-w/heretic) abliteration using **ARA (Arbitrary-Rank Ablation)** on
layers 26–56, tuned by an Optuna search with `preserve_good_behavior_weight` 0.9432 against
`steer_bad_behavior_weight` 0.0009 — roughly 1000:1 toward preservation. Its card reports **KL
divergence 0.0535** from stock with **0/100 refusals** (stock: 99/100). So the outlier distribution
z-lab's rotations were fitted to on stock *is*, to within rounding, heretic's distribution. A heavy
finetune (a merge or a NEO-CODE/Cold-Fusion style tune) would not have been a safe base for this.

**What was NOT done:** Launch80's published checkpoint is the **`-ft`** variant — a stage-2 fine-tune of
weights and the per-block e8m0 exponent bias under the MXFP4 grid, rotations frozen (256 samples × 2
epochs). On upstream's own ladder that is worth GSM8K 96.96% (one-shot) → 97.60% (fine-tuned, served).
**Ours is the one-shot**, so everything below is a floor, not a ceiling. The fine-tune needs the
optimizer's 55.6 GiB CPU-resident footprint; this box has 31 GB and its only swap is **zram** (compressed
RAM, not disk), so it needs a real disk swapfile first. The GPU side would fit the 16 GB card — after
capture only one layer (~0.5 GiB) is touched per iteration — so it could run on GPU1 while GPU0 serves.

### Core-19 on the heretic MXFP4 build

Run against the live vLLM endpoint with `core19_external.sh` (no server management — a model switch
costs 8–9 min of JIT, so the campaign must not restart anything). Effort forced to **xhigh** via a
`TB_EXTRA_BODY` hook; the GGUF arms used their embedded template, which
`{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}` — **so effort is matched**.

| | score | pass@1 | quant | bits |
|---|---|--:|---|--:|
| **MXFP4 W4A8 (this build, vLLM)** | **18/19** | **17** | paroquant_mxfp4 | 4.25 |
| stock UD-Q4_K_XL (llama.cpp, Strix Halo, *medium*) | 18/19 | 17 | unsloth dynamic | ~4.5 |
| heretic Q4_K_S (llama.cpp, R9700) | 17/19 | 16 | K-quant small | ~4.0 |

**It matches Donato's stock reference and beats the source GGUF by one task, at the lowest bit depth of
the three.** Against the GGUF it *gained* `extract-elf` and `configure-git-webserver` — both genuine
GGUF failures, no infra involved, the agent ran to completion on both attempts — and *lost*
`mteb-retrieve`, which is marginal for the whole family (the GGUF needed a retry for it; Turbo failed it
twice).

**So strict 4-bit did not cost reasoning.** The most economical reading is not that MXFP4 improved the
model but that **Q4_K_S was costing heretic a task and MXFP4 at 4.25 bpw with rotations is not.**

### Wall clock: 1.57×, and it is a slope

Agent minutes, attempt 1, both sides (container build and verifier excluded from both).
Chart: [data/vllm-mxfp4/core19_time.html](../data/vllm-mxfp4/core19_time.html).

| | MXFP4 vLLM | Q4_K_S llama.cpp |
|---|--:|--:|
| total, 19 tasks | **240.7 min** | 378.1 min |
| faster / slower / parity | **8 / 10 / 1** | |

| task | ours | baseline | |
|---|--:|--:|--:|
| mailman | 17.0 | 68.6 | **4.0×** |
| cobol-modernization | 16.5 | 45.9 | **2.8×** |
| headless-terminal | 13.1 | 28.3 | **2.2×** |
| llm-inference-batching-scheduler | 31.0 | 66.3 | **2.1×** |
| … | | | |
| mteb-retrieve | 9.1 | 6.0 | 0.66× |
| sparql-university | 13.1 | 8.6 | 0.65× |
| regex-log | 17.6 | 5.7 | **0.32×** |

The aggregate is faster while a *majority of tasks are slower*, because the gain concentrates entirely
in the long tail. This is §3 and §4 showing up at application level: prefill is 3.2–3.6× and decode is
~23% slower, and an agent re-prefills its whole context on every step — so a 60-step task pays the
prefill advantage 60 times while a 3-step task is decode-bound and gives a little back.

### The 65k context window never bound

§6d flagged three tasks whose baseline peaks exceeded the 57,536 summarisation trigger
(`SUMMARIZATION_FREE_TOKENS = 8000`, absolute, so the trigger is `context_length − 8000`). **None of them
compacted, because none of them got near it.** Measured peaks this run:

| task | baseline peak | this run |
|---|--:|--:|
| mailman | 89,089 | **33,610** (27 steps) |
| llm-inference-batching-scheduler | 65,936 | **53,727** |
| cobol-modernization | 60,439 | **33,118** |

Maximum across all 19 was **53,727**, under the trigger. The run simply used less context — `mailman`
solved it in roughly a third of the baseline's tokens *and* a quarter of the wall clock. So the 65k
per-request ceiling, which §2 identified as this stack's real weakness, cost nothing on this workload.
That is a narrower claim than it looks: Core-19 agent contexts run a 10–40k median, and a workload that
genuinely needs 262k would still be excluded.

### CAVEAT THAT QUALIFIES EVERYTHING ABOVE: this serve is inside a documented degeneration envelope

The Qwen3.5-family behaviour envelope (Qwen's own Qwen3.6-27B model card, and the owner's hands-on
priors) states two failure conditions:

- **allocated context below ~128k → degeneration**: the model rambles and emits tokens until the cap.
  **Window *size* and window *occupancy* are separate levers** — ≥128k must be *allocated* to stay out
  of the loop, independent of how much is actually used.
- **heavy KV-cache quantization → the same catastrophic loop** (q4-class is fatal; q8-class is
  tolerable but wants flash attention on).

**This stack runs `--max-model-len 65536` — half the stated floor — with `--kv-cache-dtype fp8`.** It
leans on both conditions at once, and §2 shows why: a TP=1 serve on one 32 GB card *cannot* allocate
131,072 (the launcher's own comment caps it at 65,536).

It was observed live. On the stock arm's `mailman` retry the KV cache traced a clean sawtooth —
climbing ~15.5 points of the 103,268-token pool (~16k tokens), resetting, and climbing again on a
~4-minute cycle for **57 minutes without the trajectory advancing a single step**, ending in the
3-hour `AgentTimeoutError`. That is the documented signature, not a novel failure.

**What this does and does not change:**

- It does **not** invalidate heretic's 18/19. That score was achieved *inside* the envelope, which
  makes it a floor rather than a ceiling.
- It **does** mean the heretic-vs-stock divergence on `mailman` cannot be cleanly attributed to the
  weights. The honest statement is that stock tripped a known envelope and heretic did not — plausibly
  because heretic reached the answer in 28 steps against stock's 45 and so never accumulated the
  context to trigger it. Which model degrades *less* under KV starvation is a real and useful property,
  but it is not the same claim as which model reasons better.
- It reframes the second R9700 from a capacity upgrade to a **correctness** one: TP=2 halves weights
  per card, which is the only way this family gets its ≥128k allocation on this hardware. Until then
  every result from this serve carries this asterisk.

**Draft acceptance on agentic work: 55–57%** (mean accepted length ~3.8 of 6), against 40–46% on image
description in §6c. Agents quote files back, so the drafter hits more often — the same overlap effect
[12](12-prompt-lookup-decoding.md) measured for n-gram lookup, here in a trained drafter.

## 7. Setup gotchas (all upstream, all fixed locally)

- `setup-paroquant.sh` **rejects this checkpoint**: its validator gates on `quant_method=paroquant`
  (the int4 path) and this declares `paroquant_mxfp4`. Use `PREPARE_ONLY=1 ./serve-mxfp4.sh` for setup
  instead, which pulls the image and builds libr4d without touching checkpoints.
- The launcher's default `R4D_KEY=b9e42ab-rx5` **lags what setup builds** (`rx6`). Its own error
  message names the fix. libr4d is load-bearing: the image's own copy predates the gated-delta-net
  overflow fix and NaNs this model, and a missing build is a *silent* fallback.
- `--chat-template` is hardcoded to `/root/.cache/huggingface/qwen-fixed-v22.3.jinja`, which
  `PREPARE_ONLY` never populates. Copy `qwen-fixed-v22.3.jinja` from the repo root into `$HF_CACHE`.
- The checkpoint ships **zero MTP tensors** (verified: 0 of 2,784 match `mtp.*`), so the external
  DFlash2-FP8 drafter is its only speculative path. The drafter also warns it was trained at
  `block_size=8` (i.e. `num_speculative_tokens=7`) while the launcher defaults to `SPEC=5`.

## 8. Verdict and what is still open

**Positive, and stronger after the power cap.** At a matched 330 W against production measured the
same hour, the MXFP4 vLLM serve wins **every speed axis**: PP 3.2–3.6×, realistic decode +26–46%, and
peak concurrent aggregate **4.0×** (385.8 vs 95.7, where llama.cpp saturates at n=2). The fp8 WMMA
path is real and delivers ~2.35× the FP16 figure; PP is
2.6–3.5× production and improves with depth; decode clears every project gate once speculation is on;
and n=8 concurrency beats the production line on both axes. **Context capacity is the loss** — 89k–103k
usable tokens in a serving configuration against production's 262k, with per-request context capped at
65k on one card. That is a direct argument for the second R9700, where weights halve per card and the
freed memory becomes KV.

Now measured: the power cap (§6b), `SPEC=7` (not viable at TP=1), vision (§6c), and **quality —
Core-19 18/19 (17 pass@1) on a heretic build of this format, matching Donato's stock reference and
beating the source GGUF at the lowest bit depth of the three, at 1.57× the wall clock (§6e)**.
Still open: the `-ft` stage-2 fine-tune (blocked on a disk swapfile; ours is the one-shot, i.e. a
floor), and a heterogeneous text+image concurrency
ladder (this is the VL checkpoint — `[radiance.vit] head_dim-72 attention installed`, encoder cache
budget 16,384, `compile_mm_encoder: False` — and real traffic is not homogeneous); and quality, which
has had **no** gate at all here. Note this is stock Qwen3.8-27B, **not heretic**, so it is not a
like-for-like replacement for the production worker regardless of speed.

Data: [data/vllm-mxfp4](../data/vllm-mxfp4). Harnesses: [scripts/vllm-mxfp4](../scripts/vllm-mxfp4).
Measurement traps hit and corrected along the way are in [MISTAKES](../MISTAKES.md).

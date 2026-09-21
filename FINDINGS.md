# Findings ledger

One entry per discovery, newest first — this is the repo's stream. **Claim → evidence → link.** Dates: the doc's own date box where it has one, otherwise when the write-up was first committed; some findings were measured days before they were written up. Numbers are as measured then, under the [README](README.md) ground rules. [LEVERS.md](LEVERS.md) collects the knob-by-knob deltas; [docs/](docs/) holds the full record.

## 2026-09-20 · TurboQuant's crash on gfx1201 was a disabled flash-attn path, not the rotation quant
`vllm` `turboquant` `rocm` `gfx1201`

- Four independent bugs stood between "TurboQuant boots" and "TurboQuant serves at depth": R4D rejects the KV dtype outright; a hardcoded `compile_sizes` fights cudagraph padding under MTP; MTP + TurboQuant is a hard upstream block (`TurboQuantMetadata` isn't in vLLM's spec-decode metadata allowlist for `num_speculative_tokens > 1`); and the real crash — `fa_utils.py` only routes AITER's Triton flash-attn to `gfx1250`, so gfx1201 silently falls to an uninstalled upstream `flash_attn` package, forcing a dense `O(q_len·seq_len)` SDPA fallback that OOMs (~4.6 GiB at depth 32,000) instead of flash-attention's `O(1)`.
- Fixed locally (`scripts/turboquant-gfx1201/patch_turboquant_flash_attn.py`): try AITER unconditionally, fall back to upstream `flash_attn` only on `ImportError`. Confirmed serving cleanly through depth 120,000 (target 262,144) with correct output.
- A real vLLM PR for the upstream gate was prepared and validated (`ruff` clean, tested on this hardware) then discarded — the same gfx1250-only-gate pattern recurs elsewhere in vLLM's own codebase, so it reads as a known gap, not a novel finding.
- Even fixed, decode-vs-depth is steep: **21.4 → 13.6 → 10.0 → 6.6 tok/s** at depth 2k/32k/60k/120k (turboquant_k8v4, default splits) vs fp8's **22.3 → 20.8 → 19.8** over the same range. `tq_max_kv_splits_for_cuda_graph` 32→128 recovers **~13%** at depth 60k (10.0→11.3, confirmed on repeat boots, <0.3% spread) and plateaus there — the rest is per-token value-dequant cost (scattered scale/zero-point reads), not yet fixed at the kernel level.
- **R4D's fp8 advantage isn't in the attention kernel** — R4D vs stock `ROCM_AITER_UNIFIED_ATTN`, same fp8 dtype: 22.29/19.78 vs 22.25/19.84 (2k/60k depth), indistinguishable. R4D's edge for this model lives in GDN-layer acceleration and MXFP4 weight quant, not KV-cache attention.

→ [docs/15](docs/15-turboquant-kv-quant-gfx1201.md) · [data/turboquant-gfx1201](data/turboquant-gfx1201)

## 2026-09-14 · Dropping the drafter doubled the 60-agent throughput — the knee is at n ≈ 8–16
`vllm` `spec-decode` `concurrency` `9b`

- MTP vs no-spec, co-measured same serve (9B heretic PARO-MXFP4, 16 GB card, 350 W, 800-token streams): single stream MTP wins **+34%** (104.3 vs 77.9 tok/s); the crossover lands between n=8 (0.81×) and n=16 (**1.36×**).
- At **n=60: no-spec 1,915.5 vs MTP 862.1 tok/s — 2.22×**, and the per-stream spread is **1.00× (32–32) vs 5.3× (14–76)**: MTP's verification overhead plus draft-acceptance variance (~57% at mean length 2.1) creates stragglers that double the makespan — **55.7 s vs 25.1 s** for the same tokens. Peak is n=64: 1,988.7 vs 888.6; by n=90 the no-spec pool saturates and the ratio falls to 1.69×.
- The drafter's second cost is memory: no-spec KV pool **273,881 vs 110,649 fp8 tokens (2.4×)** — the guaranteed per-agent context floor at 60 concurrent rises from 1,844 to 4,564 tokens.
- The standing serve for 60-agent fan-out is now no-spec; MTP is kept for interactive low-concurrency use, where it wins.

→ [docs/14](docs/14-vllm-9b-mxfp4-60-agent-fanout.md) · [data/9b-paro-mxfp4](data/9b-paro-mxfp4)

## 2026-09-14 · 134 vision tasks in 31.7 s: the 60-agent workload, for real
`vllm` `vision` `agents` `9b`

- 134 rendered images described by **60 concurrent workers** against the standing no-spec serve: **31.7 s wall, 134/134, zero failures**, 644.8 tok/s aggregate (20,424 completion tokens), **93% slot utilization** (4.23 img/s vs 4.55 theoretical).
- 134/60 = 2.23 waves, overlapped by the scheduler — makespan sits between two and three sequential waves (first completion 3.3 s, last stream 23.4 s). Per-stream decode median 13.57 tok/s; TTFT max 8.84 s (first-wave queueing).
- The aggregate (644.8) is far under the synthetic-text n=60 figure (1,915.5) by construction: outputs are 150 tokens, not 800, so fixed per-stream costs dominate the 13 s wall — a vision workload is a different shape, not a regression.
- Vision bench proper: 16/16, mean prefill 3,820 tok/s (encoding included), mean decode 93.1 tok/s, TTFT 0.11–0.40 s (first request 8.74 s = one-time kernel JIT), concurrent 439.0 tok/s at 5.30× wall-clock speedup.

→ [docs/14 §7](docs/14-vllm-9b-mxfp4-60-agent-fanout.md) · [data/9b-paro-mxfp4](data/9b-paro-mxfp4) (`swarm_134img_w60.json` carries all 134 descriptions)

## 2026-09-14 · TDP is not a perf lever on bandwidth-bound decode — and two power logs read the wrong card
`power` `vllm` `9b`

- Ladder peaks (agg tok/s, n=64) by cap, co-measured on the same serve: 231 W → 881.5 · 250 W → 881.7 · 350 W → 888.6 · 374 W → 893.3. **Floor→ceiling = +1.3% — inside boot noise.** At n=90: +1.8%.
- The card's natural draw under full load is **283 W median / 382 W max** (per-card sampler at the 374 W cap, n=95) — it never approaches the operating cap, so a cap is a **PSU-safety dial for the dual-card rig** (transients, both cards loaded), not a performance choice.
- **Correction:** the first two per-cap power logs read `rocm-smi GPU[0]`, which on this box is the **R9700, not the 9070 XT** (rocm-smi's device indices are inverted relative to the power tool's). Their power readings — a "331 W peak at the 350 W cap" and a "250 W run that drew 329–356 W" — were the *other card serving chat*, and are void. The throughput numbers from those runs stand; the power column above comes from runs that sampled the correct device. [MISTAKES](MISTAKES.md).

→ [docs/14 §5](docs/14-vllm-9b-mxfp4-60-agent-fanout.md) · [data/9b-paro-mxfp4](data/9b-paro-mxfp4) (`pwr_both.log`)

## 2026-09-14 · The chunk-size cliff: 4096→2048 collapses tail-distinct for nothing
`vllm` `kernel` `9b`

- On the hybrid Gated-DeltaNet model, tuning `--max-num-batched-tokens` boot by boot (one boot each, n=60 re-measured): 8192→4096 never moved tail-distinct; **4096→2048 collapsed it 0.70 → 0.56, reproducible bit-for-bit across two runs** — for a pool delta of 110,649 → 103,028 tokens that is inside boot-to-boot noise (the same v3 config booted at 101,199 then 110,649).
- Mechanism (hypothesis, labelled): the chunked scan is numerically sensitive to chunk boundaries, drifting token choices over long generations. Rule: **chunk ≥ 4096** on this architecture; and treat any same-config boot delta under ~10% as noise.
- Four boot gates preceded the tuning: donor config `float16` rejected by the R4D dtype gate (→ bfloat16); **R4D hard-requires gqa=6 — the 9B's 16q/4kv (gqa 4) forces the unified-attention backend**; `compile_sizes` padding 8→9 under MTP (→ empty list); and the graph profiler's *estimated* CUDA-graph reservation (3.65 GiB vs 2.25 actual) ate the KV pool at util 0.92 (→ disable the estimate).

→ [docs/14 §2–3](docs/14-vllm-9b-mxfp4-60-agent-fanout.md) · [data/9b-paro-mxfp4](data/9b-paro-mxfp4) (tuning battery `v1_mtp_rungs` → `h9v4_mtp_rungs`)

## 2026-09-14 · A 9B heretic PARO-MXFP4 build with the MTP head re-attached
`vllm` `models` `build` `9b`

- ParoQuant rotations (z-lab, krot=8) grafted onto an abliterated 9B — abliteration, not a finetune, which is why stock-trained rotations transfer (27B precedent [13 §1](docs/13-vllm-mxfp4-w4a8-rdna4.md)): **200/200 modules** (32× MLP, 24× linear-attn, 8× full-attn), preflight zero missing.
- The abliterated checkpoint had dropped its MTP head (760 tensors vs 775 base); the 15 `mtp.*` tensors were grafted back from Base shards — **243.3M params, exactly the Base↔heretic tensor delta**, confirming the MTP block was never abliterated.
- Build **243 s**, worst pseudo round-trip **5.71e-07** (fp32 noise; the 27B was 8.47e-07). Output 1,575 tensors = 200×5 quantized + 575 passthrough + 15 MTP, 8.55 GB.
- Prefix caching is the fan-out multiplier: a second 55,254-token prefill on a shared prefix ran **58,350 vs 3,860 tok/s cold — 15×**. A shared system prompt is paid once, not 60×. Code-edit prefill runs to **222,971 tok/s at 28k depth** (n-gram, [12](docs/12-prompt-lookup-decoding.md)) — agent workloads are code-heavy, so real traffic gets the fast prefill.

→ [docs/14 §1, §6](docs/14-vllm-9b-mxfp4-60-agent-fanout.md) · [data/9b-paro-mxfp4](data/9b-paro-mxfp4)

## 2026-09-13 · Same format, different weights: a 0.3% dead heat per attempt, decided entirely by retries
`vllm` `models` `quality` `agents`

- Core-19 on both MXFP4 checkpoints, identical config, one R9700 at 330 W, xhigh: heretic **18/19** (17 pass@1), stock Launch80 **17/19** (16 pass@1), against heretic Q4_K_S on llama.cpp **17/19**.
- **Attempt-1 agent time is 241.5 vs 240.8 min — 0.3% apart, both exactly 1.57× llama.cpp's 378.1.** The engine-and-format change buys the wall clock; the weights buy none of it. Per-task times scatter ±2× both ways with no aggregate direction.
- **What separates them is retries.** Heretic needed 2, stock 3 (1 recovered). Grand totals including retries: **253.7 vs 471.6 min — 1.86× — for one fewer task solved.** In production that *is* the cost; it is not noise around the measurement.
- The single divergence is `mailman`, and it was **our context limit, not the weights**: `ContextWindowExceededError` at **65,537 input tokens against the 65,536 this run was configured with**, then 218.5 min spinning to the 3-hour ceiling. 131,072 serves on this hardware without the drafter, so a correctly-sized serve would likely have given stock the task. Heretic avoided it by solving in 28 steps where stock took 45 — step efficiency is the real difference, not reasoning quality.
- Capacity claims here previously omitted quant and KV config and were wrong as a result; see [MISTAKES](MISTAKES.md). MXFP4 is **18.07 GiB resident vs Q4_K_S's ~15 GiB** — "4.25 bpw" covers 400 projections while embeddings, `lm_head` and the vision tower stay bf16.

→ [docs/13 §6e](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [chart](data/vllm-mxfp4/core19_time.html) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-13 · The TP=1 serve runs at half Qwen's advised context floor, and the degeneration it warns about was observed live
`vllm` `models` `agents` `quality`

- Qwen's own Qwen3.6-27B card, *Serving* section, `[!Important]`: *"we advise maintaining a context length of at least 128K tokens to preserve thinking capabilities"* (default 262,144). This serve was configured at **65,536** — half the advised floor — with `--kv-cache-dtype fp8`. **Correction (same day): that was a choice, not a ceiling.** 131,072 boots at TP=1 (233,016 KV tokens, MXFP4 + fp8 KV, `MAXSEQS=2`, no drafter); the real constraint is **drafter-or-context** — with DFlash2 loaded it needs 5.28 GiB and has 0.90. llama.cpp serves the same 27B at 262,144 on this card at Q4_K_S + q8_0 KV. See [MISTAKES](MISTAKES.md).
- It is a **gradient, not a threshold**: Qwen ships a dial (`presence_penalty` 0–2 "to reduce endless repetition"), and the phrasing *"preserve thinking capabilities"* describes capability that degrades by degree. Repetition remains possible at 262k, just rarer; each increment of KV starvation raises the rate.
- Observed on the stock arm's Core-19 `mailman` retry: the KV cache traced a sawtooth of ~16k-token generations (~15.5 points of a 103,268-token pool) on a ~4-minute cycle for **57 minutes with zero trajectory progress**, ending in the 3-hour `AgentTimeoutError` — the documented "rambles until token cap" signature.
- Sampling also deviates: Qwen specifies thinking mode `temperature=1.0` / `top_p=0.95`; the serve applies **0.7** (the instruct-mode value) on thinking-mode requests, in the direction that raises repetition risk. Upstream and fleet-wide — `run_paroquant.sh` records the 1.0 → 0.7 change "across every vllm-switch target". `presence_penalty` sits at 0.0, per spec but leaving the one offered mitigation unused.
- Consequence: **the second R9700 is a correctness requirement, not a capacity upgrade** — TP=2 halves weights per card and is the only route to this family's advised allocation on this hardware. Results from this serve carry that asterisk, which makes heretic's 18/19 a floor rather than a ceiling.

→ [docs/13 §6e](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [OPEN-PROBLEMS](OPEN-PROBLEMS.md)

## 2026-09-12 · MXFP4 W4A8 is the first thing to actually use RDNA4's fp8 WMMA
`vllm` `rdna4` `kernel` `power`

- fp8 WMMA verified native on gfx1201 by reading kernels: 20 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` calls, no gfx942 spoof; the gate was the libraries, not the silicon.
- Hand-written HIP kernels reach 225 TF/s (~2.35x the FP16 figure); MXFP4's inner loop does zero VALU ops where int4 g128 does 16.
- On one R9700 at TP=1: 2.6-3.5x production prefill rising with depth; matched at 330 W the same hour — PP 3.2-3.6x, decode +26-46%, 4.0x concurrency peak (385.8 vs 95.7 tok/s).
- Loses on context capacity (103k usable vs 262k) — the clearest argument yet for a second card.
- Quality settled: heretic MXFP4 rebuilt from bf16 scores 18/19 Core-19 (17 pass@1), matching the stock reference; vision wallclock also wins (75.1 s vs 91.3 s).

→ [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-12 · The 250 W cap was the ceiling: 330 W lifts every axis; SPEC=7 is unreachable at TP=1
`power` `vllm` `spec-decode`

- Raising the cap 250 → 330 W on a byte-identical config (KV 5.13 GiB / 103,268 tokens both times): PP +13.0% / +14.9% / +14.1% at 2k / 32k / 60k, decode +7.0–7.1%, peak concurrent 331.4 @ n=64 → 385.8 @ n=24 (+16%).
- The plateau was power, not compute or bandwidth: at 250 W the serve drew 245–267 W on all sixteen rungs while using ~15 of 225 TF/s and ~357 of ~640 GB/s.
- At 330 W the peak arrives at a third of the streams with near-perfect fairness (16.07 / 16.10 at n=24); transients still sample up to 373 W under the 330 W cap and junction climbed 82 → 91 °C without steady state.
- SPEC=7 cannot be honoured at TP=1 — two extra speculative tokens cost roughly 4 GiB; even util 0.98 / MAXSEQS=4 leaves a 1,648-token context ("estimated maximum model length is 1648"), so SPEC=5 it is.

→ [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-12 · KV is the tuning surface; four hard gfx1201 limits close the rest
`kernel` `vllm` `multi-gpu`

- On a 32 GB card a 27B leaves 4–8 GiB for KV and every feature spends it: CUDA graphs ~56k tokens, the DFlash2 drafter ~66k even at util 0.96, and MAXSEQS 8 → 96 halves tokens/GiB (20,130 → 9,202).
- Per-request context was configured at 65,536 (MXFP4 + fp8 KV + DFlash2, `MAXSEQS=8`); ~~capped on one card~~ — **corrected 2026-09-13**, 131,072 serves without the drafter. Capacity is a function of quant, KV dtype, drafter and `MAXSEQS`, never of the card alone.
- CHUNK (--max-num-batched-tokens) cannot exceed 8192 — 16384 dies with shared memory Required: 131072, Hardware limit: 65536 (64 KiB of LDS per workgroup), so the obvious deep-concurrency fix is permanently unavailable.
- One Mamba cache block per decode sequence: 96 sequences exceed the 84 available, so slots and context compete for one budget — a tax pure-attention models don't pay.
- Auto-detection is dangerous: an 8192 MiB VRAM floor passes the 16 GB card and would pick TP=2 across the mismatched pair — always pass GPUS=0 TP=1.

→ [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-12 · A heretic MXFP4 build in 62 s, and a 1.57× Core-19 wall clock
`vllm` `models` `quality` `agents`

- One-shot rebuild from bf16 + z-lab's rotations (400/400 modules): worst pseudo round-trip rel 8.47e-07 in 62 s; it streams tensor-by-tensor, so it runs in 31 GB.
- Safe because heretic-ara is ARA abliteration (KL divergence 0.0535 from stock, 0/100 refusals) — not a finetune — so stock-trained rotations transfer.
- Ours is the one-shot; the served -ft stage-2 is worth GSM8K 96.96% → 97.60% on upstream's ladder and is blocked on a disk swapfile — everything measured here is a floor.
- Core-19 wallclock 240.7 vs 378.1 min (1.57×) but 8 faster / 10 slower / 1 parity — the gain is entirely the long tail, and the 65k window never bound (max peak 53,727).
- The box now runs two mutually exclusive productions: llama.cpp on 8080 (262,144 ctx, heretic Q4_K_S + q8_0 KV) and vLLM on 8000 (65,536/request as configured, MXFP4 + fp8 KV + drafter), Conflicts= in the unit, static so it can't race at boot.

→ [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-12 · Vision: match the image-token budgets or the comparison is fiction
`vllm` `models` `methodology`

- The budgets don't default the same: llama.cpp's ceiling is ~4,096 tokens (three of five webtoon pages clipped ~4,011) against vLLM's 16,384 (preprocessor longest_edge 16,777,216 px); at defaults vLLM would have processed 40% more image tokens.
- Matched at 4,096: vLLM 75.1 s vs llama.cpp 91.3 s total (15.0 vs 18.3 s per page), TTFT 1.78 s vs 6.42 s — 3.6× — despite losing decode 45.4 vs 59.3 tok/s.
- vLLM at full native resolution (81.0 s) still beats llama.cpp at the reduced cap; the quality-for-speed trade this workload used to require is gone.
- Quality was NOT measured: all three arms hit the 700-token cap mid-reasoning; a real OCR comparison needs xhigh or thinking-off and a larger output budget.

→ [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) · [data/vllm-mxfp4](data/vllm-mxfp4)

## 2026-09-11 · Core-19: the careful model solved more, the fast one solved faster
`agents` `quality` `models`

- Same server line and harness, 19 real terminal tasks: heretic 16/19 pass@1 (17/19 with retries); Turbo 9/14 on the tasks it ran — heretic 11/14 on those same 14 — and no retry flipped.
- On what it solves, Turbo is ~3× faster and ~4.5× cheaper in output tokens: median solved task 3.2 min · 4.9k tokens vs 9.3 min · 22.5k; on the 8 tasks both passed, 30 vs 98 min and 57k vs 258k tokens.
- Failure pattern: in four of Turbo's five genuine failures it declared success on a wrong or missing check; heretic's failures were thorough work against its own reading of the task.
- Five more Turbo tasks were never the model's: an overnight plain-HTTP apt-mirror outage broke container setup (120 s tool-install timeout); even a clean re-run caps Turbo at 14/19.
- No production change; production was restored to heretic when the Turbo arm ended.

→ [docs/08](docs/08-agentic-benchmark-core19.md) · [data/core19](data/core19)

## 2026-09-11 · Chain n-gram lookup in front of MTP: free speed where agents echo their input
`spec-decode` `llamacpp` `engine`

- `--spec-type ngram-mod,draft-mtp` (n-gram first, MTP fallback) wins everywhere and loses nowhere: copy 222.8 tok/s and code-edit 123.4 vs production MTP's 77.7 / 77.4; prose unchanged at 57.1.
- n-gram alone is a trap for mixed work — 174.6 on copy but back to the no-spec floor (34.9) on prose; the chain keeps MTP as the fallback at no cost.
- Survives concurrency: ~1.5× per-stream np1–np4 (np4 39.2 / 139.0 vs MTP 26.5 / 96.8), no collapse; a lone stream runs full speed at any -np (125.0 / 123.8 / 123.6), and n_ctx_slot stays 262,144 — -np is capacity, not a context split.
- Lossless at temp 0 except the pre-existing MTP prose flip (batched verify flips near-tie greedy tokens) — not introduced by n-gram.
- In production since 2026-09-11; live smoke 137.8 tok/s, n-gram accepting 14-token drafts (0.63).

→ [docs/12](docs/12-prompt-lookup-decoding.md) · [data/core19](data/core19)

## 2026-09-11 · What the reasoning traces show: self-checks run, answers wrong
`quality` `models` `agents`

- Rhyme, audited by hand after the grader only printed it: heretic 15/15 stanzas AABB; Turbo ~8/30 — three of six poems ABAB throughout, and its automatic "8/8" overstated them.
- Self-checks are unreliable: word counts exact on 59% of 145 Turbo lines vs 91% of 117 heretic; Turbo approved "3 thee/thou. Good." for a draft with two.
- Same gap as Core-19: in four of five genuine Turbo failures the model declared the task done — e.g. "the test script exited with code 0, confirming the bypass works" when nothing had fired.
- "Right answer, then talked out of it" did not appear: no final answer scored below an earlier draft across 9 poems; heretic diagnosed all five planted bugs in the first 7.8% of its reasoning.
- Why the two GGUFs differ: Turbo's BF16 output head (2,425 vs 995 MiB) and Q8_0 MTP block, +1.59 GiB matching a +1,633 MiB VRAM gap — the head is read every token, plausibly why it decodes slower.

→ [docs/11](docs/11-reasoning-traces-and-sanity-checks.md) · [data/traces](data/traces)

## 2026-09-11 · Both models can code; Turbo is 23× cheaper and loops one run in four
`quality` `models` `methodology`

- Both passed both code tasks; Turbo was 23× cheaper on the bug hunt (1,314 tokens · 23 s vs 30,311 · 544 s) and 5× on the physics build (15,676 · 275 s vs 82,193 · 1,658 s).
- Energy: heretic conserved to 0.0000% drift (event-driven collision timing), Turbo 0.17% — both inside the 0.5% bar.
- The poem loop: Turbo fixed one line and tried 656 variations across just 7 rhyme words, never questioning the line; cancelled after 33,539 tokens.
- Loop A/B, 3 runs per condition: 0/3 loops everywhere — intermittent, not guaranteed (1 in 4 counting the sanity run); q8_0 vs f16 KV made no difference.
- Small-sample side finding, unconfirmed: f16 KV gave Turbo +2.4 tok/s and +4 points of MTP acceptance.

→ [docs/11](docs/11-reasoning-traces-and-sanity-checks.md) · [data/core19](data/core19)

## 2026-09-11 · Nathan Wilson's Vulkan fork gives the 27B nothing
`engine` `backend` `llamacpp`

- release/v0.7.5-staging (213 commits ahead) vs production 434ddbb, heretic Q4_K_S: no-spec decode identical (34.4 / 31.3 / 24.2 vs 34.4 / 31.2 / 23.9 at 2k / 32k / 128k).
- With the production chain it is actually behind: 52.0 / 40.0 / 27.0 vs 59.3 / 46.4 / 28.2, acceptance 0.36 vs 0.41.
- Its perf lives in the DeepSeek-V4-Flash / Qwen3-Next branches (dsv4-*, qwen4exp); master was 0 commits ahead of upstream — the Strix Halo stack only pays for models that want unified memory.
- Verdict: the production build stays.

→ [docs/12](docs/12-prompt-lookup-decoding.md) · [data/core19/engine_ab.html](data/core19/engine_ab.html)

## 2026-09-11 · A 2B worker: 2/19, and it says when it doesn't know
`models` `agents` `quality`

- MiniCPM5-2B (4-bit) on a 6 GB laptop GPU, same suite, 2 tasks at a time: 2/19 attempt 1 (2/16 excluding 3 setup errors); pass@2 as run 2/19, attempt 2 stopped partway.
- Quality only — the laptop makes its speed meaningless for the intended 16 GB card.
- Failures: declared success with requirements missing (no SSH account, a date not in YYYY-MM-DD), killed its own container freeing a port, 4 tasks ran into the 3 h timeout (one repeating the same analysis verbatim).
- Useful signal: it admitted it couldn't solve extract-elf ("cannot find the reference solution values") — a worker that reports uncertainty can be escalated rather than looped.

→ [docs/08](docs/08-agentic-benchmark-core19.md) · [data/core19](data/core19)

## 2026-09-10 · llama.cpp on Vulkan wins the agent seat — via speculative decoding, not raw speed
`engine` `spec-decode` `llamacpp` `vllm`

- For one or two interactive agents at deep context (median ~66k), llama.cpp + MTP decodes 61.7 tok/s single-stream, 42 tok/s each for two streams; vLLM's MTP can't combine with pipeline parallel and TP needs matched cards.
- Without speculation the engines are close once vLLM is configured correctly — and vLLM batches far better: aggregate 28.4 → 346.8 (12×) where llama.cpp of that build peaked 82.6 at n=2, then 7.8 at n=4.
- That collapse traced in source as dispatch-bound: slot scheduling is one thread, one core at 100% while the GPU drew 70–90 W of a 330 W cap. Superseded: 434ddbb held 4 streams (72.7 aggregate).
- Keeping llama.cpp current beat every knob: prefill at 42k 766.9 → 916.5 tok/s (+19.5%) at the same 250 W; every knob (ubatch, KV type) moved ±5%, depth moved −42%.

→ [docs/02](docs/02-engines-llamacpp-vs-vllm.md)

## 2026-09-10 · vLLM's "3× slower at depth" was configuration, not the hardware
`vllm` `backend`

- Three defaults stacked: AMD's custom paged-attention kernel only runs ≤ 16k and the fallback decays (34.8 → 10.3 tok/s from 68 tokens to 35k); --enforce-eager costs ~72% of decode; prefix caching is off.
- Configured correctly: TRITON_ATTN + prefix caching (TTFT 30–33 s → 1.08 s, 65.7% hit rate) + trimmed CUDA graphs → unpatched vLLM 0.27.1 at 32.0 tok/s @ 35k (3 reps, 0.1% spread), level with llama.cpp without speculation.
- The default backend is load-bearing (3× at depth); AITER, patched onto RDNA4, was a wash — 29.1–32.8 tok/s, no faster than TRITON_ATTN.
- The same settings on a Qwen3.6-35B-A3B MoE held 81–82 tok/s at ~36k. Still missing for the agent seat: MTP under PP, TP with matched cards.

→ [docs/02](docs/02-engines-llamacpp-vs-vllm.md) · [docs/05](docs/05-rocm-vs-vulkan.md)

## 2026-09-10 · ROCm wins shallow, loses deep — mechanism still unknown
`backend` `llamacpp` `methodology`

- Same commit, both backends: ROCm prefills +53% at depth 0 (1,625 vs 1,064) but page-faults at 32,768 depth; Vulkan decodes faster at both depths (Vulkan 37.6 / 34.5 vs ROCm 24.5 / 20.3).
- The fault ("Memory access fault by GPU … Page not present") appeared only in llama-bench -d 32768 with q8_0 KV, second repetition; a "fixed" call from one clean run recurred on the next.
- September, one ruler, four images at 38.7k: ROCm loses ~21% prefill (675.1 / 684.3 vs 857.4) and decode 27.7 vs 31.4–31.5; the llama.cpp version itself moved prefill 22%.
- Shallow vs deep inverts the ranking: 244 tokens ROCm 318.6 vs Vulkan 250.8 (+27%); 38,724 tokens Vulkan 857.4 vs ROCm 675.1 (+27%).
- The earlier "16k cliff" explanation belongs to vLLM's ROCm backend — borrowed, not tested. Treat the mechanism as unexplained.

→ [docs/05](docs/05-rocm-vs-vulkan.md)

## 2026-09-10 · MTP n_max=2 is production; content type dominates acceptance
`spec-decode` `llamacpp` `models`

- The flag defaults to off: an MTP-capable GGUF was benchmarked for a whole session as "no difference" before anyone set --spec-type; correctly set, MTP took short-prompt generation ~21.8 → ~41–42 tok/s.
- Content type dominates everything: code acceptance up to 0.88 vs prose at best 0.51; prose plateaus near 2.9 mean accepted length at any n_max — a drafter tuned on one workload looks broken on the other.
- Single stream: DFlash2 n_max=4 fastest on code (79.0 vs MTP's 68.3), MTP n_max=2 best on prose (52.5); at 2+ streams MTP n_max=2 wins (42.2 / 79.1 vs 29.7 / 59.0) — DFlash2 is a net loss beyond one stream (and 4.65 GB more than no-spec).
- Aggregate never exceeded the best single-stream figure: concurrency shares capacity here, it doesn't add it.
- Higher acceptance ≠ faster — Turbo accepts 84% but decodes 59.1 vs heretic's 79% / 68.0 — and vendor MTP=2 guidance is right for mixed work, wrong for code-only (n4: +15%).

→ [docs/04](docs/04-speculative-decoding.md)

## 2026-09-10 · llama.cpp across two cards: capacity, not speed — and the tail tax
`multi-gpu` `llamacpp` `engine`

- For agent turns, one card won: -sm layer pipelined at ~50% utilisation each, and the real server decoded 26.2 vs 33.5 on one card; with the drafter it prefilled 740 (the drafter blocks half the pipeline).
- The tail tax: whichever card holds the last layers also gets the LM head and the whole MTP block, whatever -ts says — the 2:1 split left ~1.3 GB free on the 16 GB card vs ~9.2 GB on the 32 GB.
- The fix is --override-tensor: `output.*=Vulkan0` recovered headroom at −6.9% generation (34.51 vs 37.06); moving the MTP block too cost −12.7%.
- Row split won't load on Vulkan; tensor split loads at about half speed (568 tok/s prefill).
- On ROCm the 128k default-batch start OOM'd on the compute buffer (not KV): -b 1024 -ub 256 fitted at 668–673 tok/s. Using -ot instead of the batch fix passed a smoke test, then page-faulted mid-prefill.

→ [docs/03](docs/03-multi-gpu.md)

## 2026-09-10 · vLLM PP=2 is a capacity lever; TP=2 serves — until the pair is mixed
`multi-gpu` `vllm` `kernel`

- PP=2 works (the "single-GPU only" belief came from a TP deadlock, a different mode) and it is a capacity lever: uneven layer splits bought 3.6× the KV cache, 317,970 tokens at 52/12, with the last stage carrying the LM head at ~1.05 GB/layer vs ~0.53.
- TP=2 serves with RCCL 2.28.9 where upstream issues report deadlocks on 2.27.7: matched at 225 W, prefill 855.8 → 1,605.0 at 23k (+87%, 91% of the ceiling) and decode +39.5% at 3k and 23k.
- The 16 GB card caps TP's context (32,768 booting a 42,325-token pool); against the right baseline (single-card DFlash2 near 12k) that's 3.4× the context and 3.85 GB freed.
- The mixed pair dies in a kernel: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION in Tensile GEMM, always on the RX 9070 XT — "needs matched cards" is about kernel selection, not only memory; TP also runs at the slower card's pace (the R9700).
- One boot isn't enough: single-GPU decode across three boots read 34.20 / 34.19 / 23.52 (ROCm slow-start mode); one boot would have reported TP as +103%.

→ [docs/03](docs/03-multi-gpu.md)

## 2026-09-10 · A 250 W cap does not bound transients
`power` `methodology`

- Sub-millisecond peaks of 488 W (R9700) and 584 W (9070 XT) were measured under the 250 W cap, with a coincident 934 W combined peak, and no crash.
- Two event classes: hard locks (dead until the PSU switch) near vLLM PP/TP alternating load at ~558–610 W combined; self-recovering reboots as low as ~230 W (summed telemetry), where a per-card trace fits a load transition between cards, not either card's steady state.
- The UPS theory is dead (removed; events continued); the leading suspect is an ageing 1000 W PSU. A memory-clock cut to 1359 MHz with −50 mV coincided with reboots stopping — the discriminating test hasn't run.
- What the cap costs: llama.cpp + DFlash2 at 40k −8.3% prefill / −16.0% decode from 374/330 → 250 W; vLLM single card only −1.9% to −2.8% — plain decode is bandwidth-bound.
- Rules: log each card separately from a machine that survives the crash; sample faster than the question; a watchdog isn't armed until it printed a real non-zero reading.

→ [docs/06](docs/06-power-and-stability.md)

## 2026-09-10 · Nothing was disqualified; the real difference is how much they think
`models` `quality` `methodology`

- Four 27B candidates, none disqualified: the lowest TG anywhere was 23.3 tok/s at 184k, and no batching ladder collapsed (closest: Turbo IQ4_XS slowest stream 19.9 at 4 streams).
- PP doesn't separate the leaders: heretic and Turbo Q4_K_S within 0.9% at every depth (2.4k–184.2k); IQ4_XS 1–2% behind, Unleashed 2–3%.
- Turbo generates 6–8% slower at every depth — but thinks far less: on the code prompt, 335 thinking tokens vs a full 2048-token budget and zero answer (~6 s vs ≥30 s).
- Why forced length: natural answers differ ~5× (Turbo 92–173 vs heretic 461–721 tokens); natural-length aggregate mostly measured verbosity.
- Decision: Turbo Q4_K_S into the agentic A/B; the fine-tuner's separate non-MTP files aren't needed once MTP acceptance cleared 50%.

→ [docs/07](docs/07-model-qualification.md) · [data/qualification](data/qualification)

## 2026-09-10 · The agent framework was the biggest speed problem
`agents` `methodology`

- From 422 real agent turns: 17.5% prefill / 82.5% decode of wall time; median turn 85% decode, p90 97%; median prompt 874 tokens; completion median 624, max 24,531.
- The fixed prefix was tools, not personality: a 40,466-token cold prompt was 69% tool schemas; cutting two toolsets + tightening the prompt saved 5,322 tokens (~7 s per cold start).
- The second request nobody could see: Hermes' background_review (on by default) forked a full-context request (14,000–18,000 tokens) after every turn; prefill with both live fell to 330–516 tok/s; re-pointing it cut local cost to 87 tokens per turn.
- A systemd edit silently dropped -cram for two days (evictions 45 → 0 with it; warm turns became 4–12-token deltas) — caught by diffing the running process against the saved launch line.
- Reasoning plumbing decides the rest: the previous turn's output is always re-prefilled once (4-token re-rendered header; 98% of input tokens are cache hits), clients strip earlier reasoning by default (thinking saves generation only), and "medium" effort injects nothing — the owner's order is xhigh > low > medium.

→ [docs/10](docs/10-agent-harness-lessons.md)

## 2026-09-10 · ComfyUI's default cache grows to 100% of RAM
`comfyui` `models` `methodology`

- The default --cache-ram lets the inactive model cache grow to 100% of system RAM; on 32 GB that meant 98–229 s per image instead of ~25 s, with swap climbing.
- Capped (--cache-ram 2 4, smart memory on): 25–28 s per image, ComfyUI's anonymous RAM ~29 GB → 6.9 GB, partial UNet reloads in ~3 s.
- Three wrong fixes came first: pinned memory (no change), --disable-smart-memory (evicted the UNet on every prompt change, ~78 s reload), --highvram (stopped eviction, broke LoRA patching on the 16 GB card).
- Fitting the model: swap the fp8 Qwen3-VL-4B text encoder (5.0 GB) for int4 (2.68 GB); int4 convrot is ~55% the size of int8 — and NVFP4 is Blackwell-only, skip it on AMD.
- One agent workflow loaded the same UNet through a different node class, built a second 12.5 GB copy, and the OOM killer took out ComfyUI and the LLM server (7 minutes of swap thrash).

→ [docs/09](docs/09-comfyui-memory.md)

## 2026-09-10 · The rig, the ceilings, and every tool counting GPUs differently
`methodology` `models`

- Ceilings used throughout: prefill ≈ 1,772 tok/s (95.7 TFLOPS ÷ ~54 GFLOPs per token) and decode ≈ bandwidth ÷ model size (38.7 tok/s for a 16.52 GB Q4_K_M); raw llama.cpp decode reached ~81% of it.
- Every tool numbers the cards differently (ROCm/vLLM vs LACT vs nvtop); LACT accepts over-range caps and clamps them silently (375 W → 330 W on the R9700, 374 W on the 9070 XT) — confirm with rocm-smi --showmaxpower.
- Two container families were both called "therock" with incompatible ROCm layouts; setting HIP_PATH in the wrong one broke a JIT compile with a false "unsupported GPU architecture" error. Tags aren't identity: pinned-20260731 was built 06-13.
- Read the model, not the filename: Unsloth UD-Q4_K_XL reports Q4_K_S in the header; mmproj-f16 was BF16; MTP heads are an extra ordinary layer (blk.64) — verify by running --spec-type draft-mtp and reading acceptance.
- Linux traps that cost time: systemctl mask on an fstab unit skipped every local mount; /tmp is tmpfs (a power event erased a finished sweep); distrobox shares the host PID namespace; tmux kill-session prefix-matches.

→ [docs/01](docs/01-hardware-and-software.md)

# 07 — Model qualification: heretic vs Turbo vs a stock proxy (2026-09-10)

**Question:** should production move from one Qwen3.8-27B fine-tune to another? The fine-tuner's claim is a
*quality* claim (better output, much less thinking), so the real test is an agentic benchmark
([08](08-agentic-benchmark-core19.md)). This qualification decides which quant goes into that test, and whether
any candidate is disqualified on speed first.

Interactive chart with every measurement: [`data/qualification/turbo_qualification.html`](../data/qualification/turbo_qualification.html)
(open locally in a browser).

## Candidates

| Model | Source | File | Size |
|---|---|---|---|
| **heretic Q4_K_S** (current production) | trohrbaugh's "heretic" (abliterated) Qwen3.8-27B | Q4_K_S GGUF, MTP head present | 15.83 GB |
| **Turbo Q4_K_S** | DavidAU, *Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP* | Q4_K_S, Q8_0 MTP head | 17.54 GB |
| **Turbo IQ4_XS** | same repo | IQ4_XS, Q8_0 MTP head | 17.03 GB |
| **Unleashed Q4_K_M** (stock proxy) | outsourc-e, *Qwen3.8-27B-Unleashed* (Unsloth Dynamic 3.0 recipe) | Q4_K_M, MTP head present | 16.52 GB |

- **Turbo is larger at the same nominal quant** because its MTP head is Q8_0 and its output tensor is kept at
  16 bits.
- **Unleashed was added afterwards** as a comparison point for the "stock" recipe, because the Turbo files are
  believed to use the same recipe internally.

## Setup

- **Server:** the live production launch line ([02](02-engines-llamacpp-vs-vllm.md)) with **only `-m`
  changed** (and `-np` for the VRAM ladder). llama.cpp `434ddbb`, Vulkan, 262k context, q8_0 KV, 250 W cap.
- **Timing:** the three original candidates ran back-to-back in one session, with heretic re-measured as the
  control. Unleashed ran ~1 h later, after a **drift check** on the live production server: MTP code
  68.02 vs 68.01 tok/s (+0.01%), prefill at 42k 860 vs 855 (+0.6%). The later numbers are comparable.
- **Three passes:**
  - **A, MTP on:** VRAM at `-np 1–4`; 2048-token code and prose generation with per-request draft
    acceptance, at the template's default reasoning (`xhigh`); 1–3 stream ladder.
  - **B, no speculation:** prefill and generation at five depths (2.4k → 184k), thinking off; 1–4 stream
    ladder.
  - **C, forced length:** every request forced to exactly 768 generated tokens (`ignore_eos`) at 2.4k, 39k and
    129k, plus the 1–4 stream ladder.
- **Why pass C exists:** pass B let each model stop where it wanted, and the models write very different
  amounts (table below). Aggregate throughput was measuring verbosity. The gates use pass C.
- **Contamination audit:** distinct server task IDs in every server log matched the requests each script
  sent exactly. No outside traffic reached any measurement.

## The owner's criteria, in priority order

1. **PP (prompt processing) = responsiveness.** Higher wins, and it usually decides, unless a disqualifier
   fires.
2. **TG (generation) without MTP:** above 20 is usable, 30+ is the dream setup, **under 18 disqualifies**.
3. **True batching:** **a TG collapse under concurrency disqualifies.**
4. **MTP:** a bonus. More than 50% acceptance is the fine-tuner's own threshold for the MTP files being worth
   using.
5. **Test both shallow and deep context.**

## Results

### PP tok/s by depth, no MTP (TTFT in brackets)

| Depth | heretic Q4_K_S | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed Q4_K_M |
|---|---|---|---|---|
| 2.4k tokens | 990 (2 s) | 999 (2 s) | 978 (2 s) | 960 (2 s) |
| 14.8k | 1,012 (15 s) | 1,013 (15 s) | 994 (15 s) | 977 (15 s) |
| 38.7k | 896 (43 s) | 896 (43 s) | 882 (44 s) | 870 (45 s) |
| 128.9k | 622 (207 s) | 623 (207 s) | 616 (209 s) | 610 (211 s) |
| 184.2k | 523 (352 s) | 524 (352 s) | 519 (355 s) | 515 (358 s) |

### TG tok/s without MTP

| Depth | heretic | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed |
|---|---|---|---|---|
| 2.4k, forced 768 tokens | **34.6** | 31.9 | 32.5 | 33.2 |
| 38.7k, forced 768 | **31.9** | 29.7 | 30.2 | 30.8 |
| 128.9k, forced 768 | **27.0** | 25.4 | 25.7 | 26.2 |
| 184k, natural length (tokens) | 24.4 (486) | 23.3 (173) | 23.5 (133) | 23.9 (686) |

### Why forced length was needed: natural answer lengths on the same prompts

| Prompt target | heretic | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed |
|---|---|---|---|---|
| 2,500 | 721 | 128 | 148 | 917 |
| 16,000 | 461 | 125 | 240 | 540 |
| 42,000 | 573 | 141 | 208 | 679 |
| 140,000 | 597 | 92 | 160 | 757 |
| 200,000 | 486 | 173 | 133 | 686 |

### True batching, forced 768 tokens, `-np 4`: per-stream / aggregate tok/s (slowest stream)

| Streams | heretic | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed |
|---|---|---|---|---|
| 1 | 34.6 / 31.9 (34.6) | 32.1 / 29.8 (32.1) | 32.5 / 30.1 (32.5) | 33.2 / 30.6 (33.2) |
| 2 | 31.2 / 53.7 (31.1) | 29.2 / 50.6 (29.1) | 28.7 / 49.7 (28.6) | 29.6 / 51.1 (29.6) |
| 3 | 28.0 / 64.8 (27.6) | 26.4 / 62.0 (26.0) | 25.5 / 59.8 (25.1) | 26.2 / 61.2 (25.9) |
| 4 | 24.8 / **72.7** (23.8) | 23.6 / 69.6 (22.7) | 20.6 / 62.6 (**19.9**) | 22.2 / 66.8 (21.4) |

### MTP (n_max 2; code and prose at `-np 2`, streams at `-np 3`), template-default reasoning

| | heretic | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed |
|---|---|---|---|---|
| Code: tok/s | **68.0** | 59.1 | 57.5 | 53.1 |
| Code: acceptance / mean length | 79.4% / 2.59 | 83.9% / 2.68 | 85.1% / 2.70 | 58.0% / 2.16 |
| Prose: tok/s | **52.5** | 48.5 | 44.1 | 48.1 |
| Prose: acceptance / mean length | 49.6% / 1.99 | **60.0%** / 2.20 | 53.4% / 2.07 | 48.4% / 1.97 |
| **Thinking / answer tokens, code prompt** | **2048 / 0** | **335 / 1711** | **347 / 1699** | **2048 / 0** |
| Prefill @ ~39k (MTP loaded) | 855 | 854 | 840 | 832 |
| 1 stream | 61.5 | 54.7 | 53.2 | 50.9 |
| 2 streams (per / aggregate) | **41.9 / 78.6** | 38.2 / 76.3 | 25.1 / 50.2 | 28.0 / 56.0 |
| 3 streams (per / aggregate) | 22.4 / 65.3 | 21.8 / 65.0 | 20.9 / 62.0 | 19.6 / 57.4 |
| Acceptance under 3-stream load | 66.8% | 73.8% | 75.4% | 57.2% |

### VRAM after load (GiB, of 31.9)

| | heretic | Turbo Q4_K_S | Turbo IQ4_XS | Unleashed |
|---|---|---|---|---|
| MTP, `-np 1` / 2 / 3 / 4 | 25.9 / 26.3 / 26.8 / 27.2 | 27.5 / 27.9 / 28.4 / 28.8 | 27.1 / 27.5 / 27.9 / 28.4 | 26.5 / 27.0 / 27.4 / 27.9 |
| No MTP, `-np 2` / 4 | 24.2 / 24.5 | 25.6 / 25.9 | 25.2 / 25.5 | 24.8 / 25.0 |

## Where the gates landed

- **No model was disqualified.**
  - The lowest TG anywhere was 23.3 tok/s, at 184k.
  - No batching ladder collapsed: aggregate rose at every step for all four, and no stream fell below 18.
  - Turbo IQ4_XS came closest to the line: slowest stream 19.9 at 4 streams.
- **PP didn't separate the leaders.** heretic and Turbo Q4_K_S are within 0.9% at every depth. IQ4_XS is
  1–2% behind; Unleashed 2–3%.
- **TG:** Turbo is **6–8% slower** than heretic at every depth.
- **MTP:** both Turbo quants clear 50% acceptance on code and prose. heretic sits just under on prose (49.6%).
  IQ4_XS and Unleashed lose badly at 2 MTP streams (25.1 and 28.0 per stream vs 38–42).

## What it means

- **The big difference isn't speed; it's how much each model thinks before answering.** On the code prompt,
  heretic and the stock proxy both spent the entire 2048-token budget thinking and never started an answer.
  Turbo thought for ~340 tokens and then wrote ~1,700 tokens of answer.
  - At ~60 tok/s that's ≥30 s of thinking with no output, vs ~6 s.
  - So the stock-recipe thinking habit is the base model's, and Turbo's tuning changed it.
- **Break-even:** Turbo generates 6–8% slower, so it needs roughly that much less generated text per task to
  finish in the same time. The agentic benchmark measures that directly.
- **IQ4_XS:** its prefill cost was only 1–2% on this build, where the owner had seen much larger IQ-quant
  penalties before. It saves just 0.4 GiB and batches worse under MTP.
- **The fine-tuner's non-MTP files aren't needed,** because MTP acceptance cleared 50%.

**Decision (the owner):** Turbo Q4_K_S goes into the agentic quality A/B against heretic.

## Caveats

- **One run per cell.** The drift check suggests ~1% stability on this build.
- **Speed passes B and C ran with thinking off; the MTP pass ran with the template default (`xhigh`).**
  Production uses `medium`, which in these templates injects *no* reasoning instruction
  ([10](10-agent-harness-lessons.md)). Thinking-length differences at `medium` may be smaller.
- **The prompts are synthetic filler plus two fixed generation prompts.** They measure speed, not understanding.
- **A single fixed code prompt** is one data point for the thinking-length difference, not a distribution.

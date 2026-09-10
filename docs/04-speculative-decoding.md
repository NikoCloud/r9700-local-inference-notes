# 04 — Speculative decoding: MTP vs DFlash2

**Speculative decoding** means a cheap drafter proposes several tokens and the big model verifies them in
one pass. Two kinds were used on Qwen3.8-27B in llama.cpp:

| | MTP (multi-token prediction) | DFlash2 |
|---|---|---|
| What drafts | A prediction head built into the model | A separate small block-diffusion model (5 layers) |
| Flag | `--spec-type draft-mtp` | `--spec-type draft-dflash -md <drafter>` |
| Extra VRAM | Small (runs on the target weights) | The drafter model, ~3.85 GB |
| Status in llama.cpp | Stock | Merged upstream 2026-08-27 (PR #27342) |

**Production today:** MTP with `--spec-draft-n-max 2`.

## First trap: the flag defaults to off

In July an MTP-capable GGUF was benchmarked for a whole session with no difference between "MTP" and
"non-MTP". MTP had never been turned on: `--spec-type` defaults to `none`, and the model loads normally
with its heads unused. With the flag set, generation went ~21.8 → ~41–42 tok/s on short prompts, and
~20.7 → ~32–36 on an 11.9k-token prompt.

## Content type dominates everything

Draft acceptance depends on what's being written far more than on any setting. A DFlash2 sweep on the
production model, at ~35k depth with 3 reps per cell:

| n_max | Prose tok/s | Prose accept | Prose mean length | Code tok/s | Code accept | Code mean length |
|---|---|---|---|---|---|---|
| 3 | 49.2 | 0.51 | 2.51 | 69.1 | 0.84 | 3.53 |
| 4 | 46.4 | 0.43 | 2.72 | 76.4 | 0.88 | 4.50 |
| 5 | 43.5 | 0.39 | 2.94 | 72.7 | 0.79 | 4.92 |
| 7 | 35.2 | 0.29 | 3.03 | 71.1 | 0.75 | 6.17 |
| 9 | 33.7 | 0.27 | 2.91 | 71.0 | 0.74 | 6.14 |

- **The drafter's advertised ~5 mean accepted length was a code figure.** Prose plateaus near 2.9 no matter
  how deep the draft. Every earlier "we only get 2.7" observation was measuring prose.
- **Prose peaks at small n_max; code tolerates large.** A drafter tuned on one workload looks broken on the
  other. Always measure both.

## Drafter placement: same GPU as the target

With the DFlash2 drafter on the *other* GPU, PCIe ends up inside every verify step:

| Config (40k context) | Prefill | Decode |
|---|---|---|
| Drafter on GPU0, single GPU | 742 | **63.3** |
| No drafter, single GPU | 796 | 33.5 |
| No drafter, 2-GPU layer split | 1,088 | 26.2 |
| Drafter on GPU1, 2-GPU split | 740 | 46.4 |

The drafter costs ~7% of prefill on one card and buys +89% decode. `-devd Vulkan0` (drafter on the target's
GPU) was worth ~20% on its own.

## Single stream: DFlash2 vs MTP (build `434ddbb`, 2048 tokens, 250 W)

| n_max | DFlash2 code / prose | MTP code / prose |
|---|---|---|
| 0 (off) | 34.7 / 34.7 | — |
| 2 | 65.3 / 51.8 | **68.3 / 52.5** |
| 3 | 72.8 / 52.3 | 71.8 / 49.4 |
| 4 | **79.0 / 54.0** | 67.0 / 45.0 |
| 7 | 70.7 / 45.5 | — |

DFlash2 at n_max=4 is the fastest single-stream code configuration. MTP at n_max=2 has the best prose figure
of either method.

*Hypothesis that failed:* MTP was expected to be *more* consistent across content types. Its code-vs-prose gap
was wider at every tested n_max.

## Concurrency is where the choice flips

Measured with `-np 3`, per-stream / aggregate tok/s:

| Config | VRAM | 1 stream | 2 streams | 3 streams |
|---|---|---|---|---|
| No speculation | 26.2 GB | 34.7 | 31.1 / 62.2 | 25.6 / 76.8 |
| **MTP n_max=2** | ~26 GB | 61.7 | **42.2 / 79.1** | 22.5 / 65.7 |
| MTP n_max=4 | ~27 GB | 70.9 | 32.0 / 62.7 | 27.0 / 80.6 |
| DFlash2 n_max=4 | 30.8 GB | **79.1** | 29.7 / 59.0 | 25.2 / 72.7 |

- **DFlash2 is a net loss beyond one stream.** No speculation beats it per-stream and in aggregate at 2 and 3
  streams, while using 4.65 GB less.
- **MTP n_max=2 wins at 2 streams,** which is the production shape: two agents sharing one server, at 42 tok/s
  each.
- Aggregate never exceeded the best single-stream figure. Concurrency shares capacity here; it doesn't add it.

This reversed an earlier "DFlash2 supersedes MTP, settled" conclusion made on an older build (see
[MISTAKES](../MISTAKES.md)).

## Vendor guidance, tested

AMD and Alibaba's launch post recommends MTP=2 for the R9700 (measured on Windows/Vulkan). On this
Linux/RADV build:
- **Single-stream code:** MTP n_max=4 is 15% faster (70.9 vs 61.7).
- **Mixed or prose work:** n_max=2 is the best setting.

So the recommendation is right for a mixed workload and wrong for a code-only one.

## MTP acceptance across different fine-tunes

From the 2026-09-10 qualification ([07](07-model-qualification.md)), same harness, MTP n_max=2, `-np 2`,
2048 tokens:

| Model | Acceptance, code / prose | tok/s, code / prose | Acceptance at 3 streams |
|---|---|---|---|
| heretic Q4_K_S | 79% / 49.6% | 68.0 / 52.5 | 66.8% |
| Turbo Q4_K_S | 84% / 60% | 59.1 / 48.5 | 73.8% |
| Turbo IQ4_XS | 85% / 53% | 57.5 / 44.1 | 75.4% |
| Unleashed Q4_K_M | 58% / 48.4% | 53.1 / 48.1 | 57.2% |

**Higher acceptance didn't mean higher speed.** Turbo accepts more drafts but is slower overall, because the
base model generates more slowly. Acceptance is a property of the model and head together.

## Other notes

- **DFlash2 in vLLM** runs, and was fast solo: 73.6 tok/s against 48.7 with vLLM's MTP. But the drafter
  shrank vLLM's KV pool to ~1.2× concurrency, which defeats the reason to use vLLM.
- **vLLM's MTP can't run under pipeline parallel** (the drafter doesn't implement `SupportsPP`). Under TP it
  only ran out of memory on the mixed pair.
- **Drafter precision barely matters:** Q8_0 was as good as BF16.
- **Warm sampling (temperature > 0) made DFlash2 worse** in vLLM, not better.

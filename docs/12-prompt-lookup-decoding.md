# 12 — Prompt-lookup (n-gram) speculative decoding, and chaining it in front of MTP

> **Date:** 2026-09-11. Measured on heretic Q4_K_S, llama.cpp `434ddbb` Vulkan, the production server line
> (262k context, q8_0 KV, `-kvu`, 250 W). Forced 512-token generations, temperature 0, seed 42. Small samples;
> re-check before relying on the exact figures.

## The idea

Autoregressive decode on this card is memory-bandwidth bound with lots of spare compute
([01](01-hardware-and-software.md), roofline). Speculative decoding trades that spare compute for speed by
drafting several tokens and verifying them in one weight-read. MTP (a trained head on the model, n_max=2) is the
production drafter ([04](04-speculative-decoding.md)).

**Prompt-lookup / n-gram decoding** is a *free* drafter: it needs no model and no VRAM. When the recent tokens
match an earlier span in the context, it proposes the continuation of that span as the draft, then the target
verifies it exactly. It only helps when output overlaps the input — quoting a file back, editing a document,
reproducing boilerplate, RAG — which is much of what an agent does. llama.cpp exposes it via `--spec-type`
(`ngram-mod` here, defaults: match a 24-gram, draft 48–64 tokens). `--spec-type` takes a **comma-separated
list**, tried in order, first match wins — so `ngram-mod,draft-mtp` means "copy from context if there's a match,
otherwise fall back to MTP".

## Sanity gate: is it doing anything at all?

`ngram-mod` **alone** on a copy-verbatim prompt (maximal overlap), forced 512 tokens:

| | decode | draft acceptance |
|---|---|---|
| n-gram alone | **176 tok/s** | 0.979 (458/468 draft tokens) |
| no speculation | 34.9 tok/s | – |

**5.0× on the copy task.** The drafter accepts ~98% of a long draft per step. It works.

## Single-stream matrix (decode tok/s, forced 512, temp 0)

| prompt (overlap) | none | draft-mtp (prod) | ngram alone | **ngram→mtp** |
|---|---|---|---|---|
| copy (max) | 34.8 | 77.7 | 174.6 | **222.8** |
| code-edit (high) | 34.8 | 77.4 | 76.6 | **123.4** |
| prose (~zero) | 34.8 | 56.9 | 34.9 | **57.1** |

- **The chain `ngram-mod,draft-mtp` wins everywhere and loses nowhere.** On the realistic case — editing a file
  and quoting it back — it is **1.6× production MTP** and 3.5× no-spec; on pure copy 2.9× MTP.
- **n-gram *alone* is a trap for mixed work:** great on copy, but on prose it finds no match and drops to the
  no-spec floor (34.9). The chain keeps MTP as the fallback, so prose stays at MTP speed (57.1) for free.

## Correctness: is it lossless?

At temperature 0 every config *should* emit identical text (verified speculation accepts only the target's own
greedy token). Result:

- **copy, code-edit: identical across all four configs.**
- **prose: the two MTP configs diverged from no-spec; n-gram-alone stayed identical.**

This is not the n-gram drafter (its verification is exact) — it is the **MTP head**: the batched verify pass
computes marginally different logits than sequential decode, and on high-entropy prose that flips near-tie
argmaxes. It changes *which* valid greedy token is chosen, not quality, and **production already had this** (it
ran `draft-mtp`). Adding n-gram introduces nothing new. Worth knowing; not a regression.

## Concurrency: does the win survive, and does per-stream TG collapse?

`-np` is **capacity, not a per-stream tax.** A lone request runs at full speed regardless of how many slots the
server has:

| single stream | np1 | np2 | np4 |
|---|---|---|---|
| ngram→mtp, code-edit | 125.0 | 123.8 | 123.6 tok/s |

When you actually run N streams at once they share the card's bandwidth, so per-stream decode drops — this is
concurrency physics, not the `-np` setting, and np2 can't serve 3–4 concurrently at all (they queue, risking a
re-prefill). Code-edit, N identical streams on a server with `-np=N`:

| | np1 | np2 | np4 |
|---|---|---|---|
| draft-mtp per-stream / aggregate | 77.9 / 76.8 | 51.0 / 93.9 | 26.5 / 96.8 |
| **ngram→mtp** per-stream / aggregate | **124.0 / 121.0** | **72.7 / 132.8** | **39.2 / 139.0** |

- **The n-gram advantage holds at every concurrency:** ~1.5× per-stream and ~1.4× aggregate, np1 through np4.
- **No collapse.** Per-stream roughly halves as streams double — smooth batch contention, no cliff. At np4 the
  chain still holds 39 tok/s/stream (65 on copy), far above the project's 18 tok/s disqualifier; even plain MTP
  holds 26. Neither config disqualifies.
- **Aggregate is near its ceiling by np2** (MTP 93.9 → 96.8; chain 132.8 → 139 going np2→np4): the card's decode
  throughput is saturated, so extra slots split the same total rather than adding much.

## The `-np` cost, measured

Under `-kvu` (unified KV), raising `-np` is nearly free:

| | np1 | np2 | np4 |
|---|---|---|---|
| n_ctx_slot | 262144 | 262144 | 262144 |
| GPU0 used | 26.6 GiB | 27.1 GiB | 28.0 GiB |

- **No per-slot context split:** each slot can still address the full 262k. The 262k is a **shared pool** (if it
  were replicated per slot, np4 would add many GiB of KV, not 1.4). So `-np` is capacity for more concurrent
  *sessions* without re-prefill, not a division of the context window.
- **The only real constraint:** if several agents run very deep contexts *simultaneously*, they share the 262k
  budget. Core-19 agent contexts ran a 10–40k median, so four fit with room to spare.

## What went to production (2026-09-11)

Changed the live `qwen38.service` override from `--spec-type draft-mtp -np 2` to:

```
--spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 -np 4
```

Everything else unchanged. Live smoke on an overlap-heavy request: **137.8 tok/s**, n-gram accepting 14-token
drafts (0.63). Rollback is the backed-up override (`draft-mtp`, `-np 2`).

Rationale: strictly faster on agent-overlap work, free on prose, lossless (the n-gram verify is exact; the MTP
prose-flip predates this), survives concurrency, and `-np 4` is near-free headroom for the planned swarm workers
plus the two always-on agents.

## Engine A/B: Nathan Wilson's Vulkan fork gives the 27B nothing (2026-09-11)

A hobbyist ecosystem has grown around Qwen3.8-Flash-Next on AMD Strix Halo (Halogen, EngramHalo, Nathan
Wilson's Vulkan fork). Most of it is gfx1151 + unified-memory specific, but Nathan's fork is *Vulkan* — the
same backend as production here — so it was the one worth testing on gfx1201. Built `release/v0.7.5-staging`
(213 commits ahead of upstream), evicted production, and A/B'd it against the production build (`434ddbb`) on
heretic Q4_K_S, one stream, forced 512-token decode. Chart: [data/core19/engine_ab.html](../data/core19/engine_ab.html).

| config | decode 2k/32k/128k | prefill 2k/32k/128k | MTP acc |
|---|---|---|---|
| Nathan v0.7.5, no-spec | 34.4 / 31.3 / 24.2 | 979 / 829 / 446 | – |
| production 434ddbb, no-spec | 34.4 / 31.2 / 23.9 | 1045 / 843 / 443 | – |
| Nathan v0.7.5, ngram→mtp | 52.0 / 40.0 / 27.0 | 959 / 787 / 419 | 0.36 |
| **production 434ddbb, ngram→mtp** | **59.3 / 46.4 / 28.2** | 989 / 803 / 418 | 0.41 |

- **No-spec decode is identical** — the fork does not move 27B dense generation.
- **Production is slightly ahead on prefill** and **wins with the PLD chain** (59.3 vs 52.0 at 2k, higher
  acceptance). So the fork is equal-to-slightly-worse on every axis for this model.
- **Why:** its real work is in the DeepSeek-V4-Flash / Qwen3-Next branches (`dsv4-*`, `qwen4exp`), not general
  dense 27B. `master` was 0 commits ahead of upstream — the perf lives only in the feature/release branches.
- **Consistency check:** PLD acceptance here (0.34–0.44) is low because these were low-overlap filler prompts;
  that matches the overlap-dependence measured above. The engine harness and the PLD findings agree.

**Verdict: the production build stays.** Nathan's fork (and the wider Strix Halo stack) only pays off for the
Flash-Next / DeepSeek-V4-Flash model class, which wants unified memory this discrete card doesn't have.

## Caveats and follow-ups

- Single greedy runs at temp 0; real agents sample. The chain is still lossless in distribution, but the exact
  tok/s will differ with sampling and with real (non-repeated) prompts.
- `ngram-mod` drafts long spans (48–64) on a recurrent-hybrid model; a rejected long draft costs a larger verify
  and a state-checkpoint rollback. Not isolated here — watch it if long-draft configs misbehave.
- The idea and the fact-check came from a chat with another assistant; its concept was right, but several
  specifics were wrong (FP16 TFLOPS ~2× too high, "no per-request toggle", grammar "injects" tokens for free in
  llama.cpp — it doesn't). Measure, don't quote.

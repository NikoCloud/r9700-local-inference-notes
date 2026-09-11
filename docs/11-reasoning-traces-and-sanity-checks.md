# 11 — Sanity checks, a loop A/B and what the reasoning traces show

> **Date:** 2026-09-11. Small samples (1–4 runs per cell) at the models' default temperature of 1.0. Treat every
> rate here as a first measurement, not a settled property of either model.

The first Turbo Core-19 run spent **1 h 50 min on its first task** (77 steps, context grown to 98k tokens,
cycling through variations of one idea) and was killed ([08](08-agentic-benchmark-core19.md)). Before re-running
the benchmark, the owner asked for a quick sanity check with the reasoning streamed live, to make sure the model
was working as intended.

## The sanity harness

Three tasks, sent straight to the Core-19 server line (`127.0.0.1:8085`, same flags as the benchmark). The
request shape copies Core-19's agent: **no `max_tokens`, no temperature or sampler overrides, no
`reasoning_effort`**, earlier reasoning not sent back. Script: [scripts/core19/sanity_live.py](../scripts/core19/sanity_live.py).

| Task | What it tests | How it's graded |
|---|---|---|
| 1. Poem | Instruction following under coupled constraints: exact title, 5×4 stanzas, AABB rhyme, fixed first words (Wind/Tick/Rust/Chime/Stop), 140–160 words, the word "time" banned, ≥ 3 "thee/thou", last line a question | Automatic checks, **except rhyme**, which was only printed for eyeballing (see below) |
| 2. Bug hunt | Two small functions with 5 planted bugs (mutable default, `<` vs `<=`, in-place sort, index out of range at p=100, missing `ValueError`) | 12 hidden tests on the corrected module |
| 3. Physics build | Stdlib-only 2D elastic-collision simulation with gravity; energy within 0.5% | The checker **recomputes energy and overlaps from the printed initial and final states**, so the numbers can't be faked; failures are fed back for up to 3 fix turns |

## Results

| | Turbo Q4_K_S | heretic Q4_K_S |
|---|---|---|
| **Bug hunt** | 13/13 · **1,314 tokens · 23 s** | 13/13 · 30,311 tokens · 544 s |
| **Physics build** | 11/11 on turn 1 · **15,676 tokens · 275 s** · energy drift 0.17% | 11/11 on turn 1 · 82,193 tokens · 1,658 s · energy drift **0.0000%** |
| **Poem** | **Reasoning loop, cancelled** after 33,539 tokens (below) | see the loop A/B |

Both models passed both code tasks. Turbo was **23×** cheaper on the bug hunt and **5×** on the simulation.
Heretic's simulation conserved energy exactly (it built event-driven collision timing and chased every
remaining drift source in its reasoning); Turbo's stayed comfortably inside the bar.

## The poem loop

Turbo's poem reasoning locked into a cycle:

- it fixed one line ("Till finally it ceases;") and swapped only the rhyme word on the line before it
- **656 attempts, cycling through just 7 words** (boom, call, ring, sing, sound, strike, toll), repeating from
  attempt ~15 onwards
- it never considered changing the fixed line, and produced no answer before it was cancelled

The same shape appeared in the killed Core-19 run: many surface variations, one unquestioned assumption.

### Is it the KV cache, the model, or the prompt? (loop A/B)

Same poem prompt, **3 runs per condition**, stopped only if a reasoning line repeated 30+ times (no token cap).
Script: [scripts/core19/loop_ab.py](../scripts/core19/loop_ab.py).

| Condition | Loops | Avg tokens | Avg time | Decode | MTP acceptance |
|---|---|---|---|---|---|
| A: Turbo, q8_0 KV, 262k (the benchmark line) | 0/3 | 6,057 | 106 s | 57.3 tok/s | 0.81 |
| B: Turbo, **f16 KV**, 131k (so f16 fits) | 0/3 | 6,488 | 109 s | **59.7 tok/s** | **0.85** |
| C: heretic, q8_0 KV, 262k | 0/3 | 14,436 | 235 s | 61.5 tok/s | 0.71 |

- **The loop is intermittent, not guaranteed:** Turbo looped **1 time in 4** on this prompt with q8_0 KV
  (counting the sanity run); heretic 0 in 3. That's too few runs to call the loop Turbo-only.
- **KV compression didn't cause it** (0/3 either way). Turbo's model card notes that its example generations
  used "no cache compression of any kind", which is why it was tested.
- **Side finding:** f16 KV gave Turbo **+2.4 tok/s and +4 points of MTP acceptance** in these 3 runs. Plausibly
  less numerical noise improves draft agreement. Unconfirmed.
- On this single-shot task Turbo's "thinks less" claim held: **2.4× fewer tokens, 2.2× faster**.

## What the traces show

All reasoning was saved (see [data/traces/](../data/traces/)). Two detectors were built and validated by hand
against the text: [scripts/core19/analyze_traces.py](../scripts/core19/analyze_traces.py).

### The rhyme rule, audited by hand

The grader never checked rhyme; it printed each stanza's end words and nobody read them. Reading them:

| Model | Stanzas actually AABB |
|---|---|
| heretic (3 poems) | **15/15** |
| Turbo (6 poems) | **~8/30**. Three of the six poems were **ABAB throughout**, and one more had an unrhymed stanza |

Near-rhymes are judgement calls. The "8/8" automatic scores for Turbo's poems therefore overstate them (a
grader mistake, recorded in [MISTAKES](../MISTAKES.md)).

### Self-verification: the checks run, but the answers are wrong

Both models write checklists into their reasoning. Turbo's checklists often approve things that are false:

- **Word counts per line:** against the real count of the line they annotate, **Turbo was exact on 59% of 145
  lines, heretic on 91% of 117** (ten random rows per model were checked by hand; the parser was right on all
  twenty).
  - Heretic numbers every word ("Wind(1) gentle2 blow3 … =6"). Turbo estimates per line, mostly undercounting
    by 1–2, and the errors add up: one draft was declared "141 words ✓" at a real 159.
- **Rule checks:** in one run Turbo counted "thou" twice and approved "3 thee/thou. Good." for a draft with 2,
  approved "Stanza 1: boy/today (A), stand/near (B) – Yes" for a stanza that doesn't rhyme, and labelled an
  ABAB stanza as passing AABB. In the same pass it did catch a missing question mark and one bad rhyme, so the
  checks aren't useless, just unreliable.
- **The same failure in Core-19:** in four of Turbo's five genuine attempt-1 failures it declared the task
  complete when it wasn't ([08](08-agentic-benchmark-core19.md#failure-analysis)), e.g. "the test script exited
  with code 0, confirming the bypass works" when no alert had fired.

### "Reaches the right answer, then talks itself out of it"

Qwen3.6-27B, an earlier model in the family, often did this: its reasoning reached the correct conclusion and
then argued itself away from it before answering. The owner asked whether Qwen3.8 does it too.

**Not found in these traces.**
- **Poems:** every complete draft in the reasoning was graded with the final checker. No final answer scored
  below an earlier draft in any of the 9 finished poems.
- **Bug hunts:** each planted bug was located where it was first *diagnosed* (outside copied code), and no
  diagnosis was followed by a dismissal. Heretic had diagnosed all five within the first **7.8%** of its
  reasoning; the other ~28k tokens were deliberation over spec ambiguities it correctly declined as non-bugs,
  plus hand-verification of the fix.

Nine poems and two bug hunts can't rule the quirk out, only show it isn't common on these tasks.

### Two reasoning styles

Visible even while streaming:
- **heretic** writes clipped shorthand ("Need count words. Need AABB…"), questions the rules before drafting
  ("Word count likely whitespace tokens?"), plans, and counts obsessively.
- **Turbo** restates the rules as a numbered list, then drafts whole stanzas immediately and fixes them ("Wait,
  'mechanic stray' doesn't rhyme with 'decay.' Let me fix.").

Both start from the same weights: Turbo's first build stage is the same heretic decensor of Qwen3.8-27B. The
style difference therefore comes from Turbo's later training, which its card describes as including
"reformatting the thinking block" and light Claude Opus and Fable reasoning traces.

## Why the two GGUFs differ in size and speed

Read from the tensor headers (neither is an Unsloth "UD" dynamic quant, despite a belief that Turbo was):

| | heretic Q4_K_S | Turbo MTP-Q4_K_S |
|---|---|---|
| Quantised by | mradermacher (i1 importance matrix) | DavidAU (own "NEO-CODER" importance matrix) |
| Layer mix | standard llama.cpp Q4_K_S | **identical**, same Q5_K counts |
| Output head | Q6_K, 995 MiB | **BF16, 2,425 MiB** |
| MTP block | Q4_K | **Q8_0** |
| Total tensors | 14.73 GiB | 16.32 GiB (+1.59 GiB) |

- The +1.59 GiB matches the measured VRAM gap (+1,633 MiB).
- A 16-bit output head is read for every generated token, a plausible cause of Turbo's slower decode
  ([07](07-model-qualification.md), [08](08-agentic-benchmark-core19.md#speed-during-the-run)).
- The Q8_0 MTP block (and the cleaner output head) plausibly explain Turbo's higher MTP acceptance (0.84 vs
  0.71 in Core-19). A Turbo card table reports a KL divergence of 0.0025, but against its *own previous build
  stage*, not against base Qwen3.8; Turbo's distance from the base model isn't reported.

## Open follow-ups

- Rhyme needs a real check (the automatic grader skipped it).
- Loop rate needs more runs (dozens, not 3–4) before calling it a property of the fine-tune.
- Precision vs training: requantise heretic with a BF16 output head and Q8_0 MTP block and see whether its MTP
  acceptance rises to Turbo's.
- Measured KL divergence of both models against base Qwen3.8 on the same text.

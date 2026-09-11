# Open problems

What is still broken, unexplained, or waiting on hardware, as of 2026-09-10. Resolved items live in
the topic docs; this list is what someone picking this up would still have to deal with.

## vLLM on this hardware

| Problem | What's known | What would unblock it |
|---|---|---|
| **Default vLLM is ~3× slower at depth** | Worked around, not fixed upstream. AMD's custom paged-attention kernel is limited to ≤ 16k context, and the default `ROCM_ATTN` backend's fallback above that is slow. With `--attention-backend TRITON_ATTN`, CUDA graphs on (trimmed capture sizes) and prefix caching, unpatched vLLM 0.27.1 does ~32 tok/s at 35k. None of those three settings are defaults. | Defaults that suit RDNA4. Until then, set all three explicitly. |
| **AITER on RDNA4 needs local patches** | vLLM enables AITER only when `get_cdna_version() > 2`, which no RDNA part can satisfy; vLLM 0.22 didn't gate it, so it's a regression. With that gate patched plus a stage-count clamp (for a 256-byte shared-memory overshoot), AITER unified attention works, but it's no faster than `TRITON_ATTN`. Likely root cause of the overshoot: AITER's `_LDS_CAP_BYTES` table has no entry for `gfx1200`/`gfx1201`. | Two upstream reports: the gate regression, and the missing capacity entry. Low priority, since `TRITON_ATTN` already matches it. |
| **Bimodal decode speed per process start** | Another R9700 owner reports vLLM decode landing in a fast (~33) or slow (~26 tok/s) state at process spawn, ~33% slow, recoverable only by restarting (ROCm/ROCm#6347, in triage). Any single vLLM decode number here carries ~21% uncertainty. | AMD fix. Meanwhile, restart-and-repeat before trusting a vLLM decode number. |
| **Tensor parallel on a mixed pair** | TP=2 on R9700 + RX 9070 XT dies with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` in Tensile GEMM kernels, always on the 9070 XT, across two kernel configs. Routing AWQ through Triton did not help. | A second R9700, or a hipBLASLt kernel-selection fix. |
| **MTP + pipeline parallel** | vLLM's MTP drafter doesn't implement `SupportsPP`, so speculation is unavailable in the only parallelism mode that fits a mismatched pair. | Upstream support, or matched cards for TP (where MTP was only blocked by memory). |

## llama.cpp

| Problem | What's known | Next step |
|---|---|---|
| **ROCm page fault at depth with quantised KV** | `Memory access fault by GPU … Page not present`, on llama-bench `-d 32768` with q8_0 KV, on the *second* repetition only. rocWMMA reduced but didn't fix it. Never reproduced through the real server. | File upstream with the sharp repro (synthetic depth priming only). |
| **Local vision + speculative decoding patch** | Upstream `master` still fails image input with a separate draft model (`failed to process mtmd chunk`). The local patch skips M-RoPE embedding rows in the draft cache and zero-fills the gap. Upstream is taking a different approach (PR #24669, open). See [patches/](patches/). | Re-evaluate when #24669 merges; the patch will likely conflict. |
| **Multi-slot scaling on older builds** | Build 200 collapsed from 82.6 to 7.8 tok/s aggregate at 4 slots. Build `434ddbb` holds (72.7 tok/s aggregate at 4 forced-length streams). The cause of the old collapse was never pinned down. | Mostly moot on current builds; don't reuse old `-np` guidance. |

## Power and stability

| Problem | What's known | Next step |
|---|---|---|
| **Hard power-offs (manual PSU reset needed)** | Tracked vLLM PP/TP "see-saw" load and a rapid overclock sweep, i.e. repeated fast transitions. Not reproduced since moving to single-card llama.cpp. | A larger ATX 3.1 PSU is on order. vLLM TP stays off until it's in. |
| **Self-recovering reboots** | One per-card trace fits the second card spinning up while the first is already at its cap. Transients reach 1.9–2.3× the power cap. A memory-clock reduction coincided with them stopping. | Retest the memory clock *alone* (keep undervolt and cap) to separate signal-integrity margin from transient size. |
| **An overclock adopted without isolating its failure mode** | A memory-clock setting that once hard-locked the box was re-validated for ~1 minute in a cooler room with a longer cooldown, then used for weeks. | The untested voltage-margin retest. |
| **Telemetry that contradicts a reboot** | One capture shows continuous 150–250 ms power samples across a confirmed new boot ID. A cold boot can't complete inside that polling gap. | Unexplained. Check for a warm-reboot path, or a watcher bug. |

## Models and quality

| Problem | What's known | Next step |
|---|---|---|
| **Five Turbo Core-19 tasks lost to a mirror outage** | Both arms finished ([08](docs/08-agentic-benchmark-core19.md)). Five Turbo tasks (all `ubuntu:24.04` images) never reached the agent because the plain-HTTP Ubuntu apt mirror hung overnight and Terminus-2's tool install has a 120 s limit per command. Heretic passed all five the previous afternoon. | Re-run only those five Turbo tasks once the mirror is healthy. Even if all pass, Turbo reaches 14/19 vs heretic's 16/19. |
| **Turbo's reasoning loops** | 1 loop in 4 on a constrained poem at temp 1.0 (heretic 0 in 3), plus a 2 h spiral on a Core-19 task in a killed first run. q8_0 vs f16 KV made no difference in 3 runs each ([11](docs/11-reasoning-traces-and-sanity-checks.md)). | Dozens of runs per condition before calling it a property of the fine-tune. |
| **Rhyme and self-check accuracy not auto-graded** | The sanity grader skipped rhyme; the rhyme and word-count results in docs/11 are from a manual audit and a hand-validated parser. | A scored rhyme check (pronunciation dictionary) and word-count check. |
| **Why Turbo's MTP acceptance is higher** | Turbo's GGUF has a BF16 output head and a Q8_0 MTP block (heretic: Q6_K and Q4_K). Its card's KL figure is measured against its own previous stage, not base Qwen3.8. | Requantise heretic with the same output-head and MTP-block precision and re-measure acceptance; measure both models' KL against base on the same text. |
| **Concurrency realism (partly done)** | Speculative-decoding concurrency measured at np 1/2/4 ([docs/12](docs/12-prompt-lookup-decoding.md)): per-stream TG halves smoothly with stream count (no collapse), aggregate saturates by np2, and the n-gram→MTP chain keeps ~1.5×/stream throughout. Still not done: a full Core-19 *agent* run under real concurrency (two agents, distinct tasks). | Run Core-19 with `--concurrency 2` on both models. |
| **Speculative-decoding follow-ups** | Production now runs `--spec-type ngram-mod,draft-mtp -np 4` ([docs/12](docs/12-prompt-lookup-decoding.md)). Open: (a) `ngram-mod` drafts 48–64 tokens on a recurrent-hybrid model — a rejected long draft costs a bigger verify + a state-checkpoint rollback, not isolated here; (b) MTP flips near-tie greedy tokens on high-entropy prose (batched-verify logit noise), harmless but real; (c) exact gains under real sampling (not temp 0) and non-repeated prompts. | Long-draft stress test; measure the chain under the production sampler on real agent traces. |
| **A ~2B worker model** | MiniCPM5-2B on a 6 GB laptop GPU: 2/19 on Core-19 attempt 1 (preliminary). Speed there is card-limited. | Quality with a critic in the loop, and the same model on the intended 16 GB card. |
| **Reasoning effort in daily use** | The owner runs `medium` day to day. In these Qwen3.8 templates `medium` injects *no* reasoning instruction, while `xhigh` and `low` do. Owner's experience: medium generates the fewest tokens and gives the worst deliverables; low is close to xhigh in tokens. The benchmark only tests xhigh. | A `medium` arm, if the owner wants the practical comparison (Turbo @ xhigh vs heretic @ medium). |
| **No direct quality metric for quants** | KL-divergence against a BF16 reference can't run: BF16 27B doesn't fit 48 GB VRAM / 32 GB RAM. | More RAM, or borrowed hardware. |
| **Reasoning replay in the agent framework** | Hermes strips earlier reasoning by default (`model.reasoning_echo: false`). Turning it on would let models keep their earlier reasoning, at a large context cost. That's also where a short-thinking model saves most. | Owner decision; if tried, measure prompt growth per turn from server logs. |

## Hardware limits

- **32 GB system RAM** is the wall for ComfyUI video models, host-side prompt caching, and MoE models whose
  lookup tables would otherwise live in RAM. 64 GB+ is the most valuable upgrade on the list.
- **A second R9700** would enable TP=2 with matched cards (and MTP under TP), but it puts RAM and VRAM back
  at 1:1. That's the ratio that already makes model staging painful, so it would make the RAM problem
  worse.

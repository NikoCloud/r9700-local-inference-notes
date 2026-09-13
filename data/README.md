# data/

## `qualification/`: the 2026-09-10 four-model speed qualification

Raw JSON written by the scripts in [../scripts/qualification](../scripts/qualification). Machine-specific paths are
generalised; values are untouched.

| File | Pass | Models |
|---|---|---|
| `results.json` | A: MTP on (VRAM ladder, code/prose generation with draft acceptance, prefill @ ~39k, 1–3 stream ladder) | heretic Q4_K_S, Turbo Q4_K_S, Turbo IQ4_XS |
| `results_nospec.json` | B: no speculation (prefill/generation depth ladder 2.4k → 184k at natural length, 1–4 stream ladder, 2 × ~33k) | same three |
| `results_fixedlen.json` | C: forced 768-token generation (`ignore_eos`) at 2.4k / 39k / 129k and 1–4 streams | same three |
| `results_unleashed_mtp.json`, `_nospec.json`, `_fixedlen.json` | A, B, C | Unleashed Q4_K_M (added about an hour later) |
| `results_unleashed_drift.json` | Control re-check on the live production server before the later window: MTP code tok/s and prefill @ 42k vs the first window | — |
| `turbo_qualification.html` | Interactive chart of all of the above (open in a browser; no network needed except fonts) | all four |

**Rebuild the chart:** `python scripts/qualification/build_page.py`

**Field notes:**
- `*_vram_gib*` is GPU0 used memory right after load (GiB).
- **`accept.pooled`** is accepted ÷ generated draft tokens, summed over that request's own server-log lines;
  `mean_len` is the server's mean accepted length.
- **In pass A, `reasoning_chunks` / `content_chunks`** are streamed deltas carrying thinking vs answer text. In
  llama.cpp that's one token per delta for these requests.
- **`tail_distinct_word_ratio`** (pass B) is a crude degeneration check: distinct ÷ total words in the last 300
  characters of output.
- **`forced_length_honored`** (pass C) is true when `completion_tokens == max_tokens`.
- **`disqualified_accept_lt_0.50`** (pass A) is a script flag that does *not* match the owner's criteria (MTP
  acceptance was a bonus, not a gate). It's kept as written; see [MISTAKES](../MISTAKES.md).

## `core19/`: the 2026-09-10/11 agentic A/B ([docs/08](../docs/08-agentic-benchmark-core19.md), [docs/11](../docs/11-reasoning-traces-and-sanity-checks.md))

| File | What it is |
|---|---|
| `core19_ab.html` | Interactive chart page (per-task outcomes, minutes and tokens per task, speed and MTP acceptance by depth). Open in a browser; fully offline. |
| `core19_report.json` | Per task, per arm, per attempt: reward, exception, setup-error flag, agent minutes, steps, input/cached/output tokens; plus arm summaries. From `scripts/core19/c19_report.py`. |
| `per_trial_server_stats.json` | Per trial, from the server log: model calls, peak context, decode/prefill tok/s, MTP acceptance, generated tokens. |
| `speed_by_depth.json` | Token-weighted decode/prefill/acceptance per context-depth bin, both arms. |
| `sanity_results_turbo.json`, `sanity_results_heretic.json`, `sanity_timeline_turbo.json` | The three-task sanity check: per-call stats and checks, and a timestamped timeline. |
| `loop_ab_results.json` | Loop A/B: 3 poem runs each for Turbo q8_0 KV, Turbo f16 KV, heretic q8_0; includes each finished poem. |
| `quirks_report.json` | Output of `scripts/core19/analyze_traces.py` (draft reversal, per-line count accuracy, stated totals, bug-diagnosis positions). |
| `core19_minicpm5_laptop_partial.json` | **Preliminary** MiniCPM5-2B laptop run, snapshot taken while attempt 2 was still running. |
| `pld_matrix.json` | Single-stream n-gram / MTP matrix ([docs/12](../docs/12-prompt-lookup-decoding.md)): 4 configs × copy/code/prose, forced 512 tokens, with the output-identity result. From `scripts/core19/pld_test.py`. |
| `pld_concurrency.json` | Concurrency scaling (per-stream and aggregate decode) at `-np` 1/2/4 for draft-mtp vs ngram→mtp, the single-stream-per-np check, and the KV/VRAM figures. Assembled from the `scripts/core19/pld_conc.py` run outputs. |
| `engine_ab.html` / `engine_ab_results.json` | Engine A/B ([docs/12](../docs/12-prompt-lookup-decoding.md#engine-ab-nathan-wilsons-vulkan-fork-gives-the-27b-nothing-2026-09-11)): Nathan Wilson's Vulkan fork (`v0.7.5-staging`) vs the production build on heretic, prefill+decode by depth, no-spec and PLD chain. Interactive chart + raw results. From `scripts/core19/engine_ab.py`. |

**Field notes:**
- **`infra_error`** is true only when the trial raised `RuntimeError` *and* the agent never started. A
  `RuntimeError` after the agent ran (e.g. the agent killed its own container) counts as a model failure.
- **`git-leak-recovery`** for heretic is the smoke run's result, reused by the runner (`reused_from_smoke`).

## `vllm-mxfp4/`: the 2026-09-12 RDNA4 fp8-WMMA qualification ([docs/13](../docs/13-vllm-mxfp4-w4a8-rdna4.md))

`Launch80/Qwen3.8-27B-PARO-MXFP4` on one R9700, TP=1, via the radiance vLLM container. Written by the
harnesses in [../scripts/vllm-mxfp4](../scripts/vllm-mxfp4).

| File | What it is |
|---|---|
| `vllm_ab_nospec.json` | PP + TG by depth (2k/32k/60k), compiled, no speculation. **PP valid** (unique random prompts, no cache hits); TG valid (content-independent). |
| `vllm_ab_dflash2_spec5.json` | Same depths with the DFlash2-FP8 drafter at `SPEC=5`. **PP valid; the decode column is NOT** — `ignore_eos` on random filler made the model loop, inflating acceptance to 88–100%. Kept as the record of that trap. |
| `vllm_ab2_dflash2_spec5_real.json` | Realistic tasks (novel prose, code edit) on real-prose context. **Decode valid** (acceptance 47–63%, `tail_distinct` 0.74–0.82); **PP is NOT** — shared prefixes gave a 45.6% prefix-cache hit rate. |
| `vllm_conc_spec5.json` | First concurrency ladder, 8k context, N=1→8 plus a depth probe. **Superseded**: the aggregate is an end-to-end figure diluted by prefill, and `MAXSEQS=8` capped the ladder. Kept because the TTFT collapse (2.45 s → 112.7 s) is real and is the interactive-usability limit. |
| `vllm_conc2_spec5_ms96.json` | Concurrency ladder, DFlash2 `SPEC=5`, `MAXSEQS=96`, using `scripts/harness/conc_test.py`'s method. Peak **293 tok/s @ n=48**. |
| `vllm_conc2_nospec_ms84.json` | Same ladder, no speculation, `MAXSEQS=84` (the Mamba-block ceiling). Peak **331 tok/s @ n=64**; near-perfect fairness to n=16. |

**At the 330 W cap** ([docs/13 §6b](../docs/13-vllm-mxfp4-w4a8-rdna4.md)) — the R9700 raised to its
330 W ceiling with the undervolt and memory OC kept, plus llama.cpp production re-measured the same
hour with the same harnesses so the comparison is matched on power, harness and date:

| File | What it is |
|---|---|
| `vllm_ab_spec5_330w.json` | PP by depth, SPEC=5, byte-identical config to `vllm_ab_dflash2_spec5.json` (KV 5.13 GiB / 103,268 both times). **PP valid**, decode column is the degenerate-loop artifact. |
| `vllm_ab2_spec5_330w_real.json` | Realistic-task decode at 330 W. **Decode valid**, PP contaminated by prefix caching. |
| `vllm_conc2_nospec_ms84_330w.json` | Concurrency ladder at 330 W. Peak **385.8 tok/s @ n=24**, all 24 resident and fair (16.07/16.10). |
| `vllm_ab_prod_llamacpp_330w.json` | llama.cpp production PP by depth at 330 W, incl. 128k (which vLLM cannot reach at TP=1). Decode column is the artifact — and worse here, because production's n-gram PLD drafter is near-perfect on a repeating loop. |
| `vllm_ab2_prod_llamacpp_330w_real.json` | llama.cpp production realistic-task decode at 330 W. |
| `vllm_conc2_prod_llamacpp_330w.json` | llama.cpp production concurrency at 330 W (`-np 4`): peak **95.7 @ n=2**, flat thereafter, per-stream 70.1 → 12.75 with min 5.73 / max 35.26 at n=16. |

**Vision A/B** ([docs/13 §6c](../docs/13-vllm-mxfp4-w4a8-rdna4.md)) — five webtoon pages,
800 x 3,520-7,190 px, identical images and prompt, both engines at 330 W:

| File | What it is |
|---|---|
| `img_bench_llamacpp_prod_330w.json` | llama.cpp production. Its `--image-max-tokens` model default clips at ~4,096, so 3 of 5 pages ran below native. 91.3 s total. |
| `img_bench_vllm_native.json` | vLLM at the checkpoint default (16,384-token cap) = native resolution on every page, 23% more image tokens than llama.cpp. **81.0 s** — still faster than llama.cpp at reduced resolution. |
| `img_bench_vllm_matched4096.json` | vLLM with `MM_KWARGS` pinning the budget to llama.cpp's ~4,096. Token counts match within ~1%. **75.1 s, mean TTFT 1.78 s vs 6.42 s.** |

**`sample` in these files is the first 180 characters of output, which is reasoning preamble, not the
transcription** — all three arms hit the 700-token cap. These files measure **wallclock only**; no OCR
quality conclusion can be drawn from them.

**Core-19 on the heretic MXFP4 build** ([docs/13 §6e](../docs/13-vllm-mxfp4-w4a8-rdna4.md)):

| File | What it is |
|---|---|
| `core19_mxfp4_results.tgz` | The exported results tree (per-task `results-*.json`, transcripts, run + attempt-2 metadata) for the 18/19 run. Platform id `r9700-330w`, deliberately distinct from the 250 W `r9700` corpus so neither set's metadata is falsified. |
| `core19_time.html` | Wall-clock chart vs the Q4_K_S baseline: per-task bars, a speedup-vs-task-length scatter, and the table. Agent minutes, attempt 1 both sides. Offline. |
| `img_bench2_heretic_mxfp4_nothink.json` | Vision run on the heretic build with thinking disabled: sequential and 5-way concurrent passes, PP/TG/TTFT per image, plus draft acceptance sampled from the server around each pass. |

**The two harnesses calibrate tokens/word differently on the same prose** (0.65 vs 0.86), so a
nominal "32000" depth is ~55k tokens against vLLM and ~42k against llama.cpp. The deep-decode rows
therefore *understate* vLLM's advantage rather than flattering it.

**Field notes:**
- **`aggregate` means decode concurrency** in the `conc2_*` files (~70-token prompt, `max_tokens=800`,
  so prefill is negligible) and **end-to-end throughput** in `vllm_conc_spec5.json` (8k prompts). Same
  word, different quantity — see [MISTAKES](../MISTAKES.md).
- **`tail_distinct`** is distinct ÷ total words in the last 300 characters of output: the degeneration
  guard added after the looping-filler trap. Healthy runs sit at 0.74–0.93.
- **`cpu_phys_max` / `cpu_sib_max`** are per-core busy % over the rung for CPUs 0–11 and 12–23. On this
  5900X `N` and `N+12` are siblings, so `sib_max` near zero confirms the `--cpuset-cpus=0-11` pin held.
- **`gpu_power_w`** is sampled mid-rung. It reads 245–267 W against a 250 W cap on every rung.
- KV-pool and Mamba-block figures per configuration are tabulated in
  [docs/13 §2](../docs/13-vllm-mxfp4-w4a8-rdna4.md) rather than in these files; they come from the
  server's startup log, not the harness.

## `traces/`: full reasoning and answers

JSONL, one record per model call: model/condition, label, prompt, stats, checks, full `reasoning` and `answer`.
Extracted from the live terminal logs by `scripts/core19/extract_traces.py` (lengths verified against the
harness's own character counts).

| File | Contents |
|---|---|
| `sanity_and_loopab_traces.jsonl` | Turbo sanity check (including the 103k-character looping poem) and all 9 loop-A/B poem runs |
| `sanity_heretic_traces.jsonl` | Heretic's bug hunt and physics build |

Core-19's own per-step reasoning lives in each trial's ATIF `trajectory.json` in the runner's job directories
(not copied here; they're large).

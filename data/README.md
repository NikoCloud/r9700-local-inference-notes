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

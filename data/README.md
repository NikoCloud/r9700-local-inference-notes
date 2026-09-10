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

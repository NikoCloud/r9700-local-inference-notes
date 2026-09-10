# patches/

## `llama.cpp-draft-mrope-vision-434ddbb.patch`

**What it fixes:** image input combined with a separate draft model (DFlash2 speculative decoding) on Qwen3.8,
which uses M-RoPE positions. Against `master` @ `434ddbb`.

**Symptom without it:** the first request containing an image fails with `failed to process mtmd chunk`. Image
input alone (no draft model) works on unpatched `master`.

**Why:**
- The target model's prefill can contain multimodal embedding batches whose rows carry non-linear M-RoPE
  positions.
- The draft model's cache is 1-D. Injecting those rows either breaks the cache's consecutive-position check
  (chunks wider than one ubatch) or lands them at bogus positions.

**What the patch does:**
- It **skips** embedding batches when seeding the draft cache.
- It **zero-fills** the hole when the next token batch arrives.

**Verified by:**
- the patched build describing a generated test image correctly, and
- the server logging `draft cache hole ... seeding with zero features`, which proves the new path ran rather
  than the bug simply not triggering.

**History:**
- The original version targeted the DFlash2 PR branch.
- DFlash2 itself merged upstream on 2026-08-27 (PR #27342), but this fix did not.
- Upstream is addressing M-RoPE differently: it *represents* embedding positions (4 rows per token) instead of
  skipping them. That's incompatible with this approach, so the patch had to be **rewritten**, not re-applied,
  for `434ddbb` (56 insertions vs the original 74).

**Status:** local only. Revisit when llama.cpp PR #24669 merges; this patch will probably conflict with it.

**Production note:** production later switched from DFlash2 to MTP ([docs/04](../docs/04-speculative-decoding.md)).
The patch stays in the build as the vision-plus-draft-model path.

**Apply:**

```bash
cd llama.cpp && git checkout 434ddbb
git apply ../patches/llama.cpp-draft-mrope-vision-434ddbb.patch
```

Licensed MIT, like llama.cpp.

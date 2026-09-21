# 15 — TurboQuant KV quantization on gfx1201: the crash was a disabled flash-attn path, not the rotation quant

> **Date:** 2026-09-20. One Radeon AI PRO R9700 (gfx1201), **TP=1**, single-GPU, production (llama.cpp)
> evicted for every run. Model: `qwen3.8-27b-heretic-paro-mxfp4-mtp` (the paroquant W4A8 MXFP4 chain
> from [13](13-vllm-mxfp4-w4a8-rdna4.md), MTP-fused checkpoint) served through
> [`radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (image
> `stilldeadcode/vllm-radiance:0.9.3`, **vLLM 0.27.1**, ROCm 7.14, torch 2.11). All runs `NOSPEC=1`
> (no drafter) unless stated — MTP is a separate, still-open blocker (§4). Not covered: dflash2 against
> TurboQuant (untried), and anything above depth 120,000.
>
> **Whose work this is:** TurboQuant is upstream vLLM (PR #38479 and follow-ons) — PolarQuant (WHT
> rotation + Lloyd-Max scalar quantization) for keys, uniform quantization for values. AITER's Triton
> MHA kernel is AMD's. The radiance/paroquant/R4D stack is credited in [13](13-vllm-mxfp4-w4a8-rdna4.md).
> Everything below is measurement and bug-hunting against their work, plus one local patch — not a
> contribution to any of it.

**Short version:** TurboQuant's `turboquant_k8v4` KV cache boots, serves, and holds up to depth 120,000
(target 262,144) — but it took four independent bugs to get there, only one of which was actually about
compression. The crash at depth was `vllm/v1/attention/backends/fa_utils.py` silently losing
flash-attention on gfx1201 (an `on_gfx1250()`-only gate), forcing TurboQuant's continuation-prefill into
a dense, unbounded-memory SDPA fallback. Fixed locally. Decode still degrades far faster with depth than
fp8 (10.0 vs 19.8 tok/s at depth 60k, default split count) — a real, reproducible **~13%** of that gap
closes with one config flag (`tq_max_kv_splits_for_cuda_graph` 32→128), the rest is the per-token
value-dequant cost itself, not yet fixed.

## 1. Four bugs, in the order they were hit

**(a) R4D rejects the KV dtype outright.** `--attention-backend R4D` (this stack's default) fails at
boot for any `turboquant_*` KV dtype: `ValueError: Selected backend AttentionBackendEnum.R4D is not
valid for this configuration. Reason: ['kv_cache_dtype not supported']`. R4D is a hand-written kernel
that never had a TurboQuant path built into it. Fix: `--attention-backend TURBOQUANT` for `turboquant_*`
dtypes, R4D stays default for everything else.

**(b) `compile_sizes` fights cudagraph padding under MTP speculative decoding.** `run_paroquant_nospec.sh`
hardcodes `compilation-config.compile_sizes = [1,2,4,8]`. With MTP active this throws
`ValueError: compile_sizes contains 8 which would be padded to 9. All compile_sizes must be values that
won't be changed by cudagraph padding.` — dropping 8 just moves the failure (`contains 1 which would be
padded to 3`). This is independent of KV dtype; it would have broken an fp8+MTP control run the same
way. Fix: omit `compile_sizes` from the compilation-config entirely and let vLLM derive safe values.
Forked `run_paroquant_nospec.sh` → `scripts/turboquant-gfx1201/run_turboquant_test.sh` for this rather
than patching the production launcher in place.

**(c) MTP + TurboQuant is a hard upstream block, not a config problem.** Even with (a) and (b) fixed,
any `num_speculative_tokens > 1` throws immediately on first request:
```
ValueError: Unsupported attention metadata type for speculative decoding with
num_speculative_tokens > 1: <class 'vllm.v1.attention.backends.turboquant_attn.TurboQuantMetadata'>.
Supported types are: (TritonAttentionMetadata, RocmAttentionMetadata, ..., FlexAttentionMetadata)
```
vLLM's speculative-decode proposer hardcodes an allowlist of attention-metadata classes it knows how to
feed multi-token draft verification into, and `TurboQuantMetadata` isn't on it. Not fixable from the
outside. `num_speculative_tokens=1` is explicitly excluded from this check per the error text and was
not tried. dflash2 (an external drafter, doesn't share vLLM's native spec-decode metadata plumbing) is
the untried workaround — see [OPEN-PROBLEMS](../OPEN-PROBLEMS.md).

**(d) The real crash: flash-attention is silently disabled on gfx1201.** With (a)–(c) clear, nospec
TurboQuant OOM'd reliably once a request's context passed one prefill chunk (~depth 32,000 at
`--max-num-batched-tokens 8192`):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.64 GiB. GPU 0 has a total capacity
of 31.86 GiB of which 758.00 MiB is free.
```
Traced to `turboquant_attn.py`'s `_continuation_prefill`: when `_HAS_FLASH_ATTN` is `False` it falls
back to a dense SDPA call with an explicit causal mask (`F.scaled_dot_product_attention(attn_mask=...)`),
which forces PyTorch's non-fused "math" kernel to materialize the full `(Hq, q_len, seq_len)` score
matrix — `O(q_len * seq_len)` instead of flash-attention's `O(1)`. At depth 32,000, `q_len=8192`, 24
heads: `24 * 8192 * 32000 * 2B ≈ 4.6 GiB` in one call — matches the OOM size exactly.

Root cause is in `vllm/v1/attention/backends/fa_utils.py`:
```python
if on_gfx1250():
    from aiter.ops.triton.mha import flash_attn_varlen_func
else:
    from flash_attn import flash_attn_varlen_func   # not installed on this image
```
Only `gfx1250` gets routed to AITER's Triton MHA; every other ROCm arch — gfx1201 included — falls to
the upstream `flash_attn` pip package, which this image's own comment says isn't installed. That import
fails silently, `_HAS_FLASH_ATTN` comes out `False`, and nothing at startup indicates flash-attention was
ever the missing piece. Confirmed AITER's Triton MHA imports and runs correctly on this hardware
(`aiter.ops.triton.mha.flash_attn_varlen_func`, verified with real GPU devices attached, not just an
import check). Patched locally (`scripts/turboquant-gfx1201/patch_turboquant_flash_attn.py`): try AITER
unconditionally, fall back to upstream `flash_attn` only on `ImportError`.

**Follow-on:** AITER's Triton MHA needs a per-arch autotune config
(`aiter/ops/triton/configs/<arch>-MHA-DEFAULT.json`) and ships one for gfx1151/gfx1250/gfx942/gfx950 —
not gfx1201. Stood in gfx1151's config (same RDNA4-family Triton kernels, modest block sizes —
`BLOCK_M=64, BLOCK_N=32, num_stages=2` — comfortably inside gfx1201's 64 KiB LDS budget). Unblocks
correctness; not a real gfx1201 autotune. A wrong config here fails loudly (kernel launch error), not
silently, so the risk of this stand-in producing silently-wrong output is low, but it wasn't validated
against a proper autotune sweep.

**On not upstreaming (d):** prepared a PR extending `on_gfx1250()` to `on_gfx1250() or on_gfx1201()`
(narrow, matches the existing per-arch helper pattern, `ruff check`/`ruff format` clean against vLLM's
own config, validated end-to-end on this hardware). Discarded before submission — the same
"gated to gfx1250 only, never extended to sibling RDNA4 archs" pattern recurs elsewhere in vLLM's own
codebase (an unrelated gfx1250-only `AITER_MXFP4_BF16` condition surfaced in the same search), so this
reads as a known category of gap to maintainers already, not a novel finding worth a standalone PR under
their own contribution policy (`vllm-project/vllm`'s `AGENTS.md`: no low-value mechanical PRs). Kept as
a local patch only.

## 2. With the fix: depth 2,000 → 120,000, fp8 vs turboquant_k8v4

`GPU_UTIL=0.94–0.97` (bumped for fp8 to clear a KV-cache-sizing margin miss, see
[MISTAKES](../MISTAKES.md)), `CHUNK=8192`, `MAXLEN=262144`, splits at TurboQuant's default (32) unless
noted. Decode tok/s, prose task (code_edit tracks within 0.1 throughout, prefill numbers are inflated by
prefix caching at depth per [MISTAKES](../MISTAKES.md) Y.5 and not compared here):

| depth | fp8 (R4D) | turboquant_k8v4 (fixed, splits=32) |
|---|---|---|
| 2,000 | 22.29 | 21.36 |
| 32,000 | 20.81 | 13.61 |
| 60,000 | 19.78 | 10.01 |
| 120,000 | *(not run)* | 6.55 |

fp8 barely degrades (22.3 → 19.8, ~11% over 60k depth). TurboQuant collapses far faster (21.4 → 13.6 →
10.0 → 6.6, roughly halving per depth-doubling). The crash is fixed; the compression tax is not flat.

## 3. Why the decay is steep, not flat, and what it cost to check

Read the actual decode kernel (`triton_turboquant_decode.py`) rather than guess. Two contributing
mechanisms, confirmed from source, not both fixed:

**Fixed split count vs. growing depth.** `turboquant_attn.py` bakes `NUM_KV_SPLITS` from
`vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph` (default **32**) — fixed because cudagraph
grid dimensions must be constant across captures. Per-split serial work is `context_len / splits`, so a
fixed split count under growing depth means each of the (fixed number of) parallel program instances does
strictly more sequential iterations as context grows — with no added parallelism to compensate. Swept
32/64/128/256/512 at depth 60,000 (fp8-lottery sanity-checked via each run's own depth-2,000 point, all
landing 20.5–21.4 tok/s — consistent with fast-mode, see [MISTAKES](../MISTAKES.md) on the gfx1201 boot
lottery):

| splits | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| decode @ 60k | 10.0 / 10.01† | 11.0 | 11.3 / 11.26† | 11.43 | 11.34 |

† repeat boot, n=2 total per value; both landed within 0.3% of the first boot.

Clean gain 32→128, then a plateau (128/256/512 statistically indistinguishable) — **not** a decline past
some optimum. But depth-2,000 decode drifts down monotonically with split count (21.3 → 21.3 → 21.2 →
20.9 → 20.5, 32→512): a small, real, ever-present tax (more stage-2 reduction, more grid launch overhead)
that buys nothing once the plateau is reached. **128 is the practical setting** — captures the gain,
avoids the tax. 32 CUs on this card are already oversubscribed at the *default* split count (24 heads ×
32 splits = 768 program instances for one request), so there was no risk in sweeping upward.

**Per-token value-dequant cost (not fixed).** For 4-bit-quantized values (`k8v4`'s "v4"), every attended
cached token, every decode step, costs: a packed-byte load + bit-unpack for a 4-bit index, plus **four
more small scattered loads** for a per-token scale and zero-point (`sc_lo`/`sc_hi`/`zr_lo`/`zr_hi`, each a
separate byte read reconstructed to fp16) that fp8 simply doesn't need — fp8 is one byte load and one
bitcast, full stop. This is inherently `O(context)` per step, same complexity class as the base attention
cost fp8 also pays, just with a much larger per-token constant. Not coalesced, not fixed this session.

Even at the split=128 plateau, TurboQuant is still **~43% slower** than fp8 at depth 60,000 (11.3 vs
19.8). The split-count fix recovered a real, measured slice of the gap; most of it is this dequant cost.

## 4. R4D's fp8 advantage isn't in the attention kernel

Ran fp8 through vLLM's stock, non-R4D `ROCM_AITER_UNIFIED_ATTN` backend as a control against R4D's own
fp8 path:

| | R4D | stock `ROCM_AITER_UNIFIED_ATTN` |
|---|---|---|
| decode @ 2,000 | 22.29 | 22.25 |
| decode @ 60,000 | 19.78 | 19.84 |

Indistinguishable. R4D's custom attention kernel buys nothing measurable for fp8 specifically on this
model. Noticed while it booted: `RADIANCE_USE_R4D=1` gates GDN/Mamba-layer acceleration through a
*separate* hook (`patch_r4d.py` §2/§3), independent of which attention backend is selected — so both
runs above shared R4D's GDN kernels regardless. Since 48 of this model's 64 layers are GDN, R4D's real
engineering value for this checkpoint lives there and in the MXFP4 weight quant ([13](13-vllm-mxfp4-w4a8-rdna4.md)),
not in the 16-full-attention-layer KV-cache kernel. Also notable: R4D's own split mechanism (`radiance_r4d_attn.py`)
bakes its split count from `max_ctx_bound` — the engine's actual configured max context — not a fixed
constant, letting "the kernel's split law" (their comment) choose. R4D never had this bug's shape to begin
with; it also never had a rotated/quantized KV path.

## Raw data
Data: [data/turboquant-gfx1201](../data/turboquant-gfx1201). Harnesses:
[scripts/turboquant-gfx1201](../scripts/turboquant-gfx1201) (`run_turboquant_test.sh`, forked from the
production launcher; `patch_turboquant_flash_attn.py`, the local fix; benchmarking via the existing
`vllm_ab2.py`). Measurement traps hit along the way: [MISTAKES.md](../MISTAKES.md).

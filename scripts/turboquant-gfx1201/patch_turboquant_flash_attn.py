#!/usr/bin/env python3
"""Let TurboQuant's continuation-prefill path use real flash-attention on gfx1201.

fa_utils.py gates AITER's Triton flash-attn behind `on_gfx1250()` -- any other ROCm arch,
including gfx1201, falls through to `from flash_attn import ...`, the upstream pip package,
which this image does not install (see the module's own comment: "the upstream flash-attn
package is not installed/available"). That raises ImportError, `_ROCM_FLASH_ATTN_AVAILABLE`
stays False, and `_HAS_FLASH_ATTN` in turboquant_attn.py comes out False.

The consequence lands specifically in TurboQuantAttentionImpl._continuation_prefill's SDPA
fallback branch: it builds a dense (q_len, seq_len) causal mask and calls
F.scaled_dot_product_attention with attn_mask=..., which forces PyTorch's non-fused "math"
kernel to materialize the full (Hq, q_len, seq_len) score matrix -- O(q_len * seq_len) memory
instead of flash-attention's O(1). At depth 32000 with Hq=24, that is
24 * 2048 * 32000 * 2B ~= 2.9 GiB for one call, which is exactly the scale of the CUDA OOMs
observed serving Qwen3.8-27B-PARO-MXFP4 nospec at 262144 ctx (2026-09-20).

AITER's Triton MHA (aiter.ops.triton.mha.flash_attn_varlen_func) already works on gfx1201 --
it is the same kernel family this stack uses elsewhere via VLLM_ROCM_USE_AITER_UNIFIED_ATTN --
so the fix is simply to try it on any arch, not just gfx1250, falling back to upstream
flash-attn only if AITER's module genuinely is not importable.

Idempotent.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = (Path(sysconfig.get_paths()["purelib"])
     / "vllm/v1/attention/backends/fa_utils.py")

OLD = """    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    try:
        if on_gfx1250():
            from aiter.ops.triton.mha import (  # type: ignore[no-redef]
                flash_attn_varlen_func,
            )
        else:
            from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        _ROCM_FLASH_ATTN_AVAILABLE = True
"""

NEW = """    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    try:
        # --- RADIANCE_FA_ANY_ARCH (patch_turboquant_flash_attn.py) ---
        # Upstream only tries AITER's Triton MHA on gfx1250; every other ROCm arch
        # (including gfx1201) fell through to the uninstalled upstream flash-attn
        # package and silently lost flash-attention entirely. AITER's Triton MHA
        # works fine on gfx1201 too, so try it unconditionally first.
        try:
            from aiter.ops.triton.mha import (  # type: ignore[no-redef]
                flash_attn_varlen_func,
            )
        except ImportError:
            from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        _ROCM_FLASH_ATTN_AVAILABLE = True
"""


def main():
    apply(F, OLD, NEW, "RADIANCE_FA_ANY_ARCH",
          "turboquant: AITER Triton flash-attn on any ROCm arch, not just gfx1250")


if __name__ == "__main__":
    main()

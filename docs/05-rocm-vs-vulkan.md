# 05 — ROCm vs Vulkan

**Summary:** for llama.cpp on this card, Vulkan (Mesa RADV) is the better choice for long-context work. ROCm
wins on short prompts and loses at depth, and it has crashed in ways Vulkan hasn't. For vLLM (ROCm only), the
defaults are badly wrong at depth, but they can be fixed.

## llama.cpp: July, first comparison (Qwen3.6-27B Q6_K, 2-GPU split, 128k context)

| | ROCm/HIP | Vulkan |
|---|---|---|
| Fits 128k at default batch sizes | **No, OOM at startup** | Yes |
| Prefill, pp512 (empty context) | 638 | **900 (+41%)** |
| Generation, tg128 (empty context) | **23.5 (+8%)** | 21.8 |
| Failure when short of memory | GPU page fault, driver recovery | Clean OOM error |

The HIP out-of-memory failure turned out to be fixable by shrinking batch/ubatch sizes. The *kind* of failure
still decided it: a clean error you can handle beats a driver fault.

## llama.cpp: August, same commit on both backends (single card)

| Test | ROCm/HIP | Vulkan |
|---|---|---|
| pp8192 at depth 0 | **1,625 (+53%)** | 1,064 |
| pp8192 at depth 32,768 | **page fault** | 616 |
| tg128 at depth 0 | 24.5 | **37.6** |
| tg128 at depth 32,768 | 20.3 | **34.5** |
| Real server, decode | 11.7–16.3 | **63.3** (with a drafter) |

**The fault:** `Memory access fault by GPU … Page not present`.
- It only appeared in `llama-bench -d 32768` with quantised KV, and only on the **second** repetition.
- Rebuilding with rocWMMA reduced it but didn't fix it.
- It never reproduced through the real server.
- A "fixed" conclusion was once drawn from a single clean repetition; it recurred on the next one.

## llama.cpp: September, one ruler across four toolbox images

Same model (Qwen3.8-27B Q4_K_S), `-c 262144`, q8_0 KV, speculation off, 38.7k-token prompt:

| Image | llama.cpp build | Backend | Prefill @ 38.7k | TTFT | Decode |
|---|---|---|---|---|---|
| vulkan-radv | 10884 | Vulkan RADV | **857.4** | 45.2 s | 31.4 |
| vulkan-rocmfpx | 257 | Vulkan RADV | 698.9 | 55.4 s | 31.5 |
| therock-nightly | 10884 | ROCm | 684.3 | 56.6 s | 27.7 |
| rocm-10.0 | 10884 | ROCm 10.0 | 675.1 | 57.4 s | 27.7 |

**Three single-variable readings:**
- **Backend beats version.** Same build, and ROCm loses ~21% prefill. The newest ROCm doesn't rescue it.
- **Decode splits cleanly by backend:** 31.4–31.5 on both Vulkan builds, 27.7 on both ROCm builds.
- **The llama.cpp version itself moved prefill 22%** between two Vulkan builds on the same driver.

**Shallow vs deep inverts the ranking:**

| Prompt | Vulkan RADV | ROCm 10.0 |
|---|---|---|
| 244 tokens | 250.8 | **318.6 (+27%)** |
| 38,724 tokens | **857.4 (+27%)** | 675.1 |

A shallow benchmark picks ROCm and is wrong for agent workloads, which arrive at depth.

> **Mechanism unknown.** An earlier write-up attributed this to "the 16k paged-attention cliff". That limit
> belongs to vLLM's ROCm backend (below), not llama.cpp, so the attribution was borrowed, not tested.

Building the winning commit natively (`GGML_NATIVE=ON`) instead of using the container added another 6.9%.
That's the 916.5 tok/s production figure in [02](02-engines-llamacpp-vs-vllm.md).

## vLLM on ROCm: the 16k kernel limit, and the fix

AMD's ROCm vLLM notes say the custom paged-attention kernel is used for bf16/fp16, block size 16, head size
128, GQA ratio 1–16 and **max context ≤ 16k**. Qwen3.8-27B meets every condition except context length. Above
16k the default `ROCM_ATTN` backend logs `Cannot use ROCm custom paged attention kernel, falling back to Triton
implementation`, and decode collapses:

| Context | Decode (default backend) |
|---|---|
| 68 tokens | 34.8 |
| 3.7k | 27.8 |
| 35k | **10.3** |

**That wasn't the whole story.** Those runs also had `--enforce-eager` on and prefix caching off. The
configuration that works, verified unpatched at 3 reps with 0.1% spread:

```bash
vllm serve <awq-int4 Qwen3.8-27B> \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
  --max-model-len 49152 --gpu-memory-utilization 0.96 --max-num-seqs 32 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --block-size 16
```

| Context | Decode |
|---|---|
| ~3k | 35.1 |
| ~35k | **32.0** (3 reps) |

- **`TRITON_ATTN` is a different Triton path** from `ROCM_ATTN`'s fallback, and ~3× faster at depth.
- **`--enforce-eager` costs ~72% of decode.**
- **Default CUDA-graph capture sizes** reserve 1.5–3 GiB for shapes you may never serve. Trimming them is a
  memory lever only; it doesn't change speed.
- **The same settings on a Qwen3.6-35B-A3B MoE (AWQ) held 81–82 tok/s at ~36k.**

### AITER on RDNA4

- vLLM only enables AITER when `get_cdna_version() > 2`, which excludes every RDNA part, even though AITER
  lists `gfx1201`. vLLM 0.22 didn't have that gate.
- **Locally:** patching the gate, plus a stage-count clamp for a 256-byte shared-memory overshoot, made AITER
  unified attention work: **29.1–32.8 tok/s at depth, no faster than `TRITON_ATTN`.**
- **Likely root cause of the overshoot:** AITER's `_LDS_CAP_BYTES` table has no entry for `gfx1200`/`gfx1201`.

### Other vLLM-on-ROCm hazards

- **Bimodal decode per process start** (ROCm/ROCm#6347, another R9700 owner). About a third of starts land
  ~25% slower, fixed until restart. It was seen here too (23.5 vs 34.2 tok/s across three boots). Never trust
  one boot.
- **"FP8 MoE plus sliding-window attention unlocks depth"** (inferred from a third-party repo) didn't hold.
  Before the configuration fixes above, a MoE decayed as hard as the dense model: 67.4 tok/s at 3k to 5.36 at
  120k. With the fixes, it held 81–82 tok/s at ~36k; 120k wasn't re-measured.

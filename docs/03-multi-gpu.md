# 03 — Multi-GPU: layer, row and tensor splits; PP=2; TP=2

The pair is an R9700 (32 GB) and an RX 9070 XT (16 GB): same silicon, different VRAM, power and board. Most
of what's below is shaped by that mismatch.

**Short version:**
- **llama.cpp:** splitting across the two cards bought capacity, not speed. Single-card was faster for this
  workload.
- **vLLM pipeline parallel (PP=2):** works, and is a *capacity* lever.
- **vLLM tensor parallel (TP=2):** faster than one card, but its context is capped by the smaller card, and
  on the mixed pair it breaks in a kernel.
- **Power:** multi-GPU vLLM coincided with the machine's hard power-offs ([06](06-power-and-stability.md)).

## llama.cpp

### `-sm layer` is a sequential pipeline, with a tail tax

- `-sm layer` hands activations from card to card, so at any instant only one card computes (~50% utilisation
  each).
- **Whichever card holds the *last* layers also gets the output projection (LM head) and, for MTP models, the
  whole extra MTP block,** whatever `-ts` ratio you set. `-ts` only moves the boundary between ordinary layers.
- **Observed:** with a 2:1 split, the 16 GB card had ~1.3 GB free and the 32 GB card ~9.2 GB.
- Changing `-ts 2,1` to `3,1` produced a byte-identical out-of-memory crash, because the tail doesn't move.

**The fix is `--override-tensor` (`-ot`),** which pins tensors by regex. The separator between rules is a comma,
despite one `--help` output showing a semicolon.

| Variant (Qwen3.6-27B MTP, 128k context, Vulkan) | `-ot` | 16 GB card free | Generation | Cost |
|---|---|---|---|---|
| Baseline | none | 1.26 GB | 37.06 | — |
| Move output *and* MTP block | `blk\.64\..*=Vulkan0,output.*=Vulkan0` | 3.6 GB | 32.35 | −12.7% |
| **Light touch: output only** | `output.*=Vulkan0` | 3.3 GB | **34.51** | **−6.9%** |

Moving the MTP block too adds a cross-device round-trip on every draft; moving only the output projection
recovers most of the headroom.

### Other split modes

| Mode | Result |
|---|---|
| `-sm row` | Won't load on Vulkan: `device Vulkan0 does not support split buffers` |
| `-sm tensor` | Loads, at about half the speed (568 tok/s prefill) |
| `-sm layer` vs one card, in llama-bench | 770 vs 616 tok/s prefill at 32k: a bench win |
| `-sm layer` vs one card, in the real server | 1,088 vs 796 prefill, but **26.2 vs 33.5 decode** without a drafter. With the drafter it fell to 740 prefill (the drafter blocks half the pipeline) |

For agent turns, which are mostly decode, a single card won.

### ROCm-specific multi-GPU traps

- At 128k context with default batch sizes (`-b 2048 -ub 512`), the ROCm/HIP build ran out of memory at
  startup. The failing allocation was the *compute buffer*, which scales with batch size, not KV cache or
  slots. `-b 1024 -ub 256` fitted and prefilled a large prompt at 668–673 tok/s, against 466–468 at
  `-b 512 -ub 128`. The default couldn't start, so there's no direct speed comparison with it.
- `-np` wasn't the cause: `--kv-unified` was already on, and forcing `-np 1` gave byte-identical failure sizes.
- **Using `-ot` *instead of* the batch fix on ROCm started fine and handled light load,** then page-faulted
  mid-prefill on a large prompt. Worse than a clean OOM, because it passes a smoke test. On ROCm the batch fix
  is required; `-ot` only complements it.

## vLLM pipeline parallel (PP=2)

**It works.** An earlier belief that "vLLM on this box is single-GPU only" came from a TP deadlock, which is a
different mode.

### It's a capacity lever, not a speed lever

- **Throughput and prefill are flat versus one GPU** (+3–4%, inside noise).
- **KV capacity moves a lot with the layer split,** set with `VLLM_PP_LAYER_PARTITION`:

| Layers on R9700 / 9070 XT | R9700 used | 9070 XT used | KV cache tokens |
|---|---|---|---|
| 44 / 20 | 58.6% | 85.5% | 88,594 |
| 48 / 16 | 72.1% | 86.2% | 190,539 |
| 50 / 14 | 81.8% | 86.2% | 254,862 |
| **52 / 12** | 90.3% | 83.0% | **317,970** |

- **3.6× the KV cache on identical hardware.** Each stage only stores KV for its own layers, so capacity is set
  by whichever stage fills first: the 16 GB card.
- `gpu_memory_utilization` is a fill *target*, so the small card sits at ~83–96% at every split. Freed space
  becomes more KV.
- Capacity scaled faster than the layer count fell, and a simple 1/L model mispredicted it in both directions.
  Measure; don't extrapolate.
- **The last stage carries the LM head,** so 12 layers there cost ~1.05 GB/layer against ~0.53 on the first
  stage. A symmetric split silently overloads the small card.
- There's **no per-tensor placement** (no `-ot` equivalent) in vLLM.
- **vLLM only logs memory for rank 0,** so the binding card never appears in the logs. Sample it externally.

On an older configuration, PP=2 decode was poor: 2.35 tok/s with fp8 KV, 7.86 with unquantised KV. MTP under
PP isn't implemented (`NotImplementedError`: the drafter lacks `SupportsPP`).

## vLLM tensor parallel (TP=2)

**Upstream issues** (rocm-systems#5480, vllm#40980) report an RCCL deadlock on dual R9700 with RCCL 2.27.7.
The image used here ships **RCCL 2.28.9 and serves.** An earlier "initialises but hangs" conclusion was
wrong: the `shm_broadcast … 60 seconds` line is a warning, and the empty result cells came from a probe
script wiped by a power-off.

### Matched comparison: 225 W, context cap 32,768, median of 3 boots

| | One R9700 | TP=2 | Gain |
|---|---|---|---|
| Prefill, 3k | 848.1 | 1,129.1 | +33% |
| Decode, 3k | 34.19 | 47.70 | **+39.5%** |
| Prefill, 23k | 855.8 | 1,605.0 | **+87%** (91% of the theoretical ceiling) |
| Decode, 23k | 32.42 | 45.23 | +39.5% |
| KV pool | 79,872 | 42,325 | −47% |

- **Prefill** is compute-bound, so TP nearly doubles it.
- **Decode** gains well short of 2×, because of per-layer all-reduce over PCIe 4.0 x8.
- **TP runs at the slower card's pace.** That's the R9700 (slightly less compute than the factory-OC 9070 XT).
  Replacing the 9070 XT with a second R9700 would cost essentially nothing in speed.

### The smaller card caps context

- vLLM sizes the KV pool from the worst rank. At a 49,152 cap, TP refused to start: 1.65 GiB of KV needed,
  1.47 GiB available on the 16 GB card.
- 32,768 boots, with a 42,325-token pool.
- **But against the right baseline** (the single-card DFlash2 setup actually used for speed, capped near 12k
  context), TP gives 3.4× the context and frees 3.85 GB. The first write-up compared against the wrong
  baseline; the owner caught it.

### One boot isn't enough

Single-GPU decode across three boots: **34.20 / 34.19 / 23.52**. That's ROCm/ROCm#6347's slow start-up mode.
TP's three boots agreed within 0.5%. One boot per arm would have reported TP as +103%.

### The mixed pair breaks TP in a kernel (09-09)

- Qwen3.8-27B AWQ at TP=2 died with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` in Tensile GEMM kernels. The
  error came **only on the RX 9070 XT** (`0x7550`), never the R9700, across two different kernel
  configurations.
- **Likely cause:** hipBLASLt picks a kernel from device properties, and the pick is invalid on the consumer SKU.
  "Needs matched cards" is about kernel selection, not only memory.
- **Routing AWQ through Triton** (`VLLM_USE_TRITON_AWQ=1`, from kyuz0's toolbox environment) didn't help.

**Four wrong diagnoses along the way:**
1. **Out of memory.** Real, but not the cause. Different `--gpu-memory-utilization` values gave byte-identical
   failures, and nvtop showed headroom at the moment of death.
2. **fp8 KV cache.** `TRITON_ATTN` doesn't support quantised KV (`illegal memory access`).
3. **Contention.** One run was invalid because other engines were still resident.
4. **"100% utilisation" on the surviving card.** It drew 83 W of 250 W: a rank spin-waiting on a dead peer.

### What matched cards would buy

A projection from the 250 W measurements:
- TP=2 decode, corrected for depth and power, with MTP: **~30 tok/s per stream.**
- That's against llama.cpp + drafter at 63.3 single-stream on one card.

So a second R9700 is a **concurrency** upgrade for many parallel agents, not a faster single agent. It would
also put system RAM and VRAM back at 1:1 (64 GB each), which makes the RAM problem worse
([OPEN-PROBLEMS](../OPEN-PROBLEMS.md)).

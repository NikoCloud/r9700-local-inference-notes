# 02 — Engines: llama.cpp vs vLLM

**Short answer:** this box is used two ways, and they want different engines.

| Use | Winner | Why |
|---|---|---|
| **One or two interactive agents,** deep contexts (median ~66k) | llama.cpp (Vulkan) + MTP | Speculative decoding: 61.7 tok/s single-stream, 42 tok/s each for two streams |
| **Many short parallel requests** (batch jobs, ≤ 16k each) | vLLM | Real continuous batching: aggregate throughput scaled 12× where llama.cpp's didn't |

Without speculative decoding the two engines are close on decode, **once vLLM is configured correctly**.
Getting there took most of a month.

## Session 1: the first matrix (August, builds of that time)

Three models, both engines, three backends where applicable, single-stream and concurrent:

| Model | Engine | Prefill (solo) | Decode (solo) | Peak concurrent aggregate | Ceiling |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | llama.cpp Vulkan, 1 GPU | 2,133 | **112** | 524 @ n=2 | collapses at n≥4 |
| Qwen3.6-35B-A3B | vLLM, 1 GPU | **2,995** | 21 | **1,634 @ n=80** | clean to n=96 |
| Gemma-4-12B | llama.cpp Vulkan, 1 GPU | 843 | **35** | 155 @ n=2 | fails at n=4 |
| Gemma-4-12B | vLLM, 1 GPU | **2,150** | 32 | **2,485 @ n=80** | clean to n=96 |
| Gemma-4-26B-A4B | llama.cpp Vulkan, 1 GPU | 1,164 | **61** | 260 @ n=2 | fails at n=4 |
| Gemma-4-26B-A4B | vLLM, 1 GPU | **1,532** | 33 | **1,702 @ n=96** | still climbing |

ROCm/HIP builds of llama.cpp and 2-GPU splits were worse than single-GPU Vulkan on every model tested (full
detail in [05](05-rocm-vs-vulkan.md) and [03](03-multi-gpu.md)).

**Why llama.cpp collapsed under concurrency** (traced in source on the build of that time):
- The HTTP layer is multithreaded, but slot scheduling (`update_slots()`) is one function on one thread.
- One core pinned at 100% while the GPU drew 70–90 W of a 330 W cap: dispatch-bound, not compute-bound.
- **Superseded on newer builds:** `master` @ `434ddbb` held up to 4 streams (per-stream 34.6 → 24.8 tok/s,
  aggregate 72.7) with forced-length requests.

## Session 2: what actually made the agent faster

The goal was to cut 120-second agent turns. **The engine was irrelevant; the fixes were elsewhere.** Details
are in [10](10-agent-harness-lessons.md). In short:
- Cut 5,322 tokens of tool schemas from the fixed prompt.
- Enabled llama.cpp's host-RAM prompt cache (`-cram 12288`): evictions 45 → 0.
- Found a systemd unit that had silently dropped that flag.
- Stopped a framework feature that fired a second full-context request after every turn.

**Configuration knobs barely matter; depth does:**

| Knob (pp4096) | Range tested | Effect |
|---|---|---|
| `-ub` (ubatch) | 256 → 4096 | 1,033 → peak 1,167 @ 1024 → 1,127. **+3.5% over default** |
| KV cache type | f16/f16 → q8_0/q8_0 | 1,182 → 1,151. **2.7%**, at enormous VRAM cost at 262k context |
| Context depth | 0 → 32k | **−42%** |

## Session 4: the engine floor, one ruler, no speculation (08-25)

Speculation is a layer added after an engine is chosen, so it was disabled on both. Depth was reached through
a 12-turn simulated conversation (not synthetic depth priming, which triggers a ROCm bug; see
[05](05-rocm-vs-vulkan.md)).

| | llama.cpp Vulkan (build of 08-25) | vLLM 0.27.1 + prefix caching |
|---|---|---|
| Cold prefill | 766.9 tok/s @ 42k | 1,113 tok/s @ 35k |
| Warm TTFT | 1.2 s | 0.68–1.42 s |
| Decode at depth | **31.1–31.6** | 9.5–10.4 |
| Aggregate batching, n=1 → max | 72.5 → 82.6 at n=2, then **7.8 at n=4** | 28.4 → **346.8** (12×) |

That decode gap was later shown to be mostly configuration.

## vLLM configured correctly (08-26)

Three defaults were costing more than any tuning:

| Default | Cost | Fix |
|---|---|---|
| Prefix caching **off** | Re-prefilled the full conversation every turn: TTFT 30–33 s | `--enable-prefix-caching`: 1.08 s, 65.7% hit rate |
| `ROCM_ATTN` backend above 16k context | Falls back to a slow Triton path: ~10 tok/s at 35k | `--attention-backend TRITON_ATTN` |
| `--enforce-eager` (used in earlier runs) / 11 default CUDA-graph shapes | Eager: −72% decode. Default shapes: 1.5–3 GiB reserved | Graphs on, `cudagraph_capture_sizes: [1,2,4,8]` |

**Result:** an unpatched vLLM 0.27.1, dense int4 AWQ Qwen3.8-27B on one R9700, decoded **32.0 tok/s at 35k**
(3 reps, 0.1% spread) and 35.1 at 3k. A Qwen3.6-35B-A3B MoE held **81–82 tok/s at ~36k**. Full command in
[05](05-rocm-vs-vulkan.md).

**Still missing for the agent seat:** vLLM's MTP can't run under pipeline parallel, and tensor parallel needs
matched cards ([03](03-multi-gpu.md)). llama.cpp with MTP stays ahead single-stream.

## Keeping llama.cpp current

Four toolbox images measured on one ruler (09-09), then a native build (09-10). All at 42k context, no
speculation, 250 W:

| Build | Prefill @ 42k | TTFT | Decode |
|---|---|---|---|
| Old native (build 200, patched PR) | 766.9 | — | 31.1–31.6 |
| Toolbox container, build 10884 (Vulkan RADV) | 857.4 | 45.2 s | 31.4 |
| **Native `master` @ `434ddbb`, `GGML_NATIVE=ON`, plus patch** | **916.5** | **42.3 s** | 32.0 |

**+19.5% prefill from software alone,** at lower power than the numbers it beat. Rebuilding also meant rebasing
the local vision patch ([../patches](../patches)).

## Qwen3.8-27B checkpoints qualified (08-16, older build)

| Checkpoint | Engine | Quant | Prefill | Decode |
|---|---|---|---|---|
| heretic (trohrbaugh) | llama.cpp Vulkan + MTP | Q4_K_S | 305–960 | **63.5–71.0** |
| Unsloth | llama.cpp Vulkan + MTP | UD-Q4_K_XL | 340–756 | 56.8–59.0 |
| cyankiwi | vLLM (before the fixes above) | AWQ int4 | 550–742 | 5.4–7.3 |

Both GGUFs have working MTP heads, which is why they beat every other 27B-class checkpoint tried, including a
near-lossless Qwen3.6 Q8_0 at 17.6–18.2 tok/s. The heretic GGUF ships no vision projector; Unsloth's loads on
it with no measured cost. For the September four-model qualification, see [07](07-model-qualification.md).

## Production launch line (2026-09-10)

```bash
llama-server \
  -m qwen3.8-27b-heretic-trohrbaugh-q4_k_s.gguf \
  --mmproj mmproj-BF16.gguf --image-min-tokens 1024 \
  -dev Vulkan0 -ngl 99 -c 262144 -ctk q8_0 -ctv q8_0 -fa on --ctx-checkpoints 2 -np 2 -kvu \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -cram 12288 \
  --host 0.0.0.0 --port 8080 --log-timestamps --log-prefix
```

**Notes:**
- `--reasoning-format deepseek` is needed for reasoning tokens to arrive as `reasoning_content`.
- `-cram` is the host prompt cache.
- `-np 2 -kvu` gives two agents one unified KV pool.
- Readiness is `GET /health` returning 200; `/v1/models` answers while still loading.
- Benchmark servers use a different port on loopback ([METHODOLOGY](../METHODOLOGY.md) #23).

# The docs

Thirteen dated write-ups **in the order they were written** — the numbering is chronological, not a
reading order. [LEVERS.md](../LEVERS.md) has the knob-by-knob deltas, [FINDINGS.md](../FINDINGS.md) the
one-line-per-discovery ledger, [MISTAKES.md](../MISTAKES.md) the corrections.

**By goal**

| Goal | Docs |
|---|---|
| The machine and its traps | [01](01-hardware-and-software.md) |
| Engine and backend choice | [02](02-engines-llamacpp-vs-vllm.md), [05](05-rocm-vs-vulkan.md), [13](13-vllm-mxfp4-w4a8-rdna4.md) |
| Two GPUs | [03](03-multi-gpu.md) |
| Speculative decoding | [04](04-speculative-decoding.md), [12](12-prompt-lookup-decoding.md) |
| Power and stability | [06](06-power-and-stability.md) |
| Models and quality | [07](07-model-qualification.md), [08](08-agentic-benchmark-core19.md), [11](11-reasoning-traces-and-sanity-checks.md) |
| Neighbours on the same box | [09](09-comfyui-memory.md), [10](10-agent-harness-lessons.md) |

**All docs**

| # | What's in it |
|---|---|
| [01](01-hardware-and-software.md) | Hardware and software stack: the two cards, host, drivers, and the traps specific to them |
| [02](02-engines-llamacpp-vs-vllm.md) | Engines — llama.cpp vs vLLM: configuration knobs and what actually made things faster |
| [03](03-multi-gpu.md) | Multi-GPU: layer/row/tensor splits, PP=2, TP=2, the mixed-SKU failure |
| [04](04-speculative-decoding.md) | Speculative decoding: MTP vs DFlash2, n_max sweeps, content-type effect, concurrency |
| [05](05-rocm-vs-vulkan.md) | ROCm vs Vulkan: backend comparison and the 16k cliff |
| [06](06-power-and-stability.md) | Power and stability: hard power-offs, transients, caps — what was and wasn't the cause |
| [07](07-model-qualification.md) | Model qualification: the four-model speed qualification (PP, TG, batching, MTP, VRAM) |
| [08](08-agentic-benchmark-core19.md) | Agentic quality A/B (Core-19): results, the setup-error outage, failure analysis |
| [09](09-comfyui-memory.md) | ComfyUI on the 16 GB card: RAM, eviction, quantised text encoders |
| [10](10-agent-harness-lessons.md) | What the agent framework did to the inference server, and reasoning-trace replay |
| [11](11-reasoning-traces-and-sanity-checks.md) | Sanity checks, a reasoning-loop A/B, and what the traces show |
| [12](12-prompt-lookup-decoding.md) | Prompt-lookup (n-gram) decoding, chaining it in front of MTP, the production switch |
| [13](13-vllm-mxfp4-w4a8-rdna4.md) | MXFP4 W4A8 on RDNA4's fp8 WMMA via vLLM: kernels, KV budgets, depth, concurrency, hard limits |

---

**Writing a new doc?** Start from [_TEMPLATE.md](_TEMPLATE.md). The update checklist is in
[../CLAUDE.md](../CLAUDE.md).

# Levers

What turning a knob was measured to do on a card like this (RDNA4, R9700 class) — **comparisons, not
a build guide.** Someone with similar hardware should be able to look up "what does a 250 → 330 W cap
buy?" or "what does depth do to this?" and get a measured answer instead of re-deriving it.

Each row links the write-up; rows are added or updated as sweeps land, and dated in
[FINDINGS.md](FINDINGS.md). Nothing here tracks what this particular box currently runs —
configurations churn, deltas transfer.

| Lever | Swept | Measured effect | Evidence |
|---|---|---|---|
| **Engine**, matched on the same card | llama.cpp (Vulkan) vs vLLM | llama.cpp wins the interactive agent seat via speculation (61.7 tok/s single-stream); vLLM batches far better (28.4 → 346.8 tok/s, 12×) | [02](docs/02-engines-llamacpp-vs-vllm.md) |
| **Weights format** | MXFP4 W4A8 (fp8 WMMA) vs Q4_K_S GGUF, same hour at 330 W | PP 3.2–3.6×; decode +26–46%; 4.0× concurrency peak (385.8 vs 95.7 tok/s); quality 18/19 Core-19 | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) |
| **Power cap** (vLLM) | 250 → 330 W, byte-identical config | PP +13.0 / +14.9 / +14.1 % at 2k / 32k / 60k; decode +7.0–7.1 %; concurrent peak 331.4 → 385.8 tok/s (+16 %) | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) §6b |
| **Power cap** (llama.cpp + DFlash2) | 374/330 W → 250 W, at 40k | −8.3 % prefill / −16.0 % decode | [06](docs/06-power-and-stability.md) |
| **Context depth** | 2.4k → 184k | moves prefill 42 % — more than every software knob combined (±5 % for ubatch / KV type); shallow and deep rankings invert | [02](docs/02-engines-llamacpp-vs-vllm.md) · [05](docs/05-rocm-vs-vulkan.md) |
| **Backend** (same build) | ROCm vs Vulkan | ROCm +27 % at 244-token prompts, −21 % at 38.7k; Vulkan decodes faster at both depths | [05](docs/05-rocm-vs-vulkan.md) |
| **Speculative decoding** | off → MTP → n-gram→MTP chain | ~21.8 → ~41–42 tok/s on short prompts (MTP); chain 1.6× MTP on code edits, 2.9× on copy, no extra VRAM | [04](docs/04-speculative-decoding.md) · [12](docs/12-prompt-lookup-decoding.md) |
| **Content type** | code vs prose | draft acceptance up to 0.88 (code) vs 0.51 (prose) — a drafter tuned on one looks broken on the other | [04](docs/04-speculative-decoding.md) |
| **Concurrency** | n = 1 → 64, matched at 330 W | vLLM peaks 385.8 tok/s @ n=24; llama.cpp 95.7 @ n=2, then flat | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) §6b |
| **Multi-GPU split** | PP=2, uneven 52/12 layers | 3.6× the KV cache (317,970 tokens); the last stage also carries the LM head, ~1.05 vs ~0.53 GB/layer | [03](docs/03-multi-gpu.md) |
| **llama.cpp build** | old build → `master` @ `434ddbb` | prefill @ 42k 766.9 → 916.5 tok/s (+19.5 %) at the same cap — beat every tuning knob | [02](docs/02-engines-llamacpp-vs-vllm.md) |
| **Prefix caching** (vLLM) | off → on | TTFT 30–33 s → 1.08 s (65.7 % hit rate); ships off by default | [02](docs/02-engines-llamacpp-vs-vllm.md) |
| **vLLM feature costs** | CUDA graphs / drafter / MAXSEQS | KV budget: graphs ~−56k tokens; DFlash2 ~−66k; MAXSEQS 8 → 96 halves tokens/GiB (20,130 → 9,202) | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) §2 |
| **Vision image budget** | llama ~4,096 clip vs vLLM 16,384 default | matched at 4,096: 75.1 s vs 91.3 s total; vLLM at full native (81.0 s) still ahead | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) §6c |
| **ComfyUI cache cap** | default → `--cache-ram 2 4` | 98–229 s → 25–28 s per image; anonymous RAM ~29 GB (incl. swap) → 6.9 GB | [09](docs/09-comfyui-memory.md) |

**Not swept yet, or blocked** (see [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md)): caps above 330 W are
tooling-clamped on this card (LACT: 375 W → 330 W); `SPEC=7` was unreachable at TP=1;
heterogeneous text+image concurrency is unmeasured; **context *allocation* against Qwen's advised 128K floor is unswept** (this box can only reach 65,536 at TP=1 — see [FINDINGS](FINDINGS.md) 2026-09-13); and the undervolt / memory-clock cut
(MCLK 1450 → 1359 MHz, −50 mV) has no throughput numbers yet — only its stability effect is
documented ([06](docs/06-power-and-stability.md)).

# State of the box

> **As of 2026-09-13.** This is the only file here that describes the *current* setup — everything in
> [docs/](docs/) is a dated snapshot. It was seeded from the repo, not read off the live machine;
> verify the live system first ([METHODOLOGY.md](METHODOLOGY.md) #3). Update it whenever the running
> setup changes, and bump the date.

## What runs now

Two mutually exclusive productions share the single R9700 — they never run together
([docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) §6b):

| | llama.cpp — the context seat | vLLM — prefill / vision / concurrency |
|---|---|---|
| Port | 8080 | 8000 |
| Model | Qwen3.8-27B "heretic" Q4_K_S | Qwen3.8-27B PARO MXFP4 (stock weights, not heretic) |
| Key config | `--spec-type ngram-mod,draft-mtp`, `-np 4`, 262k ctx, q8_0 KV, `-kvu` | radiance container (vLLM 0.27.1), TP=1, util 0.96, MAXLEN ≤ 65,536 per card |
| Why it wins here | context capacity; long agent sessions | 2.6–3.5× prefill; vision; concurrency |
| Introduced | [docs/12](docs/12-prompt-lookup-decoding.md), 2026-09-11 | [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md), 2026-09-12 |

**Power cap — verify before quoting:** production ran a **250 W** cap (undervolt + reduced memory
clock, set with LACT). The MXFP4 qualification in docs/13 §6b was measured at **330 W**; whether
production moved to 330 W is not recorded here. Check the live setting.

## Hardware

R9700 (32 GB, card 0 — LLM serving) + RX 9070 XT (16 GB, card 1 — image/video), Ryzen 9 5900X,
32 GB DDR4, X570 with bifurcated PCIe 4.0 x8/x8, 1000 W PSU (a larger ATX 3.1 unit is pending —
[docs/06](docs/06-power-and-stability.md)). Details and traps: [docs/01](docs/01-hardware-and-software.md).

## Settled

- **Engine strategy**: llama.cpp + Vulkan for the single-card agent seat; vLLM for prefill, vision
  and batching ([docs/02](docs/02-engines-llamacpp-vs-vllm.md), [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md)).
- **Speculative decoding**: `ngram-mod` chained in front of MTP, in production
  ([docs/12](docs/12-prompt-lookup-decoding.md)).
- **ComfyUI**: cap `--cache-ram` (here: `2 4`) or it grows to all of system RAM
  ([docs/09](docs/09-comfyui-memory.md)).
- **Backends**: Vulkan for depth, ROCm only for shallow work
  ([docs/05](docs/05-rocm-vs-vulkan.md)).

## Not settled

→ [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md) — the single list of what is broken, unexplained or blocked.
Do not duplicate its items here; remove entries there as they resolve.

---
*Update protocol: [CLAUDE.md](CLAUDE.md).*

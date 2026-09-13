# R9700 local inference field notes

[![hardware: R9700 + 9070 XT](https://img.shields.io/badge/hardware-R9700%20%2B%209070%20XT-ED1C24)](#the-rig)
[![OS: CachyOS](https://img.shields.io/badge/OS-CachyOS-1793D1)](#the-rig)
[![engines: llama.cpp · vLLM](https://img.shields.io/badge/engines-llama.cpp%20%C2%B7%20vLLM-555555)](#the-rig)
[![status: living notes](https://img.shields.io/badge/status-living%20notes-2EA44F)](FINDINGS.md)
[![license: CC BY 4.0 / MIT](https://img.shields.io/badge/license-CC%20BY%204.0%20%2F%20MIT-97CA00)](LICENSE)

A month of running large language models at home on AMD's **Radeon AI PRO R9700** (RDNA4, `gfx1201`),
written up for anyone else trying it: engines (llama.cpp vs vLLM), backends (Vulkan vs ROCm),
multi-GPU, speculative decoding, power stability, model qualification — and one diffusion side quest.
It keeps growing as the machine does.

The most useful part is probably **[MISTAKES.md](MISTAKES.md)**. A lot of this project was being
confidently wrong and then finding out why. Those errors are kept, not cleaned up, because the same
traps will catch the next person.

> **Living notes.** [LEVERS.md](LEVERS.md) = what each knob was measured to do · [FINDINGS.md](FINDINGS.md) = every
> discovery, newest first. Last material update: **2026-09-13**. Everything is dated and measured on
> one machine — re-check anything you plan to rely on; software on this platform moves weekly.

---

## Start here

| If you want to… | Go to |
|---|---|
| Set up one R9700 for local LLMs | [docs/02](docs/02-engines-llamacpp-vs-vllm.md) for the engine + configs, then [LEVERS.md](LEVERS.md) for what each knob buys |
| Not re-derive what was already got wrong | [MISTAKES.md](MISTAKES.md) — read before trusting your own first results |
| Benchmark honestly | [METHODOLOGY.md](METHODOLOGY.md) — 47 rules, each one from an incident |
| See what is still broken or blocked | [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md) |
| Weigh a setting before flipping it (power cap, depth, speculation, caching) | [LEVERS.md](LEVERS.md) — measured deltas per knob, linked to the evidence |
| Catch up on what's new | [FINDINGS.md](FINDINGS.md) — the ledger, newest first |
| Pick engines / backends / quants | [docs/02](docs/02-engines-llamacpp-vs-vllm.md), [docs/05](docs/05-rocm-vs-vulkan.md), [docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md) |
| Browse the deep-dives | [docs/](docs/) — 13 dated write-ups, indexed in [docs/README.md](docs/README.md) |

---

## The rig

| | |
|---|---|
| GPU 0 | **AMD Radeon AI PRO R9700, 32 GB** GDDR6 (~640 GB/s): LLM serving |
| GPU 1 | AMD Radeon RX 9070 XT, 16 GB (same `gfx1201` architecture, consumer SKU): image/video generation |
| CPU / RAM | Ryzen 9 5900X, 32 GB DDR4 (RAM is a real limit, see below) |
| Board / link | X570, bifurcated PCIe 4.0 x8 + x8 |
| PSU | 1000 W (a larger ATX 3.1 unit is pending; see [docs/06](docs/06-power-and-stability.md)) |
| OS / drivers | CachyOS (Arch-based), Mesa RADV 26.x, ROCm 7.2 on the host, ROCm 7.14 / 10.0 inside containers |
| Main engine | llama.cpp `master` @ `434ddbb` built natively for Vulkan, plus a local vision patch ([patches/](patches/)) |
| Power profile | 250 W cap, undervolt, reduced memory clock (LACT); what changing the cap buys: [LEVERS.md](LEVERS.md) |

Workload: two always-on AI agents sharing one llama.cpp server (long, deep contexts, tool use), plus
ComfyUI on the second card.

---

## What we learned

The one-line version of the project. Full detail in the linked docs; dates and the complete ledger in
[FINDINGS.md](FINDINGS.md).

| # | Finding | The number | Detail |
|---|---|---|---|
| 1 | llama.cpp + Vulkan wins the single-card agent seat — via speculative decoding, not raw speed | MTP: 61.7 tok/s single-stream | [02](docs/02-engines-llamacpp-vs-vllm.md) |
| 2 | vLLM's "3× slower at depth" was three wrong defaults, not the card | 32.0 tok/s @ 35k, unpatched, once configured | [02](docs/02-engines-llamacpp-vs-vllm.md) · [05](docs/05-rocm-vs-vulkan.md) |
| 3 | ROCm wins shallow, loses deep on the same llama.cpp build | +27% @ 244-token prompt · −21% @ 38.7k | [05](docs/05-rocm-vs-vulkan.md) |
| 4 | Staying on llama.cpp `master` beat every tuning knob | prefill @ 42k: 766.9 → 916.5 tok/s (+19.5%) | [02](docs/02-engines-llamacpp-vs-vllm.md) |
| 5 | Defaults cost more than tuning did | prefix caching off: TTFT 30–33 s → 1.08 s | [02](docs/02-engines-llamacpp-vs-vllm.md) · [10](docs/10-agent-harness-lessons.md) |
| 6 | MTP `n_max=2` is the production setup; content type dominates acceptance | acceptance on code runs up to 3× prose | [04](docs/04-speculative-decoding.md) |
| 7 | Multi-GPU: PP=2 is a capacity lever, TP=2 works on RCCL 2.28.9, the mixed pair dies | 3.6× KV via an uneven layer split | [03](docs/03-multi-gpu.md) |
| 8 | A power cap does not bound transients | 488 W / 584 W peaks under a 250 W cap | [06](docs/06-power-and-stability.md) |
| 9 | Same harness, same card: prefill identical, thinking wildly different | Turbo: 335 vs 2048 thinking tokens, −6–8% speed | [07](docs/07-model-qualification.md) |
| 10 | ComfyUI's default cache will eat all of system RAM | image runs 98–229 s → 25–28 s after the cap | [09](docs/09-comfyui-memory.md) |
| 11 | Core-19: the careful model solved more, the fast one solved faster | 16/19 vs 9/14 · ~3× faster, ~4.5× cheaper in output tokens | [08](docs/08-agentic-benchmark-core19.md) |
| 12 | Reasoning traces show *how* they differ — and where the fast model fails its own rules | exact line counts on 91% vs 59%; broke its AABB rule in 3 of 6 poems — and self-approved | [11](docs/11-reasoning-traces-and-sanity-checks.md) |
| 13 | Chaining n-gram lookup in front of MTP is free speed where agents echo their input | 1.6× production MTP (code edit), 2.9× (copy) | [12](docs/12-prompt-lookup-decoding.md) |
| 14 | MXFP4 W4A8 is the first thing to actually use RDNA4's fp8 WMMA | 2.6–3.5× prefill; beats llama.cpp on every speed axis at a matched 330 W | [13](docs/13-vllm-mxfp4-w4a8-rdna4.md) |

![Aggregate decode throughput vs concurrency — one R9700, matched runs at 330 W](data/vllm-mxfp4/concurrency_330w.svg)

*vLLM scales to 385.8 tok/s aggregate at n=24; llama.cpp production peaks at 95.7 at n=2 and stays
flat. One R9700, 330 W cap, matched harnesses in the same hour ([docs/13 §6b](docs/13-vllm-mxfp4-w4a8-rdna4.md), raw: [data/vllm-mxfp4](data/vllm-mxfp4)).*

---

## What's in this repo

| | |
|---|---|
| [LEVERS.md](LEVERS.md) | each knob, and what sweeping it was measured to do |
| [FINDINGS.md](FINDINGS.md) | every discovery, newest first |
| [MISTAKES.md](MISTAKES.md) | every wrong belief, and what corrected it |
| [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md) | still broken, unexplained, or blocked |
| [METHODOLOGY.md](METHODOLOGY.md) | measurement rules, learned the hard way |
| [docs/](docs/) | 13 dated deep-dives — index in [docs/README.md](docs/README.md) |
| [scripts/](scripts/) | the benchmark harnesses, sanitised ([README](scripts/README.md)) |
| [data/](data/) | raw results and offline interactive charts ([README](data/README.md)) |
| [patches/](patches/) | the local llama.cpp patch: vision + speculative decoding |
| [CLAUDE.md](CLAUDE.md) | how this repo is updated (protocol for the AI assistants) |

---

## Ground rules for the numbers

- **One machine, mostly one run per cell,** in a specific thermal and power state. Where a repeat
  exists it is noted; run-to-run spread on the current build was ~1%.
- **Speeds are tokens per second unless marked.**
  - *PP* (prompt processing, also called prefill) is measured from time-to-first-token.
  - *TG* (token generation, also called decode) is measured after the first token.
- **"Depth" means how many tokens are already in the context.** Almost every number on this hardware
  depends on it.
- **Dates matter.** A finding dated 08-25 was true for the builds on 08-25.

## How this was done

Measurements, scripts and write-ups were produced over many sessions by the owner working with AI
coding assistants and local agents. Several of the mistakes recorded here were the assistants'. They
are labelled as such in [MISTAKES.md](MISTAKES.md), because "the AI said so" was itself one of the
failure modes.

## Credits

- **kyuz0 (Donato)** for the AMD R9700 toolboxes and
  [terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini) (Core-19).
- Model authors: **trohrbaugh** (Qwen3.8-27B heretic), **DavidAU** (Turbo Fable Cold-Fusion),
  **outsourc-e** (Unleashed), **Unsloth** and the Qwen team.
- The llama.cpp, vLLM, ROCm, Mesa and ComfyUI projects.

## License

Documentation: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scripts and patches: MIT. See [LICENSE](LICENSE).

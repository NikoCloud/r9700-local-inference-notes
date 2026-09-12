# R9700 local inference field notes

A month of running large language models at home on AMD's **Radeon AI PRO R9700** (RDNA4, `gfx1201`),
written up for anyone else trying it. It covers engines (llama.cpp vs vLLM), backends (Vulkan vs
ROCm), multi-GPU, speculative decoding, power stability, model qualification and one diffusion
side quest.

The most useful part is probably **[MISTAKES.md](MISTAKES.md)**. A lot of this project was being
confidently wrong and then finding out why. Those errors are kept, not cleaned up, because the same
traps will catch the next person.

> **Status (2026-09-11):** living document. The two-model agentic benchmark (Core-19) has finished
> ([docs/08](docs/08-agentic-benchmark-core19.md), [docs/11](docs/11-reasoning-traces-and-sanity-checks.md)),
> and production moved to an n-gram→MTP speculative chain ([docs/12](docs/12-prompt-lookup-decoding.md)).
> Everything here is dated and measured on one machine. Software on this platform moves weekly, so re-check
> anything you plan to rely on.

---

## The rig

| | |
|---|---|
| GPU 0 | **AMD Radeon AI PRO R9700, 32 GB** GDDR6 (~640 GB/s): LLM serving |
| GPU 1 | AMD Radeon RX 9070 XT, 16 GB (same `gfx1201` architecture, consumer SKU): image/video generation |
| CPU / RAM | Ryzen 9 5900X, 32 GB DDR4 (RAM is a real limit, see below) |
| Board / link | X570, bifurcated PCIe 4.0 x8 + x8 |
| PSU | 1000 W (a larger ATX 3.1 unit is pending; see power) |
| OS / drivers | CachyOS (Arch-based), Mesa RADV 26.x, ROCm 7.2 on the host, ROCm 7.14 / 10.0 inside containers |
| Main engine | llama.cpp `master` @ `434ddbb` built natively for Vulkan, plus a local vision patch ([patches/](patches/)) |
| Power profile | 250 W cap, undervolt, reduced memory clock (set with LACT) |

Workload: two always-on AI agents sharing one llama.cpp server (long, deep contexts, tool use), plus
ComfyUI on the second card.

---

## Headline findings

1. **For a single-card agent seat, llama.cpp on Vulkan wins because of speculative decoding, not raw
   engine speed.** With MTP it decodes 61.7 tok/s single-stream. vLLM can't combine its MTP with
   pipeline parallel, and tensor parallel needs matched cards. Without speculation the two engines are
   close once vLLM is configured correctly, and vLLM batches far better.
   ([docs/02](docs/02-engines-llamacpp-vs-vllm.md))
2. **vLLM's "3× slower at depth" was configuration, not the hardware.** Three things stacked:
   - AMD's custom paged-attention kernel only runs at **≤ 16k context**, and the default backend's
     fallback above that is slow (34.8 → 10.3 tok/s).
   - `--enforce-eager` costs ~72% of decode.
   - Prefix caching is off by default.

   With `--attention-backend TRITON_ATTN`, CUDA graphs on (trimmed capture sizes) and prefix caching
   enabled, an **unpatched** vLLM 0.27.1 decoded **32.0 tok/s at 35k** (3 reps, 0.1% spread), level
   with llama.cpp without speculation. ([docs/02](docs/02-engines-llamacpp-vs-vllm.md),
   [docs/05](docs/05-rocm-vs-vulkan.md))
3. **ROCm wins shallow and loses deep.** In llama.cpp the same build on ROCm 10.0 was **27% faster** on a
   244-token prompt and **21% slower** on a 38.7k one than on Vulkan RADV. A shallow benchmark picks
   the wrong backend for agents. ([docs/05](docs/05-rocm-vs-vulkan.md))
4. **Keeping llama.cpp current was worth more than any tuning knob.** Moving from an old patched build to
   `master` 434ddbb (built natively): prefill at 42k went **766.9 → 916.5 tok/s (+19.5%)**, at the
   same 250 W cap. Every software knob (ubatch, KV type) moved prefill by ±5%; depth moved it 42%.
   ([docs/02](docs/02-engines-llamacpp-vs-vllm.md))
5. **Defaults cost more than tuning did.**
   - vLLM ships with prefix caching **off**: TTFT 30.4 s → 1.08 s once enabled.
   - vLLM's CUDA-graph capture silently reserves ~3 GiB.
   - A systemd unit had quietly dropped the host prompt-cache flag for two days.
   - An agent framework was firing a second full-context request after every turn.
   ([docs/02](docs/02-engines-llamacpp-vs-vllm.md), [docs/10](docs/10-agent-harness-lessons.md))
6. **Speculative decoding: MTP n_max=2 is production.** A separate DFlash2 draft model is faster for one
   stream (79.0 vs 68.3 tok/s on code), but loses at 2+ concurrent streams and costs ~2.5 GB more
   VRAM. **Content type dominates everything:** draft acceptance on code runs up to 3× higher than on
   prose, so a drafter tuned on one looks broken on the other. ([docs/04](docs/04-speculative-decoding.md))
7. **Multi-GPU:**
   - vLLM **pipeline parallel (PP=2) works** and is a capacity lever: uneven layer splits bought
     3.6× the KV cache.
   - **Tensor parallel (TP=2) serves** with RCCL 2.28.9, where open upstream issues report
     deadlocks on 2.27.7.
   - TP with a **mixed pair** (R9700 + RX 9070 XT) dies on an invalid GEMM kernel picked for the
     consumer card. ([docs/03](docs/03-multi-gpu.md))
8. **Power: a 250 W cap does not bound transients.** Sub-millisecond peaks of **488 W and 584 W** were
   measured under that cap. Hard power-offs tracked *synchronised* dual-GPU load changes, not steady
   load. ([docs/06](docs/06-power-and-stability.md))
9. **Model qualification, same harness, same card** (heretic vs DavidAU "Turbo" vs a stock proxy):
   - Prompt processing is identical within ~1–3%.
   - Turbo generates 6–8% slower but thinks far less: 335 thinking tokens vs a full 2048-token
     budget on the same prompt.
   - Turbo Q4_K_S was chosen for the quality test.
   ([docs/07](docs/07-model-qualification.md), interactive chart in [data/](data/))
10. **ComfyUI's default `--cache-ram` lets its inactive model cache grow to 100% of system RAM.** On
    32 GB that pushed everything else into swap. Capping it (`--cache-ram 2 4`) took image runs from
    98–229 s to 25–28 s. ([docs/09](docs/09-comfyui-memory.md))
11. **Agentic quality A/B (Core-19): the careful model solved more; the fast one solved faster.** Same server
    line and harness, 19 real terminal tasks:
    - **heretic: 16/19** first attempt, 17/19 with retries.
    - **Turbo: 9/14** on the tasks it actually ran (heretic 11/14 on those same tasks), and no retry flipped. Five
      tasks were lost to an overnight Ubuntu apt-mirror outage during container setup.
    - On what it solves, Turbo is **~3× faster and ~4.5× cheaper in output tokens**.
    - In four of its five genuine failures Turbo **declared success on a wrong or missing check**.
    ([docs/08](docs/08-agentic-benchmark-core19.md))
12. **Reasoning traces show *how* they differ.** On a constrained poem:
    - Word counts: heretic numbered every word and was exact on 91% of lines; Turbo estimated per line, 59%.
    - Rhyme: Turbo wrote alternating rhyme against an AABB rule in 3 of 6 poems, and its own checklist approved it.
    - Turbo's reasoning looped in 1 of 4 runs.
    - Qwen3.6's "right answer, then talked out of it" quirk did not show up.
    ([docs/11](docs/11-reasoning-traces-and-sanity-checks.md))
13. **Free speed on agent work: chain n-gram lookup in front of MTP.** llama.cpp's context n-gram drafter needs
    no model and no VRAM, and helps exactly where agents quote their input back. `--spec-type ngram-mod,draft-mtp`
    (n-gram first, MTP fallback) decoded **1.6× production MTP on a code-edit** and 2.9× on pure copy, stayed at
    MTP speed on prose, and held its ~1.5× edge per-stream at 1/2/4 concurrent. `-np` is capacity, not a
    per-stream tax (a lone stream runs full speed at any `-np`), and under `-kvu` raising it doesn't split the
    context. Now in production. ([docs/12](docs/12-prompt-lookup-decoding.md))

14. **RDNA4's fp8 WMMA works, and MXFP4 W4A8 is the first thing to use it.** `gfx1201` has native
    `wmma_f32_16x16x16_fp8_fp8` and no Instinct spoof is needed — the gate was always the *libraries*, not the
    silicon. Hand-written HIP kernels reach **225 TF/s** (~2.35× the FP16 figure), and the reason MXFP4 beats
    int4 is that e2m1 needs no zero point and its e8m0 scale folds at weight staging, leaving an inner loop with
    **zero VALU ops** (their ladder: 16 ops → 180–188 TF/s, 8 → 200, 0 → 225). On one R9700 at TP=1 this gives
    **2.6–3.5× production prefill, rising with depth** where llama.cpp collapses, and n=8 concurrency that beats
    the production line on per-stream *and* aggregate. It loses on **context capacity** (89–103k usable vs 262k),
    because 19 GB of weights leave 4.5 GiB of KV — the clearest argument yet for the second card.
    ([docs/13](docs/13-vllm-mxfp4-w4a8-rdna4.md))

---

## How to read this

| File | What's in it |
|---|---|
| [MISTAKES.md](MISTAKES.md) | Every wrong number, wrong explanation, misleading source and misunderstanding, and what corrected it |
| [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md) | What is still broken, unexplained, or blocked on hardware |
| [METHODOLOGY.md](METHODOLOGY.md) | The measurement rules this project learned the hard way |
| [docs/01-hardware-and-software.md](docs/01-hardware-and-software.md) | The stack in detail, and the traps specific to it |
| [docs/02-engines-llamacpp-vs-vllm.md](docs/02-engines-llamacpp-vs-vllm.md) | Engine comparison, configuration knobs, what actually made things faster |
| [docs/03-multi-gpu.md](docs/03-multi-gpu.md) | Layer/row/tensor splits, PP=2, TP=2, mixed-SKU failure |
| [docs/04-speculative-decoding.md](docs/04-speculative-decoding.md) | MTP vs DFlash2, n_max sweeps, content-type effect, concurrency |
| [docs/05-rocm-vs-vulkan.md](docs/05-rocm-vs-vulkan.md) | Backend comparison and the 16k cliff |
| [docs/06-power-and-stability.md](docs/06-power-and-stability.md) | Hard power-offs, transients, caps, what was and wasn't the cause |
| [docs/07-model-qualification.md](docs/07-model-qualification.md) | The four-model speed qualification (PP, TG, batching, MTP, VRAM) |
| [docs/08-agentic-benchmark-core19.md](docs/08-agentic-benchmark-core19.md) | Quality A/B on 19 agentic terminal tasks: results, setup-error outage, failure analysis, speed by depth |
| [docs/09-comfyui-memory.md](docs/09-comfyui-memory.md) | Diffusion on the 16 GB card: RAM, eviction, quantised text encoders |
| [docs/10-agent-harness-lessons.md](docs/10-agent-harness-lessons.md) | What the agent framework did to the inference server, and reasoning-trace replay |
| [docs/11-reasoning-traces-and-sanity-checks.md](docs/11-reasoning-traces-and-sanity-checks.md) | Live sanity checks, a reasoning-loop A/B, what the traces show (self-checks, counting, rhyme, reversals), and why the two GGUFs differ |
| [docs/12-prompt-lookup-decoding.md](docs/12-prompt-lookup-decoding.md) | n-gram (prompt-lookup) speculative decoding, chaining it in front of MTP, concurrency scaling, and the production switch |
| [docs/13-vllm-mxfp4-w4a8-rdna4.md](docs/13-vllm-mxfp4-w4a8-rdna4.md) | MXFP4 W4A8 on RDNA4's fp8 WMMA via vLLM: kernel verification, KV-availability budget per config, PP/TG by depth, concurrency to n=64, and four hard gfx1201 limits |
| [scripts/](scripts/) | The benchmark harnesses used, sanitised |
| [data/](data/) | Raw qualification results (JSON) and the interactive chart |
| [patches/](patches/) | The local llama.cpp patch for vision + speculative decoding |

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

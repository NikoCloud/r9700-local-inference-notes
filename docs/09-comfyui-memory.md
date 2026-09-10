# 09 — ComfyUI on the 16 GB card: memory, eviction, text encoders

A side quest that turned out to be about system RAM, not VRAM.

**Setup:** ComfyUI on ROCm, pinned to the RX 9070 XT (16 GB). **Workload:** a Krea-2-based image model with
LoRAs, plus video generation (MiniMax H3) on the bigger card when the LLM isn't running. **System RAM:
32 GB,** shared with a llama.cpp server holding a 27B model.

## Symptom

- Runs took 98–229 s instead of ~25 s.
- System RAM sat at 91–100%, and swap climbed.
- Changing only the prompt reloaded weights that should already have been in memory.

## Three wrong fixes, then the real one

| Attempt | Theory | What happened |
|---|---|---|
| 1 | Pinned host memory is eating RAM | No change. `--disable-pinned-memory` was already set, and `Mlocked` read 0. |
| 2 | `--disable-smart-memory` | **Made it worse:** the UNet was evicted on *every* prompt change (~78 s reload each). |
| 3 | `--highvram` | Stopped eviction, but removed the offload fallback, so LoRA patching ran out of memory on the 16 GB card. |
| **4** | **Cap the RAM cache** | ComfyUI's default caching mode (`--cache-ram`) lets the *inactive* model cache grow to **100% of system RAM**. |

**Validated flags:**

```
--cache-ram 2 4 --disable-pinned-memory --use-pytorch-cross-attention --fast fp8_matrix_mult
```

(normal VRAM mode, smart memory **on**)

**Result:**
- **25–28 s per image,** down from 98–229 s.
- ComfyUI's anonymous RAM went from ~29 GB, including swap, to **6.9 GB**.
- The partial UNet unload around VAE decode now reloads from the capped cache in ~3 s.

`--cache-ram` and `--cache-none` are mutually exclusive. If a large video text encoder starts reloading on
prompt changes, the cache cap is the setting to tune, by measurement.

## Fitting the model: quantise the text encoder first

| Component | Size |
|---|---|
| UNet (int8 "convrot" quant) | 12.24 GiB |
| Qwen3-VL-4B text encoder, fp8 | 5.0 GB |
| VAE | 0.24 GB |

That's more than the 16.3 GB card. Swapping to an **int4 text encoder (2.68 GB)** made everything fit
without evicting the UNet. Notes:
- int4 "convrot" quants are ~55% the size of int8 convrot.
- NVFP4 variants are Blackwell-only (NVIDIA); skip them on AMD.
- The CLIP loader's `type` must match the model family (`krea2` here, not `qwen_image`).

## One instance, per-node GPU pinning

A single ComfyUI instance with workflow tabs and per-node device pinning (`comfyui-multigpu`) replaced the
idea of one instance per card. With the LLM running, any loader without an explicit device defaults to
`cuda:0`, which is the LLM's card.

## How an agent workflow crashed everything

An AI agent's workflow and the GUI workflow loaded the **same UNet file through different node classes**
(`UNETLoader` vs a ROCm-specific loader). ComfyUI caches by node and inputs, so it treated them as different
models and started building a **second 12.5 GB copy** while the first was still held. On 32 GB of RAM the
kernel OOM killer took down ComfyUI **and** the LLM server. The failure took 7 minutes of swap thrash to
arrive.

**Lessons:**
- Any workflow that hardcodes model names or loader classes must match the one the GUI uses, or it
  silently doubles memory.
- Check workflow model names against the live server (`GET /object_info/<LoaderName>`), and pull the
  loaders from the last successful run (`GET /history`) instead of guessing.

## Video notes (larger card, LLM stopped)

- **ROCm-native custom nodes** for the sampler, VAE and loader gave roughly 6× per step on video.
- **Wan 2.2 "Lightspeed"** went from 706 s to 62 s for a 4-step, 5-second clip.
- **LTX-Video 2.3 (22B)** didn't fit: VRAM, and 32 GB of system RAM.
- **MiniMax H3, int8 DiT plus a 14 GB int4 text encoder,** doesn't fit 32 GB of VRAM together. The int4 DiT
  plus text encoder plus VAE (~25–26 GB) does.
- `expandable_segments` in the allocator config does nothing on ROCm/HIP.

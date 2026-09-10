# 01 — Hardware and software stack

## The two cards

Both GPUs use the same Navi 48 silicon (`gfx1201`), with different memory, power limits and boards.

| | Radeon AI PRO R9700 | Radeon RX 9070 XT (factory-OC board) |
|---|---|---|
| Role here | LLM serving | Image / video generation |
| VRAM | 32 GB GDDR6 | 16 GB GDDR6 |
| Memory bandwidth | ~640 GB/s | ~640 GB/s (tie) |
| Power ceiling | 330 W | 374 W |
| PCI device ID | `0x7551` | `0x7550` |

The R9700 is marginally *slower* in compute than the consumer card, though it has twice the VRAM.
That matters for tensor parallel, which runs at the slower card's pace ([03](03-multi-gpu.md)).

**Theoretical ceilings used throughout** (for a dense 27B model):
- **Prefill ≈ 1,772 tok/s**: published FP16 compute (~95.7 TFLOPS) ÷ ~54 GFLOPs per token (2 × 27B
  parameters). Any measured prefill above this is a cache hit, not real.
- **Decode ≈ bandwidth ÷ model size.** For a 16.52 GB Q4_K_M file that's ~38.7 tok/s. Raw llama.cpp decode
  without speculation reached ~81% of it at real depth.

**Platform:** Ryzen 9 5900X, 32 GB DDR4, X570 with the two cards bifurcated at PCIe 4.0 x8 + x8 (P2P
present). System RAM is the tightest resource on the box; see [09](09-comfyui-memory.md).

## GPU numbering: every tool disagrees

| Card | PCI address | `rocm-smi` / ROCm / vLLM | LACT | nvtop |
|---|---|---|---|---|
| R9700 32 GB | `0000:11:00.0` | GPU 0 | id 1 | shown as the *other* device |
| RX 9070 XT 16 GB | `0000:14:00.0` | GPU 1 | id 0 | shown as the *other* device |

LACT and nvtop both invert the order ROCm uses. Pick one canonical numbering (here: ROCm's), and translate
when reading the other tools. Never assume two tools agree.

## Software

- **OS:** CachyOS (Arch-based), kernel 7.1.x.
- **Vulkan driver:** Mesa RADV (26.1.6 in late August, 26.2.2 by 2026-09-10).
- **ROCm:** 7.2 on the host; containers used ROCm 7.14 (vLLM) and ROCm 10.0 / TheRock nightly (llama.cpp
  toolboxes).
- **llama.cpp:** built natively for Vulkan. Build history:
  - an old patched PR build (build 200)
  - `master` @ `434ddbb` (September), plus the local patch in [../patches](../patches)
- **vLLM:** 0.27.1 in `gfx1201` container images from kyuz0's R9700 toolbox project.
- **Power and fan control:** LACT (`lactd`).

### Container trap: same name, incompatible ROCm layouts

Two vLLM image families were both called "therock":

| | Lineage A | Lineage B |
|---|---|---|
| ROCm location | system `/opt/rocm` | Python wheels inside a venv |
| ROCm source | nightly | stable multi-arch wheels |
| Image size | ~41 GB | ~16 GB |

Setting `HIP_PATH=/opt/rocm` inside a lineage-B image broke AITER's JIT compile. The resulting error claimed
the **GPU architecture was unsupported**, which was false. Check the layout before touching environment
variables:

```bash
docker inspect <image> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'HIP_PATH|ROCM_PATH|TOOLBOX'
```

Also: **tags are not identity.** An image tagged `pinned-20260731` was built on 06-13. Record image IDs next
to results.

## Power control with LACT

- With a profile active (e.g. `current_profile: Vram1`), **edits to the top-level per-GPU entries do
  nothing.** Only the active profile's block applies.
- LACT accepts an over-range power value and the driver silently clamps it per card. Setting both cards to
  375 W yields 330 W on the R9700 and 374 W on the 9070 XT.
- `lact cli` works over its socket for members of the right group, so no sudo is needed, and it persists the
  change.
- **Confirm the effective cap with `rocm-smi --showmaxpower`,** never by reading the YAML.

Power behaviour itself is in [06](06-power-and-stability.md).

## Models: read the header, not the filename

- `general.file_type` in the GGUF header is the truth. Unsloth `UD-Q4_K_XL` files report `Q4_K_S`;
  "XL/M/S" are Unsloth upcast tiers over a base type. A projector named `mmproj-f16` was `BF16`.
- **MTP heads in these Qwen3.8 GGUFs are an extra ordinary layer** (`blk.64` on a 64-layer model), not
  tensors with "mtp" in the name. The practical check is to launch with `--spec-type draft-mtp` and read the
  draft acceptance in the server log.
- Vision projectors were byte-identical in payload across two differently named repos. One copy is enough.

## Storage

Models lived on a QLC SSD. Reads are fine; sustained writes collapse to ~90 MB/s once the SLC cache fills.
That makes multi-GB downloads and conversions slow, but loading is unaffected.

## Linux lessons that cost time

- **Never `systemctl mask` an fstab-generated mount unit.** systemd then fails `local-fs.target` and skips
  **every** local mount, not just that one. Use `nofail` in fstab instead.
- **`ntfs3` refuses a volume marked dirty after a power loss.** Without `nofail` that blocks boot.
- **`/tmp` is tmpfs:** a power event erases it. Keep results and harness scripts elsewhere.
- **Distrobox containers share the host PID namespace.** `pkill -f` inside one kills host processes. Use
  `docker stop`.
- **`tmux kill-session -t name` prefix-matches,** and `pkill -f` / `pgrep -f` can match your own shell. Use
  exact names, and patterns like `pgrep -f "[m]ain.py --port 8188"`.
- **Scripts written on Windows:** strip `\r` before running them under bash.

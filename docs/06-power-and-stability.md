# 06 — Power and stability

The machine lost power without warning several times during this project. Most of the lessons are about
**how easy it is to misdiagnose that**. The final cause is still not proven ([OPEN-PROBLEMS](../OPEN-PROBLEMS.md)).

## Two classes of event

| | Hard lock | Self-recovering reboot |
|---|---|---|
| Recovery | Dead until the PSU switch is toggled by hand | Restarts on its own |
| When | vLLM PP/TP runs (both cards alternating heavily); a rapid memory-clock sweep with 3 s cooldowns | Mixed workloads; one event looked like a steady ~230 W |
| Combined draw nearby | ~558–610 W | As low as ~230 W (summed, see below) |
| Logs | Journal stops mid-stream; no panic, no shutdown, no GPU reset | Same |

## What was wrong along the way

1. **"The dead-battery UPS causes it."**
   - **Proposed mechanism:** its AVR relay's break-before-make transfer drops the output for a few milliseconds.
   - **Result:** the UPS was removed from the circuit and the events continued. Dead theory.
2. **"Empty `/sys/fs/pstore` proves a power loss rather than a kernel panic."** No pstore backend was
   registered, so it would be empty either way. Uninformative, not negative.
3. **"It only happens under simultaneous dual-GPU load."** One event read a flat ~230 W just before death, 20
   minutes after the same run handled 467 W.
   - That 230 W was a **sum** of both cards, so which card drew it was never known.
   - The watchdog meant to abort above a threshold piped through `bc`, which wasn't installed. It logged 0
     and could never fire.
4. **"A per-card trace shows one card failing on its own."** The first reading of a properly instrumented
   event (a remote watcher logging each card every 1–2 s from another machine) said so. The owner
   corrected it:
   - That card had run pegged at its cap for hours on other days without trouble.
   - The event coincided with starting a **second** job on the other card while the first was already at
     its cap.
   - **Better-supported theory:** a *load transition* across both cards, not either card's steady state.
5. **"The power cap bounds the spikes."** A sub-millisecond watcher measured **488 W on the R9700 and
   584 W on the RX 9070 XT, both under a 250 W cap**, with a coincident 934 W combined peak. It didn't
   crash. Caps limit average power; they don't clamp transients.
6. **"`rocm-smi` shows the ramp."** `rocm-smi` reports "Average Graphics Package Power". A smooth decline
   in its log can be the averaging window lagging an instant change.

## Current working theory (unconfirmed)

- **An ageing 1000 W PSU is the leading suspect.** Weak hold-up capacitance explains a reboot on a brief
  sag at any load. A tired unit tripping over-current on RDNA4 transients explains the hard locks under
  alternating dual-GPU load.
- **The self-recovering class may instead be memory-clock related.** After lowering the memory clock (MCLK
  1450 → 1359 MHz) with a −50 mV undervolt, transients were *larger* than before but caused no reboot.
  - That fits a signal-integrity threshold rather than "transients are too big".
  - **The discriminating test (MCLK alone back to 1450) hasn't been run.**
- **A larger ATX 3.1 PSU is on order.** vLLM tensor-parallel work, the alternating-load pattern, stays
  disabled until it's installed.

## What the power cap costs

| Engine and workload | Cap change | Prefill | Decode |
|---|---|---|---|
| llama.cpp + DFlash2 drafter, 40k | 374/330 W → 250 W | −8.3% | −16.0% |
| vLLM single card, no drafter | 330 W → 225 W | — | −1.9% to −2.8% |

Plain decode is memory-bandwidth-bound, and bandwidth doesn't scale with power. Speculative verification
and K-quant dequantisation are compute work, which is why the drafter configuration paid more.

**The 250 W profile was kept for stability.** The software gains in [02](02-engines-llamacpp-vs-vllm.md)
were all measured at 250 W, so the hardware ceiling is higher than anything reported here.

## Rules that came out of this

- **Log each card separately,** from a machine that survives the crash.
- **Sample fast enough for the question.** One-second averages can't see transients.
- **A watchdog isn't armed until you've seen it print a real non-zero reading.**
- **Note what's running on each GPU** whenever anything odd happens. Reconstructing it afterwards failed
  more than once.
- **Don't treat the absence of evidence** (empty pstore, no kernel message) as evidence of a cause.

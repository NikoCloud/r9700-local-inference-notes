# Mistakes, misunderstandings and misinformation

Everything in this file was believed or stated at some point. Each entry says what was believed, what
turned out to be true, and how it was caught. Nothing here is hypothetical.

Why keep it: most of these produced numbers or explanations that *looked* right. The failure mode
was rarely a crash; it was a confident, plausible, wrong answer. If you are benchmarking on similar
hardware, skim this before trusting your own first results.

Who made them is recorded where it helps the lesson:
- **"The owner"** is the person who runs the machine.
- **"The assistant"** is the AI coding assistant doing much of the measurement and writing.
- **"An agent"** is one of the local AI agents that run on the box.

Nobody comes out of this file looking infallible, which is the point.

**Contents**

1. [Measurement errors that produced confident wrong numbers](#1-measurement-errors-that-produced-confident-wrong-numbers)
2. [Explanations stated as findings before anyone tested them](#2-explanations-stated-as-findings-before-anyone-tested-them)
3. [Misleading sources: filenames, tools, docs, tags, model cards](#3-misleading-sources)
4. [Misunderstandings between the owner and the assistants](#4-misunderstandings-between-the-owner-and-the-assistants)
5. [Patterns](#5-patterns)

---

## 1. Measurement errors that produced confident wrong numbers

### Prefill numbers that were really cache hits or warm starts

- **"vLLM pipeline-parallel gives a 2.23× prefill win"** (4,790 tok/s, 2026-08-09).
  - **What happened:** the container measured had already served a smoke test, so its "first"
    request was pre-warmed while every other configuration's was cold. Every request after the
    first was identical across configs (second-request prefill 3,082.8 vs 3,037.5 tok/s).
  - **Caught by:** comparing the non-first requests.
  - **Rule:** large-prompt probes are cold-start measurements. Never reuse a container between runs.
- **A 37,931 tok/s "cold" prefill.** A calibration probe had filled the prefix cache before the timed
  run.
- **A 39,420 tok/s prefill at 225 W.** A repeated prompt hit the prefix cache. It was 22× the
  theoretical ceiling for a dense 27B on this card (~1,772 tok/s, from 95.7 TFLOPS).
  - **Rule:** any prefill figure above the compute ceiling is a cache hit until proven otherwise.
    Every request needs a genuinely unique prompt.

### Comparisons that weren't like-for-like

- **"The new model is 2.6% faster."** It was measured at 34.7k tokens of context against the old model
  at 39.8k. Prefill falls with depth, so the shallower run was flattered. At matched depth the sign
  flipped: **−2%**.
- **A 2-GPU "win" contaminated by a second job.** Another GPU job started mid-benchmark and decode
  dropped from 48.77 to 9.32 tok/s. The tell was an audible fan ramp. Checking that the card is idle
  before launch is not enough if someone else starts after you.
- **An old container image was reused** without checking for a newer build. It reproduced a
  two-month-old regression and nearly led to "vLLM is hopeless on this card."
- **"MTP speculation is ~3×."** That was computed against a baseline from a different context length
  and harness. A matched control gave ~2.4× on the old kernel, and a fully matched test **1.41×** on
  the new one.
- **"fp8 KV costs 42% of decode."** True at a 35,840 context cap, only 17% at 8,192. Quote the matched
  number, not the dramatic one.

### Harness bugs that returned clean-looking wrong answers

- **Decode measured over an 80-token window.** Startup transients dominated. A real 2.9 tok/s
  reading was nearly dismissed as an artifact. Decode is now measured over 320+ tokens.
- **Prompt sizes estimated by characters-per-token.** It missed by up to 70% on filler text, three
  separate times. One apparent TTFT "win" was just a smaller prompt. Sizes now come from the
  server's `/tokenize`.
- **Counting streamed chunks instead of tokens.** With speculative decoding, vLLM packs several tokens
  into one SSE chunk. Counting chunks made speculation look *harmful* (8.29 → 7.76 → 7.43 as
  `n_max` rose). Fixed by reading the server-reported `completion_tokens`.
- **A patch that silently did nothing.** A `str.replace()` used single-quoted match strings against
  double-quoted source and matched nothing, so a run labelled "MTP" never enabled MTP. Every patch
  now asserts that it applied.
- **A missing tool returned "no match".**
  - `llama-gguf-dump` doesn't exist in that build. Piping the shell's "not found" error through
    `grep -i mtp` produced a clean empty result, which was recorded as "this model has no MTP head".
  - The grep was also looking for the wrong thing: this GGUF conversion stores the MTP head as an
    extra ordinary layer (`blk.64` on a 64-layer model), with no "mtp" in any tensor name.
  - **Caught by:** running the model with `--spec-type draft-mtp` and reading the live draft
    acceptance. It had MTP, and it outperformed the model it was being compared to.
- **Readiness probed with `/v1/models`,** which llama-server answers *while still loading*. A whole
  sweep was benchmarked against a loading server that returned `503 Loading model`. Probe
  `/health` for a 200 instead.
- **`reasoning_effort` defaulting to `xhigh` ate the whole token budget.** Requests came back with zero
  answer tokens and empty timing fields, which looked like hangs. The server log showed normal fast
  prefill. Fix: `enable_thinking: false` for speed measurements.
- **Missing `--reasoning-format deepseek`.** Thinking tokens were generated but swallowed, so 80
  "completion tokens" arrived with empty content, and decode speed was measuring reasoning, not
  answers. A coherence check now gates every run.
- **A concurrency sweep run with `-np 1`.** 16 requests took 268 s ≈ 16 × 16.8 s: pure queueing,
  reported as batching.
- **Benchmarking at `-c 32768`.** That's inside a context range where this model family was known to
  degrade, and 8× off production. The numbers looked sane and were comparable to nothing.
- **Setting `VLLM_ATTENTION_BACKEND`,** which no longer exists in vLLM 0.27.1. Three benchmark cells were
  silent no-ops, and the spread between them was attributed to attention backends. It was actually a
  separate bimodal-startup bug.
- **Reading a log while it was still being written.** A grep raced the writer and shifted result rows by
  one.

### Shell and infrastructure traps

- **Error capture read the wrong stream** (the end of `docker logs` stdout instead of stderr). Three runs in a
  row reported empty failure reasons; the real cause was out-of-memory.
- **Teardown removed only its own container name.** A leftover server held the port, and every launch in the
  next script failed.
- **`pkill -f` inside a distrobox container killed host processes.** Those containers run with the host
  PID namespace. It killed the controlling ssh session twice. Use `docker stop`.
- **`tmux kill-session` does prefix matching,** and `pkill -f` matches its own shell. Scripts killed
  themselves, twice.
- **Scripts written on Windows kept CRLF line endings.** `set -uo pipefail\r` died before doing any
  work.
- **Results written to `/tmp`,** which is tmpfs here. A power event wiped a completed sweep, the logs,
  and the harness itself. Results now always go under the project directory.
- **Power "watchdogs" that could never fire.** They piped through `bc`, which wasn't installed, so
  every reading logged as 0, including during the run that caused a power-off.
- **A watchdog summed both GPUs' power.** It could not tell one card at 230 W from two at 115 W. A
  whole incident analysis rested on that ambiguity. Log per card.

### Conclusions from too little data

- **"The ROCm page fault is fixed."** One successful repetition. It recurred on the next run with
  `-r 2`.
- **"Storage is the binding constraint."** One mount was checked and the result was generalised to the
  whole array. There was 4.6 TB free.
- **One boot per configuration would have reported TP=2 as +103% faster.** vLLM's single-GPU decode
  landed in its slow start-up mode on one of three boots (23.5 vs 34.2 tok/s). The median of three boots
  gave the real gain: +39.5%.
- **"An empty pstore means a hard power loss, not a kernel panic."** No pstore backend was registered on
  this box, so it would also be empty after a panic. Several incident write-ups leaned on it before this
  was noticed.
- **An overclock adopted after one clean minute.** A memory-clock setting went into weeks of production
  after about one minute of validation. The retest that would have separated a thermal cause from a
  voltage-margin cause was never run.

### 2026-09-10 model qualification (the assistant)

- **Generation speed and aggregate throughput judged at each model's natural output length.** The
  Turbo model writes ~5× fewer tokens on the filler prompts (90–175 vs 460–720), so its speed was
  averaged over much shorter windows. Its "aggregate" at 4 streams read 36 vs 65 tok/s, which measured
  verbosity, not speed.
  - **Caught by:** noticing the token counts per cell.
  - **Fix:** a third pass that forces every request to exactly 768 tokens (`ignore_eos`). Under forced
    length the gap was 69.6 vs 72.7 tok/s. The production model's forced and natural numbers matched
    to 0.1%, so the confound was specific to the short-answer model.
- **"Two streams at ~33k context each" read as batching.** The second request was still prefilling while
  the first generated, so the per-stream figure measured prefill overlap. It's a realism check, not a
  batching test.
- **A script labelled the production model "disqualified"** because its prose MTP acceptance was
  49.6%. The owner's rules made MTP acceptance a bonus, not a gate. The flag was simply wrong for
  the criteria.
- **A contamination audit first counted request log lines** that this llama-server build doesn't emit
  at default verbosity (zero `POST` lines). Counting distinct server task IDs against the requests
  each script sent gave an exact match.

### 2026-09-11 Core-19, sanity checks and trace analysis (the assistant)

- **A poem grader that never checked rhyme.** It auto-checked title, stanza shape, word count, first words,
  banned word, thee/thou and the closing question mark, and only *printed* each stanza's end words "for
  eyeballing". Nobody eyeballed them. Turbo's poems scored "8/8" while three of six were written ABAB
  throughout against an AABB rule.
  - **Caught by:** a manual audit of the end words while checking a trace. Heretic 15/15 stanzas correct, Turbo
    ~8/30.
  - **Rule:** a check that is displayed but not scored is not a check.
- **Parser false starts presented as findings.** The per-line word-count accuracy for heretic was reported as
  58%, then "~69%", then 91%, as successive parser bugs were fixed: running totals mistaken for line counts,
  and label tokens (`Line2:`, `Count:`) counted as words. A "draft regression 8 → 6 → 8" was a numbered recount
  of the same poem, not a new draft.
  - **Caught by:** pulling the exact text behind each flagged hit, and a 10-row hand spot-check per model before
    stating the final numbers.
- **An environment variable leaked between runs.** A resumed Turbo run exported `TASK1_NOTE` in a script that
  ended with `exec bash`, so the tmux shell kept it. The heretic sanity run launched later from that shell
  inherited it, and wrote Turbo's "reasoning loop, aborted" note into *heretic's* results file.
  - **Caught by:** the heretic summary listing a task heretic never ran. The entry was removed with a backup and
    a note.
- **The first relaunch of a test server copied the campaign's launch command verbatim**, including its
  `>> full-turbo/llama-server.log` redirect, so 16 startup lines of an unrelated server went into the finished
  Turbo arm's log. Split out with a backup before any analysis.
- **A watcher that could never fire.** It used bash process substitution (`<(…)`) inside a command sent over ssh
  to a host whose login shell is fish. It would have silently polled for 33 hours. Replaced with an explicit
  `bash -s` heredoc before the event it watched for.
- **Scheduled one-shot checks that didn't run.** Two session-scheduled health checks came due while the session
  was busy handling other events and never fired; both checks were run manually a few minutes late.
- **A quick in-run speed claim reused the wrong power data.** GPU power statistics computed for the killed first
  Turbo run were nearly carried into the final write-up of the re-run; recomputed from the right arm first.

### 2026-09-11 prompt-lookup speculative-decoding test (the assistant)

- **Conflated "a server with `-np 4`" with "four concurrent streams".** The first recommendation warned that
  raising `-np` from 2 to 4 "lowers per-agent speed". Wrong: `-np` is capacity, not a per-stream tax. The owner
  corrected the mental model, and a direct test confirmed a *lone* stream runs at full speed (125 tok/s) on an
  np1/np2/np4 server alike. The per-stream drop only happens when N streams actually run at once — unavoidable
  bandwidth sharing, and something np2 can't even do (it queues). Corrected recommendation: `-np 4` is near-free
  headroom.
  - **Rule:** benchmark the thing you'll actually run. "N slots" and "N simultaneous streams" are different
    measurements; don't let one stand in for the other.
- **Left production down by calling `main()` past its restore guard.** The concurrency harness stops production
  and restores it in a `try/finally` inside `if __name__ == "__main__"`. Running it via `import harness; harness.main()`
  executed `main()` (which stops production) but never the `__main__` `finally`, so `qwen38.service` was left
  stopped and 8080 returned 000. Caught on the next status read and restarted immediately.
  - **Rule:** cleanup that must always run belongs in `main()`'s own `try/finally`, not the module's `__main__`
    guard. Import-and-call is a supported entry point.
- **`pgrep -f "... --port 8085"` matched its own command line** and reported a "stray server up" that did not
  exist (the ssh/grep command string contained `--port 8085`). This is the exact trap already in
  [METHODOLOGY](METHODOLOGY.md) #36 — and it still caught me. Re-checked with `pgrep -af` and it was gone/absent.
- **Env-prefixed commands (`VAR=x python …`) and bash `for` loops sent over ssh to a fish login shell** parse-fail
  ("Missing end to balance this for loop"). Same class as earlier fish traps; the fix is `ssh host 'bash -s' <<'EOF'`
  with the env `export`ed inside, which [METHODOLOGY](METHODOLOGY.md) #47 already says to do.
- **A single-stream re-run overwrote the concurrency results file** because it didn't set the merge flag, dropping
  the `draft-mtp` grid. The numbers survived in the run transcript and were reassembled into the published data
  file, but the on-disk JSON was clobbered. Incremental result files need append/merge, not overwrite, when a
  later pass only fills part of the grid.

---

## 2. Explanations stated as findings before anyone tested them

These are worse than bad numbers because they spread into later decisions.

- **"vLLM decodes slowly on this model because of its hybrid Mamba/attention layers."** Carried for about
  two weeks.
  - **Actually:** AMD's own docs say the ROCm custom paged-attention kernel only runs at **≤ 16k
    context**. Above that vLLM falls back to Triton, and the log says so. Decode went 34.8 (68
    tokens) → 27.8 (3.7k) → 10.3 (35k). Every good vLLM number was below 16k and every bad one above it.
  - **Lesson:** the search should have happened before the measurements.
  - **And that was only half the story (08-26).** Those 10 tok/s runs also had `--enforce-eager` on
    (−72% decode) and used the default `ROCM_ATTN` backend's degraded fallback. The dedicated
    `--attention-backend TRITON_ATTN` with CUDA graphs on gave **32.0 tok/s at 35k**, unpatched.
  - **A correction to the correction:** it was then written that "the attention backend is a wash".
    AITER vs `TRITON_ATTN` is a wash, but the *default* backend is 3× slower at depth, so the backend
    choice is load-bearing.
- **"FP8 MoE plus hybrid sliding-window attention is what unlocks deep context."** This was inferred from a
  third-party repo. The MoE model decayed as hard as dense here: 67.4 tok/s at 3k → 5.36 at 120k.
- **"`cudagraph_capture_sizes` produced the 2× decode gain."** Five variables had changed and one was
  credited. A 2×2 ablation showed it's memory-only (decode 67.4 ± 0.06 in every cell). The speed came
  from the engine's int4 linear kernel.
- **"Newer vLLM containers are greedier because of the kernel."** It was ~3.7 GiB of CUDA-graph capture
  reserve that one flag recovers.
- **"llama.cpp's collapse at 4 parallel slots is an MTP interaction."** It was repeated from existing
  notes, and it reproduced with MTP absent.
  - *Later:* on the newer build (`434ddbb`), forced-length tests show no collapse up to 4 streams (72.7 tok/s
    aggregate), so the old collapse belonged to that older build.
- **"The permutation space contains no fix."** False; see the no-op attention variable above.
- **"TP=2 initialises but hangs."** TP=2 serves normally. The empty result cells came from a probe script
  wiped by a power-off.
- **"Replacing the RX 9070 XT would remove a straggler in TP."** Backwards: the R9700 is the slower card,
  so TP already runs at its pace.
- **"Tensor parallel costs context"** (a 42,325-token KV pool vs 79,872 on one GPU). Wrong baseline,
  caught by the owner. The configuration actually used for speed was single-GPU with a DFlash2 drafter,
  capped near 12k context. Against that, TP gave 3.4× the context and freed the drafter's 3.85 GB.
- **"llama.cpp on ROCm loses at depth because of the same 16k paged-attention cliff"** (09-09). That
  kernel limit belongs to vLLM's ROCm backend; llama.cpp uses its own attention kernels. The pattern
  (ROCm wins shallow, loses deep) was measured, but the mechanism was borrowed from a different engine
  and never tested. Treat it as unexplained.
- **"The matched-cards requirement is about VRAM."** Partly. The mixed R9700 + RX 9070 XT pair also fails
  because hipBLASLt picks a GEMM kernel configuration that is invalid on the consumer card
  (`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, always on that card).
- **"DFlash2 supersedes MTP for this model: settled."** Written 08-26, reversed 09-10.
  - After a llama.cpp rebuild, DFlash2 is still fastest for a single stream, but MTP wins at every
    concurrency level above one and uses ~2.5 GB less VRAM.
  - "Settled" was the wrong word for a result on an old build.
- **"MTP will be more style-consistent than a separate drafter."** The opposite: its code-vs-prose gap was
  wider at every tested `n_max`.
- **"The UPS with a dead battery causes the power-offs."**
  - **Proposed mechanism:** the AVR relay's break-before-make transfer.
  - **Actually:** the UPS was removed from the circuit and the power-offs continued. Current working
    theory is an ageing PSU failing to absorb load transients, not yet confirmed.
- **"The power-offs only happen under simultaneous dual-GPU load."** An event on apparently one card at a
  steady 230 W contradicted it. But that reading came from *summed* telemetry (see section 1).
  - A later per-card trace fits a *load-transition* theory: the second card spinning up while the first is
    already loaded.
- **"A power cap bounds the spikes."** A sub-millisecond watcher measured **488 W and 584 W peaks under a
  250 W cap** (1.9× and 2.3×). Caps limit averages, not transients.
- **"The 250 W cap is only needed for vLLM's dual-GPU startup."** Production was single-GPU, so the cap
  was costing ~9% prefill and ~19% decode for no live reason. It was later restored, as part of a
  profile that also brought the undervolt and a lower memory clock, when unexplained reboots
  continued.
- **ComfyUI RAM thrash (2026-09-10), three wrong theories in a row:**
  1. **Pinned memory.** No: the flag was already set and `Mlocked` read 0.
  2. **Add `--disable-smart-memory`.** That *caused* the UNet to be evicted on every prompt change (~78 s
     reload each time).
  3. **Add `--highvram`.** That fixed eviction but removed the offload fallback, so LoRA patching ran
     out of memory on the 16 GB card.
  - **The real cause:** ComfyUI's default `--cache-ram` lets the inactive model cache grow to 100% of
    system RAM. The owner corrected the framing more than once along the way.
- **"Turbo's shorter thinking will save prefill on every later turn"** (the assistant, 2026-09-10). Only if
  the client sends earlier reasoning back.
  - The benchmark's agent (Terminus-2) doesn't: a transcript showed 145.6k tokens generated but context
    peaking at 46.5k.
  - Hermes, the agent framework in production, strips all earlier reasoning by default.
  - So in both, the saving is generation time only.

---

## 3. Misleading sources

- **GGUF filenames.** Read `general.file_type` from the header instead.
  - Unsloth's `UD-Q4_K_XL` has file type `Q4_K_S`; "XL/M/S" are Unsloth layer-upcast tiers over a
    base type.
  - A file named `mmproj-f16` was `BF16`.
- **Tool GPU numbering.** `nvtop` and LACT number the two cards the opposite way from `rocm-smi`, ROCm and
  vLLM. Read each tool's mapping; never assume.
- **`rocm-smi`'s power figure** is literally "Average Graphics Package Power". A smooth ramp in its log can be
  the averaging window catching up with an instant change. Don't read electrical distress from its shape.
- **vLLM pipeline-parallel logs** only report memory for rank 0. The 16 GB card that actually bound the
  allocation never appeared in the logs and had to be sampled externally.
- **Container names.** Two vLLM image families were both named "therock" but used incompatible ROCm layouts
  (system `/opt/rocm` vs Python-wheel SDK).
  - Setting `HIP_PATH=/opt/rocm` in the wheel layout broke a JIT compile.
  - The error blamed an **unsupported GPU architecture**, which was false.
- **Image tags.** A tag reading `pinned-20260731` was on an image built on 06-13. Compare image IDs, not tags.
- **LACT configuration.** With a profile active, the top-level per-GPU entries do nothing. It also accepts
  over-range power values and clamps them silently. Confirm the effective cap with
  `rocm-smi --showmaxpower`, never by reading the YAML.
- **A handoff document from an earlier session:**
  - It misidentified a local NTFS disk as a network share.
  - Its suggested fix, `systemctl mask` on an fstab-generated mount unit, made systemd abandon
    `local-fs.target` and skip **every** local mount.
  - The real fix was `nofail`.
- **Docs vs the live system.**
  - Research docs described a "production" configuration while the running service had rolled back to
    a different model.
  - A systemd unit had silently lost the `-cram` prompt-cache flag for two days.
  - Diffing the running process against the saved launch line caught both. Check `systemctl` and the
    live process, not the doc.
- **The record is biased toward problems.**
  - Broken setups get documented because they need debugging; working ones get used and forgotten.
  - One recipe recorded a model at 30.88 tok/s, the owner remembered ~70, and a re-measurement gave
    67.41.
  - For a configuration that once worked, recollection can outrank the notes.
- **Vendor guidance.** AMD/Alibaba's launch post recommends MTP=2 for the R9700.
  - On this Linux/RADV build, MTP=4 is 15% faster single-stream on code.
  - MTP=2 is best for mixed or prose work.
  - So it's right for one workload and wrong for another, and the post doesn't say which.
- **Model cards.** A fine-tune card leads with ARC-C/ARC-E multiple-choice scores (non-thinking mode, MLX
  quants), a speed claim on unspecified hardware, and a "Q4_K_M minimum for tool calling" rule of thumb.
  None of these is wrong exactly, but none is decision data for an agentic GGUF deployment. That's why a
  real task benchmark was run.
- **"Prerequisites are already installed."** A handoff said another agent had installed Docker Compose v2
  before the benchmark. On the box: `docker compose` was unknown in every shell, no plugin existed in any
  standard path, and no package was installed. The same handoff's own checklist had recorded it as
  missing. The owner installed it (a user-level plugin, checksum-verified).
- **An upstream unit test in terminal-bench-mini fails on a fresh clone.** It expects 3 committed runs of one
  model and finds 5, because it's stale against newer committed results. The runner itself passes its
  other 99 tests. Not a blocker, but it looks like one.
- **A setting name that reads backwards.** Terminus-2's `proactive_summarization_threshold=8000` looks like
  "summarise at 8k tokens" (the owner read it that way). Its docstring says it's the number of *free* tokens
  below which summarisation starts, i.e. at ~254k used on a 262k window.
- **"Unsloth" on a model card.** A fine-tune card tagged `unsloth` and describing an Unsloth collaboration led to
  the belief that its GGUFs were Unsloth "UD" dynamic quants. The collaboration was the *training* stack. The
  tensor headers show a standard llama.cpp Q4_K_S layer mix with the fine-tuner's own importance matrix, a BF16
  output head and a Q8_0 MTP block.
- **A KL-divergence figure without its reference.** The same card reports KL 0.0025, which reads as "very close
  to base Qwen". The table's reference is the model's *own previous build stage*. Distance from base isn't
  reported.
- **Harness setup errors that look like model failures.** `RuntimeError: Command timed out after 120 seconds`
  on five tasks, only in one arm. The cause was external: plain-HTTP Ubuntu apt mirrors hanging overnight,
  hitting only `ubuntu:24.04` task images during the agent's tool install.

---

## 4. Misunderstandings between the owner and the assistants

Recorded in both directions, because either side can be the one who is wrong.

**Where the owner's belief was off:**

- **A UPS as a "load balancer."** What it actually does is AVR voltage regulation, which was the
  suspected harm at the time, plus surge protection via MOVs that degrade with age. It was removed.
- **"The reference Flash-Next benchmark ran on Strix Halo."** That run was on dual R9700s; the other
  published runs were on Strix Halo. Also, Strix Halo's unified-memory bandwidth is ~256 GB/s in theory
  (quad-channel LPDDR5X-8000), not "under 200", still ~2.5× below the R9700's GDDR6.
- **"The agent framework keeps the previous turn's reasoning and drops older reasoning."**
  - The owner's model of the trade-off was right: preserving reasoning is cache-stable but grows context;
    dropping it saves context but invalidates cache.
  - The framework (Hermes) actually strips *all* earlier reasoning unless `model.reasoning_echo: true`.
  - The "keep only since the last user message" behaviour is the Qwen chat template, and it only applies
    when reasoning is sent back.

**Where the assistant was off:**

- **Raised "confounds" the harness already controls.** Model-card sampling defaults, preserved-thinking
  flags and template kwargs were all identical by construction, because both models ran on the same
  harness. The owner: *"it's literally just plug in the model, run the benchmark."*
- **Suggested higher quants** (Q4_K_M minimum, Q6 recommended) without the owner's constraint: 262k context
  on one 32 GB card, leaving the second card for image generation.
- **Misread "non-MTP is not necessary"** as "skip the no-MTP tests." It meant the fine-tuner's separate
  non-MTP weights weren't needed.
- **Misattributed a statistic.** "7 of 9 models passed within two attempts" was said about
  `configure-git-webserver`; the source says it about two *other* tasks.
- **Two ComfyUI workflow errors.** An agent workflow used the wrong CLIP type (`qwen_image` instead of
  `krea2`). It also loaded the UNet with a different node class than the GUI, so ComfyUI treated it as a
  new model and built a second 12.5 GB copy. On 32 GB of RAM the kernel OOM killer took out ComfyUI *and*
  the LLM server.
- **Misread normal agent behaviour as failure.** A reported "4 passed, 10 failed" during a benchmark task
  was the agent's *own* self-test mid-iteration, not the grader. The grader later passed the task.
- **Took a stale belief as current.** A large speed tax on IQ quants, from the owner's past experience,
  showed as only 1–2% prefill on the current build. Neither side was wrong about their data; the software
  had moved.

---

## 5. Patterns

- **Silence is not a result.** A grep that returns nothing, a patch that matches nothing, a probe that
  answers "ready", a watchdog that reads 0: each looked like data. Check exit codes and add a positive
  control.
- **An explanation that was never measured is a hypothesis,** however settled it feels. The worst entries
  above were all explanations for results that already looked understood.
- **Search upstream before measuring.** The 16k kernel limit was documented the whole time.
- **Match everything but the variable:** depth, output length, power, thermal window, image build,
  harness.
- **Most corrections came from the owner questioning a confident claim,** not from an internal check. Build
  in the questioning.

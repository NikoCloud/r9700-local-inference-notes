# 08 — Agentic quality A/B: Core-19

> **Status, 2026-09-11:** both arms finished. Heretic ran 2026-09-10 15:00–22:31; Turbo ran 2026-09-11
> 01:54–08:57 (after a killed first attempt, below). A third, much smaller model ran on a laptop as a side
> experiment; its results are preliminary and listed at the end.

## Why this benchmark

The Turbo fine-tune claims better output with far less thinking ([07](07-model-qualification.md)). Speed tests
can't check that. This runs both models through real multi-step terminal tasks and records pass rate plus what
each pass cost in time and tokens.

## The suite

[kyuz0/terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini) (Terminal-Bench-Local), suite
**Core-19**:
- **Tasks:** 19 tasks vendored from Terminal-Bench 2.1, in isolated Docker containers. 6 software
  engineering, 4 system administration, 9 across debugging, security, data and file operations. Upstream
  labels them 3 easy, 13 medium and 3 hard.
- **Runner:** Harbor 0.20.0, with the **Terminus-2** agent driving each container.
- **Attempts:** up to **2 per task**; the second runs only if the first fails, after all first attempts finish.
  **3-hour agent timeout** per attempt; no turn or output-token caps. One task at a time.
- **Grading:** a hidden per-task verifier runs once at the end of each attempt. The agent never sees it. It can
  only test itself.
- **Context:** Terminus-2 summarises only when *free* context drops below ~8k tokens (at ~254k used on a 262k
  window). The setting name, `proactive_summarization_threshold=8000`, reads like "summarise at 8k"; it isn't.
- **Recorded per task:** duration, input/cached/output tokens, agent steps, and a full step-by-step transcript
  (ATIF format, including each step's reasoning).

## Arms

| | Arm 1 | Arm 2 |
|---|---|---|
| Model | heretic Q4_K_S | Turbo Q4_K_S |
| Server | identical: production flags (MTP n_max 2, `-np 2`, 262k context, q8_0 KV), llama.cpp `434ddbb` Vulkan, 250 W | same |
| Endpoint | `127.0.0.1:8085` (production stopped for the whole campaign) | same |
| Reasoning | template default (`xhigh`); the harness passes nothing | same |

**Run identity passed to the runner:** `--platform r9700 --engine llama.cpp --engine-version 434ddbb+vision-patch
--backend vulkan --backend-version mesa-26.2.2-radv --quant Q4_K_S --inference-profile mtp-n2`.

**Captured during the run:** the server log per arm (per-request timings, cache hits, draft acceptance) and GPU
power and temperature every 30 s.

## Before it could run

- **Docker Compose v2 was missing.** A handoff note said another agent had already installed the prerequisites.
  The owner installed the official user-level plugin (checksum verified).
- **One upstream unit test fails on a fresh clone** (a stale count of committed results). Not a blocker.
- **Loopback works.** Terminus-2 makes its model calls from the host, so a server bound to `127.0.0.1` is
  reachable.
- **The runner reuses a matching earlier result.** The smoke run's `git-leak-recovery` pass was reused in the
  full heretic arm. The Turbo arm ran all 19.

## The first Turbo run was killed

The first Turbo arm started 2026-09-10 22:31. After **1 h 50 min on its first task**
(`break-filter-js-from-html`: 77 agent steps, context grown to 98k, repeatedly trying variations of the same
kind of XSS bypass) the owner stopped it. Heretic had solved that task in 20 minutes with 12 steps.

That run is archived and **not counted**. The owner then asked for sanity checks with the reasoning streamed
live before re-running ([11](11-reasoning-traces-and-sanity-checks.md)). They found the model working (both code
tasks passed) but prone to intermittent loops, so the benchmark was re-run on the unchanged server line.

## Results

### Summary

| | heretic Q4_K_S | Turbo Q4_K_S |
|---|---|---|
| **pass@1** (attempt 1) | **16 / 19** | 9 / 19 |
| **pass@1 excluding setup errors** (see below) | 16 / 19 | **9 / 14** |
| **pass@1 on those same 14 tasks** | **11 / 14** | 9 / 14 |
| **pass@2** (with conditional retry) | **17 / 19** | 9 / 19 (no retry flipped) |
| Agent time, attempt 1 | 378 min | 133 min (14 tasks that ran) |
| Output tokens, attempt 1 | 914k | 214k (14 tasks that ran) |
| Minutes per solved task (attempt 1) | 23.6 | **14.7** |
| Output tokens per solved task (attempt 1) | 57.1k | **23.7k** |
| Median solved task | 9.3 min · 22.5k tokens | **3.2 min · 4.9k tokens** |
| Tasks solved per agent-hour (attempt 1) | 2.5 | **4.1** |
| Peak context per task, median / max (attempt 1) | 20.0k / 89.1k | 9.6k / 145.4k (262k in one retry) |

- **Heretic solves more;** Turbo solves what it solves **much faster and cheaper**.
- **On the 8 tasks both passed,** Turbo used **30 minutes vs 98** and **57k output tokens vs 258k**.
- **On the 14 tasks both could run,** Turbo won one heretic lost (`configure-git-webserver`) and lost three
  heretic won (`break-filter-js-from-html`, `build-cython-ext`, `llm-inference-batching-scheduler`).
- **Retries helped heretic only:** `mteb-retrieve` flipped to a genuine pass. None of Turbo's ten retries did.

An interactive chart of everything on this page: [data/core19/core19_ab.html](../data/core19/core19_ab.html).

### Setup errors: an outside outage, not the model

Five Turbo tasks never reached the agent in attempt 1. In attempt 2, four of them failed setup again (`mailman`
got through and ran into the 3 h timeout instead), and so did `extract-elf`, whose attempt 1 had set up fine.
Every one was `RuntimeError: Command timed out after 120 seconds` during agent setup:

- **What Terminus-2 does first:** installs `tmux` and `asciinema` inside the task container, with a hard-coded
  **120 s limit per command** (`_TOOL_INSTALL_TIMEOUT_SEC = 120` in Harbor's `tmux_session.py`).
- **What broke:** overnight, the **plain-HTTP Ubuntu apt mirrors hung** (archive.ubuntu.com, security.ubuntu.com:
  connect, then no response), from two machines on the same connection. HTTPS to the same mirror, Debian's mirror,
  PyPI, GitHub and Docker Hub all responded normally.
- **Who it hit:** every failing task is `FROM ubuntu:24.04`; every `python:3.13-slim-bookworm` task set up fine.
  Heretic's arm ran the same Ubuntu tasks the previous afternoon without errors.
- A read-only log of the mirror checks (every 10 minutes) showed it still failing intermittently at 08:30.

These are counted as **setup errors, not model failures**, and the table shows pass@1 both ways. The fair
comparison is the **14 tasks both models actually ran**. Re-running only the five affected Turbo tasks once the
mirror is healthy is an open item.

### Per task

`a1` / `a2` = attempt 1 / 2. Minutes and output tokens are per attempt.

| Task | heretic a1 | heretic a2 | Turbo a1 | Turbo a2 |
|---|---|---|---|---|
| break-filter-js-from-html | **pass** · 19.7 · 47.1k | – | fail · 3.1 · 7.1k | fail · 2.8 · 7.2k |
| build-cython-ext | **pass** · 23.3 · 36.3k | – | fail · 9.5 · 5.4k | fail · 20.4 · 12.8k |
| cobol-modernization | **pass** · 45.9 · 145.6k | – | **pass** · 7.7 · 22.9k | – |
| configure-git-webserver | fail · 6.5 · 16.9k | fail · 6.7 · 15.2k | **pass** · 8.8 · 13.7k | – |
| extract-elf | fail · 59.6 · 138.1k | fail · 39.7 · 130.2k | fail · 6.6 · 16.9k | setup error |
| fix-git | **pass** · 1.9 · 6.3k | – | **pass** · 0.9 · 2.2k | – |
| fix-ocaml-gc | **pass** · 12.3 · 14.7k | – | setup error | setup error |
| git-leak-recovery | **pass** · 2.7 · 8.7k (smoke run) | – | setup error | setup error |
| headless-terminal | **pass** · 28.3 · 52.9k | – | **pass** · 2.1 · 4.9k | – |
| llm-inference-batching-scheduler | **pass** · 66.3 · 179.2k | – | fail · 67.8 · 109.2k | fail · 34.5 · 42.7k |
| mailman | **pass** · 68.6 · 154.4k | – | setup error | 3 h timeout · 180 · 140.7k |
| mteb-retrieve | fail · 6.0 · 9.8k | **pass** · 7.9 · 15.5k | fail · 6.5 · 4.1k | fail · 7.3 · 4.5k |
| nginx-request-logging | **pass** · 3.8 · 12.2k | – | **pass** · 3.2 · 8.3k | – |
| openssl-selfsigned-cert | **pass** · 1.2 · 4.2k | – | **pass** · 1.2 · 3.5k | – |
| overfull-hbox | **pass** · 10.0 · 28.4k | – | setup error | setup error |
| pypi-server | **pass** · 2.1 · 3.5k | – | **pass** · 2.6 · 3.1k | – |
| regex-log | **pass** · 5.7 · 22.3k | – | setup error | setup error |
| sparql-university | **pass** · 8.6 · 22.8k | – | **pass** · 6.6 · 9.5k | – |
| sqlite-with-gcov | **pass** · 6.0 · 10.9k | – | **pass** · 5.9 · 3.0k | – |

### Failure analysis

**Heretic** (3 tasks failed attempt 1):
- **`configure-git-webserver`** (serve git pushes through a web server over SSH): the verifier's SSH clone got
  `Connection refused`. The agent built the repository and a web server, verified HTTP 200 carefully, but **never
  started an SSH server**, and declared the task done after 7 minutes. Attempt 2 failed the same way.
- **`extract-elf`** (parse an ELF binary and export memory values): **66.7% of the expected values against a
  required 75%**. Its own final check reported "93.1% coverage", measured against *its own estimate* of the
  total. Attempt 2: another 40 minutes, still short.
- **`mteb-retrieve`** (rank documents with a pinned embedding model): attempt 1 passed an invalid
  `task_name="retrieval"` to the model's query/passage prompt mode, then fell back to plain encoding and ranked
  the right document 7th instead of 5th. **Attempt 2 fixed the actual bug** (a valid retrieval task name with the
  proper prompt types) and passed.

**Turbo** (5 genuine attempt-1 failures, all failing again on retry except `extract-elf`, whose retry hit the
setup outage):
- **`break-filter-js-from-html`** (craft HTML that survives an XSS filter and still fires an alert): used a meta
  refresh to a base64 `data:` URL, which browsers block. Its own test script produced no output and exited 0, and
  Turbo concluded "the test script exited with code 0, confirming the bypass works".
- **`build-cython-ext`:** verifier: `chelpers Cython extension is not built`. Turbo's last message: "All three
  Cython extensions … are compiled and loaded from .so files."
- **`extract-elf`:** **0%** of expected values (heretic got 66.7%). Turbo's last message: "The extract.js program is
  complete and working."
- **`llm-inference-batching-scheduler`:** one latency metric 2.3% over its limit (2.76e8 vs 2.7e8). Turbo
  *knew* ("> 2.7e8 by 2.3%") and stopped at what it called the best achievable configuration. Attempt 1 ran 68
  minutes and grew the context to 145k. Heretic passed it.
- **`mteb-retrieve`:** returned the wrong document, then "The task is complete."
- **`mailman` (attempt 2 only; attempt 1 was a setup error):** the full 3 hours, 397 model calls, context filled
  to **262,143 tokens**, no result.

**Pattern:** heretic's failures were thorough work against its *own reading* of the task. Four of Turbo's five
were **confident completion claims on a check that was wrong or never run**, the same self-verification gap
found in the sanity-check traces ([11](11-reasoning-traces-and-sanity-checks.md)).

### Speed during the run

From each arm's server log, token-weighted per context-depth bin, all requests in both attempts:

| Context depth | heretic decode | Turbo decode | heretic MTP acc. | Turbo MTP acc. | heretic prefill | Turbo prefill |
|---|---|---|---|---|---|---|
| 0–10k | **64.6** | 55.9 | 0.75 | **0.79** | 794 | 795 |
| 10–20k | **60.2** | 55.3 | 0.71 | **0.82** | 767 | 716 |
| 20–40k | **56.0** | 52.1 | 0.70 | **0.81** | 677 | 595 |
| 40–60k | **51.0** | 50.5 | 0.71 | **0.86** | 598 | 543 |
| 60–90k | **48.1** | 45.8 | 0.71 | **0.86** | 433 | 447 |
| 90k+ | – | 35.2 | – | 0.86 | – | 244 |

Decode and prefill are in tok/s. Prefill is counted only for prompts adding ≥ 256 new tokens.

- **Turbo decodes 1–14% slower at every depth despite much higher MTP acceptance.** That fits its heavier output
  head (16-bit, 2.4 GB) being read for every token ([11](11-reasoning-traces-and-sanity-checks.md#why-the-two-ggufs-differ-in-size-and-speed)).
- **The gap is larger than the 6–8% measured in qualification** at shallow depth, and it shrinks with depth.
- **Turbo's 90k+ bin is almost entirely one timed-out retry** (`mailman`).
- **Power and heat:** both arms averaged ~250 W while busy (heretic 251 W, Turbo 249 W), at the 250 W cap.
  The driver's averaged power read above 300 W in **4.4% of Turbo samples vs 0.3% of heretic's** (peaks 359 vs
  345 W; caps don't bound short windows, see [06](06-power-and-stability.md)). Turbo ran cooler (junction
  67 °C vs 71 °C while busy).

### Verdict

Whether Turbo replaces heretic in production is the owner's decision; no production change was made. Production
was restored to heretic when the Turbo arm ended.

Evidence for that decision, as measured:
- **Reliability:** heretic 16/19 (17/19 with retries); Turbo 9/14 on the tasks it ran, no retries rescued, plus
  intermittent reasoning loops and confident false completion claims.
- **Cost:** Turbo is ~3× faster and ~4.5× cheaper in output tokens on the tasks both solve.
- **Unresolved:** five Turbo tasks lost to the mirror outage. Heretic passed all five, so even a clean re-run
  can at most bring Turbo to 14/19.

## Side experiment: MiniCPM5-2B on a laptop (preliminary)

A test of whether a ~2B model could serve as a cheap worker or critic in a swarm. **Quality only;** the hardware
makes the speed numbers meaningless for the intended target (a 16 GB RX 9070 XT).

- **Setup:** openbmb/MiniCPM5-2B official 4-bit (asymmetric GPTQ W4 g128, repacked in AWQ format), vLLM 0.29.0,
  GTX 1660 Ti Max-Q 6 GB, 65,536-token context (78k-token KV pool), **2 tasks at a time**, same runner and suite.
- **Attempt 1 (final):** **2 / 19 passed** (`git-leak-recovery`, `sqlite-with-gcov`); **2 / 16** excluding 3
  setup errors from the same Ubuntu mirror outage.
- **Attempt 2: stopped by the owner partway, published as is.** 5 retries finished with no flips (4 fails, 1
  setup error: `build-cython-ext`, `extract-elf`, `openssl-selfsigned-cert` and `sparql-university` failed again;
  `configure-git-webserver` hit the mirror). Unfinished retries aren't counted; given the model's first-attempt
  pattern they were unlikely to flip. **pass@2 as run: 2 / 19.**
- **How it failed:**
  - Declared success with requirements missing: no SSH account; a date not in the required YYYY-MM-DD format.
  - Admitted it couldn't solve a task (`extract-elf`: "cannot find the reference solution values"). That's a
    useful signal: a worker that reports uncertainty can be escalated rather than looped.
  - Killed its own container while trying to free a port (`nginx-request-logging`).
  - Ran into the 3 h timeout on 4 tasks, in one of them repeating the same analysis verbatim step after step
    (`sparql-university`).
- **Throughput collapsed at depth on this card:** ~20 tok/s combined early on, then turns dominated by
  30+ second prompt processing at 40–60k context (no FlashAttention on Turing). That's a laptop limit, not a
  model result.

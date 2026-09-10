# 08 — Agentic quality A/B: Core-19 (in progress)

> **Status, 2026-09-10 evening:** the heretic arm is running (7 of 9 attempt-1 tasks passed so far), and the
> Turbo arm follows automatically. This page will be updated with final results.

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
  **3-hour agent timeout** per attempt; no turn or output-token caps.
- **Grading:** a hidden per-task verifier runs once at the end of each attempt. The agent never sees it. It can
  only test itself.
- **Context:** Terminus-2 summarises only when free context drops below ~8k tokens.
- **Recorded per task:** duration, input/cached/output tokens, agent steps, and a full step-by-step transcript
  (ATIF format).

## Arms

| | Arm 1 | Arm 2 |
|---|---|---|
| Model | heretic Q4_K_S | Turbo Q4_K_S |
| Server | identical: production flags (MTP n_max 2, `-np 2`, 262k context, q8_0 KV), llama.cpp `434ddbb` Vulkan, 250 W | same |
| Endpoint | `127.0.0.1:8085` (production stopped for the whole campaign) | same |
| Reasoning | template default (`xhigh`); the harness passes nothing | same |

**Run identity passed to the runner:** `--platform r9700 --engine llama.cpp --engine-version 434ddbb+vision-patch
--backend vulkan --backend-version mesa-26.2.2-radv --quant Q4_K_S --inference-profile mtp-n2`.

**Captured during the run** (nothing else is recoverable afterwards):
- the server log per arm (per-request timings, cache hits, draft acceptance, any stray requests)
- GPU power and temperature every 30 s

## Before it could run

- **Docker Compose v2 was missing.** A handoff note said another agent had already installed the prerequisites.
  No `docker compose` existed in any shell, plugin path or package. The owner installed the official user-level
  plugin (checksum verified).
- **One upstream unit test fails on a fresh clone** (a stale count of committed results). The runner's other 99
  tests pass. Not a blocker.
- **Loopback works.** Terminus-2 makes its model calls from the host, so a server bound to `127.0.0.1` is
  reachable. Confirmed by a smoke run.
- **The runner reuses a matching earlier result.** The smoke run's `git-leak-recovery` pass (same model, quant,
  engine and profile) was reused in the full heretic arm, which ran 18 tasks. The Turbo arm has no smoke run,
  so it runs all 19.

## Heretic arm: attempt 1 so far

| Task | Result | Minutes | Output tokens | Input tokens | Cache hit |
|---|---|---|---|---|---|
| git-leak-recovery (from smoke run) | pass | 3 | 8,730 | 35,674 | 76% |
| break-filter-js-from-html | pass | 20 | 47,146 | 89,775 | 85% |
| build-cython-ext | pass | 24 | 36,272 | 1,090,970 | 96% |
| cobol-modernization | pass | 46 | 145,618 | 704,855 | 93% |
| configure-git-webserver | **fail** | 7 | 16,907 | 57,396 | 82% |
| extract-elf | **fail** | 60 | 138,079 | 440,363 | 87% |
| fix-git | pass | 2 | 6,263 | 25,239 | 75% |
| fix-ocaml-gc | pass | 20 | 14,742 | 358,704 | 94% |
| headless-terminal | pass | 29 | 52,943 | 935,228 | 95% |
| llm-inference-batching-scheduler | running | | | | |

### Why the two failures failed

- **`configure-git-webserver`** (serve git pushes through a web server over SSH):
  - The verifier's first step, a clone over SSH, got `connect to host localhost port 22: Connection refused`.
  - The agent had built the bare repository and a web server, and carefully verified HTTP 200 on two
    addresses, but **never started an SSH server**. It also used Python's `http.server` instead of Nginx.
  - It declared the task done after 7 minutes.
- **`extract-elf`** (parse an ELF binary and export memory values):
  - Output format passed, and no values were wrong, but it found **66.7% of the expected values against a
    required 75%**.
  - The agent's own final check reported "93.1% coverage", measured against *its own estimate* of the total.
  - That's 60 minutes and 138k output tokens of confident near-miss.

**Common pattern:** thorough self-testing against the agent's *own* reading of the task. It verified what it
remembered to build, not what was asked. Two tasks aren't enough to call it a trait.

### Observations while watching

- **"4 passed, 10 failed" mid-task was the agent's own test script, not the grader.** It fixed the
  implementation within two minutes (14/14 on its own tests), and the task passed the verifier.
- **Context stayed small:** peaks of 5–18% of 262k on finished tasks, and no summarisation. Terminus-2 doesn't
  send earlier reasoning back ([10](10-agent-harness-lessons.md)), so 145k generated tokens produced a 46.5k
  peak context.
- **The GPU alternates between ~270 W bursts and ~15 W idle** while the agent waits on long commands in its
  container, polling every 20–30 s. That's normal: a stall looks different (no new transcript steps, no server
  activity).
- **The server log stayed clean:** no truncated prompts, context shifts or error lines.

## Planned comparison

- **Pass rate:** pass@1 and pass@2 per arm, broken down per task. The hard tier is where differences are
  expected to show.
- **Efficiency per task,** from the runner's own records, **attempt 1 only** (attempt 2 only exists for
  failures, which would skew totals):
  - minutes and output tokens per *solved* task
  - tasks solved per hour
- **What efficiency can and can't show:**
  - Turbo generates 6–8% slower ([07](07-model-qualification.md)), so it needs about that much less output per
    task to break even on time.
  - In this harness, a model thinking less saves generation time only. It doesn't shrink later prompts.

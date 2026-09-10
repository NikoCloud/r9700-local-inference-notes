# Methodology: rules learned the hard way

Each rule points at the incident that created it ([MISTAKES.md](MISTAKES.md)). They're written for
llama.cpp / vLLM on AMD, but most apply to any local-inference benchmarking.

## Before measuring

1. **Search upstream first.** Kernel limits, known regressions and hardware quirks are often documented.
   The 16k ROCm paged-attention limit was, and weeks were spent on a wrong theory instead.
2. **Pre-register the decision rule.** Write down what result would change the decision *before* running
   anything, and don't renegotiate afterwards. Example: "switch only if throughput gain ≥ 25% and no
   detectable quality regression"; "under 18 tok/s disqualifies"; "any TG collapse under concurrency
   disqualifies".
3. **Verify the live system, not the docs.** Diff the running process's command line against the saved
   launch line before and after changes. Docs drift; systemd units lose flags.
4. **Save a restore line before stopping anything.** Record the exact live command (`pgrep -af`) to a
   file first, and restore production in a `finally` or `trap`, whatever happens.

## One ruler for every candidate

5. **Change one variable.** Same binary, same flags, same context size, same power profile, same thermal
   window. For a model swap, *only* `-m` changes.
6. **Co-measure in one session.** The machine is non-stationary (decode once drifted 36% across two days).
   Re-measure the control alongside candidates. If a candidate must be added later, re-check the control
   on the live server first; here that meant a ~2 minute drift check with no swap.
7. **Speed runs use thinking off** (`chat_template_kwargs.enable_thinking: false`) and temperature 0 with a
   fixed seed. A model's reasoning default can consume the whole token budget.
8. **Unique prompt per request.** Identical prompts hit the prefix cache and turn a prefill or batching
   measurement into a cache-hit measurement.
9. **Measure at real depth.** Shallow and deep rankings can invert (ROCm vs Vulkan did, by ~27% each way).
   Include at least one shallow and one deep point; this project used 2.4k / 15k / 39k / 129k / 184k.
10. **Separate content types.** Speculative-decoding acceptance differs up to 3× between code and prose.
    Report both.

## Measuring correctly

11. **Readiness = `GET /health` returning 200,** not `/v1/models`, which answers while the model is still
    loading.
12. **Prefill from time-to-first-token on a streamed request,** counting reasoning *and* content deltas.
    Never infer it by subtraction.
13. **Token counts from the server's `usage`,** not streamed chunks. Speculative decoding packs several
    tokens per chunk.
14. **Force output length when comparing generation speed.** Use `ignore_eos: true` with a fixed
    `max_tokens`. Models stop at different lengths, and aggregate tok/s at natural length mostly
    measures verbosity. Check `completion_tokens == max_tokens` on every request.
15. **Decode windows of 300+ tokens.** Short windows are dominated by startup transients.
16. **Size prompts with `/tokenize`.** Character-per-token estimates were off by up to 70%.
17. **Concurrency means `-np` ≥ the number of streams.** Report per-stream speed *and* aggregate. A
    realistic deep-context concurrency test measures prefill overlap, so label it as such.
18. **Any prefill above the compute ceiling is a cache hit.** For a dense 27B on the R9700 that's about
    1,772 tok/s (95.7 TFLOPS ÷ 2N FLOPs per token).
19. **Coherence-check outputs.** A distinct-word ratio on the tail catches degenerate repetition. A fast
    number from a looping model is worthless.

## Capability checks

20. **Verify a feature by running it.** Launch with `--spec-type draft-mtp` and read the live draft
    acceptance in the server log. Don't reverse-engineer tensor names; this GGUF conversion stores MTP
    as an extra ordinary layer.
21. **Read the GGUF header** (`general.file_type`), not the filename.
22. **Every patch asserts it applied.** Every grep that "finds nothing" gets a positive control.

## Isolation and contamination

23. **Benchmark servers bind `127.0.0.1:8085`; production stays on `0.0.0.0:8080`.** Crons, context
    compaction, agent turns and anything else pointed at production then fail loudly instead of
    silently sharing the GPU with a test.
24. **Audit every run.** Count distinct server task IDs (`launch_slot_: … task N`) and compare against
    the requests the script sent (or agent steps, for agentic runs). A count that's identical across
    models and time windows is the harness's own; a random excess is contamination.
25. **Evict everything else from the GPU first, and watch for jobs starting after you.** A second job
    silently dropped decode from 48.8 to 9.3 tok/s.
26. **Never reuse a warmed container for a cold measurement.**

## Long runs

27. **Everything long runs in named tmux sessions** (servers, benchmarks, downloads). They survive dropped
    connections, and humans or agents can attach (read-only: `tmux attach -t <name> -r`).
28. **Before a long run, set up only the capture that's unrecoverable afterwards:**
    - the inference server log to a per-run file (per-request timings, cache hits, draft acceptance,
      stray requests)
    - per-card GPU power and temperature at a fixed interval

    Build analysis tooling later, against the real data.
29. **Results never go to `/tmp`** if it's tmpfs. A power event erased a completed sweep.
30. **Monitors emit milestones and alerts, not logs:** one line per finished task, plus any abort,
    missing session or unreachable host. Silence must not be able to hide a crash.

## Power and hardware telemetry

31. **Read board power alongside utilisation.** 100% utilisation at 83 W of a 250 W cap was a rank
    spin-waiting on a dead peer, not work.
32. **Log each card separately, from a machine that survives the crash.** Summed power made one incident
    undiagnosable.
33. **Averaged telemetry can't see transients.** Sub-millisecond peaks need a fast watcher; `rocm-smi`'s
    "Average Graphics Package Power" smooths them away.
34. **Confirm a power cap with `rocm-smi --showmaxpower`,** not the config file.

## Shell hygiene

35. **Strip CRLF** from any script written on Windows (`sed -i 's/\r$//'`).
36. **Never `pkill -f` inside a container that shares the host PID namespace;** use `docker stop`. Guard
    `pgrep`/`pkill` patterns against matching their own shell (`pgrep -f "[m]ain.py --port 8188"`).
37. **Check the command's exit status, not only its output.**

## Writing it up

38. **Mark superseded claims inline, where they appear,** and keep an errata table at the top. Don't delete
    wrong conclusions; strike them and point to the correction.
39. **An explanation that was never measured is a hypothesis.** Label it that way.
40. **Record working configurations too.** The record naturally fills with problems.

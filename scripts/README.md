# scripts/

The harnesses behind the numbers, lightly sanitised (paths and machine names generalised). They were written for
one machine, so expect to edit paths, model files and ports. They're here so the methodology can be checked,
not as a polished tool.

## `harness/`: shared measuring tools

| File | What it measures |
|---|---|
| `bench_sweep.py` | OpenAI-compatible endpoint benchmark: large-prompt prefill (TTFT-based) and decode, a small-prompt sanity request, and a concurrency sweep. Unique prompt per request; temperature 0, seed 42; thinking on, off or template default. Reads `usage` for token counts. |
| `conc_test.py` | 1/2/3 simultaneous streams; per-stream and aggregate tok/s. |
| `session_sim.py` | A multi-turn conversation that reaches depth the way real use does. Used instead of synthetic depth priming, which triggered a ROCm-only fault. |
| `nmax_sweep_mtp.py` | Speculative-decoding `--spec-draft-n-max` sweep on code vs prose prompts, reading draft acceptance from the server log. |

## `qualification/`: the 2026-09-10 four-model qualification ([docs/07](../docs/07-model-qualification.md))

| File | Pass |
|---|---|
| `phase1_turbo_ab.py` | **Pass A, MTP on.** VRAM at `-np` 1–4; code and prose generation with per-request draft acceptance; prefill @ 42k; 1–3 stream ladder. Saves the production restore line and restores production in `finally`. |
| `phase1b_nospec.py` | **Pass B, no speculation.** Depth ladder (2.4k → 184k) for prefill and generation at natural length; 1–4 stream batching ladder; 2 streams at ~33k. |
| `phase1c_fixedlen.py` | **Pass C, forced length** (`ignore_eos`, 768 tokens). Generation at depth and the batching ladder, with verbosity removed as a variable. Added after pass B showed length-confounded results. |
| `phase1_unleashed.py` | Runs a later-added model through all three passes, after a drift check against the live control. |
| `build_page.py`, `turbo_qual_template.html` | Build the interactive chart in `data/` from the JSON results. |

**Launch shape:**
- Every pass uses the production launch line with only `-m` (and `-np`) changed. Pass B/C also drop the two
  `--spec-*` flags.
- After this campaign the test server was moved to `127.0.0.1:8085`, while production-health checks stay on
  8080 ([METHODOLOGY](../METHODOLOGY.md) #23).

## `core19/`: the agentic A/B campaign ([docs/08](../docs/08-agentic-benchmark-core19.md))

| File | Role |
|---|---|
| `core19_campaign.sh` | Stops production (restore line first), starts each arm's server on `127.0.0.1:8085` with the production flags, starts per-arm capture (server log + GPU sampler), runs `doctor`, then kyuz0/terminal-bench-mini with full run identity. Modes: `smoke`, `full` (both arms back-to-back), `full-arm <name>`. Restores production on exit. |
| `gpu_sampler.sh` | One JSON line every 30 s: timestamp plus `rocm-smi` power and temperature for both GPUs. |
| `monitor_poll.sh` | Prints only *new* events since the last call (campaign log lines, one line per finished trial with reward/time/tokens, liveness alerts). Designed to be polled by a remote monitor so every line is worth a notification. |
| `c19_report.py` | Builds the per-task, per-attempt report and arm summaries (pass@1 raw and excluding setup errors, pass@2, minutes and tokens per solved task) from the runner's job directories. |
| `c19_live_speed.py` | Parses each arm's llama-server log: per-request prefill/decode/MTP acceptance by context depth, and per-task stats matched to each trial's agent time window. |
| `sanity_live.py` | The three-task sanity check (constrained poem, bug hunt with hidden tests, stdlib physics build with recomputed checks), streamed live; saves every call as a JSONL trace. |
| `loop_ab.py` | Loop A/B: relaunches the server per condition (KV type, model) and runs the poem N times, cancelling only on a detected reasoning loop. |
| `extract_traces.py` | Turns tee'd terminal logs from the two scripts above into structured JSONL traces and readable per-run markdown. |
| `analyze_traces.py` | Heuristic trace detectors: drafts graded against the final answer (reversal), per-line word-count accuracy, stated totals, and where each planted bug is first diagnosed. Validate its hits against the text before quoting them. |
| `build_core19_page.py` | Builds `data/core19/core19_ab.html` from the JSON in `data/core19/`. |
| `pld_test.py` | Prompt-lookup / n-gram speculative decoding ([docs/12](../docs/12-prompt-lookup-decoding.md)): a sanity gate (n-gram alone must accept drafts on a copy task, else abort), then a matrix of none / draft-mtp / ngram-mod / ngram-mod,draft-mtp across copy/code/prose, with an output-identity check. Stops and restores production. |
| `pld_conc.py` | Concurrency scaling for the same configs at `-np` 1/2/4 (relaunches the server with matching `-np`), reporting per-stream and aggregate decode. Reuses `pld_test.py` as a module. |

## Licence

MIT (see [../LICENSE](../LICENSE)).

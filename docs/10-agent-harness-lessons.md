# 10 — What the agent framework did to the inference server

Two AI agents run on top of the llama.cpp server here, built on
[Hermes Agent](https://github.com/NousResearch/hermes-agent). Some of the largest speed problems in this
project came from how an agent framework uses the server, not from the engine. These notes are
framework-specific, but the mechanisms are general.

## Where agent turn time actually goes

Parsed from 422 real agent turns in the server journal, not a benchmark:

| | |
|---|---|
| Share of wall time: prefill / decode | **17.5% / 82.5%** |
| Decode share of a median turn / p90 turn | 85% / 97% |
| Median prompt processed | 874 tokens (prefix caching works) |
| Completion: median / mean / max | 624 / 1,556 / 24,531 tokens |

A cold 40k-token prefill is a p90 event; the common case is a few hundred new tokens plus a long generation.
Optimising decode matters most day to day. Cold prefill matters for the first turn and after anything that
invalidates the cache.

## The fixed prefix: tools, not personality

- A cold prompt was **40,466 tokens**, measured by serialising a real request's tool schemas. **Tools were
  69% of it,** not the system prompt or the skills index.
- Removing two unused toolsets and tightening the system prompt cut **5,322 tokens**, ~7 s off every cold
  start. The server's own cold prefill dropped to 35,314.
- Tools from MCP servers cost nothing per turn when the framework defers them behind a search tool (118 tools,
  zero prefix cost).

## The second request nobody could see

Hermes has a `background_review` feature, **on by default**:
- After every turn, it forks the agent and replays the full conversation against the same model and server.
- So every turn fired a **second full-context inference** (14,000–18,000 tokens) into the same slot pool.
- Prefill with both requests live fell from ~743 tok/s to 330–516.
- It survived restarts and fresh sessions, which is why "a simple hi takes 120 s" reproduced on a clean
  session.

Pointing that feature at a different provider cut its local cost to 87 tokens per turn.

## The host prompt cache that silently turned off

`-cram 12288` lets llama.cpp keep prompt state in host RAM:
- **With it:** cache evictions went from 45 to 0, and warm turns became 4–12-token deltas instead of full
  re-prefills.
- **Then the systemd unit was edited and lost that flag.** The cache was off on every boot for two days.
- **Caught by:** diffing the running process's command line against the saved launch line.

## Keep automatic traffic off benchmark servers

Crons, context compaction, skill updates and agent turns all target the production port. Benchmark servers
now bind a different port on loopback only (`127.0.0.1:8085`), so that traffic fails loudly instead of
contaminating a measurement ([METHODOLOGY](../METHODOLOGY.md) #23–24).

## Reasoning traces: kept or dropped between turns?

This decides how much a model's *thinking* costs beyond generation time.

**On this hybrid model, the previous turn's output is always prefilled again once.**
- In a committed Core-19 transcript (86 steps), the server's prompt cache covered exactly the previous
  prompt, less a 4-token re-rendered header, at every step. The model's own output was never reused.
- **Plausible explanation, not confirmed:** llama.cpp only saves restore points for the recurrent state at
  the end of prompt processing, not after generation.
- Everything older stays cached: 98% of input tokens were cache hits.

**What gets sent back depends on the client:**

| Client | Earlier `reasoning_content` sent back? | Effect of a model thinking less |
|---|---|---|
| Hermes, default (`model.reasoning_echo: false`) | **No,** stripped from every earlier assistant message | Saves generation time only |
| Hermes with `model.reasoning_echo: true` | Yes, for every earlier message | Also saves prefill of that reasoning on the next turn, and context growth |
| Terminus-2 (the Core-19 agent) | No | Generation time only. One task generated 145.6k tokens while its context peaked at 46.5k. |

**The Qwen3.8 chat templates** (heretic and Turbo) keep reasoning on assistant messages when
`preserve_thinking` is unset or true, and otherwise only after the last user message. The server logs "chat
template supports preserving reasoning, enabled by default". None of that matters unless the client sends
the reasoning back.

**The trade-off,** once reasoning is being sent back:
- **Preserve all reasoning:** earlier turns never change, so the cache stays stable, but context fills with
  thinking.
- **Keep only reasoning since the last user message:** context stays small, but each new user message removes
  the previous exchange's reasoning from the rendered history, which invalidates the cache from that point.

## Reasoning effort is not a dial

In these Qwen3.8 templates:
- `xhigh` (the default) and `low` inject a reasoning instruction into the system prompt.
- **`medium` injects nothing,** so it's the model's unsteered behaviour.

The owner's experience across many deliverables: `medium` generates the fewest tokens but gives the worst
results, and `low` is about as long as `xhigh`. Benchmarks that don't pass `reasoning_effort` run at `xhigh`.

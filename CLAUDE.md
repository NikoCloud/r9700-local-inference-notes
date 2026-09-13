# Updating these notes

This repository is written while the machine is being used. Anything just measured or discovered
belongs here in the same session. The rules below are for *writing*; the rules for *measuring* live in
[METHODOLOGY.md](METHODOLOGY.md) and apply to anything published.

## Where each kind of change goes

| What changed | Where it goes |
|---|---|
| A new discovery, experiment or result | New `docs/NN-topic.md` (start from [docs/_TEMPLATE.md](docs/_TEMPLATE.md)) **plus** a ledger entry |
| A lever was swept, or its numbers moved (power cap, depth, speculation, backend, caching) | Update that row in [LEVERS.md](LEVERS.md) — swept range + deltas. Effects only; never a "current config" snapshot |
| Something previously published was wrong | [MISTAKES.md](MISTAKES.md) — strike through and correct; never delete ([METHODOLOGY.md](METHODOLOGY.md) #38) |
| A problem opened, changed or got resolved | [OPEN-PROBLEMS.md](OPEN-PROBLEMS.md) |
| A *headline* finding that changes the front page | The relevant digest row in [README.md](README.md), and its "last material update" date |

## The update checklist

When a finding lands:

1. **Doc first.** Full detail in `docs/NN-topic.md`: date box, short version, sections, raw-data
   footer. One finding per doc — unless it continues an existing thread, in which case add a section
   to that doc (`docs/13 §6b` style) rather than a new file.
2. **Ledger.** Prepend one entry to [FINDINGS.md](FINDINGS.md), newest first: `## YYYY-MM-DD · title`,
   a tag line, 2–5 bullets each carrying its exact numbers, then `→` links to doc and raw data. The
   ledger states the *current understanding*; error history belongs in MISTAKES.md only.
3. **Levers.** If the finding sweeps a knob or moves a lever's numbers, update that row in
   [LEVERS.md](LEVERS.md): swept range, deltas, link. This repo publishes *effects* — configurations
   churn, deltas transfer. There is deliberately no "what this box currently runs" file.
4. **Corrections.** If anything previously published was wrong, add the MISTAKES.md entry in the same
   session. Corrected numbers get an inline strikethrough + pointer at the point of the original claim.
5. **Digest.** Only headline-level findings touch the README — keep it one line, keep it true.
6. **Check.** Run `python scripts/check_docs.py` before pushing. Fix what it flags.
7. **Commit.** Message style follows the log: `docs/NN §X: <what changed>` or `<FILE>: <what changed>`.

## Writing rules

- **Dated snapshots stay snapshots.** Don't rewrite old docs to match new knowledge; correct inline
  (strikethrough + link) or write the new doc. Numbers in a dated doc stay as measured that day.
- **Numbers** carry units, depth, power cap, date and sample size. Copy exact values from raw data —
  never re-derive from memory. A number that can't be traced to raw data or a logged run gets said so.
- **Hypotheses get labelled.** An explanation that was never measured is a hypothesis, not a finding.
- **Voice**: dense, plain, honest. No marketing adjectives. "We were wrong about X" is normal here.
- **Links**: relative paths matching the existing files (`docs/…`, `data/…` from the repo root;
  `../` from inside a subdirectory).
- **Line endings**: LF everywhere — `.gitattributes` enforces it, don't fight it.

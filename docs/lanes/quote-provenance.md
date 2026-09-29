# Quote provenance for LEAD-74 / LEAD-77

## Goal
Resolve historical and self-captured quote clocks and supply a metadata-only validator for both lanes.

## State
Protocol declared before outcomes: 2023-2025 NFL archive plus one live example; CFB separately for LEAD-77/archive reconciliation. Target: provenance eligibility. Terms: request, returned snapshot, bookmaker/market update, receipt, cutoff. Pair: two opposing spread prices at one event/book/snapshot/line. Metrics: exact counts and exclusions; no fitted parameters/folds or temporal deadline join. Outcome looks 0; effect intervals, probability_positive, IS/OOS and gap unestimated. Existing research protocols unchanged.
**Measured:** 8,104 historical snapshots / 5,031,980 rows; all file/clock checks pass. 2,220,422 pregame as-of pairs (NFL 2,056,874; CFB 163,548); 0 historical availability-proven pairs under the strict evidence rule. Bovada: 16/16 spread pairs pass at receipt time; 0 have bookmaker ages. Exact census, no sampling intervals.
**Read/inferred:** historical observed time is the returned archive clock, not local ingestion. Local evidence does not establish an archive correction guarantee; no revision was demonstrated. Bovada stamps after response receipt; ESPN stamps before fetching. See docs/quote_provenance.md for file:line evidence and the source rules.

## Tried
**Measured:** ran `UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/quote_provenance_check.py --include-snapshot data/market/raw/20260929T203730Z` once; exit 0, zero structural errors. Aggregate log: `%TEMP%/quote-provenance-check.json`; durable counts: docs/quote_provenance.md. Scoped Ruff format/check passed. One streaming audit, capped threads; no scores, tests added, registry writes, serving changes, or Git mutations.

## Record commands
None: source provenance inventory has no fitted effect, interval, probability_positive, or research verdict to register. Inventing a weak-signals/rotation command would invent evidence. The orchestrator owns any future serial record commands with --plain-summary.

## Next
Orchestrator: resolve the provider's contemporaneous archive/correction contract using the source rule, then certify actual frozen-opener/deadline joins for LEAD-74/77 unit 2. The validator supports a single snapshot and explicit --deadline; default request cutoffs are not pool deadlines.

## Open
Historical public availability remains unproven by local evidence; missing local ingestion alone neither proves leakage nor refutes either signal. No research closure. Only the assigned three files changed; the orchestrator owns review and commits.

## Decision (root, 2026-09-29)
Research units treat a historical snapshot's provider `observed_at_utc` as the availability time (**reported**: the provider documents historical snapshots as point-in-time; **read**: the served market-move term and LEAD-69/73 already rely on the same snapshots). Each unit states this assumption in its declaration and reports results as conditional on it. Self-captured Bovada quotes use the post-receipt clock; ESPN pre-fetch clocks do not certify availability.

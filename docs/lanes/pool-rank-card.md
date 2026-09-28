# Pool rank card

## Goal
POOL-01: measure pool-field behavior chronologically using genuine predeadline inputs.

## State
**Measured:** corrected `scripts/pool_field_share_fit.py` fails before fitting on
ATL-PIT's baseline lineage after publication. The complete readiness audit also
finds no historical public capture with verified response completion before publication.
Thirty-one declared games retain prepublication request-start evidence and
`2026_02_DET_BUF` has none. No games were dropped, no future capture substituted,
and no corrected fit was run. The prior field-fit verdict is superseded pending
valid remeasurement; this is not a negative result.

## Tried
Frozen sides/probabilities now come from the actual published ledger. Input lines,
push outcomes, source capture time, kickoff identity, and pool deadlines are checked.
Malformed locked probability rows fail rather than disappearing from the population.
Baseline forecasts, decision bindings, and public predictors must now exist by the
exact frozen publication, not merely by the later pick deadline. Each market quote
must also record an observation no later than its forecast creation. A missing quote
time is unavailable provenance; a later observation is a potential leakage flag. Complete
published forecast bindings take precedence over the legacy decision ledger; a partial
or mutated binding never falls back. `scripts/research_input_readiness.py` retains each
declared row with its readiness reason and immutable source hashes.
The timestamped market proxy also requires one bookmaker in a completed immutable raw
snapshot to match the game, opposing lines and both prices. Nominal or differently timed
prices remain lawful forecast inputs but are unavailable as a measured market baseline.
Legacy public captures are reparsed from their immutable raw HTML with explicit game
home/away team IDs; their raw HTML, saved index, manifest and parser hashes are retained.
This corrects side identity in memory without rewriting the original index. Captured
home/away lines and prices remain separate even when they are not opposing. Their
legacy timestamps record request start, so they remain visible as unverified chronology
and cannot enter a fit. New captures must declare a response-received time.
The future corrected replay retains all 32 Week 1–2 games for input/proxy audits,
fits only 16 Week 1 games, and scores only 16 Week 2 games. Week 1 has no prior
training week and cannot have a fitted-field replay. Future-to-past, same-week,
and leave-one-out fits are invalid for this chronological question.
[Current evidence](../seven_backlog_completion_20260928.md) records every unavailable input;
[history](done/pool-rank-card-before-20260928-checkpoint.md) preserves previous work.

## Next
The scheduler now includes Tuesday noon and Thursday noon public-split captures, in
addition to Saturday and Sunday. Tuesday precedes the 12:20 lock and Thursday covers
later changes. Collect genuine prepublication inputs and
predeclare a new chronological population before fitting; do not reduce or backfill
the failed historical population. The earlier deadline-only proxy replay is superseded.

## Open
Public-split captures omit contemporaneous spread lines. Rank simulation uses
conditional non-push probabilities and remains an unserved counterfactual.
No valid registry effect unit exists for ranks/week; do not force it into accuracy
pooling. A field estimate is required before the standing-aware variant. Zero
crossing closes no signal; one fitted probability selects each served side.

# Pool rank card

## Goal
POOL-01: measure pool-field behavior chronologically using genuine predeadline inputs.

## State
**Measured:** corrected `scripts/pool_field_share_fit.py` fails closed because
`2026_02_DET_BUF` lacks a saved predeadline public split. No games were dropped,
no future capture substituted, and no corrected fit was run. The prior field-fit
verdict is superseded pending valid remeasurement; this is not a negative result.

## Tried
Frozen sides/probabilities now come from the actual published ledger. Input lines,
push outcomes, source capture time, kickoff identity, and pool deadlines are checked.
Malformed locked probability rows fail rather than disappearing from the population.
The future corrected replay retains all 32 Week 1–2 games for input/proxy audits,
fits only 16 Week 1 games, and scores only 16 Week 2 games. Week 1 has no prior
training week and cannot have a fitted-field replay. Future-to-past, same-week,
and leave-one-out fits are invalid for this chronological question.
[Checkpoint evidence](../backlog_research_20260928.md) records the missing input;
[history](done/pool-rank-card-before-20260928-checkpoint.md) preserves previous work.

## Next
Collect genuinely predeadline public inputs prospectively. Predeclare a new
chronological population before fitting; do not reduce the failed historical population.

## Open
Public-split captures omit contemporaneous spread lines. Rank simulation uses
conditional non-push probabilities and remains an unserved counterfactual.
No valid registry effect unit exists for ranks/week; do not force it into accuracy
pooling. A field estimate is required before the standing-aware variant. Zero
crossing closes no signal; one fitted probability selects each served side.

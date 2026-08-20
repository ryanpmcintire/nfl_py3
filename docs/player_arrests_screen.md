# Pregame player-arrest disruption screen

## Predeclaration

**Written before either ATS result was computed.** This family asks whether a
team identified by USA Today's incident-time team field covers less often when
one of its players had a reported arrest/citation event during the 14 calendar
days strictly before the pool's Tuesday decision date.

The mechanism is short-lived roster uncertainty and organizational distraction,
so the direction is fixed in advance: **fade the affected team** (`sign=-1`).
The 14-day window was selected from exposure counts without reading outcomes:
it yields enough historical team-games to measure while staying close to the
late-arriving mechanism. It is not tuned after seeing ATS results.

Two commensurable views are declared together:

1. Close grade, 2009-2025, to use the full schedule/source overlap.
2. Opener grade, 2020-2025, because the pool settles against its frozen opener
   and that grade governs any decision.

Both are subset screens, not model retraining. Each compares the affected
team's cover rate with every other REG-season team-game and scales the gap by
the affected fraction of the slate. They are overlapping views of one family,
not independent inputs to a pooled estimate.

## Point-in-time contract

The only input is
`data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet`.
Outcome/resolution text, descriptions, and links are mechanically absent.
Incident dates equal to the Tuesday decision date are excluded because the
source provides dates but no intra-day publication timestamps. Future incidents
and incidents more than 14 days old cannot flag a game. `Free agent` rows and
unmapped team codes are excluded.

The source's team attribution is canonicalized to the schedule (`OAK→LV`,
`SD→LAC`, `STL→LA`, `JAC→JAX`, `IN→IND`). This screen does not infer suspension,
availability, guilt, or later case outcome.

## Adjudication

Sparse, small effects are evaluated continuously with `probability_positive`;
uncertainty alone is never a reason to discard one. Unless the mechanism is
genuinely refuted under an admissible closing ground, each result is category 3
and must be recorded `unresolved_below_power` through the declarative
experiment runner.

No production or prospective policy change is predeclared from this first
screen. A positive expected-value opener read can become a parameter-free
prospective challenger without claiming the historical mechanism is resolved.

## Recorded result

The close-grade run is recorded at
`artifacts/experiment_runner/20260820T160312Z`: 247 flagged and 8,387
complement team-games produced a fixed-fade effect of -0.1371 accuracy points,
a week-blocked interval of [-0.3095, +0.0356], and
`probability_positive=0.0572`. The result is
`unresolved_below_power`; no admissible closing ground was established.

The opener-grade run is recorded at
`artifacts/experiment_runner/20260820T160322Z`: 50 flagged and 2,956 complement
team-games produced a fixed-fade effect of -0.1015 accuracy points, a
week-blocked interval of [-0.3190, +0.1258], and
`probability_positive=0.1653`. It too is `unresolved_below_power` with no
admissible closing ground. The six-season secondary bootstrap is below the
runner's measured block-count floor, so it is not used for the decision.

Because the sign was fixed as a fade, these negative estimates mean the raw
historical split leans toward backing the affected team instead. That opposite
direction is a post-result lead, not a revision of this predeclaration. It
requires a direct comparison with the 53.4% opener-grade production policy and
a live Tuesday source-refresh contract before it can become an active
prospective challenger.

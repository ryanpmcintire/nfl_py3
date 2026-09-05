# Injury trajectory refresh screens (CX14)

## Predeclaration — 2026-09-05, before arm computation

Read (ROADMAP.md:935–938): these four leads prescribe fading the affected
team. Freeze all four definitions below before scoring any arm. The shared
family is `injury_trajectory_refresh_cx14`; units are full-card accuracy points
on the same 2022–2025 regular-season opener archive, paired by game. No fitting
or outcome-based threshold selection is performed. Chronological season results
are descriptive screens on previously mined seasons, not independent confirmation.

Use the active-model-matched opener per-game artifact and reconstruct the played
four-member union (coach fade, division revenge, arrests, spread gap), applying
each member to the initial probability-rule pick and flipping once. Each new
refresh arm flips that production pick only if it backs the sole flagged team;
both teams flagged leaves the pick unchanged. Freeze the Tuesday opener spread.
Read (src/nfl_ats/nfl_week.py:20): the refresh deadline is the earlier of kickoff
and Sunday 16:00 Eastern. Require real revision timestamps strictly before it;
exclude and count proxy rows before canonicalization. Do not backdate archived
NFL.com fetch timestamps to their historical game weeks.

Report coverage by season before scores. Use latest visible practice observation
per player and Eastern calendar weekday; require that weekday to be Friday for
Friday constructs, Wednesday for the rest construct, within the game's own NFL
week. A final report with no earlier revision cannot measure a trajectory.
Unobserved constructs leave the production pick intact but are not evidence of
a zero biological effect. Keep coverage separate from an observed false flag.

Bootstrap paired game deltas by season/week blocks, 20,000 draws, seed 20260905,
95% percentile intervals. Report strict `probability_positive` and tie fraction;
an identically unchanged card has no directional evidence, regardless of its
strict probability of improvement. Reliability is Pearson correlation of odd-
and even-week team-season flag rates among covered observations; constant or
missing halves yield unavailable reliability, not measured zero reliability.

Positive control: use the actual Sunday inactive list as an oracle to identify
which flagged players were inactive; fade the sole team with those actual
absences, keeping the production comparator. This is explicitly an oracle,
never a pregame feature. Do not substitute zero snaps, final Out designations,
or realized ATS results for an actual inactive list. If that list is absent,
report the control unavailable and make no positive-control closure.

### LEAD-08 — Friday designation momentum

Construct: per-player Wednesday/Thursday latest practice versus Friday latest
practice, DNP=0, LP=1, FP=2. Score deterioration as +1, recovery as -1, static
as 0, sum across players with both observations. Flag a team when sum > 0;
FADE it. Coverage requires at least one genuinely paired early/Friday player.

### LEAD-09 — Questionable QB, Friday DNP

Construct: flag any QB whose latest Friday practice is DNP and latest visible
game designation is Questionable; BACK its opponent. Use position carried by
the real timestamped report (the QB availability schema already includes it).
Do not condition the playable flag on eventual participation. Report coverage
of Friday reports separately from flag incidence.

### LEAD-10 — Multi-illness team waves

Construct: count unique players whose latest Friday practice/reported injury
text includes illness, flu, or sickness; flag at least three and FADE that
team, home or away. Report all seasons, with no post-hoc flu-season restriction.
Join injury text only by the exact player/team/week/revision key, never from a
later or merely same-week report. A known empty reason is not a missing schema.

### LEAD-11 — Veteran Wednesday rest DNP

Construct: flag at least one player with Wednesday DNP, explicit rest text,
and chronological age >=30 at the game date, and FADE the team. Age requires
a birth date keyed by player ID; years of experience is not age. Missing birth
dates are counted and leave this construct unobserved. Birth dates must be
provided as static biographical data, never inferred from later performance.

## Binding verdict taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: (1) refuted mechanism - a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is unresolved_below_power: record it, report probability_positive, never the binary "contains zero". If a record command errors, the verdict is wrong, not the validator. Decisions are expected value: probability_positive above 0.5 favours playing it as a refresh overlay; state what each result implies for the DECISION before what is wrong with it. Never say a lead needs more games; the data is fixed and the project is model-limited.


## Measured results ? 2026-09-05

Inferred (decision): retain production for LEAD-09 and LEAD-10 given their
negative paired expected gains; LEAD-08 and LEAD-11 supply no measured direction
because their required constructs have zero covered team-games. These are four
unresolved records, not mechanism closures.

Measured (`scripts/injury_trajectory_leads_on_production.py`, results below):
source `artifacts/opener_evaluation/20260905T194919Z/per_game.parquet`, active
model `ab29832a4e099766`, 1,047 paired non-push games / 72 season-week blocks,
production 571/1,047 = 54.5368%. The four production overlays were unioned
against the original probability-rule picks before each injury refresh arm.
Measured (paired artifact audit): zero pushes and zero raw probability ties.

Measured (results.json coverage):

| Season | Raw regular-season rows | Real timestamps | Proxy rows excluded | Paired opener games | Friday-covered team-games |
|---|---:|---:|---:|---:|---:|
| 2022 | 5450 | 5450 | 0 | 248 | 394 |
| 2023 | 5451 | 5451 | 0 | 266 | 439 |
| 2024 | 5954 | 5954 | 0 | 266 | 436 |
| 2025 | 5783 | 0 | 5783 | 267 | 0 |

Measured (team_flags.parquet audit): 4,940 / 5,324 / 5,830 / 0 real rows
are visible on paired games in those seasons; one 2024 revision is excluded at
the decision deadline. Wednesday rest-DNP rows missing birth dates number
108 / 139 / 30 / 0. Read (`src/nfl_ats/age_curves.py:5`): the existing age axis
is career experience. Measured (weekly roster schema inspection): no birth-date
or chronological-age column is present, so no experience-to-age conversion was
used. Measured (NFL.com parquet/schema inspection): historical NFL.com rows have
August 2026 fetch timestamps and only one practice-status field, not a stored
Wednesday/Thursday/Friday revision sequence.

Measured (`results.json`, 20,000 week-blocked draws, seed 20260905):

| Lead | Covered team-games | Flagged team-games | Pick flips | Candidate accuracy | Delta accuracy points | 95% interval | probability_positive | Tie fraction | Odd/even reliability |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| LEAD-08 | 0 | 0 | 0 | 54.5368% | +0.000000 | [+0.000000, +0.000000] | 0.00000 | 1.00000 | unavailable |
| LEAD-09 | 1269 | 17 | 11 | 54.4413% | -0.095511 | [-0.575264, +0.384986] | 0.28265 | 0.14280 | -0.086044 |
| LEAD-10 | 1269 | 39 | 18 | 53.9637% | -0.573066 | [-1.350048, +0.191205] | 0.05510 | 0.03445 | 0.404453 |
| LEAD-11 | 0 | 0 | 0 | 54.5368% | +0.000000 | [+0.000000, +0.000000] | 0.00000 | 1.00000 | unavailable |

Measured (results.json): LEAD-08 and LEAD-11 have deterministic zero card deltas
and tie fraction 1. Their strict bootstrap probability_positive=0 is a property
of identical cards, not evidence favoring the opposite biological hypothesis.
Their registry entries omit uncertainty and directional probability so these
coverage audits cannot enter an inverse-variance pool as precise zero effects.
Measured (results.json): LEAD-09 and LEAD-10 reliability each uses 96 paired
team-seasons; a negative finite estimate alone does not establish zero trait
reliability. No terminal classification is made.

Measured (results.json), chronological season magnitudes:

| Lead | Season | Delta accuracy points | 95% interval | probability_positive | Flips |
|---|---:|---:|---|---:|---:|
| LEAD-09 | 2022 | +0.000000 | [-1.606426, +1.606426] | 0.40395 | 4 |
| LEAD-09 | 2023 | +0.000000 | [-1.107011, +1.132075] | 0.34620 | 4 |
| LEAD-09 | 2024 | -0.375940 | [-1.136364, +0.000000] | 0.00000 | 3 |
| LEAD-09 | 2025 | +0.000000 | [+0.000000, +0.000000] | 0.00000 | 0 |
| LEAD-10 | 2022 | +0.000000 | [-2.016129, +1.968504] | 0.41705 | 8 |
| LEAD-10 | 2023 | -1.503759 | [-3.053728, +0.000000] | 0.02150 | 6 |
| LEAD-10 | 2024 | -0.751880 | [-2.631579, +0.757576] | 0.13990 | 4 |
| LEAD-10 | 2025 | +0.000000 | [+0.000000, +0.000000] | 0.00000 | 0 |

Measured (`Test-Path data/players/inactives`): the local actual inactive-list
capture directory is absent. The required Sunday-inactive oracle is therefore
unavailable and no positive-control detection bound was established. Neither
zero snaps nor final Out labels were substituted. LEAD-08 cannot identify
trajectories from these snapshots; LEAD-11 cannot identify chronological age.
The builders and leakage tests implement both constructs for explicit supported
inputs, but the biological screens remain unmeasured. Friday in the available
nflverse source means the observation's Eastern weekday, not a separately
verified practice-session date; this limitation applies to LEAD-09 and LEAD-10.

Measured (four successful `nfl-ats weak-signals record` commands): recorded
`lead_08_injury_trajectory_refresh_cx14` through
`lead_11_injury_trajectory_refresh_cx14` under shared family
`injury_trajectory_refresh_cx14`, classification `unresolved_below_power`, units
`accuracy_points`. Read (`docs/rotation_registry.md:58`, rule 8): this is a
descriptive screen without an assigned confirmation window, so no rotation
confirmation was claimed or consumed. Measured (`rotation record-look --help`):
that subcommand does not exist; `rotation record` is the available confirmation
recorder. All four registry records retain the mined-season and overlap caveats.

Measured artifact bundle:
`artifacts/experiments/injury_trajectory_leads/20260905T203456Z/` contains
`results.json`, `paired_predictions.parquet`, and `team_flags.parquet`; the
provenance helper's experiment sidecar is inside the same allowed lane root.

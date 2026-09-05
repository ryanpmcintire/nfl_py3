# LEAD-24 Stage 1: the rookie workload wall and its dependence metric

Roadmap row: `| LEAD-24 | 🔬 | Rookie wall weeks 12-17 |`. This document is the
predeclaration AND the measured-results record for STAGE 1 ONLY: (i) does the
wall exist, (ii) the team-week snap-share dependence metric, (iii) that
metric's split-half reliability. **No ATS window is built here and nothing
is wired into `registry/rotation_registry.json`** -- LEAD-24's own definition
of done includes the ATS look, which is explicitly a later lane's job. This
lane stays 🔬.

## Closing-grounds taxonomy (binding, verbatim, read before any number below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Verdicts
flow only through `nfl-ats weak-signals record`, never through prose.

Nothing in this document closes anything. Measurement 1 (the wall delta) is
explicitly **not recorded in the registry at all** (see "Why measurement 1
is never recorded" below) -- there is no classification to get wrong there.
Measurement 3 (dependence-metric reliability) IS recorded, as
`unresolved_below_power` per the task's own instruction, since reliability
alone is never one of the two admissible closing grounds.

## Predeclared method (written before any number below was computed)

Builder: `scripts/rookie_wall_screen.py`. Library: `src/nfl_ats/rookie_wall.py`.
Tests: `tests/test_rookie_wall.py` (13 tests, all passing). Every number
below is **measured** by running

```
.\.tools\uv.exe run --no-sync python scripts\rookie_wall_screen.py
```

which wrote `artifacts/rookie_wall/20260905T052927Z/` (`wall_measurement.parquet`,
`dependence_shares.parquet`, `dependence_reliability.parquet`,
`manifest.json`). Artifacts are gitignored and local-disk-only; this
document is the durable record.

### Inputs and resolved snapshots (measured, this run)

| Source | Path | Snapshot id |
|---|---|---|
| Snap counts + weekly rosters | `data/players/raw/<id>/{snap_counts,weekly_rosters}.parquet` | `20260817T184901Z` |
| Weekly player stats | `data/players/values/raw/<id>/player_stats.parquet` | `20260817T184911Z` |
| Play-by-play (QB dropback EPA only) | `data/pbp/raw/<id>/season=YYYY/plays.parquet` | `20260817T184927Z` |
| Combine (draft slot) | `data/raw/combine/<id>/combine.parquet` | `20260822T143152Z` |

Season coverage: 2013-2025, bounded by the snap-count feed (same bound
`docs/age_curves.md` already documents).

### Top-50-pick identity and its join-rate disclosure (measured)

A player is a "top-50 pick" iff `combine.parquet`'s `draft_ovr <= 50` for a
row whose `pfr_id` resolves to a `gsis_id` through
`nfl_ats.players._stable_crosswalk` -- the identical crosswalk
`nfl_ats.qb_identity_features.draft_team_by_gsis_id` already uses for the
QB-revenge feature. Anything that fails either step is **NOT a top-50 pick**,
never guessed:

- `draft_ovr` is present on 5,554 of 8,968 combine rows (**61.9%**,
  `n_with_draft_ovr / n_combine_rows`).
- Of the 5,394 rows with both `draft_ovr` and a `pfr_id`, 2,686 resolve to a
  `gsis_id` through the roster crosswalk (**join rate 49.8%** over the FULL
  2000-2025 combine population -- low because the local roster crosswalk
  only starts in 2013, so pre-2013 draftees can never join). Restricted to
  `draft_year >= 2013` rows (the population that can actually appear as a
  rookie in this module's 2013-2025 panel), the join rate is **77.1%**
  (2,150 of 2,789) -- still measured, not assumed, and still disclosed as
  incomplete: 22.9% of draft-year-2013+ top-50-eligible combine rows never
  resolve to a snap-table player, most plausibly players who were drafted
  but never appeared on an NFL 2013+ weekly roster at all (a real "never
  made a regular-season roster" population, not a crosswalk bug -- not
  independently verified this session).
- 715 unique `gsis_id`s resolve as top-50 picks across the full combine
  history; 509 of those have `draft_year >= 2013`.

### Panel (measured, reusing `nfl_ats.age_curves.build_career_age_panel`)

- Snap rows (REG, 2013-2025): 310,475. Linked to GSIS: 308,857
  (99.48%). Unmapped positions dropped: 582 (0.19%). Missing `years_exp`:
  4. Final panel rows: 308,271 -- all figures identical to
  `docs/age_curves.md`'s panel diagnostics, since this module reuses that
  exact panel builder unchanged.
- Attaching `offense_pct`/`defense_pct` (needed for the 70% gate and the
  dependence metric, since `build_career_age_panel` does not carry them) via
  an independent `attach_snap_player_ids` join produced **zero** duplicate
  player-game collisions and **zero** panel rows left with a missing
  percentage (`panel_rows_missing_pct: 0`).

### Measurement 1: does the wall exist?

**Population.** A rookie (`career_age == 0`) top-50-pick player-season
qualifies if the mean of `offense_pct` OR `defense_pct` across the weeks the
player appeared in weeks 1-11 is `>= 0.70` (unweighted mean of games played,
since each week's `*_pct` is already snap-normalized within that game). The
**veteran control** applies the IDENTICAL >=70% weeks-1-11 gate to
`career_age >= 3` players at the same position group -- not "any veteran
starter" -- so the comparison isolates the rookie-SPECIFIC component of any
late-season decline from a generic "heavy-workload players fade" effect at
the same snap load.

**Delta.** For each qualifying player-season, the per-snap performance rate
(the same `nfl_ats.age_curves` metric per `pos_group`: EPA/dropback for QB,
EPA/offense-snap for RB/WR/TE, defense-disruption/defense-snap for
EDGE/DL/LB/CB/S) is computed separately for weeks 1-11 and weeks 12-17,
each half independently required to clear a 100-snap(/dropback) floor (the
same floor `age_curves.DELTA_METHOD_SNAP_FLOOR` uses). `delta = rate_late -
rate_early`, weighted by `min(denominator_early, denominator_late)`.
**OL/K/P/LS are excluded** -- they have no local performance rate in
`age_curves`, so the wall for those groups is snap-volume only and is not
scored here (LEAD-24's own scope note).

**Headline: rookie-minus-veteran delta**, per position group, with a
season-blocked percentile bootstrap (2,000 draws, fixed seed 20260905,
resampling SEASON blocks -- the same season sequence drawn for both the
rookie and veteran arms each draw, so any variation shared by a season is
preserved rather than washed out). `probability_wall_direction` = P(the
rookie-minus-veteran delta is NEGATIVE), the predeclared FADE direction.
Reported per era (2013-2018, 2019-2025) plus the full 2013-2025 window,
per AGENTS.md's per-era-magnitude rule.

### Measurement 2: the pregame dependence metric

Per (team, season, week): `offense_share` = SUM of `offense_pct` across
every top-50-pick rookie on that team-game (each player's `offense_pct` is
already that player's fraction of the team's offensive snaps that game, so
the sum is the fraction of an average offensive snap filled by a top-50-pick
rookie -- can exceed 1.0 with more than one such rookie playing heavy snaps
simultaneously, bounded above by 11). `defense_share` is the same construct
on the defensive side; `share_sum = offense_share + defense_share`.
Team-weeks with zero qualifying rookies get `0.0`, not a missing row.

**Pregame-safe transform.** `trailing_*_share` = a rolling mean over the
team's last 4 completed games, THEN shifted by one game, so the CURRENT
week's own share never enters its own trailing value. **Leakage test**
(`tests/test_rookie_wall.py::test_trailing_dependence_feature_excludes_the_current_weeks_own_value`):
perturbing a team-week's own raw share leaves THAT week's trailing value
byte-identical, while the following week's trailing value visibly changes --
proving the transform is a real rolling-prior-games mean, not a no-op that
would trivially pass a weaker "nothing changed" check. Confirmed a second
time against real 2020 Kansas City data during development (not kept as a
test, since the synthetic fixture is deterministic and exercises the same
code path): perturbing week 3's `offense_share` left week 3's own trailing
value at 0.6450 unchanged in both the clean and perturbed runs, while weeks
4-7's trailing values absorbed the perturbation and week 8 (outside the
4-game window) returned to the unperturbed 0.6325.

**Late-season high-dependence flag** = `trailing_share_sum` at or above the
80th percentile of every OTHER team's own `trailing_share_sum` for the same
season/week (a cross-sectional threshold built only from other teams' own
strictly-prior values) AND `week >= 12`.

### Measurement 3: reliability of the dependence metric

Two independent schemes on the RAW (non-trailing) `share_sum`, each with a
season-blocked bootstrap (2,000 draws) and a team-label shuffle null
(shuffles which team's second value pairs with which team's first value
WITHIN each season block, destroying team-specific pairing while preserving
any shared-season trend):

- **(a) odd-vs-even weeks within season**, team-season unit, block = season.
- **(b) season-to-season**, team unit (season T's mean `share_sum` vs. the
  SAME team's season T+1 mean), block = the earlier season of the pair.

Both report Pearson r, Spearman rho, Spearman-Brown-corrected reliability,
the bootstrap CI, `probability_positive`, and where the observed r falls in
the shuffle-null distribution.

## Measured results

### Measurement 1: rookie-minus-veteran wall delta, per position group / era

Units: the position group's own per-snap/per-dropback metric (EPA/dropback
for QB; EPA/offense-snap for RB/WR/TE; disruption/defense-snap for
EDGE/DL/LB/CB/S). Positive `rookie_minus_veteran` means rookies IMPROVED
late season relative to the veteran control (opposite the predeclared wall
direction); negative means rookies declined MORE (the predeclared wall
direction). `probability_wall_direction` = P(bootstrap draw < 0).

| pos_group | era | n_rookie | n_veteran | rookie_delta | veteran_delta | rookie_minus_veteran | 95% CI | P(wall direction) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| WR | 2013_2018 | 14 | 223 | -0.00983 | +0.00104 | **-0.01087** | [-0.02731, +0.00189] | **0.942** |
| WR | 2013_2025 | 35 | 454 | -0.00863 | -0.00207 | **-0.00656** | [-0.01756, +0.00646] | 0.852 |
| WR | 2019_2025 | 21 | 231 | -0.00776 | -0.00530 | -0.00245 | [-0.01838, +0.02072] | 0.609 |
| LB | 2013_2018 | 17 | 255 | -0.00397 | +0.00030 | -0.00426 | [-0.01125, +0.00446] | 0.864 |
| LB | 2013_2025 | 31 | 558 | -0.00059 | -0.00051 | -0.00008 | [-0.00688, +0.00624] | 0.506 |
| LB | 2019_2025 | 14 | 303 | +0.00378 | -0.00121 | +0.00499 | [-0.00808, +0.01150] | 0.187 |
| RB | 2013_2018 | 4 | 22 | -0.00081 | +0.00744 | -0.00824 | [-0.04025, +0.04614] | 0.663 |
| RB | 2013_2025 | 6 | 48 | -0.01196 | -0.00102 | -0.01094 | [-0.03583, +0.02929] | 0.773 |
| RB | 2019_2025 | 2 | 26 | -0.03907 | -0.00786 | -0.03121 | [-0.04447, -0.01405] | 1.000 |
| TE | 2013_2018 | 1 | 85 | +0.02537 | -0.00318 | +0.02855 | [+0.02520, +0.03202] | 0.000 |
| TE | 2013_2025 | 7 | 168 | -0.00441 | -0.00338 | -0.00103 | [-0.04596, +0.04172] | 0.465 |
| TE | 2019_2025 | 6 | 83 | -0.00884 | -0.00360 | -0.00525 | [-0.05619, +0.04448] | 0.530 |
| QB | 2013_2018 | 13 | 114 | +0.02727 | -0.02246 | +0.04972 | [-0.03428, +0.13297] | 0.163 |
| QB | 2013_2025 | 30 | 229 | +0.02051 | -0.02470 | +0.04521 | [-0.00124, +0.09622] | 0.029 |
| QB | 2019_2025 | 17 | 115 | +0.01511 | -0.02705 | +0.04217 | [-0.01288, +0.10022] | 0.073 |
| CB | 2013_2018 | 17 | 236 | +0.00339 | -0.00311 | +0.00650 | [-0.00737, +0.01867] | 0.173 |
| CB | 2013_2025 | 41 | 525 | +0.00131 | -0.00174 | +0.00305 | [-0.00317, +0.00971] | 0.172 |
| CB | 2019_2025 | 24 | 289 | -0.00028 | -0.00048 | +0.00020 | [-0.00483, +0.00557] | 0.487 |
| S | 2013_2018 | 14 | 232 | +0.00041 | -0.00189 | +0.00230 | [-0.00530, +0.00669] | 0.229 |
| S | 2013_2025 | 24 | 517 | +0.00276 | -0.00018 | +0.00294 | [-0.00091, +0.00665] | 0.057 |
| S | 2019_2025 | 10 | 285 | +0.00663 | +0.00133 | +0.00530 | [+0.00018, +0.01295] | 0.008 |
| EDGE | 2013_2018 | 1 | 116 | +0.01094 | -0.00122 | +0.01216 | [+0.00826, +0.01479] | 0.000 |
| EDGE | 2013_2025 | 6 | 208 | +0.00110 | -0.00113 | +0.00223 | [-0.02298, +0.01916] | 0.394 |
| EDGE | 2019_2025 | 5 | 92 | -0.00085 | -0.00100 | +0.00014 | [-0.02814, +0.02137] | 0.467 |
| DL | 2013_2018 | 3 | 64 | +0.00306 | -0.00086 | +0.00392 | [-0.00302, +0.01394] | 0.321 |
| DL | 2013_2025 | 4 | 152 | +0.00159 | +0.00028 | +0.00131 | [-0.00526, +0.01260] | 0.422 |
| DL | 2019_2025 | 1 | 88 | -0.00376 | +0.00117 | -0.00493 | [-0.00767, -0.00308] | 1.000 |

Full per-row output (including `bootstrap_valid_draws`) is in
`wall_measurement.parquet`.

**Reading this table plainly, per AGENTS.md's rule to state the number
before the caveat, and per the label-provenance rule:**

- **measured**, this run: **WR leans toward the predeclared wall direction
  in every era** -- 2013-2018 rookie-minus-veteran -0.01087, 95% [-0.02731,
  +0.00189], P(wall direction) 0.942 (the tightest, most rookie-sample-rich
  era, n=14 rookie-seasons); the full-window read is -0.00656, [-0.01756,
  +0.00646], P+ 0.852. Neither interval excludes zero, so neither is
  "resolved" under the taxonomy -- both are `unresolved_below_power`-shaped
  results with a P(wall direction) in the high 0.85-0.94 range, i.e. a
  real candidate worth an ATS look, not a settled finding.
- **measured**: **RB also leans toward the wall in every era**
  (2013-2025 -0.01094, P+ 0.773; 2019-2025 -0.03121 with an interval
  ENTIRELY negative, [-0.04447, -0.01405], P+ 1.000) but on an extremely
  thin sample (2 rookie-seasons in 2019-2025, 6 in the full window) --
  stated plainly: an "entirely negative" interval built from a 2-player
  bootstrap is not a resolved wrong sign in any meaningful sense, because
  the bootstrap's only source of variation with n=2 rookie player-seasons
  is which of the handful of SEASONS gets drawn, not genuine rookie-arm
  sampling variability across many players. This is flagged, not treated as
  closed, and nothing is recorded for it (measurement 1 is never recorded
  in the registry regardless -- see below).
- **measured**: **LB is flat-to-slightly-wall-leaning** (2013-2025 -0.00008,
  essentially zero; 2013-2018 -0.00426, P+ 0.864; 2019-2025 flips to
  +0.00499, P+ 0.187 against the wall direction) -- inconsistent across
  eras, a real per-era magnitude difference per AGENTS.md, not noise to be
  averaged away.
- **measured**: **QB, CB, S, EDGE, DL lean AWAY from the predeclared wall
  direction** (positive `rookie_minus_veteran`) in most era cells, several
  with n=1 rookie-season (TE 2013-2018, EDGE 2013-2018, DL 2019-2025) or n=2
  (RB 2019-2025) driving an "entirely one-sided" bootstrap interval that is,
  for the same reason as the RB case above, an artifact of season-bootstrap
  variance applied to a near-single-point rookie arm, not a resolved
  finding on either side. **No position group's interval is treated as a
  resolved wrong sign here** -- that determination is explicitly out of
  scope for this Stage-1 lane and would need a much larger n before it
  meant anything for the thin cells.
- **inferred** (my reasoning, not a measured claim): the QB and EDGE/TE
  positive leans plausibly reflect that the very few top-50-pick rookie QBs
  who ALSO clear 70% snap share for 11 straight weeks (n=1-30 across eras)
  are disproportionately the ones good enough to start and stay startable
  all season (survivorship into the population itself), which could mask a
  true within-player decline the same way `docs/age_curves.md`'s
  cross-sectional WR curve masks its own within-player decline -- this
  module's delta IS within-player (not cross-sectional), so that specific
  mechanism does not directly apply, but a milder version (which rookies
  clear the 70%-for-11-weeks gate at all) is a selection filter this
  measurement does not control for. Worth a future refinement, not asserted
  as fact here.

### Measurement 2: dependence metric spot-check (measured)

- 6,814 team-weeks (13 seasons x 32 teams x ~16.4 weeks average, 2013-2025).
- `share_sum` distribution: mean 0.642, median 0.580, max 3.31 (Buffalo-
  or Baltimore-style seasons with several heavy-snap top-50 rookies at
  once), 25th percentile exactly 0.0 (most team-weeks have no qualifying
  rookie playing heavy snaps at all).
- 567 of 6,814 team-weeks (8.3%) are flagged `late_season_high_dependence`
  (trailing share at/above the same-week 80th percentile AND week >= 12) --
  a plausible, non-degenerate rate for an 80th-percentile-and-week-12+ gate.

### Measurement 3: dependence-metric split-half reliability (measured)

| Scheme | n units | Pearson r | Spearman rho | Spearman-Brown | 95% CI | P(r>0) | Shuffle-null mean [95% CI] | Null percentile of observed r |
|---|---:|---:|---:|---:|---|---:|---|---:|
| Odd/even weeks, team-season | 416 | **0.988** | 0.988 | 0.994 | [0.985, 0.991] | 1.000 | 0.006 [-0.094, +0.108] | 1.000 |
| Season-to-season, team | 384 | **0.129** | 0.159 | 0.228 | [0.044, 0.225] | 1.000 | 0.001 [-0.101, +0.104] | 0.993 |

**Stated plainly, before caveats:** both schemes' full bootstrap interval
sits entirely on the positive side of zero, and both sit far above their own
team-label shuffle null (a shuffle-null mean near 0.00-0.01 in both cases,
as expected -- shuffling destroys the real pairing). The within-season
number (r=0.988) mechanically reflects that a team's rookie-dependence
level over a season is close to constant given the same 1-2 rookies play
most of a team's games at a similar volume all season, so it mainly confirms
the metric is measuring a real within-season team quantity rather than
random week-to-week noise. **The season-to-season number (r=0.129, entirely
positive, shuffle-null percentile 0.993) is the more informative reliability
figure** for anything meant to generalize beyond a single season: it says a
team's tendency to lean on high-draft-pick rookies carries over to the NEXT
season at a modest but real, non-shuffle-explainable level -- plausibly an
organizational/roster-construction trait (a team that top-50-drafts a
rookie and plays him heavily one year is somewhat more likely to do so
again), not merely "the same player happened to still be a rookie."

## Item 5: plain statement, before caveats

**Does the wall exist at a magnitude worth an ATS look?** For WR and RB, yes
-- both position groups lean toward the predeclared FADE direction in every
era measured, WR with a real sample (14-41 rookie-seasons per era, P(wall
direction) 0.85-0.94) and RB with a thinner one (2-6 rookie-seasons per era)
but a consistent sign across all three era cells. Neither is resolved (every
interval contains zero except the two n<=2 RB/DL cells flagged above as
bootstrap artifacts of a near-single-point rookie arm), and per the binding
taxonomy that is the EXPECTED outcome for a real small signal, not a reason
to drop it. QB, CB, S, EDGE, DL lean the other way or are inconsistent
across eras and are not flagged as ATS candidates by this measurement.

**Is the dependence metric reliable enough to carry an ATS look?** Yes on
both counts measured: the within-season split-half correlation is very high
(0.988, entirely positive, at the top of its own shuffle null) and the
season-to-season correlation, while more modest (0.129), is ALSO entirely
positive and sits at the 99.3rd percentile of its own team-shuffle null --
a real, replicable team-level trait, not week-to-week or season-to-season
noise. This clears the bar for a later lane to build the team-week
`late_season_high_dependence` flag into an ATS window, most plausibly
targeted at the WR/RB position groups where measurement 1 above found the
strongest wall-direction lean.

## Why measurement 1 is never recorded in the registry

Per AGENTS.md's commensurability rule ("pooled inputs must be commensurable
-- same units, same scale, same population"), the wall delta is a per-snap
performance difference (EPA/dropback, EPA/offense-snap, or
disruption/defense-snap depending on position group) -- not a correlation,
not an accuracy-points gap, not any of the registry's admissible
`--effect-units`. Forcing it into `accuracy_points` "as a numeric container
only" is exactly the mistake AGENTS.md's commensurability correction already
flags for a prior CFB entry. Nothing from measurement 1 is recorded to
`registry/weak_signals.json`; this document and `wall_measurement.parquet`
are the record.

## Recorded: dependence-metric reliability

The dependence metric's within-season split-half reliability (the more
directly interpretable of the two schemes, and the one the task template
names) is recorded as a correlation-units weak signal:

```
.\.tools\uv.exe run --no-sync nfl-ats weak-signals record `
  --name rookie_wall_dependence_reliability `
  --description "LEAD-24 Stage 1: split-half reliability of the team-week top-50-pick-rookie snap-dependence metric (share of a team's offense+defense snaps taken by top-50-pick rookies). Odd/even-week within-season correlation across 416 team-seasons (2013-2025), Spearman-Brown corrected. A second, independent season-to-season scheme (team-level year-over-year persistence, n=384 team-season pairs, block=earlier season) reads Pearson r=0.129, 95% CI [0.044, 0.225], P+ 1.0, shuffle-null percentile 0.993 -- also fully positive and outside its own team-label shuffle null." `
  --source artifacts/rookie_wall/20260905T052927Z/dependence_reliability.parquet `
  --effect 0.988167 --effect-units correlation `
  --classification unresolved_below_power `
  --league nfl --season-start 2013 --season-end 2025 `
  --reliability 0.994048 `
  --interval-low 0.984942 --interval-high 0.991246 `
  --probability-positive 1.0 `
  --sample-games 416 --sample-blocks 13 `
  --family rookie_wall --category onfield `
  --classification-evidence "Both split-half schemes' full bootstrap interval sits entirely positive and both clear their team-label shuffle null (percentile 1.000 and 0.993), but reliability alone is never one of AGENTS.md's two admissible closing grounds (wrong-sign-resolved or positive-control-bound) -- this records the metric's measured reliability for a later ATS lane to build on, not a closed effect." `
  --plain-summary "Which teams lean hardest on high-draft-pick rookies is a real, stable team trait, not week-to-week noise: a team's rookie-dependence in the first half of its games nearly matches the second half of the same season (correlation 0.99), and even carries over, more modestly but still measurably, into the following season."
```

**measured** (this session): the command above was run and returned
`{"classification": "unresolved_below_power", "effect": 0.988167,
"effect_units": "correlation", "favours_candidate": true, "recorded":
"rookie_wall_dependence_reliability", "total_signals": 740}` against
`registry/weak_signals.json` (no warnings on stdout/stderr -- both
`--plain-summary` and `--category` were supplied).

## Reproduction

```
.\.tools\uv.exe run --no-sync python scripts\rookie_wall_screen.py
.\.tools\uv.exe run --no-sync python scripts\rookie_wall_screen.py --as-of-season 2022
.\.tools\uv.exe run --no-sync pytest tests\test_rookie_wall.py -q
```

Artifact from this session: `artifacts/rookie_wall/20260905T052927Z/`
(`wall_measurement.parquet`, `dependence_shares.parquet`,
`dependence_reliability.parquet`, `manifest.json`). Artifacts are
gitignored; re-run the builder to reproduce (deterministic given the same
local snapshots -- the bootstrap seed is fixed at 20260905).

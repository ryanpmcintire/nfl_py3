# PBP-06: special-teams battery predeclaration

Written 2026-08-19 for ROADMAP.md `PBP-06` ("Special teams -- Kicking,
punting, returns, field position above expectation"), previously unbuilt.

**This document is written and frozen BEFORE any cell in it has been
screened against a cover outcome.** The reliability audit below (which
dimensions persist year-over-year, and therefore which are eligible to enter
a cell) WAS run first and is allowed to shape which dimensions enter a cell
-- that is the one predeclared, admissible exception, identical to the
PBP-08 team-style precedent (`docs/team_style.md` "Reliability gate") -- but
no cover-rate sign has been looked at for any cell defined here.

## Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Every cell
below is recorded to the registry regardless of sign or interval shape.

## Why special teams is a live lane, not a team-quality proxy in disguise

`PBP-05`'s measured ceiling (ROADMAP.md, `docs/play_level_audit.md`):
features that only estimate team QUALITY better are bounded near zero,
because the market already prices quality well from primary offense/defense
efficiency signals. Special-teams units are a plausible exception for two
reasons: (1) they contribute a comparatively small share of a team's overall
EPA/points, so a market model built primarily on offense/defense efficiency
has less reason to price a specialist's marginal skill precisely; (2) the
four traits built here (kicker accuracy, punter net yards, return yards,
and their composite) are specifically SKILL/TENDENCY measures of a
specialist unit, not restatements of how good the team's offense or defense
is. The caveat, stated plainly rather than hidden: a team's kicking/punting/
return units are not fully independent of overall team quality either (a
well-run organization tends to be good at everything), so this battery
should be read as "underweighted specialist skill," not as a construct
proven orthogonal to quality the way PBP-08's play-calling-tendency
dimensions were.

## Data source and pregame-safety argument

`nfl_ats.pbp`'s `PBP_SNAPSHOT_COLUMNS` (read, `src/nfl_ats/pbp.py`)
intentionally narrows the stored local PBP snapshot; it has no
`kick_distance`, `field_goal_result`, `return_yards`, `return_team`,
`touchback`, or `punt_blocked` -- confirmed by reading the constant, no
special-teams column survives the trim. Per the referee-battery precedent
(`docs/referee_battery.md` "Data source"), this battery fetches fresh
play-by-play directly via `nflreadpy.load_pbp(seasons=[season])`, ONE SEASON
AT A TIME, and aggregates each season down to small per-team-game and
per-team-season tables IMMEDIATELY -- the raw ~740k-row 2009-2025 PBP is
NEVER concatenated across seasons and NEVER written to disk (stricter than
the PBP-08 team-style precedent, which does cache a narrowed raw parquet;
this module follows the referee-battery convention of "the raw PBP itself is
NOT persisted, only this small derived aggregate" instead). MEASURED this
session: 781,712 REG-season rows processed across 17 seasons (2009-2025),
none persisted; outputs written to
`data/raw/special_teams/<UTC timestamp>/{team_game,team_season}.parquet`
(gitignored, `data/raw/**`).

Every trait is a TRAILING PRIOR-SEASON value at screening time (see
"Method" below) -- nothing here reads a game's own outcome, the same
pregame-safety argument every other battery in this repo relies on.

## Trait definitions

Built by `scripts/special_teams_features.py`. Every raw dimension is pooled
DIRECTLY from the underlying play-level attempts at each level (team-game,
team-season), never averaged from a lower granularity (Simpson's-paradox
precaution, the PER-07/PBP-08 convention), then centered against its OWN
SEASON's unweighted team mean (era/rule-change-drift removal -- special
teams saw two significant rule changes inside this window: the 2023
fair-catch-anywhere-inside-25 rule and the 2024 "dynamic kickoff" format,
both of which visibly moved the league-wide raw kickoff-return rate in this
data, MEASURED: mean `n_kickoff_returns`/team fell from 32.4 in 2022 to
18.3 in 2023 then to 28.7 in 2024; centering absorbs the level shift, not
the underlying persistence, exactly PER-07's own justification for the same
convention). `TEAM_ABBREVIATION_ALIASES` (OAK->LV, SD->LAC, STL->LA) is
applied to every team-identifying column so specialist continuity survives
relocations -- MEASURED: 544 team-season rows = exactly 32 teams x 17
seasons, confirming no relocation-era duplication.

| dimension | definition |
|---|---|
| `fg_oe` | field-goal makes minus a season-and-distance-bucket (<30, 30-39, 40-49, 50+) expected make rate, averaged per attempt. The bucket rate is computed from THAT SEASON's own league-wide attempts -- self-referential in the same documented sense nflverse's own `pass_oe` already is (a team's own attempts contribute a small amount to the season baseline it is compared against; with 32 teams the dilution is minor). Blocked kicks count as misses (matches the official statistic). |
| `punt_net_yards` | standard net-punting formula: `kick_distance - return_yards` for a normal punt, capped at `yardline_100 - 20` on a touchback (the ball is only ever "worth" up to the receiving team's 20). MEASURED and hand-verified against a real 2024 row: yardline_100=56, kick_distance=56, touchback=1 -> net=36, matching the standard net-punting convention (an uncapped formula would have wrongly credited 56). |
| `punt_return_yards` | mean `return_yards` on punts with a genuine return (excludes fair catch, touchback, blocked, downed, out-of-bounds), grouped by the RETURNING team. |
| `kickoff_return_yards` | mirror of the above for kickoffs (excludes touchback, fair catch, out-of-bounds, downed). |
| `block_rate` | share of this team's own FG+punt attempts that were blocked (protection-unit trait). MEASURED to have the weakest reliability of the five (see below); reported, and explicitly SCOPED OUT of the predeclared cells below -- see "Scope decision on block_rate." |

Composite dimensions (built in `scripts/special_teams_screen.py`, at
screening time, from the reliable raw dimensions only): each raw centered
dimension is z-scored (divided by its pooled standard deviation across the
544-row 2009-2025 team-season panel, the identical convention `team_style`'s
`add_identity_distance` uses). `return_composite_z` = mean of
(`punt_return_yards_z`, `kickoff_return_yards_z`) -- the "returns" trait
named in the ROADMAP row. `special_teams_composite_edge_z` = mean of all
four z-scored raw dimensions (`fg_oe_z`, `punt_net_yards_z`,
`punt_return_yards_z`, `kickoff_return_yards_z`) -- the "field position
above expectation" trait named in the ROADMAP row: a team that is
simultaneously good at hidden points (kicking), field-position denial
(punting), and field-position generation (returns) is, by construction,
gaining field position above what the market's generic team-quality signals
would predict.

**Trailing basis**: prior FULL REG season (not trailing-N-games), matching
PER-07/PBP-08's convention. A team's first tracked season (2009) has no
prior season and is excluded (missing, not defaulted) from every cell.

## Reliability audit (MEASURED, run first, before any cell existed)

`scripts/special_teams_reliability.py`,
`artifacts/special_teams_reliability/20260819T232538Z/results.json`. Two
independent estimates per dimension, both on the CENTERED value:
**year-over-year** (primary; Pearson r between a team's centered value in
season t and t+1, pooled across all same-franchise consecutive-season pairs
2009-2025, block-bootstrapped 95% CI over pairs, 20,000 samples) and
**within-season split-half** (secondary cross-check; odd/even-week
team-season split, Spearman-Brown corrected, via
`nfl_ats.cfb_qb_dependence.split_half_reliability` reused directly).

| dimension | YoY r | YoY 95% CI | YoY P+ | YoY SB step-up | split-half SB | n pairs |
|---|---|---|---|---|---|---|
| `fg_oe` | +0.065 | [-0.022, +0.153] | 0.931 | +0.123 | +0.063 | 512 |
| `punt_net_yards` | +0.313 | [+0.233, +0.391] | 1.000 | +0.477 | +0.424 | 512 |
| `punt_return_yards` | +0.109 | [+0.019, +0.196] | 0.992 | +0.196 | +0.163 | 512 |
| `kickoff_return_yards` | +0.158 | [+0.073, +0.243] | 1.000 | +0.272 | +0.210 | 508 |
| `block_rate` | -0.024 | [-0.105, +0.060] | 0.277 | -0.049 | +0.109 | 512 |

**Zero dimensions are excluded on the admissible reliability ground** (a
dimension whose YoY 95% CI upper bound sits at or below zero is not a
persistent trait): every dimension's CI upper bound sits above zero,
including `block_rate`'s (+0.060). Per the PBP-08 precedent's own reading of
this same result shape, this is a genuine, positive finding in its own
right for the four kept dimensions: `punt_net_yards` is a moderately
reliable trait (+0.313, comparable to PER-07's own +0.320 "genuinely
reliable" bar), `kickoff_return_yards` and `punt_return_yards` are weaker
but measurably persistent, and `fg_oe` is the weakest of the four kept
dimensions -- a kicker-accuracy-over-distance-expectation signal that
persists only marginally year over year (consistent with the well-known
result in kicking literature that FG accuracy is dominated by
distance/weather/pressure noise more than by any one kicker's stable skill,
though this data cannot separate a kicker's own skill from year-to-year
kicker TURNOVER on the same team -- a limitation stated, not hidden).

### Scope decision on `block_rate` (a priori, before any cover-rate sign)

`block_rate`'s YoY interval, [-0.105, +0.060], straddles zero almost
symmetrically around a NEGATIVE point estimate (-0.024) -- this is about as
close to "indistinguishable from a random redraw each season" as a measured
result gets without crossing the one admissible exclusion bar (upper bound
at or below zero). Per AGENTS.md, this is NOT grounds to classify it as
refuted -- the interval is not entirely below zero, so `wrong_sign_resolved`
does not apply, and it clears the >0 reliability-exclusion bar on a
technicality. This is a SCOPE decision, not a taxonomy closure: no predeclared
screening cell is built for `block_rate` and it does not enter either
composite, because folding a dimension this close to pure noise into a
composite would only dilute the other three kept dimensions' real
(measured, positive) persistence. `block_rate` remains reported here,
MEASURED, for the record -- a future revisit with a larger sample (e.g.
pooling across a rolling multi-season window, since single-season blocked-
kick counts are typically 0-1 per team) could plausibly sharpen this, but
that is out of scope for this predeclaration.

## Predeclared cells (8: 4 traits x top/bottom quartile)

**Population**: NFL REG, close grade (`schedules.parquet` `spread_line`,
the same convention `scripts/nfl_weather_battery_screen.py` /
`scripts/team_style_screen.py` use), full local history 2009-2025. 2009
games are excluded per cell for missing prior-season special-teams data
(reported, not silently dropped from the declared range).

**Design**: team-perspective long table (one row per team per game,
`team_covered`), the exact pattern `scripts/team_style_screen.py::build_long_table`
/ `scripts/nfl_bias_battery_screen.py::build_long_table` use -- chosen over
the referee-battery precedent's home-only one-sided design because a
special-teams trait belongs to whichever of the two competing teams has it
(home or away), unlike a referee assignment which is an external factor to
both teams; a home-only design would silently discard every away-side
appearance of a flagged team, halving the usable population for no
methodological reason.

**Method**: joint week-blocked bootstrap (block = season*100+week) PRIMARY,
season-blocked bootstrap SECONDARY, both `block_bootstrap_two_group`-
identical to `scripts/nfl_weather_battery_screen.py` /
`scripts/team_style_screen.py` (read both; same algorithm, vectorized
multinomial block resample, jointly resampling both arms from the same
drawn blocks). Full-slate effect scaling via
`nfl_ats.experiment_runner.scale_subset_effect` (imported directly, not
reimplemented) -- `sign * raw_gap_fraction * 100 * fraction_of_slate`,
`accuracy_points` units. 20,000 bootstrap samples, seed 20260819 (today's
date, the repo's own convention), fixed and deterministic.

Quartile thresholds are the empirical top-quartile (0.75 quantile) /
bottom-quartile (0.25 quantile) of each dimension across the full 544-row
2009-2025 team-season panel, computed ONCE and reused identically by
`scripts/special_teams_screen.py` -- not re-derived per season, matching
the PBP-08 precedent's own anti-drift convention.

**Mechanism (shared across all 8 cells)**: a team's special-teams unit
contributes hidden points (kicking), field-position denial (punting), and
field-position generation (returns) that a market model built primarily on
offense/defense efficiency plausibly underweights relative to its actual
marginal value. Top-quartile teams by a given trait are predicted to
outperform their spread (POSITIVE `team_covered`); bottom-quartile teams are
predicted to underperform it (NEGATIVE `team_covered`) -- the mirror-sign
pairing convention `docs/referee_battery.md` cells 1/2 and 5/6 use.

1. **`special_teams_fg_kicker_top_quartile`** -- top-quartile teams by
   prior-season `fg_oe` vs the field. Sign: **+1**. Reliability: YoY
   +0.065, 95% CI [-0.022, +0.153] (the weakest of the four kept
   dimensions -- read this cell's result as the most exploratory of the
   eight).
2. **`special_teams_fg_kicker_bottom_quartile`** -- same trait, bottom
   quartile. Sign: **-1**. Same reliability as cell 1.
3. **`special_teams_punt_net_top_quartile`** -- top-quartile teams by
   prior-season `punt_net_yards` vs the field. Sign: **+1**. Reliability:
   YoY +0.313, 95% CI [+0.233, +0.391] (the strongest of the four kept
   dimensions).
4. **`special_teams_punt_net_bottom_quartile`** -- same trait, bottom
   quartile. Sign: **-1**. Same reliability as cell 3.
5. **`special_teams_return_top_quartile`** -- top-quartile teams by
   prior-season `return_composite_z` (mean of z-scored `punt_return_yards`
   and `kickoff_return_yards`) vs the field. Sign: **+1**. Reliability:
   componentwise YoY +0.109 [+0.019,+0.196] (punt) and +0.158
   [+0.073,+0.243] (kickoff); the composite's own reliability is not
   separately measured (an average of two positively-reliable inputs is not
   guaranteed to inherit either input's exact r, but cannot be LESS
   reliable than pure noise given both inputs individually clear the
   admissible bar).
6. **`special_teams_return_bottom_quartile`** -- same trait, bottom
   quartile. Sign: **-1**. Same reliability as cell 5.
7. **`special_teams_composite_edge_top_quartile`** -- top-quartile teams by
   prior-season `special_teams_composite_edge_z` (mean of all four
   z-scored raw dimensions) vs the field -- the "field position above
   expectation" cell named in the ROADMAP row. Sign: **+1**. Reliability:
   composite of all four kept dimensions (+0.065 to +0.313); reported as
   the weakest input's figure per the PBP-08 team-style precedent's own
   "weakest-link" convention for a composite cell's recorded reliability
   figure.
8. **`special_teams_composite_edge_bottom_quartile`** -- same trait, bottom
   quartile. Sign: **-1**. Same reliability as cell 7.

**Scope note**: a ninth cell interacting kicker accuracy with game-time
wind (mirroring `docs/team_style.md`'s B3 mechanism screen and the weather
battery's own `high_wind_outdoor` threshold) was considered -- explicitly
named as a candidate in the task brief ("kicker cold/wind robustness if
joinable to schedules weather actuals") -- but is deliberately deferred to
keep this predeclaration inside the task's stated ~4-8 cell range and avoid
uncorrected-multiplicity creep beyond the two precedent batteries' own
5-6-cell scope. Not built here; a future addition, not a finding.

**Every cell is recorded to `registry/weak_signals.json` after screening,
regardless of sign or interval shape**, per AGENTS.md; a sign prediction
above is a predeclaration of what would count as the mechanism working, not
a promise about what the data will show.

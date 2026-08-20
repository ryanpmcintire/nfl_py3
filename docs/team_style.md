# PBP-08: team "personality" / style predeclaration

Written 2026-08-19, in response to the owner's question: "I wonder if we can
model team 'personalities' -- like the Patriots under Belichick had a love
for the short game, screen plays and what not. Does that sort of thing offer
signal not in the market spread? And have we tried the experiment yet?" The
answer to "have we tried it" was **no** -- ROADMAP.md `PBP-08` (scheme/
matchup interactions) had never been built. This is the first version.

**This document is written and frozen BEFORE any cell in it has been
screened against a cover outcome.** The reliability audit below (which
dimensions persist year-over-year) WAS run first and is allowed to shape
which dimensions enter a cell -- that is the one predeclared, admissible
exception (see "Reliability gate" below) -- but no cover-rate sign has been
looked at for any cell defined here.

## Binding taxonomy (owned verbatim, per AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two things
justify closing a line of work: (1) a refuted mechanism -- a RESOLVED wrong
sign (the whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". If a record command errors, the verdict is wrong,
not the validator. Forced-pick pool: EV decides, P+ > 0.5 favours. Every
cell below is recorded to the registry regardless of its sign or interval
shape.

## Why style can carry signal the spread doesn't already have

`PBP-05` and the team-quality ceiling (`docs/play_level_audit.md`,
ROADMAP.md) measured that features which only estimate team QUALITY better
are bounded near zero, because the market already prices quality well. Style
is a live lane only where it is NOT a quality proxy in disguise. Every
dimension built here is a play-calling TENDENCY -- how a team chooses to
play, not how well it executes -- so none of them are effect-per-play
quality metrics (no EPA, no success rate, no points). The one place quality
creeps in is the defensive proxy for cell B1 (below): raw sack/pressure rate
is explicitly avoided because it is quality-laden (a genuinely better pass
rush produces both more sacks and more shotgun response from the offense,
confounding style with quality); instead the proxy is opposing offenses'
OWN formation response (`shotgun_rate_faced`), which reflects how offenses
treat this defense's style, not a direct measure of its effectiveness.

`PER-07` is the one personality trait already measured -- 4th-down
aggressiveness relative to league norm has year-over-year reliability
+0.320 and is a pooling input (RWB-18). It is the precedent that coach/team
tendencies are stable, persistent traits worth measuring this way, and its
"express relative to season league norm" convention (to strip well-
documented league-wide drift from team identity) is used verbatim below.

## Data and method

**Full nflverse PBP, not the in-repo trimmed snapshot.** `nfl_ats.pbp`
(read, `src/nfl_ats/pbp.py`) intentionally narrows the STORED play-by-play
snapshot to `PBP_SNAPSHOT_COLUMNS`, which does not include `air_yards`,
`pass_length`, `shotgun`, `no_huddle`, `run_location`, or `run_gap` -- there
is nothing to model style with in the existing snapshot contract, and no
untrimmed snapshot exists in this local clone. `scripts/team_style_features.py`
fetches nflreadpy's full-column PBP directly (one network call per season,
cached locally under the gitignored `data/pbp/team_style/`), reusing
`nfl_ats.pbp.analysis_plays`/`build_drive_table` (the same play-eligibility
filter -- real scrimmage plays, WP 0.05-0.95 "competitive" filter, kneels/
spikes/aborted plays excluded -- already used by the production PBP
pipeline) and `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` for franchise
continuity (OAK->LV, SD->LAC, STL->LA). **Measured**: the independent fetch
totals 781,712 REG-season plays across 2009-2025, exactly matching
`data/processed/game_features_pbp.manifest.json`'s `pbp_rows` figure for the
production pipeline's own snapshot -- cross-validating that this is the same
underlying nflverse data, just with more columns kept.

Nine style dimensions, each a TRAILING, prior-season, pregame-safe team
value, computed by pooling the team's own season's competitive-play-filtered
plays directly (not averaging per-game rates, to avoid Simpson's-paradox
bias from uneven per-game play counts):

| dimension | definition |
|---|---|
| `short_pass_share` | share of pass attempts with `air_yards <= 5` |
| `deep_share` | share of pass attempts with `air_yards >= 20` |
| `avg_air_yards` | mean `air_yards` among pass attempts |
| `screen_rate` | share of pass attempts that are COMPLETIONS with `air_yards <= 0` |
| `proe` | mean nflverse `pass_oe` (pass rate over its own down/distance/score/time-adjusted expectation -- adjusted AND cheap, since it ships with the play-by-play) over offensive snaps |
| `shotgun_rate` | share of offensive snaps in shotgun |
| `no_huddle_rate` | share of offensive snaps no-huddle |
| `run_direction_hhi` | Herfindahl index (sum of squared shares) of called-run `run_location` in {left, middle, right}; excludes scrambles |
| `seconds_per_play_pace` | season sum(drive time-of-possession) / sum(drive play count), from `build_drive_table` |

Every dimension is **centered against its own season's unweighted team
mean** (era-drift-proof, the exact PER-07 convention). This is not a
theoretical concern: measured 2009-vs-2025 league means show `shotgun_rate`
moving 34.9%->65.9% and `no_huddle_rate` 4.7%->9.0% -- raw values would read
league-wide drift as team identity.

**Trailing basis**: prior FULL REG season (not trailing-N-games), matching
PER-07's convention exactly and directly supporting the year-over-year
reliability measurement below, whose unit IS the team-season. A team's
first tracked season (2009) has no prior season and is excluded (missing,
not defaulted) from every cell that depends on trailing style.

## Reliability gate (measured, run first, before any cell existed)

`scripts/team_style_reliability.py`, `artifacts/team_style_reliability/20260819T205232Z/results.json`.
Two independent estimates per dimension, both on the CENTERED value:
**year-over-year** (primary; Pearson r between a team's centered value in
season t and t+1, pooled across all 512 same-franchise consecutive-season
pairs 2009-2025, 32 teams x 16 gaps, block-bootstrapped 95% CI over pairs)
and **within-season split-half** (secondary cross-check; odd/even-week
team-season split, Spearman-Brown corrected, via
`nfl_ats.cfb_qb_dependence.split_half_reliability` reused directly -- the
same function PBP-05's 0.80/0.46 figure and the CFB role-continuity
0.719/0.680 figures were built on).

| dimension | YoY r | YoY 95% CI | YoY SB step-up | split-half SB | n pairs |
|---|---|---|---|---|---|
| short_pass_share | +0.408 | [+0.324, +0.486] | +0.579 | +0.750 | 512 |
| deep_share | +0.306 | [+0.226, +0.382] | +0.468 | +0.549 | 512 |
| avg_air_yards | +0.358 | [+0.276, +0.439] | +0.528 | +0.675 | 512 |
| screen_rate | +0.366 | [+0.283, +0.445] | +0.536 | +0.669 | 512 |
| proe | +0.434 | [+0.351, +0.514] | +0.606 | +0.854 | 512 |
| shotgun_rate | +0.634 | [+0.563, +0.698] | +0.776 | +0.956 | 512 |
| no_huddle_rate | +0.557 | [+0.402, +0.690] | +0.715 | +0.952 | 512 |
| run_direction_hhi | +0.653 | [+0.587, +0.714] | +0.790 | +0.539 | 512 |
| seconds_per_play_pace | +0.489 | [+0.405, +0.567] | +0.657 | +0.733 | 512 |
| `shotgun_rate_faced` (defensive proxy, B1 only) | +0.278 | [+0.200, +0.353] | -- | -- | 512 |

**All ten dimensions clear PER-07's own "genuinely reliable" bar (+0.320)
except `deep_share` (+0.306) and the defensive proxy (+0.278), both of which
still have a 95% CI entirely and comfortably above zero.** Team play-calling
identity is, empirically, a MUCH more reliable trait than 4th-down
aggressiveness (a single discrete decision, noisier by construction) -- this
is a clean, positive finding on its own: scheme/formation/pace tendencies
persist. **Zero dimensions are excluded on reliability grounds.** Per the
task's one admissible reliability-based exclusion (a dimension whose YoY 95%
CI upper bound sits at or below zero is not a personality and is dropped
from cell-building) -- the bar is not met by any dimension, so all nine
(plus the defensive proxy) are eligible inputs.

## Predeclared cells (5: 2 identity, 3 matchup)

Population for all cells: NFL REG, close grade (schedules `spread_line`,
the same convention `scripts/nfl_weather_battery_screen.py` uses), full
local history 2009-2025 (`data/raw/*/schedules.parquet`, newest snapshot).
2009 games are excluded per cell for missing prior-season style (reported,
not silently dropped from the declared range). Method: joint week-blocked
bootstrap (block = season*100+week) PRIMARY, season-blocked bootstrap
SECONDARY, both `block_bootstrap_two_group`-identical to
`scripts/nfl_weather_battery_screen.py`/`scripts/nfl_bias_battery_screen.py`
(read both; same algorithm, vectorized multinomial block resample, jointly
resampling both arms from the same drawn blocks). Full-slate effect scaling
via `nfl_ats.experiment_runner.scale_subset_effect` (imported directly, not
reimplemented) -- `sign * raw_gap_fraction * 100 * fraction_of_slate`,
`accuracy_points` units. 20,000 bootstrap samples, seed 20260819 (today's
date, the repo's own convention), fixed and deterministic.
Quartile thresholds are the empirical top-quartile (0.75 quantile) of the
CENTERED dimension across the full 544-row 2009-2025 team-season panel
(544 = 32 teams x 17 seasons), computed once and reused identically by
`scripts/team_style_screen.py` -- not re-derived per season, so a threshold
cannot silently drift with each new season added.

**Every cell is recorded to `registry/weak_signals.json` after screening,
regardless of sign or interval shape**, per AGENTS.md; a sign prediction
below is a predeclaration of what would count as the mechanism working, not
a promise about what the data will show.

### Identity cells (team's own tendency vs the field)

**A1. `team_style_distinct_identity`** -- top-quartile teams by L2 distance
of their prior-season centered style vector (z-scored across the 9 reliable
offensive dimensions, pooled sd) from the league-mean origin, vs the field.
Team-perspective long table (one row per team per game, `team_covered`,
reused pattern from `scripts/nfl_bias_battery_screen.py::build_long_table`).
**Unsigned** -- no clean a priori mechanism for whether an unconventional
identity systematically helps or hurts against the spread; this cell tests
whether "weirdness" itself carries value either direction.

**A2. `team_style_short_game_identity`** -- top-quartile teams by prior-
season centered `short_pass_share` (the owner's own Patriots/Belichick
example), vs the field. Team-perspective long table, `team_covered`.
**Unsigned** -- a directional prediction here would require conditioning on
favorite/underdog status (a low-variance short-game identity plausibly
helps a favorite close out a game and hurts an underdog needing a
comeback), which this simple vs-field cell does not do; declaring a sign
without that conditioning would be inventing a mechanism the cell can't
actually test.

### Matchup cells (style-vs-style or style-vs-environment interactions)

**B1. `team_style_short_game_vs_pressure_defense`** -- AWAY team is a
short-game identity offense (prior-season centered `short_pass_share` top
quartile) AND HOME team's defense is a pressure-style-proxy identity
(prior-season centered `shotgun_rate_faced` top quartile -- opposing
offenses go to shotgun far more than league average against this defense).
Game-level, `home_cover`. **Predicted sign: NEGATIVE** on `home_cover` (a
quick-release short-game offense is mechanically better equipped to
neutralize a defense that draws heavy shotgun/pass-pro response elsewhere;
the market prices the defense's generic pressure quality, not this specific
offense-defense fit, so the away short-game offense should outperform its
line more than the market accounts for).

**B2. `team_style_pace_mismatch_dog_cover`** -- games in the top quartile of
`|home_seconds_per_play_pace_centered - away_seconds_per_play_pace_centered|`
(the biggest prior-season pace mismatches), vs the field. Value column is
`dog_cover` (favorite/underdog framing via `spread_line` sign; true
pick'ems, `spread_line == 0`, are excluded as missing, not defaulted).
**Predicted sign: POSITIVE** on `dog_cover` -- the stated variance
mechanism: a pace mismatch compresses total offensive possessions toward
the slower team's preference, and fewer possessions favor the underdog
(less opportunity for the true talent gap to assert itself). This is the
one cell in the battery with a clean, textbook predicted sign independent
of any team-quality confound.

**B3. `team_style_deep_ball_outdoor_wind`** -- AWAY team is a deep-ball
identity offense (prior-season centered `deep_share` top quartile) AND the
game is outdoor/open roof AND wind >= 15mph (identical threshold to
`scripts/nfl_weather_battery_screen.py`'s `high_wind_outdoor`/
`high_wind_road_favorite` cells, for direct comparability -- this bridges
the weather family, testing whether a team's SPECIFIC offensive identity
compounds the market's already-priced generic weather effect). Game-level,
`home_cover`. **Predicted sign: POSITIVE** on `home_cover` (the away
team's deep-ball-dependent offense is disproportionately degraded by wind;
the market prices generic wind effects, per the weather battery's own
leakage caveat that even those are actual-weather, not forecast-time, but
plausibly still underprices a team-specific vulnerability on top of the
generic one).

**Leakage caveat inherited from the weather battery**: `wind`/`temp` in
`schedules.parquet` are GAME-TIME ACTUALS, not pregame forecasts. B3 is
therefore a MECHANISM SCREEN like the weather battery's own cells -- an
upper bound on what a forecast-time feature could capture, not itself a
usable pregame predictor, independent of the style-identity half of the
cell (which IS pregame-safe, being prior-season).

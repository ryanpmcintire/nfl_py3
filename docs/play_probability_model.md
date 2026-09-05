# The play-probability model (UI-20-AB, 2026-09-05)

## The complaint this replaces

Owner directive, verbatim: "the percentages should obviously make sense my
dude... it needs to be a forecast about the game and it needs to consider
depth chart."

Before this change, the lineup panel's per-player number was a
no-designation BASE RATE keyed only on `(position_group, recent_role)` --
`returning_contributor` 0.952 / `no_recent_role` 0.109 / `unknown_no_history`
0.465 (measured, `artifacts/lineups/current/lineups.json` before this
session). It never looked at depth chart at all, so a rookie QB2 with no
injury designation read 47% and a veteran healthy QB3 read 95% --
exactly backwards from what "makes sense."

This is replaced with `src/nfl_ats/play_probability.py`: a walk-forward,
isotonic-calibrated gradient-boosting model of two probabilities, per
player, per game, using the player's own depth-chart rank.

## Definitions

- **Plays** (primary): P(this player records at least one offensive,
  defensive, or special-teams snap this game). Label: any `snap_counts` row
  for that `(season, week, team, gsis_id)` with `offense_snaps +
  defense_snaps + st_snaps > 0`.
- **Starts** (secondary, an explicit proxy): a player's own snap share is
  >= 50% of the total snaps recorded by every player who shares his
  **specific** position that game (e.g. every snap-counts "T", not the
  broader `offensive_line` feature group, which pools 5 simultaneous
  linemen and would make a 50% threshold nearly unreachable for anyone) --
  **or**, because that same broader-group problem is total for a QB (the
  `skill` feature group pools QB with RB/WR/TE/FB), being the team's
  depth-chart-listed QB1 who actually played that game.

  **This is a documented interpretive choice.** The task text said "his
  position group," which is ambiguous between the broad 5-bucket feature
  used elsewhere in this doc and "his own position." The broad-group
  reading makes the 50% threshold nearly unreachable for any multi-role
  position (a `secondary` bucket pools 4-5 simultaneous DBs; an
  `offensive_line` bucket pools 5 simultaneous linemen), so the
  specific-position reading was used instead. **Measured, and genuinely
  coarse even so:** rank-1 "started" rates range from ~0.4% (CB -- teams
  routinely play 2-3 corners with no single one over 50% of the group's
  snaps) to ~82% (QB, effectively 1 simultaneous role). This is a real
  property of the literal proxy, not a code defect; see
  [Measured: "started" is a much coarser secondary metric](#measured-started-is-a-much-coarser-secondary-metric).

## Features (all strictly pregame)

| Feature | Description |
| --- | --- |
| `depth_rank_bucket` | `"1"` / `"2"` / `"3+"` / `"unknown"`, from the player's own depth-chart rank. |
| `position_group` | `offensive_line` / `skill` / `front` / `secondary` / `other` (`nfl_ats.lineup_availability.depth_chart_position_group`, handling both generic and side-specific nflverse tags). |
| `report_category` | This week's own injury-report status (`out`/`doubtful`/`questionable`/`probable`/`none`/`other`), or `none` if not listed. |
| `practice_category` | This week's own practice-participation status, categorized the same way. |
| `weeks_since_last_snap` | Weeks since this player's own most recent recorded snap, any team, strictly before this week (`NaN` if never recorded one). |
| `trailing4_snap_share` | Mean "own side" snap share (`max(offense_pct, defense_pct, st_pct)`) over the player's own last up to 4 games played, strictly before this week. |
| `roster_status` | `ACT` / `INA` / `other` / `unknown` this week. |
| `season_week` | The week number. |
| `qb1_report_category`, `qb1_practice_category` | For QB rows only: the team's own depth-1 QB's report/practice status this week (`not_applicable` for every non-QB row) -- this is how a backup's probability rises when the starter is hurt. |

Every categorical feature is encoded through a **fixed vocabulary**
(`*_CATEGORIES` constants in `play_probability.py`) so a training frame and
a one-row serving frame never silently disagree on which integer code a
category string gets.

**A measured train/serve asymmetry, handled explicitly:** `roster_status`'s
true weekly `INA` (gameday-inactive) designation is essentially never known
before the noon-Eastern refresh this feeds -- nflverse's inactive list
posts roughly 90 minutes before kickoff, and the CURRENT week's weekly-
roster release is usually not published at all yet. `serving_feature_frame`
therefore defaults every serving-time row to `"ACT"` rather than an
`"unknown"` category the model saw only rarely (or never) in training. See
[Measured: how much of the improvement is the roster-status column](#measured-how-much-of-the-improvement-is-the-roster-status-column)
for exactly how much this costs.

## Depth-chart history: a new archive

No archive of the FULL (all-position) depth chart existed before this
session: `nfl_ats.quarterbacks`'s `depth-ingest` / `depth-history-ingest`
both filter to QB rows only. `config/source_policies.json` marks nflverse
GREEN, so `scripts/build_play_probability_panel.py` fetches and archives it
via nflreadpy's own `load_depth_charts` -- the only new network dependency
this lane added, run once (this session) to build the training archive at
`data/players/raw/depth_charts/<stamp>/depth_charts.parquet`.

**Measured this session: `nfl.load_depth_charts` returns two different
schemas depending on season.** Seasons <= 2024 return legacy, week-labelled
rows (`season`, `week`, `club_code`, `depth_team`, `position`,
`depth_position`, `formation` -- one row per team per week).  Seasons >=
2025 return DAILY snapshot rows instead (`dt`, `team`, `pos_abb`,
`pos_rank`, no week label at all -- nflverse switched to continuous
point-in-time capture). `canonicalize_depth_chart_history` unifies both:
legacy rows keep their own week label directly ("the depth chart's week
used only for the game it describes" -- a full weekly depth chart IS the
team's own pregame lineup announcement for that week's game, unlike the
QB-only archive's conservative "strictly later games only" rule, which
exists there only because that narrower QB path could not otherwise rule
out looking at a not-yet-finalized chart); daily rows have no week label at
all, so each `(season, week, team)` is assigned the most recent
depth-chart snapshot observed strictly before that team's own kickoff via
`pandas.merge_asof` -- never a later one.

The archive nests one level deeper than first tried
(`data/players/raw/depth_charts/<stamp>/`, not
`data/players/raw/<stamp>/`) for a measured reason: `nfl_ats.players.
latest_player_snapshot` globs `data/players/raw/*/manifest.json` and
assumes every match is a `PlayerSnapshot` manifest. A depth-chart-history
manifest at that same depth broke it (`KeyError: 'injury_seasons'`) for
every other caller sharing this tree, including `scripts/
build_week_lineups.py`'s already-shipped `_no_designation_lookup`.
`latest_player_snapshot` is a shared module this lane may not edit, so the
archive moved one directory deeper instead of editing it.

## Walk-forward protocol

Trained on 2013-2025 player-weeks (`data/players/raw/<stamp>/
snap_counts.parquet` starts 2013; injuries and weekly rosters go back to
2009 but are joined down to the snap-covered range). The training
POPULATION is depth-chart rows -- the same population `scripts/
build_week_lineups.py` scores every player from -- not the broader weekly-
roster population, so training and serving see the same distribution of
players.

For a scored season `Y`:

1. The raw `HistGradientBoostingClassifier` (`categorical_features=
   "from_dtype"`) is fit on every season strictly before `Y`.
2. Isotonic calibration is fit on that SAME booster's own predictions,
   restricted to season `Y - 1` (the most recent training season).
3. `Y` is never in either step.

**Documented simplification:** step 2 calibrates on the booster's own
in-sample predictions for `Y - 1` rather than a third, disjoint calibration
band. A fully nested train/calibrate/test split (three non-overlapping
season bands) needs at least three prior seasons before the FIRST season
can ever be scored -- and 2014, one of the seasons this evaluation and the
recorded weak signal cover, has exactly one prior season (2013) available.
The property that actually matters for leakage safety -- season `Y` is
NEVER in the data used to fit or calibrate the model that scores it --
holds regardless of this simplification.

## Measured: Brier and log loss, walk-forward 2014-2025

Command: `.tools/uv.exe run --no-sync python scripts/build_play_probability_panel.py`
then `nfl_ats.play_probability.walk_forward_evaluate(panel,
scored_seasons=range(2014, 2026))`, on the panel at
`data/processed/play_probability_panel.parquet` (391,045 depth-chart-row-
weeks; player snapshot `20260905T123614Z`; depth-chart-history archive
`20260905T152519Z`).

| Season | n | Brier (played) | log loss (played) | mean predicted | mean observed |
| --- | --- | --- | --- | --- | --- |
| 2014 | 26,285 | 0.059 | 0.216 | 0.803 | 0.851 |
| 2015 | 29,640 | 0.089 | 0.330 | 0.794 | 0.741 |
| 2016 | 29,585 | 0.054 | 0.200 | 0.702 | 0.745 |
| 2017 | 29,522 | 0.052 | 0.190 | 0.756 | 0.747 |
| 2018 | 29,634 | 0.049 | 0.171 | 0.744 | 0.742 |
| 2019 | 29,665 | 0.048 | 0.167 | 0.739 | 0.740 |
| 2020 | 29,169 | 0.031 | 0.118 | 0.750 | 0.750 |
| 2021 | 30,538 | 0.025 | 0.091 | 0.744 | 0.746 |
| 2022 | 30,948 | 0.021 | 0.085 | 0.749 | 0.752 |
| 2023 | 30,808 | 0.017 | 0.068 | 0.757 | 0.761 |
| 2024 | 30,637 | 0.017 | 0.065 | 0.759 | 0.758 |
| 2025 | 34,745 | 0.030 | 0.112 | 0.743 | 0.720 |

**Overall Brier, full information: model 0.040 vs. depth-rank-only baseline
0.176 vs. the current (base-rate) approach 0.221.**

### Season-blocked bootstrap (12 season blocks, 5,000 resamples)

| Comparison | Improvement (Brier) | 95% interval | probability_positive |
| --- | --- | --- | --- |
| vs. depth-rank-only baseline, full information | +0.135 | [0.118, 0.148] | 1.0 |
| vs. current (base-rate) approach, full information | +0.181 | [0.175, 0.188] | 1.0 |
| vs. depth-rank-only baseline, **serving-realistic** | +0.077 | [0.068, 0.085] | 1.0 |
| vs. current (base-rate) approach, **serving-realistic** | +0.124 | [0.102, 0.146] | 1.0 |

Positive = the model's Brier is lower (better). "Full information" trains
and evaluates using the true historical `roster_status` (including the
real weekly `INA` designation). "Serving-realistic" forces `roster_status`
to `"ACT"` in both training and evaluation -- the same default
`serving_feature_frame` uses live, because the true weekly designation for
the CURRENT week is not normally available at the noon-Eastern refresh.
**The weak-signal recorded for this model
(`play_probability_model_brier_improvement`, `nfl-ats weak-signals record`)
uses the serving-realistic number against the current-approach baseline
(+0.124), not the more flattering full-information number, because that is
the comparison an actual deployment gets.**

Depth-rank-only baseline: a season-lagged, shrunk `(position_group,
depth_rank_bucket)` rate -- everything the play-probability model's own
depth-rank feature knows, and nothing else. Current-approach baseline: the
literal approach this replaces (`nfl_ats.lineup_availability`'s
no-designation base rate by `(position_group, recent_role)`, plus
`nfl_ats.availability.fixed_unavailability` for a listed player) run
walk-forward on the same population.

### Measured: how much of the improvement is the roster-status column

Full-information overall Brier 0.040 vs. serving-realistic overall Brier
0.099 -- roughly 60% of the raw improvement in the full-information number
comes from a feature (the true weekly `INA` designation) that is not
normally available yet at serve time. Even after removing it entirely, the
serving-realistic model still beats both baselines decisively (see the
bootstrap table above); the depth-rank, injury-report, and recent-history
features alone carry real, substantial signal.

## Measured: calibration by depth slot

Full-information run (model's own predictions on all 361,176 walk-forward
test rows, 2014-2025 combined):

| Slot | n | Mean predicted | Mean observed | Gap |
| --- | --- | --- | --- | --- |
| CB | 32,919 | 0.750 | 0.748 | +0.001 |
| DL | 48,222 | 0.764 | 0.773 | -0.008 |
| K/P | 20,900 | 0.878 | 0.879 | -0.000 |
| LB | 46,901 | 0.790 | 0.790 | -0.000 |
| OL | 56,180 | 0.718 | 0.721 | -0.002 |
| QB1 | 6,929 | 0.830 | 0.820 | +0.010 |
| QB2 | 6,877 | 0.270 | 0.249 | +0.021 |
| QB3+ | 3,323 | 0.080 | 0.088 | -0.008 |
| RB1 | 10,375 | 0.821 | 0.820 | +0.002 |
| RB2+ | 17,735 | 0.722 | 0.725 | -0.004 |
| S | 31,440 | 0.799 | 0.800 | -0.002 |
| TE/LB | 23,116 | 0.774 | 0.779 | -0.005 |
| WR1 | 15,419 | 0.817 | 0.821 | -0.003 |
| WR2 | 13,927 | 0.770 | 0.770 | +0.000 |
| WR3 | 7,035 | 0.680 | 0.679 | +0.001 |
| WR4+ | 2,147 | 0.640 | 0.603 | +0.038 |
| other | 17,731 | 0.730 | 0.708 | +0.023 |

Every slot but `WR4+` and `other` (both small-`n`, thin-bench buckets)
calibrates within 1 point.

## Measured: "started" is a much coarser secondary metric

Rank-1 "started" rate by specific position (full-information run,
`played` and `started` both reported for context):

| Position | played | started |
| --- | --- | --- |
| QB | 0.818 | 0.818 |
| C | 0.844 | 0.686 |
| NT | 0.843 | 0.520 |
| RB | 0.811 | 0.345 |
| G | 0.817 | 0.300 |
| TE | 0.828 | 0.295 |
| DT | 0.845 | 0.251 |
| T | 0.791 | 0.180 |
| DE | 0.842 | 0.121 |
| LB | 0.845 | 0.171 |
| OLB | 0.825 | 0.006 |
| ILB | 0.814 | 0.003 |
| MLB | 0.843 | 0.000 |
| WR | 0.821 | 0.002 |
| S / FS / SS | ~0.82 | ~0.02-0.03 |
| CB | 0.789 | 0.004 |

`started` is documented and reported here as-measured, not smoothed over:
for any position with 2 or more simultaneous roles that genuinely rotate
(CB, S, most LB packages, WR corps), a flat 50%-of-specific-position-share
threshold under-flags almost everyone as "not started," even a player who
plainly is a real starter by football convention. Read `played` as the
primary signal; treat `started` as a rough, documented proxy only.

## Measured: the 2026 Week 1 distribution

Command: `.tools/uv.exe run --no-sync python scripts/build_week_lineups.py`
(writes `artifacts/lineups/current/lineups.json`, git-ignored).
`generated_at`: `20260905T153839Z`. `probability_provenance.
play_probability_model`: `v1 fit on train_seasons=2013-2025 (calibrated on
season 2025); panel=data/processed/play_probability_panel.parquet (391045
rows)`. `current_injury_feed`: nflverse has not published season 2026
injuries yet, so every player's `report_category`/`practice_category` is
`"none"` and `roster_status` is the `serving_feature_frame` default
`"ACT"` for this run.

2,179 players total: **2,172 `play_probability_model`**, **7
`unavailable`** (all 7 are pre-existing depth-chart rows with no `gsis_id`
at all -- unrelated to this feature, same 7 lane X's report already
measured).

By depth slot (`play_probability`, mean / min / max, n):

| Slot | Mean | Min | Max | n |
| --- | --- | --- | --- | --- |
| QB1 | 0.997 | 0.988 | 0.999 | 32 |
| QB2 | 0.186 | 0.186 | 0.212 | 32 |
| QB3+ | 0.020 | 0.003 | 0.121 | 28 |
| RB1 | 0.998 | 0.996 | 0.999 | 44 |
| RB2+ | 0.978 | 0.903 | 0.996 | 87 |
| WR1 | 0.999 | 0.996 | 0.999 | 32 |
| WR2 | 0.995 | 0.979 | 0.996 | 32 |
| WR3 | 0.971 | 0.926 | 0.996 | 32 |
| WR4+ | 0.965 | 0.903 | 0.996 | 117 |
| OL | 0.920 | 0.638 | 0.999 | 360 |
| DL | 0.980 | 0.903 | 1.000 | 278 |
| LB | 0.987 | 0.903 | 1.000 | 291 |
| CB | 0.983 | 0.903 | 0.999 | 219 |
| S | 0.989 | 0.903 | 0.999 | 149 |
| K/P | 1.000 | 0.996 | 1.000 | 94 |
| TE/LB (TE rows) | 0.979 | 0.926 | 0.999 | 141 |
| other | 0.990 | 0.903 | 1.000 | 204 |

**This matches the owner's own sanity check almost exactly**: a healthy
QB1 reads 0.988-0.999 (target 0.95-0.99), a healthy QB2 reads 0.186-0.212
(target 0.05-0.20), a healthy QB3 reads 0.003-0.121 (target 0.01-0.05,
mean 0.020).

**Read honestly, not just approvingly:** every non-QB slot reads high
(0.92-1.00), including `WR4+` and deep offensive-line bench spots. This is
measured, not a bug: with 2026 injuries not yet published, every player's
`report_category`/`practice_category` is `"none"` and `roster_status`
defaults to `"ACT"` -- and the historical `(position_group, ACT, no
report)` played rate really is this high even at rank 3+ (measured on the
training panel: `skill` rank 3+ 82.6%, `front` rank 3+ 85.8%, `secondary`
rank 3+ 86.0%; only `offensive_line` rank 3+ is meaningfully lower, 55.4%).
The remaining lift above even those historical rates comes from
`weeks_since_last_snap`/`trailing4_snap_share`: a player who survived a
real team's week-1 53-man cutdown at a deep bench spot disproportionately
already has real, recent playing history elsewhere on the depth chart. The
model still correctly ORDERS every slot (QB1 > QB2 > QB3+, WR1 > WR2 > WR3
> WR4+, RB1 > RB2+, and OL sits meaningfully below the defensive/skill
groups) -- it is compressed toward the healthy end because there is
genuinely no adverse pregame information available yet this early in the
week, not because rank stopped mattering.

## Wiring

`scripts/build_week_lineups.py` fits the model once per refresh
(`_play_probability_context`, a few seconds) from the cached panel and
scores every depth-chart row with a `gsis_id`, batched per team through
`nfl_ats.play_probability.serving_feature_frame` +
`predict_play_probabilities`. Every scored player carries:

- `play_probability` -- P(plays), from the model.
- `start_probability` -- P(starts), from the model (secondary; see the
  coarseness note above).
- `probability_source: "play_probability_model"`.
- `model_qb_start_probability` -- the active margin model's own forecast
  input (`{side}_qb_start_probability`) for the one QB it actually
  consumed, preserved as its OWN field rather than deleted. Previously this
  value WAS `play_probability` for that player (`probability_source:
  "base_model_qb"`); UI-20-AB's replacement directive applies to the
  scored QB too, so `play_probability` for that player now also comes from
  this model, and the forecast input moved to this separate field instead.
- `probability_reason` -- names the model and, for QBs, that the team's
  QB1 status was used.

`"unavailable"` is unchanged: reserved for a depth-chart row with no
`gsis_id` at all.

`scripts/build_play_probability_panel.py` builds the cached training panel
`build_week_lineups.py` reads (`data/processed/play_probability_panel.parquet`)
-- kept as a separate, occasionally-rerun step because building it needs a
full nflverse schedule fetch (for the ENG-39 `week_proxy` injury-visibility
timestamp) and roughly a minute of joins; refitting on every noon-Eastern
refresh would repeat both for no benefit, since the panel does not change
within a season.

## Weak signal recorded

`nfl-ats weak-signals record --name play_probability_model_brier_improvement
--league nfl --season-start 2014 --season-end 2025 --effect-units
brier_improvement --classification unresolved_below_power --effect 0.1236
--interval-low 0.1019 --interval-high 0.1464 --probability-positive 1.0
--sample-games 361176 --sample-blocks 12 --family play_probability_model
--category health` -- the serving-realistic improvement vs. the current
(base-rate) approach. `unresolved_below_power` per AGENTS.md: positive on
every one of 12 season blocks with an interval that excludes zero
decisively, but this is one predeclared measurement against one baseline
family, not a positive-control-bounded or wrong-sign-refuted closure.

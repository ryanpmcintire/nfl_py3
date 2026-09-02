# Totals model, wave 2 — predeclared design (not yet run)

Predeclared 2026-09-01T19:06:53Z, before any wave-2 model has been fit. Wave 1
(`docs/totals_model.md`) reserved this in its own frozen text: the
`*_ats_residual`/graph/schedule/bias/surface columns were excluded "because
a second wave may screen them as totals features, separately declared." This
document is that declaration, scoped narrower than that sentence promised —
it screens the drive-pace family only, not the full spread-oriented set,
because that family is the one wave 1's own text called out as most likely to
carry totals signal that graph/bias/schedule constructs do not ("Pace and
points-per-drive are the most natural totals predictors that wave 1 did not
have"). **Nothing in this file is a result.**

## Why this family

Wave 1 (measured, `docs/totals_model.md` Results, read 2026-09-01) found the
raw model total point estimate WORSE than the market (MAE 10.5495 vs
10.4249) and a blend at k=0.1 worth +0.0008 MAE points, week-blocked 95%
[-0.0062, +0.0077], `probability_positive` 0.583 — the same shape the margin
side found. Wave 1's 41-column allowlist carries efficiency-per-play
(EPA, CPOE, yards/play) but nothing about PACE: how many drives a team runs,
how long each drive takes, how many plays it uses. Two teams with identical
per-play efficiency but very different pace produce very different game
totals, and pace is not reachable from a per-play average. `data/processed/
game_features_pbp.parquet` (4,902 rows x 201 cols, measured 2026-09-01) adds
exactly that: the drive-level state family built by `enrich_with_pbp_features`
(`src/nfl_ats/pbp.py`).

**Verified this session, before freezing the list below** (measured
2026-09-01, `python -c` against the checked-out parquet files):
- `data/processed/game_features_pbp.parquet` is a superset of `data/processed/
  game_features.parquet` (wave 1's source): same shape on the row axis
  (4,902 rows), IDENTICAL `game_id` sets (4,902 of 4,902 in common), and the
  41 wave-1 allowlist columns carry numerically identical values in both
  files (spot-checked `home_off_epa_per_play`, `total_line`, `elo_diff` —
  `np.allclose(..., equal_nan=True)` all `True`). So switching the feature
  table for wave 2 changes nothing about the games scored or wave 1's own
  columns; it only adds columns.
- The drive family is built by the SAME point-in-time mechanism as every
  other PBP state column: `_build_pbp_states` (`src/nfl_ats/pbp.py:591-636`)
  EWM-smooths each metric within a team's own game history, and
  `enrich_with_pbp_features` (`src/nfl_ats/pbp.py:639-709`) attaches the
  state via `np.searchsorted(dates, game_date, side="left") - 1`
  (`pbp.py:678-687`) — the index strictly before the target game's date, the
  same guard pattern `nfl_ats.totals.walk_forward_predictions` uses at the
  season/week level. `PBP_ENRICHMENT_STATE_METRICS = PBP_STATE_METRICS +
  DRIVE_STATE_METRICS` (`src/nfl_ats/constants.py:209`) means the drive
  columns run through this exact code path, not a separate one.
- **Leakage test already covering this mechanism** (read 2026-09-01,
  `tests/test_pbp.py:200-210`,
  `test_current_game_plays_cannot_change_current_pregame_features`): inflates
  one play's EPA in a game and asserts the game's OWN pregame feature is
  unchanged while the FOLLOWING game's is. It exercises
  `enrich_with_pbp_features` generically over
  `PBP_ENRICHMENT_STATE_METRICS`, which includes every `DRIVE_STATE_METRICS`
  entry, so the drive family is covered by an existing, passing test. No new
  leakage test is owed for the feature-build mechanism itself, per this
  work package's instructions.
- What IS newly owed, and covered in `tests/test_totals_wave2.py`: proof
  that WAVE2's OWN join — merging the wider parquet onto the schedules
  population via `game_id` — does not corrupt or reorder anything, and that
  the walk-forward guard (train strictly before the target week) still holds
  when the drive columns drive the signal, not just when `wind` (wave 1's
  synthetic driver) does.

## Frozen contract

### Candidate allowlist (65 columns total: wave 1's 41 + these 24, nothing else)

Wave 1's `nfl_ats.totals.TOTALS_FEATURES` (41 columns, unchanged, imported
not copied) PLUS the following 24 columns, added as
`nfl_ats.totals_wave2.WAVE2_DRIVE_FEATURES`:

```
home_drive_points_per_drive          away_drive_points_per_drive
home_drive_yards_per_drive           away_drive_yards_per_drive
home_drive_plays_per_drive           away_drive_plays_per_drive
home_drive_seconds_per_drive         away_drive_seconds_per_drive
home_drive_scoring_rate              away_drive_scoring_rate
home_drive_turnover_rate             away_drive_turnover_rate
home_drive_points_per_drive_allowed  away_drive_points_per_drive_allowed
home_drive_yards_per_drive_allowed   away_drive_yards_per_drive_allowed
home_drive_plays_per_drive_allowed   away_drive_plays_per_drive_allowed
home_drive_seconds_per_drive_allowed away_drive_seconds_per_drive_allowed
home_drive_scoring_rate_allowed      away_drive_scoring_rate_allowed
home_drive_takeaway_rate             away_drive_takeaway_rate
```

This is exactly `{home,away}` crossed with `nfl_ats.constants.
DRIVE_STATE_METRICS` (12 entries, read 2026-09-01), i.e. the walk-forward
state version of every column `build_pbp_team_game_metrics`
(`src/nfl_ats/pbp.py:450-588`) derives from `build_drive_table`.

**Deliberately excluded, and named here so the boundary is explicit rather
than implicit:**
- `{home,away}_pbp_drives` (drive COUNT). The introductory context for this
  work package named it alongside the 12-metric family but fixed the added
  count at "24," which the 12-metric x 2-side family matches exactly and
  `pbp_drives` does not; honoring the literal number keeps the freeze
  unambiguous. A drive-count feature is a reasonable wave-3 candidate but is
  out of scope here BY THIS CHOICE, not by data absence — recorded so it is
  not silently forgotten.
- `diff_drive_*` (12 columns) and `diff_pbp_drives` — for the identical
  reason wave 1 excluded every `diff_*` column: a total rides the SUM of the
  two teams' production, not their difference, and ridge can form whatever
  sum weighting it wants directly from the `home_*`/`away_*` pair. Verified
  both families are present and excludable without loss (measured
  2026-09-01: `game_features_pbp.parquet` carries `diff_drive_*` and
  `diff_pbp_drives` columns; neither is in the frozen 65).
- Every other wave-1-excluded family (`*_ats_residual`, graph, schedule
  rating, bias, surface, weather, player, quarterback, weak-stack) — still
  out of scope; this is a drive-pace screen only, not "wave 2 in full."

### Comparator

Wave 1, EXACTLY, reproduced fresh inside this run rather than read from the
prior artifact: same population rule (newest schedules, non-null
`home_score`/`away_score`/`total_line`), same feature source class (joined
on `game_id`), same pipeline
(`SimpleImputer(median,+indicator) -> StandardScaler -> Ridge(alpha=10.0)`),
same walk-forward protocol (`min_train_games=500`, expanding window by
`(season, week)`), same blend grid `{0.0, 0.1, ..., 1.0}`, same bootstrap
seed `20260901`. Concretely: `nfl_ats.totals.load_population`,
`nfl_ats.totals.walk_forward_predictions` (default `features=TOTALS_FEATURES`),
`nfl_ats.totals.blend_sweep`, `nfl_ats.totals.choose_weight` — called
unmodified, not re-derived — against `data/processed/game_features.parquet`,
the same file wave 1 read. The wave-1 blend weight used for the paired
comparison is **k = 0.1**, wave 1's own already-chosen (not re-swept) value,
per this work package's instructions — wave 2 does not get to re-pick wave
1's operating point.

### Population

Identical row set to wave 1 (verified above: same `game_id`s, same
`home_score`/`away_score`/`total_line`-non-null filter, applied to the SAME
schedules file via `nfl_ats.totals.newest_schedules_path`). Wave 2 reads
`data/processed/game_features_pbp.parquet` instead of `game_features.parquet`
purely to reach the extra 24 columns — every row that qualifies for wave 1
qualifies for wave 2 and vice versa, by construction. Primary read on
`game_type == "REG"` (3,935 games, matching wave 1's count exactly);
playoffs (188 games) reported separately, never pooled, per FND-15 lineage.

### Pipeline

Unchanged from wave 1: `SimpleImputer(strategy="median", add_indicator=True)`
-> `StandardScaler` -> `Ridge(alpha=10.0)`
(`nfl_ats.totals.make_totals_estimator`, reused not reimplemented). Alphas
{1, 100} reported alongside for transparency only, exactly as wave 1 does —
never used to pick a winner.

### Protocol

Unchanged: expanding-window walk-forward by chronological `(season, week)`,
train strictly before the target week, `min_train_games = 500`. Reused
verbatim via `nfl_ats.totals.walk_forward_predictions(population,
features=WAVE2_FEATURES)` — the SAME guarded function wave 1 uses and
`tests/test_totals.py` already covers, called with the wider column list.

### Metrics

- **Primary**: paired per-game |error| difference between (a) the wave-2
  blend at ITS OWN MAE-minimizing k (from a fresh sweep over the same
  `{0.0, ..., 1.0}` grid, computed on wave-2's walk-forward predictions) and
  (b) the wave-1 blend at the frozen k = 0.1. Sign convention: positive =
  wave 2 closer to the actual total (wave-1 |error| minus wave-2 |error|),
  matching the sign convention already stored in the registry for
  `totals_market_residual_blend`. Run through
  `nfl_ats.clv.week_blocked_bootstrap` (reused unmodified, same block key
  `week`, same 2,000 resamples, same seed 20260901). Report
  `probability_positive` for "wave 2 is better than wave 1" — never a
  binary "the interval contains zero" read.
- **Secondary**: wave-2 blend vs market alone (same construction wave 1
  used for its own vs-market read); per-season MAE deltas for both wave-1
  and wave-2 blends; playoffs scored by the same walk-forward models and
  reported separately, never pooled into the primary.

### Decision rule

EV, per AGENTS.md ("A promotion bar is not a decision bar" / "Grade the
decision at the opener"): if `probability_positive` > 0.5 for "wave 2 beats
wave 1," wave 2 is the favourite and the served total SHOULD move to it. The
wiring change — replacing `nfl_ats.tiebreaker.TOTALS_RESIDUAL_WEIGHT`'s
source model and feature table — is applied (read, `src/nfl_ats/tiebreaker.py`
lines 647-730, 2026-09-02): the tiebreaker prefers the frozen wave-2 view when
the PBP table is present, retains the recorded blend weight `0.1`, and uses
market-only totals when a present PBP table is stale, incomplete, or
misaligned. A missing PBP table retains the explicitly labelled wave-1
fallback for fresh clones; wave-2 model failure never silently substitutes
wave 1.

### Positive control (frozen BEFORE any wave-2 outcome is computed)

Purpose: prove the walk-forward + ridge + blend-sweep + bootstrap machinery
newly wired for the extended 65-column allowlist can register a large,
unambiguous effect when one is actually present — an instrument-sanity
check, not a claim that the screen's own (unknown, not-yet-computed) result
is "bounded" by it. A positive control that only shows the pipeline detects
a MAXIMAL injected effect does not by itself establish power to detect an
effect the size of wave 1's own +0.0008 — that limitation is stated here in
advance so it cannot be quietly upgraded into a bigger claim after the run.

Method, frozen: take the wave-2 population and design matrix unchanged
except for ONE column — `home_drive_points_per_drive` (an arbitrary,
explicit, pre-chosen member of the 24) — whose values are REPLACED, for
every row (train and test alike), with that row's own `total_residual`
(the target itself, unit slope, zero noise). Every other column, the
walk-forward guard, the pipeline, the blend grid and the bootstrap are
identical to the real screen. Expected and falsifiable BEFORE running:
the MAE-minimizing k should land at or near 1.0 (the sweep should reward
leaning almost entirely on the model), the blend MAE improvement over market
should be large (order of several MAE points, not thousandths), and
`probability_positive` should be indistinguishable from 1.0. If the control
does NOT show this shape, that is a pipeline bug to fix before the screen's
own result is trusted at all — not a reason to soften the control's own
verdict.

### Recording

One registry entry: family `totals_market_residual` (same family as wave 1
— same units, same scale, same population, so pooling later is legitimate
per AGENTS.md's commensurability rule), name
`totals_market_residual_wave2_vs_wave1`, `--effect-units mae_improvement`
(confirmed present in `nfl-ats weak-signals record --help`, read
2026-09-01 — the unit built for exactly this "baseline-minus-candidate,
positive-is-better" convention per the `totals_market_residual_blend`
entry's own notes field, read from `registry/weak_signals.json`), effect =
mean paired |error| improvement of wave 2 over wave 1 (wave-1 |error| minus
wave-2 |error|, positive = wave 2 better). Recorded via `nfl-ats
weak-signals record` under the cross-process lock wrapper, never by hand.
Classification follows the outcome under the taxonomy below — decided AFTER
the numbers, never before.

**Closing-grounds taxonomy (binding, restated verbatim per this work
package's instructions since subagents and this file's own execution do not
see the session hooks):** An interval or CI that contains zero is NEVER
grounds to reject, fail, or close an experiment. Only two grounds ever close
a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero." The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

### Required tests (`tests/test_totals_wave2.py`)

1. Allowlist enforcement for the extended 65-column list — `WAVE2_FEATURES`
   equals wave 1's `TOTALS_FEATURES` plus exactly `WAVE2_DRIVE_FEATURES`
   (24, matching `{home,away}` x `DRIVE_STATE_METRICS`), no `diff_*`, no
   `pbp_drives`; a renamed/extra column is still a hard error / silently
   excluded (reusing `nfl_ats.totals.design_matrix` unmodified against the
   wider list).
2. Join point-in-time proof: (a) a synthetic-population walk-forward test
   in the style of `tests/test_totals.py`'s flip-week fixture, but with a
   DRIVE column as the signal driver, showing the guarded prediction at the
   flip week matches a model trained on strictly-earlier weeks only and
   differs from one that also saw the flip week; (b) a real-file check that
   `load_population_wave2` returns the identical `game_id` set as wave 1's
   `load_population`, and that its drive-column values exactly match
   `game_features_pbp.parquet`'s own values for a sample of games (no
   join corruption).
3. Paired-comparison math: sign convention (positive = wave 2 better) and
   the mean-improvement helper on a small hand-built example.
4. Positive-control shape check (light-weight, synthetic — the full-data
   positive-control run is produced by `scripts/totals_wave2_backtest.py
   --mode positive-control` and reported in Results): injecting a
   column that equals the target drives the chosen k toward 1.0 and the
   MAE improvement strongly positive, on a small synthetic population.

## Status

**RUN 2026-09-01.** `src/nfl_ats/totals_wave2.py`,
`scripts/totals_wave2_backtest.py`, and `tests/test_totals_wave2.py` (17
tests, all passing; measured with the focused wave-2/tiebreaker run on
2026-09-02) shipped this session. Both modes ran once, in the
predeclared order (positive control, then screen), both under stamp
`wp18run`, and the finding is recorded in
`registry/weak_signals.json` as `totals_market_residual_wave2_vs_wave1`
(family `totals_market_residual`, registry at 625 signals after the write).
The contract above is unchanged from its frozen form.

## Results (added after the run, 2026-09-01)

Produced by
`.\.tools\uv.exe run --no-sync python scripts/totals_wave2_backtest.py --mode positive-control --stamp wp18run`
and
`.\.tools\uv.exe run --no-sync python scripts/totals_wave2_backtest.py --mode screen --stamp wp18run`.
Prediction-level output: `artifacts/totals_backtest_wave2/wp18run/positive_control/`
and `artifacts/totals_backtest_wave2/wp18run/screen/` (each with
`results.json`, `predictions.parquet`/`wave1_predictions.parquet` +
`wave2_predictions.parquet`, and `paired_wave2_vs_wave1.parquet` for the
screen). Registry entry: `totals_market_residual_wave2_vs_wave1`, family
`totals_market_residual`, league nfl.

### Positive control (instrument-sanity check, run first)

`home_drive_points_per_drive` replaced by each row's own `total_residual`
(unit slope, zero noise), same walk-forward/sweep/bootstrap pipeline as the
real screen, same 3,935 regular-season games. Matches the shape frozen above
BEFORE this ran: chosen k = **1.0** (>= the frozen 0.8 threshold), MAE
improvement over market **+10.3657** points, week-blocked bootstrap 95%
**[+10.1568, +10.5809]**, `probability_positive` **1.000**
(`shape_matches_expectation: true`, computed automatically from the frozen
thresholds). The walk-forward + ridge + blend-sweep + bootstrap machinery
built for the extended 65-column allowlist registers a large, unambiguous
injected effect correctly. As stated in the frozen method: this proves the
pipeline is not silently broken; it does NOT by itself establish power to
resolve an effect the size of wave 1's own +0.0008 MAE points, and is not
used to classify the screen's own result below.

### Screen: MAE / RMSE, regular season (3,935 games, identical population to wave 1)

| arm | MAE | RMSE |
| --- | --- | --- |
| market total alone | 10.4249 | 13.1697 |
| wave 1, chosen blend (k=0.1, frozen) | 10.4241 | 13.1650 |
| wave 2, raw model (k=1.0) | 10.5799 | 13.3217 |
| **wave 2, chosen blend (k=0.1, own sweep minimum)** | **10.4221** | **13.1635** |

Market MAE reproduces wave 1's own 10.4249 exactly, confirming the wave-2
population is the identical game set (as the predeclaration's join-integrity
tests already proved on the real files). Wave 2's raw model point estimate
is again worse than the market — the same shape wave 1 and the margin side
both found — and slightly worse than wave 1's own raw model (10.5799 vs
10.5495): more columns did not make the unblended point estimate better.

### Wave-2 blend sweep, `total_line + k * predicted_residual`

| k | MAE | RMSE | MAE improvement vs market |
| --- | --- | --- | --- |
| 0.0 | 10.4249 | 13.1697 | +0.0000 |
| **0.1** | **10.4221** | **13.1635** | **+0.0028** |
| 0.2 | 10.4226 | 13.1621 | +0.0023 |
| 0.3 | 10.4277 | 13.1655 | -0.0028 |
| 0.4 | 10.4362 | 13.1736 | -0.0113 |
| 0.5 | 10.4490 | 13.1865 | -0.0241 |
| 0.6 | 10.4671 | 13.2042 | -0.0422 |
| 0.7 | 10.4890 | 13.2265 | -0.0641 |
| 0.8 | 10.5152 | 13.2536 | -0.0903 |
| 0.9 | 10.5458 | 13.2853 | -0.1209 |
| 1.0 | 10.5799 | 13.3217 | -0.1550 |

Wave 2's own MAE-minimizing k is **0.1** — the same numeric value as wave
1's frozen operating point, by coincidence of this sweep rather than by
construction (wave 2 re-swept its own grid independently). Wave 2 vs market
alone (secondary metric): **+0.0028** MAE points, about 3.5x wave 1's own
+0.0008.

### Primary: paired wave-2-vs-wave-1 |error| improvement

Wave 1 graded at its frozen k=0.1; wave 2 graded at its own chosen k=0.1
(same value, independently derived). Week-blocked bootstrap
(`nfl_ats.clv.week_blocked_bootstrap`, 2,000 resamples, 261 week blocks,
seed 20260901), positive = wave 2 closer to the actual total:

**+0.0020 total points, 95% [-0.0024, +0.0063], `probability_positive` 0.8235.**

### Per-season MAE (positive = wave 2 better than wave 1)

| season | games | wave1 MAE | wave2 MAE | wave1 vs wave2 delta |
| --- | --- | --- | --- | --- |
| 2010 | 16 | 10.787 | 10.864 | -0.0774 |
| 2011 | 256 | 9.383 | 9.386 | -0.0028 |
| 2012 | 256 | 10.407 | 10.421 | -0.0143 |
| 2013 | 256 | 11.088 | 11.084 | +0.0044 |
| 2014 | 256 | 10.752 | 10.769 | -0.0166 |
| 2015 | 256 | 10.529 | 10.517 | +0.0120 |
| 2016 | 256 | 9.935 | 9.934 | +0.0006 |
| 2017 | 256 | 11.142 | 11.143 | -0.0010 |
| 2018 | 256 | 10.595 | 10.580 | +0.0149 |
| 2019 | 256 | 10.838 | 10.832 | +0.0057 |
| 2020 | 256 | 10.159 | 10.154 | +0.0056 |
| 2021 | 272 | 10.778 | 10.760 | +0.0177 |
| 2022 | 271 | 10.404 | 10.392 | +0.0122 |
| 2023 | 272 | 10.253 | 10.250 | +0.0028 |
| 2024 | 272 | 9.728 | 9.739 | -0.0107 |
| 2025 | 272 | 10.384 | 10.380 | +0.0040 |

10 of 16 seasons favour wave 2, 6 favour wave 1. The single largest
deviation is 2010, the 16-game warm-up stub (the first regular-season block
`min_train_games=500` clears), where wave 2 is 0.0774 points worse — every
other season sits within +-0.02. Reported as measured, not smoothed over.

### Playoffs, reported separately (188 games, never pooled into the primary)

Scored by the same walk-forward models as the regular-season read:

| arm | MAE | vs market |
| --- | --- | --- |
| market | 10.9229 | — |
| wave 1, k=0.1 | 10.8991 | +0.0237 |
| wave 2, k=0.1 | 10.8907 | +0.0321 |

Wave-1's number reproduces `docs/totals_model.md`'s own playoff figure
exactly (10.8991 vs market 10.9229, +0.0237), a second population-identity
check. Wave-2-vs-wave-1 paired mean improvement on playoffs: **+0.0084**,
same direction as the regular season and about 4x the size, on under 5% of
the sample — a separate report, per FND-15 lineage, not folded into the
primary.

### What this implies for the decision, before what is wrong with it

At `probability_positive` 0.8235, wave 2 is the clear favourite over wave 1
under this project's EV decision rule — the pool serves a tiebreaker total
every week regardless, so "wait for a tighter interval" is not one of the
available choices; declining to prefer the 82%-favourite side is choosing
the 18% side. The primary interval crosses zero — [-0.0024, +0.0063] — and
per AGENTS.md that is the EXPECTED shape for a real small signal at this
evaluator's resolution, not grounds to reject it. The direction is
consistent everywhere it was checked: wave 2 beats wave 1 vs market
(+0.0028 vs +0.0008), on the majority of seasons (10/16), and on playoffs
(+0.0084 vs +0.0237 the wave-1 number, i.e. wave 2 pulls further ahead of
market there too) — no measurement in this run pointed the other way at the
aggregate level.

**Applied wiring (read, `src/nfl_ats/tiebreaker.py` lines 647-730,
2026-09-02):**
1. `model_total_view_wave2` mirrors `nfl_ats.totals.model_total_view` but is
   built on `WAVE2_FEATURES` / `data/processed/game_features_pbp.parquet` via
   `load_population_wave2` instead of `TOTALS_FEATURES` /
   `game_features.parquet`.
2. `tiebreaker_report` prefers `model_total_view_wave2` whenever the PBP
   table is present.  A present table that is missing, stale, incomplete, or
   misaligned fails closed to the market-only total; it never silently falls
   back to wave 1.  If the entire PBP table is absent, the existing explicitly
   labelled wave-1 fallback remains for fresh clones.
3. `TOTALS_RESIDUAL_WEIGHT` stays numerically **0.1** — wave 2's own
   MAE-minimizing k matches wave 1's by coincidence of this sweep. What
   changes on adoption is which fitted model produces the residual being
   blended in (65-column ridge instead of 41-column ridge), not the blend
   weight itself.

What the number is not: an edge, or a settled question. +0.0020 points of
improvement on a 10.4-point error is still on the order of two parts in ten
thousand, and the honest headline of this wave, like wave 1's, is that the
raw (unblended) model point estimate is worse than the market with or
without the drive-pace columns. The positive control confirms the pipeline
CAN find a real effect when one is injected at full strength; it does not
certify that a 0.002-point effect specifically survives at the resolution
this evaluator has.

Classification: `unresolved_below_power`. Not `refuted_mechanism` (the
interval is not wholly on the wrong side of zero and `probability_positive`
0.8235 favours the candidate). Not `bounded_by_control` (the positive
control proved the pipeline can register a MAXIMAL injected effect, not that
it was calibrated to detect one this small — stated as a limitation in the
frozen method above, before the screen ran).

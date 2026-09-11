# ENV-05: weather interactions (wind x style, heat x pace, surface x traits)

Written **before** `scripts/weather_interactions_screen.py` scores anything.
ENV-05's row (`ROADMAP.md`) ran a 5-cell battery on 2026-08-19
(`scripts/nfl_weather_followup_screen.py`, `docs/weather_followup.md`) using
GAME-TIME ACTUAL weather, not a forecast: `weather_followup_temp_gap_cold_visitor`,
`weather_followup_wind_gap_visitor`, `weather_followup_high_wind_pass_heavy_visitor`
(wind x pass-heavy, but restricted to the VISITING team's own actual-weather
game, not a general pass-heavier-side rule, and never re-tried with a forecast),
`weather_followup_rest_disadvantage_cold`, and a fifth cell. All five are
`unresolved_below_power`, already recorded. ENV-01's row separately ran a
12-cell forecast-timed family that includes `forecast_weather_kn_wind_passing_away_favorite`
-- but that cell is restricted to the AWAY team specifically being BOTH the
market favorite AND pass-heavy, not a general "fade whichever side is
pass-heavier" rule. **This doc runs the four ENV-05 mechanisms the row lists
as not yet tried in a general, forecast-timed form**: wind x pass-heavier side
(general, not away-favorite-restricted), wind x kicking reliance (never
tried), heat x pace (never tried), and surface change x visiting unit traits
(never tried). Nothing below repeats a 2026-08-19 cell.

## Binding closing-grounds taxonomy (governs every verdict in this doc)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record` (or `nfl-ats rotation record` for a production screen),
report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command
errors, the verdict is wrong, not the validator. Every cell below is
recorded regardless of sign. Decisions on what to play are made on expected
value (`probability_positive` above 0.5 favours the candidate); a 0.90
threshold gates only what the docs may claim as resolved, never what gets
played.

## Data sources (read this session)

- **Forecast archives, two cutoffs, both full 2009-2025 REG populations**
  (`docs/forecast_archive_build.md`, `docs/forecast_weather_screen.md`):
  - `data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet` --
    the **deadline-visible cutoff**, `min(kickoff, Sunday 16:00 ET)` per
    game, the actual pool decision timestamp. **This is the playable read**
    used for every production screen below. 4,431 rows, `fetch_status=='ok'`
    for 4,380 (domestic), 51 `unmappable_international_stadium`.
  - `data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet`
    -- a **second, labelled read**: issuance closest-before-kickoff, no
    Sunday-16:00 floor for SNF/MNF. 4,431 rows, 4,379 `ok`, 51 unmappable,
    1 `transport_error`. Reported alongside the primary read for every
    screen-level cover-rate cell; NOT separately re-run at the production
    stage (a second production window per cell would double-spend rotation
    windows on two cutoffs of the same mechanism -- the same overlap
    discipline `docs/forecast_weather_screen.md` already applies between
    `tuesday_noon` and `kickoff_nearest` siblings).
  - Both archives carry `forecast_temp_f`, `forecast_wind_mph`
    (knots-converted), `fetch_status`, `roof` (schedules' own, actual,
    disclosed as a known-in-advance stadium fact for fixed roofs, unchanged
    convention from `docs/forecast_weather_screen.md`).
- **Team style traits, PRIOR-season, pregame-safe by construction**:
  - Pass rate: `nfl_ats.experiment_runner._team_season_pass_rate` (raw PBP,
    `pass_attempt/(pass_attempt+rush_attempt)` per team-season, posteam-based),
    reused unchanged from the existing `forecast_weather_kn_wind_passing_away_favorite`
    cell's own machinery, not rebuilt.
  - Kicking reliance: **new this session**, built the same way (posteam-grouped
    over the same raw PBP snapshot) as `n_field_goal_attempts / n_team_games`
    per team-season. **Disclosed limitation**: this repo's PBP snapshot
    carries `play_type` but no `field_goal_result` field (checked this
    session -- absent from every column list probed), so this is an
    ATTEMPT-based proxy for "a team's scoring leans on field goals," not a
    makes-based one; a team that settles for more field-goal tries (whether
    made or missed) is read as more kick-reliant. Disclosed, not resolved.
  - Pace: `data/pbp/team_style/team_season_style.parquet`'s
    `seconds_per_play_pace` (lower = faster tempo), the same team-season
    style cache `pace_mismatch_dog_tilt_overlay.py` already reads in
    production. Reused unchanged, not rebuilt.
  - Every trait uses the project's existing `_lag_and_quartile` /
    `_year_over_year_reliability` helpers (`nfl_ats.experiment_runner`,
    already used by the referee-crew and `wind_passing_away_favorite`
    cells): global `qcut(4)` over every (team, season) pair with a valid
    year-over-year lag (this season = prior season + 1), so quartile
    membership for game G always uses only data from BEFORE G's season.
- **Surface**: the latest `data/raw/*/schedules.parquet` snapshot's own
  `surface` column (confirmed present this session: `grass`, `fieldturf`,
  `sportturf`, `matrixturf`, `astroturf`, `a_turf`, `astroplay`,
  `dessograss`, and a few blank rows). **Surface history exists on disk** --
  cell 4 runs. Reuses `nfl_ats.surface_switch_tilt_overlay.surface_switch_flag_by_game`
  (the already-registered, already-live `surface_switch_tilt_overlay`
  challenger's own flag builder: away team's modal home surface this season
  is grass AND this game's surface is turf) unchanged, not rebuilt --
  exactly the "reuse the outdoor-venue logic rather than rewriting it"
  instruction, applied to the surface analogue of that logic.
- **Base game table**: `data/processed/game_features.parquet` (REG rows,
  `home_cover`/`spread_line`/`result`) plus schedules' `roof`/`surface`,
  reusing `scripts/snow_game_home_prep_screen.py`'s `_latest_schedules()`
  loader and `OUTDOOR_ROOFS` constant directly (imported, not
  reimplemented), and `nfl_ats.experiment_runner._base_team_game_table` for
  the team-long (one row per team per game) shape cells 1-3 need.
- **Production**: `data/processed/game_features_weak_stack.parquet`, the
  active `weak_stack` model via `resolve_active_model_config` /
  `opener_pick_evaluation` (opener grade), rotation windows via `nfl-ats
  rotation declare --grade opener --acknowledge-mined` / `assign` / `record`.
  All four families drew **[2020, 2021]** on `assign` (the earliest
  eligible opener block, the same window other lanes drew tonight) --
  **measured**, not requested by size.

## Method (shared across all 4 cells)

Week-blocked joint bootstrap primary (block=`season*100+week`),
season-blocked secondary (block=`season`), 20,000 samples, seed `20260910`,
using this repo's shared `probability_positive_from_draws` helper
throughout -- same method, sample count, and seed convention as every prior
weather battery in this project.

For cells 1-3 (team-level traits), the population is **team-long**: one row
per (game, team) via `_base_team_game_table`, outcome = `team_covered`
(1 if that team covered the spread, pushes dropped). A flag on a team's own
row means "fade this team" -- report `sign=-1` (predicted: flagged team's
own cover rate is LOWER than the complement). This resolves the
home/away-favorite-only ambiguity `docs/forecast_weather_screen.md` flagged
for its `wind_passing_away_favorite` cell (a home-favorite mirror needing the
opposite sign under the same flag name): team-long framing needs no
one-sided restriction, because "fade this team" is well-defined regardless
of whether the flagged team is playing at home or away. Effect is reported
SIGNED so that positive always confirms the predicted (fade) direction: for
a raw group gap `g = mean(team_covered | flag) - mean(team_covered | not
flag)`, the reported effect is `-100*g` (accuracy points) and reported
`probability_positive = 1 - raw_probability_positive` (exact under sign
negation for `probability_positive_from_draws`'s `P(>0)+0.5*P(==0)`
definition).

Cell 4 stays `home_cover`-framed directly (not team-long): the flag it reuses
(`surface_switch_flag_by_game`) is already one-sided by construction (only
ever true for the away team), so `home_cover` unambiguously represents
"the fade succeeded" and needs no re-signing.

Every cell reports, vs TWO complements: (a) the full population (every other
team-row / game, matching "cover rate ... vs complement" literally), and
(b) a narrower control isolating the interaction from the weather main
effect -- "same weather threshold met, trait NOT top/bottom quartile" -- the
same design snow_game_home_prep.md uses ("all other outdoor cold games") to
separate a compound condition's marginal contribution from its weather-only
parent.

Reliability is reported two ways per cell 1-3 (cell 4 reuses the already-registered
`surface_switch_tilt_overlay` trait, itself a same-season aggregate with no
year-over-year value, matching that overlay's own convention): (a) the
underlying team trait's own year-over-year Pearson correlation (pass rate /
FG-attempt rate / pace), attached as `--reliability` on each `weak-signals
record` call -- the field AGENTS.md calls decisive for the
`no_split_half_reliability` closing ground; and (b) an odd-vs-even-season
split of the interaction cell's own cover-rate gap, matching
`docs/snow_game_home_prep.md`'s convention for a per-game situational
condition that itself has no year-over-year value.

## The 4 predeclared cells (frozen before scoring)

1. **`weather_interactions_wind_pass_heavy_fade`.** Flag (team-row): outdoor
   AND `forecast_wind_mph >= 15` AND this team's PRIOR-season pass-attempt-rate
   global quartile == 4 (most pass-heavy). Predicted: FADE this team (its own
   `team_covered` rate is lower when flagged). Narrow control: same wind/outdoor
   condition, trait quartile != 4.
2. **`weather_interactions_wind_fg_reliant_fade`.** Flag (team-row): outdoor
   AND `forecast_wind_mph >= 15` AND this team's PRIOR-season FG-attempts-per-game
   global quartile == 4 (most kick-reliant, attempt-based proxy). Predicted:
   FADE this team (wind disrupts field-goal distance/accuracy, hurting a team
   whose scoring leans on kicks). Narrow control: same wind/outdoor condition,
   trait quartile != 4.
3. **`weather_interactions_heat_pace_fade`.** Flag (team-row): outdoor AND
   `forecast_temp_f >= 85` AND this team's PRIOR-season pace global quartile
   == 1 (fastest tempo, lowest seconds-per-play). **Predeclared direction:
   FADE pace in heat** (mechanism: a high-tempo team runs more plays per
   game, compounding heat-driven fatigue over four quarters, a cost a
   slower-tempo team pays less of) -- this row's own text offered either
   direction; FADE is the one scored. Narrow control: same heat/outdoor
   condition, trait quartile != 1.
4. **`weather_interactions_surface_switch_rush_fade`.** Flag (game-level,
   home_cover-framed): `surface_switch_flag` (away team's modal home surface
   this season is grass AND this game's surface is turf, reused from
   `nfl_ats.surface_switch_tilt_overlay` unchanged) AND the away team's
   PRIOR-season pass-attempt-rate global quartile == 1 (most run-heavy).
   Predicted: home_cover HIGHER when flagged (fade a run-heavy visitor
   thrown onto an unfamiliar surface -- traction-dependent rushing schemes
   are read as more surface-sensitive than pass-heavy ones). No forecast
   archive dependency; surface is a known-in-advance fact at every decision
   cutoff. Narrow control: `surface_switch_flag` true, away pass-rate
   quartile != 1.

## Recording commitment

Every screen-level cell (cover-rate gap vs. both complements, both cutoffs
for cells 1-3, both season halves) records to `registry/weak_signals.json`
via `nfl-ats weak-signals record`, `league=nfl`, `family` set per cell,
`effect_units=accuracy_points`, `category=environment`,
`unresolved_below_power` regardless of what the interval looks like, unless
a whole interval sits on the wrong side of zero (a resolved wrong sign) or a
positive control is run (none is, in this doc) -- the only admissible
alternative classifications, per AGENTS.md. Every production screen records
to `registry/rotation_registry.json` via `nfl-ats rotation record` the same
way, spending each family's assigned `[2020, 2021]` window.

## Results

**Measured this session**, `.\.tools\uv.exe run --no-sync python scripts\weather_interactions_screen.py screen` and `screen-surface`
(`artifacts/weather_interactions/screen_results.json`,
`artifacts/weather_interactions/surface_switch_results.json`), then
`production --cell {wind_pass_heavy,wind_fg_reliant,heat_pace,surface_switch_rush}`
(`artifacts/weather_interactions/production_*_results.json`). Every number
below is the script's own output, unmodified, then recorded via
`scripts/record_weather_interactions.py` (28 `nfl-ats weak-signals record`
calls, all succeeded, `artifacts/weather_interactions/record_log.txt`) and
`scripts/record_weather_interactions_production.py` (4 `nfl-ats rotation
record` calls, all succeeded, `artifacts/weather_interactions/record_production_log.txt`).

### Trait reliability (year-over-year, 512 team-season pairs each, `nfl-ats` PBP snapshot `20260817T184927Z` for pass/FG, `data/pbp/team_style/team_season_style.parquet` for pace)

| trait | reliability |
|---|---:|
| prior-season pass-attempt rate | 0.3929 |
| prior-season FG-attempts/game (attempt proxy) | 0.0874 |
| prior-season pace (seconds/play) | 0.5233 |

None of these are zero -- the `no_split_half_reliability` closing ground
does not apply to any of the three traits. The FG-attempt-rate trait is the
weakest (0.087), a real caution on cell 2's underlying persistence, reported
here rather than used to close anything (0.087 is not zero).

### Cell 1: `weather_interactions_wind_pass_heavy_fade`

| cutoff | n_flag / n_narrow | flag cover / complement cover | gap vs population (week-blocked) | P+ | gap vs narrow control | odd / even seasons P+ |
|---|---|---|---|---:|---|---|
| pool_decision (playable) | 160 / 558 | 45.00% / 50.10% | +5.096 pts, 95% [-1.730, +11.954] | 0.9276 | +6.434 pts [-2.343, +15.003] | 0.914 (n=83) / 0.746 (n=77) |
| kickoff_nearest (secondary) | 154 / 540 | 46.10% / 50.07% | +3.968 pts, 95% [-2.888, +10.730] | 0.8744 | +5.007 pts [-3.644, +13.619] | 0.871 (n=81) / 0.664 (n=73) |

Both cutoffs lean the predicted (fade) direction strongly, both intervals
cross zero. This is the strongest of the three team-style-trait cells at
the screen level. **Production (opener grade, window [2020, 2021], active
model `d49194e04945a5e5`):** 15 fade-home + 19 fade-away flagged games, 16
picks actually changed. Candidate 53.73% vs. baseline 53.29%, **+0.439
accuracy points**, week-blocked 95% [-1.101, +2.155], `probability_positive`
**0.698**. Season-blocked reads [+0.424, +0.455], P+ 1.0 -- a degenerate
artifact of only 2 season blocks in a 2-season window (matches this
project's own documented caution about season-blocked CIs on short
windows), not a second, stronger confirmation; read the week-blocked number.

### Cell 2: `weather_interactions_wind_fg_reliant_fade`

| cutoff | n_flag / n_narrow | flag cover / complement cover | gap vs population (week-blocked) | P+ | gap vs narrow control | odd / even seasons P+ |
|---|---|---|---|---:|---|---|
| pool_decision (playable) | 126 / 592 | 52.38% / 49.96% | -2.417 pts, 95% [-10.311, +5.450] | 0.2732 | -2.888 pts [-12.322, +6.672] | 0.387 (n=58) / 0.294 (n=68) |
| kickoff_nearest (secondary) | 117 / 577 | 51.28% / 49.98% | -1.300 pts, 95% [-9.374, +6.794] | 0.3716 | -1.542 pts [-11.064, +8.154] | 0.443 (n=55) / 0.385 (n=62) |

Both cutoffs lean AGAINST the predicted fade direction, and both halves of
the odd/even split agree with each other (both lean negative), unlike cell
3 below. Neither interval sits entirely on the wrong side of zero (the
upper bound is comfortably positive in every read), so this is not a
resolved wrong sign -- `unresolved_below_power`, leaning toward "wind does
not fade the kick-reliant team," not proof of it. **Production:** 2
fade-home + 7 fade-away flagged games, 8 picks changed. Candidate 53.29% vs.
baseline 53.29%, **effect exactly 0.0 accuracy points** (a coin-flip result
at this window's tiny n), week-blocked P+ **0.488**, season-blocked P+
0.501.

### Cell 3: `weather_interactions_heat_pace_fade`

| cutoff | n_flag / n_narrow | flag cover / complement cover | gap vs population (week-blocked) | P+ | gap vs narrow control | odd / even seasons P+ |
|---|---|---|---|---:|---|---|
| pool_decision (playable) | 64 / 212 | 54.69% / 49.96% | -4.723 pts, 95% [-14.506, +4.581] | 0.1574 | -6.103 pts [-18.324, +5.986] | 0.051 (n=29) / 0.589 (n=35) |
| kickoff_nearest (secondary) | 61 / 211 | 54.10% / 49.97% | -4.128 pts, 95% [-14.292, +5.591] | 0.1988 | -5.283 pts [-18.004, +6.786] | 0.051 (n=27) / 0.688 (n=34) |

The screen-level read leans AGAINST the predeclared FADE direction (i.e.,
leans toward the opposite, backing pace in heat) at both cutoffs, and the
odd/even split is highly inconsistent (odd P+ 0.051, even P+ 0.589-0.688) --
real heterogeneity, disclosed rather than averaged away, and this alone is
a reason for caution about the trait's stability within this specific
interaction, not grounds to close it (neither half's interval sits wholly
on one side, and the trait's own year-over-year reliability, 0.523, is not
zero). **Production:** 7 fade-home + 5 fade-away flagged games, 5 picks
changed. Candidate 53.95% vs. baseline 53.29%, **+0.658 accuracy points**,
week-blocked 95% [-0.229, +1.895], `probability_positive` **0.883**. This
on-production read leans FOR the fade direction even though the broader
screen-level population leaned against it -- a real divergence between the
full 17-season population and this narrow 2020-2021, model-stacked window,
reported as measured rather than reconciled by picking one.

### Cell 4: `weather_interactions_surface_switch_rush_fade`

**Measured**, `screen-surface`: 1,215 REG games have a grass-to-turf
surface-switch flag on the away team (2009-2025); 265 of those also have the
away team in the run-heaviest prior-season quartile (the compound flag), 806
are surface-switch games where the away team is NOT run-heavy (narrow
control). Cover rate: flagged (fade) games 54.23% home cover vs. 48.56% in
the full population vs. 51.84% in the narrow control.

- Gap vs. population (week-blocked): **+5.673 accuracy points**, 95%
  [-0.709, +11.962], `probability_positive` **0.9594**; season-blocked
  +5.673, [-2.940, +12.612], P+ 0.9078.
- Gap vs. narrow control (week-blocked): +2.388 pts, [-4.613, +9.447], P+
  0.7468 -- most of the raw population-level gap is shared with the
  surface-switch main effect (`surface_switch_tilt_overlay`'s own
  registered lead); the run-heavy interaction narrows it but does not
  reverse it.
- Odd seasons: +12.110 pts (n=118), P+ **0.9947**. Even seasons: +0.167 pts
  (n=142), P+ 0.5109. A large odd/even split -- the strongest single read in
  this whole battery lives entirely in odd-numbered seasons, disclosed
  rather than smoothed over.
- Trait reliability (prior-season pass rate, reused from cell 1): 0.3929,
  512 pairs.

This is the strongest screen-level read in the family (P+ 0.96 vs.
population), but per the taxonomy above a P+ this high on an interval that
still crosses zero is unresolved, not resolved -- the pool-relevant next
step is whether it survives on production. **Production (opener grade,
window [2020, 2021]):** 20 flagged games in the window, 17 picks changed.
Candidate 53.07% vs. baseline 53.29%, **-0.219 accuracy points**,
week-blocked 95% [-1.948, +1.376], `probability_positive` **0.408**;
season-blocked [-0.909, +0.424], P+ 0.247. The on-production read leans
slightly AGAINST the screen-level lean -- reported as measured, a real
divergence between the 17-season screen population and the narrow
2020-2021 stacked-on-production window, same pattern as cell 3.

### What this does and does not establish

Every interval above crosses zero. Per the binding taxonomy, none of that is
grounds to close any of the four cells: no interval sits entirely on the
wrong side of zero (cell 2 and cell 3 lean negative but their upper bounds
sit comfortably above zero), no trait has zero split-half reliability
(weakest is FG-attempt rate at 0.087, still nonzero), and no positive
control was run this session. All eight production/screen-level readings
(four cells, screen + production) are recorded `unresolved_below_power`.
Two honest, reportable patterns: (1) cell 1 (wind x pass-heavy) and cell 4
(surface switch x run-heavy) lean consistently toward their predicted fade
direction at every cut (both forecast cutoffs, both narrow and population
controls, and — for cell 1 — the production screen too), while cell 2 (wind
x FG-reliant) leans mildly against its predicted direction at every cut with
no inconsistency between odd/even; (3) cell 3 (heat x pace) and cell 4's
odd/even split both show real within-family heterogeneity (a sign flip
across season halves or across screen-vs-production populations) that is
itself worth tracking rather than a single number.

### Files

- `docs/weather_interactions.md` -- this doc.
- `scripts/weather_interactions_screen.py` -- `screen` (cells 1-3, both
  cutoffs), `screen-surface` (cell 4), `production --cell {wind_pass_heavy,
  wind_fg_reliant,heat_pace,surface_switch_rush}`.
- `scripts/record_weather_interactions.py`,
  `scripts/record_weather_interactions_production.py` -- read the computed
  results JSON and pass every numeric field through unmodified to `nfl-ats
  weak-signals record` / `nfl-ats rotation record`.
- `artifacts/weather_interactions/screen_results.json`,
  `surface_switch_results.json`, `surface_switch_table.parquet`,
  `production_{wind_pass_heavy,wind_fg_reliant,heat_pace,surface_switch_rush}_results.json`
  and `_paired.parquet`, `record_log.txt`, `record_production_log.txt`.
- `registry/weak_signals.json` -- 28 entries under the four
  `weather_interactions_*_fade` families.
- `registry/rotation_registry.json` -- 4 families
  (`weather_interactions_{wind_pass_heavy,wind_fg_reliant,heat_pace,surface_switch_rush}_on_production`),
  each spent on window [2020, 2021], verdict `unresolved`.

### Provenance

- **measured**: every count, cover rate, bootstrap interval, and
  `probability_positive` above, from this session's script runs; the two
  forecast archives' row counts and `fetch_status` breakdowns; the three
  traits' year-over-year reliabilities; the `nfl-ats rotation
  declare`/`assign` outputs (all four families drew window [2020, 2021]);
  all 28 `weak-signals record` and 4 `rotation record` command outputs.
- **read**: `ROADMAP.md`'s ENV-05 and ENV-01 rows, `docs/forecast_weather_screen.md`,
  `docs/forecast_archive_build.md`, `docs/snow_game_home_prep.md`,
  `docs/pool_edge_plan.md`, `src/nfl_ats/experiment_runner.py` (team-long
  table builder, pass-rate/quartile/reliability helpers, forecast-weather
  cutoff conventions), `src/nfl_ats/pace_mismatch_dog_tilt_overlay.py` and
  `src/nfl_ats/surface_switch_tilt_overlay.py` (reused trait/flag builders),
  `data/processed/game_features.parquet` and `game_features_weak_stack.parquet`
  column lists, `data/pbp/raw/20260817T184927Z/season=2023` column list and
  `play_type` value counts (confirming no `field_goal_result` field exists).
- **inferred**: the "what this does and does not establish" summary is a
  reading of the measured numbers above, not a new measurement; the
  FG-attempt-based kicking-reliance proxy's football interpretation
  (attempts, not makes, signal reliance on the kicking game) is a stated
  judgment call, not a validated equivalence.

## 2026-09-11 follow-up: two cells wired as prospective challengers

The two cells above whose on-production read leaned the predicted (fade)
direction -- cell 1 (`weather_interactions_wind_pass_heavy_fade`, +0.439
accuracy points, 16 of 456 picks changed, `probability_positive` 0.698) and
cell 3 (`weather_interactions_heat_pace_fade`, +0.658 pts, 5 picks changed,
`probability_positive` 0.883) -- are wired as no-window-cost, dual-tracked
prospective challengers: `forecast_weather_kn_wind_pass_heavy_fade` and
`forecast_weather_kn_heat_pace_fade` in `artifacts/prospective/challengers.json`,
both `ACTIVE_PROSPECTIVE`. Per the binding taxonomy restated at the top of
this doc, expected value favours playing both (`probability_positive` above
0.5 for each) even though nothing is resolved -- both intervals cross zero,
and this is exactly the case a promotion bar must not veto.

**Wiring mirrors the ENV-01 precedent exactly.** Each challenger is a
post-prediction, parameter-free pick transform over the active model's own
weekly card (`nfl_ats.forecast_weather_kn_wind_pass_heavy_fade_overlay`,
`nfl_ats.forecast_weather_kn_heat_pace_fade_overlay`), never a retrained
model -- the same shape as `forecast_weather_kn_warm_team_cold_late_tilt`
and `forecast_weather_kn_precip_high_total_tilt`. Both are team-long framed
(fade whichever team the model's own pick is already on, home or away, no
away-favorite restriction), matching this doc's own cell design rather than
ENV-01's away-favorite-restricted `wind_passing_away_favorite` cell.

- **Wind cell.** Flag: outdoor AND live kickoff-nearest (`pool_decision`
  cutoff, model=GFS) forecast wind >= 15mph AND the model's picked team's
  own prior-season pass-attempt-rate quartile == 4 (most pass-heavy, raw
  PBP, `posteam pass_attempt/(pass_attempt+rush_attempt)`, reusing
  `nfl_ats.experiment_runner._team_season_pass_rate` /
  `_lag_and_quartile` unchanged). The trait is built fresh at record time
  from the newest raw PBP snapshot, wrapped fail-open (a missing/broken
  snapshot degrades to zero flags with a warning, never an error).
- **Heat cell.** Flag: outdoor AND live kickoff-nearest forecast
  temperature >= 85F AND the model's picked team's own prior-season pace
  quartile == 1 (fastest tempo, `data/pbp/team_style/team_season_style.parquet`'s
  `seconds_per_play_pace`, reusing the same cache
  `pace_mismatch_dog_tilt_overlay` already reads in production), also
  fail-open.
- **Shared live fetch, extended, not duplicated.** Both reuse
  `nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.fetch_shared_kickoff_nearest_forecasts_fail_open`
  -- the SAME weekly Iowa Environmental Mesonet GFS-MOS bulletin fetch
  `forecast_weather_kn_warm_team_cold_late_tilt` and
  `forecast_weather_kn_precip_high_total_tilt` already make. The heat cell
  needed no change (that fetch already carries `forecast_temp_f`). The
  wind cell needed a `forecast_wind_mph` field, added to
  `fetch_one_game_kickoff_nearest`'s return on every code path (parsed
  from the same MOS bulletin response's `wsp` field, knots-converted with
  the identical `KNOTS_TO_MPH = 1.15078` constant `scripts/ingest_forecast_archive.py`
  uses) -- zero additional network calls.
- **Registration.** `artifacts/prospective/challengers.json`, both
  `ACTIVE_PROSPECTIVE`, config fingerprint `bc77638d47e2748c` (the current
  active model `d49194e04945a5e5`'s own configuration, unchanged since the
  ENV-01 registrations). Display names added to `CHALLENGER_DISPLAY_NAMES`
  in `src/nfl_ats/dashboard/findings_content.py`: "Wind fades the passing
  team", "Heat fades the fast team".
- **Wired into publishing.py** the same way as the two precedent
  challengers: imports, `PUBLISH_CHALLENGER_RESULT_KEYS` entries, a
  try/except recorder call inside `orchestrate_publish_predictions`'s
  `if request.record_decisions:` branch (reusing the same
  `shared_kn_forecasts` object the precedent challengers already fetch),
  and a `--record-decisions`-not-passed skip placeholder in the `else:`
  branch.

**Confirmed on 2026 Week 1** (measured this session, active model
`d49194e04945a5e5`, 16-game card, without touching the served card): both
challengers recorded cleanly against the live card -- 14 pre-kickoff
decisions each, 2 already-past-kickoff games skipped, no errors. **0 flips
for either challenger this week.** The live MOS fetch returned
`fetch_status=ok` for only 1 of the 16 games (14 `transport_error`, 1
`unmappable_international_stadium`); a direct single-request probe of the
same API succeeded immediately, so this reads as a transient/rate-limited
run rather than a blocked source, and even that one successfully-fetched
game did not cross either the 15mph wind or 85F heat threshold. The
FAIL-OPEN design handled the mostly-failed fetch exactly as intended --
zero flags, no error, nothing recorded incorrectly.

Provenance: **measured** -- the wiring, the `lockday_contract.audit()`
static check (0 errors, both challengers dispatch on the `publish` path),
`ruff format`/`ruff check`/`mypy src`, the filtered pytest run (478 passed),
and this session's live confirmation run and its fetch-status counts, all
from this session. **read** -- `docs/weather_interactions.md`'s own cells 1
and 3 above, `artifacts/prospective/challengers.json`'s existing
`forecast_weather_kn_warm_team_cold_late_tilt` /
`forecast_weather_kn_precip_high_total_tilt` entries as the wiring
template. **inferred** -- the "transient/rate-limited, not blocked" read on
this run's mostly-failed live fetch, from one successful direct probe of
the same endpoint; not independently re-verified across multiple runs.

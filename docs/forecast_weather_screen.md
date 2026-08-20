# ENV-01 payoff screen: forecast-vs-actual weather battery (predeclaration)

Written **before** `scripts/nfl_forecast_weather_screen.py` scores anything.
This is the ENV-01 "payoff screen" the sourcing/build docs deferred: the
actual-weather battery cells (`weather_battery_*`, `weather_followup_*`,
predeclared in `docs/weather_followup.md`) were **explicitly recorded as
upper bounds** on any forecast-time feature (see each cell's
`description`/`classification_evidence` field in `registry/weak_signals.json`
-- "actual-weather mechanism screen, NOT pregame-available; upper bound for a
forecast-time feature"). This doc predeclares the re-measurement of the four
strongest of those mechanisms with the **Tuesday-noon GFS-MOS forecast**
substituted for the game-time actual, on the population that forecast now
covers.

## Binding closing-grounds taxonomy (governs every verdict from this screen)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign.

## What this screen is, and is not

This is a **timing re-screen, not an independent discovery**. All four cells
below test the SAME mechanisms already predeclared and recorded in
`docs/weather_followup.md` / `scripts/nfl_weather_battery_screen.py` --
only the information-timing changes (Tuesday-noon forecast instead of
game-time actual). A future pool across the weak-signal registry must not
treat a `forecast_weather_*` cell and its `weather_battery_*`/
`weather_followup_*` sibling as independent evidence points: they overlap in
mechanism and (for 2020-2025) in population.

## Data source (read from `docs/forecast_archive_build.md` and measured this
session against the parquet directly)

- `data/raw/forecast_archive/full_2020_2025/forecasts.parquet`: the
  `tuesday_noon`-cutoff GFS-MOS (model=MEX) archive, one row per REG game,
  2020-2025. **Measured** directly this session: 1,615 rows, `fetch_status`
  `ok` for 1,598 (domestic), `unmappable_international_stadium` for 17;
  `forecast_temp_f`/`forecast_wind_mph` null only on those 17 international
  rows (100% coverage of domestic games). Columns used:
  `forecast_temp_f`, `forecast_wind_mph` (already knots->mph converted),
  `fetch_status`, joined to schedules on `game_id`.
- ~~The pool's picks lock **Tuesday at 12:00** (`docs/pool_edge_plan.md` line
  80), essentially the opener -- so `tuesday_noon` is the correct, and only,
  cutoff mode for a pool-playable feature; `kickoff_nearest` (also built,
  2024 + a 2020-2023 spot check) is NOT pool-playable and is out of scope
  for this screen.~~ **Owner-corrected 2026-08-20:** wrong. Only the pool's
  LINE locks Tuesday at 12:00; our picks are editable up to each game's real
  deadline (**refined 2026-08-20: min(kickoff, Sunday 16:00 ET) -- SNF/MNF
  lock early at Sunday 4pm**) (`docs/pool_edge_plan.md`). `tuesday_noon` is used in this screen
  because it mirrors the grading line's own information set (a legitimate
  reason to screen it), not because it is the only pool-playable cutoff --
  `kickoff_nearest` is, if anything, the MORE pool-playable construction for
  a late-week-refreshed pick, since it validates far tighter against actuals
  (temp r=0.972 vs. 0.897, `docs/forecast_archive_build.md`) and is not
  ruled out of scope by anything about pick timing. This screen's four
  cells were scored against `tuesday_noon` only and were not re-run against
  `kickoff_nearest`; that is a real gap left by this correction, not
  resolved here.
- **Narrower window than the actual-weather originals, disclosed up front**:
  this screen's population is REG **2020-2025** (archive coverage), vs. the
  2009-2025 population the `weather_battery_*`/`weather_followup_*` siblings
  were recorded on. Six seasons of week-blocks instead of seventeen -- the
  season-blocked secondary bootstrap here has far fewer blocks (n<=6) and is
  correspondingly weaker as a robustness check; this is reported, not
  hidden.

## Leakage posture -- the one thing this screen changes

Every cell below is built by taking its `weather_battery_*`/
`weather_followup_*` sibling's exact subset definition and swapping ONLY
the "this game's own weather" term from the schedules parquet's game-time
`temp`/`wind` (actual) to the forecast archive's `forecast_temp_f`/
`forecast_wind_mph` (Tuesday-noon forecast). Nothing else changes:

- `roof` (outdoor/dome/closed) and `surface` stay the schedules parquet's
  own actual-recorded values, unchanged from the parent scripts. This is a
  disclosed, precedented convention already carried by both prior batteries,
  not a new leakage source: for fixed roofs (the large majority of outdoor
  stadiums and domes) roof status is a known-in-advance stadium fact, not a
  forecast; the caveat is narrower than the temp/wind caveat it replaces and
  applies identically to a small number of retractable-roof games whose
  Tuesday-noon roof decision is not yet public. This screen does not resolve
  that narrower caveat -- it is inherited, not introduced.
- `away_modal_roof` / `away_modal_surface` (cells 2 and the surface half of
  nothing new here -- surface itself is not gapped in this battery) and the
  `climate_temp` away-team-own-climate baseline (cell 4) are same-season
  aggregates over the away team's OTHER home games' ACTUAL weather -- exactly
  the `weather_followup_screen.py` convention (not season-causal, disclosed
  there, unchanged here). Only the FOCAL game's own temp/wind term swaps to
  forecast; the climatological baseline a game is compared against is still
  built from actual weather at other games.
- `forecast_temp_f` / `forecast_wind_mph` themselves ARE genuinely
  pregame-available before kickoff (and, for this specific
  `tuesday_noon`-cutoff construction, at the pool's Tuesday-noon LINE lock
  too -- **owner-corrected 2026-08-20:** that lock constrains the grading
  line, not our own pick timing, which runs up to each game's real
  deadline, min(kickoff, Sunday 16:00 ET) -- SNF/MNF lock early at Sunday
  4pm)
  -- this is the whole
  point of the screen: these four cells, unlike their actual-weather
  siblings, are candidate POOL-PLAYABLE features, not mechanism-screen upper
  bounds, to the extent the roof/surface/climate-baseline caveats above do
  not apply.

## The 4 predeclared cells (mirrors, exact subset definitions, frozen before scoring)

All cells score `home_cover` (pushes dropped, `add_ats_outcomes`) on REG
2020-2025 games with a successful (or attempted) forecast join. Week-blocked
joint bootstrap primary (block=`season*100+week`), season-blocked secondary
(block=`season`), 20,000 samples, seed `20260819` -- same method, sample
count, and seed as both prior batteries, for direct comparability.
Full-slate-scaled effect (`raw_gap_pts * fraction_of_slate`), reused
verbatim from `scripts/nfl_weather_battery_screen.py`.

1. **`forecast_weather_high_wind_outdoor`** -- mirrors
   `weather_battery_high_wind_outdoor` (registry: +0.1585pts, 95%
   [-0.2863, +0.6037], P+ 0.7567, n=4,317 REG 2009-2025). Flag: outdoor/open
   roof AND `forecast_wind_mph >= 15`. Unsigned (no predicted direction --
   same as the sibling).

2. **`forecast_weather_dome_team_outdoors_cold`** -- mirrors
   `weather_battery_dome_team_outdoors_cold` (registry: +0.1052pts, 95%
   [-0.118, +0.3264], P+ 0.8249). Flag: away team's modal home roof this
   season is dome/closed AND this game is outdoor/open AND
   `forecast_temp_f <= 40`. **Predicted: positive home_cover edge.**

3. **`forecast_weather_warm_team_cold_late`** -- mirrors
   `weather_battery_warm_team_cold_late` (registry: +0.1576pts, 95%
   [-0.0043, +0.3094], P+ 0.9723, the strongest first-generation cell other
   than surface-switch). Flag: away team's season code in the static
   warm-winter-metro list AND outdoor AND `forecast_temp_f <= 35` AND
   `week >= 13`. **Predicted: positive home_cover edge.**

4. **`forecast_weather_temp_gap_cold_visitor`** -- mirrors
   `weather_followup_temp_gap_cold_visitor` (registry: +0.3836pts, 95%
   [+0.0017, +0.7541], P+ 0.9755 -- the strongest follow-up-battery cell, and
   the only registry sibling among these four whose recorded interval
   already excludes zero). Flag: away team's own climatological-normal
   OUTDOOR home temp this season (actual, same-season aggregate) MINUS this
   game's `forecast_temp_f` `>= 25`F, AND outdoor. **Predicted: positive
   home_cover edge.**

## Diagnostic: forecast-vs-actual flag agreement (per cell, not a registry field)

For each cell, also compute the SAME subset definition using the game's
ACTUAL temp/wind (the sibling's original construction) on the identical
2020-2025 population, then report:

- the agreement rate (fraction of rows where the forecast-based flag and the
  actual-based flag agree), restricted to rows where BOTH flags have their
  required inputs present;
- the confusion breakdown (both-true / both-false / forecast-only /
  actual-only counts);
- the same-population actual-weather effect/interval/P+ (computed here, not
  looked up from the registry) alongside the forecast-weather effect, so the
  "playable fraction" (forecast full-slate effect / actual full-slate effect
  on the IDENTICAL 2020-2025 games) is measured on matched populations, not
  confounded by the registry sibling's broader 2009-2025 window. The
  registry sibling's original 2009-2025 number is reported alongside as a
  second, disclosed-as-broader-population comparison.

This diagnostic is exploratory/descriptive, not itself recorded to the
registry as a signal (it has no predicted direction of its own -- it is an
information-timing decay measurement, not a home_cover mechanism).

## Recording commitment

Every one of the 4 cells above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` under a `forecast_weather_*` name (checked for
collisions against the current registry before this doc was written -- none
exist), `league=nfl`, `effect_units=accuracy_points`, `unresolved_below_power`
regardless of what the interval looks like, via a script that reads the
computed results JSON and passes every numeric field through unmodified (no
hand-typed numbers) -- same discipline as
`scripts/record_weather_followup.py`. The only admissible alternative
classification under AGENTS.md would be a RESOLVED wrong sign (whole
interval on the wrong side of the predicted direction, cells 2-4 only, cell
1 has no predicted direction) or a positive-control bound (not run in this
screen); if neither applies the classification is
`unresolved_below_power`, full stop -- an interval crossing zero is never
itself that ground.

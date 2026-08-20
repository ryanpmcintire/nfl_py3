# Weather follow-up battery (second generation) — predeclaration

Written **before** `scripts/nfl_weather_followup_screen.py` scores anything.
Frozen copy also at
`scratchpad/agent_weather_followup/predeclaration.json`. Per AGENTS.md, this
is a mined/exploratory screen family: every cell here is predeclared to
record `unresolved_below_power` regardless of interval shape (crossing zero
is the EXPECTED outcome for a real small signal at this evaluator's
resolution, never a rejection ground). Method, population, and leakage
posture are locked before any cell's sign is seen.

## Rank of the 8 first-generation cells (read from `registry/weak_signals.json`, recorded 2026-08-19)

Ranked by `|effect| / SE`, SE approximated from the recorded week-blocked 95%
CI half-width (`(hi-lo)/(2*1.96)`) since `standard_error` is null on every
entry — **measured** from the registry file, not re-run this session:

| rank | cell | effect (pts) | 95% CI | P+ | \|effect\|/SE |
|---|---|---|---|---|---|
| 1 | `weather_battery_surface_switch_grass_to_turf` | +1.1618 | [+0.2896, +2.038] | 0.995 | 2.61 |
| 2 | `weather_battery_warm_team_cold_late` | +0.1576 | [-0.0043, +0.3094] | 0.9723 | 1.97 |
| 3 | `weather_battery_thursday_outdoor_cold` | +0.0495 | [-0.0318, +0.121] | 0.9022 | 1.27 |
| 4 | `weather_battery_extreme_cold` | +0.1042 | [-0.1033, +0.335] | 0.8391 | 0.93 |
| 5 | `weather_battery_dome_team_outdoors_cold` | +0.1052 | [-0.118, +0.3264] | 0.8249 | 0.93 |
| 6 | `weather_battery_high_wind_road_favorite` | -0.0979 | [-0.3407, +0.1467] | 0.2176 | 0.79 |
| 7 | `weather_battery_high_wind_outdoor` | +0.1585 | [-0.2863, +0.6037] | 0.7567 | 0.70 |
| 8 | `weather_battery_high_altitude_road` | +0.0007 | [-0.2691, +0.2716] | 0.5042 | 0.005 |

`surface_switch_grass_to_turf` is the strongest cell by a wide margin (also
the project's strongest cross-league weather/venue lead, replicated on CFB
per ROADMAP ENV-02) and `warm_team_cold_late` is a clear second. This
battery deepens both: one cell compounds surface-switch with cold, and two
cells generalize the temperature-mismatch mechanism behind
`warm_team_cold_late`/`dome_team_outdoors_cold` from a raw-temperature
threshold into a **temperature/wind GAP relative to the away team's own
climate**, per the task's explicit steer away from raw-temperature
thresholds. A fifth cell tests the wind x pass-style interaction named
explicitly in the task brief.

## Data source and leakage posture (read from `scripts/nfl_weather_battery_screen.py`)

- `temp`/`wind`/`roof`/`surface` in `data/raw/<latest>/schedules.parquet` are
  **game-time actuals**, not pregame forecasts (documented leakage caveat in
  the parent script, reused verbatim here). Every cell below that touches
  this game's own `temp`/`wind`/`roof` is therefore a **mechanism screen**,
  an upper bound on a future forecast-time (ENV-01) feature -- not itself
  pregame-usable. This is the same posture as all 8 generation-1 cells.
- The away team's "own climate" gap baseline (cells 1-2 below) is computed
  the same way `away_modal_roof`/`away_modal_surface` were computed in the
  parent script: a full-season aggregate over that team's OWN home games in
  the SAME season (`groupby(["home_team","season"])`), which is not
  season-causal (it can include games later than the one being scored). This
  is a disclosed, precedented convention carried over unchanged from the
  parent script, not a new leakage source -- a genuinely pregame-safe
  version would need a prior-season or multi-year climatological baseline
  instead, and that gap is called out explicitly, not hidden.
- Cell 3 (`wind_gap_visitor`) uses the same full-season-aggregate convention
  for the away team's own wind baseline.
- Cell 4 (`high_wind_pass_heavy_visitor`) uses `data/raw/<latest>/team_stats.parquet`
  attempts/carries aggregated to team-season and shifted **one full season
  forward** (away team's PRIOR-season pass rate) -- genuinely season-causal
  and pregame-safe on its own terms (no same-season leakage), independent of
  the temp/wind leakage caveat that still applies to the wind>=15 half of
  the flag.
- Cell 5 (`surface_switch_x_outdoor_cold`) reuses `away_modal_surface`/
  `surface_norm` verbatim from the parent script (already precedented, no
  new leakage posture) plus `temp`/`roof` (same caveat as generation-1).

## The 5 predeclared cells (exact subset definitions, mechanism, predicted direction -- frozen before scoring)

All cells score `home_cover` (pushes dropped, reused from `add_ats_outcomes`)
on the same REG 2009-2025 population as the parent script, subset vs.
complement, week-blocked joint bootstrap primary (block=`season*100+week`),
season-blocked secondary (block=`season`), 20,000 samples, seed `20260819`
(same seed/sample count as the parent battery, for direct comparability),
full-slate-scaled effect (`raw_gap_pts * fraction_of_slate`), reused verbatim
from `scripts/nfl_weather_battery_screen.py`.

1. **`weather_followup_temp_gap_cold_visitor`** -- Generalizes
   `dome_team_outdoors_cold` + `warm_team_cold_late` into one continuous-gap
   mechanism instead of two raw-threshold special cases. Away team's own
   climatological-normal temp (mean actual temp across the away team's own
   OUTDOOR home games that season; teams with zero outdoor home games that
   season, e.g. full-time dome teams, get a missing baseline and are
   excluded via the missing mask, not silently zero-filled) minus this
   game's temp is `>= 25`F (this game much colder than what the away team's
   own stadium sees), AND this game is outdoor. **Predicted: positive
   home_cover edge** (visitor unaccustomed to this much cold relative to its
   own climate).

2. **`weather_followup_wind_gap_visitor`** -- Same gap construction, wind
   axis. Away team's own climatological-normal wind (mean wind across the
   away team's own outdoor home games that season) is `<= 8`mph (a
   low-wind/sheltered or dome-adjacent home climate) AND this game's wind is
   `>= 15`mph AND this game is outdoor. **Predicted: positive home_cover
   edge** (visitor unaccustomed to wind this strong relative to its own
   climate).

3. **`weather_followup_high_wind_pass_heavy_visitor`** -- Named explicitly
   in the task brief. Outdoor AND wind `>= 15`mph AND away team's PRIOR-season
   pass rate (`attempts / (attempts + carries)`, team_stats.parquet, REG
   games only) is above that PRIOR season's median across teams ("pass-heavy
   visitor", season-causal). Teams with no prior-season row (first season in
   the sample, expansion/relocation-adjacent codes) get a missing baseline,
   excluded via the missing mask. **Predicted: positive home_cover edge**
   (wind degrades a pass-heavy offense disproportionately more than a
   run-heavy one).

4. **`weather_followup_rest_disadvantage_cold`** -- Travel/fatigue x weather
   compounding, distinct from the parent battery's purely-calendar
   `thursday_outdoor_cold` cell. Away team's rest is strictly less than the
   home team's rest (`away_rest < home_rest`, both already in schedules,
   pregame-known matchup fact) AND this game is outdoor AND temp `<= 35`F.
   **Predicted: positive home_cover edge** (fatigue and cold compounding
   favor the more-rested home side).

5. **`weather_followup_surface_switch_x_outdoor_cold`** -- Direct deepening
   of the strongest generation-1 cell: tests whether the surface-switch
   mechanism (away team's modal home surface grass, this game's surface
   turf) is amplified when this game is ALSO outdoor and cold, vs. the full
   population (same subset-vs-complement convention as every other cell in
   both batteries, not vs. the surface-switch-only population). Flag =
   `away_modal_surface == grass AND surface_norm == turf AND outdoor AND
   temp <= 45`F. **Predicted: positive home_cover edge, and larger in
   magnitude than the +1.16pt parent cell** (compounding, not competing,
   mechanisms).

## Recording commitment

Every cell above records to `registry/weak_signals.json` via
`nfl-ats weak-signals record` as `unresolved_below_power`, `league=nfl`,
`effect_units=accuracy_points`, regardless of what the interval looks like,
via a script that reads the computed results JSON and passes every numeric
field through unmodified (no hand-typed numbers). The only exception
admissible under AGENTS.md would be a RESOLVED wrong sign (whole interval on
the wrong side of the predicted direction) or a positive-control bound,
neither of which this measure-only screen is designed to produce.

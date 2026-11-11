# SIM-15 Weather-conditioned play-calling

## Goal
Measure how weather changes play-calling, as a simulator mechanism. ROADMAP row SIM-15.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Wind and rain change what teams call, which changes variance and total as well as level; the simulator draws calls independent of weather. Estimate within-team elasticity of pass rate and explosive-pass share to wind and precipitation. Data: nflverse pbp with weather fields for outdoor games, 2009-2025 (local).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: sim09_personnel_catalogue.md B5 (listed only), LEAD-75 wind forecast innovation (market pricing of wind), ENV-02 (roof), forecast_weather_screen.md (weather versus cover). None measures play-calling elasticity.

## Next
Build the within-team panel and run the roof placebo. First unit: First unit: team-fixed-effect regression of pass rate and deep pass share on wind speed and precipitation indicator with a roof placebo; no engine edit.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Weather-forecast timing versus observed conditions: use observed for mechanism, forecast for any pregame term.

## Unit 1 in flight (2026-11-03)
Script scripts/sim15_weather_calls.py (ruff clean, no comments). Family sim15_weather_calls, 42 looks declared before outcomes: 7 outcomes (pass_rate, deep_share, fg_le20, fg_21_30, fg_31_40, epa_total, epa_fixed_call) x 3 specs (main_observed, outdoor_forecast, placebo_roof_forecast) x 2 terms (wind10, temp10). Offence team-season FE, game-state controls, 300 game-cluster bootstrap, LOSO per-season folds.
Data: pbp raw 20260929 (2009-2025) plus weather/roof from aware forecast archive (actual_* for observed, forecast_* for placebo and forecast spec). Air yards only exist in nflverse_pbp_full 2009-2017, so deep_share covers 2009-2017 only; FG bands by yardline_100 on 4th-down decisions.
Run `run` then `behaviour` (total minus fixed-call EPA slope = behavioural part) launched in background; outputs artifacts/sim15/{cells,loso,behaviour}.csv, run.log, beh.log.
## Next
Read the csvs, record cells with nfl-ats weak-signals record (unresolved_below_power, epa_per_play/rate_pp units), write results and engine hook spec if identified.

## Unit 1 results (2009-2025, 48 cells recorded unresolved_below_power, family sim15_weather_calls; artifacts/sim15/)
Per 10 mph wind / per 10 F, [game-cluster 95%], p+; main = observed weather outdoors, placebo = dome/closed using that city-day outdoor forecast.
- Pass rate: wind -0.91 pp [-1.29,-0.43] p+ 0.00; temp +0.36 pp [0.25,0.49]. Placebo wind -0.17 pp [-0.63,0.35] p+ .30; temp +0.05 pp [-0.11,0.19]. LOSO wind negative 17/17 (fold sd 0.07 pp).
- Deep share (2009-2017 only, air yards): wind -0.49 pp [-0.90,-0.04] p+ .007; temp -0.05 [-0.20,0.07]; placebo wind -0.22 [-0.89,0.39]. LOSO wind negative 9/9.
- FG attempt per 4th-down decision, wind: kick ~37 or less -0.27 pp [-1.7,1.1]; 38-47 -1.78 pp [-3.8,-0.08]; 48-57 -6.94 pp [-9.3,-4.8]. Temp +0.76, +1.35, +3.18 pp (cold cuts attempts). Placebos wind +1.8/+0.9/+2.1 pp with intervals spanning 0; temp placebos near 0. LOSO 48-57 wind negative 17/17.
- EPA per play wind: total -0.0171 [-0.0274,-0.0071]; fixed-call -0.0164; behavioural (total minus fixed-call) -0.00065 [-0.00104,-0.00027]; temp behavioural +0.00023 [0.00011,0.00034]. Placebo wind total +0.003 [-0.016,0.019]. Temp EPA placebo +0.005 equals main, so the temperature EPA level is not identified (within-season calendar confound); only the wind EPA level and the call-shift contrasts are clean.
- Looks: 42 declared plus 6 behavioural contrasts = 48.
Read: calls shift identified for wind (fewer passes, fewer deep, fewer long field goal tries) and cold (fewer passes, fewer long FG tries), placebo flat. The shift is slightly costly: -0.0007 EPA/play per 10 mph, about 4 percent of the wind EPA drop; weather's direct effect on outcomes is nearly all of it. So the behavioural term is small; the level effect is the mechanism.

## Engine hook spec (not applied)
Where: sim10_drive.py Pool.draw (line 1200), the neighbour kernel pick; add observed wind w (mph) per game as a kernel feature or a log-odds shift on pass vs run neighbours: logit shift = beta*(w/10) with LOSO beta on pass rate -0.0091 pp/0.5 ~ about -0.036 logit per 10 mph (convert from -0.91 pp at base pass rate ~0.58: -0.0091/(0.58*0.42)); deep: air_yards>=20 share -0.49 pp per 10 mph on the pass neighbours; 4th-down FG decision (fgdec_fit, line 2405) add wind10 and temp10 terms with kick-band slopes above. Check against outdoor-windy games total points; no compensating constants.
## Next
Engine owner decides; drive-level check of simulated totals in wind > 15 mph. Possible: precipitation (not in data).

## Unit 2 weather join (2026-11-03 16:50, no engine edit)
Table: artifacts/sim10/real_games.parquet (sim10_drive.py:338 writes, :1613 and :2333 read; 2304 games, 2009-2017, unique game_id, no weather column). New file beside it: artifacts/sim10/real_games_weather.parquet (2304 rows; game_id, roof_state, indoor, outdoor, temp_f, wind_mph [observed], forecast_temp_f, forecast_wind_mph, forecast_precip_prob_pct, fetch_status). Built by scripts/sim15_weather_join.py from the same forecast archive as unit 1 (ruff clean). Existing table untouched.
Coverage: observed wind/temp 0.715-0.750 of all games per season (rest are domes/closed); outdoors 1686/1686 observed; roof open 43 games have no observed wind (2.5% of 1729 outdoor+open games missing wind; forecast covers them). Forecast wind 0.980-0.992 by season. Precipitation: only forecast probability (outdoor 98.8% non-null); no observed precipitation exists.
Consistency: all 2304 ids identical to unit 1 values (wind, temp, forecast wind, forecast temp).
Join key: game_id (e.g. 2009_01_TEN_PIT) = real_games.game_id.
Engine sites (read, sim10_drive.py): Pool.draw line 1206 (neighbour pick, pass/run log-odds shift); fg decision shift at 1266-1269 (cf_ per regime from fgdec, fgdec_fit line 2435); real_schedules 1613 and cmd_real 2333 are where real games load. Unit 1 lane cited 1200/2405; current lines are above.
Measured effects the hook would use (per 10 mph wind, unit 1): pass rate -0.91 pp [-1.29,-0.43]; deep share -0.49 pp [-0.90,-0.04] (2009-2017); FG attempts 48-57 yd -6.94 pp [-9.3,-4.8], 38-47 -1.78 pp; temp per 10 F pass +0.36 pp. Observed weather for mechanism, forecast for any pregame term.
## Next
Engine owner wires wind_mph per game into the sim; the Tier B chain must finish reading real_games.parquet first.
Recorded in ROADMAP 2026-11-11

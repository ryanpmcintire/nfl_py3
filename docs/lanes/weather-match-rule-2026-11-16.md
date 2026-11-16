# Weather match rule (2026-11-16)

## Goal
Derive the MOS projection-to-kickoff match rule for the v4 weather feature from measured error, replacing the underived 3 h `GFS_MOS_PROJECTION_STEP` window (also misapplied to 12-hourly MEX). Measurement only: nothing served changes and no picks are graded. The rule is for the post-season pinned batch.

## State (done)
- scripts/weather_match_rule.py: `fetch` (IEM full bulletins: GFS and MEX at each game's deadline issuance, MEX at Tuesday 00Z; 2975 open-air 2009-25 games, 4808 ok, 4037 MEX 404 because the IEM MEX archive starts in 2020), `analyze`, `live --schedules <parquet> --cut-h 5`. ruff clean. Outputs in artifacts/weather_match_rule/: report.json, gap_curve.csv, nearest_gap_curve.csv, per_bulletin.parquet, live_2026.csv/.json, bulletins/.
- Baseline: climatology = leave-one-season-out station-month mean temp and station mean wind; MAE 7.16 F and 3.70 mph.
- GFS (deadline issuance): nearest-projection MAE beats climatology at gaps under 5 h and loses from 5 h (unbracketed rows: 4-5 h temp 5.36 vs clim 8.01, wind 3.46 vs 4.11, n=97; 5-6 h temp 6.58 vs 6.28, wind 3.64 vs 3.28, n=96). At 1-3 day lead the wind curve first loses at 3 h (4.13 vs 3.54, n=96), then wins again at 4 h, so the wind result is marginal.
- Interpolation vs nearest, GFS 3 h bracket (n=2085): temp -0.029 F [-0.067, +0.011], P(neg) 0.92; wind -0.089 mph [-0.115, -0.067], P(neg) 1.00 (season bootstrap).
- MEX: wind is worse than climatology at every gap and lead (Tuesday, 0-1 h gap: 5.0/6.2 vs 3.7/3.8 mph). Tuesday MEX temp: nearest beats climatology under 5 h and loses at 5 h (10.89 vs 5.99, n=203). For 1 PM ET games, linear interpolation across the 12 h bracket gives 8.14 vs clim 5.99 in EDT and 7.22 vs 7.96 in EST, so widening the MEX window for 1 PM games is not supported. In the 2-4 h gap bins, interpolation is worse than nearest (diurnal curvature).
- Looks: 13 one-hour gap bins x 6 model-lead cells x 2 variables on the curve, plus 3 x 2 interpolation comparisons.

## Tried
- The archives store one projection per game, so the gap curve was measured from refetched full bulletins (archive values reproduce: 2975/2975 GFS temps equal).

## Next (proposed rule for the post-season pinned batch)
1. Name the cut `MOS_FORECAST_SKILL_HORIZON_H = 5`: the first projection gap where the measured MOS error is no longer below LOSO climatology (GFS temp and wind at ≤1 d lead, MEX temp). Use a strict `gap < 5 h`.
2. Temperature: newest GFS bulletin before the cutoff. When kickoff is bracketed, linearly interpolate between the bracketing 3-hourly projections. Otherwise use the nearest projection if the gap is under 5 h. Otherwise use the MEX nearest projection under 5 h, without interpolation. Otherwise NaN.
3. Wind: GFS only, with the same interpolation and cut. Never take wind from MEX.
4. Apply the same rule to training rows (today the training rows take the archive's nearest projection with no window: 194 of 2975 rows are more than 3 h from kickoff).
- Change counts: training keeps 2781 rows under 3 h and 2878 under 5 h. 97 rows in 3-5 h are gained and 97 at or beyond 5 h are dropped, against today's no-window training. Interpolation changes values on 1127 of 2085 bracketed rows. Live 2026 (179 open-air rows): temp coverage is 102 under both rules, 36 values change, none gained or lost; wind loses 3 (the MEX-wind rows). On played games covered by both rules (n=41): temp MAE 2.44 -> 2.43, wind 2.86 -> 2.75 mph.

## Open
- The MEX daytime max (`n_x`) is a candidate mechanism for 1 PM ET games when only Tuesday MEX exists. Not measured.
- The cause of the MEX wind error (quantity definition or bias) is not diagnosed. Do not offset it; investigate the definition.

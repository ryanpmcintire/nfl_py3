# live-weather-features

## Goal
Decide if 100% missing temp/wind on 2026 live cards (vs ~35% domes in history) skews picks, and the pregame-safe fix.

## State
Answered, wk6 only. Served ridge uses temp and wind (constants.py:319-320, 353-354 "context" family; in every weak_stack set). Imputer is SimpleImputer(median, add_indicator=True) (margin.py:603): live outdoor games get median temp 60 / wind 8 plus missing=1 (the dome signature). Coefs tiny: temp -0.125 pts per SD (13.9F), wind -0.049 per SD (4.2mph), indicators 0.007; 65 of 90 features larger. Wk6 counterfactual with pregame GFS/MEX MOS (9 of 50 outdoor rows had a bulletin, i.e. 9 of ~10 outdoor games): mean |shift| 0.057 pts, max 0.264 pts of 12.6 pt residual sd, zero side changes. pick_probability.py hard-codes flag_precip=0 (:630); active_pick_probability.json has no weather term. Scratch: tests/scratch/live_weather/wk6n.py.
## Tried
Forecast sources: data/raw/forecast_archive/mos_bulletins/{GFS,MEX}/<ICAO>/ (cached_at before deadline) is usable; paired_wind_forecast tuesday dir is empty, deadline captures partial/late and wind-only.
## Next
Wire pregame MOS temp/wind (outdoors only) into live feature build; wks 4-5 not run.
Deferred to after Week 18 (orchestrator 2026-10-19): the fix edits features.py, a PINNED_NAMES file in src/nfl_ats/independent_validation.py, so editing it mid-season breaks the v3 validation cohort; measured effect is at most 0.26 pts and zero side changes. Batch with the other post-Week-18 edits, or with a forecast-trained weak_stack_v4 if that is enrolled as v4.
## Open
Training temp/wind are observed (nflverse) -> train/serve mismatch beyond missingness; forecast-trained variant is weak_stack_v4 (forecast_weather family).

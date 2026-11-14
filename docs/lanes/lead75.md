# LEAD-75 - prospective paired wind forecasts

## Goal
Collect comparable Tuesday/deadline forecasts (`scripts/capture_paired_wind_forecast.py`, `paired_wind_*` jobs in `scripts/capture_scheduler.py`); keep the forward log leak-free. Declaration: `docs/forward_pooling_declaration_2026-11-03.md`. Protocol: Tuesday 12:05-12:20 ET leg, deadline leg in final 90 min before `pool_decision_cutoff`, same product/station/kickoff/valid-time key, MEX and GFS separate, no catch-up, outcome looks 0 at collection.

## State (audit 2026-11-14, read-only)
Inputs per game (read): (1) Tuesday wind forecast, product MEX or GFS, IEM MOS, `issuance_runtime_utc`, `captured_at_utc`, `request_started_at_utc`, sha256 in `data/raw/paired_wind_forecast/<Tue>/tuesday/<run>/forecasts.parquet`; (2) deadline forecast, same columns under `.../deadline/<run>/`; (3) pass-rate: `diff_pbp_off_pass_rate` = home minus away `pbp_off_pass_rate` EWM state (span 8, min 3, offseason retention 0.67) in `data/processed/game_features_pbp.parquet`, built by `enrich_with_pbp_features` in `src/nfl_ats/pbp.py:768-828`. The collector itself stores no pass-rate; the interim look script is not in the repo (only the lane text). Pass-rate lookup (pbp.py:801-818) takes the last team state with `gameday` strictly before the game date (`searchsorted side=left`, minus 1), so own and later plays are excluded by construction (read). Outcome input: home ATS margin from game_features (settled only).
Capture (measured, scheduler_log + dirs): Tuesday leg exists with parquet for 10-20, 10-27, 11-03 (RUN/OK 12:05 ET, 30 rows, dir 20261103T170513Z), 11-10. Lost: 09-29 (job added that evening), 10-06 and 10-13 (atomic_bytes error, empty dirs). The 11-03 leg was NOT missed; the "MISSED paired_wind_tue 11-03" lines at 11-04 08:36 and 08:53 are catch-up notices after the OK. Deadline legs run OK now (11-08 sun, 11-12 thu, 11-13 fri, 11-14 sat all OK or in-window SKIP). Earlier FAILs 10-01..10-18 and 11-01 11:45 (timeout; retry fix since added).
Leak check: `tests/scratch/lead75/leak_check.py` launched (PID 28388, 53 forward games on 5 truncation dates, truncating pbp to games dated before each date and rebuilding vs stored); output to `artifacts/lead75/leak_check.json` and `tests/scratch/lead75/run.log` - FINISHED (measured): 53 games, largest absolute difference 0.0 on home, away and diff pass-rate after removing each game's own and all later plays; full rebuild vs stored also 0.0; min gap from prior game 6 days, 0 prior games in same or later week. No leak. Pairs (measured, ok, captured before kickoff, issuance ordered): 39 (wk7 14, wk8 8, wk9 15, wk10 2 so far; 36 MEX, 4 GFS), outdoor-mapped only.

## Tried
Read of builder and logs only; no capture jobs, no edits to src/scripts.

## Next
Read `artifacts/lead75/leak_check.json` (largest_abs_difference_truncated_vs_stored must be 0 or float noise; full_rebuild_vs_stored shows reproducibility); compute usable pair count from the parquets (Tuesday and deadline rows sharing pair_key, ok, captured before kickoff) and the earliest look date from declaration power (~23 weeks at 7 pairs a week from the interim, so not before the end of the 2026 season); confirm PID 28388 exited.

## Open
Interim look script and its pass-rate definition are unversioned; Tuesday-leg weeks 4-6 unrecoverable. Stale MISSED lines in scheduler log after a successful run can mislead audits.

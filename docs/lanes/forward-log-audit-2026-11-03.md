# Forward-log audit 2026-11-03

## Goal
Confirm every forward-only ROADMAP row is accumulating its 2026 record. Read-only; root dispatches fixes.

## State (measured 2026-11-03 07:50 ET, scheduler_log last 2 weeks: each writer job OK=2, MISSED=1 from the 10-25..10-29 daemon outage, run after catch-up)
HEALTHY
- POL-10, POL-11, MKT-08: artifacts/prospective/*.parquet have Weeks 1-8 (challenger_decisions 549-706/wk; refresh/consensus/crew/inactives all weeks). half_line only Weeks 1-3 (retired).
- LEAD-53: best_pick_refresh_decisions.parquet has Weeks 1,3-8. Week 2 lost (recorder failure, history in lead53 lane); not recoverable.
- LEAD-29: interim_playcaller_first_game_back_overlay in challenger_decisions Weeks 1-8 (Wk1 14 rows not 16).
- LEAD-65: pbp08_protection_mismatch_early_window_v1 Weeks 2-8 (Week 1 never recorded, lost; gate is week<=4 so done after Wk4).
- MOD-07: weak_stack_deadline_drag / expected_lineup_loss Weeks 1-8 (14-16/wk).
- MOD-23: artifacts/armb_forward/runs Wk7 x6, Wk8 x2 (Thu/Sun jobs OK 2, MISSED 1).
- MOD-24: artifacts/mod24_forward/runs Weeks 4-8 each Thu+Sun (started Wk4). MOD-25: artifacts/mod25_forward/runs Weeks 4-8 Thu+Sun; mod25_wfj_forward Week 9 started 11-03.
- LEAD-75: data/raw/paired_wind_forecast dirs 09-29..10-27 (Wks 4-8); paired_wind_* jobs OK, one MISSED per job; one FAIL(1) paired_wind_sun_1145 11-01 TimeoutError (other Sunday runs OK). 11-03 dir not yet present (jobs later today).
- MOD-27: artifacts/pool_relative_forward_log/log.parquet Week 8 = 14 rows (backfill). 8 pool_move_log_* jobs exist; Week 9 first live.
GAP
- LEAD-99: artifacts/lead99_llm_read/runs has Wk7 Tue + 2 deadline, Wk8 deadline x4 (10-29, 11-01 x3). Week 8 Tuesday view LOST: 10-27 13:30 FAIL "no captured pool week has a game still to kick off" because splash_board_tue FAILED 10-27 (BLOCKED weekly_lock 14:16-14:19), so no Week 8 pool snapshot existed and the newest capture (Wk7) was fully kicked off. The 10-31 MISSED line is the same window replayed. Not recoverable (post-kickoff view is not blind). Week 9 Tuesday run is today 13:30: depends on splash_board_tue + weekly_lock OK at ~12:20.
- MOD-23/draftcap (armb sibling): artifacts/draftcap_forward/runs Week 7 only (3 files 10-18). Week 8 FAIL 10-29 19:15 and 11-01 11:15 "draft capital arm coefficients differ from the enrolled frozen set"; manual run 11-02 10:50 OK but wrote nothing for Wk8. Week 8 lost; MOD-27 item 13 says float-noise hash gate repaired, so Week 9 should work: verify Thursday 11-05 19:15 run.
BROKEN
- independent_validation_*: FAIL(2) x6 "Independent-validation source changed; do not mix model versions" (10-28..11-02, daily). Not one of the 18 rows but the same class; root decide.
NO-WRITER
- LEAD-64: no scheduler job; last artifacts/injury_headline_designations run 20260924T183645Z. Nothing will run it Friday 11-06. Fix: add Job (after nflverse_injuries_fri_pm) argv [UV,run,--no-sync,python,scripts/lead64_freshness_audit.py,--as-of,<date>] (required arg --as-of; needs a date template) or run by hand Fri 11-06 pm; scripts/friday_designation_momentum_screen.py is the other tool. Lane lead64 judged unresolved_below_power (n=3), so low priority.
- LEAD-80: data/pool_observables has only manifest.json/observations.json, no entrant tiebreaker guesses; capture is manual lock step 1c in docs/lanes/week9-lock-2026-11-03.md (data/splash/2026_week09_field_*.json + scripts/validate_pool_field_capture.py). No job in capture_scheduler.py. Fix: do step 1c today before lock; ongoing weeks need the same manual step or a job.
- LEAD-81: scripts/capture_travel_interruptions.py exists, one capture data/raw/travel_interruptions/20261102T215836Z, zero jobs in scripts/capture_scheduler.py. Fix: week9 lane step 3b (travel_interruptions_<tue..sun> 08:00 + sun_1000, timeout 120, argv [UV,run,--no-sync,python,scripts/capture_travel_interruptions.py]).
- ENV-02 surface term: artifacts/forward_terms/env02/frozen_coefficients.json exists, but the forward_terms Tuesday Job is not in scripts/capture_scheduler.py (0 hits). Fix: add Job entry from docs/lanes/forward-terms-2026-11-03.md "Scheduler Job entry" after weekly_lock. Weeks 1-8 not recorded and were deliberately deleted (not forward); Week 9 is first.
- SKY-01: artifacts/forward_terms has sky03, sky04, env02 only; nothing writes a sky01 forward score; earliest score Week 9 (after 11-10). Fix: extend scripts/forward_terms_week.py with sky01 frozen coefficients (artifacts/sky01), or score post hoc from timestamped data/raw/injury_news + beat_reporters captures (recoverable as long as the coefficients are frozen before Week 9 kickoffs).
- MOD-27 sky03/sky04 forward share the same forward_terms Job gap.

## Tried
Scheduler counts via grep of data/scheduler_log.txt 10-20..11-03; week counts from parquet/dir listings; no edits besides this lane.

## Next
Root: add forward_terms + travel_interruptions jobs post-lock (week9 lane 3b); do step 1c; confirm lead99 Tue 13:30 and draftcap Thu 19:15 land; decide LEAD-64 and independent_validation.

## Open
Whether LEAD-64 deserves a job given n=3 power; draftcap row ID (not named in ROADMAP).

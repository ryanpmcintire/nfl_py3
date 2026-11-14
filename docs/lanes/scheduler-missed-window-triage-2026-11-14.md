# Scheduler missed-window triage 2026-11-14

## Goal
Make `capture_scheduler.py --health` stop failing on 116 unacknowledged MISSED rows (2026-10-28..11-10) without hiding a real miss.

## State
Before: OVERALL FAIL, 116 unacknowledged. After: 113 acknowledged, 3 left; see the OVERALL line in the final report.
- (a) RERUN, 89 acknowledged: scheduler_log shows OK (or MANUAL-RUN OK) for the same job the same day after the window start. The MISSED rows were written by restart sweeps (10-31 07:17, 11-03 14:22, 11-04 08:33-09:07, 11-09 12:35, 11-10 14:05-14:26) although the job ran. 11-03 and 11-10 lock dependents ran by hand after the splash_board failure (11-10 weekly_lock 20:37, dependents 20:56).
- (b) UNRECOVERABLE, 24 acknowledged, each job OK since:
  - 4: daemon down after the 10-31 01:18 and 08:25 crashes (log gaps 00:33-07:07, 09:00-13:01): injury_news_sat_0400, odds_private_sat, prediction_markets_sat_1200, refresh_last_call_sat_1215.
  - 6: independent_validation FAIL(2) "source changed" in window (10-28..11-03).
  - 2: draftcap gate float noise FAIL (fixed 7f88f2cf); 1 paired_wind_sun_1145 TimeoutError; 1 refresh_sun FAIL (11-01 10:02).
  - 4: build_staff_state, per07_staff_state, travel_interruptions, mod54_forward on 11-03: no RUN in window, first swept at the 11-03 restarts (inferred: jobs not yet in the daemon schedule).
  - 2: prediction_markets_wed_1200, independent_validation_wed_1210 on 11-04: daemon busy with serial catch-up runs 08:53-13:38.
  - 4: odds_espn mon/thu/fri_evening/sat_midday: no RUN logged, swept at 11-09 12:35 restart (inferred: new to schedule).
- (c) STILL BROKEN, 3 left unacknowledged: odds_espn_sun_morning, odds_espn_sun_close, odds_espn_sun_late @ 2026-11-08. No OK since 2026-09-20.

## Tried
Parsed `--health` `!!` lines; per key compared MISSED log time, same-day OK, next OK, and log gaps over 25 min. Read sweep_missed (blocked_by only labels a MISSED row) and acknowledge_missed (appends an `acknowledged` field only, never deletes).

## Next
Class (c): odds_espn_sun_* have not run OK since 09-20 and the three Sunday windows passed with no RUN line. Confirm they are enabled and scheduled in SCHEDULE, then watch 2026-11-15 11:00, 12:20, 16:15; if OK, acknowledge the three (a later OK proves the job works). If no RUN line appears, fix the schedule entry.

## Open
The 89 reasons say the row came from a restart sweep; why the sweep re-marked jobs that had run (state rewind or loss) is not diagnosed.

# Lane: scheduler job batch, apply after the Week 9 lock

## Goal
One unapplied patch adding the Week 9+ jobs to scripts/capture_scheduler.py, applied with one edit and one daemon restart.

## State
Patch: tests/scratch/scheduler-jobs-2026-11-03.patch (211 lines). `git apply --check` exit 0; scripts/capture_scheduler.py restored, `git diff` empty. Apply: `git apply F:\Repos\nfl_py3\tests\scratch\scheduler-jobs-2026-11-03.patch`, then restart the daemon once (check the process list first).
Jobs (added_on 2026-11-03; 15 entries):
- travel_interruptions_{mon..sun} 08:00 + travel_interruptions_sun_1000, grace 120, capture_travel_interruptions.py.
- prospective_scorecard_tue 09:30 (20, catch_up), prospective_score_tue 09:40 (30, catch_up, nfl-ats prospective-score ... --bootstrap-samples 1000).
- odds_tape_sun sun 12:12, grace 35, `--duration-minutes 48` (ends 13:00); constant ODDS_TAPE_CAPTURE. The 4th Job field is the start grace; the run cap is JOB_SUBPROCESS_TIMEOUT_SECONDS 3600 s, which covers 48 min.
- odds_2h_probe_thu_20261105 thu 21:25, grace 90: no single-date mechanism and no existing one-off exist, so the argv is a `python -c` that exits 0 unless today (ET) is 2026-11-05, then runs probe_bovada_live_periods.py --repeat-minutes 70. Other Thursdays log a no-op OK; remove after 11-05.
- forward_terms_tue tue 12:35 grace 105, `--week auto`, requires=("weekly_lock",) only.
- mod25_wfj_forward_tue tue 12:50 grace 90, requires weekly_lock. The script has NO `--week auto` (no flag = backtest read). argv is `python -c` calling mod25_wfj_forward.record_week(forward_terms_week.resolve_week('auto')) (auto = newest served-card week, available after the lock). Imports verified; resolve_week('auto') returns 8 today (Week 9 card not yet written).
- lead64_freshness_fri fri 10:00 grace 120: `python -c` runpy of lead64_freshness_audit.py with `--as-of` set to the current UTC time (no templating exists in the scheduler).
Verification (measured): imported capture_scheduler, listed the 15 jobs with day/time/requires/catch_up/occurrence and printed dry_command argv for each; names unique. Jobs were NOT run (fence), so no --run-job rehearsal; AGENTS requires one per added job after apply (use --run-job NAME --dry; for the python -c jobs --dry does not stop writes: mod25_wfj and the probe write on run, so rehearse only after the lock for mod25_wfj, and the probe only on the guard day).
`pytest tests -k scheduler -q`: 64 passed, 144 teardown PermissionErrors in tests/scratch basetemp (environment), no failures, no job-list pin hit.

Revision 2 (patch now 291 lines, check exit 0, file restored): travel_interruptions_mon removed (tue..sun 08:00 + sun_1000 remain, 7 jobs); added nflverse_injuries_fri_2000 fri 20:00 (copy of daily argv, dedupe 60); build_staff_state_tue tue 07:45 (requires coordinators_tue) and per07_staff_state_tue tue 07:55 (requires build_staff_state_tue; two jobs, no chained argv); mod27_pool_offset_forward_tue tue 13:05 grace 60, requires weekly_lock, python -c mod27_pool_offset_observed.forward(resolve_week('auto')) (script --week is int default 9, no auto). Now 19 jobs. Import verified; pytest scheduler 64 passed, 144 teardown errors only.

## Tried
Searched for a templated week/date or one-off mechanism in the scheduler: none.

## Next
Root applies the patch after the lock, restarts the daemon once, runs `--run-job` rehearsals for the capture jobs, confirms forward_terms_tue and mod25_wfj_forward_tue fire after weekly_lock.

## Open
Long travel job-name line in the patch exceeds 100 columns (ruff E501 if enforced). The lead64 job is low priority (n=3 power). Whether mod25_wfj_forward should get a real `--week auto` flag instead of the inline argv.

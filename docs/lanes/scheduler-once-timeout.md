# scheduler --once timeout

## Goal
`capture_scheduler.py --once` (the mandated session-startup step) must finish
in seconds when the daemon is healthy instead of blocking ~30 min inside a
refit it duplicates. Done when --once defers to a live daemon, one process
can never run a job twice, and the scheduler suites stay green.

## State
- 2026-09-16: diagnosed (measured). Daemon ran `lineups_wed` 12:00:37; two
  `--once` calls at 12:04:54 and 12:26:00 re-ran the same 12:00 window
  in-process and were killed by the caller at 180s/300s; daemon's own run
  went OK at 12:30:25 with 3 PICK-CHANGEs. No orphans left (verified via
  process list). `execute_job_with_output` allows 3600s per job; a refit
  takes ~30 min, so any `--once` inside a refit window hangs.
- Fix SHIPPED in `scripts/capture_scheduler.py`: per-job lockfile beside
  the state file (`<state-dir>/scheduler_locks/<job>.lock`, pid + liveness
  via `pid_is_alive`, stale takeover, release-only-own) held inside
  `run_job`, so daemon, `--once` and catch-up can never double-run a job
  and SKIP writes no state; `--once` defers due jobs to a live daemon
  (DEFER log, no execution) and runs inline only when the daemon is down.
  Lock dir derives from `STATE_PATH`, so the existing test isolation
  (`STATE_PATH` patched to tmp) covers locks with no test edits.
  `run_job_manually` deliberately unchanged (supervised argv rehearsal;
  tests stay hermetic).
- Verified: lock scratch (acquire, contention, holder pid, stale takeover,
  foreign-release refusal) OK; DEFER scratch (live daemon, due job, no
  subprocess, state saved) OK; live `--once` exit 0 under 1s;
  `test_capture_scheduler.py` + `test_scheduled_lock.py` 70 passed (one
  xdist order failure during development, fixed by the STATE_PATH-derived
  lock dir); ruff format + check clean on the script. Served card
  consistent: CURRENT_PREDICTIONS.md already carries the noon 3-pick
  refresh section. Nothing committed.

## Tried
- Nothing before this session; the Sept-9 log (4 RUNs, 1 OK for
  `lineups_wed`) shows the same overlap class on retry.

## Next
- Full contract gates after the edit (ruff format/check repo-wide, mypy
  src, pytest -q); then session report. No new jobs added, so no
  `--run-job` obligation; argv unchanged.

## Open
- Residual: `--run-job` can still overlap a live daemon run on purpose;
  left as supervised use, operator-visible via RUN + MANUAL-RUN lines.
- `READ_ONLY_EXCEPTIONS` line ledger is already stale (numbers do not match
  current write sites); not extended.

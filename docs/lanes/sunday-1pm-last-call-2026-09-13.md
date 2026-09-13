# sunday-1pm-last-call-2026-09-13

## Goal

Before the Week 1 13:00 ET kickoffs: every pick flip reaches the owner's
phone, and one more look at the 1pm games happens after the 12:00 refit and
the 12:30 closing odds. Done when `refresh_last_call_sun_1245` has run once
for real, the daemon is restarted on the new schedule, and the lane says
which picks (if any) flipped.

## State

- Gap found (read, `scripts/refresh_lineup_forecast.py` and
  `notify_after_job`): the 12:00 `lineups_sun` refit ends with
  `refresh-picks --publish-card` but the daemon only alerted on `refresh_*`
  jobs, so a flip at the 12:28 republish was silent. Fixed: `notify_after_job`
  covers `lineups_*` too, `parse_job_json` takes the last JSON object holding
  `changed_picks`, and manual `--run-job` (non-dry) alerts as well.
- Gap found (measured): this agent shell has no `NFL_ATS_NTFY_TOPIC`
  (length 0) while `HKCU\Environment` has it. The 07:27 daemon was started
  from a session shell, so it likely had none either and would have sent
  nothing. Fixed: `send_notification` resolves the topic at send time from
  env, then the user registry; logs `NOTIFY-SKIP` when absent. Test message
  sent 12:16 ET, HTTP OK (measured).
- New job `refresh_last_call_sun_1245` (sun 12:45, grace 10, records
  decisions, no catch-up). Scheduler tests: 57 passed (measured).
- Daemon's 12:00 `lineups_sun` refit logged OK 12:28:21 and republished
  the card with no pick change (measured: `git diff CURRENT_PREDICTIONS.md`
  shows only DEN at KC's decision score 57.0% -> 56.9%). Daemon restarted
  12:28:45 on the new code, pid 19136, 195 enabled jobs (measured,
  scheduler log). Uncommitted.
- A duplicate `lineups_sun` refit was started by this session's startup
  `--once` at 12:05 and stopped at 12:09; the daemon's own run continued.
  The 12:08 arrests snapshot without a manifest was the daemon's live
  capture, not an orphan; nothing was deleted.
- Board improvement: the week timeline's refresh-pass list
  (`board_content.WEEK_REFRESH_PASSES`) was missing the four MKT-08
  last-call passes and now carries all 17 passes including Sunday 12:45;
  its contract test now derives the expected list from the live
  `SCHEDULE` instead of an AST walk that skipped the comprehension-built
  jobs (`tests/test_board_terminal.py`). Board tests 88 passed (measured).
- Orchestration running in the background (scratch `sun1245.sh`): waits for
  the daemon's refit, restarts the daemon, runs the 12:45 job, then
  `card-ledger-check`. Its output decides the Next list below.
- Card unchanged since Saturday 12:54 ET (ledger has no newer revision
  rows; scheduler log has no PICK-CHANGE since then).

- 12:45 pass (measured, scheduler log): `MANUAL-RUN OK
  refresh_last_call_sun_1245` at 12:47:18, and the daemon's own scheduled
  run of the same job logged OK at 12:47:26. One revision row: CHI at CAR,
  the pick-revision ledger's Saturday heavy-money side (CHI -2.5) went back
  to the model side CAR +2.5 (51%), `movement_policy=model_only`; the
  served card already read CAR +2.5 since the owner's Sunday realign, so
  the card did not change. ntfy message sent (PICK-CHANGE logged, no
  NOTIFY-FAIL/SKIP). `card-ledger-check`: 16 paper rows, 6 revision rows,
  no disagreements (measured).
- Board already carries the 17-pass list: the refit's own `publish-board`
  ran after the edit (`docs/index.html` names the Sunday 12:45 PM ET
  refresh; measured).

## Next

- Commit with handoff refresh, push; then move this lane to done.
- Reader-facing follow-up (queued, not done): the card's late-week table
  still lists CHI at CAR as a Saturday change to CHI while the picks table
  and both ledgers say CAR; the table should show the current side.

## Open

- None.

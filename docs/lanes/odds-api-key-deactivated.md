# Odds API key deactivated

## Goal

Restore the Tuesday/Wednesday/Thursday/Friday/Saturday odds captures. Done when
`odds_sat` (or any bulk odds job) records OK again and the halves jobs stop
blocking on their parents.

## State

- 2026-09-19 (measured): the three mid-week bulk captures failed identically —
  `odds_wed_opener` (09-16), `odds_thu_tnf` (09-17), `odds_fri_1230` (09-18)
  each `FAIL(1)` in `data/scheduler_log.txt`, and `capture_log.txt` shows the
  cause on all three: `The Odds API returned HTTP 401`. One live diagnostic
  read the error body: `DEACTIVATED_KEY` ("API key is deactivated. This could
  be due to cancelation or a failed payment"). The 32-char key is present in
  both the process environment and HKCU, so this is billing, not a missing
  secret. Zero credits spent on the failures; quota intact at 98,517.
- Cascade (measured): `odds_wed_opener_halves`, `odds_thu_tnf_halves` read
  `MISSED` (parent prerequisite never successful), and `odds_fri_1800` read
  `MISSED` in its window. Those point-in-time snapshots are unrecoverable.
- Daemon healthy: RUNNING, `--once` exits 0, all non-odds jobs OK.

## Tried

- Single raw-HTTPS diagnostic against the bulk endpoint to capture the 401
  body (key value never logged). No retries beyond that; 401s cost nothing
  but re-running a dead key proves nothing further.

## Next

- Owner: check The Odds API billing/subscription, reactivate the key, then
  confirm with `capture_scheduler.py --run-job odds_sat` (or the next due
  bulk job) and watch the halves job follow on its `requires=(...)` link.
- If the key is replaced rather than reactivated, update it in the user
  environment where `scripts/odds_capture.ps1` reads it (process env, then
  HKCU) and on the second capture host, which shares the quota.

## Open

- Whether the three lost mid-week snapshots matter for any open read (refresh
  attribution, market-move terms). The Tuesday opener and Saturday snapshots
  that anchor the served card are unaffected.

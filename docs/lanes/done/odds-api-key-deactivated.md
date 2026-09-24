# Odds API key deactivated

CLOSED. The owner cancelled the subscription deliberately and will not renew it. This is not an open item or an owner action; current odds come from free sources (`docs/lanes/free-odds-sources.md`).

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

- Owner decision 2026-09-19: NO re-subscribe (too expensive). The paid API
  was only ever needed for the historical backfill; present odds move to
  free sources. Nothing in this lane needs billing action.
- DONE 2026-09-19: all 18 paid jobs disabled (`enabled=False`): 10 bulk
  `odds_*` via `odds_capture.ps1`, all six `*_halves` via
  `odds-ingest-halves`, and both `player_props_*` (same dead key).
  `weekly_lock` requires now `("splash_board_tue",)` only, otherwise the
  Tuesday lock could never fire; scheduler suites green (70 passed),
  daemon restarted (pid 19452), `--status --brief` shows the paid rows
  `no` with no new FAILs possible. Owner's free-source answer: no
  specific agreement exists ("assumed multiple sources").
- Open: which free sources to build on (no record in the repo); candidates
  visible in-tree are the hand-captured Splash board and
  `public_betting_live_capture.py` (Action Network, free, already
  scheduled). Next unit: free-source feasibility probe.
- Open: what happens to the board features that read the latest capture
  (Books-now column, market-move labels, Best Pick dispersion pool,
  Sunday re-nomination pool) with no cross-book feed.
- Open: whether the historical backfill finished before deactivation.

## Open

- Whether the three lost mid-week snapshots matter for any open read (refresh
  attribution, market-move terms). The Tuesday opener and Saturday snapshots
  that anchor the served card are unaffected.

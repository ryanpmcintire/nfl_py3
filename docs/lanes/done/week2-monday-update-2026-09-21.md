# Week 2 Monday update — 2026-09-21

## Goal
Update the played card through Sunday's finals, leaving Giants at Rams pending.

## State
- **Measured:** `nfl-ats settle --season 2026 --week 2 --write-graded`
  loaded live nflverse scores: 15 Week 2 finals; played card **13-2**, one pending.
  Production served-ledger grading gives **22-9** season and **1-1** Best Pick.
  Denver -2.5 won; Tampa Bay -8.5 and Philadelphia -6.5 were the two losses.
  Giants at Rams remains pending with **LA -7.5** on the frozen card.
- **Measured:** `nfl-ats card-ledger-check` found no disagreements across the
  16 paper rows, one revision row, and served card. No decision rows were added.
  Republishing restores the Markdown star to locked Denver, matching the ledger.
- **Measured:** `nfl-ats publish-predictions --with-board` and
  `nfl-ats publish-board` succeeded. Rendered record strip changed from
  `1-0 so far / 10-7 / 0-1` to `13-2 so far · 1 game left / 22-9 / 1-1`.
  Sunday rows now show final scores and Covered/No cover; scheduled lineup and
  source freshness changes are included. The pending-game count is computed
  from the same served decisions and final scores as the weekly record.

## Tried
- **Measured:** required `ruff format --check .` (1,114 formatted files),
  `ruff check .`, `mypy src` (235 source files), and `pytest -q`
  (**4,529 passed, 9 skipped**) all succeeded through `.tools/uv.exe run --no-sync`.
  Ruff initially crashed while reporting an inaccessible `.pytest-best-pick-gap`
  scratch directory; adding it to the existing exclusions fixed both full scans.
  Full logs and served grading are in ignored `.tmp/session-week2-*.log`.
- **Measured:** scheduler `--once` and `--status --brief` succeeded; daemon was
  already running. Its 10 MISSED entries were acknowledged historical windows;
  there were no NEVER RUN jobs or open windows. No scheduler jobs changed.
- **Reviewed:** this display change adds no tests, mocks, research constants,
  or runtime research assertions. No test removals are proposed by this change.

## Next
After Monday's final, run settlement and republish to close Week 2. The scheduler
continues to own capture cadence. Do not rebuild played picks from final scores.

## Open
Three scheduler-generated registry JSON files predated this work and remain
untracked: margin-backtest `20260921T160806Z`, margin-predict
`2026-week-02-20260921T161008Z`, and waterfall-feed `20260921T161510Z`.
Commit, push, and Pages deployment verification are recorded in the session report.

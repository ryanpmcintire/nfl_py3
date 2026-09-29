# Week 4 Tuesday lock (2026-09-29)

## Goal
Publish the 2026 Week 4 card and dashboard from the pool's Tuesday lines.

## State
Done. Board read in Chrome and captured to
`data/splash/2026_week04_20260929_1514.json` (16 games; slate
`slate_01KZKRMEGDTQMHREA5666Y6N2S`). `--run-job splash_board_tue` MANUAL-RUN OK.
`--run-job weekly_lock` ran every step through publish-predictions
`--record-decisions` (19:29 UTC) and the prospective steps (19:42 UTC). The
1800 s wrapper timeout killed only the final read-only drift-report, so the job
logged FAIL(1). Card: model `b578fbea1c5c706f`, Best Pick PIT -2.5 (60.6%).
UI-20: the status rail and screen-reader summary now read "1 STRONG READ" or
"1 GAME" instead of plural forms (`board_terminal.py`).

## Tried
- The capture parser needs `AWAY  Sun, Oct 4 1:00 PM  HOME` headers and
  `Nickname TEAM -N.5` rows; the page text had to be reshaped into that form.

## Next
- None for this lane. The scheduled Tuesday job still depends on a manual board
  read; the lock's 1800 s wrapper timeout is tight (the run took ~33 min).

## Open
- `scheduled_weekly_lock.py:96` timeout=1800 is shorter than a full run with
  drift-report; consider raising it.

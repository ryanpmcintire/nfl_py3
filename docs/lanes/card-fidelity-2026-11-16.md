# Card fidelity check (2026-11-16)

## Goal
After every card refresh, recompute each open card game's served probability through the
card path (`card_term_recorder.capture_plan` -> `plan_refresh`) and fail loudly when the
card file, pick-revision ledger or card-term recorder disagree on side or displayed
percentage, or when an input behind the pick is stale. Part of goal-65 (card 76-73 while
the current formula replays better; differing sides went 4-12).

## State
- `scripts/card_fidelity_check.py` written; ruff check and format clean.
- Checks per game: side and displayed % vs `CURRENT_PREDICTIONS.md` main table; side vs
  ledger (latest revision, else published side); side and probability vs newest
  `data/card_terms/<season>_week_<ww>/*.parquet` row; newest leader-book move quote vs the
  latest move-feeding capture slot before the last refresh slot (threshold derived from
  `capture_scheduler.SCHEDULE`: enabled `odds_` jobs whose script produced leader-book
  quotes this week, minus that job's `dedupe_minutes`); move books (one-book move fails,
  reusing `check_served_terms.ANCHOR_MINIMUM_BOOKS`); `temp`/`wind` present for open-air
  games (`mos_fetch` roof rule) once a `mos_fetch_` slot has passed; projected QB id vs
  newest depth-chart QB1 at the refresh instant.
- Open games fail (exit 1, `capture_scheduler.send_notification("NFL card check failed")`,
  which logs NOTIFY-OK/FAIL); locked games go to `locked` (report only). `--dry-run` never
  notifies; `--print-schedule` prints argv and 21 slots (refresh slot + 5 min).
- 2026-11-16 Week 10 run (all locked, exit 0): sides agree on all 14 games across card,
  ledger and recorder; QB inputs all match depth QB1; move quotes 14-18 min old at the
  last refresh; MIN at GB move rests on 1 book; every open-air game (8) has null
  `temp`/`wind` in the served input; locked % gaps (post-lock inputs) e.g. CAR at NO card
  50.4 v now 63.8, MIN at GB 50.9 v 53.8. Week 11: no original card yet (`card: none`).

## Tried
- Probes in scratchpad only. Leader books are bovada/williamhill_us/mybookieag; ESPN
  captures (DraftKings) do not feed the move, hence the data-attributed capture set.

## Next
- Root: add scheduler jobs `card_fidelity_<day>_<hhmm>` from `--print-schedule`. Do not
  add the prefix to `ALARM_JOB_PREFIXES` (the script notifies itself; double alert).
- Confirm whether the active feature profile reads `temp`/`wind`; if yes, the null served
  weather is a live stale-input defect for goal-65.

## Open
- 5 minutes after a refresh start can precede refresh completion (graces up to 300 min);
  consider gating on the refresh job's completion instead.
- No Tuesday slot: the Tuesday card is publish-predictions, not refresh-picks.

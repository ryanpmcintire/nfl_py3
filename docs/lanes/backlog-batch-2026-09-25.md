# Backlog batch 2026-09-25

## Goal

Owner asked (2026-09-25) for a standing third lane that works through the
remaining backlog in bounded units alongside SIM-04 and XLG-09. Done when each
item below is shipped, recorded, or blocked with its reason.

## State

- (a) DONE: drafted weak-signals records reconciled across seven lanes; 30
  already recorded, one missing companion recorded
  (`pooled_signal_sixth_fit_vs_model_only`, verified against its artifact).
- (b) DONE, published: UI-20 queue (a)-(h) were all already shipped, so the
  History page's pick table gained a "Vs. the close" column (secondary context,
  never changes the Outcome) and a caption. `board_site_content.py`
  (`HistoryPickRow.close_outcome_text`, `_history_pick_rows(close_reference=)`)
  and `board_terminal.py`. Rendered: 23 of 32 settled picks also covered the
  close, 9 missed, 1 had no archived close.
- (c) DONE: Week 3 challengers all recorded by scheduled passes (lockday_verify: 59 recorded, 4 skipped, 0 missing); the tiebreaker shade is gated by design; lane moved to done/.
- (d) MKT-08 per-player news value decomposition (lane `news-trigger-refresh`) running.

## Tried

- (b) verification: ruff format/check clean, mypy src clean,
  `pytest -q -k "board or history"` 312 passed, publish-board exit 0; mobile
  stacking via the existing `max-width:680px` data-label rule.

## Next

- On each return: verify, commit, push; refill from `docs/lanes/README.md`.

## Open

- None.

# Week 2 pool board and lock

## Goal

Get the Week 2 Splash board into `data/splash/`, clear the `splash_board_tue`
gate, and run the real Tuesday lock so the card is formed on the numbers the
pool grades. Done when the lock records Week 2 decisions and the board
publishes.

## State

Done. Week 2 is locked, recorded and published.

- **Board captured (measured):** `data/splash/2026_week02_20260915_1302.json`,
  16 games, read from the contest page at 13:02 ET. Contest
  `contest_01KZKRMEGF4NNVRFG54TN5P5RH`, channel BIG-DAN-2276, 250 entries,
  picks lock Sun 2026-09-20 16:00 ET, tiebreaker `2026_02_NYG_LA`.
- **Pool's own record (read):** Week 1 finished 9-7, rank T86/250 — the pool
  agrees with the served card.
- **Override reached the model (measured):** all 16 Week 2 rows in
  `game_features.parquet` carry the board's half-point numbers as
  `spread_line`.
- Lock ran 13:03:36–13:31:15, steps 1-14 OK, new model `2b7790c47566874a`.
  Best Pick `2026_02_JAX_DEN` (DEN -2.5, 57.7%).
- `lockday_verify`: paper ledger 16 rows, challengers 48 recorded / 11 skipped
  / 2 MISSING. Board and card published; full suite 4529 passed, 9 skipped.

## Tried

- `splash_board_tue` had failed after 4 retries and `weekly_lock` read
  `waiting for ...`. The board read is a hand step by design, so the failure
  was correct.
- **Scheduler defect fixed:** `run_job_manually` wrote only the `job_health`
  entry, never `state["runs"]`, which is the only thing
  `unsatisfied_prerequisites` (`clv`-style check in `capture_scheduler.py`)
  reads. A prerequisite recovered by hand left dependents blocked for the rest
  of the window. A successful non-dry manual run now records `OK-MANUAL` and
  that status joins the accepted set.
- **The lock failed closed at step 15 `publish-board`** on the
  `overlay_union_..._v3` OR-union invariant. Reproduced exactly: since MKT-19
  the served card is the four-term calibrated probability, so
  played = raw + member flips + calibration, and the calibration alone can
  cross 0.5. Three Week 2 games sit within 0.005 of 0.5 —
  `LV_LAC` (declared flip, calibration pulled it back to HOME),
  `MIN_CHI` and `WAS_DAL` (no member fired, calibration crossed the
  threshold). The ledger rows are CORRECT under the column's own meaning:
  `card_ledger_check.py:94` reads `composed_overlay_flip` as union membership
  and all three match. The stale part was the invariant's second clause,
  `observed_flip == declared_flip`, a derived consequence of the pre-MKT-19
  world. Removed that clause in `clv.py`; kept the real OR-union clause
  (`member_flip & ~declared_flip`). The served side is still checked live by
  `card_ledger_check` against a freshly resolved card, so no guard was lost.
- The 11 MISSING challengers were fallout from the same error (the lock log
  shows `prospective-record`/`prospective-score` skipping on it). Nine were
  recovered with `prospective-record --challenger`; `publish-predictions
  --record-decisions` recovered the rest down to 2.
- `fit-pick-probability` was re-run because `active_pick_probability.json`
  still named the old model. **It was a no-op**: coefficients are bit-identical,
  which is correct — the fit uses 2020-2025 opener-graded games whose
  predictions are made walk-forward from prior data, so a new week cannot
  change them. Only the stale `active_model_id` stamp was wrong.
- `tests/test_model_ledger.py` pinned `games == 2075`; the lock legitimately
  made it 2091. Changed to read the count from the active-model manifest
  rather than re-pinning a number that breaks every week.

## Next

- Dashboard: shipped `pool_line_note` — the week page now tells the reader the
  spreads are the pool's own locked numbers (`board_content._build_pool_line_note`,
  `board_terminal._pool_line_note_html`, `.pool-line-note` css), sourced from
  the capture, absent when no capture exists.

## Open

- **Owner decision: `best_pick_sunday_renomination` has no Week 2 Tuesday
  anchor and cannot get one.** `best_pick_refresh_prospective.py:108` requires
  the recording instant to equal the paper ledger's `recorded_at_utc` exactly,
  so the anchor is only writable inside the same pass that writes the ledger —
  the pass that died on the invariant. Recovering it needs a full-week
  `--replace-week` re-record, which is blocked pending the owner. Cost is one
  week of one paired challenger; the served card and the Best Pick itself are
  unaffected.
- **Owner call on the invariant change above.** It is a release-blocking
  contract and the reasoning is recorded here; reverse it if the intended
  meaning of `composed_overlay_flip` is the served side change rather than
  union membership — in which case the recorder, not the validator, is what
  needs changing.
- `tiebreaker_low_side_shade` shows MISSING but has a documented reason (the
  served total already carries the shade, so the arms are no longer a paired
  contrast); it emits no gate string, which is a reporting gap only.
- `slate_id` not recorded for Week 2 (no `slate`-matching network request after
  page load). Optional provenance.

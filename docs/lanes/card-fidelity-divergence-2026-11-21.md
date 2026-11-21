# Card fidelity divergence (2026-11-21)

## Goal
Explain the ~2.7-point card-vs-recompute gaps (BAL at CAR, LV at DEN, PIT at PHI, TEN at DAL)
that `scripts/card_fidelity_check.py` reports; name the term, the stale file, its writer, and
which numbers are the served formula.

## State (2026-11-21 10:45 EST, verified 10:46)
- Stale file: `CURRENT_PREDICTIONS.md` (`card_fidelity_check.py:30`), mtime Fri 13:30:49 ET,
  written by `refresh_last_call_fri_1330` (scheduler_log 8359-8363, commit 4d6cad3f) on the
  ARI at KC flip; that write syncs every row at the Fri 13:30 plan (`pick_refresh.py:2136-2145`).
- Writer gate (read): `cli_commands/publishing.py:1593-1595` writes the card only on
  `--publish-card`, a recorded side change, or a served star move. `refresh_sat`
  (`capture_scheduler.py:2147`) has neither flag nor a flip, so the card kept Friday's numbers;
  `publish-board` renders docs from the current recompute.
- Measured (`at_instants.log`): fit pick_probability/20261120T171725Z at every instant; model,
  flag, hook identical. Recompute at Fri 13:30:01 ET reproduces the card exactly; from Sat 10:26
  only `move_toward_home` changed by one 0.5 step (x 0.220012 = 0.110 logit):
  BAL +0.5 -> 0 (54.0 -> 51.3), LV 0 -> +0.5 (57.7 -> 60.3), PIT +0.5 -> +1.0 (PIT 53.6 -> 50.9),
  TEN 0 -> -0.5 (53.7 -> 50.9). Hook hypothesis refuted (LV 0.0440, TEN 0.1306 both paths).
- Served formula: both numbers are the declared formula; the recompute (= docs after 10:55
  publish-board, = card_terms 20261121T154013Z) is the current served value; the card file is
  the same formula on Friday 13:30 inputs. refresh_sun 10:00 `--publish-card` resyncs all rows.
- Card != board flags are right: the card is stale until the 12:15 recorded refresh.
- Recorder gate (2026-11-21 10:55 EST, uncommitted, `scripts/card_fidelity_check.py`):
  recorder side/probability/missing-row compared only when the newest recorder parquet's
  max `run_at_utc` >= the latest write at or before the game's as-of: the scheduled
  refresh slot the check already computes (`refresh_at`) or the `CURRENT_PREDICTIONS.md`
  mtime. Older -> non-failing `pending` line "<game>: recorder pending after <job> (<ET>)",
  row field `recorder`; JSON adds `recorder_written_utc`, `card_written_utc`, `pending`.
  Side, card %, ledger, move, weather, QB checks unchanged. ruff format/check clean.
- Dry run 10:53 ET (exit 1): recorder 20261121T154013Z (10:40 ET) newer than refresh_sat and
  card write -> all 13 compared, 0 recorder problems, 0 pending; only the 4 card-% problems.
  Gate exercised for refresh_last_call_sat_1215: recorder 10:40 -> pending "Sat 12:15 ET";
  a 12:25 recorder -> compared.
- Card % (read): check reads `CURRENT_PREDICTIONS.md` rows above the late-week marker
  (`card_fidelity_check.py:30,98`); writer `_sync_card_table_rows` rewrites "Cover chance"
  `f"{estimate:.1%}"` from `new_home_cover_probability` (`pick_refresh.py:1998-2022,2134-2145`);
  every sat/sun refresh carries `--record-decisions`, so 12:15 rewrites all eligible rows.

## Tried
- `tests/scratch/card_fidelity_diag/decompose.py`, `at_instants.py` (+ .log), `dry_run.log`.
- Gate run output: scratchpad `fidelity_dry.out` / `.err` (session-local).

## Next
- Root decision, card writer (not owned here), proposed diff at publishing.py:1595:
  `-    if args.publish_card or recorded_change or (args.record_decisions and star_served):`
  `+    if args.publish_card or args.record_decisions:`
  plus publish_site after every recorded refresh (scheduler `notify_after_job` 3508-3517 only
  publishes on PICK-CHANGE). Corrected rows now: CAR -2.5 51.3%, DEN -9.5 60.3%, PIT -4.5 50.9%,
  DAL -7.5 50.9% (late-week BAL row 51.3%); they move with each move step until Sunday 10:00.

- Root: review and commit the checker gate; after 12:20 ET read `card_fidelity_sat_1220` in
  scheduler_log: expect exit 0 with "recorder pending" lines (no card_terms slot until 16:00)
  unless an odds capture between 12:15 and 12:20 moves the recompute.

## Open
- Order: card_fidelity_* at refresh+5, card_terms_* at +10 (none after 12:15 Sat), so every
  scheduled check now reports recorder pending; the comparison bites only on off-slot runs.
  Swap the slots at the next daemon restart (scheduler, root) to restore it on schedule.
- Inferred: if the served move input is read at "now", an odds capture between a refresh and
  its +5 check can still raise card-% gaps; not observed in today's runs.
- Thursday 15:31 BAL 50.7 v 53.5 (other direction) not re-run; same mechanism inferred
  (pre-flip card under fit 20261118T171615Z).

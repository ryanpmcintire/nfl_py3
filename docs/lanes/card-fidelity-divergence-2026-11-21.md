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
- Dry run 10:46 ET (exit 1): only the four card-% problems; no side, ledger, recorder, move,
  weather or QB problems. No checker change made (it is right to flag card != board).

## Tried
- `tests/scratch/card_fidelity_diag/decompose.py`, `at_instants.py` (+ .log), `dry_run.log`.

## Next
- Root decision, card writer (not owned here), proposed diff at publishing.py:1595:
  `-    if args.publish_card or recorded_change or (args.record_decisions and star_served):`
  `+    if args.publish_card or args.record_decisions:`
  plus publish_site after every recorded refresh (scheduler `notify_after_job` 3508-3517 only
  publishes on PICK-CHANGE). Corrected rows now: CAR -2.5 51.3%, DEN -9.5 60.3%, PIT -4.5 50.9%,
  DAL -7.5 50.9% (late-week BAL row 51.3%); they move with each move step until Sunday 10:00.

## Open
- Order: card_fidelity_* runs at refresh+5 min, card_terms_* at +10, so checks after a refresh
  compare the previous recorder file (Fri "NO at CHI recorder differs by 0.0040"). Swap the
  slots (scheduler, root) or gate the recorder comparison on run_at >= refresh slot.
- Thursday 15:31 BAL 50.7 v 53.5 (other direction) not re-run; same mechanism inferred
  (pre-flip card under fit 20261118T171615Z).

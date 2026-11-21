# Card fidelity divergence (2026-11-21)

## Goal
Explain card_fidelity_check.py failures on the Week 11 card; name the term, stale file and
writer; fix the checker (non-pinned) where its rule is wrong.

## State (2026-11-21 12:51 EST)
- Card-% gaps (10:45): `CURRENT_PREDICTIONS.md` kept Fri 13:30 numbers (writer gate
  `publishing.py:1593-1595`); only `move_toward_home` moved one 0.5 step. Card rewritten 12:33 ET
  by `refresh_last_call_sat_1215`; those gaps are gone.
- Recorder gate (10:55, uncommitted, in the checker): recorder compared only when the newest
  card_terms `run_at_utc` >= the latest refresh slot or card write; else a `pending` line.
- 12:40 move-quote alarm on all 12 open games = FALSE ALARM (B), measured:
  - Served move = `leader_median_net_move` over LEADER_BOOKS bovada/williamhill_us/mybookieag
    (`pick_probability.py:771-772`, `sharp_book_movement_features.py:24,153-157`).
  - ESPN pickcenter writes only `draftkings` (`capture_espn_pickcenter.py:122`); 10:00 and
    12:29 ESPN dirs = 48 draftkings rows each. Newest leader quote on disk = 14:56:36Z
    (09:56 ET dry-run capture); 0 leader rows after it.
  - Move recomputed at 12:15 slot, 12:32 run and at run time (~12:45) = card `market_move_toward_home`
    = 10:40 recorder for all 13 games (e.g. PIT +1.0, TB -1.5).
  - Checker bugs: `move_capture_slots` marked ESPN "feeding" because its 10:00/12:30/18:00
    windows coincide with private captures; and private jobs have `dedupe_minutes=0` while the
    script's own `--max-age-minutes` guard (default 30, `capture_private_sunday_odds.py:78,112`)
    is what skipped 10:00.
- Fix (uncommitted, `scripts/card_fidelity_check.py`): a script feeds only if a leader quote
  falls in its slot windows and no other script's; skip window = max(dedupe_minutes, argv
  `--max-age-minutes` or the script's argparse default read via ast). Sat skips: espn 15,
  odds_private_sat 30, deadline jobs 1. ruff clean.
- Dry run 12:50 ET: exit 0, 0 problems, 0 locked, 12 "recorder pending" (recorder 10:40 <
  12:15 refresh; expected until card_terms 16:00). Capture used: odds_private_sat.

## Tried
- `tests/scratch/card_fidelity_diag/` (decompose, at_instants, dry_run.log); scratchpad
  `move_fresh.py`, `move_compare.py`, `dry2.out` (session-local).

## Next
- Root: review and commit the checker (recorder gate + move-source fix).
- Root decision, card writer (not owned): `publishing.py:1595` write the card on every
  `--record-decisions` refresh; publish_site after every recorded refresh.
- Old rule would also have cleared at 15:50: odds_private_sat_1535 always fetches (age 1).

## Open
- card_fidelity_* runs at refresh+5, card_terms_* at +10, so scheduled checks report recorder
  pending; swap slots at the next daemon restart (scheduler, root).
- Attribution uses sole-coverage windows: an off-schedule leader capture landing only in an
  ESPN window would re-mark ESPN as feeding (fail-loud, not silent).

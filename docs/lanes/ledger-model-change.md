# ledger-model-change

## Goal
A routine same-week model re-sync must not raise a stale-flag disagreement, without rewriting the frozen Tuesday ledger.

## State
- Frozen (defines the pick): pick_side, decision_home_spread, bet_side, kickoff, game_id (clv.py:1385-1410 PAPER_DECISION_COLUMNS). Revision ledger holds later side changes.
- Re-sync-derived (legitimately differ): model_id, model_pick_side, probabilities/edge, composed_overlay_flip and member flips. The ledger invariant (clv.py:1508-1520) defines composed_overlay_flip as model_pick_side != pick_side, so it moves whenever the model's own side moves.
- TEN at IND: Tuesday model HOME no flip; after re-sync (403b07a3 -> 2d286390 -> d1d6f128) model side is a 0.498 coin flip, coach_fade carries IND. Side IND -6.5 on every surface; only the derived flag differs.
- Change made: card_ledger_check.py:116 flag disagreement now counts only when the side also disagrees (served_side != ledger_side). Side checks (pick_side vs latest revision, revision vs board) untouched.
- Verified: nfl-ats card-ledger-check -> "no disagreements" (was TEN flag mismatch). Real side disagreement path unchanged (lines 102-113).

## Tried
- Comparing to latest revision flag: revisions only exist on side changes, so TEN at IND has none; does not help.

## Next
Owner/primary: decide if flags should be recorded per re-sync (pick_refresh.py:1378) for explanation text (board_content.py:1544 uses locked vs recorded flag). Not needed for correctness.

## Open
publish-predictions "Repair with --replace-week" hint remains; --replace-week stays owner-blocked.

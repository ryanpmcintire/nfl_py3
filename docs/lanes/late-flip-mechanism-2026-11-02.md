# late-flip-mechanism-2026-11-02

## Goal
Name the mechanism behind post-feed and Tuesday->final card flips, 2026 Weeks 1-8, and whether any is a defect.

## State
Scratch: C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/352ecbdc-246f-44c0-9f81-c287a6ef4436/scratchpad/agentLF (tl.py, mv.py, an.py, bk.py, cl.py, flips.pkl).
- Card != feed (8): W1 NE_SEA is a stale Aug-22 feed vs a reconstructed card (no live job). W1 WAS_PHI, ARI_LAC, CHI_CAR were the retired follow rule (probability set to 0.50). HOU_TEN (Sun 12:42 ET, no move quotes, injury and player-value families arrive: base .540->.574), GB_DET (Sun 14:19 ET, continuity family +0.98 pt, a new coefficient set 18:13Z the same day), LV_NYJ (move 1.75), ATL_TB (williamhill_us half-point tick moves the 3-book median .5->0).
- Tue->final changes: 26 (W1 3 retired-rule, W2-8 23). W2-8 final 13-10, Tuesday 10-13. Classes (counterfactual: Tuesday logit plus move part 0.221*M, or plus the rest): move alone crosses 11 (final 7-4); either alone 6 (4-2); non-move alone 5 (2-3); both needed 1 (0-1). 9 of 23 finals within 0.03 of 0.5.
- Thin-book moves: leader_books 2 in SF_ATL, IND_JAX, LV_NYJ; 0 in HOU_TEN.
- Recorded two cells, family late_flip_mechanism_2026, unresolved_below_power.

## Tried
Production market_move_toward_home at each publication time; per-book moves via sharp_book_movement_features one book at a time.

## Next
Measure a baseline-outlier guard leave-one-season-out on 2020-25 before any fix.

## Open
Defect: sharp_book_movement_features.py:136-139 takes each book's first pre-Wednesday quote as the baseline with no cross-book check. LV_NYJ mybookieag baseline -1.0 against a consensus +1.0 or +1.5 counts +2.5 toward NYJ and the 2-book median 1.75 flips the side. Fix to measure: baseline from the cross-book median at the anchor, or drop a book whose baseline departs from it by a LOSO-derived distance. Feed fields differ from the card because the feed is written by fewer jobs than publish.

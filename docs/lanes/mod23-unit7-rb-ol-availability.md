# MOD-23 unit 7: RB/OL availability from snap counts (inventory)

## Goal
Decide whether RB/OL availability from snap counts is novel as a base-model family on the fair opener test (base 802-701 on 2020-25; 2011-19 graded at the SBR open).

## State
Inventory 2026-10-18, no fits. Verdict: OL is a REPEAT; RB alone is untested.
- acc-u7a (scripts/mod24_u7a.py, family mod24_availability_history): injury-report Out/Doubtful and OL-missing adjusted history. A3 (+OL) -0.107 pooled P+ 0.39 (2020-25 -0.200). Unresolved.
- unit 6b B3 (scripts/mod23_unit6b.py): own-side snap-count absence shares per group, ol = top 5 OL, sk = top 3 WR/TE, df = top 11, own coefficients from prior seasons. +0.20 2020-25 P+ 0.66, pooled +0.03 P+ 0.53. Unresolved. RB is NOT in any group.
- docs/ol_continuity_screen.md: OL-start continuity from snap_counts (2013-25) as cover signal; C1-C4 all unresolved, C2 P+ 0.085. docs/ol_crosswalk_defect.md: pfr_id crosswalk drops OL (0.41% resolved) in players.py features.
- Also touching: docs/injury_value_lost.md and availability_confirmation.md (value lost positive, reliable), docs/absence_dependence.md, roster_availability_flag_features.py, inactives_channel.md (current-game inactives).
- Data: data/players/raw/*/snap_counts.parquet (position, offense snaps, 2013-2025 REG) is post-game, so usable only as history for prior games; pregame current-game availability already comes from inactives/injury captures (src/nfl_ats/inactives_*). Pre-2013 has no snaps (flag 0), as in unit 6.
- Family mod23_unit6b_availability_depth plus u7a/u6/u9 = 19 looks already.

## Tried
B5 run 2026-10-18 (scripts/mod23_unit7.py, artifacts/mod23_unit7/report.json), measured. Leak check 3494 games diff 0.0; base 802-1503 reproduced. 2 looks recorded (unresolved_below_power), family total 21.
- Decisive (flips) B5 v base: 2011-19 85 flips 44-41; 2020-25 80 flips 40-40; pooled 165 flips 84-81.
- B5 accuracy: 2011-19 1145-1086 v 1142-1089 +0.13 pts [-0.58, 0.86] P+ 0.62; 2020-25 802-701 v 802-701 0.00 [-0.63, 0.92] P+ 0.43; pooled +0.08 [-0.42, 0.64] P+ 0.59 (season-block bootstrap).
- B3-only (not a look): 2011-19 -0.09 P+ 0.41; 2020-25 +0.20 P+ 0.66 (805-698); pooled +0.03 P+ 0.53.
- LL/Brier arm minus base (positive = worse): B5 pooled +0.00022/+0.00010; B3 +0.00035/+0.00016.

## Next
Done: RB group adds nothing resolvable over B3; no further availability-from-snaps unit. Arm B forward log is the clean test. (Declaration kept below.) Only untested piece was: B5 = unit 6b B3 with an RB group added (top 2 RB by prior snap share; absence share = sum max(0, expected - actual offense snaps)/sum expected; maps to off rush epa, yards, ats_residual only). Declare before scoring: LOSO by season, coefficients from seasons strictly before the game's season, same leak check as unit 6 (zero after cutoff leaves earlier features identical). Baselines: served base 802-701 and market-only/model-only LL/Brier alongside. Opener grade, 2 cells (2011-19, 2020-25) = 2 new looks (family total 21, state count and family). Reference arm B is not a look. Closing grounds: only whole interval below zero (wrong_sign_resolved) or a proven positive control; else unresolved_below_power via weak-signals record. Prior expectation is small: B3 gave +0.03 pooled, and RB value is mostly priced.

## Open
Is a 2-cell RB-only add worth CPU while sims run? Arm B forward log (docs/lanes/armb-forward-log.md) remains the clean test. No OL re-test: it would repeat u7a A3 and 6b B3.
- Orchestrator 2026-10-18: B5 prep log prints 'raw repro max abs diff 16.49' v features file; base 802-701 reproduced and leak check passed, so the scored arms stand; cause of the 16.49 line unexplained (open).

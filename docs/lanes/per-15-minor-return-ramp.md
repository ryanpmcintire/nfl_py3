# PER-15

## Goal
Non-QB starter returning from 1-3 missed games: snap-share loss in first game back, and whether it is a fitted opener-grade term (ROADMAP PER-15).

## State
Unit 1 done 2026-11-03 17:01 ET via scripts/per15_return_ramp.py (outputs tests/scratch/per15/result.json). Candidates 5267 (2014-25): starter (baseline unit share >= prior-season group median), missed 1-3 consecutive team games, on the injury report during absence, not Out in return week. Only 50.6% play (m1 59%, m2 46%, m3 36%). Played-only loss vs own baseline 6.2 pts [5.3,7.1]; m1 5.7, m2 6.6, m3 7.7 (intervals overlap); starters after a played game lose 5.5 [5.3,5.7] (regression to mean), so no ramp-up beyond control among those who play. By group played-only: WR 8.3, LB 6.5, DL 6.4, TE 5.6, DB 5.6, RB 4.7. The loss is nearly all whether he plays.
Term (LOSO 2020-25, 1503 opener games, 1155 with feature, value = prior-season margin regression of slot-loss, noisy mixed signs): served+x_value dLL -0.00007 [-0.00116,0.00093] P+ .43, flips 78-76, wins 802 vs served 800; unweighted dLL -0.00115 P+ .20, flips 129-147; shuffled control +0.00018. Coefs stable (0.27-0.44 on z; x -0.03 to -0.07). Deadline-view term (Friday after opener).
Looks: 28 snap cells + 3 arms. Recorded unresolved_below_power: per15_x_value, per15_x_unweighted.

## Tried
Candidate gating via Friday status != Out; position values from prior seasons only.

## Next
Position values are noisy (OL/TE/RB negative, WR/DL positive): try snap-weighted points from player value data; Friday-time vs opener split.

## Open
Does not duplicate LEAD-106 ramp (played-only loss equals control).

## Unit 2 (2026-11-14 18:21 ET, Get-Date; ENG-54 crosswalk repair)
The injury-listing lookup used the weekly-roster PFR crosswalk (64.7% of snap rows mapped; an unmapped player defaulted to never listed). scripts/per15_return_ramp.py --xwalk players maps through nfl_ats.players.attach_snap_player_ids with the players table (99.98%); --xwalk roster (default) reproduces Unit 1 exactly. Outputs tests/scratch/per15/xwalk_roster and xwalk_players (result.json). Measured, old v new: candidates 5267 v 7302; played rate 50.6% v 50.4%; snap cells 28 v 32; games with feature 1155 v 1210 of 1503.
- per15_x_value: -0.000066 [-0.001159, +0.000935] P+ .432, flips 78-76 -> -0.000641 [-0.001447, +0.000148] P+ .064, flips 52-45, 807-696 v base 800-703, positive in 3 of 6 seasons.
- per15_x_unweighted: -0.001151 [-0.003741, +0.001525] P+ .198, flips 129-147 -> -0.001675 [-0.003691, +0.000333] P+ .053, flips 108-124.
- shuffled control: +0.000181 P+ .574 -> +0.000066 [-0.001443, +0.001513] P+ .527.
Both cells re-recorded with replace (batch tests/scratch/eng54/per15_xwalk_rerecord_batch.json), unresolved_below_power: both intervals reach above zero, so wrong_sign_resolved is not admissible. 35 looks (32 snap cells, 3 arms). The base in this script is home_cover_probability_at_open (model-only), not the served probability; a served-base retest is still owed (ENG-54 list). Nothing served.
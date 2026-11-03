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

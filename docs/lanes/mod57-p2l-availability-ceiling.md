# MOD-57 P2-L availability ceiling (deliberate leak, positive control)

## Goal
Ceiling of exact who-plays knowledge for the base model at the opener, all/quiet/moved. All numbers are LEAK, not servable.

## State
2026-11-11 10:03 DONE. Script scripts/mod57_p2l_availability_ceiling.py; outputs artifacts/mod57_p2l/ (results.json, cells.json, batch.json, oracle_features_LEAK_NOT_SERVABLE.parquet). 36 cells recorded, family mod57_availability_ceiling, unresolved_below_power.
- Novelty (read): no realised-availability opener oracle existed (PER-14 oracle is season-final unit ratings 803 v 801).
- Base reproduced: 802-701 all, 439-398 quiet, max abs diff to artifact 0.
- Records all / quiet / moved (base 802-701 / 439-398 / 359-294): O1 actives 813-690 / 446-391 / 363-290; O2 snap share 794-709 / 447-390 / 343-310; O3 QB 812-691 / 442-395 / 365-288; ALL 835-668 / 466-371 / 366-287; shuffled control 790-713 / 432-405 / 355-298.
- ALL v base: all +2.20 pts [0.07, 4.31] P+ .977; quiet +3.23 [0.49, 5.97] P+ .989; moved +1.07 P+ .70. O1 alone +0.73 all, +0.84 quiet (P+ .88, .85); O3 +0.67, +0.36.
- Coefficients (scaled units): O1 negative every season (-0.14 to -0.45), O3 negative (about -1.1), O2 POSITIVE (0.31-0.66). Inferred: O2 positive is garbage-time rest after home leads, a margin leak, so ALL is inflated by it.
- O1 nonzero on 100% of games (someone is always out); O3 on 33% (QB 271-222 v base 249-244 there).

## Tried
- Two crashes fixed (snapshot loaders, duplicate QB rows).

## Next
- Optional: O1+O3 only arm to remove the garbage-time confounded O2 (not declared; would be a new look).
- Decision: perfect actives alone are worth under 1 point; availability model cannot reach 57% on quiet games.

## Open
- No-roster team-games 670 of 9804 (no oracle rows there).

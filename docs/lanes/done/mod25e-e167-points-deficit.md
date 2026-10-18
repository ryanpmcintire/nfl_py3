# mod25e-e167-points-deficit

## Goal
Decompose the -1.6 pts/game gap (provisional base XFA+LDR+RGR, label ...ldrg, seeds 11-13: 43.59 v real 45.21) into drives/game, start spots, progress, zone reach, conversion by deepest zone, FG make by distance, after-TD points, defensive/ST scores, safeties, OT; name the carrier with file:line. Offline, no installs.

## State
Scripts (new, tests/scratch/e167/): an.py builds per-game tables (R.pkl real 2009-17 REG incl OT for scoring, qtr<=4 for drives; S.pkl sim play_*_*.parquet s>=2, 3 seeds); rep.py prints sim-real diffs with 400-rep game bootstrap, 90%.
build3.py from e155 is NOT usable standalone: importing e131/ch.py runs a long analysis on import (killed my two build3 PIDs).
Results (measured, 90% game bootstrap, tests/scratch/e167/rep.txt, an2.txt; ~60 cells looked at): pts/game 44.76 real v 43.61 sim by this accounting (-1.15 [-1.67,-0.67]). Offense TD pts -1.08, after-TD -0.59, FG -0.52, def/safety/ST/other +1.03 (other includes PAT after those TDs, so it nets against after-TD). Drives/game 22.37 v 22.66 (-0.28 [-0.40,-0.17]); plays/drive 6.14 v 6.18; plays/game about -2.7. Starts equal (all +0.23 yd worse; after turnover +0.68). Reach opp20 per game -0.40, opp10 -0.33; conversion given zone equal (z0 TD .644 v .643, FG .255 v .253). FG make .835 v .837. Down 1/2 net gain -0.07/-0.15 yd/play; 3rd conv equal. Carrier: volume (fewer snaps/drives, shallower progress), not conversion/kicking. Inferred link: clock_elapsed kept from pre-repick row (mod25e_geo.py:380-386, E160) lengthens plays +1.6 s; GSTOP hook (mod25e_gstop.py) is the untested fix.

## Tried
- Reusing e155/build3 (G_TAG): import side effects, abandoned.

## Next
Test GSTOP 3-seed for drives/game and reach; check safeties +0.038/game and def TD +0.05 (sim high).

## Open
Sim kick/punt return TDs are not in the play log; counted as the residual (sim_games total minus logged).

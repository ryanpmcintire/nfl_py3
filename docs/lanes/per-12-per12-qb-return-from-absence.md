# PER-12 Quarterback return from absence

## Goal
First starts back after 3 or more missed starts as a Tuesday term (ROADMAP PER-12). Family per12_qb_return.

## State
Unit 1 done 2026-11-03 via scripts/per12_qb_return.py (outputs tests/scratch/per12/: events.csv, return_games_opener.csv, result.json). Measured: 174 return events 2009-2025 (74 in 2020-25); missed 3 games 65, 4 to 6 76, 7 or more 33; reason from injury report 83, other status 15, none reported 76. Starter = most dropbacks per team game, regular season.
EPA per dropback vs own pre-absence baseline (QB-clustered bootstrap, dropback weighted): return 1 +0.063 [0.019, 0.107] pp 0.997; return 2 +0.005 [-0.051, 0.060] pp 0.55; return 3 -0.018 pp 0.30; return 4 +0.054 pp 0.91. No rust dip; vs matched backup +0.066 / +0.022.
Opener 2020-25, returns 1-2: record vs opener 55-64-1 (return 1 31-35-1, return 2 24-29); cover margin -1.07 [-3.61, 1.35] pp 0.19. Adjusted line move from prior close +0.41 [-0.64, 1.41] pp 0.80. Move on expected swap value (baseline minus backup EPA times dropbacks) slope 0.05 [-0.06, 0.17] pp 0.79; mean swap value is only -1.06 pts, so there is little to price.
LOSO diagnostic term (6 folds, one look): out-of-sample MSE 165.47 vs intercept-only 165.15, in-sample 164.39 vs 164.49, coefficient negative in all 6 folds; no gain, no term fitted.
Looks counted: 4 EPA cells, 1 record, 1 cover, 1 move, 1 slope, 1 LOSO = 9 in family. 5 cells recorded unresolved_below_power with weak-signals record.

## Tried
Single definition of starter and baseline (all prior dropbacks, no window). Reason is from nflverse injuries status only.

## Next
Baseline for low-sample returners is noisy; try baseline shrunk toward league; backup quality as pre-absence pregame input; timestamps from injury captures not yet used (nflverse injuries have no capture time).

## Open
Opener cover cell is below power (120 games). No served term proposed. Swap value mean is negative because many returners are below their backups' level.

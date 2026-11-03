# MKT-26

## Goal
Move size as a dispersion signal: does the size of the opener-to-deadline move predict how far the margin lands from the line, not which side (ROADMAP MKT-26).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: A big move reflects news that sharpens or scrambles the game; if absolute moves predict larger absolute residuals, the cover probability should be pulled toward 50 percent in those games. Fitted term: absolute move in a shrinkage coefficient on the served logit, fitted leave-one-season-out.

## Tried
Novelty check only. Nearest: MKT-23 (direction of moves, done unresolved), MKT-25 (revision size), mod25d-variance-source; grep 'heterosced' and 'residual variance' found none.

## Next
First unit for one agent: Regress absolute opener residual on absolute opener-to-deadline move with season-block bootstrap and a within-week permutation null; then LOSO add-one of the shrink coefficient on cover log loss. Data: Local lines archive 2013-2025 (free).

## Open
Opener grade only; the deadline move is a feature of the line, never the graded number. Zero crossing closes nothing; one fitted probability selects the side.

## Unit 1 result (2026-11-03 17:20 ET, `date` 5:06 PM run start)
Script scripts/mkt26_move_dispersion.py; artifacts/mkt26/{result.json,record_batch.json}. n 784 REG 2023-25 non-push at deadline line (hs1 = median last quote per book at min(kickoff, Sun 4PM ET); margin = result - hs1). Mean |resid| 10.12, near-push (|m|<=0.5) 4.97%. 583 moved games.
Dispersion (OLS slope controlling |line| and total; game bootstrap; permutation within 5 |line| quintiles): |move| -> |resid| +0.305 pts per point of move [-0.32,0.96] P+ .82, perm p .09, split-half -0.18/+0.88, seasons +.27/+.71/+.10. |move| -> near-push -0.0016 [-.017,.018]. Book disagreement (sd) -> |resid| -1.14 [-4.8,2.6] P+ .26, perm .50; -> near-push -0.014 [-.12,.10]. Direction of |move| positive (wider) but weak; disagreement not.
Shrink (logit x |move| / x disagree interaction, LOSO, vs refit base), DEADLINE view (deadline-view Sunday re-nomination input; base = served lattice at close line, near-0.5 so decisive flips are vs a weak base): ll gain c_move -0.0015 [-.0035,.0004] P+ .06; d_disagree -0.0022 [-.0056,.0013]; e_both -0.0036 [-.0075,.0002]. Decisive (arm right/base right of flips): c 56/68, d 48/87, e 67/99. Gaps oos-in .0022/.0034/.0048. Fitted move coefficient on logit x move +0.5/-0.06/+0.48 (unstable sign across folds).
Tuesday-opener-graded (information arriving later, move unknown on Tuesday): c -0.0023 [-.0056,.0010] P+ .09, decisive 13/22; d +0.0001 [-.0009,.0011] P+ .57, 11/17; e -0.0023.
10 looks (4 dispersion + 3 + 3), recorded unresolved_below_power (family mkt26_move_dispersion). Nothing closed. Caveat: deadline base is the close-line lattice (mean |close-deadline| 0.09 pts), not a re-served deadline read.
## Next
Unit 2: re-serve the lattice at hs1 for the deadline base; extend to 2026 captures; dispersion test on 2013-2025 openers-to-close pairs for power.
- 2026-11-03 17:09 ET (root): two dispersion-slope cells had been written with an invalid unit (margin_points) and broke publish-board; re-recorded with --replace as ats_points (slope of absolute ATS residual per unit of predictor).

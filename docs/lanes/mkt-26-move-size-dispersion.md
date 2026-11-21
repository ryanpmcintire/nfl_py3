# MKT-26

## Goal
Move size as a dispersion signal: does the size of the opener-to-deadline move predict how far the margin lands from the line, not which side (ROADMAP MKT-26).

## State
Units 1 (2026-11-03) and 2 (2026-11-21) done; all 16 cells unresolved_below_power. Mechanism: A big move reflects news that sharpens or scrambles the game; if absolute moves predict larger absolute residuals, the cover probability should be pulled toward 50 percent in those games. Fitted term: absolute move in a shrinkage coefficient on the served logit, fitted leave-one-season-out.

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
## Unit 2 result (2026-11-21 10:11 local; measured, artifacts/mkt26/unit2_result.json)
Source checks: SBR close vs nflverse close r .996, Tuesday close r .998, same sign convention; my band reads match band_read exactly (max diff 0). n 3361 games, 80% moved (SBR 82%, Tuesday 74%).
D1 excess |r| per point of move +0.089 pts [-0.107,+0.277] P+ .80, perm p .21, halves +.04/+.15, 9/13 seasons positive. D2 excess key landing +0.11 pp per point [-1.05,+1.21] P+ .58, perm .42, halves +.47/-.21 pp, 5/13. D3 variance ratio moved/unmoved 1.063 [0.94,1.22] P(>1) .84; SBR 1.16, Tuesday 0.89 (sources disagree).
L1 move-class lattice, opener-graded, n 3288: decisive 89-89 of 178 flips; log loss gain -0.00017 [-.0010,+.0006] P+ .33, perm .86. L4 LOSO logit x move: decisive 3-7 of 10; gain -0.00005 [-.0006,+.0006] P+ .44; interaction coefficient +.024..+.059 positive in all 13 folds (moves sharpen, opposite of the pull-to-50 story) yet no held-out gain; in-sample +.00014, gap -.00019.
L2 re-served deadline lattice 2023-25 opener-graded n 799: decisive 24-18 of 42; gain +0.0008 [-.0014,+.0030] P+ .76, perm .10, all 3 seasons positive. Base 0.6890 vs served opener probability 0.6818 on the same games. L3 deadline-graded n 784: 31-21 of 52; +0.0002 [-.0020,+.0025] P+ .55. Re-serve fidelity vs stored close read: mean |diff| 0.012, max 0.025 (inferred cause: served residual location not stored).
7 looks; 6 cells recorded unresolved_below_power (mkt26u2_*, registry/weak_signals/mkt26_move_dispersion.json); pool loads. Nothing closed, nothing served.
ROADMAP MKT-26 update: Unit 2 done 2026-11-21: 2013-2025 n 3361, move to excess |r| +0.089 [-0.11,+0.28] P+ .80; move-class lattice opener-graded 89-89, gain -0.0002 P+ .33; re-served deadline lattice move-class 24-18, +0.0008 [-0.0014,+0.0030] P+ .76; 6 cells unresolved. Next: 2026 deadline captures for L2.
## Next
Unit 3: extend L2 (re-served deadline lattice, move-class pool) with 2026 captures through the played card; store the residual location to close the 0.012 fidelity gap; L4 sign test of moves sharpening rather than widening.
## Unit 2 declaration (2026-11-21 10:06 local, before outcomes joined)
Adds: 13 seasons of opener-to-close pairs (power) and a re-served lattice at the deadline line (removes unit 1's close-line caveat). Script scripts/mkt26_unit2_dispersion.py, output artifacts/mkt26/unit2_result.json. Members: REG 2013-2025; opener/close from SBR (2013-2021, one book feed) and the pool Tuesday opener / evaluation close (2022-2025); move = |close - open| within source; r = result - close. Line-implied baseline = production lattice (mass_preserving_lattice band read + tilt, imported unedited), walk-forward trailing pool, anchor = close, point = close.
D1 excess |r| (|r| minus lattice E|r|) on |move|, OLS with |close|, total, season dummies; D2 excess key landing (|result| in 3,7,10,14,17 minus lattice mass there) on |move|; D3 variance ratio moved vs unmoved (lane only, no unit). Game bootstrap 2000, within-season permutation of the predictor 2000, split halves, by season.
L1 opener-graded log loss, market lattice (anchor close, point close, line opener) from the full pool vs from the pool of the same move class (moved vs unmoved, natural zero cut, nothing fitted), 2013-2025. L4 LOSO (season folds) logit x |move| shrink on L1's base. L2 re-served deadline lattice 2023-25 (anchor deadline line, point deadline + served residual, production pool), opener-graded, full vs same-class pool; L3 same graded at the deadline line. 7 looks, family mkt26_move_dispersion. Mechanism sign: bigger move, wider margin, pull to 50 helps.
- 2026-11-03 17:09 ET (root): two dispersion-slope cells had been written with an invalid unit (margin_points) and broke publish-board; re-recorded with --replace as ats_points (slope of absolute ATS residual per unit of predictor).

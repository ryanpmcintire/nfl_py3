# mod25d-variance-source

## Goal
Make the neutral/strength generator realistic enough to produce training data: margin variance, strength-to-margin link, count-points coupling. Parent docs/lanes/mod25-synthetic-data.md. Rule: nothing targets mass at 3/7/10/14/17; shape is a held-out diagnostic.

## State
Code scripts/mod25d_variance.py (commands decomp, tilt, grid, gate, ipwcheck, corrdiag, pacediag, ratediag, tiltcoef). Outputs artifacts/mod25d/ (run_*.sh, gate_*.json/txt, grid_*.txt, tilt_coef.json, diag/). Engine and src untouched. Sim cache in scratchpad.
Variants: c = MOD-25c; cr = c + shrunk team-season EPA effects in kernel; cre = cr + epa residualized; crw = cre + inverse-propensity pool weights; crk = crw + clock artifacts cleaned (E1); crt = crk + outcome tilts (E2).

## Tried
Real decomposition (train/eval): margin var 221/205; random-effects noise 165/153, strength-diff var 58/52; possessions var 10.7/9.2; corr(possessions, points) .054/.048.
D1-D3: variance excess is selection leak in the pool (offenses at big leads are +.043 EPA vs -.027 at deficits), fixed by BLUP residualization plus propensity weights (net tilt .044 -> .005). crw gate (scale 1.0, 8x8, real 2018-25): SD 14.34, nonstrength var 195 (177.5), autocorr .148, R2 .028/.063/.053 (real .105/.126/.150), mass3 .123 (.144), le3 .213 (.247), corr(n,pts) .31.
E1 (this round): count-points coupling traced to clock artifacts. 0.15% of pool rows have clock_elapsed above 60 s (519 of 346552 rows above 100 s), 59% of them equal to the remaining game clock (misordered next-play pairing). They add about 211 s per game of fake time; drawn in the sim they end a game early. Fix crk: those rows get the class x zone x score-cell mean of clean rows (threshold 60 s is the physical play-clock bound, not fitted to margin). Neutral decomp 12000 games: corr(n,pts) .307 -> .191 (real .05); plays per game sd 14.9 -> 7.9 (real 9.5-10); plays mean 143.9 -> 149.2 (real 146-149); possession var 9.45 (real 9.2-10.7); quarter 2/4 play-count sd 7.8/8.5 -> 4.2/5.0 (real 4.4/5.5).
E2: outcome tilts (exponential tilt of neighbor weights keyed to team latent; turnover via the grid policy probability). Coefficients = (real train between-team spread of the rate minus spread the yard channel already makes) / (p(1-p) x latent sd), sign from effect correlation with EPA effect (tilt_coef.json). Rates: red-zone TD per play, 3rd/4th-down conversion, explosive play, sack per dropback, turnover. crk grid (loss): .75 47.7, 1.0 14.9, 1.5 43.2 (R2 at 1.5 matches real .106/.126/.161 but autocorr .296 vs .156).

E2 grid crt loss: .5 42.4, .75 7.3, 1.0 42.1, 1.25 108 (min .75). Gate g4 (8 worlds x 8 seasons, real 2018-25; crk at 1.0, crt at .75; tables gate_g4_crk.txt, gate_g4_crt.txt, gate_g3_crw.txt): SD crw 14.34 / crk 14.54 / crt 15.16 (real 14.31, interval 13.97-14.87); nonstrength var 195/199/208 (177.5); autocorr .148/.161/.167 pass; home edge pass; R2 w1-4 .028/.036/.076, w5-9 .063/.065/.109, w10-18 .053/.062/.094 (real .105/.126/.150); mass3 .123/.120/.120 (.144); mass7 .078/.076/.075 (.085); mass14 .032/.035/.033 (.051); le3 .213/.212/.213 (.247). Tilted rate spreads (sd, real train / crt .75): conv off .036/.028, expl off .0126/.0082, sack off .015/.010, tov off .0043/.0030.

## Next
Verdict: not ready. Fails: margin SD (crt), nonstrength variance 208 vs 177.5, R2 w10-18, all shape checks (not converging). Next mechanism: score-state reversion. Neutral crk decomp: late slope of future margin on current lead -.33 at 300 s vs real -.10/-.08, -.26 vs -.10/-.08 at 900 s, early slope at 2700 s -.10 vs +.09/+.06; cov_total -87 vs -40/-64. Test whether remaining pool selection by score state (policy swaps and clock cells conditioned on score) makes leads revert too fast; fit to that slope, not to margin shape. Then re-run grid and gate.

## Open
Red-zone TD defense spread unstable (train .022, eval 0.0): tilt may be overfit. Tilts load all rates on one latent per side (perfect cross-rate correlation, inferred). Remaining corr(n,pts) .19 vs .05 unexplained (game-level pace heterogeneity missing: plays covariance between quarters -.19 sim vs +1.5-2.1 real). Tilt spreads attenuate below target (swaps and scale .75).

# mod25d-variance-source

## Goal
Find where excess within-game margin variance comes from (neutral sim 216 vs real strength-removed noise 165 train / 153 eval), fix mechanistically. Parent docs/lanes/mod25-synthetic-data.md. Rule: nothing targets mass at 3/7/10/14/17.

## State
Code scripts/mod25d_variance.py (commands decomp, tilt, grid, gate; imports mod25c_noise; engine and src untouched). Outputs artifacts/mod25d/ (decomp_d2.json/txt, tilt.csv, grid_cr.*, gate_g1.*, run_all.sh). Sim frame cache in scratchpad.
Measured decomposition (real train / eval / neutral sim c): margin var 221/205/216; random-effects noise 165/153 and strength-diff var 58/52 (hfa 2.5/1.6). Possessions per game mean 22.7/21.3/22.6, var 10.7/9.2/13.0; per-possession points var 7.9/8.3/7.4, TD share .201/.227/.184 (sim LIGHTER); corr(possession count, total points) .054/.048/.27; lag-1 across-team corr -.044/-.023/-.044.
Candidates: (b) possession count var 21% high, count-points coupling too strong, but signed mean 0 so no margin effect; (c) refuted (sim tails lighter); (d) refuted as missing: late reversion slope sim -.20/-.31 at 900/300 s vs real -.10/-.10 (sim stronger). Quarter margin variances sim 208.8 vs real 231.6 sum, so sim's variance is not too big per quarter; excess is dependence: centred possession covariance at lags 5-21 about +7 sim vs about -41 real nonstrength (inferred, strength share spread evenly), early-game momentum slope at 2700 s +.045 sim vs about -.09 real nonstrength.
(a) confirmed as selection leak, not per-play variance: team effects explain only .6% of EPA and .4% of yards variance, but pool plays at leads >13 come from offenses +.043 EPA vs -.027 at deficits >14 (sd of advantage .078; tilt.csv). Neutral sim draws that tilt as if it were in-game momentum.
Fix implemented: variant cr = c + shrunk (BLUP) offense/defense team-season EPA effects (fit 2009-2017) replace game-level ratings in kernel weights and yard shift. Strength grid (loss, same as MOD-25c): 0.75 71.2, 1.0 44.8, 1.5 5.5, 2.0 11.3, 3.0 94.4 (min 1.5; 1.5/0.75/1.0 declared after edge result). Held-out gate g2 (scale 1.5, 8 worlds x 8 seasons, real 2018-25) vs c (mod25c g2): SD 15.63 vs 16.41 (real 14.31); nonstrength var 223 vs 247 (real 177.5); mass3 .110 vs .102 (.144); le3 .190 vs .182 (.247); autocorr .147 vs .154 (both pass); R2 w10-18 .098 vs .101 (real .150, now under). Partial: closes about a third of the excess.

## Tried
Neutral variant c only; no balanced-pool neutral test (neutral path ignores ratings).

## Next
Residualize scoring on the drawn play (TD/FG share by team-season) and reweight pool for score-state tilt; re-measure early-game momentum slope (+.045 sim) with decomp.

## Open
Scoring outcomes (points on drawn play) are not residualized, only yards and neighbour weights.

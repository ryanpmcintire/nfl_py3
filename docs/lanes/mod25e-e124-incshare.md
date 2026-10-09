# mod25e E124 incomplete-pass share deficit along the draw chain (diagnosis only; no full sim, no script edits)

## Goal
Locate the stage(s) that lose incompletions (sim inc share of nonend passes ~1.4 pts below real at every hs bin; E123) in base ...kekudwtpq2ugky; specify the fix. Scratch tests/scratch/e124/.

## State (DIAGNOSED, 2026-10-09; measured unless labelled)
- run1.pkl = 150 games season 2 (22,035 plays); drv.py now uses sys.monitoring (no wrapper frames; wrappers broke sys._getframe(1) in tdc), records per-stage input/output; an1.py prints pre/post flows per stage. Games take 74 s; init 341 s. No process left.
- Share of inc in nonend passes: pool all .3634, pool excl pen rows .3681; pick-level expected (kernel+IPW+tilt) .3653 vs uniform-excl-pen .3631; realized engine row .3669. PASS_W weight std .002 within state (no effect). ADJ/picks: no loss. 4th down: none.
- Stage table (inc share before -> after): engine .3669 -> F3 (sim09_f3.py pol :674-679 -> Overlay.apply :550-640) .3544 [426 inc leave, 5 enter of 3,883 inc; -1.25 pts] -> GZ (mod25e_gz.py) .3544->.3579 [97 leave/116 enter, +0.35] -> every other stage 0/0. Final .3579 (SE .005 at 150 games; flows are paired and exact).
- F3 leavers: 340 -> code 6 (accepted uncounted penalty), 82 -> nonzero pass (counted DPI-type). Hazard is outcome-blind (inc 11.4% v pass+ 10.7%); the asymmetry is acceptance: defensive-group penalties on inc accepted 1.00 (in_def 181, pre_def 51) v 0.79/0.63 on pass+.
- Defect (read+inferred): the base pool is real NON-flagged survivors (f3_install :662-666 IPW *= ~pen), already net of real accepted wipes/conversions (survivor inc share .3681 v unfiltered .3634 after counted rows are re-added), and Overlay then removes incs a second time at the same fitted hazard x acceptance. Real excess is -0.47 pts; sim F3 is -1.25 pts, excess ~ -0.8, plus E123's remainder.

## Fix (specified, not applied)
Inverse-probability the survivor pool by F3's own removal probability: at f3_install, IPW_j *= 1/(1 - r_j), r_j = P(row j is wiped or converted by Overlay) = sum over groups of hazard_g(x_j) (team and crew multipliers at their mean 1) x E_type[ accept(gv_j, type) x (1 if uncounted wipe or counted conversion else 0) ], computed once per pool row from row j's own real state, outcome yards and the typepmf (no constant, derived from the already-fitted hazard/accept/typepmf in f3_params.json). Then clear nn_weight_cache_cond (already done there). Expected (inferred) final share ~ .364 v real .363. Score: rerun drv.py 150 games, an1.py; check inc/game, plays/game, and held-out bins hs<=120.
Fallback if owner rejects reweight: LOSO logit P(inc|pass) on down, log dist, yardline (+58e-4 over down+dist), hs, as a pass-class row reweight; 7 specs = 7 looks, probability_positive 1.00; coefficients per fold not yet printed.

## Tried
Static suspects PASS_W, ADJ, endgame DECIDE: no loss (measured). Wrapper-frame driver (failed), monitoring driver (works).

## Next
1. Implement r_j in sim09_f3.py f3_install (owner of scripts/ edit task), smoke 150 games with drv.py, confirm F3 pre-effect share change ~ -0.47 pts not -1.25.
2. Check GZ +0.35 pts (97/116) is intended.

## Open
r_j for counted conversions needs gain_dec per row (vectorised, ~150k rows x type pmf); 120-game CI on final share not significant alone.

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

## F3W reweight attempt (2026-10-09, measured; scratch tests/scratch/e124f/)
- Implemented behind F3W=1: sim09_f3.removal_prob (hazard x accept x wipe-or-convert from row's own state, type pmf sampled WDRAWS=1 per group, nom drawn from the type pmf, DPI at air yards) and f3_install IPW /= (1-r); mod25e_crH label "fw". F3W unset: byte-identical IPW path (git diff). Full-pool r costs 294 s (346,552 rows, mean r over r>0 .080); cached to e124f/r_0_346552.npy for drv.
- 150 games (22,266 plays): engine row inc share .3669 -> .3677 (+.08 pt); F3 still removes 420 inc (enter 1): .3677 -> .3564; GZ +109/109 -> .3576; final .3576 v base .3579 v real .3634. NO FIX: reweight by total removal r does not touch the asymmetry because r(inc) ~ r(pass+) (30k-row check: .095 v .084); the loss is class-specific (inc leaves class on wipe AND on counted conversion, pass+ stays pass+ on conversion).
- Class mix final (real/base/fw): run .3315/.3360/.3332, pass+ .2845/.2834/.2851, inc .1624/.1580/.1587, pass_end .0424/.0402/.0383, run_end .0166/.0177/.0168, noplay .0617/.0619/.0634. No-play +.15 pt, pass_end -.19 pt vs base.
- Reading: weighting the pool cannot fix it; overlay's inc-class departure (10.8%) exceeds what real S->real implies (-0.43 pt). Next: check the overlay's own hazard/accept by outcome class against real (penalty rate on inc v completion among all real snaps incl. flagged), or fit the fallback LOSO inc-share reweight.

## E124c outcome-blind hazard (2026-10-09, measured; scratch tests/scratch/e124c/: real.py stub.py loso.py)
- Real 2009-17 REG dropbacks (179,184; underlying outcome from incomplete_pass or desc): committed flag INC .1104 v COMP .0586; in_def INC .0686 v COMP .0235; accepted wipe (no_play) INC .0898 v pass+ .0330; counted conv INC .0085 v pass+ .0150. Frame yards_gained = field-position diff (sim04_engine.py:644), so counted inc leaves the y==0 class in real too. Shares: underlying .3733, survivors .3607, logged .3571 (S->logged -0.36).
- Overlay on 60k pool rows (stub of removal_prob split wipe/conv): inc wipe .080 conv .015; pass+ wipe .069 conv .015 v real pass+ wipe .033. Hazard sim09_f3.py:630/:518 is a function of x only (ispass, deep), outcome-blind; real in_def hazard is 2.9x on inc. LOSO (9 seasons) inc coef in_def +0.864 (sd .016, 9/9 +), LL gain .0031; pre_def +.058, in_off +.048, pre_off -.148 (tiny).
- Mismatch: F3W IPW (:747-748) uses blind r (inc .095 v pass+ .084 v real .098 v .048) so U share barely moves (+0.08); blind overlay then over-wipes pass+ ~2x and the loss is inc-specific.
- Fix: add underlying-incomplete feature (and x deep) to hazard_fit/fx/Xg/Overlay.haz/removal_prob; pool outcome known per row; IPW uses same r. Then S->logged ~ -0.36 not -1.25.

# E159 X2 strength excess decomposition

## Goal
Name what carries X2 strength_re_var 61.8 v real 58.0 (B 56.1, 3 seeds). Offline only; no edits to existing scripts/; scratch in tests/scratch/e159/ (lk.py, lk2.py, lk3.py, lk4.py, logs, lk.pkl).

## State (measured unless marked)
- E159-E159c summary (about 100 looks, 4 labels, unresolved_below_power, no refuted mechanism): X2-B drive-point gain is offense only (+0.52 [+.45,+.60]); in-sample team-season def slope was biased up about 40 by same-game ratings; per-channel sim v real-held-out slopes show no channel carrying a missing half; source-row def transfer slope 0.464 v off 0.768 traces to one scalar kernel h on both axes (sim04_engine.py:524, mod25d_variance.py:934, sim04_engine.py:35; same pattern mod25e_geo.py:456-460, mod25e_gz.py:103-105, mod25e_draw.py:32-35, mod25e_rulestr.py:108-110). Hook scripts/mod25e_axish.py default off (def diff times std(eo)/std(ef) about 1.85); deferred crH install and pass bar: in mod25e_crH.py init wrap inspect.getsource with mod25e_axish.patched_source(dv, inspect.getsource) under a new cfg letter, then mod25e_axish.install(dv, eo, ef) with eo, ef = dv.blup_rows(pbp, trans, dv.TRAIN); bar: def transfer 0.70-0.85 with off 0.768 +/-0.02, held-out def slope not above -62.7, strength_re_var 58.0 +/-4, drive off slope >= 5.18, GZ pass off slope in [.35,.58], E154 guards unchanged, the four other kernel sites patched first.

## E159d (this round): like-for-like held-out (lk.py)
Same fit_play_effects (mod25d_variance.py:249) on odd-week plays, margin of even-week games and reversed; sim epa = source-row epa, margin = play-point sums; 3 seeds x 8 worlds x 6 seasons (144 seasons); season-block bootstrap 90%. Real reproduces 59.3 / -50.0.
- Slopes off/def (sim minus real, 90% interval, P(sim>real)): B 53.3/-62.0 (-6.1 [-11.6,-0.5] .05; -12.0 [-24.9,0.2] .06); X2 54.9/-61.6 (-4.5 [-9.9,0.9] .08; -11.6 [-24.6,1.1] .07). X2-B +1.6 [-0.2,3.8] off (P .94), +0.4 def (P .52). So the E159c offense overshoot (65.9 v 59.3) was the true-latent v noisy-rating mismatch; like for like sim offense is BELOW real and sim defense is stronger than real, in both B and X2.
- strength_re_var (dv.random_effects on play-sum margins, per season mean): real 55.8; B 56.2 (+0.4 [-6.1,6.7] P .52); X2 65.0 (+9.3 [3.0,15.6] P .988); odd-half X2 +9.7 [0.8,19.0], even-half +10.3 [0.3,20.4].
- Variance decomposition of held-out team-season margin (var_m / explained by held-out ratings / residual): real 53.3/14.9/38.3; B 53.9/17.3/36.6; D1 51.8/18.5/33.3; X 55.3/19.9/35.4; X2 58.5/22.4/36.1. X2-B explained +5.1 (P 1.00), residual -0.5 [-1.9,1.0] (P .29). Held-out offense rating sd real .0597, B .0658, D1 .0758, X .0760, X2 .0744; defense .0290/.0309/.0300/.0312/.0320.
- Reading (inferred): the X2 strength excess is rating-linked (rating spread x slope), not residual margin and not slope; the offense rating spread step B->D1 (.0658 to .0758, label suffix g2rtgw13 = Geo, RTD, GZT cols 1,3) carries it, with X and X2 steps adding explained variance (+1.4, +2.5) at the same spread. Source of spread is the offense source-row kernel/Geo-GZ sites above (mod25e_geo.py:456-460, mod25e_gz.py:103-105); component-level file:line attribution not run. No no-constant fix identified; no hook written. Status unresolved_below_power: X2-B explained-variance is resolved, the cause is not.

## E159e: ablation B -> g2 -> g2rt -> D1, seed 11 only (lk5.py, runs run_g2.sh/run_g2rt.sh, labels ...ugkyg2 and ...ugkyg2rt), 48 seasons, season-block bootstrap 90%
Held-out (odd/even) per season: offense rating SD / defense SD / team margin var / strength_re_var.
- B .0677 / .0316 / 56.7 / 59.2; g2 .0640 / .0237 / 44.2 / 42.9; g2rt .0618 / .0274 / 44.7 / 41.0; D1 .0753 / .0283 / 51.1 / 54.3.
- Steps: GEO=2 off SD -.0037 [-.0051,-.0023], RE var -16.4 [-20.4,-12.4]; RTD off -.0022 [-.0037,-.0007], RE -1.8 [-5.2,+1.6]; GZT=3 cols 1,3 off SD +.0135 [+.0116,+.0153], def +.0009, var_m +6.4 [4.1,8.7], RE +13.2 [9.3,17.0] (P 1.00). D1-B off +.0077 [.0061,.0092].
- Carrier: GZT tilt. Geo and RTD narrow strength; GZT restores it past B in offense SD. Mechanism (read): mod25e_geo.py:413-416 tilt_factor = exp(T[rows][:,cols] @ (AO*(o_rt-LO)+AD*(d_rt-LD))) multiplies kernel weights (mod25e_geo.py:456-460, mod25e_gz.py:103-105 for non-Geo draws), adding a rating-linear reweight of drawn rows on top of the Gaussian kernel, so offense rating shifts outcome distribution of every drawn row on cols 1,3 and team means spread with the rating (inferred: rating-linear gain stacked on the kernel's own rating dependence). Fit constants AO/AD/LO/LD/T come from ns TILT_*; not re-derived here.
- Geo narrows because it redraws violators (mod25e_geo.py:309-313, 447-460 with GEO=2 always) under the same kernel, shrinking row-level rating dependence (inferred). RTD (mod25e_rtd.py:158-195) only injects half-opening return points, spread effect small.
- No no-constant fix evident; no hook. Unresolved_below_power for cause of GZT widening vs real (real RE var 55.8; D1 seed 11 54.3 is within noise; X2 excess is the later gx2 step per E159d).

## Tried
Cell/drive regressions; split-half slopes; 16-channel per-play slopes; kernel read; like-for-like held-out slopes, RE var and decomposition for B, D1, X, X2 (looks added this round: 2 slopes x 2 labels, 3 RE x 2, 5 labels x 3 decomposition stats, about 40; family: held-out comparison).

## Next
Record E159d+E159e family with weak-signals record (about 40 + 16 looks). Next: ablate GZT3 cols 1 vs 3 and AO/AD split (needs new runs); X2 gx2 step vs rating spread. Hook run waits for in-flight runs.

## Open
Does cols 1 or 3 of GZT carry the widening, and is the tilt gain fitted LOSO?

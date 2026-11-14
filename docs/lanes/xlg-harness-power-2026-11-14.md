# XLG harness power (MOD-36 instrument on the college and pooled harnesses)

## Goal
Say what a no-information term produces, and how big a true effect must be to show, on the college-only and NFL-plus-college harnesses, beside MOD-36's NFL figure (1,503 games: about +1.6 points). An instrument; it closes nothing.

## State
Done 2026-11-14 16:17 local (**measured**): `scripts/xlg_harness_power.py` (788 s, one thread, seed 36, streams [36, 1 add-one | 2 shared, template, cell]); `artifacts/xlg_harness_power/` run.log, summary.json, draws.parquet, weak_signals_batch.json. 200 draws; templates short-week side / rest difference, shuffled within season.

Harnesses (**read**, counts **measured**):
- Shared 14-input (XLG-10, XLG-13): `scripts/pooled_screen.py`; ridge alpha 10 refit walk-forward by week, opener as input and grade (fixed harness, commit 35f7f070; earlier numbers used the closing line); the candidate is a 15th input. NFL 2020-25 1,503 games (base 754-749); college 2013-25 without 2020, 4,702 (2,370-2,332); pooled = inverse-variance mean, 6,205. The script reproduces the XLG-10 short-week arm exactly (23-14, 51-31, pooled +0.466 P+ .993).
- College-first (XLG-14): `scripts/xlg14_family_screen.py`; 8,925 college games 2012-19, 2021-25; one logistic term on a leave-one-season-out ridge cover probability at the opener. Its NFL confirmation is the served add-one MOD-36 measured. XLG-09 (done) grades NFL only.

Null mean [5th, 95th], then the smallest planted size (logit per SD = true points) with P+ >= .90 in 80% of draws; short-week / rest:

| Harness (games) | Accuracy pts | Log loss | Changed; record; losing | False alarm acc / LL | Seen by log loss | Seen by accuracy |
|---|---|---|---|---|---|---|
| NFL served add-one, MOD-36 (1,503) | -0.17 [-0.74,+0.47] / -0.21 [-1.00,+0.60] | -.00038 / -.00047 | 37; 16-20; 67% / 50; 22-29; 68% | 1.5-4% / 0-0.5% | .265 = +1.57 / .262 = +1.64 | .265 = +1.57 / .315 = +2.11 |
| NFL arm, shared 14 (1,503) | +0.12 [-0.47,+0.87] / +0.13 [-0.60,+1.07] | -.00011 / -.00010 | 35; 19-17; 39% / 49; 26-23; 36% | 11.5-14.5% / 3-4% | .178 = +1.52 / .167 = +1.76 | .208 = +1.71 / .194 = +2.09 |
| College arm, shared 14 (4,702) | +0.08 [-0.23,+0.43] / +0.09 [-0.30,+0.45] | -.00014 / -.00020 | 93; 48-46; 34% / 137; 69-65; 32% | 12-13% / 0.5-1% | .127 = +1.05 / .111 = +1.07 | .104 = +0.83 / .121 = +1.23 |
| Pooled, shared 14 (6,205) | +0.08 [-0.15,+0.41] / +0.08 [-0.22,+0.41] | -.00012 [-.00036,+.00004] / -.00013 [-.00035,+.00010] | 140; 72-68; 34% / 192; 102-93; 30% | 15-16.5% / 1% | .102 = +0.83, LL +.0012 / .111 = +1.06, +.0015 | .102 = +0.83 / .097 = +0.90 |
| College add-one (8,925) | -0.01 [-0.22,+0.26] / -0.01 [-0.26,+0.34] | -.00006 [-.00017,+.00010] / -.00006 [-.00018,+.00012] | 158; 80-80; 56% / 227; 112-114; 57% | 6.5-8.5% / 0% | .086 = +0.69, +.0009 / .109 = +1.09, +.0014 | .082 = +0.68 / .082 = +0.78 |

Detection by log loss at the sizes units fitted: .023 per SD 1-4% (college add-one and pooled); .044-.048 9-18%; .082 71-74% add-one, 55-59% pooled; .146 96-100% (NFL: 0-12% through .082, 8-41% at .146). XLG-10's short-week arm in its null: pooled +0.47 P+ .993 is the 96.5th percentile on accuracy, 18th on log loss; college 95th / 36.5th; NFL 85.5th / 20th.

Transfer arithmetic. XLG-13 (**measured** from report_fixed.json): 4 estimator-swap pairs, served = -1.90 + 0.53 x screen, Pearson +0.28 (P+ .85), sign agreement 1 of 4, intervals [-1, 1]; unresolved. **Inferred**: at that slope the pooled screen's smallest visible gain (+0.83 to +1.06) is +0.44 to +0.56 served points, a third of what the NFL look confirms; +1.6 served needs a +3.0 screen gain. Pass and confirm (equal size in both leagues): >= .26 per SD about 80%; .146 8-41%; .082 at most 9%; <= .05 at most 2%. If the college size is a third of the NFL size, an NFL .26 effect passes 71-81% and an NFL .15 effect at most 18%; XLG-14 unit 1 read 1/30 with the opposite sign (**read**).

Answer: college-first lowers the visible effect from .26 to .08-.11 logit per SD (+1.6 to +0.7-1.1 points), a factor 2.4-3, the square root of the game ratio. It is the default first screen for families whose input exists in both leagues under one construction, since only it resolves .08-.15. It does not reach this season's sizes (.02-.05) and does not lower what the NFL look confirms. Cost: a second league's inputs, magnitude differences, an unresolved 14-input-to-served relation.

Recorded: 18 null cells, family `xlg_harness_power` (6 college add-one, 6 shared college arm, 6 shared NFL arm; category control); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## How to use it
1. Screen on the college add-one and read log loss against the parameter cost (-.00006) or the percentile in the matched null (add a dict to TEMPLATES); state that only .09-.11 per SD shows 80% of the time.
2. On the shared 14-input harness never take accuracy P+ at face value: P+ >= .90 occurs in 12-17% of no-information draws and the null mean is +0.08 to +0.13 points; use log loss or the null percentile.
3. The NFL look after a college pass states a sign and a percentile; below +1.6 points it stays `unresolved_below_power`.

## Tried
- MOD-36 functions loaded from its source without the trailing `main()` (an import reruns it and overwrites `artifacts/mod36`); its plant re-implemented with cached tables, same random stream.
- Shared harness refit re-implemented with cumulative sums; column planted on every training row (4,353 NFL, 9,116 college), which is why the NFL arm sees .17-.21 where the served add-one sees .26.
- Cover-flag plant as MOD-36 unit 2; the ridge fits margins, so fitted log-loss gain is 70-85% of the true gain.
- Pooled null cells and all planted cells unrecorded (registry league is nfl or cfb; planted cells would enter the pool).
- Looks: 4 new arms x 2 templates x 3 metrics x 2 levels = 48 detectable sizes, 24 null rows.

## Next
Margin plant (MOD-36 unit 3) on these harnesses; planted magnitude ratios between leagues; XLG-13 pairs for added terms; XLG-14 declarations state the visible size and expected NFL confirm share first.

## Open
- 200 draws: shares resolve to about 5.5 points, the 80% size to one grid step. Positive-direction control only.
- The shared-harness accuracy null is positive and its P+ over-alarms; cause not measured (base at 50.2-50.4% with log loss worse than a constant).

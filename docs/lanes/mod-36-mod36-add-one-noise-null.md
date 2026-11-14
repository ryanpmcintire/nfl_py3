# MOD-36 Add-one noise ensemble

## Goal
Say what a no-information term produces in the add-one test on the served pick probability, and how big a real effect must be to show (ROADMAP MOD-36). An instrument; it closes nothing.

## State
Unit 2 done 2026-11-14 15:24 ET (**measured**): `scripts/mod36_addone_noise_floor.py` (190 s, single thread, seed 36, streams [36, template, cell]); `artifacts/mod36/` run.log, summary.json, draws.parquet. Base `out_of_season_home_probability`, pick_probability 20261114T184614Z, opener 2020-25, leave-one-season-out, 2,000-draw game bootstrap. The harness reproduces the three units exactly (11-13, 17-26, 36-41).

Null: real column shuffled within season, 200 draws. Mean [5th, 95th]:

| Template (n; fit vs reference) | Accuracy pts | Log loss | Changed 5/50/95; median record; losing | P+ acc <=.10 | P+ LL <=.10 |
|---|---|---|---|---|---|
| Short-week side, 3 values, 12% nonzero (1,503; 1 coef vs served) | -0.17 [-0.74,+0.47] | -.00038 [-.00128,+.00079] | 14/37/74; 16-20; 67% | 18% | 38% |
| Rest difference, 15 values, 34% nonzero (1,503; 1 coef vs served) | -0.21 [-1.00,+0.60] | -.00047 [-.00159,+.00061] | 18/50/117; 22-29; 68% | 21% | 38% |
| Questionable habit, continuous (1,503; intercept + coef vs served) | -0.27 [-1.13,+0.54] | -.00048 [-.00157,+.00058] | 32/76/187; 36-42; 68% | 17.5% | 41% |
| Playoff leverage, weeks 10-16 (576; coef vs intercept-only) | -0.90 [-2.78,+1.56] | -.00110 [-.00358,+.00158] | 23/50/103; 22-28; 78% | 29% | 44.5% |

Medians: accuracy -0.20/-0.27/-0.27/-1.04. Brier means -.00019/-.00023/-.00023/-.00053. P+ >= .90 on accuracy in 1.5-4% of null draws, on log loss 0-0.5%. Mean log loss is the cost of the fitted parameter; 1/(2 x training games) gives .00040 and .00104 (**inferred** check).

Planted control: column resampled within season, tilted by the realised cover so the true effect is b logit per SD; grid = the six sizes the three units fitted (.019-.146 per SD: full-sample and largest fold) plus 1-10 null coefficient SDs. Smallest size reaching P+ >= .90 / .95 in at least 80% of 200 draws, with the true gain (true coefficient applied):

| Template | Log loss .90 | Log loss .95 | Accuracy .90 | Accuracy .95 |
|---|---|---|---|---|
| Short-week | .265 = +1.57 pts, +.0080 | .265 | .265 | .265 |
| Rest | .262 = +1.64, +.0079 | .262 | .315 = +2.11, +.0110 | .367 = +2.75, +.0148 |
| Habit | .242 = +1.36, +.0069 | .291 = +2.10, +.0098 | .339 = +2.70, +.0129 | .387 = +3.24, +.0167 |
| Leverage (576) | .393 = +3.62, +.0176 | .472 = +5.15, +.0265 | .472 | .472 |

At the sizes units fitted, detection at P+ >= .90 is 0-12% through .082 and 8-41% at .146; false alarms at zero are 1-5% (accuracy), 0% (log loss).

Today's results, percentile in the matched null (accuracy / log loss / record / P+ accuracy / coefficient): short-week -0.13, 11-13: 56 / 46 / 56 / 54 / 68. Habit -0.60, 17-26: 27 / 60 / 27 / 15 / 35 (vs intercept-only, 13-24, P+ .032: 11th). Leverage -0.87, 36-41: 56 / 65 / 56 / 62 / 88.
Licence: all three sit inside the 5-95 band of a no-information term; a small negative and a losing record are the null's usual output, so they are no evidence of a wrong sign. They are no evidence of absence either: the control only detects effects about 3 to 13 times the fitted sizes. All three stay `unresolved_below_power`; nothing closed.

Recorded: 12 cells, family `mod36_addone_noise_floor` (4 templates x 3 metrics, category control); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Use in future packets
1. Report a candidate's percentile in its matched null (add one dict to TEMPLATES), never its raw sign: P+ <= .10 on accuracy occurs in 18-29% of no-information terms, a losing changed-pick record in 67-78%.
2. Compare log loss against the parameter cost (-.0004 per parameter at 1,503 games, -.0011 at 576), not zero.
3. State power: at 1,503 games only a true effect of about .25 logit per SD (+1.4 to +1.6 points) is seen 80% of the time, at 576 games .39 (+3.6); smaller effects stay unresolved and neither null nor control closes them.

## Tried
- Unit 1 (2026-11-03): 50 N(0,1) draws per K on the 88-input ridge refit, family mod36_noise_floor; still the read for refit-harness candidates, not for served-base add-ones.
- Differs from positive-control-power: that curve measured the retrain evaluator on two flag controls across bootstrap precision; this is the offset add-one harness, null matched to candidate marginals, power by true logit size.
- Leverage is not heavy-tailed (excess kurtosis -0.08; short-week 5.4, rest 4.0); its distinguishing property is 576 games.
- Planted cells unrecorded: the pool takes every unresolved cell whatever its category (**read** src/nfl_ats/weak_signals.py:1583), so a constructed +1.6-point cell would enter the pooled estimate; detection shares have no unit.
- True gain at small planted sizes on the 576-game frame reads -0.2 to -0.5 pts: 2-3 fixed coin-flip games, not the term.

## Next
Finer grid between .19 and .29 per SD; a declared negative-direction control; detectable size for a pooled family of small terms.

## Open
- 200 draws: percentiles resolve to 0.5, detection shares to about 5.5 points; the 80% size is known to one grid step.
- Control is positive-direction only, conditional on realised covers.
- The 12 null cells (and unit 1's) sit in the pools as control category; excluding controls from pooled estimates is a src decision not made here.

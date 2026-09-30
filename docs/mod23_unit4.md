# MOD-23 unit 4: blend and shrink on the model-alone opener test

Script `scripts/mod23_unit4.py`; outputs `artifacts/mod23_unit4/report.json`, per-arm per_game parquet, `weak_signals_batch.json`. Numbers **measured** this session. Baseline reproduced through the same machinery (w=1): 802-701 on 1,503 decisive games, identical to the artifact.

## Declared before scoring (3 outer arms, one family `mod23_unit4_blend_and_shrink`)
Each arm yields one predicted margin-vs-line that goes through the unchanged served mapping (gaussian_median, discrete reader, home-side offset). The out-of-time residual set is refit from the same 80/20 split on the arm's own predictor. Inner walk-forward for season Y uses the four seasons before Y, each fit only on earlier seasons; criterion **margin MSE on ats_margin**; ties go to the larger weight.
- (a) blend_compact_net: w*served + (1-w)*compact_net (60 columns), w in {0,.25,.5,.75,1}.
- (b) blend_unit1_trimmed: same with the unit 1 greedy family/alpha choice for Y (`artifacts/mod23_unit1/selection.json`). Inner fits reuse that choice.
- (c) shrink_served: s*served, s in {.25,.5,.75,1}.
Looks: 3 outer arms, 84 inner grid looks (30+30+24).

## Results (baseline 802-701, LL 0.6969, Brier 0.2517; even 0.6931/0.25; 107 season-week blocks)
| arm | record | diff pts | 95% CI | prob_positive | LL | Brier | picks changed vs served |
|---|---|---|---|---|---|---|---|
| blend_compact_net | 785-718 | -1.13 | [-3.20, 0.81] | 0.131 | 0.6944 | 0.2506 | 195 |
| blend_unit1_trimmed | 785-718 | -1.13 | [-3.26, 0.92] | 0.134 | 0.6928 | 0.2498 | 321 |
| shrink_served | 785-718 | -1.13 | [-3.61, 1.32] | 0.176 | 0.6933 | 0.2501 | 333 |

The three identical records were recomputed from the pick columns (each 785 correct, 0 pushes); the pick sets differ (79% agreement between a and c). Coincidence, not a shared frame.

Chosen weights: every season chose the grid edge. Blends: w=0 in all six seasons (pure alternate). Shrink: s=0.25 in all six. Arm (a) therefore reproduces unit 3 compact_net exactly (785-718, residuals identical). Boundary choices mean the grid did not bracket the optimum; the MSE-best served weight is at or below the lowest tried.

Per-season accuracy diff pts (2020..2025): a -0.5,+0.8,-4.4,+1.5,-5.3,+1.1; b +0.5,0,-4.0,+3.4,-4.1,-2.2; c 0,0,-0.4,-2.3,-3.0,-0.7. Per-season LL diff (arm minus base) negative in 4/6 (a), 4/6 (b), 5/6 (c).

Inner vs outer MSE gain over served (2020..2025): a inner 2.04,1.58,1.48,1.17,1.37,1.12, outer 0.52,2.86,1.32,1.84,0.56,2.05; b inner 7.82,6.74,4.57,4.06,2.83,3.51, outer 1.94,2.21,2.04,1.09,2.15,-0.08; c inner 7.33,5.84,3.65,3.72,2.22,2.68, outer 1.74,0.70,5.21,-0.47,2.54,2.00. Out-of-sample MSE gain is positive in 16/18 seasons-arms but smaller than inner (shrinkage of the selection gap), and it does not convert to picks.

Reliability, five equal-count bands of pick-side probability (mean p / realized): served 0.509/.545, .529/.497, .551/.561, .581/.540, .631/.525. a .509/.482, .527/.517, .548/.505, .572/.557, .621/.551. b .508/.482, .523/.507, .539/.555, .559/.520, .593/.548. c .506/.518, .518/.520, .531/.518, .546/.527, .572/.528. No arm is monotone; top band still overconfident in all.

## Reading
- Shrinking the served lean to 0.25 improves LL to 0.6933 (even is 0.6931) and Brier to 0.2501, so the served lean is overconfident in probability terms. Picks still move 333 times, because the home-side offset and discrete lattice are not shrunk; this confounds the arm as a test of the raw lean.
- Blending toward a smaller model gives the same probability gain without a pick gain. The served 90 columns have the record edge, the smaller ones the fit edge; a blend that never keeps served weight (w=0) cannot test the mixture. A finer grid below 0.25 for shrink and above 0 for blends was not declared and not run.
- Implication for the decision: nothing served changes; 785-718 is -1.13 pts with probability_positive 0.13-0.18, not a refutation (no wrong-sign resolved, no positive control).

Recorded `unresolved_below_power` x3 (batch `artifacts/mod23_unit4/weak_signals_batch.json`, 107 blocks counted).

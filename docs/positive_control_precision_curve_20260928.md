# Full positive-control precision curve: September 28

**Measured:** the predeclared historical 16-cell family completed with 200 fixed
synthetic replicates per cell, 50 fit iterations, seed 20260923, and strict
400/2,000/20,000 hierarchical season-week bootstrap prefixes. Four exact-hash
null/strong cells were reused and 12 remaining cells computed. All original
400-draw results must reproduce exactly. The full artifact preserves 4,809,600
prediction rows, 9,600 replicate-prefix rows, fitted coefficients and optimizer
diagnostics. These are evaluator diagnostics, not a serving or closure decision.

Evidence: `artifacts/positive_control_precision_curve/historical_v1_checkpoint/`.
The original declaration is preserved as `predeclaration.md` in that bundle.

**Measured:** independent verification passed all 16 exact 400-draw reproductions,
prediction and replicate arithmetic, 96 rate/interval summaries, reused hashes,
and unchanged input hashes. A post-launch runner write added sorting to two final
summary lists. The exact original source was recovered and verified against the
manifest, then archived as `extension_runner.py`; the exact diff and current hash
are retained separately. No numeric calculation or original manifest was changed.

The table reports every coefficient, mean injected accuracy-point effect,
detection percentage at each prefix, the 20,000-draw Wilson 95% interval, and
400-versus-20,000 classification disagreement. Each rate has denominator 200.

| Control | Coefficient | Mean effect (points) | 400 | 2,000 | 20,000 [95%] | Disagreement [95%] |
|---|---:|---:|---:|---:|---:|---:|
| DPI | 0.00 | -0.090818 | 0.5% | 0.5% | 0.5% [0.09%, 2.78%] | 0.0% [0.00%, 1.88%] |
| DPI | 0.10 | +0.200599 | 2.5% | 3.0% | 3.0% [1.38%, 6.39%] | 0.5% [0.09%, 2.78%] |
| DPI | 0.25 | +1.021623 | 5.5% | 6.5% | 6.0% [3.47%, 10.19%] | 1.5% [0.51%, 4.32%] |
| DPI | 0.45 | +1.688623 | 22.0% | 21.5% | 21.0% [15.93%, 27.16%] | 5.0% [2.74%, 8.96%] |
| DPI | 0.70 | +1.931138 | 19.5% | 21.5% | 22.0% [16.82%, 28.24%] | 3.5% [1.71%, 7.05%] |
| DPI | 1.00 | +2.068530 | 20.5% | 20.0% | 20.0% [15.05%, 26.09%] | 2.5% [1.07%, 5.72%] |
| DPI | 1.40 | +2.186959 | 24.0% | 23.5% | 24.5% [19.06%, 30.90%] | 4.5% [2.39%, 8.33%] |
| DPI | 1.80 | +2.348636 | 38.5% | 35.5% | 34.5% [28.26%, 41.32%] | 8.0% [4.98%, 12.60%] |
| Holding | 0.00 | -0.072522 | 0.5% | 0.5% | 0.5% [0.09%, 2.78%] | 0.0% [0.00%, 1.88%] |
| Holding | 0.10 | +0.290419 | 1.0% | 1.0% | 1.0% [0.27%, 3.57%] | 0.0% [0.00%, 1.88%] |
| Holding | 0.25 | +0.943114 | 11.5% | 10.5% | 9.5% [6.17%, 14.36%] | 3.0% [1.38%, 6.39%] |
| Holding | 0.45 | +1.670659 | 28.0% | 26.5% | 25.5% [19.96%, 31.96%] | 5.5% [3.10%, 9.58%] |
| Holding | 0.70 | +2.190286 | 37.5% | 36.5% | 35.5% [29.20%, 42.35%] | 4.0% [2.04%, 7.69%] |
| Holding | 1.00 | +2.568862 | 53.5% | 52.5% | 54.0% [47.08%, 60.77%] | 5.5% [3.10%, 9.58%] |
| Holding | 1.40 | +2.832668 | 64.0% | 60.5% | 60.5% [53.59%, 67.02%] | 4.5% [2.39%, 8.33%] |
| Holding | 1.80 | +3.126747 | 78.0% | 74.0% | 75.0% [68.57%, 80.49%] | 6.0% [3.47%, 10.19%] |

All 48 detection estimates and 48 paired disagreement summaries are retained,
including the other two prefix pairs. The original research family has 16
coefficient looks; the precision extension reports 96 diagnostic looks. Exact
reproduction checks are integrity checks, not selected findings.

**Read:** the natural DPI estimate is -0.133 accuracy points; holding is -0.599
(the original declarations are retained with this artifact). The injected non-null
controls all point in the opposite direction. No interpolation or absolute-value
matching is authorized. This curve does not bound or close either natural signal.
A separately declared direction/size-matched control would be required.

**Inferred under AGENTS.md:** additional bootstrap precision does not establish
sensitivity to these natural effects. No natural weak-signal status changes, no
model selection, and no serving promotion follow. Zero crossing closes nothing;
one fitted calibrated probability continues to select every served side.

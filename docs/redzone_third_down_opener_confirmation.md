# Red-zone third-down fade opener confirmation: predeclaration

Written before this opener-grade comparison is scored. New family
`redzone_third_down_over_fade_on_production_opener` inherits
`redzone_reversion_on_production`; it receives its new eligible opener block only
from the rotation CLI, never from manual selection.

Baseline is production `weak_stack` / `market_residual` / ridge alpha 10. Candidate
is `weak_stack_redzone_third_down`, adding only
`redzone_third_down_over_fade_diff`. Primary is paired candidate-minus-baseline
Tuesday-opener forced-pick accuracy under the production probability rule; sign and
close reads are secondary. Training remains strictly pre-prediction.

Run once, in order: a 200-draw within-week frozen-pick opener null; a positive
control replacing only the candidate column with realized `ats_margin`; then one
real screen if the control fires. Keep prediction-level pairs. Feature leakage and
identity coverage is in `tests/test_redzone_reversion_production_feature.py`, with
confirmation scope/profile identity separately tested.

No interval crossing zero is grounds to close this work. Every non-terminal result
is recorded through both registries as `unresolved_below_power` with
`probability_positive`. Opener EV governs play; claim thresholds do not veto it.

## Result (recorded after the screen)

**Measured** in `artifacts/redzone_third_down_opener_confirmation/20260902T174651Z/results.json`: the assigned 2020-2021 window held 456 paired non-push games in 35 weeks. The probability-rule candidate was 53.070% versus production at 53.728%, a **-0.658 accuracy-point** delta with week-blocked `probability_positive=0.111` and 95% [-1.978, +0.451]. The frozen-pick null centred +0.106 points; the observed delta was at the 4.0th percentile. The positive control measured +43.860 points, P+ 1.000. Both registry records are `unresolved`: this block currently favors the baseline for forced-pick EV, without a terminal conclusion.

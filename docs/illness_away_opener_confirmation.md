# Away active-illness opener confirmation: predeclaration

Written before any opener outcome is scored. New family
`illness_away_active_ge1_on_production_opener` inherits `illness_on_production`.
Its assigned opener window comes only from the rotation CLI and is disjoint from
the close-grade family's 2011-2013 window.

Baseline is production `weak_stack` / `market_residual` / ridge alpha 10. Candidate
is `weak_stack_illness_away`, adding only `illness_away_active_ge1`. Primary is the
paired opener forced-pick accuracy delta under the production probability rule;
sign-rule and close grades are secondary. All fitting is forward-chained.

Before its sole screen: run one 200-draw within-week frozen-pick opener null, then
one `ats_margin` leak positive control. Run the screen exactly once only after the
control confirms the full-profile harness can detect the planted large effect.
Retain prediction-level output. `tests/test_illness_production_feature.py` supplies
the as-of feature leakage tests; confirmation scope and profile identity are tested.

Intervals crossing zero do not settle this line. Non-terminal outcomes must be
recorded in both registries as `unresolved_below_power`, reporting
`probability_positive`. The opener result informs forced-pick EV, while thresholds
limit claims only.

## Result (recorded after the screen)

**Measured** in `artifacts/illness_away_opener_confirmation/20260902T174255Z/results.json`: the assigned 2020-2021 window held 456 paired non-push games in 35 weeks. The probability-rule candidate was 53.289% versus production at 53.728%, a **-0.439 accuracy-point** delta with week-blocked `probability_positive=0.312` and 95% [-2.655, +1.532]. The frozen-pick null centred -0.378 points and the observed delta was at the 44.5th percentile. The positive control measured +43.860 points, P+ 1.000. Both registry records are `unresolved`: this block currently favors the baseline for forced-pick EV, without a terminal conclusion.

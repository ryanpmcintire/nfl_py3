# Team-style pace mismatch opener confirmation: predeclaration

Written before this opener-grade comparison is scored. New family
`team_style_pace_mismatch_on_production_opener` inherits
`team_style_pace_on_production`; the rotation CLI assigns a new eligible opener
window rather than reusing the close-grade 2011-2013 window.

Baseline: production `weak_stack` / `market_residual` / ridge alpha 10. Candidate:
`weak_stack_team_style_pace`, differing only by `team_style_pace_mismatch_flag`.
Primary quantity: paired candidate-minus-baseline forced-pick accuracy at the
Tuesday opener under the production probability rule. The sign rule and close
settlement read are secondary. Training is strictly chronological.

Run exactly one frozen-pick within-week opener null (200 draws), then one
`ats_margin`-leak positive control, then the real screen once if the control
detects the planted effect. Retain prediction-level pairs. Existing feature
leakage/additivity coverage is `tests/test_team_style_pace_production_feature.py`;
the confirmation evaluator also has scope and profile-identity tests.

An interval crossing zero is not a closing ground. Record any non-terminal result
in both registries as `unresolved_below_power` and report `probability_positive`.
Opener expected value governs the forced pick, independent of claim thresholds.

## Result (recorded after the screen)

**Measured** in `artifacts/team_style_pace_opener_confirmation/20260902T173905Z/results.json`: the assigned 2020-2021 window held 456 paired non-push games in 35 weeks. The probability-rule candidate was 53.509% versus production at 53.728%, a **-0.219 accuracy-point** delta with week-blocked `probability_positive=0.286` and 95% [-1.325, +0.889]. Its frozen-pick null centred +0.063 points; the observed delta was at the 32.0th percentile. The positive control measured +43.860 points, P+ 1.000. Both registry records are `unresolved`: this block currently favors the baseline for forced-pick EV, without a terminal conclusion.

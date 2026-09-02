# Reddit home-comment-ratio opener confirmation: predeclaration

Written before this opener-grade comparison is scored. This is a new, independent
opener family, `reddit_home_comment_ratio_elevated_on_production_opener`, inheriting
the close-grade family `reddit_attention_on_production`; it does not reuse its
2011-2013 close window. The rotation CLI, not this document, assigns the earliest
eligible two-season opener block.

The baseline is the active production `weak_stack` / `market_residual` / ridge
alpha 10 profile. The one candidate is `weak_stack_reddit_ratio_home`, which adds
only `reddit_home_comment_ratio_elevated`. The primary measure is paired
candidate-minus-baseline forced-pick accuracy at the Tuesday opener, using the
production probability pick rule; the sign-rule and close grades are secondary
diagnostics. Fits are chronological and use only games before each predicted week.

Before the screen, run exactly once on the assigned window: (1) a 200-draw
within-week frozen-pick opener-settlement permutation null and (2) a positive
control that replaces only the candidate column with realized `ats_margin`.
The screen runs only after the control demonstrates that the complete-profile
harness can detect its deliberate large leak. Per-game paired output is retained.
Existing feature leakage/additivity tests in `tests/test_reddit_attention_production_feature.py`
and confirmation scope/identity tests cover the as-of column and evaluator.

An interval crossing zero never closes this work. Only a resolved wrong sign,
zero split-half reliability, or a positive control that bounds this effect can
close it. Every other outcome is recorded through both registries as
`unresolved_below_power`, with `probability_positive` reported. Opener expected
value informs the forced-pick decision; claim thresholds do not veto that decision.

## Result (recorded after the screen)

**Measured** in `artifacts/reddit_attention_opener_confirmation/20260902T173438Z/results.json`: the CLI-assigned 2020-2021 opener window produced 456 paired non-push games in 35 weeks. The probability-rule candidate was 54.167% versus production at 53.728%, a **+0.439 accuracy-point** delta with week-blocked `probability_positive=0.665` and 95% [-0.895, +1.978]. The frozen-pick null centred +0.333 points and the observed delta was at its 42.0th percentile. The prior positive control measured +43.860 points, P+ 1.000. Both registry records are `unresolved`; this block favors the candidate for forced-pick EV but establishes no general claim or card change.

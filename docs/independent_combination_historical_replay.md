# Historical combination selection replay

## Decision and evidence

**Measured:** the declared full replay scored **269-264** versus raw **282-251**
on the same **533 decisive regular-season games** in 2024-2025. Its primary
Brier improvement was **-0.002283**, week-block 95% interval
**[-0.009488, +0.005479]**, probability positive **0.266050**.
**Inferred:** this full experimental pipeline does not establish an improvement
over raw and should not replace the served model.

**Measured, post-result diagnostic:** before the extra calibration stage, the
selected combination scored **300-233** versus raw **282-251**. Brier improvement
was **+0.002057** **[-0.005241, +0.010013]**, probability positive **0.700550**;
accuracy improvement was **+3.38 points** **[-0.94, +7.99]**, probability positive
**0.931525**. On the **180 disagreements**, it went **99-81**, with exact
two-sided fair-coin null p **0.204992**. These diagnostic results do not replace
the primary result or become a new selected headline experiment.

**Measured:** the additional calibrator learned a slope of **-0.584377** from
62 late-2023 games for the nested arm, and **-0.120518** for the fixed arm.
That reverses the base probability ordering before the 2024 outer year. In the
next fold the respective slopes are **+0.550112** and **+0.398022**, fitted on
63 late-2024 games. The complete fitted coefficients below show the mechanism.

**Read:** this extra two-parameter calibration is introduced by
scripts/independent_combination_replay.py; it is absent from the served
four-term PickProbabilityModel.home_probability at src/nfl_ats/pick_probability.py:182.
No served prediction source,
coefficient, switch, or dashboard was changed by this audit.
**Inferred:** the poor full replay cannot be attributed solely to feature
selection or treated as evidence of a production calibration bug.

**Inferred decision:** these measurements do not justify replacing the current
combination with raw because of Week 3. The selected base combination retains a
positive point estimate, but its Brier result varies by season and its selected
flags change substantially. The existing fixed recipe also retains positive
historical results. This supports continuing the declared prospective comparison
while withholding claims of independently established advantage. No individual
signal is closed and no automatic serving change is made.

## Design fixed before scores

The final declaration is
[the historical protocol](../registry/studies/combined_vs_raw_historical_nested_20260928.json).
The first whole-year declaration is preserved alongside it. **Measured input
audit, before any new performance scoring:** movement availability is zero for
2020-2022 and complete for the saved 2023-2025 populations (266, 266, 267 games).
The declaration was amended before scoring to use successive blocks within the
preceding season, allowing all candidate families actual movement observations.

For each outer test year Y:

1. Train candidate coefficients on all data through week 6 of Y-1.
2. Select among all 256 candidates on weeks 7-10 of Y-1: every subset of the
   seven active flags, with or without the movement/availability pair.
   Intercept and raw-model logit are always present; included flags share one
   fitted coefficient. Held members and new thresholds are excluded.
3. Refit the selected features through week 10, then select ridge from
   0.001, 0.01, 0.1, 1, 10 on weeks 11-14 of Y-1.
4. Refit the chosen base model through week 14. Fit the separate two-parameter
   logit calibrator on weeks 15-18 of Y-1.
5. Score the entire outer year Y. All fitted values, standardization,
   selection and calibration precede that outer year.

The fixed-combination control uses all seven flags plus movement, the same
chronology, fixed ridge 0.001, and the same separate calibration stage.
Raw-calibrated fits only the raw logit on the same calibration block. Raw and
neutral market are unmodified references. Fixed-forward is the existing saved
chronological four-term recipe, refitted using all earlier seasons.

Both model arms start from the saved discrete cover distribution at the same
original opener and use nonpush-conditional probabilities. **Measured:** all
544 opener games in the two outer years are accounted for: 533 decisive games
and 11 excluded pushes; no decisive games are missing. Uncertainty uses 36
season/week blocks, 20,000 draws, seed 20260928. Push mass and prediction-level
rows are retained. Close grades never select an arm or veto a play.

## Complete paired results

**Measured:** lower Brier and log loss are better. Neutral-market ATS uses the
common home-side tie convention at probability 0.5; it is not a wagering edge.

| Arm | ATS record | Accuracy | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Raw model | 282-251 | 52.91% | 0.250898 | 0.695016 |
| Neutral market | 271-262 | 50.84% | 0.250000 | 0.693147 |
| Raw plus extra calibration | 264-269 | 49.53% | 0.252913 | 0.699042 |
| Fixed combination plus extra calibration | 271-262 | 50.84% | 0.250585 | 0.694266 |
| Nested selection plus extra calibration (primary) | 269-264 | 50.47% | 0.253181 | 0.699494 |
| Fixed recipe, forward fit | 305-228 | 57.22% | 0.245340 | 0.683647 |
| Nested selection before extra calibration (diagnostic) | 300-233 | 56.29% | 0.248841 | 0.691867 |
| Fixed combination before extra calibration (diagnostic) | 308-225 | 57.79% | 0.246160 | 0.685748 |

The two before-calibration rows are explicitly post-result diagnostics.
Their declaration was written after observing the primary calibration slope,
before computing their outer metrics. They reuse already stored base models,
perform zero new fits, and reconstruct the original final predictions exactly.

### Season stability

| Arm | Season | ATS record | Brier | Log loss |
| --- | --- | ---: | ---: | ---: |
| Raw model | 2024 | 145-121 | 0.250570 | 0.694413 |
| Raw model | 2025 | 137-130 | 0.251226 | 0.695617 |
| Fixed recipe, forward fit | 2024 | 152-114 | 0.245098 | 0.683547 |
| Fixed recipe, forward fit | 2025 | 153-114 | 0.245581 | 0.683748 |
| Nested selection plus extra calibration (primary) | 2024 | 128-138 | 0.258536 | 0.710421 |
| Nested selection plus extra calibration (primary) | 2025 | 141-126 | 0.247846 | 0.688609 |
| Nested selection before extra calibration (diagnostic) | 2024 | 149-117 | 0.252780 | 0.701435 |
| Nested selection before extra calibration (diagnostic) | 2025 | 151-116 | 0.244917 | 0.682335 |
| Fixed combination before extra calibration (diagnostic) | 2024 | 155-111 | 0.246415 | 0.687081 |
| Fixed combination before extra calibration (diagnostic) | 2025 | 153-114 | 0.245906 | 0.684421 |

The selected base arm's Brier is worse than raw in 2024 and better in 2025.
Its positive aggregate hit rate is not proof of stable probability improvement
or profitability.

### Choices made before each outer season

| Outer year | Selected flags | Movement | Ridge | Calibration games |
| --- | --- | --- | ---: | ---: |
| 2024 | cold_visitor | Included | 10 | 62 |
| 2025 | coach, division, bye, protection, tank_zone | Included | 10 | 63 |

### Fitted coefficients and stability

All coefficients are in original input units. The separate calibrator transforms
the base logit; its intercept and slope are composed with these coefficients.

| Outer year | Arm/stage | Intercept | Raw logit | Flag sum | Movement | Available |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | nested_selected, before extra calibration | -0.0906 | +0.2034 | +0.6687 | +0.3333 | +0.0826 |
| 2024 | nested_selected, after extra calibration | +0.2800 | -0.1189 | -0.3908 | -0.1948 | -0.0483 |
| 2024 | fixed_combination, before extra calibration | -0.0501 | +0.3041 | +0.2826 | +0.3480 | +0.0634 |
| 2024 | fixed_combination, after extra calibration | +0.1530 | -0.0366 | -0.0341 | -0.0419 | -0.0076 |
| 2025 | nested_selected, before extra calibration | -0.0199 | +0.3003 | +0.2859 | +0.1869 | +0.0174 |
| 2025 | nested_selected, after extra calibration | +0.2052 | +0.1652 | +0.1573 | +0.1028 | +0.0096 |
| 2025 | fixed_combination, before extra calibration | -0.0617 | +0.2685 | +0.3185 | +0.1890 | +0.0312 |
| 2025 | fixed_combination, after extra calibration | +0.1816 | +0.1069 | +0.1268 | +0.0752 | +0.0124 |

The extra calibration reverses all directional terms in the first outer fold.
This is a diagnosis of the experimental calibration stage, not a side-flip rule
to add to production.

### Apparent fit versus outer performance

The before-calibration rows use the fitted training population through week 14.
The after-calibration rows use all earlier development games under the completed
composed model; these are apparent scores, not independent validation.
Positive accuracy gap means the apparent score exceeded the outer score.

| Outer year | Arm | Apparent accuracy | Outer accuracy | Gap, points | Apparent Brier | Outer Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | pre_nested | 55.62% | 56.02% | -0.40 | 0.244819 | 0.252780 |
| 2024 | pre_fixed | 57.38% | 58.27% | -0.89 | 0.243319 | 0.246415 |
| 2025 | pre_nested | 57.72% | 56.55% | +1.16 | 0.245080 | 0.244917 |
| 2025 | pre_fixed | 57.97% | 57.30% | +0.67 | 0.243900 | 0.245906 |
| 2024 | fixed_combination (apparent development) | 48.66% | 49.62% | -0.96 | 0.253848 | 0.252903 |
| 2024 | nested_selected (apparent development) | 45.77% | 48.12% | -2.35 | 0.262078 | 0.258536 |
| 2024 | raw_calibrated (apparent development) | 48.66% | 47.37% | +1.29 | 0.260400 | 0.255464 |
| 2025 | fixed_combination (apparent development) | 50.81% | 52.06% | -1.25 | 0.249207 | 0.248275 |
| 2025 | nested_selected (apparent development) | 50.81% | 52.81% | -2.00 | 0.249375 | 0.247846 |
| 2025 | raw_calibrated (apparent development) | 49.68% | 51.69% | -2.01 | 0.250650 | 0.250372 |

## All declared paired metrics

Positive improvement favors the first arm. All comparisons are paired on the
same 533 decisive games; accuracy units are percentage points.

### Declared primary family

| Contrast | Units | Improvement | 95% interval | Probability positive |
| --- | --- | ---: | --- | ---: |
| nested_selected_vs_raw | brier_improvement | -0.002283 | [-0.009488, +0.005479] | 0.266050 |
| nested_selected_vs_raw | log_loss_improvement | -0.004479 | [-0.019166, +0.011327] | 0.273250 |
| nested_selected_vs_raw | accuracy_points | -2.44 | [-8.07, +3.23] | 0.193250 |
| nested_selected_vs_raw_calibrated | brier_improvement | -0.000268 | [-0.004444, +0.004190] | 0.440400 |
| nested_selected_vs_raw_calibrated | log_loss_improvement | -0.000453 | [-0.009044, +0.008770] | 0.448700 |
| nested_selected_vs_raw_calibrated | accuracy_points | +0.94 | [-2.63, +4.45] | 0.698325 |
| nested_selected_vs_fixed_combination | brier_improvement | -0.002597 | [-0.005219, -0.000005] | 0.024750 |
| nested_selected_vs_fixed_combination | log_loss_improvement | -0.005229 | [-0.010663, +0.000118] | 0.028050 |
| nested_selected_vs_fixed_combination | accuracy_points | -0.38 | [-2.79, +1.93] | 0.379750 |
| nested_selected_vs_fixed_forward | brier_improvement | -0.007841 | [-0.016026, +0.000071] | 0.025850 |
| nested_selected_vs_fixed_forward | log_loss_improvement | -0.015847 | [-0.032756, +0.000557] | 0.029300 |
| nested_selected_vs_fixed_forward | accuracy_points | -6.75 | [-12.50, -1.13] | 0.009850 |
| fixed_forward_vs_raw | brier_improvement | +0.005558 | [-0.001792, +0.013738] | 0.926950 |
| fixed_forward_vs_raw | log_loss_improvement | +0.011368 | [-0.004040, +0.028573] | 0.921950 |
| fixed_forward_vs_raw | accuracy_points | +4.32 | [-0.57, +9.53] | 0.955075 |

### Post-result calibration diagnostic

| Contrast | Units | Improvement | 95% interval | Probability positive |
| --- | --- | ---: | --- | ---: |
| pre_nested_vs_raw | brier_improvement | +0.002057 | [-0.005241, +0.010013] | 0.700550 |
| pre_nested_vs_raw | log_loss_improvement | +0.003149 | [-0.012685, +0.020179] | 0.642050 |
| pre_nested_vs_raw | accuracy_points | +3.38 | [-0.94, +7.99] | 0.931525 |
| pre_fixed_vs_raw | brier_improvement | +0.004738 | [-0.003228, +0.013471] | 0.872750 |
| pre_fixed_vs_raw | log_loss_improvement | +0.009267 | [-0.007938, +0.027893] | 0.850300 |
| pre_fixed_vs_raw | accuracy_points | +4.88 | [+0.37, +9.55] | 0.983200 |
| pre_nested_vs_pre_fixed | brier_improvement | -0.002681 | [-0.005828, +0.000245] | 0.035500 |
| pre_nested_vs_pre_fixed | log_loss_improvement | -0.006119 | [-0.012956, +0.000114] | 0.027200 |
| pre_nested_vs_pre_fixed | accuracy_points | -1.50 | [-4.36, +1.31] | 0.150725 |
| pre_nested_vs_nested_selected | brier_improvement | +0.004340 | [-0.004058, +0.012567] | 0.849300 |
| pre_nested_vs_nested_selected | log_loss_improvement | +0.007627 | [-0.010203, +0.025036] | 0.804450 |
| pre_nested_vs_nested_selected | accuracy_points | +5.82 | [-0.38, +11.68] | 0.966725 |
| pre_fixed_vs_fixed_combination | brier_improvement | +0.004424 | [-0.002562, +0.011556] | 0.892750 |
| pre_fixed_vs_fixed_combination | log_loss_improvement | +0.008518 | [-0.006344, +0.023669] | 0.867350 |
| pre_fixed_vs_fixed_combination | accuracy_points | +6.94 | [+1.13, +12.69] | 0.990575 |

### Disagreement nulls

These exact nulls assume independent games and are descriptive across the
declared family. The week bootstrap supplies the clustered intervals above.

| Family and contrast | Candidate-baseline wins | Disagreements | Exact two-sided null p |
| --- | ---: | ---: | ---: |
| Primary: fixed_forward_vs_raw | 108-85 | 193 | 0.113051 |
| Primary: nested_selected_vs_fixed_combination | 22-24 | 46 | 0.882996 |
| Primary: nested_selected_vs_fixed_forward | 109-145 | 254 | 0.027887 |
| Primary: nested_selected_vs_raw | 112-125 | 237 | 0.435759 |
| Primary: nested_selected_vs_raw_calibrated | 42-37 | 79 | 0.652964 |
| Diagnostic: pre_fixed_vs_fixed_combination | 134-97 | 231 | 0.017663 |
| Diagnostic: pre_fixed_vs_raw | 115-89 | 204 | 0.079803 |
| Diagnostic: pre_nested_vs_nested_selected | 145-114 | 259 | 0.062098 |
| Diagnostic: pre_nested_vs_pre_fixed | 33-41 | 74 | 0.415985 |
| Diagnostic: pre_nested_vs_raw | 99-81 | 180 | 0.204992 |

## Reliability

**Measured:** the four bands were fixed before scoring. Every arm and empty cell
is shown; no band was selected as a finding.

| Arm | Estimated selected-side chance | Games | Mean estimate | Observed cover rate |
| --- | --- | ---: | ---: | ---: |
| raw | 50-55% | 297 | 52.40% | 53.20% |
| raw | 55-60% | 163 | 57.40% | 53.37% |
| raw | 60-65% | 67 | 61.85% | 49.25% |
| raw | 65-100% | 6 | 66.32% | 66.67% |
| neutral_market | 50-55% | 533 | 50.00% | 50.84% |
| neutral_market | 55-60% | 0 | unavailable | unavailable |
| neutral_market | 60-65% | 0 | unavailable | unavailable |
| neutral_market | 65-100% | 0 | unavailable | unavailable |
| raw_calibrated | 50-55% | 280 | 52.72% | 49.29% |
| raw_calibrated | 55-60% | 220 | 56.86% | 48.64% |
| raw_calibrated | 60-65% | 33 | 61.24% | 57.58% |
| raw_calibrated | 65-100% | 0 | unavailable | unavailable |
| fixed_combination | 50-55% | 357 | 53.19% | 48.18% |
| fixed_combination | 55-60% | 147 | 56.88% | 54.42% |
| fixed_combination | 60-65% | 26 | 61.58% | 61.54% |
| fixed_combination | 65-100% | 3 | 67.46% | 100.00% |
| nested_selected | 50-55% | 206 | 52.90% | 53.40% |
| nested_selected | 55-60% | 256 | 57.04% | 47.27% |
| nested_selected | 60-65% | 56 | 62.14% | 48.21% |
| nested_selected | 65-100% | 15 | 68.75% | 73.33% |
| fixed_forward | 50-55% | 255 | 52.35% | 54.90% |
| fixed_forward | 55-60% | 154 | 57.20% | 57.79% |
| fixed_forward | 60-65% | 77 | 62.17% | 61.04% |
| fixed_forward | 65-100% | 47 | 69.26% | 61.70% |
| pre_nested | 50-55% | 292 | 52.30% | 55.14% |
| pre_nested | 55-60% | 133 | 57.02% | 60.15% |
| pre_nested | 60-65% | 49 | 62.31% | 48.98% |
| pre_nested | 65-100% | 59 | 70.49% | 59.32% |
| pre_fixed | 50-55% | 230 | 52.38% | 57.39% |
| pre_fixed | 55-60% | 168 | 57.18% | 55.36% |
| pre_fixed | 60-65% | 72 | 62.63% | 59.72% |
| pre_fixed | 65-100% | 63 | 70.49% | 63.49% |

## Selection accounting and limits

The primary family includes 512 feature-subset scores, 10 ridge-validation
scores, four final base fits, and six calibration fits: 532 continuous fits in
total. It also contains six scored arms, 18 pooled metric cells, 36 season
metric cells, 24 reliability cells, 15 paired metric contrasts, five
disagreement nulls, and 12 apparent-versus-outer accuracy/Brier gap cells.
The post-result diagnostic adds no fits, two scored arms, six pooled metric
cells, 12 season metric cells, eight reliability cells, 15 paired contrasts,
five disagreement nulls, and eight training-versus-outer gap cells. These are
correlated observations, not independent votes.
The bootstrap conditions on the two fitted historical folds. It does not
measure the uncertainty of repeating the original human research process;
probability positive is the fraction of bootstrap effects favoring the first arm.

**Read:** artifacts/pick_probability/20260927T161329Z/metadata.json:156 explicitly
states that the signal definitions were selected using the evaluated seasons.
**Inferred:** this replay controls
the newly declared subset and ridge selection, not the earlier invention of the
signals, signs, thresholds, market definition, or raw-model recipe. It is a
selection-adjusted historical audit, not an untouched outer experiment.
Only two outer seasons and small internal blocks are available for the complete
movement feature. Neither an observed loss nor an interval spanning positive
and negative effects closes an underlying signal.

**Measured:** all 15 primary and 15 diagnostic paired metrics are recorded in
registry/weak_signals.json, in their separate declared families, as
unresolved_below_power. Under the research-closure rules in AGENTS.md, no
admissible refuted mechanism or positive-control power bound has been
established for closing an underlying signal. No threshold, family, or serving
choice was changed after seeing the outer results.

## Reproduction and verification

Use the locked uv environment:

~~~powershell
.\.tools\uv.exe --cache-dir .tmp/uv-cache run --no-sync --offline python scripts/independent_combination_replay.py
.\.tools\uv.exe --cache-dir .tmp/uv-cache run --no-sync --offline python scripts/independent_combination_calibration_audit.py
.\.tools\uv.exe --cache-dir .tmp/uv-cache run --no-sync --offline nfl-ats weak-signals record --batch artifacts/independent_historical_validation/20260928_nested/weak_signals_batch.json
.\.tools\uv.exe --cache-dir .tmp/uv-cache run --no-sync --offline nfl-ats weak-signals record --batch artifacts/independent_historical_validation/20260928_nested/calibration_diagnostic/weak_signals_batch.json
~~~

The evaluation refuses to overwrite an existing output. A replay can use
--output PATH to produce a separate verification artifact. Inputs, declarations,
executed source copies, reports, selection scores, fold coefficients, paired
intervals, registry payloads and game-level predictions are retained under
artifacts/independent_historical_validation/20260928_nested/; the diagnostic
has its own subdirectory. The registry-export formatting correction did not
change fits or scores; executed source copies preserve the original hashes.

Verification commands are ruff format --check ., ruff check ., mypy src,
and pytest -q --basetemp .tmp/pytest-nested-validation, with the workspace uv
cache. A second real replay verifies the corrected export and compares all
game-level predictions with the first run. The separate arithmetic readback
checks scores, intervals' point estimates, selection winners, row coverage and
registry sample counts. No test files or functions were added.

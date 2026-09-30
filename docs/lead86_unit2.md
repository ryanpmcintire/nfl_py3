# LEAD-86 unit 2: price-consistent market regularization

**Measured:** one fixed-protocol six-season replay. All intervals below are 95% paired,
season-stratified week-block percentile intervals (10,000 draws). Positive improvement
favors the candidate. Accuracy is percent; differences are percentage points.
RPS is the unnormalized sum over integer margins -100 through 100.

## Decisive games first

**Measured:** decisive games are non-push opener games where candidate and comparator
choose different sides. Their intervals resample whole weeks; exact-null p-values
are two-sided binomial diagnostics.

| Fold | Against | Candidate W-L | Win %, interval | probability_positive (above half) | Exact-null p |
| --- | --- | --- | --- | --- | --- |
| 2020 | served_loso | 1-4 | 20.000000 [0.000000, 100.000000] | 0.1120 | 0.375000 |
| 2020 | matched_zero | 0-0 | unavailable | 0.5000 | unavailable |
| 2020 | model_only | 30-29 | 50.847458 [43.076923, 61.702128] | 0.5797 | 1.000000 |
| 2020 | market | 47-48 | 49.473684 [41.176471, 58.163265] | 0.4486 | 1.000000 |
| 2021 | served_loso | 3-2 | 60.000000 [0.000000, 100.000000] | 0.6690 | 1.000000 |
| 2021 | matched_zero | 3-1 | 75.000000 [0.000000, 100.000000] | 0.8482 | 0.625000 |
| 2021 | model_only | 33-31 | 51.562500 [40.740741, 63.157895] | 0.6104 | 0.900653 |
| 2021 | market | 51-53 | 49.038462 [39.639640, 59.793814] | 0.4256 | 0.921949 |
| 2022 | served_loso | 5-6 | 45.454545 [14.285714, 72.727273] | 0.3597 | 1.000000 |
| 2022 | matched_zero | 0-0 | unavailable | 0.5000 | unavailable |
| 2022 | model_only | 53-37 | 58.888889 [52.054445, 67.010583] | 0.9957 | 0.113344 |
| 2022 | market | 56-45 | 55.445545 [47.169811, 65.116812] | 0.8909 | 0.319727 |
| 2023 | served_loso | 6-7 | 46.153846 [12.500000, 75.000000] | 0.3833 | 1.000000 |
| 2023 | matched_zero | 0-0 | unavailable | 0.5000 | unavailable |
| 2023 | model_only | 49-37 | 56.976744 [47.824879, 65.168539] | 0.9382 | 0.235380 |
| 2023 | market | 34-26 | 56.666667 [40.816327, 69.354839] | 0.8185 | 0.366294 |
| 2024 | served_loso | 6-21 | 22.222222 [11.111111, 38.095238] | 0.0017 | 0.005925 |
| 2024 | matched_zero | 0-0 | unavailable | 0.5000 | unavailable |
| 2024 | model_only | 41-50 | 45.054945 [35.714286, 55.102041] | 0.1654 | 0.401813 |
| 2024 | market | 32-35 | 47.761194 [38.571429, 57.746479] | 0.3252 | 0.807195 |
| 2025 | served_loso | 13-30 | 30.232558 [17.073171, 42.501330] | 0.0006 | 0.013718 |
| 2025 | matched_zero | 0-0 | unavailable | 0.5000 | unavailable |
| 2025 | model_only | 41-45 | 47.674419 [37.349398, 57.954545] | 0.3184 | 0.746534 |
| 2025 | market | 53-56 | 48.623853 [40.186916, 57.024793] | 0.3647 | 0.848195 |
| pooled | served_loso | 34-70 | 32.692308 [24.137931, 41.837825] | 0.0003 | 0.000534 |
| pooled | matched_zero | 3-1 | 75.000000 [0.000000, 100.000000] | 0.8464 | 0.625000 |
| pooled | model_only | 247-229 | 51.890756 [48.163265, 55.691057] | 0.8350 | 0.435897 |
| pooled | market | 273-263 | 50.932836 [47.024952, 54.887293] | 0.6716 | 0.697506 |

## Candidate versus served LOSO

**Measured:** Brier and accuracy are the two declared decision comparisons.
Log loss and RPS are descriptive; none supplies an additional selected variant.

| Metric | Improvement, interval | probability_positive |
| --- | --- | --- |
| accuracy | -2.750191 [-4.211332, -1.243153] | 0.0003 |
| brier | -0.000901 [-0.002573, +0.000616] | 0.1343 |
| log_loss | -0.002144 [-0.005799, +0.001183] | 0.1104 |
| rps | -0.015029 [-0.037386, +0.005379] | 0.0809 |

**Inferred:** proposed registry dispositions await the orchestrator: primary Brier
remains unresolved_below_power. This specified training workflow has resolved
negative accuracy versus served; its accuracy cell can use refuted_mechanism with
closing ground wrong_sign_resolved under AGENTS.md:65-85. That endpoint does not
close the broader market-regularization mechanism: its matched-zero comparison
isolates the penalty, while the served comparison also changes training design.
No serving change is authorized. One calibrated probability selects each candidate
side (AGENTS.md:87-100).

## Frozen protocol

- Population: 2020–2025 regular-season historical openers, conditional non-push
  rows with complete common-book Tuesday/Sunday panels after the price filter.
  Historical opener is the pool-line proxy; no pre-2026 pool captures required.
- Filter: de-vig each two-sided moneyline/spread pair. For home-margin threshold
  h>0 require p(win)>=p(cover), h<0 reverse, h=0 equality, allowing 0.0051.
  This fixed tolerance bounds four one-point American-odds rounding errors:
  4 × 0.25 × 100/(99×199) < 0.0051. No outcome informs exclusions.
- Base: artifacts/loso_base/20260929T232711656489Z; canonical
  base_home_probability, hash/coverage checks and one-to-one game/season join.
- Terms: exact cached four served terms; logistic label loss plus lambda market
  cross-entropy, ridge 0.001; lambda grid 0/0.1/1, selection Brier, ties smaller.
- Six outer seasons. In the cyclic 2020–2025 order, preceding season calibrates,
  second preceding selects; remaining three fit. No outer outcomes enter any fit.
  Separate nonnegative-slope logit calibration; no refit after selection.
- Market: Unit 1 discrete lattice, prior on fit seasons only (including pushes),
  filtered alphabetically first common book; all arms share conditional lattice.
- Arms: candidate, pinned served LOSO, matched zero-penalty, raw model, market.
  Base IS uses its outer-fold coefficients on candidate fit rows; OOS uses artifact.
- Metrics: opener Brier primary, accuracy secondary; log loss/RPS descriptive.
  Two decision looks (candidate versus served Brier/accuracy); full diagnostic
  accounting: 756 metric/contrast cells + 28 decisive + 25 reliability + 42
  fit/coefficient summaries = 851 cells/looks. No diagnostic-driven revision.
- IS/OOS and OOS-minus-IS gaps, each season and pooled; 10,000 paired,
  season-stratified week-block draws, seed 20260929, fixed predictions;
  probability_positive counts ties half. Five fixed equal-width reliability bins.
- Zero crossing closes nothing; one fitted probability selects each side.
  Retrospective upstream inputs prevent an end-to-end holdout/promotion claim.


## Source and population checks

**Measured:** 262 capture files;
316,530 raw quotes;
49,032 price pairs checked;
24 rejected book/capture panels
across 18 games.
Exclusions precede book selection and use prices only.
1309 of
1503 base games have eligible complete
panels; every eligible game joined once to the pinned base.

**Measured:** maximum pre-filter ordering violation
48.803504 percentage points;
retained maximum 0.334734.
Across 7,854 lattice fits, maximum quoted-price
projection error 0.000972 points;
0 exceed the fixed 0.51-point tolerance.
Projection diagnostics do not change the population after outcomes.
**Read:** docs/loso_base_artifact.md:26-41 states that the base holds out the
four-term fitting season while upstream model inputs retain retrospective training
provenance; Sunday features cannot represent Tuesday or Thursday decisions.
**Inferred:** this is retrospective season-separated recalibration, not end-to-end
season-held-out forecasting or an untouched chronological test. Historical opener
is the grading line, with Sunday archived pre-kick information.

**Inferred:** candidate and matched-zero fit three seasons, select on a fourth,
calibrate on a fifth, and score the sixth. The served comparator fits all five
other seasons on its larger eligible population and retains its existing fit
convention. Candidate-versus-served therefore combines training-design differences
with regularization; matched-zero isolates the penalty inside the candidate design.
Old-versus-filtered differences cannot be attributed solely to the filter because
the comparator and folds also changed.

**Measured:** artifact hash, base coefficient reproduction, complete eligible coverage,
one-to-one game/season joins, disjoint stages, NFL identity, capture/update/kickoff
clocks and conditional non-push targets passed runtime checks. Exact cached served
terms retain their missing-move policy; the reconstructed filtered move is not
substituted into that design.

## Fold choices and calibrated coefficients

| Outer | Fit seasons | Select | Calibrate | Fit games | OOS games | Penalty | Selection Brier 0 / 0.1 / 1 | Prior games / pushes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 2021, 2022, 2023 | 2024 | 2025 | 656 | 189 | 0.1 | 0.245371 / 0.245216 / 0.245586 | 671 / 15 |
| 2021 | 2022, 2023, 2024 | 2025 | 2020 | 675 | 213 | 1.0 | 0.244659 / 0.244316 / 0.243790 | 693 / 18 |
| 2022 | 2023, 2024, 2025 | 2020 | 2021 | 697 | 210 | 0.0 | 0.245937 / 0.246135 / 0.247496 | 714 / 17 |
| 2023 | 2020, 2024, 2025 | 2021 | 2022 | 653 | 233 | 0.0 | 0.245658 / 0.245766 / 0.247089 | 671 / 18 |
| 2024 | 2020, 2021, 2025 | 2022 | 2023 | 634 | 232 | 0.0 | 0.242838 / 0.243183 / 0.245487 | 649 / 15 |
| 2025 | 2020, 2021, 2022 | 2023 | 2024 | 612 | 232 | 0.1 | 0.246377 / 0.246235 / 0.246732 | 628 / 16 |

| Fold | Arm | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | candidate | -0.063670 | +0.302034 | +0.242711 | +0.174915 | +0.045641 |
| 2020 | matched_zero | -0.064545 | +0.303574 | +0.245967 | +0.168011 | +0.047770 |
| 2020 | served_loso | -0.045251 | +0.341246 | +0.244789 | +0.220730 | +0.028113 |
| 2021 | candidate | -0.031407 | +0.283282 | +0.317860 | +0.309683 | -0.032898 |
| 2021 | matched_zero | -0.045279 | +0.229919 | +0.334798 | +0.184898 | -0.029553 |
| 2021 | served_loso | -0.062332 | +0.177916 | +0.271867 | +0.223716 | +0.044183 |
| 2022 | candidate | -0.079668 | +0.173538 | +0.279915 | +0.239377 | +0.000000 |
| 2022 | matched_zero | -0.079668 | +0.173538 | +0.279915 | +0.239377 | +0.000000 |
| 2022 | served_loso | -0.055698 | +0.249647 | +0.238476 | +0.222903 | +0.039674 |
| 2023 | candidate | +0.021133 | +0.300052 | +0.407587 | +0.330909 | -0.017224 |
| 2023 | matched_zero | +0.021133 | +0.300052 | +0.407587 | +0.330909 | -0.017224 |
| 2023 | served_loso | -0.038064 | +0.341480 | +0.246114 | +0.230272 | +0.002286 |
| 2024 | candidate | +0.074660 | +0.261915 | +0.210974 | +0.232051 | -0.008437 |
| 2024 | matched_zero | +0.074660 | +0.261915 | +0.210974 | +0.232051 | -0.008437 |
| 2024 | served_loso | -0.056178 | +0.256116 | +0.262262 | +0.246066 | +0.039146 |
| 2025 | candidate | -0.056514 | +0.371772 | +0.307943 | +0.000000 | +0.000000 |
| 2025 | matched_zero | -0.056482 | +0.369459 | +0.308034 | +0.000000 | +0.000000 |
| 2025 | served_loso | -0.054099 | +0.292328 | +0.295997 | +0.188739 | +0.047462 |

**Measured:** coefficient stability across the six candidate folds:

| Term | Minimum | Maximum | Positive / zero / negative folds |
| --- | --- | --- | --- |
| intercept | -0.079668 | +0.074660 | 2 / 0 / 4 |
| model_logit | +0.173538 | +0.371772 | 6 / 0 / 0 |
| composition_flag_sum | +0.210974 | +0.407587 | 6 / 0 / 0 |
| market_move_toward_home | +0.000000 | +0.330909 | 5 / 1 / 0 |
| market_move_available | -0.032898 | +0.045641 | 1 / 2 / 3 |

## IS, OOS and gaps

**Measured:** gaps are OOS minus IS, with the same sampled season/week weights applied
to both. Pooled IS repeats each fit game across three outer folds; intervals preserve
those shared weeks. IS is optimistic. Fixed-prediction intervals omit refit and
feature-selection uncertainty. Served IS uses its corresponding outer-fold
coefficients on candidate fit rows, never another row's LOSO prediction.

| Fold | Scheme | Arm | Metric | Estimate, interval |
| --- | --- | --- | --- | --- |
| 2020 | IS | candidate | accuracy | 58.231707 [54.984424, 61.479461] |
| 2020 | IS | candidate | brier | 0.243471 [0.239102, 0.247871] |
| 2020 | IS | candidate | log_loss | 0.680080 [0.671200, 0.689089] |
| 2020 | IS | candidate | rps | 7.431744 [7.017446, 7.892747] |
| 2020 | IS | served_loso | accuracy | 58.079268 [54.602160, 61.503067] |
| 2020 | IS | served_loso | brier | 0.243325 [0.238697, 0.248051] |
| 2020 | IS | served_loso | log_loss | 0.679808 [0.670303, 0.689534] |
| 2020 | IS | served_loso | rps | 7.430197 [7.014436, 7.890763] |
| 2020 | IS | matched_zero | accuracy | 58.689024 [55.503858, 61.840121] |
| 2020 | IS | matched_zero | brier | 0.243466 [0.239082, 0.247887] |
| 2020 | IS | matched_zero | log_loss | 0.680071 [0.671144, 0.689080] |
| 2020 | IS | matched_zero | rps | 7.431595 [7.016961, 7.891976] |
| 2020 | IS | model_only | accuracy | 53.201220 [49.467275, 56.893954] |
| 2020 | IS | model_only | brier | 0.251552 [0.245712, 0.257599] |
| 2020 | IS | model_only | log_loss | 0.696609 [0.684642, 0.709032] |
| 2020 | IS | model_only | rps | 7.532772 [7.092848, 8.018492] |
| 2020 | IS | market | accuracy | 55.335366 [51.674568, 58.866335] |
| 2020 | IS | market | brier | 0.244882 [0.241531, 0.248323] |
| 2020 | IS | market | log_loss | 0.682692 [0.675866, 0.689694] |
| 2020 | IS | market | rps | 7.434150 [7.001251, 7.904919] |
| 2020 | OOS | candidate | accuracy | 53.439153 [46.739130, 60.326087] |
| 2020 | OOS | candidate | brier | 0.246610 [0.240128, 0.253093] |
| 2020 | OOS | candidate | log_loss | 0.686299 [0.673231, 0.699366] |
| 2020 | OOS | candidate | rps | 7.055124 [6.396897, 7.823194] |
| 2020 | OOS | served_loso | accuracy | 55.026455 [47.646982, 62.295082] |
| 2020 | OOS | served_loso | brier | 0.246721 [0.240289, 0.253280] |
| 2020 | OOS | served_loso | log_loss | 0.686518 [0.673534, 0.699789] |
| 2020 | OOS | served_loso | rps | 7.058471 [6.400482, 7.825927] |
| 2020 | OOS | matched_zero | accuracy | 53.439153 [46.739130, 60.326087] |
| 2020 | OOS | matched_zero | brier | 0.246591 [0.240055, 0.253132] |
| 2020 | OOS | matched_zero | log_loss | 0.686260 [0.673077, 0.699464] |
| 2020 | OOS | matched_zero | rps | 7.054816 [6.395923, 7.822162] |
| 2020 | OOS | model_only | accuracy | 52.910053 [45.502646, 59.223697] |
| 2020 | OOS | model_only | brier | 0.258187 [0.245163, 0.271881] |
| 2020 | OOS | model_only | log_loss | 0.710793 [0.683749, 0.739381] |
| 2020 | OOS | model_only | rps | 7.098257 [6.420307, 7.903299] |
| 2020 | OOS | market | accuracy | 53.968254 [48.837209, 58.549223] |
| 2020 | OOS | market | brier | 0.246065 [0.238954, 0.253211] |
| 2020 | OOS | market | log_loss | 0.684829 [0.669840, 0.699620] |
| 2020 | OOS | market | rps | 7.082915 [6.415261, 7.875724] |
| 2020 | gap | candidate | accuracy | -4.792554 [-12.320745, +2.656807] |
| 2020 | gap | candidate | brier | +0.003139 [-0.004721, +0.011139] |
| 2020 | gap | candidate | log_loss | +0.006218 [-0.009657, +0.022398] |
| 2020 | gap | candidate | rps | -0.376620 [-1.195056, +0.490999] |
| 2020 | gap | served_loso | accuracy | -3.052813 [-11.338553, +4.960584] |
| 2020 | gap | served_loso | brier | +0.003396 [-0.004637, +0.011616] |
| 2020 | gap | served_loso | log_loss | +0.006710 [-0.009627, +0.023345] |
| 2020 | gap | served_loso | rps | -0.371727 [-1.187888, +0.494986] |
| 2020 | gap | matched_zero | accuracy | -5.249871 [-12.767923, +2.171975] |
| 2020 | gap | matched_zero | brier | +0.003125 [-0.004777, +0.011231] |
| 2020 | gap | matched_zero | log_loss | +0.006189 [-0.009780, +0.022507] |
| 2020 | gap | matched_zero | rps | -0.376778 [-1.194673, +0.490772] |
| 2020 | gap | model_only | accuracy | -0.291167 [-8.662876, +7.067139] |
| 2020 | gap | model_only | brier | +0.006636 [-0.007710, +0.021571] |
| 2020 | gap | model_only | log_loss | +0.014184 [-0.015512, +0.045238] |
| 2020 | gap | model_only | rps | -0.434515 [-1.281868, +0.473326] |
| 2020 | gap | market | accuracy | -1.367112 [-7.557260, +4.507789] |
| 2020 | gap | market | brier | +0.001183 [-0.006722, +0.009021] |
| 2020 | gap | market | log_loss | +0.002138 [-0.014324, +0.018408] |
| 2020 | gap | market | rps | -0.351234 [-1.190014, +0.549202] |
| 2021 | IS | candidate | accuracy | 58.962963 [55.847908, 61.876997] |
| 2021 | IS | candidate | brier | 0.243461 [0.237702, 0.249405] |
| 2021 | IS | candidate | log_loss | 0.680633 [0.668283, 0.693337] |
| 2021 | IS | candidate | rps | 7.108381 [6.715245, 7.515220] |
| 2021 | IS | served_loso | accuracy | 57.629630 [54.472694, 60.615047] |
| 2021 | IS | served_loso | brier | 0.243119 [0.238665, 0.247729] |
| 2021 | IS | served_loso | log_loss | 0.679443 [0.670166, 0.689006] |
| 2021 | IS | served_loso | rps | 7.101303 [6.708702, 7.509176] |
| 2021 | IS | matched_zero | accuracy | 59.259259 [56.363569, 62.099354] |
| 2021 | IS | matched_zero | brier | 0.242954 [0.238053, 0.248083] |
| 2021 | IS | matched_zero | log_loss | 0.679225 [0.669105, 0.689920] |
| 2021 | IS | matched_zero | rps | 7.096820 [6.705334, 7.504491] |
| 2021 | IS | model_only | accuracy | 53.185185 [49.553571, 56.788321] |
| 2021 | IS | model_only | brier | 0.252060 [0.248047, 0.256200] |
| 2021 | IS | model_only | log_loss | 0.697580 [0.689390, 0.706081] |
| 2021 | IS | model_only | rps | 7.202196 [6.797131, 7.621147] |
| 2021 | IS | market | accuracy | 53.629630 [50.303928, 56.975037] |
| 2021 | IS | market | brier | 0.245124 [0.241858, 0.248290] |
| 2021 | IS | market | log_loss | 0.683194 [0.676507, 0.689647] |
| 2021 | IS | market | rps | 7.110814 [6.709956, 7.529383] |
| 2021 | OOS | candidate | accuracy | 57.276995 [49.268293, 65.094888] |
| 2021 | OOS | candidate | brier | 0.245353 [0.235099, 0.255502] |
| 2021 | OOS | candidate | log_loss | 0.683714 [0.662930, 0.704275] |
| 2021 | OOS | candidate | rps | 8.293151 [7.514273, 9.236716] |
| 2021 | OOS | served_loso | accuracy | 56.807512 [49.536929, 64.150943] |
| 2021 | OOS | served_loso | brier | 0.245718 [0.237160, 0.254273] |
| 2021 | OOS | served_loso | log_loss | 0.684497 [0.667160, 0.701816] |
| 2021 | OOS | served_loso | rps | 8.297866 [7.519508, 9.241113] |
| 2021 | OOS | matched_zero | accuracy | 56.338028 [48.571429, 63.963964] |
| 2021 | OOS | matched_zero | brier | 0.245825 [0.235517, 0.256103] |
| 2021 | OOS | matched_zero | log_loss | 0.684666 [0.663624, 0.705588] |
| 2021 | OOS | matched_zero | rps | 8.299679 [7.523664, 9.243590] |
| 2021 | OOS | model_only | accuracy | 56.338028 [47.906977, 64.444708] |
| 2021 | OOS | model_only | brier | 0.247460 [0.231854, 0.263736] |
| 2021 | OOS | model_only | log_loss | 0.688277 [0.656183, 0.721814] |
| 2021 | OOS | model_only | rps | 8.301692 [7.451143, 9.306407] |
| 2021 | OOS | market | accuracy | 58.215962 [53.211009, 63.303098] |
| 2021 | OOS | market | brier | 0.247033 [0.239916, 0.254437] |
| 2021 | OOS | market | log_loss | 0.687061 [0.672437, 0.702238] |
| 2021 | OOS | market | rps | 8.270335 [7.489722, 9.202994] |
| 2021 | gap | candidate | accuracy | -1.685968 [-10.263267, +6.731974] |
| 2021 | gap | candidate | brier | +0.001892 [-0.009805, +0.013506] |
| 2021 | gap | candidate | log_loss | +0.003081 [-0.021058, +0.027078] |
| 2021 | gap | candidate | rps | +1.184771 [+0.295756, +2.201798] |
| 2021 | gap | served_loso | accuracy | -0.822118 [-8.616773, +7.075689] |
| 2021 | gap | served_loso | brier | +0.002599 [-0.007084, +0.012163] |
| 2021 | gap | served_loso | log_loss | +0.005054 [-0.014585, +0.024597] |
| 2021 | gap | served_loso | rps | +1.196562 [+0.309653, +2.208680] |
| 2021 | gap | matched_zero | accuracy | -2.921231 [-11.194094, +5.102215] |
| 2021 | gap | matched_zero | brier | +0.002871 [-0.008660, +0.014237] |
| 2021 | gap | matched_zero | log_loss | +0.005441 [-0.018104, +0.028619] |
| 2021 | gap | matched_zero | rps | +1.202860 [+0.311950, +2.215607] |
| 2021 | gap | model_only | accuracy | +3.152843 [-5.855912, +11.865376] |
| 2021 | gap | model_only | brier | -0.004600 [-0.020528, +0.012246] |
| 2021 | gap | model_only | log_loss | -0.009304 [-0.042097, +0.025281] |
| 2021 | gap | model_only | rps | +1.099496 [+0.150126, +2.189821] |
| 2021 | gap | market | accuracy | +4.586333 [-1.443359, +10.675837] |
| 2021 | gap | market | brier | +0.001909 [-0.005985, +0.010088] |
| 2021 | gap | market | log_loss | +0.003867 [-0.012354, +0.020669] |
| 2021 | gap | market | rps | +1.159521 [+0.267475, +2.175996] |
| 2022 | IS | candidate | accuracy | 57.962697 [54.871060, 61.048346] |
| 2022 | IS | candidate | brier | 0.243693 [0.237784, 0.249488] |
| 2022 | IS | candidate | log_loss | 0.680478 [0.667945, 0.692805] |
| 2022 | IS | candidate | rps | 7.191460 [6.779209, 7.617836] |
| 2022 | IS | served_loso | accuracy | 57.675753 [54.847813, 60.497324] |
| 2022 | IS | served_loso | brier | 0.243478 [0.238185, 0.248746] |
| 2022 | IS | served_loso | log_loss | 0.679929 [0.668708, 0.691037] |
| 2022 | IS | served_loso | rps | 7.188920 [6.774438, 7.613750] |
| 2022 | IS | matched_zero | accuracy | 57.962697 [54.871060, 61.048346] |
| 2022 | IS | matched_zero | brier | 0.243693 [0.237784, 0.249488] |
| 2022 | IS | matched_zero | log_loss | 0.680478 [0.667945, 0.692805] |
| 2022 | IS | matched_zero | rps | 7.191460 [6.779209, 7.617836] |
| 2022 | IS | model_only | accuracy | 53.371593 [49.783550, 56.860325] |
| 2022 | IS | model_only | brier | 0.252024 [0.247926, 0.256117] |
| 2022 | IS | model_only | log_loss | 0.697446 [0.689102, 0.705796] |
| 2022 | IS | model_only | rps | 7.285218 [6.860547, 7.715088] |
| 2022 | IS | market | accuracy | 52.797704 [49.566443, 55.903384] |
| 2022 | IS | market | brier | 0.245988 [0.242635, 0.249210] |
| 2022 | IS | market | log_loss | 0.684953 [0.678110, 0.691521] |
| 2022 | IS | market | rps | 7.203317 [6.795334, 7.625698] |
| 2022 | OOS | candidate | accuracy | 60.476190 [54.976303, 66.183575] |
| 2022 | OOS | candidate | brier | 0.242741 [0.236189, 0.249328] |
| 2022 | OOS | candidate | log_loss | 0.678532 [0.665253, 0.691871] |
| 2022 | OOS | candidate | rps | 6.730143 [6.164911, 7.357172] |
| 2022 | OOS | served_loso | accuracy | 60.952381 [55.700963, 66.497525] |
| 2022 | OOS | served_loso | brier | 0.243087 [0.237464, 0.248728] |
| 2022 | OOS | served_loso | log_loss | 0.679244 [0.667860, 0.690658] |
| 2022 | OOS | served_loso | rps | 6.741843 [6.172540, 7.374989] |
| 2022 | OOS | matched_zero | accuracy | 60.476190 [54.976303, 66.183575] |
| 2022 | OOS | matched_zero | brier | 0.242741 [0.236189, 0.249328] |
| 2022 | OOS | matched_zero | log_loss | 0.678532 [0.665253, 0.691871] |
| 2022 | OOS | matched_zero | rps | 6.730143 [6.164911, 7.357172] |
| 2022 | OOS | model_only | accuracy | 52.857143 [46.788991, 58.571429] |
| 2022 | OOS | model_only | brier | 0.251442 [0.244728, 0.259042] |
| 2022 | OOS | model_only | log_loss | 0.696349 [0.682592, 0.711942] |
| 2022 | OOS | model_only | rps | 6.883685 [6.276936, 7.573125] |
| 2022 | OOS | market | accuracy | 55.238095 [48.469388, 61.207098] |
| 2022 | OOS | market | brier | 0.242777 [0.236193, 0.249354] |
| 2022 | OOS | market | log_loss | 0.678364 [0.664840, 0.691768] |
| 2022 | OOS | market | rps | 6.735143 [6.133965, 7.409169] |
| 2022 | gap | candidate | accuracy | +2.513493 [-3.756172, +8.855899] |
| 2022 | gap | candidate | brier | -0.000952 [-0.009633, +0.007882] |
| 2022 | gap | candidate | log_loss | -0.001946 [-0.019887, +0.016362] |
| 2022 | gap | candidate | rps | -0.461316 [-1.173698, +0.270404] |
| 2022 | gap | served_loso | accuracy | +3.276628 [-2.711054, +9.378027] |
| 2022 | gap | served_loso | brier | -0.000391 [-0.008007, +0.007268] |
| 2022 | gap | served_loso | log_loss | -0.000685 [-0.016475, +0.015213] |
| 2022 | gap | served_loso | rps | -0.447077 [-1.160793, +0.291879] |
| 2022 | gap | matched_zero | accuracy | +2.513493 [-3.756172, +8.855899] |
| 2022 | gap | matched_zero | brier | -0.000952 [-0.009633, +0.007882] |
| 2022 | gap | matched_zero | log_loss | -0.001946 [-0.019887, +0.016362] |
| 2022 | gap | matched_zero | rps | -0.461316 [-1.173698, +0.270404] |
| 2022 | gap | model_only | accuracy | -0.514450 [-7.534168, +6.286010] |
| 2022 | gap | model_only | brier | -0.000582 [-0.008598, +0.007971] |
| 2022 | gap | model_only | log_loss | -0.001097 [-0.017462, +0.016506] |
| 2022 | gap | model_only | rps | -0.401533 [-1.146947, +0.388838] |
| 2022 | gap | market | accuracy | +2.440391 [-4.940269, +9.162945] |
| 2022 | gap | market | brier | -0.003211 [-0.010506, +0.004221] |
| 2022 | gap | market | log_loss | -0.006589 [-0.021606, +0.008554] |
| 2022 | gap | market | rps | -0.468174 [-1.203415, +0.312155] |
| 2023 | IS | candidate | accuracy | 57.427259 [54.173228, 60.591900] |
| 2023 | IS | candidate | brier | 0.246492 [0.239349, 0.253280] |
| 2023 | IS | candidate | log_loss | 0.686392 [0.670652, 0.701347] |
| 2023 | IS | candidate | rps | 7.119868 [6.728229, 7.527630] |
| 2023 | IS | served_loso | accuracy | 57.886677 [54.237155, 61.469265] |
| 2023 | IS | served_loso | brier | 0.244826 [0.239424, 0.249926] |
| 2023 | IS | served_loso | log_loss | 0.682601 [0.671148, 0.693317] |
| 2023 | IS | served_loso | rps | 7.092741 [6.707028, 7.498200] |
| 2023 | IS | matched_zero | accuracy | 57.427259 [54.173228, 60.591900] |
| 2023 | IS | matched_zero | brier | 0.246492 [0.239349, 0.253280] |
| 2023 | IS | matched_zero | log_loss | 0.686392 [0.670652, 0.701347] |
| 2023 | IS | matched_zero | rps | 7.119868 [6.728229, 7.527630] |
| 2023 | IS | model_only | accuracy | 54.211332 [50.155763, 58.104079] |
| 2023 | IS | model_only | brier | 0.252607 [0.247199, 0.258097] |
| 2023 | IS | model_only | log_loss | 0.698806 [0.687687, 0.710111] |
| 2023 | IS | model_only | rps | 7.163732 [6.770747, 7.577389] |
| 2023 | IS | market | accuracy | 54.670750 [51.931845, 57.401813] |
| 2023 | IS | market | brier | 0.246531 [0.242909, 0.250006] |
| 2023 | IS | market | log_loss | 0.685999 [0.678502, 0.693228] |
| 2023 | IS | market | rps | 7.106473 [6.730264, 7.501044] |
| 2023 | OOS | candidate | accuracy | 55.793991 [51.754386, 59.493932] |
| 2023 | OOS | candidate | brier | 0.242750 [0.230202, 0.255111] |
| 2023 | OOS | candidate | log_loss | 0.680230 [0.652400, 0.707824] |
| 2023 | OOS | candidate | rps | 7.378090 [6.579609, 8.223611] |
| 2023 | OOS | served_loso | accuracy | 56.223176 [51.931330, 60.169492] |
| 2023 | OOS | served_loso | brier | 0.242638 [0.233115, 0.252105] |
| 2023 | OOS | served_loso | log_loss | 0.678669 [0.658757, 0.698596] |
| 2023 | OOS | served_loso | rps | 7.378143 [6.579068, 8.223995] |
| 2023 | OOS | matched_zero | accuracy | 55.793991 [51.754386, 59.493932] |
| 2023 | OOS | matched_zero | brier | 0.242750 [0.230202, 0.255111] |
| 2023 | OOS | matched_zero | log_loss | 0.680230 [0.652400, 0.707824] |
| 2023 | OOS | matched_zero | rps | 7.378090 [6.579609, 8.223611] |
| 2023 | OOS | model_only | accuracy | 50.643777 [46.280355, 55.285182] |
| 2023 | OOS | model_only | brier | 0.255391 [0.249714, 0.260878] |
| 2023 | OOS | model_only | log_loss | 0.704461 [0.692758, 0.715936] |
| 2023 | OOS | model_only | rps | 7.511568 [6.687139, 8.367076] |
| 2023 | OOS | market | accuracy | 52.360515 [46.188293, 58.723404] |
| 2023 | OOS | market | brier | 0.245686 [0.240688, 0.250733] |
| 2023 | OOS | market | log_loss | 0.684412 [0.674280, 0.694654] |
| 2023 | OOS | market | rps | 7.408366 [6.581872, 8.281457] |
| 2023 | gap | candidate | accuracy | -1.633267 [-6.753793, +3.338497] |
| 2023 | gap | candidate | brier | -0.003743 [-0.018114, +0.010648] |
| 2023 | gap | candidate | log_loss | -0.006161 [-0.038174, +0.025775] |
| 2023 | gap | candidate | rps | +0.258222 [-0.647398, +1.191405] |
| 2023 | gap | served_loso | accuracy | -1.663501 [-7.354014, +3.723984] |
| 2023 | gap | served_loso | brier | -0.002188 [-0.013088, +0.008639] |
| 2023 | gap | served_loso | log_loss | -0.003932 [-0.026856, +0.018840] |
| 2023 | gap | served_loso | rps | +0.285402 [-0.620907, +1.214876] |
| 2023 | gap | matched_zero | accuracy | -1.633267 [-6.753793, +3.338497] |
| 2023 | gap | matched_zero | brier | -0.003743 [-0.018114, +0.010648] |
| 2023 | gap | matched_zero | log_loss | -0.006161 [-0.038174, +0.025775] |
| 2023 | gap | matched_zero | rps | +0.258222 [-0.647398, +1.191405] |
| 2023 | gap | model_only | accuracy | -3.567555 [-9.469333, +2.545618] |
| 2023 | gap | model_only | brier | +0.002783 [-0.005081, +0.010596] |
| 2023 | gap | model_only | log_loss | +0.005655 [-0.010495, +0.021724] |
| 2023 | gap | model_only | rps | +0.347836 [-0.580119, +1.290333] |
| 2023 | gap | market | accuracy | -2.310235 [-9.048533, +4.569456] |
| 2023 | gap | market | brier | -0.000844 [-0.006950, +0.005304] |
| 2023 | gap | market | log_loss | -0.001587 [-0.014025, +0.011004] |
| 2023 | gap | market | rps | +0.301894 [-0.621788, +1.255651] |
| 2024 | IS | candidate | accuracy | 53.312303 [49.606268, 57.009346] |
| 2024 | IS | candidate | brier | 0.245578 [0.240322, 0.250370] |
| 2024 | IS | candidate | log_loss | 0.683956 [0.672821, 0.693954] |
| 2024 | IS | candidate | rps | 7.420357 [7.005099, 7.875801] |
| 2024 | IS | served_loso | accuracy | 56.782334 [52.866242, 60.663507] |
| 2024 | IS | served_loso | brier | 0.244912 [0.239178, 0.250124] |
| 2024 | IS | served_loso | log_loss | 0.682610 [0.670482, 0.693597] |
| 2024 | IS | served_loso | rps | 7.412727 [6.998548, 7.864006] |
| 2024 | IS | matched_zero | accuracy | 53.312303 [49.606268, 57.009346] |
| 2024 | IS | matched_zero | brier | 0.245578 [0.240322, 0.250370] |
| 2024 | IS | matched_zero | log_loss | 0.683956 [0.672821, 0.693954] |
| 2024 | IS | matched_zero | rps | 7.420357 [7.005099, 7.875801] |
| 2024 | IS | model_only | accuracy | 54.258675 [50.240732, 58.201954] |
| 2024 | IS | model_only | brier | 0.252098 [0.245203, 0.259227] |
| 2024 | IS | model_only | log_loss | 0.697838 [0.683601, 0.712543] |
| 2024 | IS | model_only | rps | 7.490243 [7.054539, 7.967253] |
| 2024 | IS | market | accuracy | 55.362776 [52.495974, 58.179082] |
| 2024 | IS | market | brier | 0.245717 [0.242182, 0.249227] |
| 2024 | IS | market | log_loss | 0.684280 [0.676996, 0.691489] |
| 2024 | IS | market | rps | 7.414942 [7.008956, 7.854932] |
| 2024 | OOS | candidate | accuracy | 52.155172 [46.581197, 57.692308] |
| 2024 | OOS | candidate | brier | 0.246026 [0.238858, 0.253616] |
| 2024 | OOS | candidate | log_loss | 0.685389 [0.670408, 0.701331] |
| 2024 | OOS | candidate | rps | 7.239322 [6.662821, 7.829289] |
| 2024 | OOS | served_loso | accuracy | 58.620690 [53.246753, 63.636364] |
| 2024 | OOS | served_loso | brier | 0.244951 [0.237068, 0.253651] |
| 2024 | OOS | served_loso | log_loss | 0.683323 [0.666814, 0.701567] |
| 2024 | OOS | served_loso | rps | 7.238240 [6.662346, 7.826526] |
| 2024 | OOS | matched_zero | accuracy | 52.155172 [46.581197, 57.692308] |
| 2024 | OOS | matched_zero | brier | 0.246026 [0.238858, 0.253616] |
| 2024 | OOS | matched_zero | log_loss | 0.685389 [0.670408, 0.701331] |
| 2024 | OOS | matched_zero | rps | 7.239322 [6.662821, 7.829289] |
| 2024 | OOS | model_only | accuracy | 56.034483 [48.260870, 63.333333] |
| 2024 | OOS | model_only | brier | 0.249274 [0.241568, 0.257349] |
| 2024 | OOS | model_only | log_loss | 0.691784 [0.676183, 0.708139] |
| 2024 | OOS | model_only | rps | 7.255877 [6.673122, 7.830753] |
| 2024 | OOS | market | accuracy | 53.448276 [49.145115, 57.692731] |
| 2024 | OOS | market | brier | 0.248572 [0.244233, 0.252539] |
| 2024 | OOS | market | log_loss | 0.690250 [0.681468, 0.698298] |
| 2024 | OOS | market | rps | 7.231402 [6.676967, 7.796014] |
| 2024 | gap | candidate | accuracy | -1.157130 [-7.809781, +5.509875] |
| 2024 | gap | candidate | brier | +0.000448 [-0.008273, +0.009440] |
| 2024 | gap | candidate | log_loss | +0.001432 [-0.016826, +0.020352] |
| 2024 | gap | candidate | rps | -0.181035 [-0.913103, +0.552196] |
| 2024 | gap | served_loso | accuracy | +1.838355 [-4.622207, +8.202586] |
| 2024 | gap | served_loso | brier | +0.000040 [-0.009452, +0.009994] |
| 2024 | gap | served_loso | log_loss | +0.000713 [-0.019120, +0.021779] |
| 2024 | gap | served_loso | rps | -0.174487 [-0.903420, +0.560459] |
| 2024 | gap | matched_zero | accuracy | -1.157130 [-7.809781, +5.509875] |
| 2024 | gap | matched_zero | brier | +0.000448 [-0.008273, +0.009440] |
| 2024 | gap | matched_zero | log_loss | +0.001432 [-0.016826, +0.020352] |
| 2024 | gap | matched_zero | rps | -0.181035 [-0.913103, +0.552196] |
| 2024 | gap | model_only | accuracy | +1.775808 [-6.798726, +10.231178] |
| 2024 | gap | model_only | brier | -0.002824 [-0.013324, +0.007774] |
| 2024 | gap | model_only | log_loss | -0.006054 [-0.027486, +0.015513] |
| 2024 | gap | model_only | rps | -0.234366 [-0.987711, +0.503115] |
| 2024 | gap | market | accuracy | -1.914500 [-7.075249, +3.315110] |
| 2024 | gap | market | brier | +0.002855 [-0.002671, +0.008289] |
| 2024 | gap | market | log_loss | +0.005971 [-0.005241, +0.017014] |
| 2024 | gap | market | rps | -0.183540 [-0.893893, +0.530838] |
| 2025 | IS | candidate | accuracy | 57.679739 [53.611557, 61.677708] |
| 2025 | IS | candidate | brier | 0.244437 [0.239202, 0.249595] |
| 2025 | IS | candidate | log_loss | 0.681925 [0.671305, 0.692421] |
| 2025 | IS | candidate | rps | 7.327130 [6.925811, 7.769337] |
| 2025 | IS | served_loso | accuracy | 57.026144 [53.074434, 60.876817] |
| 2025 | IS | served_loso | brier | 0.244441 [0.239667, 0.249169] |
| 2025 | IS | served_loso | log_loss | 0.681921 [0.672276, 0.691516] |
| 2025 | IS | served_loso | rps | 7.327644 [6.927815, 7.770141] |
| 2025 | IS | matched_zero | accuracy | 57.679739 [53.611557, 61.677708] |
| 2025 | IS | matched_zero | brier | 0.244435 [0.239217, 0.249581] |
| 2025 | IS | matched_zero | log_loss | 0.681920 [0.671321, 0.692408] |
| 2025 | IS | matched_zero | rps | 7.327120 [6.925802, 7.769503] |
| 2025 | IS | model_only | accuracy | 54.084967 [49.918167, 58.143322] |
| 2025 | IS | model_only | brier | 0.252139 [0.244918, 0.259455] |
| 2025 | IS | model_only | log_loss | 0.698000 [0.683123, 0.713110] |
| 2025 | IS | model_only | rps | 7.409659 [6.979297, 7.884117] |
| 2025 | IS | market | accuracy | 56.045752 [52.782345, 59.283415] |
| 2025 | IS | market | brier | 0.245267 [0.241906, 0.248697] |
| 2025 | IS | market | log_loss | 0.683382 [0.676461, 0.690404] |
| 2025 | IS | market | rps | 7.324839 [6.915826, 7.781338] |
| 2025 | OOS | candidate | accuracy | 51.724138 [46.846847, 56.722689] |
| 2025 | OOS | candidate | brier | 0.248925 [0.241722, 0.255977] |
| 2025 | OOS | candidate | log_loss | 0.691152 [0.676407, 0.705624] |
| 2025 | OOS | candidate | rps | 7.150288 [6.492957, 7.833434] |
| 2025 | OOS | served_loso | accuracy | 59.051724 [53.508772, 64.680851] |
| 2025 | OOS | served_loso | brier | 0.244287 [0.232693, 0.254659] |
| 2025 | OOS | served_loso | log_loss | 0.681144 [0.656330, 0.703033] |
| 2025 | OOS | served_loso | rps | 7.048873 [6.377730, 7.760750] |
| 2025 | OOS | matched_zero | accuracy | 51.724138 [46.846847, 56.722689] |
| 2025 | OOS | matched_zero | brier | 0.248924 [0.241729, 0.255965] |
| 2025 | OOS | matched_zero | log_loss | 0.691149 [0.676423, 0.705602] |
| 2025 | OOS | matched_zero | rps | 7.150258 [6.493122, 7.833390] |
| 2025 | OOS | model_only | accuracy | 53.448276 [47.234043, 59.111111] |
| 2025 | OOS | model_only | brier | 0.251394 [0.244022, 0.259270] |
| 2025 | OOS | model_only | log_loss | 0.696063 [0.681112, 0.712107] |
| 2025 | OOS | model_only | rps | 7.177303 [6.496665, 7.890200] |
| 2025 | OOS | market | accuracy | 53.017241 [48.068670, 58.222222] |
| 2025 | OOS | market | brier | 0.244917 [0.239066, 0.250217] |
| 2025 | OOS | market | log_loss | 0.682795 [0.670889, 0.693585] |
| 2025 | OOS | market | rps | 7.051861 [6.421638, 7.708700] |
| 2025 | gap | candidate | accuracy | -5.955601 [-12.447679, +0.523755] |
| 2025 | gap | candidate | brier | +0.004488 [-0.004384, +0.013227] |
| 2025 | gap | candidate | log_loss | +0.009226 [-0.008973, +0.027221] |
| 2025 | gap | candidate | rps | -0.176842 [-0.981593, +0.637194] |
| 2025 | gap | served_loso | accuracy | +2.025580 [-4.822519, +8.832070] |
| 2025 | gap | served_loso | brier | -0.000154 [-0.012800, +0.011462] |
| 2025 | gap | served_loso | log_loss | -0.000777 [-0.027375, +0.023640] |
| 2025 | gap | served_loso | rps | -0.278771 [-1.087758, +0.549104] |
| 2025 | gap | matched_zero | accuracy | -5.955601 [-12.447679, +0.523755] |
| 2025 | gap | matched_zero | brier | +0.004489 [-0.004372, +0.013213] |
| 2025 | gap | matched_zero | log_loss | +0.009229 [-0.008946, +0.027198] |
| 2025 | gap | matched_zero | rps | -0.176862 [-0.981476, +0.636782] |
| 2025 | gap | model_only | accuracy | -0.636691 [-8.028131, +6.377095] |
| 2025 | gap | model_only | brier | -0.000745 [-0.011262, +0.009986] |
| 2025 | gap | model_only | log_loss | -0.001937 [-0.023408, +0.020028] |
| 2025 | gap | model_only | rps | -0.232355 [-1.067253, +0.613602] |
| 2025 | gap | market | accuracy | -3.028510 [-8.931949, +3.019457] |
| 2025 | gap | market | brier | -0.000350 [-0.007019, +0.005963] |
| 2025 | gap | market | log_loss | -0.000587 [-0.014226, +0.012303] |
| 2025 | gap | market | rps | -0.272978 [-1.046284, +0.508238] |
| pooled | IS | candidate | accuracy | 57.295646 [55.059036, 59.420290] |
| pooled | IS | candidate | brier | 0.244502 [0.240619, 0.248461] |
| pooled | IS | candidate | log_loss | 0.682209 [0.673877, 0.690577] |
| pooled | IS | candidate | rps | 7.263512 [6.972514, 7.568219] |
| pooled | IS | served_loso | accuracy | 57.524828 [55.052375, 59.897455] |
| pooled | IS | served_loso | brier | 0.243997 [0.240440, 0.247612] |
| pooled | IS | served_loso | log_loss | 0.681013 [0.673607, 0.688528] |
| pooled | IS | served_loso | rps | 7.255924 [6.965997, 7.559523] |
| pooled | IS | matched_zero | accuracy | 57.422969 [55.214704, 59.523810] |
| pooled | IS | matched_zero | brier | 0.244414 [0.240654, 0.248284] |
| pooled | IS | matched_zero | log_loss | 0.681964 [0.673968, 0.690087] |
| pooled | IS | matched_zero | rps | 7.261498 [6.970436, 7.565523] |
| pooled | IS | model_only | accuracy | 53.705118 [50.927321, 56.419669] |
| pooled | IS | model_only | brier | 0.252078 [0.248047, 0.256226] |
| pooled | IS | model_only | log_loss | 0.697705 [0.689464, 0.706200] |
| pooled | IS | model_only | rps | 7.344594 [7.041441, 7.661185] |
| pooled | IS | market | accuracy | 54.596384 [52.409853, 56.735964] |
| pooled | IS | market | brier | 0.245589 [0.243235, 0.247919] |
| pooled | IS | market | log_loss | 0.684093 [0.679278, 0.688841] |
| pooled | IS | market | rps | 7.262978 [6.972027, 7.564608] |
| pooled | OOS | candidate | accuracy | 55.080214 [52.651806, 57.448635] |
| pooled | OOS | candidate | brier | 0.245404 [0.241868, 0.249017] |
| pooled | OOS | candidate | log_loss | 0.684251 [0.676776, 0.691952] |
| pooled | OOS | candidate | rps | 7.311439 [7.023276, 7.610219] |
| pooled | OOS | served_loso | accuracy | 57.830405 [55.392912, 60.197138] |
| pooled | OOS | served_loso | brier | 0.244503 [0.240989, 0.248078] |
| pooled | OOS | served_loso | log_loss | 0.682106 [0.674788, 0.689498] |
| pooled | OOS | served_loso | rps | 7.296410 [7.008789, 7.598063] |
| pooled | OOS | matched_zero | accuracy | 54.927426 [52.529123, 57.282282] |
| pooled | OOS | matched_zero | brier | 0.245478 [0.241919, 0.249114] |
| pooled | OOS | matched_zero | log_loss | 0.684400 [0.676892, 0.692129] |
| pooled | OOS | matched_zero | rps | 7.312452 [7.023979, 7.611091] |
| pooled | OOS | model_only | accuracy | 53.705118 [50.927321, 56.419669] |
| pooled | OOS | model_only | brier | 0.252078 [0.248047, 0.256226] |
| pooled | OOS | model_only | log_loss | 0.697705 [0.689464, 0.706200] |
| pooled | OOS | model_only | rps | 7.375170 [7.075671, 7.686392] |
| pooled | OOS | market | accuracy | 54.316272 [52.099519, 56.501565] |
| pooled | OOS | market | brier | 0.245869 [0.243457, 0.248268] |
| pooled | OOS | market | log_loss | 0.684681 [0.679729, 0.689570] |
| pooled | OOS | market | rps | 7.299082 [7.007504, 7.601226] |
| pooled | gap | candidate | accuracy | -2.215432 [-3.881340, -0.515038] |
| pooled | gap | candidate | brier | +0.000903 [-0.001088, +0.003055] |
| pooled | gap | candidate | log_loss | +0.002042 [-0.002300, +0.006778] |
| pooled | gap | candidate | rps | +0.047927 [+0.018742, +0.078732] |
| pooled | gap | served_loso | accuracy | +0.305577 [-0.385230, +1.007771] |
| pooled | gap | served_loso | brier | +0.000507 [+0.000073, +0.000936] |
| pooled | gap | served_loso | log_loss | +0.001093 [+0.000178, +0.002005] |
| pooled | gap | served_loso | rps | +0.040486 [+0.021836, +0.058195] |
| pooled | gap | matched_zero | accuracy | -2.495544 [-4.102177, -0.867109] |
| pooled | gap | matched_zero | brier | +0.001065 [-0.000982, +0.003273] |
| pooled | gap | matched_zero | log_loss | +0.002435 [-0.002044, +0.007294] |
| pooled | gap | matched_zero | rps | +0.050954 [+0.020326, +0.082922] |
| pooled | gap | model_only | accuracy | +0.000000 [+0.000000, +0.000000] |
| pooled | gap | model_only | brier | +0.000000 [-0.000000, +0.000000] |
| pooled | gap | model_only | log_loss | -0.000000 [-0.000000, +0.000000] |
| pooled | gap | model_only | rps | +0.030577 [+0.012641, +0.048017] |
| pooled | gap | market | accuracy | -0.280112 [-1.058201, +0.481256] |
| pooled | gap | market | brier | +0.000280 [-0.000317, +0.000859] |
| pooled | gap | market | log_loss | +0.000588 [-0.000649, +0.001790] |
| pooled | gap | market | rps | +0.036104 [+0.017213, +0.054092] |

## Paired candidate improvements

**Measured:** losses use comparator minus candidate; accuracy uses candidate minus
comparator. Improvement gaps are OOS improvement minus IS improvement. Baselines
share each scored population and discrete lattice; conditional cover mass differs.

| Fold | Scheme | Against | Metric | Improvement, interval | probability_positive |
| --- | --- | --- | --- | --- | --- |
| 2020 | IS | served_loso | accuracy | +0.152439 [-1.047904, +1.257911] | 0.5983 |
| 2020 | IS | served_loso | brier | -0.000146 [-0.000749, +0.000490] | 0.3233 |
| 2020 | IS | served_loso | log_loss | -0.000272 [-0.001588, +0.001129] | 0.3465 |
| 2020 | IS | served_loso | rps | -0.001546 [-0.010163, +0.007247] | 0.3680 |
| 2020 | IS | matched_zero | accuracy | -0.457317 [-0.955414, +0.000000] | 0.0215 |
| 2020 | IS | matched_zero | brier | -0.000005 [-0.000120, +0.000107] | 0.4750 |
| 2020 | IS | matched_zero | log_loss | -0.000009 [-0.000252, +0.000224] | 0.4771 |
| 2020 | IS | matched_zero | rps | -0.000149 [-0.001641, +0.001339] | 0.4285 |
| 2020 | IS | model_only | accuracy | +5.030488 [+1.538462, +8.384146] | 0.9970 |
| 2020 | IS | model_only | brier | +0.008081 [+0.002889, +0.013599] | 0.9992 |
| 2020 | IS | model_only | log_loss | +0.016529 [+0.005824, +0.027908] | 0.9993 |
| 2020 | IS | model_only | rps | +0.101028 [+0.029412, +0.175260] | 0.9973 |
| 2020 | IS | market | accuracy | +2.896341 [-2.259121, +8.096158] | 0.8679 |
| 2020 | IS | market | brier | +0.001411 [-0.004153, +0.007090] | 0.6903 |
| 2020 | IS | market | log_loss | +0.002611 [-0.008723, +0.014146] | 0.6749 |
| 2020 | IS | market | rps | +0.002406 [-0.068659, +0.073918] | 0.5319 |
| 2020 | OOS | served_loso | accuracy | -1.587302 [-4.255319, +0.588235] | 0.1158 |
| 2020 | OOS | served_loso | brier | +0.000111 [-0.000414, +0.000597] | 0.6739 |
| 2020 | OOS | served_loso | log_loss | +0.000219 [-0.000843, +0.001198] | 0.6715 |
| 2020 | OOS | served_loso | rps | +0.003347 [-0.002991, +0.009240] | 0.8596 |
| 2020 | OOS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2020 | OOS | matched_zero | brier | -0.000019 [-0.000088, +0.000050] | 0.3054 |
| 2020 | OOS | matched_zero | log_loss | -0.000039 [-0.000182, +0.000101] | 0.3009 |
| 2020 | OOS | matched_zero | rps | -0.000307 [-0.001297, +0.000708] | 0.2793 |
| 2020 | OOS | model_only | accuracy | +0.529101 [-4.787745, +6.077348] | 0.5797 |
| 2020 | OOS | model_only | brier | +0.011577 [+0.001161, +0.021790] | 0.9851 |
| 2020 | OOS | model_only | log_loss | +0.024495 [+0.002757, +0.045946] | 0.9865 |
| 2020 | OOS | model_only | rps | +0.043133 [-0.089185, +0.176502] | 0.7335 |
| 2020 | OOS | market | accuracy | -0.529101 [-8.955224, +8.064516] | 0.4486 |
| 2020 | OOS | market | brier | -0.000545 [-0.011193, +0.010275] | 0.4580 |
| 2020 | OOS | market | log_loss | -0.001469 [-0.023408, +0.020695] | 0.4481 |
| 2020 | OOS | market | rps | +0.027792 [-0.118800, +0.169918] | 0.6489 |
| 2020 | gap | served_loso | accuracy | -1.739741 [-4.653592, +0.971266] | 0.1121 |
| 2020 | gap | served_loso | brier | +0.000256 [-0.000558, +0.001044] | 0.7319 |
| 2020 | gap | served_loso | log_loss | +0.000491 [-0.001215, +0.002153] | 0.7123 |
| 2020 | gap | served_loso | rps | +0.004893 [-0.005668, +0.015469] | 0.8181 |
| 2020 | gap | matched_zero | accuracy | +0.457317 [+0.000000, +0.955414] | 0.9784 |
| 2020 | gap | matched_zero | brier | -0.000014 [-0.000145, +0.000121] | 0.4211 |
| 2020 | gap | matched_zero | log_loss | -0.000030 [-0.000301, +0.000251] | 0.4179 |
| 2020 | gap | matched_zero | rps | -0.000159 [-0.001948, +0.001646] | 0.4340 |
| 2020 | gap | model_only | accuracy | -4.501387 [-10.876688, +2.016891] | 0.0856 |
| 2020 | gap | model_only | brier | +0.003497 [-0.008230, +0.015134] | 0.7260 |
| 2020 | gap | model_only | log_loss | +0.007966 [-0.016439, +0.032388] | 0.7421 |
| 2020 | gap | model_only | rps | -0.057895 [-0.210471, +0.094888] | 0.2269 |
| 2020 | gap | market | accuracy | -3.425442 [-13.412453, +6.641675] | 0.2512 |
| 2020 | gap | market | brier | -0.001956 [-0.013995, +0.010239] | 0.3732 |
| 2020 | gap | market | log_loss | -0.004081 [-0.028719, +0.020866] | 0.3714 |
| 2020 | gap | market | rps | +0.025386 [-0.137529, +0.187498] | 0.6228 |
| 2021 | IS | served_loso | accuracy | +1.333333 [-0.452489, +3.288613] | 0.9236 |
| 2021 | IS | served_loso | brier | -0.000343 [-0.001906, +0.001169] | 0.3316 |
| 2021 | IS | served_loso | log_loss | -0.001190 [-0.004787, +0.002269] | 0.2556 |
| 2021 | IS | served_loso | rps | -0.007077 [-0.025934, +0.011317] | 0.2293 |
| 2021 | IS | matched_zero | accuracy | -0.296296 [-2.189781, +1.639405] | 0.3779 |
| 2021 | IS | matched_zero | brier | -0.000507 [-0.002474, +0.001400] | 0.3072 |
| 2021 | IS | matched_zero | log_loss | -0.001408 [-0.005831, +0.002816] | 0.2644 |
| 2021 | IS | matched_zero | rps | -0.011561 [-0.037679, +0.013568] | 0.1891 |
| 2021 | IS | model_only | accuracy | +5.777778 [+1.203008, +9.913793] | 0.9935 |
| 2021 | IS | model_only | brier | +0.008599 [+0.002050, +0.014860] | 0.9944 |
| 2021 | IS | model_only | log_loss | +0.016947 [+0.003017, +0.030166] | 0.9910 |
| 2021 | IS | model_only | rps | +0.093815 [+0.013250, +0.172125] | 0.9883 |
| 2021 | IS | market | accuracy | +5.333333 [+1.230722, +9.516837] | 0.9945 |
| 2021 | IS | market | brier | +0.001662 [-0.003949, +0.007256] | 0.7250 |
| 2021 | IS | market | log_loss | +0.002561 [-0.009398, +0.014292] | 0.6696 |
| 2021 | IS | market | rps | +0.002434 [-0.065619, +0.070220] | 0.5357 |
| 2021 | OOS | served_loso | accuracy | +0.469484 [-1.435407, +2.403846] | 0.6685 |
| 2021 | OOS | served_loso | brier | +0.000365 [-0.001297, +0.002224] | 0.6432 |
| 2021 | OOS | served_loso | log_loss | +0.000783 [-0.002653, +0.004620] | 0.6493 |
| 2021 | OOS | served_loso | rps | +0.004715 [-0.019972, +0.034455] | 0.6091 |
| 2021 | OOS | matched_zero | accuracy | +0.938967 [-0.925926, +2.803738] | 0.8441 |
| 2021 | OOS | matched_zero | brier | +0.000472 [-0.000241, +0.001095] | 0.9111 |
| 2021 | OOS | matched_zero | log_loss | +0.000952 [-0.000524, +0.002242] | 0.9070 |
| 2021 | OOS | matched_zero | rps | +0.006528 [-0.002615, +0.015726] | 0.9175 |
| 2021 | OOS | model_only | accuracy | +0.938967 [-5.581395, +7.426146] | 0.6104 |
| 2021 | OOS | model_only | brier | +0.002107 [-0.011514, +0.017804] | 0.5961 |
| 2021 | OOS | model_only | log_loss | +0.004562 [-0.023367, +0.036896] | 0.6021 |
| 2021 | OOS | model_only | rps | +0.008541 [-0.188509, +0.216407] | 0.5270 |
| 2021 | OOS | market | accuracy | -0.938967 [-10.891324, +9.004739] | 0.4256 |
| 2021 | OOS | market | brier | +0.001680 [-0.012229, +0.015587] | 0.5893 |
| 2021 | OOS | market | log_loss | +0.003347 [-0.025132, +0.031815] | 0.5877 |
| 2021 | OOS | market | rps | -0.022816 [-0.209906, +0.155486] | 0.4052 |
| 2021 | gap | served_loso | accuracy | -0.863850 [-3.717672, +1.860465] | 0.2698 |
| 2021 | gap | served_loso | brier | +0.000708 [-0.001568, +0.003133] | 0.7183 |
| 2021 | gap | served_loso | log_loss | +0.001972 [-0.003008, +0.007234] | 0.7746 |
| 2021 | gap | served_loso | rps | +0.011792 [-0.019685, +0.046765] | 0.7526 |
| 2021 | gap | matched_zero | accuracy | +1.235263 [-1.367833, +3.861537] | 0.8226 |
| 2021 | gap | matched_zero | brier | +0.000979 [-0.001057, +0.003066] | 0.8261 |
| 2021 | gap | matched_zero | log_loss | +0.002360 [-0.002110, +0.006945] | 0.8467 |
| 2021 | gap | matched_zero | rps | +0.018089 [-0.008869, +0.045546] | 0.9048 |
| 2021 | gap | model_only | accuracy | -4.838811 [-12.619708, +3.041870] | 0.1152 |
| 2021 | gap | model_only | brier | -0.006492 [-0.021481, +0.010053] | 0.2031 |
| 2021 | gap | model_only | log_loss | -0.012385 [-0.043293, +0.021781] | 0.2218 |
| 2021 | gap | model_only | rps | -0.085274 [-0.294196, +0.135210] | 0.2215 |
| 2021 | gap | market | accuracy | -6.272300 [-17.185949, +4.505558] | 0.1273 |
| 2021 | gap | market | brier | +0.000018 [-0.015161, +0.014947] | 0.4968 |
| 2021 | gap | market | log_loss | +0.000786 [-0.030363, +0.031239] | 0.5156 |
| 2021 | gap | market | rps | -0.025249 [-0.226298, +0.163800] | 0.4020 |
| 2022 | IS | served_loso | accuracy | +0.286944 [-1.711840, +2.312222] | 0.6028 |
| 2022 | IS | served_loso | brier | -0.000215 [-0.001523, +0.001075] | 0.3720 |
| 2022 | IS | served_loso | log_loss | -0.000549 [-0.003293, +0.002121] | 0.3439 |
| 2022 | IS | served_loso | rps | -0.002540 [-0.018971, +0.013612] | 0.3786 |
| 2022 | IS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2022 | IS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | IS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | IS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | IS | model_only | accuracy | +4.591105 [+0.000000, +9.039568] | 0.9742 |
| 2022 | IS | model_only | brier | +0.008331 [+0.001937, +0.014912] | 0.9962 |
| 2022 | IS | model_only | log_loss | +0.016968 [+0.003481, +0.030963] | 0.9949 |
| 2022 | IS | model_only | rps | +0.093759 [+0.020842, +0.170746] | 0.9946 |
| 2022 | IS | market | accuracy | +5.164993 [+1.149425, +9.262202] | 0.9931 |
| 2022 | IS | market | brier | +0.002295 [-0.002420, +0.007083] | 0.8274 |
| 2022 | IS | market | log_loss | +0.004475 [-0.005352, +0.014484] | 0.8081 |
| 2022 | IS | market | rps | +0.011858 [-0.044315, +0.066462] | 0.6595 |
| 2022 | OOS | served_loso | accuracy | -0.476190 [-3.301887, +2.347418] | 0.3597 |
| 2022 | OOS | served_loso | brier | +0.000346 [-0.000852, +0.001517] | 0.7193 |
| 2022 | OOS | served_loso | log_loss | +0.000712 [-0.001746, +0.003106] | 0.7204 |
| 2022 | OOS | served_loso | rps | +0.011700 [-0.005404, +0.029262] | 0.9111 |
| 2022 | OOS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2022 | OOS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | OOS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | OOS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | OOS | model_only | accuracy | +7.619048 [+1.851639, +13.953488] | 0.9957 |
| 2022 | OOS | model_only | brier | +0.008701 [-0.000318, +0.018057] | 0.9709 |
| 2022 | OOS | model_only | log_loss | +0.017817 [-0.000625, +0.036877] | 0.9713 |
| 2022 | OOS | model_only | rps | +0.153541 [+0.021822, +0.295118] | 0.9891 |
| 2022 | OOS | market | accuracy | +5.238095 [-2.791024, +13.846154] | 0.8909 |
| 2022 | OOS | market | brier | +0.000036 [-0.009985, +0.010308] | 0.5042 |
| 2022 | OOS | market | log_loss | -0.000168 [-0.020472, +0.020729] | 0.4935 |
| 2022 | OOS | market | rps | +0.005000 [-0.133119, +0.134885] | 0.5389 |
| 2022 | gap | served_loso | accuracy | -0.763135 [-4.268286, +2.755240] | 0.3223 |
| 2022 | gap | served_loso | brier | +0.000561 [-0.001201, +0.002313] | 0.7383 |
| 2022 | gap | served_loso | log_loss | +0.001261 [-0.002378, +0.004909] | 0.7569 |
| 2022 | gap | served_loso | rps | +0.014239 [-0.009479, +0.037617] | 0.8840 |
| 2022 | gap | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2022 | gap | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | gap | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | gap | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2022 | gap | model_only | accuracy | +3.027943 [-4.292793, +10.749404] | 0.7890 |
| 2022 | gap | model_only | brier | +0.000370 [-0.010877, +0.011885] | 0.5312 |
| 2022 | gap | model_only | log_loss | +0.000849 [-0.022451, +0.024585] | 0.5348 |
| 2022 | gap | model_only | rps | +0.059783 [-0.095561, +0.217834] | 0.7763 |
| 2022 | gap | market | accuracy | +0.073102 [-9.124385, +9.683221] | 0.5070 |
| 2022 | gap | market | brier | -0.002259 [-0.013497, +0.009025] | 0.3513 |
| 2022 | gap | market | log_loss | -0.004643 [-0.027598, +0.018587] | 0.3512 |
| 2022 | gap | market | rps | -0.006858 [-0.157645, +0.134825] | 0.4774 |
| 2023 | IS | served_loso | accuracy | -0.459418 [-2.108434, +1.111111] | 0.2923 |
| 2023 | IS | served_loso | brier | -0.001666 [-0.003812, +0.000437] | 0.0606 |
| 2023 | IS | served_loso | log_loss | -0.003790 [-0.008897, +0.001267] | 0.0701 |
| 2023 | IS | served_loso | rps | -0.027127 [-0.056884, +0.002067] | 0.0362 |
| 2023 | IS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2023 | IS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | IS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | IS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | IS | model_only | accuracy | +3.215926 [-0.614463, +7.196402] | 0.9508 |
| 2023 | IS | model_only | brier | +0.006115 [-0.001380, +0.014290] | 0.9450 |
| 2023 | IS | model_only | log_loss | +0.012415 [-0.004029, +0.030459] | 0.9315 |
| 2023 | IS | model_only | rps | +0.043864 [-0.049691, +0.145697] | 0.8163 |
| 2023 | IS | market | accuracy | +2.756508 [-0.618262, +6.332892] | 0.9417 |
| 2023 | IS | market | brier | +0.000039 [-0.006233, +0.006429] | 0.5106 |
| 2023 | IS | market | log_loss | -0.000392 [-0.014099, +0.013621] | 0.4829 |
| 2023 | IS | market | rps | -0.013395 [-0.098547, +0.070906] | 0.3902 |
| 2023 | OOS | served_loso | accuracy | -0.429185 [-3.571429, +3.347280] | 0.3833 |
| 2023 | OOS | served_loso | brier | -0.000112 [-0.003676, +0.003377] | 0.4860 |
| 2023 | OOS | served_loso | log_loss | -0.001561 [-0.010242, +0.007097] | 0.3712 |
| 2023 | OOS | served_loso | rps | +0.000053 [-0.044595, +0.045203] | 0.5031 |
| 2023 | OOS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2023 | OOS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | OOS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | OOS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | OOS | model_only | accuracy | +5.150215 [-1.731790, +10.833333] | 0.9382 |
| 2023 | OOS | model_only | brier | +0.012641 [+0.000958, +0.025076] | 0.9838 |
| 2023 | OOS | model_only | log_loss | +0.024230 [-0.001864, +0.051726] | 0.9646 |
| 2023 | OOS | model_only | rps | +0.133478 [-0.004724, +0.272594] | 0.9695 |
| 2023 | OOS | market | accuracy | +3.433476 [-4.347826, +10.460251] | 0.8185 |
| 2023 | OOS | market | brier | +0.002937 [-0.008258, +0.013399] | 0.7113 |
| 2023 | OOS | market | log_loss | +0.004182 [-0.020290, +0.027544] | 0.6471 |
| 2023 | OOS | market | rps | +0.030276 [-0.097754, +0.160131] | 0.6794 |
| 2023 | gap | served_loso | accuracy | +0.030234 [-3.559651, +4.037523] | 0.4904 |
| 2023 | gap | served_loso | brier | +0.001555 [-0.002677, +0.005606] | 0.7743 |
| 2023 | gap | served_loso | log_loss | +0.002229 [-0.007980, +0.012335] | 0.6713 |
| 2023 | gap | served_loso | rps | +0.027180 [-0.025637, +0.081337] | 0.8379 |
| 2023 | gap | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2023 | gap | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | gap | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | gap | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2023 | gap | model_only | accuracy | +1.934288 [-5.730481, +8.786053] | 0.7024 |
| 2023 | gap | model_only | brier | +0.006526 [-0.007881, +0.021201] | 0.8164 |
| 2023 | gap | model_only | log_loss | +0.011816 [-0.020221, +0.043994] | 0.7720 |
| 2023 | gap | model_only | rps | +0.089614 [-0.083466, +0.254798] | 0.8500 |
| 2023 | gap | market | accuracy | +0.676968 [-7.807348, +8.647367] | 0.5697 |
| 2023 | gap | market | brier | +0.002898 [-0.009751, +0.015153] | 0.6774 |
| 2023 | gap | market | log_loss | +0.004574 [-0.023088, +0.031785] | 0.6309 |
| 2023 | gap | market | rps | +0.043672 [-0.107254, +0.198403] | 0.7085 |
| 2024 | IS | served_loso | accuracy | -3.470032 [-6.995311, +0.000000] | 0.0258 |
| 2024 | IS | served_loso | brier | -0.000666 [-0.002752, +0.001408] | 0.2707 |
| 2024 | IS | served_loso | log_loss | -0.001346 [-0.005595, +0.002870] | 0.2721 |
| 2024 | IS | served_loso | rps | -0.007630 [-0.034355, +0.020241] | 0.2935 |
| 2024 | IS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2024 | IS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | IS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | IS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | IS | model_only | accuracy | -0.946372 [-5.705230, +3.930818] | 0.3415 |
| 2024 | IS | model_only | brier | +0.006520 [-0.001061, +0.014770] | 0.9544 |
| 2024 | IS | model_only | log_loss | +0.013882 [-0.001869, +0.031225] | 0.9585 |
| 2024 | IS | model_only | rps | +0.069886 [-0.030171, +0.178379] | 0.9062 |
| 2024 | IS | market | accuracy | -2.050473 [-6.416275, +2.419551] | 0.1844 |
| 2024 | IS | market | brier | +0.000139 [-0.004796, +0.005134] | 0.5281 |
| 2024 | IS | market | log_loss | +0.000324 [-0.009919, +0.010713] | 0.5304 |
| 2024 | IS | market | rps | -0.005415 [-0.076612, +0.064256] | 0.4510 |
| 2024 | OOS | served_loso | accuracy | -6.465517 [-11.570248, -1.762115] | 0.0017 |
| 2024 | OOS | served_loso | brier | -0.001074 [-0.004056, +0.002041] | 0.2356 |
| 2024 | OOS | served_loso | log_loss | -0.002065 [-0.008124, +0.004300] | 0.2475 |
| 2024 | OOS | served_loso | rps | -0.001082 [-0.036465, +0.034980] | 0.4730 |
| 2024 | OOS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2024 | OOS | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | OOS | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | OOS | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | OOS | model_only | accuracy | -3.879310 [-10.859974, +4.109589] | 0.1654 |
| 2024 | OOS | model_only | brier | +0.003248 [-0.005805, +0.011810] | 0.7622 |
| 2024 | OOS | model_only | log_loss | +0.006395 [-0.012602, +0.024275] | 0.7513 |
| 2024 | OOS | model_only | rps | +0.016555 [-0.090760, +0.122227] | 0.6195 |
| 2024 | OOS | market | accuracy | -1.293103 [-6.808511, +4.273963] | 0.3252 |
| 2024 | OOS | market | brier | +0.002546 [-0.003485, +0.008484] | 0.7947 |
| 2024 | OOS | market | log_loss | +0.004862 [-0.007607, +0.017066] | 0.7776 |
| 2024 | OOS | market | rps | -0.007920 [-0.084562, +0.065987] | 0.4151 |
| 2024 | gap | served_loso | accuracy | -2.995486 [-9.073053, +2.768260] | 0.1682 |
| 2024 | gap | served_loso | brier | -0.000408 [-0.003999, +0.003271] | 0.4068 |
| 2024 | gap | served_loso | log_loss | -0.000719 [-0.008033, +0.006793] | 0.4192 |
| 2024 | gap | served_loso | rps | +0.006548 [-0.038964, +0.050672] | 0.6106 |
| 2024 | gap | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2024 | gap | matched_zero | brier | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | gap | matched_zero | log_loss | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | gap | matched_zero | rps | -0.000000 [-0.000000, +0.000000] | 0.5000 |
| 2024 | gap | model_only | accuracy | -2.932938 [-11.586924, +6.468373] | 0.2586 |
| 2024 | gap | model_only | brier | -0.003272 [-0.015277, +0.008240] | 0.2933 |
| 2024 | gap | model_only | log_loss | -0.007487 [-0.032650, +0.016501] | 0.2751 |
| 2024 | gap | model_only | rps | -0.053331 [-0.204136, +0.093615] | 0.2369 |
| 2024 | gap | market | accuracy | +0.757370 [-6.302584, +7.875430] | 0.5837 |
| 2024 | gap | market | brier | +0.002407 [-0.005171, +0.010044] | 0.7234 |
| 2024 | gap | market | log_loss | +0.004538 [-0.011209, +0.020288] | 0.7037 |
| 2024 | gap | market | rps | -0.002505 [-0.105121, +0.098750] | 0.4703 |
| 2025 | IS | served_loso | accuracy | +0.653595 [-0.330579, +1.768489] | 0.8915 |
| 2025 | IS | served_loso | brier | +0.000004 [-0.000581, +0.000594] | 0.5074 |
| 2025 | IS | served_loso | log_loss | -0.000004 [-0.001213, +0.001210] | 0.4988 |
| 2025 | IS | served_loso | rps | +0.000514 [-0.007060, +0.008100] | 0.5580 |
| 2025 | IS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2025 | IS | matched_zero | brier | -0.000002 [-0.000019, +0.000014] | 0.3993 |
| 2025 | IS | matched_zero | log_loss | -0.000005 [-0.000040, +0.000029] | 0.3856 |
| 2025 | IS | matched_zero | rps | -0.000010 [-0.000231, +0.000211] | 0.4671 |
| 2025 | IS | model_only | accuracy | +3.594771 [+0.160256, +7.011686] | 0.9790 |
| 2025 | IS | model_only | brier | +0.007702 [+0.001706, +0.013940] | 0.9950 |
| 2025 | IS | model_only | log_loss | +0.016075 [+0.003736, +0.028885] | 0.9952 |
| 2025 | IS | model_only | rps | +0.082529 [-0.005276, +0.171059] | 0.9668 |
| 2025 | IS | market | accuracy | +1.633987 [-3.582694, +6.935532] | 0.7336 |
| 2025 | IS | market | brier | +0.000830 [-0.005742, +0.007756] | 0.5962 |
| 2025 | IS | market | log_loss | +0.001457 [-0.011984, +0.015558] | 0.5834 |
| 2025 | IS | market | rps | -0.002291 [-0.090484, +0.087214] | 0.4852 |
| 2025 | OOS | served_loso | accuracy | -7.327586 [-11.255411, -3.004292] | 0.0006 |
| 2025 | OOS | served_loso | brier | -0.004638 [-0.012863, +0.002086] | 0.1108 |
| 2025 | OOS | served_loso | log_loss | -0.010008 [-0.027689, +0.004255] | 0.1086 |
| 2025 | OOS | served_loso | rps | -0.101415 [-0.210133, -0.011015] | 0.0117 |
| 2025 | OOS | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2025 | OOS | matched_zero | brier | -0.000001 [-0.000018, +0.000015] | 0.4596 |
| 2025 | OOS | matched_zero | log_loss | -0.000002 [-0.000036, +0.000030] | 0.4568 |
| 2025 | OOS | matched_zero | rps | -0.000030 [-0.000264, +0.000185] | 0.4048 |
| 2025 | OOS | model_only | accuracy | -1.724138 [-9.012876, +6.035136] | 0.3184 |
| 2025 | OOS | model_only | brier | +0.002469 [-0.004573, +0.009799] | 0.7468 |
| 2025 | OOS | model_only | log_loss | +0.004911 [-0.009512, +0.019798] | 0.7417 |
| 2025 | OOS | model_only | rps | +0.027015 [-0.055953, +0.107377] | 0.7490 |
| 2025 | OOS | market | accuracy | -1.293103 [-8.974359, +6.723398] | 0.3647 |
| 2025 | OOS | market | brier | -0.004008 [-0.012749, +0.005201] | 0.1880 |
| 2025 | OOS | market | log_loss | -0.008357 [-0.026287, +0.010555] | 0.1835 |
| 2025 | OOS | market | rps | -0.098427 [-0.220796, +0.015141] | 0.0477 |
| 2025 | gap | served_loso | accuracy | -7.981181 [-12.046029, -3.507109] | 0.0005 |
| 2025 | gap | served_loso | brier | -0.004642 [-0.012851, +0.002097] | 0.1122 |
| 2025 | gap | served_loso | log_loss | -0.010004 [-0.027788, +0.004332] | 0.1108 |
| 2025 | gap | served_loso | rps | -0.101929 [-0.211054, -0.011443] | 0.0115 |
| 2025 | gap | matched_zero | accuracy | +0.000000 [+0.000000, +0.000000] | 0.5000 |
| 2025 | gap | matched_zero | brier | +0.000001 [-0.000022, +0.000025] | 0.5439 |
| 2025 | gap | matched_zero | log_loss | +0.000003 [-0.000045, +0.000050] | 0.5507 |
| 2025 | gap | matched_zero | rps | -0.000020 [-0.000336, +0.000299] | 0.4513 |
| 2025 | gap | model_only | accuracy | -5.318909 [-13.254449, +3.090865] | 0.1031 |
| 2025 | gap | model_only | brier | -0.005233 [-0.014766, +0.004323] | 0.1395 |
| 2025 | gap | model_only | log_loss | -0.011163 [-0.030633, +0.008316] | 0.1291 |
| 2025 | gap | model_only | rps | -0.055514 [-0.175607, +0.064611] | 0.1829 |
| 2025 | gap | market | accuracy | -2.927090 [-12.385296, +6.875797] | 0.2656 |
| 2025 | gap | market | brier | -0.004838 [-0.015864, +0.006617] | 0.2025 |
| 2025 | gap | market | log_loss | -0.009813 [-0.032342, +0.013611] | 0.2049 |
| 2025 | gap | market | rps | -0.096136 [-0.246977, +0.049943] | 0.0999 |
| pooled | IS | served_loso | accuracy | -0.229183 [-1.103438, +0.669784] | 0.3049 |
| pooled | IS | served_loso | brier | -0.000505 [-0.001172, +0.000133] | 0.0573 |
| pooled | IS | served_loso | log_loss | -0.001196 [-0.002703, +0.000227] | 0.0502 |
| pooled | IS | served_loso | rps | -0.007588 [-0.016111, +0.000433] | 0.0326 |
| pooled | IS | matched_zero | accuracy | -0.127324 [-0.461894, +0.203252] | 0.2188 |
| pooled | IS | matched_zero | brier | -0.000088 [-0.000439, +0.000255] | 0.3116 |
| pooled | IS | matched_zero | log_loss | -0.000244 [-0.001029, +0.000518] | 0.2723 |
| pooled | IS | matched_zero | rps | -0.002014 [-0.006692, +0.002496] | 0.1934 |
| pooled | IS | model_only | accuracy | +3.590527 [+0.869005, +6.342018] | 0.9948 |
| pooled | IS | model_only | brier | +0.007576 [+0.002980, +0.012415] | 0.9987 |
| pooled | IS | model_only | log_loss | +0.015497 [+0.005774, +0.025796] | 0.9983 |
| pooled | IS | model_only | rps | +0.081082 [+0.019547, +0.143432] | 0.9959 |
| pooled | IS | market | accuracy | +2.699262 [-0.227964, +5.619120] | 0.9636 |
| pooled | IS | market | brier | +0.001087 [-0.002910, +0.004988] | 0.7046 |
| pooled | IS | market | log_loss | +0.001885 [-0.006462, +0.010044] | 0.6693 |
| pooled | IS | market | rps | -0.000534 [-0.052162, +0.048860] | 0.4917 |
| pooled | OOS | served_loso | accuracy | -2.750191 [-4.211332, -1.243153] | 0.0003 |
| pooled | OOS | served_loso | brier | -0.000901 [-0.002573, +0.000616] | 0.1343 |
| pooled | OOS | served_loso | log_loss | -0.002144 [-0.005799, +0.001183] | 0.1104 |
| pooled | OOS | served_loso | rps | -0.015029 [-0.037386, +0.005379] | 0.0809 |
| pooled | OOS | matched_zero | accuracy | +0.152788 [-0.151976, +0.458365] | 0.8424 |
| pooled | OOS | matched_zero | brier | +0.000074 [-0.000045, +0.000178] | 0.8984 |
| pooled | OOS | matched_zero | log_loss | +0.000149 [-0.000097, +0.000365] | 0.8948 |
| pooled | OOS | matched_zero | rps | +0.001013 [-0.000508, +0.002536] | 0.9027 |
| pooled | OOS | model_only | accuracy | +1.375095 [-1.357543, +4.131678] | 0.8350 |
| pooled | OOS | model_only | brier | +0.006674 [+0.002464, +0.011079] | 0.9981 |
| pooled | OOS | model_only | log_loss | +0.013454 [+0.004573, +0.022707] | 0.9973 |
| pooled | OOS | model_only | rps | +0.063731 [+0.006862, +0.121426] | 0.9864 |
| pooled | OOS | market | accuracy | +0.763942 [-2.441165, +3.978587] | 0.6716 |
| pooled | OOS | market | brier | +0.000464 [-0.003689, +0.004549] | 0.5917 |
| pooled | OOS | market | log_loss | +0.000430 [-0.008223, +0.008997] | 0.5419 |
| pooled | OOS | market | rps | -0.012357 [-0.067102, +0.041122] | 0.3310 |
| pooled | gap | served_loso | accuracy | -2.521008 [-4.281346, -0.759851] | 0.0028 |
| pooled | gap | served_loso | brier | -0.000396 [-0.002421, +0.001501] | 0.3578 |
| pooled | gap | served_loso | log_loss | -0.000949 [-0.005417, +0.003184] | 0.3442 |
| pooled | gap | served_loso | rps | -0.007441 [-0.034030, +0.017084] | 0.2936 |
| pooled | gap | matched_zero | accuracy | +0.280112 [-0.153492, +0.726817] | 0.8920 |
| pooled | gap | matched_zero | brier | +0.000162 [-0.000196, +0.000524] | 0.8077 |
| pooled | gap | matched_zero | log_loss | +0.000393 [-0.000394, +0.001199] | 0.8306 |
| pooled | gap | matched_zero | rps | +0.003026 [-0.001704, +0.007948] | 0.8898 |
| pooled | gap | model_only | accuracy | -2.215432 [-3.881340, -0.515038] | 0.0047 |
| pooled | gap | model_only | brier | -0.000903 [-0.003055, +0.001088] | 0.1940 |
| pooled | gap | model_only | log_loss | -0.002042 [-0.006778, +0.002300] | 0.1863 |
| pooled | gap | model_only | rps | -0.017351 [-0.044950, +0.008413] | 0.0969 |
| pooled | gap | market | accuracy | -1.935320 [-3.829401, -0.052180] | 0.0225 |
| pooled | gap | market | brier | -0.000623 [-0.002753, +0.001419] | 0.2811 |
| pooled | gap | market | log_loss | -0.001454 [-0.006142, +0.002977] | 0.2669 |
| pooled | gap | market | rps | -0.011823 [-0.039179, +0.014257] | 0.1885 |

## Opener records

**Measured:** forced-pick records are diagnostics, not evidence of a profitable or
stable edge, and not individual game probabilities.

| Fold | Arm | W-L |
| --- | --- | --- |
| 2020 | candidate | 101-88 |
| 2020 | served_loso | 104-85 |
| 2020 | matched_zero | 101-88 |
| 2020 | model_only | 100-89 |
| 2020 | market | 102-87 |
| 2021 | candidate | 122-91 |
| 2021 | served_loso | 121-92 |
| 2021 | matched_zero | 120-93 |
| 2021 | model_only | 120-93 |
| 2021 | market | 124-89 |
| 2022 | candidate | 127-83 |
| 2022 | served_loso | 128-82 |
| 2022 | matched_zero | 127-83 |
| 2022 | model_only | 111-99 |
| 2022 | market | 116-94 |
| 2023 | candidate | 130-103 |
| 2023 | served_loso | 131-102 |
| 2023 | matched_zero | 130-103 |
| 2023 | model_only | 118-115 |
| 2023 | market | 122-111 |
| 2024 | candidate | 121-111 |
| 2024 | served_loso | 136-96 |
| 2024 | matched_zero | 121-111 |
| 2024 | model_only | 130-102 |
| 2024 | market | 124-108 |
| 2025 | candidate | 120-112 |
| 2025 | served_loso | 137-95 |
| 2025 | matched_zero | 120-112 |
| 2025 | model_only | 124-108 |
| 2025 | market | 123-109 |
| pooled | candidate | 721-588 |
| pooled | served_loso | 757-552 |
| pooled | matched_zero | 719-590 |
| pooled | model_only | 703-606 |
| pooled | market | 711-598 |

## Reliability

**Measured:** five predeclared equal-width bands; empty cells remain visible and counted.

| Arm | Probability band | Games | Mean probability | Observed home-cover rate |
| --- | --- | --- | --- | --- |
| candidate | 0.0-0.2 | 3 | 0.141446 | 0.333333 |
| candidate | 0.2-0.4 | 124 | 0.359449 | 0.411290 |
| candidate | 0.4-0.6 | 1054 | 0.496504 | 0.490512 |
| candidate | 0.6-0.8 | 124 | 0.645361 | 0.669355 |
| candidate | 0.8-1.0 | 4 | 0.837810 | 1.000000 |
| served_loso | 0.0-0.2 | 1 | 0.199419 | 0.000000 |
| served_loso | 0.2-0.4 | 106 | 0.358508 | 0.377358 |
| served_loso | 0.4-0.6 | 1086 | 0.494302 | 0.493554 |
| served_loso | 0.6-0.8 | 116 | 0.644224 | 0.689655 |
| served_loso | 0.8-1.0 | 0 | unavailable | unavailable |
| matched_zero | 0.0-0.2 | 3 | 0.141446 | 0.333333 |
| matched_zero | 0.2-0.4 | 134 | 0.361538 | 0.410448 |
| matched_zero | 0.4-0.6 | 1041 | 0.497194 | 0.490874 |
| matched_zero | 0.6-0.8 | 127 | 0.645194 | 0.669291 |
| matched_zero | 0.8-1.0 | 4 | 0.837810 | 1.000000 |
| model_only | 0.0-0.2 | 0 | unavailable | unavailable |
| model_only | 0.2-0.4 | 188 | 0.366231 | 0.505319 |
| model_only | 0.4-0.6 | 1058 | 0.490633 | 0.498110 |
| model_only | 0.6-0.8 | 63 | 0.622587 | 0.539683 |
| model_only | 0.8-1.0 | 0 | unavailable | unavailable |
| market | 0.0-0.2 | 0 | unavailable | unavailable |
| market | 0.2-0.4 | 23 | 0.362165 | 0.434783 |
| market | 0.4-0.6 | 1235 | 0.502594 | 0.493927 |
| market | 0.6-0.8 | 50 | 0.646732 | 0.700000 |
| market | 0.8-1.0 | 1 | 0.812216 | 1.000000 |

## Look accounting and reproducibility

**Measured:** one candidate specification and one replay; two decision comparisons.
Mandatory diagnostics are not counted as only two looks: 420 absolute metric cells
+ 336 contrast cells + 28 decisive cells + 25 reliability cells + 42 fit/coefficient
summaries = 851 declared diagnostic looks. Actual per-game lattice optimizer calls
are separately enumerated above. The 35 forced-pick records restate accuracy cells;
coefficient ranges restate the six fixed fits. No best diagnostic is promoted.

**Measured:** rows, input checksums, exclusions, market masses/targets, coefficients,
fixed-prediction bootstrap summaries and the pre-run protocol snapshot are local:
tests/scratch/codex/lead86_unit2/. No prediction rows are written under docs.
Registry commands are prepared only in docs/lanes/lead86.md, candidate-versus-served
only, for serial orchestrator execution.

```bash
export UV_CACHE_DIR=tests/scratch/codex/lead86_unit2_uv_cache
export RUFF_CACHE_DIR=tests/scratch/codex/lead86_unit2_ruff_cache
export PYTHONDONTWRITEBYTECODE=1
.tools/uv.exe run --no-sync python scripts/lead86_unit2.py
.tools/uv.exe run --no-sync ruff check scripts/lead86_unit2.py
.tools/uv.exe run --no-sync ruff format --check scripts/lead86_unit2.py
```

Replay refuses to overwrite existing predictions. --summarize-only regenerates
this report from saved summaries without fitting or scoring.

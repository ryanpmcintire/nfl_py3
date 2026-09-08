# MOD-18 Lane I: home-side prior sensitivity

## Predeclaration (frozen before computation, 2026-09-08)

Read (`src/nfl_ats/home_side_location.py:85`): S3 fits the mean home
margin error with a 100-game prior toward zero and serves only buckets
7, 7.5-10, and 10.5+. S5a uses 50 games, S5b 25, S5c 200.
Everything else stays identical: bucket pooling across both home sides,
five trailing seasons, strictly earlier weeks, completed raw archive points,
ridge fit, and gaussian_median via MarginModel.predict(center_offset=...).
This is a predeclared sensitivity in both directions, not a search.

Decision rule, fixed before scoring: take the arm with the highest through-card
accuracy only if its through-card probability_positive against S3 exceeds 0.5
AND its Brier does not worsen with improvement probability_positive below 0.3;
otherwise keep S3. Exact candidate accuracy ties retain declaration order.
Loss comparisons use standalone home-cover probabilities, as in lanes G/H.

Replay S3 first on find_matching_opener_evaluation's active-matched archive;
stop before candidate computation if any served probability differs by >1e-9.
Grade non-push OPENER outcomes: paired accuracy points, whole-week bootstrap
20,000 draws, seed 20260817, within-week correlation ZERO, never estimated or
padded. Report overall, season, bucket, Brier, log loss and through the played
three-member card. Run overlay-composition on research per_game artifacts,
research_arm S5a/S5b/S5c and active_model_id research_laneI_S5a etc.
Read the linked Week 1 sidecar and predictions.csv, replay without regenerating,
report model/card side changes and fitted offsets/counts for each bucket.
Record every comparison cell through weak-signals record in family
mod18_home_side_location_v1, names mod18_home_side_location_v1_s5_*.

Inferred mechanism: the prior controls attenuation of the diagnosed home-side
location error; weaker shrinkage may recover signal or amplify archive noise.
This sensitivity reuses the mined archive on which S2/S3 were selected, with
correlated arms, and is not independent confirmation.

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is `unresolved_below_power`: record it with `nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not the validator. Never state that anything "needs more games"; decide on expected value, and state what the numbers imply for the DECISION (which prior to serve) before what is wrong with them.

## Decision and measured results

Inferred decision from measured `artifacts/research/laneI/cells.json`: keep S3, the 100-game prior. The highest-accuracy candidate, S5a, has through-card probability_positive 0.02490, below the predeclared 0.5 decision boundary. The Brier guard does not veto any arm. No production change is recommended.

Measured (`reproduction.json` in that directory): active-matched archive `F:\Repos\nfl_py3\artifacts\opener_evaluation\20260908T115957Z`, maximum served probability gap 3.77475828e-15, offset gap 6.66133815e-16. The replay passed before any candidate computation. Measured (`cells.json`): 1,503 non-push games, 107 weeks, 2020-2025; 20,000 whole-week draws, seed 20260817, correlation zero.

Measured decision table (`cells.json`; losses use standalone probabilities):

| Arm / prior | Standalone % | Card % | Card delta [95%], probability_positive | Brier | Log loss | Week 1 model/card changes |
|---|---:|---:|---|---:|---:|---:|
| S3 / 100 | 54.55755 | 55.88822 | reference | 0.251583991 | 0.696623617 | reference |
| S5a / 50 | 54.55755 | 55.48902 | -0.39920 [-0.86898, +0.00000]; 0.02490 | 0.251580071 | 0.696625555 | 0/0 |
| S5b / 25 | 54.49102 | 55.42249 | -0.46574 [-0.98879, +0.00000]; 0.02305 | 0.251618655 | 0.696720160 | 0/0 |
| S5c / 200 | 54.02528 | 55.35595 | -0.53227 [-0.94276, -0.13378]; 0.00115 | 0.251615322 | 0.696683713 | 0/0 |

Measured paired comparisons (`cells.json`): delta [95%], probability_positive; accuracy in percentage points, losses in baseline-minus-candidate units.

| Arm | Standalone | Brier improvement | Log-loss improvement |
|---|---|---|---|
| S5a | +0.00000 [-0.53946, +0.53369]; 0.45310 | +0.00000 [-0.00023, +0.00024]; 0.52045 | -0.00000 [-0.00049, +0.00048]; 0.50400 |
| S5b | -0.06653 [-0.73284, +0.59250]; 0.38800 | -0.00003 [-0.00052, +0.00044]; 0.44950 | -0.00010 [-0.00110, +0.00088]; 0.42875 |
| S5c | -0.53227 [-1.00874, -0.06627]; 0.00935 | -0.00003 [-0.00024, +0.00018]; 0.37975 | -0.00006 [-0.00048, +0.00037]; 0.38715 |

### Per-bucket comparisons

Measured (`cells.json`): each cell is delta [95%], probability_positive against S3.

| Group | n | S3 standalone/card % | Arm | Standalone | Card |
|---|---:|---:|---|---|---|
| cell_bucket_0-3_all | 552 | 56.159/57.428 | S5a | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_0-3_all | 552 | 56.159/57.428 | S5b | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_0-3_all | 552 | 56.159/57.428 | S5c | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_10p5plus_all | 131 | 47.328/51.145 | S5a | -0.76336 [-4.72441, +3.17460]; 0.28810 | -1.52672 [-5.26316, +2.18978]; 0.14985 |
| cell_bucket_10p5plus_all | 131 | 47.328/51.145 | S5b | -0.76336 [-5.30320, +3.81679]; 0.31095 | -2.29008 [-6.40000, +1.53846]; 0.08605 |
| cell_bucket_10p5plus_all | 131 | 47.328/51.145 | S5c | -4.58015 [-8.75912, -0.75758]; 0.00585 | -4.58015 [-8.75912, -0.75758]; 0.00585 |
| cell_bucket_3p5-6p5_all | 552 | 56.159/57.790 | S5a | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_3p5-6p5_all | 552 | 56.159/57.790 | S5b | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_3p5-6p5_all | 552 | 56.159/57.790 | S5c | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_7_all | 74 | 51.351/50.000 | S5a | +6.75676 [+1.40845, +12.98701]; 0.99425 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_7_all | 74 | 51.351/50.000 | S5b | +6.75676 [+0.00000, +13.92405]; 0.96905 | +1.35135 [+0.00000, +4.34783]; 0.63480 |
| cell_bucket_7_all | 74 | 51.351/50.000 | S5c | +1.35135 [-2.89855, +6.06061]; 0.60715 | -2.70270 [-7.14286, +0.00000]; 0.00000 |
| cell_bucket_7p5-10_all | 194 | 51.546/51.546 | S5a | -2.06186 [-4.73684, +0.00000]; 0.02345 | -2.06186 [-4.73684, +0.00000]; 0.02345 |
| cell_bucket_7p5-10_all | 194 | 51.546/51.546 | S5b | -2.57732 [-5.46448, +0.00000]; 0.01130 | -2.57732 [-5.46448, +0.00000]; 0.01130 |
| cell_bucket_7p5-10_all | 194 | 51.546/51.546 | S5c | -1.54639 [-3.38185, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |

Measured paired losses for the same groups (`cells.json`):

| Group | Arm | Brier improvement | Log-loss improvement |
|---|---|---|---|
| cell_bucket_0-3_all | S5a | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_0-3_all | S5b | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_0-3_all | S5c | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_10p5plus_all | S5a | +0.00043 [-0.00164, +0.00242]; 0.66360 | +0.00075 [-0.00352, +0.00484]; 0.64000 |
| cell_bucket_10p5plus_all | S5b | +0.00060 [-0.00327, +0.00433]; 0.62590 | +0.00093 [-0.00707, +0.00857]; 0.59525 |
| cell_bucket_10p5plus_all | S5c | -0.00058 [-0.00246, +0.00135]; 0.27480 | -0.00111 [-0.00495, +0.00287]; 0.28810 |
| cell_bucket_3p5-6p5_all | S5a | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_3p5-6p5_all | S5b | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_3p5-6p5_all | S5c | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| cell_bucket_7_all | S5a | -0.00025 [-0.00186, +0.00141]; 0.38510 | -0.00058 [-0.00390, +0.00279]; 0.36525 |
| cell_bucket_7_all | S5b | -0.00072 [-0.00424, +0.00286]; 0.34450 | -0.00170 [-0.00892, +0.00568]; 0.32160 |
| cell_bucket_7_all | S5c | +0.00010 [-0.00110, +0.00125]; 0.56530 | +0.00025 [-0.00218, +0.00260]; 0.58095 |
| cell_bucket_7p5-10_all | S5a | -0.00017 [-0.00117, +0.00078]; 0.37395 | -0.00030 [-0.00233, +0.00162]; 0.39035 |
| cell_bucket_7p5-10_all | S5b | -0.00040 [-0.00269, +0.00172]; 0.36655 | -0.00073 [-0.00538, +0.00355]; 0.38170 |
| cell_bucket_7p5-10_all | S5c | +0.00011 [-0.00068, +0.00094]; 0.60440 | +0.00019 [-0.00143, +0.00186]; 0.58650 |

### Per-season comparisons

Measured (`cells.json`): each cell is delta [95%], probability_positive against S3.

| Group | n | S3 standalone/card % | Arm | Standalone | Card |
|---|---:|---:|---|---|---|
| season_2020 | 220 | 52.273/55.455 | S5a | +0.00000 [-1.86047, +1.80180]; 0.40155 | -0.90909 [-2.32558, +0.00000]; 0.00000 |
| season_2020 | 220 | 52.273/55.455 | S5b | -0.45455 [-2.40385, +1.40187]; 0.24780 | -1.36364 [-2.88462, +0.00000]; 0.00000 |
| season_2020 | 220 | 52.273/55.455 | S5c | +0.00000 [+0.00000, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |
| season_2021 | 236 | 54.237/54.237 | S5a | +0.42373 [-0.86207, +1.74672]; 0.60990 | +0.42373 [-0.86207, +1.74672]; 0.60990 |
| season_2021 | 236 | 54.237/54.237 | S5b | +0.84746 [-0.82988, +2.54237]; 0.78205 | +0.42373 [-0.86207, +1.74672]; 0.60990 |
| season_2021 | 236 | 54.237/54.237 | S5c | -0.42373 [-1.29310, +0.00000]; 0.00000 | -0.42373 [-1.29310, +0.00000]; 0.00000 |
| season_2022 | 248 | 55.242/58.065 | S5a | +0.00000 [-1.68776, +1.55642]; 0.39780 | -0.80645 [-2.10970, +0.00000]; 0.00000 |
| season_2022 | 248 | 55.242/58.065 | S5b | +0.40323 [-0.85106, +1.63265]; 0.60755 | -0.40323 [-1.26582, +0.00000]; 0.00000 |
| season_2022 | 248 | 55.242/58.065 | S5c | -1.61290 [-3.21285, -0.39216]; 0.00000 | -1.61290 [-3.21285, -0.39216]; 0.00000 |
| season_2023 | 266 | 56.391/57.895 | S5a | +0.75188 [-0.74906, +2.23881]; 0.78090 | +0.00000 [-1.12782, +1.11524]; 0.34540 |
| season_2023 | 266 | 56.391/57.895 | S5b | +0.00000 [-2.26415, +1.89394]; 0.44165 | -0.37594 [-1.55039, +0.76336]; 0.17955 |
| season_2023 | 266 | 56.391/57.895 | S5c | -1.12782 [-2.28137, +0.00000]; 0.00000 | -0.75188 [-1.85185, +0.00000]; 0.00000 |
| season_2024 | 266 | 53.008/53.008 | S5a | -0.37594 [-1.15385, +0.00000]; 0.00000 | -0.37594 [-1.15385, +0.00000]; 0.00000 |
| season_2024 | 266 | 53.008/53.008 | S5b | +0.00000 [-1.13208, +1.10701]; 0.34540 | +0.00000 [-1.13208, +1.10701]; 0.34540 |
| season_2024 | 266 | 53.008/53.008 | S5c | +0.37594 [-1.14504, +1.93050]; 0.59385 | -0.37594 [-1.55039, +0.76046]; 0.18190 |
| season_2025 | 267 | 55.805/56.554 | S5a | -0.74906 [-1.89394, +0.00000]; 0.00000 | -0.74906 [-1.89394, +0.00000]; 0.00000 |
| season_2025 | 267 | 55.805/56.554 | S5b | -1.12360 [-2.29885, +0.00000]; 0.00000 | -1.12360 [-2.29885, +0.00000]; 0.00000 |
| season_2025 | 267 | 55.805/56.554 | S5c | -0.37453 [-1.14943, +0.00000]; 0.00000 | +0.00000 [+0.00000, +0.00000]; 0.00000 |

Measured paired losses for the same groups (`cells.json`):

| Group | Arm | Brier improvement | Log-loss improvement |
|---|---|---|---|
| season_2020 | S5a | -0.00015 [-0.00084, +0.00047]; 0.33930 | -0.00031 [-0.00170, +0.00095]; 0.33520 |
| season_2020 | S5b | -0.00043 [-0.00227, +0.00118]; 0.32220 | -0.00091 [-0.00464, +0.00241]; 0.31515 |
| season_2020 | S5c | +0.00007 [-0.00028, +0.00046]; 0.64070 | +0.00015 [-0.00057, +0.00093]; 0.64270 |
| season_2021 | S5a | +0.00024 [-0.00018, +0.00073]; 0.84965 | +0.00048 [-0.00038, +0.00149]; 0.84755 |
| season_2021 | S5b | +0.00043 [-0.00045, +0.00144]; 0.81235 | +0.00086 [-0.00095, +0.00292]; 0.80770 |
| season_2021 | S5c | -0.00021 [-0.00059, +0.00010]; 0.10430 | -0.00043 [-0.00121, +0.00021]; 0.10660 |
| season_2022 | S5a | +0.00012 [-0.00057, +0.00077]; 0.64065 | +0.00022 [-0.00117, +0.00154]; 0.63310 |
| season_2022 | S5b | +0.00018 [-0.00117, +0.00147]; 0.61410 | +0.00034 [-0.00242, +0.00294]; 0.60515 |
| season_2022 | S5c | -0.00013 [-0.00068, +0.00044]; 0.31755 | -0.00026 [-0.00137, +0.00090]; 0.32095 |
| season_2023 | S5a | -0.00001 [-0.00069, +0.00062]; 0.49595 | -0.00004 [-0.00144, +0.00124]; 0.48425 |
| season_2023 | S5b | -0.00003 [-0.00130, +0.00113]; 0.48700 | -0.00011 [-0.00274, +0.00227]; 0.47445 |
| season_2023 | S5c | -0.00000 [-0.00059, +0.00063]; 0.48880 | +0.00000 [-0.00119, +0.00131]; 0.49465 |
| season_2024 | S5a | -0.00024 [-0.00073, +0.00021]; 0.15945 | -0.00051 [-0.00153, +0.00042]; 0.15260 |
| season_2024 | S5b | -0.00046 [-0.00136, +0.00036]; 0.14800 | -0.00097 [-0.00284, +0.00072]; 0.13975 |
| season_2024 | S5c | +0.00021 [-0.00024, +0.00069]; 0.81105 | +0.00044 [-0.00048, +0.00143]; 0.81585 |
| season_2025 | S5a | +0.00007 [-0.00040, +0.00054]; 0.61195 | +0.00016 [-0.00081, +0.00113]; 0.61760 |
| season_2025 | S5b | +0.00009 [-0.00073, +0.00092]; 0.58610 | +0.00021 [-0.00149, +0.00193]; 0.59155 |
| season_2025 | S5c | -0.00013 [-0.00064, +0.00038]; 0.31580 | -0.00027 [-0.00133, +0.00076]; 0.31200 |

### Week 1

Measured (`week1.json`): linked forecast `F:\Repos\nfl_py3\artifacts\margin_predictions\2026-week-01-20260908T124514Z`, sidecar replay gap 9.99200722e-16. All three arms retain all 16 model and card sides. The forecast was read, not regenerated.

Measured fitted served offsets in points (`week1.json`); small buckets remain zero:

| Bucket | Prior games | S3 / 100 | S5a / 50 | S5b / 25 | S5c / 200 |
|---|---:|---:|---:|---:|---:|
| 0-3 | 504 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 3.5-6.5 | 468 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 7 | 65 | -0.170623 | -0.244807 | -0.312808 | -0.106237 |
| 7.5-10 | 160 | +0.784211 | +0.970928 | +1.102135 | +0.566375 |
| 10.5+ | 113 | +1.915663 | +2.503290 | +2.956784 | +1.303630 |

Read limitation (predeclaration above): this is a sensitivity on the mined archive that selected S2/S3, with correlated arms, not independent confirmation. Inferred: these measurements favour retaining the existing prior, without closing the broader home-side mechanism. All cells remain unresolved_below_power. Measured (`cells.json`): identical small-bucket comparisons have probability_positive 0 because draws must be strictly positive; these are ties.


## Registry and verification

Measured (CLI record stage, then exact-name lookup in `registry/weak_signals.json`): 144 rows recorded and verified, all unresolved_below_power; 3 arms x (overall + 6 seasons + 5 buckets) x 4 metrics. Exact names are also stamped in `artifacts/research/laneI/registry_names.json`.

- `mod18_home_side_location_v1_s5_s5a_cell_bucket_0-3_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_0-3_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_0-3_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_0-3_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_10p5plus_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_10p5plus_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_10p5plus_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_10p5plus_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_3p5-6p5_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_3p5-6p5_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_3p5-6p5_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_3p5-6p5_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7p5-10_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7p5-10_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7p5-10_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_cell_bucket_7p5-10_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_overall_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_overall_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_overall_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_overall_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5a_season_2020_brier_2020_2020`
- `mod18_home_side_location_v1_s5_s5a_season_2020_card_2020_2020`
- `mod18_home_side_location_v1_s5_s5a_season_2020_log_loss_2020_2020`
- `mod18_home_side_location_v1_s5_s5a_season_2020_standalone_2020_2020`
- `mod18_home_side_location_v1_s5_s5a_season_2021_brier_2021_2021`
- `mod18_home_side_location_v1_s5_s5a_season_2021_card_2021_2021`
- `mod18_home_side_location_v1_s5_s5a_season_2021_log_loss_2021_2021`
- `mod18_home_side_location_v1_s5_s5a_season_2021_standalone_2021_2021`
- `mod18_home_side_location_v1_s5_s5a_season_2022_brier_2022_2022`
- `mod18_home_side_location_v1_s5_s5a_season_2022_card_2022_2022`
- `mod18_home_side_location_v1_s5_s5a_season_2022_log_loss_2022_2022`
- `mod18_home_side_location_v1_s5_s5a_season_2022_standalone_2022_2022`
- `mod18_home_side_location_v1_s5_s5a_season_2023_brier_2023_2023`
- `mod18_home_side_location_v1_s5_s5a_season_2023_card_2023_2023`
- `mod18_home_side_location_v1_s5_s5a_season_2023_log_loss_2023_2023`
- `mod18_home_side_location_v1_s5_s5a_season_2023_standalone_2023_2023`
- `mod18_home_side_location_v1_s5_s5a_season_2024_brier_2024_2024`
- `mod18_home_side_location_v1_s5_s5a_season_2024_card_2024_2024`
- `mod18_home_side_location_v1_s5_s5a_season_2024_log_loss_2024_2024`
- `mod18_home_side_location_v1_s5_s5a_season_2024_standalone_2024_2024`
- `mod18_home_side_location_v1_s5_s5a_season_2025_brier_2025_2025`
- `mod18_home_side_location_v1_s5_s5a_season_2025_card_2025_2025`
- `mod18_home_side_location_v1_s5_s5a_season_2025_log_loss_2025_2025`
- `mod18_home_side_location_v1_s5_s5a_season_2025_standalone_2025_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_0-3_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_0-3_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_0-3_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_0-3_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_10p5plus_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_10p5plus_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_10p5plus_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_10p5plus_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_3p5-6p5_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_3p5-6p5_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_3p5-6p5_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_3p5-6p5_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7p5-10_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7p5-10_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7p5-10_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_cell_bucket_7p5-10_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_overall_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_overall_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_overall_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_overall_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5b_season_2020_brier_2020_2020`
- `mod18_home_side_location_v1_s5_s5b_season_2020_card_2020_2020`
- `mod18_home_side_location_v1_s5_s5b_season_2020_log_loss_2020_2020`
- `mod18_home_side_location_v1_s5_s5b_season_2020_standalone_2020_2020`
- `mod18_home_side_location_v1_s5_s5b_season_2021_brier_2021_2021`
- `mod18_home_side_location_v1_s5_s5b_season_2021_card_2021_2021`
- `mod18_home_side_location_v1_s5_s5b_season_2021_log_loss_2021_2021`
- `mod18_home_side_location_v1_s5_s5b_season_2021_standalone_2021_2021`
- `mod18_home_side_location_v1_s5_s5b_season_2022_brier_2022_2022`
- `mod18_home_side_location_v1_s5_s5b_season_2022_card_2022_2022`
- `mod18_home_side_location_v1_s5_s5b_season_2022_log_loss_2022_2022`
- `mod18_home_side_location_v1_s5_s5b_season_2022_standalone_2022_2022`
- `mod18_home_side_location_v1_s5_s5b_season_2023_brier_2023_2023`
- `mod18_home_side_location_v1_s5_s5b_season_2023_card_2023_2023`
- `mod18_home_side_location_v1_s5_s5b_season_2023_log_loss_2023_2023`
- `mod18_home_side_location_v1_s5_s5b_season_2023_standalone_2023_2023`
- `mod18_home_side_location_v1_s5_s5b_season_2024_brier_2024_2024`
- `mod18_home_side_location_v1_s5_s5b_season_2024_card_2024_2024`
- `mod18_home_side_location_v1_s5_s5b_season_2024_log_loss_2024_2024`
- `mod18_home_side_location_v1_s5_s5b_season_2024_standalone_2024_2024`
- `mod18_home_side_location_v1_s5_s5b_season_2025_brier_2025_2025`
- `mod18_home_side_location_v1_s5_s5b_season_2025_card_2025_2025`
- `mod18_home_side_location_v1_s5_s5b_season_2025_log_loss_2025_2025`
- `mod18_home_side_location_v1_s5_s5b_season_2025_standalone_2025_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_0-3_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_0-3_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_0-3_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_0-3_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_10p5plus_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_10p5plus_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_10p5plus_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_10p5plus_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_3p5-6p5_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_3p5-6p5_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_3p5-6p5_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_3p5-6p5_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7p5-10_all_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7p5-10_all_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7p5-10_all_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_cell_bucket_7p5-10_all_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_overall_brier_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_overall_card_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_overall_log_loss_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_overall_standalone_2020_2025`
- `mod18_home_side_location_v1_s5_s5c_season_2020_brier_2020_2020`
- `mod18_home_side_location_v1_s5_s5c_season_2020_card_2020_2020`
- `mod18_home_side_location_v1_s5_s5c_season_2020_log_loss_2020_2020`
- `mod18_home_side_location_v1_s5_s5c_season_2020_standalone_2020_2020`
- `mod18_home_side_location_v1_s5_s5c_season_2021_brier_2021_2021`
- `mod18_home_side_location_v1_s5_s5c_season_2021_card_2021_2021`
- `mod18_home_side_location_v1_s5_s5c_season_2021_log_loss_2021_2021`
- `mod18_home_side_location_v1_s5_s5c_season_2021_standalone_2021_2021`
- `mod18_home_side_location_v1_s5_s5c_season_2022_brier_2022_2022`
- `mod18_home_side_location_v1_s5_s5c_season_2022_card_2022_2022`
- `mod18_home_side_location_v1_s5_s5c_season_2022_log_loss_2022_2022`
- `mod18_home_side_location_v1_s5_s5c_season_2022_standalone_2022_2022`
- `mod18_home_side_location_v1_s5_s5c_season_2023_brier_2023_2023`
- `mod18_home_side_location_v1_s5_s5c_season_2023_card_2023_2023`
- `mod18_home_side_location_v1_s5_s5c_season_2023_log_loss_2023_2023`
- `mod18_home_side_location_v1_s5_s5c_season_2023_standalone_2023_2023`
- `mod18_home_side_location_v1_s5_s5c_season_2024_brier_2024_2024`
- `mod18_home_side_location_v1_s5_s5c_season_2024_card_2024_2024`
- `mod18_home_side_location_v1_s5_s5c_season_2024_log_loss_2024_2024`
- `mod18_home_side_location_v1_s5_s5c_season_2024_standalone_2024_2024`
- `mod18_home_side_location_v1_s5_s5c_season_2025_brier_2025_2025`
- `mod18_home_side_location_v1_s5_s5c_season_2025_card_2025_2025`
- `mod18_home_side_location_v1_s5_s5c_season_2025_log_loss_2025_2025`
- `mod18_home_side_location_v1_s5_s5c_season_2025_standalone_2025_2025`

Measured verification (`artifacts/research/laneI/verification.json`): 15 tests passed in 24.39 seconds; scoped Ruff format and lint passed. The tests include the experiment-registry provenance scan, production parity at prior 100, all frozen prior scalings, home-side pooling, absent observations, future/same-week/old-history exclusion, fail-closed replay, and the exact decision/Brier boundaries.

Measured execution: the following commands completed under the locked environment. `commands.json` retains all three successful overlay-composition argv and the registry parser/handler invocations. The first registry attempt hit Windows access denial at atomic replacement; the authorized elevated retry completed, and all expected rows were verified.

```powershell
.\.tools\uv.exe run --no-sync python scripts/home_side_prior_opener_eval.py --stage replay
.\.tools\uv.exe run --no-sync python scripts/home_side_prior_opener_eval.py --stage map
.\.tools\uv.exe run --no-sync python scripts/home_side_prior_opener_eval.py --stage score
.\.tools\uv.exe run --no-sync python scripts/home_side_prior_opener_eval.py --stage week1
.\.tools\uv.exe run --no-sync python scripts/home_side_prior_opener_eval.py --stage record
.\.tools\uv.exe run --no-sync ruff format --check scripts/home_side_prior_opener_eval.py tests/test_home_side_prior.py
.\.tools\uv.exe run --no-sync ruff check scripts/home_side_prior_opener_eval.py tests/test_home_side_prior.py
.\.tools\uv.exe run --no-sync pytest tests/test_home_side_prior.py tests/test_experiment_registry.py -n 2 --basetemp C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/fedb09af-a2ec-4d29-af01-540025768002/scratchpad/laneI -q
```

Read scope constraint: the owner's Lane I allowlist supersedes the older fleet read-only instruction and session-wide scheduler/dashboard/handoff actions. No scheduler job was edited or invoked, no forecast regenerated, no production source changed, and no commit or push made. Measured Git status: lane files plus `registry/weak_signals.json`; local research outputs are under `artifacts/research/laneI/`.

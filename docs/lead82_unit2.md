# LEAD-82 unit 2 — hourly Tuesday move

**Measured:** one declared replay; no tuning, serving changes, or registry writes.
The lane amendment was saved before outcome access. Historical opener is the frozen pool-line proxy.

## Decisive games first

**Measured:** candidate record when its OOS pick differs from each baseline; pushes excluded from conditional-cover scoring.
| Panel | Baseline | Candidate W-L | Accuracy, 95% interval |
| --- | --- | --- | --- |
| pooled | served | 13-23 | 36.111111 [20.000000, 53.851351] |
| pooled | model | 158-126 | 55.633803 [50.000000, 61.258278] |
| pooled | market | 214-172 | 55.440415 [50.626566, 60.152284] |
| 2023 | served | 6-11 | 35.294118 [13.636364, 66.666667] |
| 2023 | model | 52-41 | 55.913978 [46.314145, 64.761905] |
| 2023 | market | 76-66 | 53.521127 [46.753247, 60.000000] |
| 2024 | served | 3-4 | 42.857143 [0.000000, 83.333333] |
| 2024 | model | 54-45 | 54.545455 [44.329604, 64.815324] |
| 2024 | market | 76-56 | 57.575758 [48.120301, 66.187050] |
| 2025 | served | 4-8 | 33.333333 [0.000000, 61.538462] |
| 2025 | model | 52-40 | 56.521739 [47.191011, 66.666667] |
| 2025 | market | 62-50 | 55.357143 [47.200000, 64.406780] |

## Protocol and integrity

**Read:** the lane fixes the signed median Tuesday-noon-to-pre-Wednesday leader-book move,
four metrics and 261-look ceiling. The owner's LOSO amendment supersedes the row's single 2025 outer season.
The audit (docs/move_feature_audit.md:3-21) attributes 164 late-move differences to later captures.
Both arms retain saved hourly late move and availability; only Tuesday is newly extracted on the hourly grid.
The predeclared terminal must occur after noon; a stale pre-noon quote is not treated as an observed afternoon block.

**Measured:** 1503 nonpush source rows across 2020-2025; 799 scored rows;
season counts {'2023': 266, '2024': 266, '2025': 267}; 1537 opener rows retained
in the feature inventory including pushes; 2348 same-book Tuesday pairs.
Zero boundary-clock violations. Served fold probability maximum reproduction error 0;
saved hourly move maximum error 0.
**Measured:** source-only Tuesday comparison with unit 1: {'compared_games': 799, 'changed_game_values': 1, 'max_abs_change': 0.25}.
The post-noon gate accounts for the removed stale boundary pair; the one changed game median is disclosed,
and no definition was revised after outcomes.

Each base four-term ridge fit uses all other 2020-2025 seasons, then freezes its coefficients.
One Tuesday coefficient uses the other two quote-covered seasons with base logit as offset.
Tuesday scales by training SD without centering; ridge 0.001; no new intercept, slope, cut point, or flip rule.
All-data IS is explicitly optimistic. One combined probability selects at 0.5; ties choose home.

**Measured:** 10,000 paired bootstrap draws over 54 season-week blocks,
stratified by season, seed 20260929. Percentile 95% intervals; probability_positive gives exact-zero draws half credit.
Positive improvement means candidate-minus-baseline accuracy or baseline-minus-candidate loss.
Primary Brier; conditional binary RPS equals Brier exactly, not full-margin RPS. No push forecast is fitted.
Fair opener market is 0.5; model-only is saved discrete conditional probability; identical opener grades/population.
136 metric panels + 48 coefficient entries + 20 reliability cells + 12 decisive records = 216 looks;
45 of 261 unspent. No extra specifications. Baseline contrasts are not separate registry candidates.

## Out-of-season scores

**Measured:** estimate [95% week-block interval]; accuracy is percentage points.
| Panel | Arm | accuracy_points | log_loss | brier | rps |
| --- | --- | --- | --- | --- | --- |
| pooled | candidate | 56.070088 [52.948557, 59.142497] | 0.681243 [0.669988, 0.692224] | 0.244102 [0.238712, 0.249359] | 0.244102 [0.238712, 0.249359] |
| pooled | served | 57.321652 [54.271261, 60.275689] | 0.681801 [0.670811, 0.692436] | 0.244345 [0.239122, 0.249428] | 0.244345 [0.239122, 0.249428] |
| pooled | model | 52.065081 [48.877805, 55.417488] | 0.697288 [0.689386, 0.705216] | 0.251974 [0.248091, 0.255854] | 0.251974 [0.248091, 0.255854] |
| pooled | market | 50.813517 [47.375000, 54.090397] | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] |
| 2023 | candidate | 54.511278 [49.806950, 59.245283] | 0.679103 [0.659301, 0.698075] | 0.242838 [0.233407, 0.251864] | 0.242838 [0.233407, 0.251864] |
| 2023 | served | 56.390977 [51.908397, 60.755633] | 0.677980 [0.659744, 0.695643] | 0.242324 [0.233595, 0.250874] | 0.242324 [0.233595, 0.250874] |
| 2023 | model | 50.375940 [46.816479, 54.296875] | 0.701840 [0.690604, 0.712421] | 0.254129 [0.248664, 0.259227] | 0.254129 [0.248664, 0.259227] |
| 2023 | market | 50.751880 [45.160989, 55.938697] | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] |
| 2024 | candidate | 57.894737 [52.290076, 63.235294] | 0.681927 [0.663946, 0.700230] | 0.244298 [0.235773, 0.252975] | 0.244298 [0.235773, 0.252975] |
| 2024 | served | 58.270677 [52.873017, 63.235851] | 0.683667 [0.667027, 0.700879] | 0.245124 [0.237230, 0.253261] | 0.245124 [0.237230, 0.253261] |
| 2024 | model | 54.511278 [47.859503, 61.090909] | 0.694413 [0.680630, 0.708462] | 0.250570 [0.243764, 0.257500] | 0.250570 [0.243764, 0.257500] |
| 2024 | market | 50.375940 [43.750000, 57.088158] | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] |
| 2025 | candidate | 55.805243 [49.811321, 61.623616] | 0.682694 [0.661925, 0.702504] | 0.245167 [0.235331, 0.254674] | 0.245167 [0.235331, 0.254674] |
| 2025 | served | 57.303371 [51.515152, 62.867926] | 0.683748 [0.662001, 0.703931] | 0.245581 [0.235377, 0.255191] | 0.245581 [0.235377, 0.255191] |
| 2025 | model | 51.310861 [45.185185, 57.142857] | 0.695617 [0.681098, 0.711345] | 0.251226 [0.244076, 0.258974] | 0.251226 [0.244076, 0.258974] |
| 2025 | market | 51.310861 [45.925926, 56.390977] | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] |

## Paired improvements

**Measured:** positive favors Tuesday; paired comparisons are not independent votes.
| Panel | Baseline | Metric | Improvement, 95% interval | probability_positive |
| --- | --- | --- | --- | --- |
| pooled | served | accuracy_points | -1.251564 [-3.101737, 0.370370] | 0.0606 |
| pooled | served | log_loss | 0.000557 [-0.002389, 0.003383] | 0.6567 |
| pooled | served | brier | 0.000242 [-0.001137, 0.001572] | 0.6467 |
| pooled | served | rps | 0.000242 [-0.001137, 0.001572] | 0.6467 |
| pooled | model | accuracy_points | 4.005006 [0.000000, 7.990139] | 0.9744 |
| pooled | model | log_loss | 0.016044 [0.003474, 0.028910] | 0.9938 |
| pooled | model | brier | 0.007872 [0.001841, 0.014010] | 0.9949 |
| pooled | model | rps | 0.007872 [0.001841, 0.014010] | 0.9949 |
| pooled | market | accuracy_points | 5.256571 [0.622665, 9.885967] | 0.9872 |
| pooled | market | log_loss | 0.011904 [0.000923, 0.023159] | 0.9828 |
| pooled | market | brier | 0.005898 [0.000641, 0.011288] | 0.9869 |
| pooled | market | rps | 0.005898 [0.000641, 0.011288] | 0.9869 |
| 2023 | served | accuracy_points | -1.879699 [-6.488550, 1.503759] | 0.1831 |
| 2023 | served | log_loss | -0.001123 [-0.006822, 0.004386] | 0.3427 |
| 2023 | served | brier | -0.000513 [-0.003135, 0.002050] | 0.3415 |
| 2023 | served | rps | -0.000513 [-0.003135, 0.002050] | 0.3415 |
| 2023 | model | accuracy_points | 4.135338 [-2.661597, 10.112360] | 0.8914 |
| 2023 | model | log_loss | 0.022737 [0.005267, 0.041804] | 0.9968 |
| 2023 | model | brier | 0.011292 [0.002874, 0.020441] | 0.9970 |
| 2023 | model | rps | 0.011292 [0.002874, 0.020441] | 0.9970 |
| 2023 | market | accuracy_points | 3.759398 [-3.409091, 11.029412] | 0.8400 |
| 2023 | market | log_loss | 0.014044 [-0.004928, 0.033846] | 0.9264 |
| 2023 | market | brier | 0.007162 [-0.001864, 0.016593] | 0.9413 |
| 2023 | market | rps | 0.007162 [-0.001864, 0.016593] | 0.9413 |
| 2024 | served | accuracy_points | -0.375940 [-2.298851, 1.515152] | 0.3581 |
| 2024 | served | log_loss | 0.001740 [-0.000752, 0.004523] | 0.9080 |
| 2024 | served | brier | 0.000825 [-0.000380, 0.002168] | 0.9041 |
| 2024 | served | rps | 0.000825 [-0.000380, 0.002168] | 0.9041 |
| 2024 | model | accuracy_points | 3.383459 [-4.395604, 10.861423] | 0.8153 |
| 2024 | model | log_loss | 0.012486 [-0.010549, 0.033621] | 0.8658 |
| 2024 | model | brier | 0.006272 [-0.004667, 0.016320] | 0.8775 |
| 2024 | model | rps | 0.006272 [-0.004667, 0.016320] | 0.8775 |
| 2024 | market | accuracy_points | 7.518797 [-1.858909, 17.110266] | 0.9415 |
| 2024 | market | log_loss | 0.011220 [-0.007083, 0.029201] | 0.8813 |
| 2024 | market | brier | 0.005702 [-0.002975, 0.014227] | 0.8972 |
| 2024 | market | rps | 0.005702 [-0.002975, 0.014227] | 0.8972 |
| 2025 | served | accuracy_points | -1.498127 [-3.831418, 1.111111] | 0.1174 |
| 2025 | served | log_loss | 0.001053 [-0.005141, 0.006934] | 0.6508 |
| 2025 | served | brier | 0.000414 [-0.002528, 0.003159] | 0.6286 |
| 2025 | served | rps | 0.000414 [-0.002528, 0.003159] | 0.6286 |
| 2025 | model | accuracy_points | 4.494382 [-1.886792, 11.808118] | 0.9022 |
| 2025 | model | log_loss | 0.012922 [-0.008800, 0.040645] | 0.8411 |
| 2025 | model | brier | 0.006059 [-0.004356, 0.019296] | 0.8359 |
| 2025 | model | rps | 0.006059 [-0.004356, 0.019296] | 0.8359 |
| 2025 | market | accuracy_points | 4.494382 [-2.573529, 11.698113] | 0.8915 |
| 2025 | market | log_loss | 0.010453 [-0.009357, 0.031222] | 0.8344 |
| 2025 | market | brier | 0.004833 [-0.004674, 0.014669] | 0.8244 |
| 2025 | market | rps | 0.004833 [-0.004674, 0.014669] | 0.8244 |

## Optimistic IS, OOS, and gap

**Measured:** positive optimism gap means better IS accuracy or lower IS loss than OOS.
| Arm | Metric | IS, 95% interval | OOS, 95% interval | Optimism gap, 95% interval |
| --- | --- | --- | --- | --- |
| candidate | accuracy_points | 56.195244 [53.112804, 59.245283] | 56.070088 [52.948557, 59.142497] | 0.125156 [-0.997506, 1.250000] |
| candidate | log_loss | 0.679704 [0.668544, 0.690470] | 0.681243 [0.669988, 0.692224] | 0.001539 [0.000335, 0.002757] |
| candidate | brier | 0.243408 [0.238075, 0.248546] | 0.244102 [0.238712, 0.249359] | 0.000694 [0.000142, 0.001251] |
| candidate | rps | 0.243408 [0.238075, 0.248546] | 0.244102 [0.238712, 0.249359] | 0.000694 [0.000142, 0.001251] |
| served | accuracy_points | 57.321652 [54.336535, 60.200250] | 57.321652 [54.271261, 60.275689] | 0.000000 [-0.974481, 0.877193] |
| served | log_loss | 0.680739 [0.669883, 0.691120] | 0.681801 [0.670811, 0.692436] | 0.001061 [0.000003, 0.002125] |
| served | brier | 0.243871 [0.238704, 0.248849] | 0.244345 [0.239122, 0.249428] | 0.000474 [-0.000014, 0.000956] |
| served | rps | 0.243871 [0.238704, 0.248849] | 0.244345 [0.239122, 0.249428] | 0.000474 [-0.000014, 0.000956] |
| Candidate vs served improvement | IS, 95% interval | OOS, 95% interval | IS-OOS gap, 95% interval |
| --- | --- | --- | --- |
| accuracy_points | -1.126408 [-2.654867, 0.252215] | -1.251564 [-3.101737, 0.370370] | 0.125156 [-0.503145, 0.755668] |
| log_loss | 0.001036 [-0.001705, 0.003733] | 0.000557 [-0.002389, 0.003383] | 0.000478 [-0.000218, 0.001198] |
| brier | 0.000463 [-0.000838, 0.001741] | 0.000242 [-0.001137, 0.001572] | 0.000220 [-0.000103, 0.000559] |
| rps | 0.000463 [-0.000838, 0.001741] | 0.000242 [-0.001137, 0.001572] | 0.000220 [-0.000103, 0.000559] |

## Natural coefficients

**Measured:** base coefficients are identical between arms; both shown to expose the constraint.
| Holdout | Arm | Base train n | Tuesday train n | Intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | tuesday_move |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | served | 1237 | 533 | -0.038064 | 0.341480 | 0.246114 | 0.230272 | 0.002286 | 0.000000 |
| 2023 | candidate | 1237 | 533 | -0.038064 | 0.341480 | 0.246114 | 0.230272 | 0.002286 | 0.276968 |
| 2024 | served | 1237 | 533 | -0.056178 | 0.256116 | 0.262262 | 0.246066 | 0.039146 | 0.000000 |
| 2024 | candidate | 1237 | 533 | -0.056178 | 0.256116 | 0.262262 | 0.246066 | 0.039146 | 0.127540 |
| 2025 | served | 1236 | 532 | -0.054099 | 0.292328 | 0.295997 | 0.188739 | 0.047462 | 0.000000 |
| 2025 | candidate | 1236 | 532 | -0.054099 | 0.292328 | 0.295997 | 0.188739 | 0.047462 | 0.217182 |
| IS | served | 1503 | 799 | -0.052065 | 0.276510 | 0.259974 | 0.221778 | 0.034174 | 0.000000 |
| IS | candidate | 1503 | 799 | -0.052065 | 0.276510 | 0.259974 | 0.221778 | 0.034174 | 0.202948 |

## Reliability

**Measured:** OOS home-side probabilities and observed cover rates; all five equal-width bins retained.
| Arm | Bin | Games | Mean probability | Observed rate |
| --- | --- | --- | --- | --- |
| candidate | 0.0-0.2 | 4 | 0.186335 | 0.250000 |
| candidate | 0.2-0.4 | 76 | 0.349648 | 0.421053 |
| candidate | 0.4-0.6 | 606 | 0.500644 | 0.493399 |
| candidate | 0.6-0.8 | 113 | 0.651899 | 0.654867 |
| candidate | 0.8-1.0 | 0 | empty | empty |
| served | 0.0-0.2 | 1 | 0.199419 | 0.000000 |
| served | 0.2-0.4 | 71 | 0.347070 | 0.408451 |
| served | 0.4-0.6 | 621 | 0.501722 | 0.495974 |
| served | 0.6-0.8 | 106 | 0.650456 | 0.650943 |
| served | 0.8-1.0 | 0 | empty | empty |
| model | 0.0-0.2 | 0 | empty | empty |
| model | 0.2-0.4 | 49 | 0.374657 | 0.530612 |
| model | 0.4-0.6 | 692 | 0.501198 | 0.505780 |
| model | 0.6-0.8 | 58 | 0.623338 | 0.517241 |
| model | 0.8-1.0 | 0 | empty | empty |
| market | 0.0-0.2 | 0 | empty | empty |
| market | 0.2-0.4 | 0 | empty | empty |
| market | 0.4-0.6 | 799 | 0.500000 | 0.508135 |
| market | 0.6-0.8 | 0 | empty | empty |
| market | 0.8-1.0 | 0 | empty | empty |

## Interpretation and limits

**Read:** upstream opener producer fits completed games before the target week's earliest game
(src/nfl_ats/clv.py:2184-2200) and passes that cutoff and target-game exclusions to the discrete reader.
The cached artifact lacks row-level upstream training ledgers; source inspection is not a fresh reconstruction.
**Read:** frozen-fit limitation: Features were selected using these seasons; this is not an untouched outer test. Weekly ranking uncertainty is assessed separately from average calibration.
**Inferred:** retrospective LOSO includes later seasons in earlier-fold training and reuses selected upstream features.
This is not chronological deployment validation or an untouched outer test. Bootstrap intervals condition on fitted
predictions and three fixed seasons; they exclude refitting uncertainty and cannot establish long-run season stability.
**Inferred:** retain unresolved_below_power pending the orchestrator's serial candidate-versus-served recording.
**Measured:** Tuesday coefficients are positive in all three folds:
2023: 0.276968, 2024: 0.127540, 2025: 0.217182.
Brier improves in two folds and worsens in one; changed-pick accuracy trails served in all three folds.
**Inferred:** the small proper-score gain and weaker changed-pick record warrant further research;
they do not establish a serving improvement or refute the Tuesday-news mechanism.
No split-half refutation or powered positive control was tested. AGENTS.md:65-84 governs closure;
AGENTS.md:115-120 separates closure from serving. Historical forced-pick rate does not establish a profitable edge.

## Reproduction

**Measured:** .tools/uv.exe run --no-sync python scripts/lead82_unit2.py completed one replay.
Use writable UV_CACHE_DIR, UV_OFFLINE=1 and at most two numerical threads. No tests added.
Rows, boundaries, feature inventory, coefficients, intervals and source hashes: tests/scratch/codex/lead82_unit2/.
Record commands are in the lane for the orchestrator; they were not executed.

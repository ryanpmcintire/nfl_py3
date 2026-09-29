# LEAD-69 CFB horizon replication

**Measured:** first-observed-opener proxy study completed. Exact market-opening-time replication remains unavailable: the stored CFB opener table has no opener timestamp.

## Decisive games before the effect

| Fit | Probability | Correct | Incorrect | Decisive games |
| --- | --- | --- | --- | --- |
| is | opener | 1201 | 1177 | 2378 |
| is | base | 1292 | 1086 | 2378 |
| is | horizon | 1293 | 1085 | 2378 |
| oos | opener | 1201 | 1177 | 2378 |
| oos | base | 1267 | 1111 | 2378 |
| oos | horizon | 1265 | 1113 | 2378 |

**Measured:** LOSO horizon-minus-base opener-graded accuracy: -0.084104 [-0.360848, +0.205362] percentage points; probability_positive 0.267000.

**Inferred:** provisional unresolved_below_power, pending the orchestrator's serial registry entry. The proxy study supplies no consistent NFL-sign replication and no demonstrated improvement; it does not refute the mechanism. No admissible closing ground has been established. AGENTS.md research rules distinguish signal closure from serving; this retrospective study changes no served pick. No positive control or split-half closure was attempted.

## Protocol and source limits

Declared in docs/lanes/lead69-cfb-replication.md before CFB outcome access. Population: completed 2023-2025 regular-season games with at least one FBS team, including neutral sites, a valid paired opener quote and a later same-book quote. The frozen grade is the historical first-observed opener proxy, not a 2026 pool capture.

**Read:** scripts/lead69_unit1.py:212 and :227 implement move x log(hours) and training-only standardization followed by ridge logistic regression. This study uses that form and FIT_RIDGE=0.001, with no tuning. Base = intercept + opener-implied home-cover logit + market move toward home; challenger adds move x log(hours from observed opener to conservative kickoff). One fitted probability chooses the side at 0.5.

**Measured:** each opener is an actual home spread at the earliest paired historical snapshot (lower observed median). Its implied probability is the median of home/away de-vigged prices only at that exact line. Movement is the median same-book opening-home-line minus latest-home-line among opener books with a later valid pair. Positive movement means the market moved toward home. This avoids inventing a smooth spread-to-probability mapping.

**Measured:** features are built before scores are loaded. Provider clocks must be at or before capture; captures and snapshots must precede the earlier of schedule kickoff and quoted kickoff. Schedule join uses normalized team names and kickoff separation at most 36 hours; ambiguous matches fail closed. Prices must have valid two-sided opposite spreads. Pushes are excluded from binary fitting and scoring.

**Read:** the CFB line loader in scripts/opener_error_transfer_eval.py:89 uses the stored opening_lines field but supplies no opening timestamp. Schedule source: data/cfb/schedules/raw/20260816T162105Z; quote source: data/market/raw/*-ncaaf/quotes.parquet and adjacent historical-response manifests. Retrospective ingestion does not certify real-time availability in an archive that existed then. First observation may be days after actual opening, and some last quotes precede kickoff by days; those limits prevent a claim of exact NFL timing replication.

One declared CFB look, zero NFL looks: one continuous interaction against a fixed base, three predeclared metrics. The three LOSO folds and fixed reliability bins are descriptive diagnostics, not searched alternatives. Each held season is excluded from coefficient and standardization fitting. In-sample fits use all seasons and are diagnostic only. LOSO can train on later seasons, so this is season-held-out replication, not a forward deployment simulation.

Uncertainty: 4,000 paired bootstrap resamples of (season, week) blocks, sampled within season, seed 6902026, 95% percentile intervals. These resample fixed held-out predictions and do not refit models. probability_positive gives half credit to exact ties. Positive improvements always favor the horizon term. Only three seasons limit season-level inference; no cross-league pooling was performed.

## Population audit

| Quantity | Measured value |
| --- | --- |
| opener_basis | "first_observed_paired_quote_proxy" |
| quote_files | 324 |
| fbs_regular_completed_by_season | {"2023": 868, "2024": 873, "2025": 888} |
| spread_quote_rows | 281362 |
| quote_events | 2817 |
| ambiguous_multi_event_games_excluded | 30 |
| matched_games_by_season | {"2023": 801, "2024": 853, "2025": 873} |
| unsafe_quote_rows_removed | 13736 |
| games_without_later_same_book_pair | 124 |
| eligible_before_outcomes_by_season | {"2023": 690, "2024": 846, "2025": 867} |
| hours_min_median_max | [4.072777777777778, 103.0725, 7490.739444444444] |
| quote_to_kickoff_hours_min_median_max | [0.05527777777777778, 1.0727777777777778, 5032.072777777777] |
| neutral_site_games | 76 |
| fbs_vs_fbs_games | 2182 |
| pushes_by_season | {"2023": 9, "2024": 8, "2025": 8} |
| decisive_by_season | {"2023": 681, "2024": 838, "2025": 859} |

## Accuracy, probability quality, and fitting gap

| Probability | Metric | In sample [95% CI] | LOSO [95% CI] | IS - OOS |
| --- | --- | --- | --- | --- |
| opener | accuracy_points | +50.504626 [+48.457113, +52.547858] | +50.504626 [+48.457113, +52.547858] | +0.000000 |
| opener | log_loss | +0.693234 [+0.692758, +0.693694] | +0.693234 [+0.692758, +0.693694] | +0.000000 |
| opener | brier | +0.250043 [+0.249805, +0.250273] | +0.250043 [+0.249805, +0.250273] | +0.000000 |
| base | accuracy_points | +54.331371 [+52.143688, +56.438440] | +53.280067 [+51.454755, +55.068394] | +1.051304 |
| base | log_loss | +0.685762 [+0.680869, +0.690786] | +0.686302 [+0.681746, +0.691134] | -0.000540 |
| base | brier | +0.246491 [+0.244141, +0.248902] | +0.246750 [+0.244537, +0.249066] | -0.000259 |
| horizon | accuracy_points | +54.373423 [+52.225874, +56.464160] | +53.195963 [+51.342815, +55.000164] | +1.177460 |
| horizon | log_loss | +0.685753 [+0.680887, +0.690747] | +0.686499 [+0.681857, +0.691336] | -0.000746 |
| horizon | brier | +0.246494 [+0.244149, +0.248909] | +0.246854 [+0.244634, +0.249162] | -0.000359 |

**Measured:** paired improvements below use identical games and week draws. Log-loss and Brier improvements are base minus challenger; accuracy is challenger minus base in percentage points.

| Fit | Improvement | Estimate [95% CI] | probability_positive |
| --- | --- | --- | --- |
| is | accuracy_points | +0.042052 [-0.088580, +0.181744] | 0.706750 |
| is | log_loss | +0.000009 [-0.000144, +0.000160] | 0.530500 |
| is | brier | -0.000004 [-0.000069, +0.000062] | 0.439750 |
| oos | accuracy_points | -0.084104 [-0.360848, +0.205362] | 0.267000 |
| oos | log_loss | -0.000197 [-0.000619, +0.000229] | 0.183500 |
| oos | brier | -0.000104 [-0.000281, +0.000072] | 0.126000 |

## Season stability

| Held season | N | Opener accuracy % | Base accuracy % | Horizon accuracy % | Accuracy gain pp | Log-loss gain | Brier gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 681 | 51.101 | 51.836 | 51.836 | +0.000 | +0.000014 | +0.000004 |
| 2024 | 838 | 49.761 | 54.773 | 54.535 | -0.239 | -0.000450 | -0.000252 |
| 2025 | 859 | 50.757 | 52.969 | 52.969 | +0.000 | -0.000116 | -0.000046 |

## Natural-unit coefficients

**Measured:** all coefficients below are on the original feature scale. NFL comparison values are read from docs/lead69_results.md:173-175; negative is the NFL direction in all three folds.

| Held season | Fit | Training N | Intercept | Opener logit | Move | Move x log(hours) | NFL horizon | Same sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in_sample | base | 2378 | +0.037129146 | -1.008650528 | +0.101600425 | +0.000000000 | - | - |
| in_sample | horizon | 2378 | +0.037436939 | -0.993209611 | +0.085709935 | +0.002732479 | - | - |
| 2023 | base | 1697 | +0.060722950 | -1.369114660 | +0.104697520 | +0.000000000 | -0.434823142 | - |
| 2024 | base | 1540 | +0.049620481 | -1.034284058 | +0.096296357 | +0.000000000 | -0.029133083 | - |
| 2025 | base | 1519 | -0.001883049 | -0.324053884 | +0.101650977 | +0.000000000 | -0.653789708 | - |
| 2023 | horizon | 1697 | +0.060885437 | -1.363111545 | +0.099343607 | +0.000900639 | -0.434823142 | no |
| 2024 | horizon | 1540 | +0.051271783 | -0.984511196 | +0.032552283 | +0.011168723 | -0.029133083 | no |
| 2025 | horizon | 1519 | -0.001912533 | -0.343315859 | +0.121787137 | -0.003508474 | -0.653789708 | yes |

**Measured:** horizon sign agrees with the corresponding NFL fold in 1/3 held seasons. Sign agreement is mechanism context; it is not three independent looks or evidence of profitable betting.

## Reliability on held-out games

**Measured:** fixed 0.1-wide bins describe home-cover calibration. Probability scores above compare both fitted models with the raw opener-implied market baseline. Sparse bins are not selected findings.

| Probability | Bin | N | Mean predicted home cover | Observed home cover |
| --- | --- | --- | --- | --- |
| opener | 0.4-0.5 | 344 | 0.492270 | 0.508721 |
| opener | 0.5-0.6 | 2034 | 0.501273 | 0.507375 |
| base | 0.1-0.2 | 6 | 0.162163 | 0.000000 |
| base | 0.2-0.3 | 15 | 0.263298 | 0.266667 |
| base | 0.3-0.4 | 63 | 0.363363 | 0.365079 |
| base | 0.4-0.5 | 888 | 0.473895 | 0.483108 |
| base | 0.5-0.6 | 1301 | 0.529344 | 0.526518 |
| base | 0.6-0.7 | 93 | 0.632459 | 0.612903 |
| base | 0.7-0.8 | 9 | 0.743511 | 0.666667 |
| base | 0.8-0.9 | 3 | 0.838065 | 1.000000 |
| horizon | 0.0-0.1 | 1 | 0.098553 | 0.000000 |
| horizon | 0.1-0.2 | 5 | 0.150153 | 0.000000 |
| horizon | 0.2-0.3 | 14 | 0.251659 | 0.214286 |
| horizon | 0.3-0.4 | 64 | 0.362017 | 0.343750 |
| horizon | 0.4-0.5 | 884 | 0.474879 | 0.486425 |
| horizon | 0.5-0.6 | 1310 | 0.529237 | 0.525954 |
| horizon | 0.6-0.7 | 85 | 0.631073 | 0.600000 |
| horizon | 0.7-0.8 | 12 | 0.745810 | 0.750000 |
| horizon | 0.8-0.9 | 2 | 0.814990 | 1.000000 |
| horizon | 0.9-1.0 | 1 | 0.928899 | 1.000000 |

## Reproduction and artifacts

Command: `.tools/uv.exe run --no-sync --no-cache python scripts/lead69_cfb_replication.py` (one scoring run; cache disabled because the shared user cache is outside the writable workspace).

Prediction rows and full summary: `tests/scratch/codex/lead69_cfb_replication/predictions.parquet` and `summary.json`. No prediction-row dump is written under docs/. BLAS and Arrow are limited to two threads. The script never writes the registry, publishes, or changes the served card. Exact bash-compatible registry commands are handed to the orchestrator in the lane.

## Review and preserved predeclaration

**Measured, source review:** 164 decisive games have observed-opener horizons above 14 days; one game's last quote is more than seven days old (maximum 5,032.07 hours). Maximum opener horizon is 7,490.74 hours. No freshness cutoff was added after outcomes were seen. Thirty games with multiple provider event IDs were excluded before scoring; the first attempt stopped at that join before loading outcomes, followed by one completed scoring run. Prediction-artifact review independently checked unique games, timestamp guards, non-push targets, finite probabilities, and the 1,267/2,378 versus 1,265/2,378 decisive records.

Protocol declared before outcome inspection. Population: FBS regular-season games in 2023–2025 with a historical opener and a pre-kickoff quote. Frozen pool-line proxy: historical opener. Horizon: hours from opener timestamp to kickoff. Target: home covers the opener, excluding pushes from binary fitting and accuracy. Base: opener-implied probability plus market move toward home. Challenger: the same fitted form as LEAD-69, adding its market-move × horizon interaction. All fitted parameters and transformations use training seasons only; leave one season out (LOSO). Report in-sample and out-of-sample metrics and gap, fold coefficients and sign agreement with NFL, log loss, Brier, opener-graded accuracy, season stability, paired week-blocked bootstrap intervals and probability_positive. One declared CFB look; no NFL look. A zero-crossing interval cannot close the signal. One fitted probability selects the side; no standalone flip.

Pre-outcome implementation declaration: include regular-season completed games with at least one FBS team (neutral sites included). At the earliest valid event snapshot, opener = an actual quoted median home line (lower middle if tied), probability = median two-sided de-vigged home-cover price among books quoting that exact line. Move toward home = median same-book (opening home line − latest home line), using opener books with a later pre-kickoff pair. Both snapshot/provider clocks precede the conservative kickoff. Fit intercept + opener logit + move; add move × log(opener-to-kickoff hours), with the NFL ridge and training-only standardization. Binary pushes excluded; one 2023–2025 LOSO fit family, no tuning. Use 4,000 paired season-stratified week-block bootstraps (seed 6902026), 95% percentile intervals and half-credit ties for probability_positive. Fixed probability bins report calibration descriptively; fold/metric summaries are diagnostics of the single declared look.

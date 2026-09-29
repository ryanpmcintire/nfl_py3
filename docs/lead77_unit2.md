# LEAD-77 unit 2: quote-freshness reliability

**Conditional assumption:** historical provider `observed_at_utc` is availability time, as authorized in the provenance decision. Historical opener is the frozen pool-line proxy; no pre-2026 Splash captures are used.

**Measured, decisive games first:** freshness won 20-19 on 39 side disagreements with the four-term base.
**Measured:** decisive accuracy 51.282051 [36.842105, 66.666667]%; probability_positive versus 50% = 0.572050.

**Measured:** 533 held-out non-push opener games; 36 season-week blocks; chronological 2024 and 2025 folds. 2023 supplies training only. No forecast/card changed.

## Declaration saved before outcomes

Historical standard: historical OPENER is the frozen pool-line proxy, never pre-2026
Splash captures. Use the quoted subset of the standard 2020-2025 regular-season
non-push population in `artifacts/pick_probability/20260929T192747Z/per_game.parquet`;
quotes cover 2023-2025. The served four-term specification is the paired base.
**Assumption:** provider `observed_at_utc` counts as historical availability, per
`docs/lanes/quote-provenance.md` Decision; every result is conditional on that assumption.
Use the validator's historical as-of/clock rules, not its strict availability flag.
NFL cutoff is min(kickoff, Sunday 12:45 ET); CFB cutoff is kickoff. All features and
response labels must precede their game's cutoff. No final score enters quote fitting.
Keep the latest capture per league/season/six-hour UTC bucket; retain seven pre-cutoff
days. Pair opposite HOME/AWAY spreads with valid prices and known update clocks.
At each event/snapshot/line require at least three books. Define q as no-vig home
probability at that exact line, fresh q as median among most-recently updated books,
d = fresh q - book q, a = log1p(age hours). Response terms are d and a*d, with no
intercept; target is next same-line consensus minus book q. The first later snapshot
within 48 hours with >=3 books all updated after the source capture supplies that
consensus. Never interpolate probabilities across spread lines or smooth key numbers.
Fit NFL-only ridge response and predeclared partially pooled NFL+CFB response; use
shared slopes plus penalized NFL deviations (penalties .001 and 1). Scale on training
rows; give each game equal total weight. Pool only identical probability units.
The pooled arm alone supplies the ATS term: latest eligible snapshot, modal spread
(ties: smallest line), mean predicted next q minus current median q. Missing response
labels never determine ATS eligibility; require only an eligible pre-cutoff feature.
Scheduled outer seasons 2023/2024/2025; chronology-purged LOSO trains on earlier quote
seasons only, excluding held/later NFL and CFB seasons. Thus 2023 is unavailable.
Fit/calibrate base four terms and base plus ONE correction jointly on earlier eligible
NFL games, ridge .001 with train-only standardization. No tuning/selection grid;
calibration uses only outer training rows. Frozen upstream model remains conditional.
One fitted probability chooses the side; no standalone flip. Simple baselines: frozen
discrete model and neutral market .5, on identical rows. No new Elo reconstruction.
Report decisive record first; opener accuracy, log loss, Brier, response MAE; IS/OOS
and gaps, every fold's coefficients and five train-quantile reliability bands per arm.
Paired season-stratified week-block bootstrap: 10,000 draws, seed 20260929, fixed
predictions, 95% percentile intervals; probability_positive=P(gain>0)+.5P(gain=0).
Primary contrast: augmented-minus-four-term accuracy; positive loss gains mean lower
loss. Report three baselines, no subgroup selection. Margin MAE omitted: this unit
changes a probability, not a predicted margin. Budget REDUCED from 217 to <=162 looks:
12 fits + 96 IS/OOS metric cells + 12 response-MAE cells + 20 reliability bands +
9 contrasts + 1 decisive comparison + 12 gaps; unavailable 2023 cells consume none.
Zero crossing never closes the signal; absent an admissible closing ground the result
is unresolved_below_power. No registry writes, serving changes, commits or publication.

## Paired held-out results

**Measured:** brackets are 95% season-stratified week-block bootstrap intervals, 10,000 draws, fixed predictions. Loss gains are comparator loss minus freshness loss; accuracy gains are freshness minus comparator, in percentage points.

| Arm | Accuracy % | Log loss | Brier |
| --- | --- | --- | --- |
| freshness | 54.596623 [50.847378, 58.174993] | 0.685551 [0.671433, 0.700133] | 0.246374 [0.239689, 0.253344] |
| four_term | 54.409006 [51.119403, 57.604718] | 0.684097 [0.670012, 0.698250] | 0.245586 [0.238912, 0.252335] |
| model | 52.908068 [48.387097, 57.353010] | 0.695016 [0.684787, 0.705278] | 0.250898 [0.245860, 0.255982] |
| market | 50.844278 [46.629213, 55.140187] | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] |

| Comparator | Metric | Gain [95% interval] | probability_positive |
| --- | --- | --- | --- |
| four_term | accuracy_points | 0.187617 [-2.044705, 2.247191] | 0.572050 |
| four_term | log_loss | -0.001454 [-0.004871, 0.001946] | 0.203200 |
| four_term | brier | -0.000788 [-0.002443, 0.000854] | 0.175400 |
| model | accuracy_points | 1.688555 [-3.619220, 7.104132] | 0.732800 |
| model | log_loss | 0.009465 [-0.008305, 0.028066] | 0.845500 |
| model | brier | 0.004524 [-0.003997, 0.013425] | 0.844500 |
| market | accuracy_points | 3.752345 [-1.679104, 9.023941] | 0.910600 |
| market | log_loss | 0.007596 [-0.006986, 0.021714] | 0.849800 |
| market | brier | 0.003626 [-0.003344, 0.010311] | 0.850700 |

## Training and held-out gap

**Measured:** training metrics pool each outer fit training appearance; 2023 appears twice. Gap is held-out minus training, so a positive loss gap means worse held-out loss. No held-out season enters its response or calibration fit.

| Arm | Metric | IS | OOS | OOS-IS |
| --- | --- | --- | --- | --- |
| freshness | accuracy_points | 57.268170 | 54.596623 | -2.671548 |
| freshness | log_loss | 0.678018 | 0.685551 | 0.007532 |
| freshness | brier | 0.242415 | 0.246374 | 0.003959 |
| four_term | accuracy_points | 57.017544 | 54.409006 | -2.608538 |
| four_term | log_loss | 0.678548 | 0.684097 | 0.005549 |
| four_term | brier | 0.242694 | 0.245586 | 0.002893 |
| model | accuracy_points | 51.754386 | 52.908068 | 1.153682 |
| model | log_loss | 0.699364 | 0.695016 | -0.004349 |
| model | brier | 0.252943 | 0.250898 | -0.002044 |
| market | accuracy_points | 50.626566 | 50.844278 | 0.217711 |
| market | log_loss | 0.693147 | 0.693147 | 0.000000 |
| market | brier | 0.250000 | 0.250000 | 0.000000 |

**Measured:** per-fold values are IS / OOS / gap.

| Held season | Arm | Train/test n | accuracy_points | log_loss | brier |
| --- | --- | --- | --- | --- | --- |
| 2024 | freshness | 266/266 | 59.774436 / 53.759398 / -6.015038 | 0.674518 / 0.687319 / 0.012801 | 0.240640 / 0.247124 / 0.006484 |
| 2024 | four_term | 266/266 | 57.518797 / 53.383459 / -4.135338 | 0.675886 / 0.684902 / 0.009016 | 0.241391 / 0.245799 / 0.004409 |
| 2024 | model | 266/266 | 50.375940 / 54.511278 / 4.135338 | 0.701840 / 0.694413 / -0.007427 | 0.254129 / 0.250570 / -0.003559 |
| 2024 | market | 266/266 | 50.751880 / 50.375940 / -0.375940 | 0.693147 / 0.693147 / 0.000000 | 0.250000 / 0.250000 / 0.000000 |
| 2025 | freshness | 532/267 | 56.015038 / 55.430712 / -0.584326 | 0.679769 / 0.683790 / 0.004021 | 0.243302 / 0.245626 / 0.002324 |
| 2025 | four_term | 532/267 | 56.766917 / 55.430712 / -1.336206 | 0.679879 / 0.683295 / 0.003416 | 0.243345 / 0.245374 / 0.002029 |
| 2025 | model | 532/267 | 52.443609 / 51.310861 / -1.132748 | 0.698126 / 0.695617 / -0.002510 | 0.252349 / 0.251226 / -0.001124 |
| 2025 | market | 532/267 | 50.563910 / 51.310861 / 0.746952 | 0.693147 / 0.693147 / -0.000000 | 0.250000 / 0.250000 / 0.000000 |

## Response fit and coverage

**Measured:** response MAE uses identical NFL response games for both response arms, averaging within game first. Pooled training additionally uses CFB. Rows are repeated quotes, never extra ATS games.

| Held | Response | Fit games/rows | NFL train/test games | IS MAE | OOS MAE | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | nfl_response | 272/84179 | 272/272 | 0.003138 | 0.003502 | 0.000363 |
| 2024 | pooled_response | 964/97954 | 272/272 | 0.003139 | 0.003502 | 0.000363 |
| 2025 | nfl_response | 544/134675 | 544/272 | 0.003320 | 0.003673 | 0.000353 |
| 2025 | pooled_response | 1971/158765 | 544/272 | 0.003320 | 0.003673 | 0.000353 |

| League/season | Feature games | Book rows | Response games | Response rows |
| --- | --- | --- | --- | --- |
| NFL_2023 | 272 | 93771 | 272 | 84179 |
| NFL_2024 | 272 | 56603 | 272 | 50496 |
| NFL_2025 | 272 | 65195 | 272 | 58608 |
| CFB_2023 | 871 | 38343 | 692 | 13775 |
| CFB_2024 | 900 | 29098 | 735 | 10315 |
| CFB_2025 | 928 | 30974 | 745 | 11296 |

## Coefficients and calibration

**Measured:** natural-unit coefficients below; response intercept fixed at zero. Pooled NFL response uses shared plus NFL-deviation slopes. ATS coefficients include the intercept and all four base terms; the correction is fitted jointly.

| Held | Fit | Term | Coefficient |
| --- | --- | --- | --- |
| 2024 | nfl_response | distance | 0.505256221 |
| 2024 | nfl_response | age_distance | 0.060637969 |
| 2024 | pooled_response | shared_distance | 0.620655967 |
| 2024 | pooled_response | shared_age_distance | -2.434234223 |
| 2024 | pooled_response | nfl_distance_deviation | -0.114180445 |
| 2024 | pooled_response | nfl_age_distance_deviation | 2.455552860 |
| 2024 | four_term | intercept | -0.002450034 |
| 2024 | four_term | model_logit | -0.037744839 |
| 2024 | four_term | composition_flag_sum | 0.319493402 |
| 2024 | four_term | market_move_toward_home | 0.215717705 |
| 2024 | four_term | market_move_available | 0.000000000 |
| 2024 | freshness | intercept | -0.002716599 |
| 2024 | freshness | model_logit | -0.031099026 |
| 2024 | freshness | composition_flag_sum | 0.315934710 |
| 2024 | freshness | market_move_toward_home | 0.218648292 |
| 2024 | freshness | market_move_available | 0.000000000 |
| 2024 | freshness | freshness_correction | -34.876123480 |
| 2025 | nfl_response | distance | 0.539779831 |
| 2025 | nfl_response | age_distance | -0.154936081 |
| 2025 | pooled_response | shared_distance | 0.550383523 |
| 2025 | pooled_response | shared_age_distance | -0.635258643 |
| 2025 | pooled_response | nfl_distance_deviation | -0.010536766 |
| 2025 | pooled_response | nfl_age_distance_deviation | 0.478117282 |
| 2025 | four_term | intercept | -0.007379773 |
| 2025 | four_term | model_logit | 0.162070695 |
| 2025 | four_term | composition_flag_sum | 0.284876847 |
| 2025 | four_term | market_move_toward_home | 0.191006481 |
| 2025 | four_term | market_move_available | 0.000000000 |
| 2025 | freshness | intercept | -0.007166137 |
| 2025 | freshness | model_logit | 0.167444331 |
| 2025 | freshness | composition_flag_sum | 0.286197273 |
| 2025 | freshness | market_move_toward_home | 0.192392957 |
| 2025 | freshness | market_move_available | 0.000000000 |
| 2025 | freshness | freshness_correction | -9.437263527 |

**Measured:** five training-quantile bands per arm, pooled across held-out seasons. Ties can leave bands empty; no band chooses the side.

| Arm | Training-quantile band | n | Mean home p | Home-cover rate |
| --- | --- | --- | --- | --- |
| freshness | 1 | 97 | 0.393963 | 0.443299 |
| freshness | 2 | 109 | 0.469432 | 0.495413 |
| freshness | 3 | 100 | 0.508572 | 0.450000 |
| freshness | 4 | 108 | 0.546395 | 0.518519 |
| freshness | 5 | 119 | 0.627188 | 0.613445 |
| four_term | 1 | 105 | 0.400819 | 0.438095 |
| four_term | 2 | 95 | 0.469553 | 0.494737 |
| four_term | 3 | 115 | 0.506615 | 0.426087 |
| four_term | 4 | 93 | 0.547272 | 0.569892 |
| four_term | 5 | 125 | 0.624728 | 0.608000 |
| model | 1 | 94 | 0.409594 | 0.500000 |
| model | 2 | 76 | 0.463550 | 0.434211 |
| model | 3 | 99 | 0.493620 | 0.484848 |
| model | 4 | 116 | 0.527092 | 0.543103 |
| model | 5 | 148 | 0.585973 | 0.540541 |
| market | 1 | 0 | n/a | n/a |
| market | 2 | 0 | n/a | n/a |
| market | 3 | 0 | n/a | n/a |
| market | 4 | 0 | n/a | n/a |
| market | 5 | 533 | 0.500000 | 0.508443 |

## Interpretation and limits

**Measured:** correction coefficients by held season: 2024: -34.876123; 2025: -9.437264. Compare the per-fold table for accuracy and loss stability.

**Inferred:** this is conditional research evidence, not a serving decision. Per AGENTS.md, an interval crossing zero is not a closing ground. The mechanism remains unresolved_below_power pending the orchestrator serial registry record; no positive-control or split-half-reliability closure was tested.

**Read:** the reused upstream artifact supplies a discrete conditional non-push model probability. This script never maps across spread lines or estimates a smooth residual margin distribution; a single fitted probability selects each side at the historical opener. Push and alternative-line serving are outside this unit.

**Inferred:** same-line synchronization excludes line-change responses, and six-hour source capping limits freshness resolution. CFB and NFL use identical paired-price probability units, but their sporting response may differ; penalized NFL deviations address that declared difference without selecting an arm afterward. Frozen upstream model/composition fits are inherited, not recertified as fully chronological here. Week-block intervals hold all fitted predictions fixed and do not quantify refit uncertainty. Two scored seasons cannot establish broad season stability.

**Measured:** 158 reporting looks within the 162-look ceiling: 8 fits + 72 IS/OOS metric cells + 8 response-MAE cells + 20 reliability bands + 9 paired contrasts + 1 decisive comparison + 12 pooled gaps + 24 fold gaps + 4 response gaps. Derived gaps are counted conservatively. One primary comparison; correlated diagnostics are not separate discoveries.

## Reproduction and provenance

```bash
UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead77_unit2.py
```

**Measured:** source hashes and structural counts:

```json
{
  "base": "artifacts/pick_probability/20260929T192747Z/per_game.parquet",
  "base_sha256": "0490c806caf9e9707abe28f3e3e42f85df312084ae52d9d1588d173bba5b01e5",
  "schedule": "data/raw/20260908T162105Z/schedules.parquet",
  "schedule_sha256": "a075d909f48e0a4301dbe7a90085c3fd4ad0c3ef5c487d0e3e29baf9a640650a",
  "declaration_sha256": "3e98198a15c01a2fae8556a6d31318705917450e59c90ddef88ab41a9b6e2b2c",
  "script_sha256": "3f38e1e37ed6b9670a151c4c3180a1a657a72c19c0f518786470210e57315605",
  "NFL_source_snapshots": 7644,
  "CFB_source_snapshots": 457,
  "validated_snapshots": 1792,
  "validator_asof_pairs": 508693,
  "paired_quotes": 349257,
  "selected_snapshot_hash": "10c241c873c5665e19d960d959879955a6624fc69636a388d4149630c792c173",
  "reporting_script_sha256": "4b1167d26ee6286d37c667b31151fc3f2af1e4ea4ca954502dd0c6b8d87d6a77",
  "post_run_edit": "Report wording, registry cell counts and compact-lane declaration reload only; no refit."
}
```

Prediction rows, response rows, result JSON and a registry batch are saved only under `tests/scratch/codex/lead77_unit2/`. The lane contains the exact registry command; this script never runs it.

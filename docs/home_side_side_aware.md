# MOD-18 Lane G: side-aware home push

## Predeclaration (frozen before computation, 2026-09-08)

Read: `docs/home_side_offset_promotion.md`, section "S3 played", defines
the incumbent as the 100-game-prior home offset in buckets 7, 7.5-10,
and 10.5+ only. Read: `src/nfl_ats/home_side_location.py:239` defines
`prior_rows_before`: earlier weeks only, seasons >= target season minus five,
completed results and valid raw points only.

S4 fits sum(actual home margin - raw archived point)/(cell count + 100)
separately in each (bucket x home side) cell in those three buckets.
S4b uses a 50-game prior, a predeclared sensitivity, not a search.
Positive opener line means home favourite; negative means home underdog.
Small buckets stay zero. Both retain the ridge and gaussian_median mapping,
using `MarginModel.predict(center_offset=...)`.

First replay S3 on the active-matched opener archive. Stop before candidate
scoring if any served probability differs by more than 1e-9.
Grade paired non-push opener accuracy, Brier and log loss; bootstrap whole
weeks with 20,000 draws, seed 20260817, within-week correlation ZERO (never
estimated or padded). Report standalone and through the played three-member
card, every season and bucket x side cell. Run overlay-composition on research
per_game artifacts with research_arm and research_laneG_S4 / research_laneG_S4b
identities. Read the linked Week 1 sidecar without generating a forecast.
Record all comparison cells through weak-signals record, family
mod18_home_side_location_v1, prefix mod18_home_side_location_v1_s4_.

Inferred mechanism: home-favourite and home-underdog games can have different
home point errors at the same spread size; fitting separate locations may
represent that difference. This is a further post-hoc split on the mined
archive on which S2/S3 were selected, not independent confirmation.
Decision: probability_positive > 0.5 through the played card favours playing
the candidate; disclose sensitivity separately, without selecting a tuned prior.

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is `unresolved_below_power`: record it with `nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not the validator. Never state that anything "needs more games"; decide on expected value (P+ above 0.5 favours playing), and state what the numbers imply for the DECISION (play S4 instead of S3?) before what is wrong with them.

## Measured results (appended after computation)

Inferred decision from measured `artifacts/research/laneG/cells.json`: keep S3; neither S4 nor the predeclared S4b sensitivity favours replacing it through the played card. This does not close the side-dependent football mechanism. All cells remain unresolved_below_power; the fitted arms lose on this reused archive.

Measured: script stages `replay`, `score`, and `week1`; matching archive `artifacts/opener_evaluation/20260908T115957Z`, 1,503 non-push games across 107 weeks. S3 maximum served-probability replay gap 3.774758e-15; offset gap 6.661338e-16. Week-blocked 20,000 draws, seed 20260817, no correlation estimation or padding.

Measured decision table (`cells.json`; loss values are standalone probabilities):

| Arm | Standalone accuracy | Card accuracy | Card delta, 95% (pts) | probability_positive | Brier | Log loss | Week 1 model-side changes |
|---|---:|---:|---|---:|---:|---:|---:|
| S3 | 54.5576% | 55.8882% | reference | ? | 0.25158399 | 0.69662362 | reference |
| S4 | 53.8257% | 55.2229% | -0.6653 [-1.1251, -0.2640] | 0.00025 | 0.25171166 | 0.69687696 | 0/16 |
| S4B | 54.2914% | 55.4225% | -0.4657 [-0.9265, -0.0668] | 0.00370 | 0.25172572 | 0.69690487 | 0/16 |

Measured: S4 standalone delta -0.7319 [-1.2987, -0.1980], probability_positive 0.00265; S4b -0.2661 [-0.8458, +0.2672], probability_positive 0.14345. S4 changes 19 standalone / 12 composed historical picks; S4b 16 / 9. Brier improvement probability_positive: S4 0.15050, S4b 0.05770; log-loss improvement: 0.15795 / 0.06320.

Measured (`week1.json`): linked sidecar `artifacts/margin_predictions/2026-week-01-20260908T120939Z/home_side_offset.json` replays to 9.992007e-16. No model-side changes in either arm; no forecast regenerated.

Measured per-cell and per-season accuracy comparisons (`cells.json`), all versus S3; each entry is delta [95%], probability_positive. Small-bucket all-zero deltas have probability_positive 0 because the bootstrap uses strictly positive draws, not because S3 is better.

| Cell | n | S4 standalone | S4 card | S4b standalone | S4b card |
|---|---:|---|---|---|---|
| cell_bucket_0-3_home_favourite | 294 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_0-3_home_underdog | 246 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_0-3_pickem | 12 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_10p5plus_home_favourite | 99 | -6.061 [-12.621, +0.000], 0.01490 | -6.061 [-11.818, -0.980], 0.00495 | -1.010 [-6.522, +4.040], 0.27780 | -3.030 [-7.779, +1.020], 0.04450 |
| cell_bucket_10p5plus_home_underdog | 32 | -3.125 [-9.375, +0.000], 0.00000 | -3.125 [-9.375, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_3p5-6p5_home_favourite | 317 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_3p5-6p5_home_underdog | 235 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_7_home_favourite | 49 | -2.041 [-6.818, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +2.041 [-4.444, +9.091], 0.60950 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_7_home_underdog | 25 | +4.000 [+0.000, +12.500], 0.63715 | -4.000 [-12.500, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |
| cell_bucket_7p5-10_home_favourite | 125 | -0.800 [-3.604, +1.724], 0.18380 | -0.800 [-2.542, +0.000], 0.00000 | +0.000 [-2.326, +2.439], 0.34605 | -0.800 [-2.542, +0.000], 0.00000 |
| cell_bucket_7p5-10_home_underdog | 69 | -4.348 [-9.677, +0.000], 0.00000 | -1.449 [-4.688, +0.000], 0.00000 | -5.797 [-11.765, -1.389], 0.00000 | -4.348 [-9.722, +0.000], 0.00000 |
| season_2020 | 220 | -0.455 [-1.382, +0.000], 0.00000 | -0.455 [-1.382, +0.000], 0.00000 | -0.909 [-2.252, +0.000], 0.00000 | -0.909 [-2.252, +0.000], 0.00000 |
| season_2021 | 236 | -0.424 [-1.282, +0.000], 0.00000 | -0.424 [-1.282, +0.000], 0.00000 | +0.424 [-0.851, +1.778], 0.61385 | +0.000 [-1.255, +1.271], 0.35180 |
| season_2022 | 248 | -0.806 [-2.459, +0.791], 0.09380 | -1.210 [-2.553, +0.000], 0.00000 | +0.806 [+0.000, +2.000], 0.87805 | +0.000 [+0.000, +0.000], 0.00000 |
| season_2023 | 266 | -1.880 [-3.333, -0.733], 0.00000 | -1.128 [-2.281, +0.000], 0.00000 | -1.128 [-3.053, +0.722], 0.07135 | -1.504 [-3.358, +0.000], 0.00000 |
| season_2024 | 266 | +0.376 [-1.149, +1.931], 0.59110 | -0.376 [-1.550, +0.760], 0.18190 | +0.000 [-1.124, +1.136], 0.34825 | -0.376 [-1.154, +0.000], 0.00000 |
| season_2025 | 267 | -1.124 [-2.281, +0.000], 0.00000 | -0.375 [-1.136, +0.000], 0.00000 | -0.749 [-1.873, +0.000], 0.00000 | +0.000 [+0.000, +0.000], 0.00000 |

Measured full loss comparisons for every season and bucket x side are retained in `artifacts/research/laneG/cells.json` and recorded alongside accuracy cells. Research composition outputs and exact successful CLI argv are in `artifacts/research/laneG/commands.json`; per_game metadata uses research identities, never the active identity.

Read/inferred limitation: the predeclaration above explicitly discloses the further post-hoc split on the S2/S3 selection archive. The sensitivity changes shrinkage as well as separating sides; it is not an independent replication.

## Registry and verification

Measured by reading `registry/weak_signals.json`: 144 matching rows, all unresolved_below_power; exact names:

- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_pickem_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_pickem_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_pickem_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_0-3_pickem_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_10p5plus_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_3p5-6p5_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_cell_bucket_7p5-10_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_overall_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4_overall_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4_overall_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4_overall_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4_season_2020_brier_2020_2020`
- `mod18_home_side_location_v1_s4_s4_season_2020_card_2020_2020`
- `mod18_home_side_location_v1_s4_s4_season_2020_log_loss_2020_2020`
- `mod18_home_side_location_v1_s4_s4_season_2020_standalone_2020_2020`
- `mod18_home_side_location_v1_s4_s4_season_2021_brier_2021_2021`
- `mod18_home_side_location_v1_s4_s4_season_2021_card_2021_2021`
- `mod18_home_side_location_v1_s4_s4_season_2021_log_loss_2021_2021`
- `mod18_home_side_location_v1_s4_s4_season_2021_standalone_2021_2021`
- `mod18_home_side_location_v1_s4_s4_season_2022_brier_2022_2022`
- `mod18_home_side_location_v1_s4_s4_season_2022_card_2022_2022`
- `mod18_home_side_location_v1_s4_s4_season_2022_log_loss_2022_2022`
- `mod18_home_side_location_v1_s4_s4_season_2022_standalone_2022_2022`
- `mod18_home_side_location_v1_s4_s4_season_2023_brier_2023_2023`
- `mod18_home_side_location_v1_s4_s4_season_2023_card_2023_2023`
- `mod18_home_side_location_v1_s4_s4_season_2023_log_loss_2023_2023`
- `mod18_home_side_location_v1_s4_s4_season_2023_standalone_2023_2023`
- `mod18_home_side_location_v1_s4_s4_season_2024_brier_2024_2024`
- `mod18_home_side_location_v1_s4_s4_season_2024_card_2024_2024`
- `mod18_home_side_location_v1_s4_s4_season_2024_log_loss_2024_2024`
- `mod18_home_side_location_v1_s4_s4_season_2024_standalone_2024_2024`
- `mod18_home_side_location_v1_s4_s4_season_2025_brier_2025_2025`
- `mod18_home_side_location_v1_s4_s4_season_2025_card_2025_2025`
- `mod18_home_side_location_v1_s4_s4_season_2025_log_loss_2025_2025`
- `mod18_home_side_location_v1_s4_s4_season_2025_standalone_2025_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_pickem_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_pickem_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_pickem_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_0-3_pickem_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_10p5plus_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_3p5-6p5_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_favourite_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_favourite_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_favourite_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_favourite_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_underdog_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_underdog_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_underdog_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_cell_bucket_7p5-10_home_underdog_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_overall_brier_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_overall_card_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_overall_log_loss_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_overall_standalone_2020_2025`
- `mod18_home_side_location_v1_s4_s4b_season_2020_brier_2020_2020`
- `mod18_home_side_location_v1_s4_s4b_season_2020_card_2020_2020`
- `mod18_home_side_location_v1_s4_s4b_season_2020_log_loss_2020_2020`
- `mod18_home_side_location_v1_s4_s4b_season_2020_standalone_2020_2020`
- `mod18_home_side_location_v1_s4_s4b_season_2021_brier_2021_2021`
- `mod18_home_side_location_v1_s4_s4b_season_2021_card_2021_2021`
- `mod18_home_side_location_v1_s4_s4b_season_2021_log_loss_2021_2021`
- `mod18_home_side_location_v1_s4_s4b_season_2021_standalone_2021_2021`
- `mod18_home_side_location_v1_s4_s4b_season_2022_brier_2022_2022`
- `mod18_home_side_location_v1_s4_s4b_season_2022_card_2022_2022`
- `mod18_home_side_location_v1_s4_s4b_season_2022_log_loss_2022_2022`
- `mod18_home_side_location_v1_s4_s4b_season_2022_standalone_2022_2022`
- `mod18_home_side_location_v1_s4_s4b_season_2023_brier_2023_2023`
- `mod18_home_side_location_v1_s4_s4b_season_2023_card_2023_2023`
- `mod18_home_side_location_v1_s4_s4b_season_2023_log_loss_2023_2023`
- `mod18_home_side_location_v1_s4_s4b_season_2023_standalone_2023_2023`
- `mod18_home_side_location_v1_s4_s4b_season_2024_brier_2024_2024`
- `mod18_home_side_location_v1_s4_s4b_season_2024_card_2024_2024`
- `mod18_home_side_location_v1_s4_s4b_season_2024_log_loss_2024_2024`
- `mod18_home_side_location_v1_s4_s4b_season_2024_standalone_2024_2024`
- `mod18_home_side_location_v1_s4_s4b_season_2025_brier_2025_2025`
- `mod18_home_side_location_v1_s4_s4b_season_2025_card_2025_2025`
- `mod18_home_side_location_v1_s4_s4b_season_2025_log_loss_2025_2025`
- `mod18_home_side_location_v1_s4_s4b_season_2025_standalone_2025_2025`

Measured checks: 12 tests passed (25.44 seconds), including the artifact provenance scanner. Commands:

```powershell
.\.tools\uv.exe run --no-sync pytest tests/test_home_side_side_aware.py tests/test_experiment_registry.py -n 2 --basetemp C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/fedb09af-a2ec-4d29-af01-540025768002/scratchpad/laneG
.\.tools\uv.exe run --no-sync ruff format --check scripts/home_side_side_aware_opener_eval.py tests/test_home_side_side_aware.py
.\.tools\uv.exe run --no-sync ruff check scripts/home_side_side_aware_opener_eval.py tests/test_home_side_side_aware.py
```

Measured Git changes: registry/weak_signals.json plus the three new lane files. No commit or push. Read: the user lane allowlist supersedes scheduler, dashboard, handoff, and full-repository workflow actions for this task.

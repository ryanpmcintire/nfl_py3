# MOD-18 Lane H: big-spread lattice

## Predeclaration (frozen before computation, 2026-09-08)

Read: `src/nfl_ats/conditional_margin.py:18` freezes K1 bandwidth at 2.5,
integer outcome atoms, and K3 side-of-nearest-key conditioning with its existing
support fallback. Read: `scripts/conditional_margin_opener_eval.py:35` builds
prior out-of-time median-centered points using the opener where available and
otherwise the historical line; preserve this construction and the four-season,
strictly-before-week cutoff in `predict_conditional_margin`.

M1 uses served S3 points with K1 ONLY in spread_bucket values 7, 7.5-10,
10.5+. M1b is the predeclared K3 sibling on precisely those buckets, not a
search. Both retain the exact served gaussian_median probabilities in 0-3 and
3.5-6.5. Target lattice centers add the S3 offset to K's median-centered point;
historical kernel samples remain K's frozen construction, without retuning.

First replay S3 against the active-matched opener evaluation; stop before any
candidate scoring if a served probability differs by more than 1e-9. Grade
paired non-push opener accuracy, Brier, log loss, overall, each season and each
bucket, plus through the played three-member overlay union. Whole-week bootstrap:
20,000 draws, seed 20260817, within-week correlation zero, never estimated or
padded. Report push probability at absolute line 3 versus realised; unchanged small-bucket push diagnostic uses the served three-way split. Read the linked Week 1
predictions and sidecar without regeneration. Research identities are
research_laneH_M1 and research_laneH_M1b. Record every comparison cell through
the CLI in mod18_conditional_margin_v1, prefix mod18_conditional_margin_v1_m1_.

Inferred mechanism: change distribution shape only where the big-spread shape
error was diagnosed, preserving the calibrated small-spread read. This is a
post-hoc restriction on the mined archive on which lanes K and S were selected,
not independent confirmation. Through-card probability_positive above 0.5
favours playing; report that decision before limitations. No production changes.

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is `unresolved_below_power`: record it with `nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not the validator.

Read clarification before push scoring: `src/nfl_ats/margin.py:860` supplies
rounded-residual three-way push mass alongside the Gaussian cover read. Preserve
that served diagnostic in small buckets; the initial zero-mass assumption was
incorrect and is not used in the reported push comparison. Arm definitions unchanged.

## Decision and measured results

Inferred decision from measured `artifacts/research/laneH/cells.json`: keep S3 on the played card. Both siblings improve standalone accuracy but their expected through-card gains are negative. Neither arm is closed; all comparisons remain unresolved_below_power.

Measured (`reproduction.json`): matching archive `F:\Repos\nfl_py3\artifacts\opener_evaluation\20260908T115957Z`, maximum S3 probability gap 3.77e-15; offset gap 6.66e-16. Paired accuracy uses 1,503 non-push games and 107 weeks; the archive has 1,537 rows including pushes. Whole-week 20,000 draws, seed 20260817, correlation zero.

Measured decision table (`cells.json`, `push_calibration.json`, `week1.json`):

| Arm | Standalone % | Card % | Card delta points [95%] | probability_positive | Brier | Log loss | Push at 3 % | Week 1 model/card changes |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| S3 | 54.5576 | 55.8882 | reference | reference | 0.25158399 | 0.69662362 | 3.499 | reference |
| M1 | 55.1564 | 55.7552 | -0.133 [-1.355, +1.077] | 0.40095 | 0.25187471 | 0.69739377 | 3.499 | 2/1 |
| M1b | 54.8902 | 55.6221 | -0.266 [-1.718, +1.195] | 0.34645 | 0.25229389 | 0.69823670 | 3.499 | 1/1 |

Measured (`push_calibration.json`): realised push rate at absolute line 3 is 10.050% on 199 games. All three arms retain the served small-bucket rounded-residual diagnostic. Inferred: M1 cannot repair the line-3 mass deficit because its lattice is switched off there; its atoms are not translated in the big buckets.

Measured per-bucket accuracy (`cells.json`), with standalone/card deltas and probability_positive versus S3:

| Bucket | n | S3 standalone/card % | M1 standalone/card % | M1 delta standalone/card; probability_positive | M1b standalone/card % | M1b delta standalone/card; probability_positive |
|---|---:|---|---|---|---|---|
| 0-3 | 552 | 56.16/57.43 | 56.16/57.43 | +0.000 [+0.000, +0.000], 0.00000 / +0.000 [+0.000, +0.000], 0.00000 | 56.16/57.43 | +0.000 [+0.000, +0.000], 0.00000 / +0.000 [+0.000, +0.000], 0.00000 |
| 3p5-6p5 | 552 | 56.16/57.79 | 56.16/57.79 | +0.000 [+0.000, +0.000], 0.00000 / +0.000 [+0.000, +0.000], 0.00000 | 56.16/57.79 | +0.000 [+0.000, +0.000], 0.00000 / +0.000 [+0.000, +0.000], 0.00000 |
| 7 | 74 | 51.35/50.00 | 51.35/43.24 | +0.000 [-12.329, +12.857], 0.45265 / -6.757 [-16.250, +2.703], 0.05615 | 51.35/44.59 | +0.000 [-15.068, +15.068], 0.46565 / -5.405 [-18.310, +7.576], 0.17495 |
| 7p5-10 | 194 | 51.55/51.55 | 54.64/52.58 | +3.093 [-5.584, +12.000], 0.74265 / +1.031 [-5.742, +8.040], 0.59490 | 50.52/49.48 | -1.031 [-10.606, +9.091], 0.40495 / -2.062 [-9.845, +6.011], 0.29035 |
| 10p5plus | 131 | 47.33/51.15 | 49.62/51.91 | +2.290 [-7.194, +12.121], 0.65060 / +0.763 [-7.200, +9.009], 0.53575 | 52.67/54.20 | +5.344 [-4.286, +15.038], 0.84445 / +3.053 [-4.839, +11.111], 0.74785 |

Measured seasonal comparisons (`cells.json`), delta points [95%], probability_positive:

| Season | M1 standalone | M1 card | M1b standalone | M1b card |
|---|---|---|---|---|
| 2020 | +1.818 [-1.852, +5.882], 0.78500 | +0.909 [-2.222, +4.327], 0.65425 | +0.455 [-4.977, +5.991], 0.53735 | +0.000 [-4.310, +4.525], 0.46090 |
| 2021 | -0.847 [-5.310, +3.766], 0.32380 | -1.695 [-4.348, +1.235], 0.08570 | -1.695 [-6.780, +3.478], 0.23335 | -1.271 [-5.128, +2.575], 0.22285 |
| 2022 | +2.823 [-0.397, +6.061], 0.94180 | +0.000 [-3.226, +2.941], 0.47455 | +2.823 [-0.787, +6.478], 0.92365 | +0.000 [-3.488, +3.292], 0.47595 |
| 2023 | +0.376 [-3.333, +4.089], 0.53940 | -0.376 [-4.135, +3.435], 0.38620 | +1.128 [-2.593, +4.669], 0.69475 | +0.000 [-3.610, +3.637], 0.45495 |
| 2024 | -0.376 [-3.042, +2.256], 0.34190 | +0.376 [-1.901, +2.353], 0.58085 | -1.880 [-5.166, +1.111], 0.08645 | -0.752 [-4.135, +1.931], 0.29255 |
| 2025 | +0.000 [-3.371, +3.435], 0.44345 | +0.000 [-2.583, +2.239], 0.44735 | +1.124 [-2.239, +4.762], 0.69170 | +0.375 [-2.583, +3.238], 0.55015 |

Measured: every bucket and season also has paired Brier/log-loss comparisons in `cells.json`, recorded alongside accuracy. All-zero differences have strict-positive bootstrap probability 0, which denotes a tie, not evidence favouring S3.

Measured (`week1.json`): linked forecast `F:\Repos\nfl_py3\artifacts\margin_predictions\2026-week-01-20260908T124514Z`; sidecar replay gap 9.99e-16; no forecast regenerated.

Measured M1 changes (`week1.json`): 2026_01_CLE_JAX: JAX -> CLE; 2026_01_NO_DET: DET -> NO.

Measured M1 card_changes (`week1.json`): 2026_01_NO_DET: DET -> NO.

Measured M1b changes (`week1.json`): 2026_01_NO_DET: DET -> NO.

Measured M1b card_changes (`week1.json`): 2026_01_NO_DET: DET -> NO.

Read limitation (predeclaration above): this is a post-hoc restriction on the mined archive that selected lanes K and S, not independent confirmation. No arm was retuned after scoring.

## Registry and verification

Measured: verified all 98 expected CLI-recorded rows in `registry/weak_signals.json`; exact names also in `artifacts/research/laneH/registry_names.json`. Family `mod18_conditional_margin_v1`; all `unresolved_below_power`.

- `mod18_conditional_margin_v1_m1_m1_cell_bucket_0-3_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_0-3_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_0-3_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_0-3_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_10p5plus_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_10p5plus_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_10p5plus_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_10p5plus_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_3p5-6p5_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_3p5-6p5_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_3p5-6p5_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_3p5-6p5_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7p5-10_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7p5-10_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7p5-10_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_cell_bucket_7p5-10_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_overall_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_overall_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_overall_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_overall_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_push_at_3_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1_season_2020_brier_2020_2020`
- `mod18_conditional_margin_v1_m1_m1_season_2020_card_2020_2020`
- `mod18_conditional_margin_v1_m1_m1_season_2020_log_loss_2020_2020`
- `mod18_conditional_margin_v1_m1_m1_season_2020_standalone_2020_2020`
- `mod18_conditional_margin_v1_m1_m1_season_2021_brier_2021_2021`
- `mod18_conditional_margin_v1_m1_m1_season_2021_card_2021_2021`
- `mod18_conditional_margin_v1_m1_m1_season_2021_log_loss_2021_2021`
- `mod18_conditional_margin_v1_m1_m1_season_2021_standalone_2021_2021`
- `mod18_conditional_margin_v1_m1_m1_season_2022_brier_2022_2022`
- `mod18_conditional_margin_v1_m1_m1_season_2022_card_2022_2022`
- `mod18_conditional_margin_v1_m1_m1_season_2022_log_loss_2022_2022`
- `mod18_conditional_margin_v1_m1_m1_season_2022_standalone_2022_2022`
- `mod18_conditional_margin_v1_m1_m1_season_2023_brier_2023_2023`
- `mod18_conditional_margin_v1_m1_m1_season_2023_card_2023_2023`
- `mod18_conditional_margin_v1_m1_m1_season_2023_log_loss_2023_2023`
- `mod18_conditional_margin_v1_m1_m1_season_2023_standalone_2023_2023`
- `mod18_conditional_margin_v1_m1_m1_season_2024_brier_2024_2024`
- `mod18_conditional_margin_v1_m1_m1_season_2024_card_2024_2024`
- `mod18_conditional_margin_v1_m1_m1_season_2024_log_loss_2024_2024`
- `mod18_conditional_margin_v1_m1_m1_season_2024_standalone_2024_2024`
- `mod18_conditional_margin_v1_m1_m1_season_2025_brier_2025_2025`
- `mod18_conditional_margin_v1_m1_m1_season_2025_card_2025_2025`
- `mod18_conditional_margin_v1_m1_m1_season_2025_log_loss_2025_2025`
- `mod18_conditional_margin_v1_m1_m1_season_2025_standalone_2025_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_0-3_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_0-3_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_0-3_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_0-3_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_10p5plus_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_10p5plus_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_10p5plus_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_10p5plus_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_3p5-6p5_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_3p5-6p5_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_3p5-6p5_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_3p5-6p5_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7p5-10_all_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7p5-10_all_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7p5-10_all_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_cell_bucket_7p5-10_all_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_overall_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_overall_card_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_overall_log_loss_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_overall_standalone_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_push_at_3_brier_2020_2025`
- `mod18_conditional_margin_v1_m1_m1b_season_2020_brier_2020_2020`
- `mod18_conditional_margin_v1_m1_m1b_season_2020_card_2020_2020`
- `mod18_conditional_margin_v1_m1_m1b_season_2020_log_loss_2020_2020`
- `mod18_conditional_margin_v1_m1_m1b_season_2020_standalone_2020_2020`
- `mod18_conditional_margin_v1_m1_m1b_season_2021_brier_2021_2021`
- `mod18_conditional_margin_v1_m1_m1b_season_2021_card_2021_2021`
- `mod18_conditional_margin_v1_m1_m1b_season_2021_log_loss_2021_2021`
- `mod18_conditional_margin_v1_m1_m1b_season_2021_standalone_2021_2021`
- `mod18_conditional_margin_v1_m1_m1b_season_2022_brier_2022_2022`
- `mod18_conditional_margin_v1_m1_m1b_season_2022_card_2022_2022`
- `mod18_conditional_margin_v1_m1_m1b_season_2022_log_loss_2022_2022`
- `mod18_conditional_margin_v1_m1_m1b_season_2022_standalone_2022_2022`
- `mod18_conditional_margin_v1_m1_m1b_season_2023_brier_2023_2023`
- `mod18_conditional_margin_v1_m1_m1b_season_2023_card_2023_2023`
- `mod18_conditional_margin_v1_m1_m1b_season_2023_log_loss_2023_2023`
- `mod18_conditional_margin_v1_m1_m1b_season_2023_standalone_2023_2023`
- `mod18_conditional_margin_v1_m1_m1b_season_2024_brier_2024_2024`
- `mod18_conditional_margin_v1_m1_m1b_season_2024_card_2024_2024`
- `mod18_conditional_margin_v1_m1_m1b_season_2024_log_loss_2024_2024`
- `mod18_conditional_margin_v1_m1_m1b_season_2024_standalone_2024_2024`
- `mod18_conditional_margin_v1_m1_m1b_season_2025_brier_2025_2025`
- `mod18_conditional_margin_v1_m1_m1b_season_2025_card_2025_2025`
- `mod18_conditional_margin_v1_m1_m1b_season_2025_log_loss_2025_2025`
- `mod18_conditional_margin_v1_m1_m1b_season_2025_standalone_2025_2025`


Measured verification (`artifacts/research/laneH/verification.json`): 40 tests
passed in 22.79 seconds; scoped Ruff format and lint checks passed. Commands:

```powershell
.\.tools\uv.exe run --no-sync ruff format --check scripts/big_spread_lattice_opener_eval.py tests/test_big_spread_lattice.py
.\.tools\uv.exe run --no-sync ruff check scripts/big_spread_lattice_opener_eval.py tests/test_big_spread_lattice.py
.\.tools\uv.exe run --no-sync pytest tests/test_big_spread_lattice.py tests/test_conditional_margin.py tests/test_experiment_registry.py -n 2 --basetemp C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/fedb09af-a2ec-4d29-af01-540025768002/scratchpad/laneH -q
```

Measured execution: `python scripts/big_spread_lattice_opener_eval.py --stage`
was run with replay, map, score, week1, and record under the locked uv environment.
`commands.json` retains successful real overlay-composition argv and registry CLI
argv. The score stage was repeated to preserve the served push diagnostic;
accuracy and cover-loss results were unchanged. Tests cover small-bucket identity,
frozen K parity, corrected-center effects, future/same-week/old-history exclusion,
and the fail-closed S3 replay gate.

Measured final `git status --short`: modified `registry/weak_signals.json`; new
`docs/big_spread_lattice.md`, `scripts/big_spread_lattice_opener_eval.py`, and
`tests/test_big_spread_lattice.py`. Research artifacts are under the allowed
`artifacts/research/laneH/`. No commit or push was made. Read scope constraint:
the owner's Lane H instructions prohibit source, scheduler, handoff, roadmap and
publication writes; the session-wide dashboard/handoff tasks were therefore not run.

# LEAD-72 push-adjusted Best Pick replay

**Measured: Decisive weeks (nominee changes):** 13 weeks; raw-cover W-L-P 7-6-0; push-adjusted W-L-P 7-6-0.
Paired gain +0.0000 grade percentage points (95% week-bootstrap interval -38.4615 to +38.4615); probability_positive=0.501750.

| Ranking | Pool-grade accuracy | 95% interval |
|---|---:|---:|
| raw_cover | 53.8462% | 30.7692% to 76.9231% |
| push_adjusted | 53.8462% | 23.0769% to 76.9231% |

Paired better/worse/equal weeks: 3/3/7.

**Measured: All declared weeks:** 107 weeks; raw-cover W-L-P 56-49-2; push-adjusted W-L-P 56-49-2.
Paired gain +0.0000 grade percentage points (95% week-bootstrap interval -4.6729 to +4.6729); probability_positive=0.501600.

| Ranking | Pool-grade accuracy | 95% interval |
|---|---:|---:|
| raw_cover | 53.2710% | 43.9252% to 62.6168% |
| push_adjusted | 53.2710% | 43.9252% to 62.6168% |

Paired better/worse/equal weeks: 3/3/101.

## Protocol and interpretation

**Read:** declaration saved in `docs/lanes/lead72.md` before outcome access. Two declared ranking looks: raw P(served-side cover), and P(cover)+0.5*P(push). The decisive subset and all six season rows are required descriptive diagnostics; no subgroup, threshold, model, timing or additional ranking was selected.

**Read:** `PoolRules` (src/nfl_ats/pool_workbench.py:37-45) fixes win/push/loss at 1/0.5/0. LEAD-53 Tuesday dispersion eligibility and its exact tie function are reused (scripts/best_pick_sunday_renomination_eval.py:54-77,228-243). Tie order: rank statistic rounded to 12 decimals, minimum dispersion, minimum absolute frozen spread, earliest kickoff, then equal weight across remaining ties.

**Measured:** ranking leaves every archived probability-selected side unchanged. Raw cover plus push plus loss sums to one; the saved non-push probability matches cover/(cover+loss). No smooth residual or future line enters ranking. Pushes remain in the outcome denominator. The normalized accuracy target uses the pool 0.5 push grade; it is not total bonus points or wagering return.

**Measured:** 1537 games, 107 weeks, six seasons; 812 eligible game-week rows; 10 dispersion fallback weeks. Policies: discrete_conditional_non_push_v1.

Paired week-block bootstrap: 20,000 draws, seed 20260929, percentile 95% intervals; probability_positive credits an exactly zero draw by 0.5. All-week and decisive-week resampling preserve both arms within each sampled week.

## Season stability and coefficients

**Read:** the row explicitly requires deterministic arithmetic and no fold choice. No parameter was fitted or selected: in-sample fit scores and IS/OOS fitting gap are N/A. Fitted fold coefficients are N/A; fixed coefficients below apply in every season. OOS means the saved chronological weekly forecasts, not an untouched outer test of this research idea. No new LOSO model was fitted.

| Season | Weeks | Changed | Raw W-L-P | Adjusted W-L-P | Raw OOS grade | Adjusted OOS grade | Difference (pp) | Raw (cover,push) | Adjusted (cover,push) |
|---:|---:|---:|---|---|---:|---:|---:|---|---|
| 2020 | 17 | 1 | 10-7-0 | 9-8-0 | 58.8235% | 52.9412% | -5.8824 | (1,0) | (1,0.5) |
| 2021 | 18 | 2 | 9-9-0 | 9-9-0 | 50.0000% | 50.0000% | +0.0000 | (1,0) | (1,0.5) |
| 2022 | 18 | 4 | 11-7-0 | 11-7-0 | 61.1111% | 61.1111% | +0.0000 | (1,0) | (1,0.5) |
| 2023 | 18 | 3 | 4-13-1 | 4-13-1 | 25.0000% | 25.0000% | +0.0000 | (1,0) | (1,0.5) |
| 2024 | 18 | 1 | 12-5-1 | 12-5-1 | 69.4444% | 69.4444% | +0.0000 | (1,0) | (1,0.5) |
| 2025 | 18 | 2 | 10-8-0 | 11-7-0 | 55.5556% | 61.1111% | +5.5556 | (1,0) | (1,0.5) |

## Source inventory

**Measured:** source hashes identify the exact local replay inputs; only the declared columns were loaded.

| Source | Rows | Required columns | SHA-256 |
|---|---:|---|---|
| `artifacts/opener_evaluation/20260920T135435Z/per_game.parquet` | 1537 | present | `dfd2ad0283328f45510290eea5a3827fd46bfc0fd18e4c4b01b930b33b60c007` |
| `artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet` | 1537 | present | `a3a31d45d3f185b9843f30f9a8642d93ef7e7798c4a2ba024a47088993784a36` |
| `artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet` | 1537 | present | `5dbbb3361937f067c57eef9627e1070b39a037a1b1d1ea07963bc2c6a17ca40a` |
| `data/processed/game_features_weak_stack.parquet` | 4902 | present | `a3eedb0323818f95b029047c5b80de3ce8d458a48b82ae4489c1fb70d16d5685` |

Paired nomination-level output: `docs/lead72_weekly.md`.

## Review and pending registry write

**Inferred:** proposed classification is `unresolved_below_power`, pending the
orchestrator's serial registry write. The effect is 0.0000 grade percentage
points [95% -4.6729,+4.6729], probability_positive=0.5016. Under AGENTS.md's
research-closure rule, no admissible closing ground is established: no resolved
wrong sign, zero split-half reliability, or demonstrated power-control bound
was measured. This replay supplies no evidence of an improvement to serve.

**Measured:** the six season effects were -5.8824, 0, 0, 0, 0, and +5.5556
grade percentage points, respectively. Fixed (cover,push) coefficients were
(1,0) versus (1,0.5) in every season. These are deterministic ranking weights;
there are no fitted per-fold coefficients, in-sample fit scores, or IS/OOS fit
gap. This explicit N/A follows the roadmap's no-fit protocol.

**Measured:** one real command completed successfully:
`.tools/uv.exe run --no-sync python -B scripts/lead72_unit1.py`.
Scoped Ruff checks passed. The run used `UV_NO_CACHE=1` because the default uv
cache was inaccessible, with numerical thread limits of 2. Output is retained
in `.tmp/lead72_unit1.log`. No new tests or registry writes were made.

The exact pending registry command is in `docs/lanes/lead72.md` under
Record commands. It records the aggregate effect, retains the correlated
decisive subset in notes, and uses a distinct family: normalized Best Pick
grade with half-credit pushes must not be pooled with ordinary ATS accuracy.

# LEAD-90 — fit probability mass near the frozen opener

## Goal
Execute the fixed LEAD-90 protocol; historical OPENER proxies the frozen pool line. No serving, registry, publication, or Git mutations.

## State
Protocol copied before any outcome computation; source/chronology inventory in progress.

Implementation fixed before outcomes: use every game with at least one clock-valid common bookmaker; recover signed composition flags for pushes and verify against frozen non-push flags. Candidate PMF is `q(k)*exp((k-opener)*Xβ)/Z`, with the same four standardized inputs plus intercept, fitted on all training margins including pushes, ridge .001 on non-intercept coefficients, weighted RPS summed without weight normalization. Reuse LEAD-83's Brier temperature selection on Y−2 and intercept calibration on Y−1, excluding pushes only there and in baseline conditional fitting. Calibration reweights cover/loss groups while retaining push mass. Binary scores/records exclude pushes; both RPS scores include them. Bootstrap 10,000 paired season/week-block draws, seed 20260990; gaps are OOS minus optimistic IS. These choices instantiate the declaration, with no additional variants.

## Tried
**Read:** ROADMAP.md:873 declaration: **Inferred mechanism:** late news moves mass across the frozen line; better far-tail fit need not improve that crossing. Fit a four-term exponential tilt of the discrete PMF using proper weighted RPS, fixed weights `2**(-abs(k-opener))` for home margin k; calibrate separately. Changes outcome loss, not a fifth term. **Measured input:** `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet`, 1,537 rows, plus D's four-term inputs. D: opener Brier primary, B=6/K=5, **605 looks**. **Read/checked:** LEAD-66/67/71/84/86, MOD-18; local outcome-loss fitting differs from whole-PMF grading, move units, book mixing and market-teacher regularization. Rank reason: directly addresses LEAD-71's RPS/side mismatch using existing machinery. Units: loss adapter 15–20 calls; cached replay 20–25.

**Read:** Protocol C/D, docs/lanes/ideation-2026-09-29c.md:10–19: population 2020–2025 archived openers, source-complete subsets; pushes retained for distributions/nomination, excluded only from conditional cover fitting. Enforce target-week, observation, bookmaker, kickoff clocks; no closing inputs. Reconstruct timestamp-matched moves for both arms in every year, replacing missing-archive zeros. Chronological outer 2023/2024/2025: fit through Y−3, selection Y−2, calibration Y−1. Verify upstream cutoffs and reuse LEAD-83 fixed-cutoff refits; retrospective archives require prospective freezing. Fixed ridge .001, no tuning grid. Five arms: candidate, served four-term recipe, model-only, timestamp-matched market, Elo. One calibrated line-conditioned discrete PMF selects every side. Report decisive records first, optimistic IS/OOS/gaps, fold coefficients/stability, accuracy/Brier/log loss/RPS/local RPS, five equal-width reliability bands, season/week-block 95% intervals and probability_positive. All population filters outcome-blind. Looks `(27*5+6+4)*(3+1)+25=605`; no unlisted variants. Zero crossing closes nothing; default unresolved_below_power; RPS alone cannot promote ATS sides. Shared sources: `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, opener artifact above, `data/processed/game_features_weak_stack.parquet`.

## Record commands
Pending the declared replay; exact bash-compatible commands with pool-player-English `--plain-summary` will be saved here for the orchestrator, never run by this worker.

## Next
Implement scripts/lead90_unit1.py; validate upstream and quote clocks; execute one replay, then scoped Ruff and report checks.

## Open
Source availability and certified chronology remain unmeasured. Only scripts/lead90_unit1.py, docs/lead90_*.md, this lane, and scratch outputs belong to this packet.

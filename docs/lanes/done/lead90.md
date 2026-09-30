# LEAD-90 — fit probability mass near the frozen opener

## Goal
Execute the fixed LEAD-90 protocol; historical OPENER proxies the frozen pool line. No serving, registry, publication, or Git mutations.

## State
Protocol copied before any outcome computation; source/chronology inventory in progress.

Implementation fixed before outcomes: use every game with at least one clock-valid common bookmaker; recover signed composition flags for pushes and verify against frozen non-push flags. Candidate PMF is `q(k)*exp((k-opener)*Xβ)/Z`, with the same four standardized inputs plus intercept, fitted on all training margins including pushes, ridge .001 on non-intercept coefficients, weighted RPS summed without weight normalization. Reuse LEAD-83's Brier temperature selection on Y−2 and intercept calibration on Y−1, excluding pushes only there and in baseline conditional fitting. Calibration reweights cover/loss groups while retaining push mass. Binary scores/records exclude pushes; both RPS scores include them. Bootstrap 10,000 paired season/week-block draws, seed 20260990; gaps are OOS minus optimistic IS. These choices instantiate the declaration, with no additional variants.

## Tried
**Read:** ROADMAP.md:873 declaration: **Inferred mechanism:** late news moves mass across the frozen line; better far-tail fit need not improve that crossing. Fit a four-term exponential tilt of the discrete PMF using proper weighted RPS, fixed weights `2**(-abs(k-opener))` for home margin k; calibrate separately. Changes outcome loss, not a fifth term. **Measured input:** `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet`, 1,537 rows, plus D's four-term inputs. D: opener Brier primary, B=6/K=5, **605 looks**. **Read/checked:** LEAD-66/67/71/84/86, MOD-18; local outcome-loss fitting differs from whole-PMF grading, move units, book mixing and market-teacher regularization. Rank reason: directly addresses LEAD-71's RPS/side mismatch using existing machinery. Units: loss adapter 15–20 calls; cached replay 20–25.

**Read:** Protocol C/D, docs/lanes/ideation-2026-09-29c.md:10–19: population 2020–2025 archived openers, source-complete subsets; pushes retained for distributions/nomination, excluded only from conditional cover fitting. Enforce target-week, observation, bookmaker, kickoff clocks; no closing inputs. Reconstruct timestamp-matched moves for both arms in every year, replacing missing-archive zeros. Chronological outer 2023/2024/2025: fit through Y−3, selection Y−2, calibration Y−1. Verify upstream cutoffs and reuse LEAD-83 fixed-cutoff refits; retrospective archives require prospective freezing. Fixed ridge .001, no tuning grid. Five arms: candidate, served four-term recipe, model-only, timestamp-matched market, Elo. One calibrated line-conditioned discrete PMF selects every side. Report decisive records first, optimistic IS/OOS/gaps, fold coefficients/stability, accuracy/Brier/log loss/RPS/local RPS, five equal-width reliability bands, season/week-block 95% intervals and probability_positive. All population filters outcome-blind. Looks `(27*5+6+4)*(3+1)+25=605`; no unlisted variants. Zero crossing closes nothing; default unresolved_below_power; RPS alone cannot promote ATS sides. Shared sources: `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, opener artifact above, `data/processed/game_features_weak_stack.parquet`.

**Measured result:** 697 outer games; candidate 358-339, four-term 362-335; decisive 3-7. Brier gain -0.000039 [-0.000308, 0.000262], probability_positive=0.38240; local RPS gain 0.001192 [-0.002121, 0.004479], probability_positive=0.76290. **Inferred:** unresolved_below_power; no serving change.

## Record commands
Orchestrator runs these 20 candidate-versus-four-term OOS commands serially.

```bash
while read -r panel first last games blocks metric units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead90_unit1_$panel-$metric" --league nfl \
  --description "Fitting the whole score spread near the pool line versus the usual four-term calculation: $panel $metric" --source docs/lead90_unit1.md --family lead90_unit1_605_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 605 declared looks reported." \
  --plain-summary "This teaches the pick calculation to care most about scores right around the pool line instead of far-off blowouts. The held-out comparison is a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points accuracy_points -1.71673819742 -3.94736842105 0.431034482759 0.0601
2023 2023 2023 233 18 brier brier_improvement -0.000306595845908 -0.00050334228612 -0.000102448418794 0.0021
2023 2023 2023 233 18 log_loss log_loss_improvement -0.00061393397695 -0.00100803405531 -0.000205137476738 0.0021
2023 2023 2023 233 18 rps rps_improvement -0.264810240142 -0.491446892877 -0.0580818204913 0.0046
2023 2023 2023 233 18 local_rps rps_improvement -0.00147249487991 -0.00601247181991 0.0028044335735 0.2568
2024 2024 2024 232 18 accuracy_points accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 brier brier_improvement 0 0 0 0.5
2024 2024 2024 232 18 log_loss log_loss_improvement 0 0 0 0.5
2024 2024 2024 232 18 rps rps_improvement 0.101080765726 0.00764510920701 0.197201361066 0.9833
2024 2024 2024 232 18 local_rps rps_improvement 0.00432077590327 0.000883055535884 0.00764472185378 0.9932
2025 2025 2025 232 18 accuracy_points accuracy_points 0 -1.74672489083 1.70212765957 0.50035
2025 2025 2025 232 18 brier brier_improvement 0.000191397225387 -0.000251865220854 0.000625774510886 0.8047
2025 2025 2025 232 18 log_loss log_loss_improvement 0.000394095666472 -0.000504732185815 0.00128590765245 0.8045
2025 2025 2025 232 18 rps rps_improvement 0.0239562050447 -0.026422562011 0.0795920996223 0.813
2025 2025 2025 232 18 local_rps rps_improvement 0.000736487859893 -0.000739096452159 0.00237627852287 0.8239
pooled 2023 2025 697 54 accuracy_points accuracy_points -0.573888091822 -2.01446105364 0.56657223796 0.20095
pooled 2023 2025 697 54 brier brier_improvement -3.8784326839e-05 -0.000307850125186 0.000262450268741 0.3824
pooled 2023 2025 697 54 log_loss log_loss_improvement -7.40551248321e-05 -0.000616940124831 0.000540103773051 0.3865
pooled 2023 2025 697 54 rps rps_improvement -0.0469955245875 -0.265258245433 0.108088187267 0.3636
pooled 2023 2025 697 54 local_rps rps_improvement 0.00119182914772 -0.00212077853022 0.0044789589475 0.7629
CELLS
```

## Next
Orchestrator reviews docs/lead90_unit1.md and runs the serial records.

## Open
Three outer seasons; optimistic IS; retrospective archives; missing Tuesday totals. No closure or promotion. No Git, publication or src edits.

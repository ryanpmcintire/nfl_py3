# LEAD-94: last-game tiebreaker population

## Goal
Execute the unchanged predeclared LEAD-94 protocol. No served-card or registry changes.

## State
**Measured:** one declared replay complete (717 looks); report docs/lead94_unit1.md; rows tests/scratch/codex/lead94_unit1/.
Pooled OOS last-game total MAE (54 games): candidate 10.852 vs Tuesday 11.019 (gain +0.167 [-0.259, 0.556], probability_positive 0.790), vs deadline +0.019 [-0.241, 0.241] (0.570), vs adjusted deadline +0.019 (0.553), vs median +0.333 (0.665). Last-game decisive vs Tuesday 18-14 (p 0.597).
All-game OOS MAE gain vs Tuesday -0.113 [-0.337, 0.108] (0.171); vs deadline -0.136 [-0.229, -0.050] (0.0009): the weighting hurts all games. Last-game IS-OOS gap +1.01 MAE [-2.69, 4.83].
ESS 16.0/34.5/54.9 of 197/413/629 train games; weighted b 0.5/2.0/2.0 vs unweighted 1.357. ATS sides equal four-term (0 decisive). **Inferred:** unresolved_below_power.

## Protocol (verbatim declaration)
| LEAD-94 | ⬜ | Train tiebreaker totals for the last-game population (rank 5/8) | **Inferred mechanism:** the week's final primetime matchup is selected; average post-Tuesday scoring-news response across all games may misprice its total. Retain LEAD-88's total-response equation; weight training loss by a class-prior-corrected density ratio for last-game membership from Tuesday total, absolute opener and kickoff slot, capped at 10 and fitted on training seasons only. No outcome enters weights. **Measured inputs:** `data/raw/20260908T162105Z/schedules.parquet` 4,902 rows; `data/market/raw/20200908T125500Z/quotes.parquet` 1,862 and `data/market/raw/20200913T162500Z/quotes.parquet` 2,088. D: last-game MAE primary, B=7/K=6, **717 looks**. **Read/checked:** POL-12, LEAD-54/80/88, MOD-17, SIM-04/08, tiebreaker study. Changes population weights, not rounding, another pass-through term, field assumptions or ATS predictors. Rank reason: cheap replay; report effective sample size, no pool-rank claim. Units 15–20/20–25 calls. |
Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.
Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.
One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.
Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.
For 90–97 use C's chronology, five controls/arms, reporting and look formula with F=3; historical OPENER is the frozen Splash-line proxy, never pre-2026 pool captures. Shared **measured** inputs: `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows/107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537 including pushes), `data/processed/game_features_weak_stack.parquet` (4,902). Reuse the upstream cutoff/refit machinery documented in `docs/lanes/lead83.md:14–19`; a downstream LOSO label alone does not certify upstream chronology. Fixed ridge .001, no tuning grid; selection and calibration years remain separate. Each row is an independent challenger, not an eight-way winner-selection exercise. All features/quotes are pre-deadline; historical scores/plays are training labels only. One calibrated line-conditioned PMF decides sides, with the served four-term recipe as the paired base. K=4 normally; 90/91/93/95 add local RPS/worst-season loss/tail RPS/regulation log score respectively; 92 adds nominee reward/Brier; 94 adds all-game/last-game total MAE. Tiebreaker controls follow C/LEAD-88; tiebreaker changes keep ATS sides fixed. All population filters are outcome-blind, all IS/OOS gaps and diagnostic looks reported. Zero crossing closes nothing; RPS or tiebreaker gains alone cannot promote ATS sides (AGENTS.md, Margins/Promotion). Freeze survivors for subsequent normal 2026 capture collection.

## Tried
**Measured:** .tools/uv.exe run --no-sync --no-cache python scripts/lead94_unit1.py, one job, one thread, 40 GB free; exit 0. First launch aborted in load() on a cutoff-guard bug (season-end date in January), before any outcome; fixed guard, rerun. Ruff check and format passed. No src, registry, commit or publish.

## Record commands
Orchestrator runs these 32 candidate-versus-control OOS total-MAE cells serially (4 panels x 4 controls x 2 endpoints); worker did not run them.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead94_unit1_$panel-$units" --league nfl \
  --description "Final-game total guess weighted toward final-game-like games versus the comparator: $panel" --source docs/lead94_unit1.md --family lead94_unit1_717_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no refuted mechanism or powered control. All 717 declared looks reported." \
  --plain-summary "This lets the last game of the week count more when learning how totals move after Tuesday, then guesses its combined score. Picks against the spread are unchanged; this is a research result only."
done <<'CELLS'
2023-tuesday-last_mae 2023 2023 18 18 mae_improvement 0.277777777778 -0.166666666667 0.777777777778 0.8843
2023-tuesday-all_mae 2023 2023 239 18 mae_improvement 0.112970711297 0.0207468879668 0.210300429185 0.99125
2023-deadline-last_mae 2023 2023 18 18 mae_improvement 0.0555555555556 -0.222222222222 0.333333333333 0.6434
2023-deadline-all_mae 2023 2023 239 18 mae_improvement -0.0836820083682 -0.195744680851 0.0158761904762 0.04985
2023-adjusted_deadline-last_mae 2023 2023 18 18 mae_improvement 0.0555555555556 -0.222222222222 0.333333333333 0.6453
2023-adjusted_deadline-all_mae 2023 2023 239 18 mae_improvement -0.0585774058577 -0.145228215768 0.0208333333333 0.0789
2023-median-last_mae 2023 2023 18 18 mae_improvement 1.33333333333 -0.944444444444 3.38888888889 0.884
2023-median-all_mae 2023 2023 239 18 mae_improvement 1.87447698745 0.874014947827 2.78111587983 0.9999
2024-tuesday-last_mae 2024 2024 18 18 mae_improvement 0.111111111111 -0.777777777778 1 0.5966
2024-tuesday-all_mae 2024 2024 238 18 mae_improvement -0.22268907563 -0.497992367878 0.0480585509313 0.0544
2024-deadline-last_mae 2024 2024 18 18 mae_improvement -0.111111111111 -0.555555555556 0.333333333333 0.322
2024-deadline-all_mae 2024 2024 238 18 mae_improvement -0.126050420168 -0.262717343659 0.0125013075314 0.0351
2024-adjusted_deadline-last_mae 2024 2024 18 18 mae_improvement -0.111111111111 -0.555555555556 0.333333333333 0.322
2024-adjusted_deadline-all_mae 2024 2024 238 18 mae_improvement -0.126050420168 -0.262717343659 0.0125013075314 0.0351
2024-median-last_mae 2024 2024 18 18 mae_improvement -0.388888888889 -2 1.11111111111 0.31075
2024-median-all_mae 2024 2024 238 18 mae_improvement 0.273109243697 -0.320516877637 0.828451882845 0.82245
2025-tuesday-last_mae 2025 2025 18 18 mae_improvement 0.111111111111 -0.555555555556 0.777777777778 0.61865
2025-tuesday-all_mae 2025 2025 237 18 mae_improvement -0.232067510549 -0.502092050209 0.00862256371814 0.03055
2025-deadline-last_mae 2025 2025 18 18 mae_improvement 0.111111111111 -0.277777777778 0.5 0.72075
2025-deadline-all_mae 2025 2025 237 18 mae_improvement -0.198312236287 -0.338983050847 -0.0635593220339 0.00215
2025-adjusted_deadline-last_mae 2025 2025 18 18 mae_improvement 0.111111111111 -0.333333333333 0.611111111111 0.6623
2025-adjusted_deadline-all_mae 2025 2025 237 18 mae_improvement -0.168776371308 -0.337606837607 -0.00847457627119 0.0192
2025-median-last_mae 2025 2025 18 18 mae_improvement 0.0555555555556 -1.72222222222 1.94444444444 0.5185
2025-median-all_mae 2025 2025 237 18 mae_improvement 0.324894514768 -0.0808510638298 0.758931260757 0.9352
pooled-tuesday-last_mae 2023 2025 54 54 mae_improvement 0.166666666667 -0.259259259259 0.555555555556 0.7903
pooled-tuesday-all_mae 2023 2025 714 54 mae_improvement -0.113445378151 -0.336643619286 0.108386251032 0.1711
pooled-deadline-last_mae 2023 2025 54 54 mae_improvement 0.0185185185185 -0.240740740741 0.240740740741 0.56965
pooled-deadline-all_mae 2023 2025 714 54 mae_improvement -0.135854341737 -0.229020695578 -0.0500695410292 0.0009
pooled-adjusted_deadline-last_mae 2023 2025 54 54 mae_improvement 0.0185185185185 -0.259259259259 0.277777777778 0.55295
pooled-adjusted_deadline-all_mae 2023 2025 714 54 mae_improvement -0.117647058824 -0.217573372481 -0.0334704756141 0.0015
pooled-median-last_mae 2023 2025 54 54 mae_improvement 0.333333333333 -0.925925925926 1.75925925926 0.6652
pooled-median-all_mae 2023 2025 714 54 mae_improvement 0.826330532213 0.0923262713572 1.86217908203 0.9937
CELLS
```

## Next
Orchestrator reviews docs/lead94_unit1.md and runs the record loop. Any refinement (finer weighted-median b, smoother membership) needs a new declaration.

## Open
Weights near zero for Sunday slots and capped at 10 for Monday slots give ESS 16-55; b is quantized (0.5/2.0). Three outer seasons, 2020-only first fold, retrospective archives. Zero crossing closes nothing; one fitted probability selects each ATS side, held fixed here.

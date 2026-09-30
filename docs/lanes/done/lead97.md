# LEAD-97 — integrate unavailable moves

## Goal
Execute the fixed LEAD-97 unit 1: replace the zero-move substitution for archive-unavailable late moves with an average over an earlier-season empirical move law. Historical OPENER is the frozen pool-line proxy. Research only.

## State
**Measured:** unit 1 ran once (exit 0): 697 outer games, 54 weeks, 497 looks. Candidate 357-340 equals four-term 357-340; decisive vs four-term 0 games; vs market 131-136 (267, exact p 0.807). Pooled OOS Brier improvement -1.05e-08 [-4.78e-08, +1.29e-08], probability_positive 0.160; log loss -2.1e-08, 0.163. IS/OOS accuracy 56.24/51.22, gap -5.02 (candidate and base identical). Legacy availability is 0 for all 2020-2022 and 1 for all 2023-2025, so no outer row is unavailable; candidate differs only through fitted move coefficient (2024 only, -0.33), and the 2024 calibration slope is 0 so no prediction changes. **Inferred:** unresolved_below_power; estimand untestable under the fixed chronology; no serving change.

## Protocol (fixed before outcomes)

Shared LEAD-90..97 declaration (verbatim from docs/lanes/lead92.md; row text: ROADMAP.md LEAD-97):

For 90–97 use C's chronology, five controls/arms, reporting and look formula with F=3; historical OPENER is the frozen Splash-line proxy, never pre-2026 pool captures. Shared **measured** inputs: `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows/107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537 including pushes), `data/processed/game_features_weak_stack.parquet` (4,902). Reuse the upstream cutoff/refit machinery documented in `docs/lanes/lead83.md:14–19`; a downstream LOSO label alone does not certify upstream chronology. Fixed ridge .001, no tuning grid; selection and calibration years remain separate. Each row is an independent challenger, not an eight-way winner-selection exercise. All features/quotes are pre-deadline; historical scores/plays are training labels only. One calibrated line-conditioned PMF decides sides, with the served four-term recipe as the paired base. K=4 normally; 90/91/93/95 add local RPS/worst-season loss/tail RPS/regulation log score respectively; 92 adds nominee reward/Brier; 94 adds all-game/last-game total MAE. Tiebreaker controls follow C/LEAD-88; tiebreaker changes keep ATS sides fixed. All population filters are outcome-blind, all IS/OOS gaps and diagnostic looks reported. Zero crossing closes nothing; RPS or tiebreaker gains alone cannot promote ATS sides (AGENTS.md, Margins/Promotion). Freeze survivors for subsequent normal 2026 capture collection.

C chronology: Chronological LOSO; outer 2023/2024/2025; fit through Y-3, tune on Y-2, calibrate on Y-1. Five equal-width reliability bands, season/week-block 95% intervals, probability_positive. Looks L=(27K+B+4)(F+1)+25 with F=3, K=4, B=6 gives 497.

Row-specific choices fixed now:
- Population: the LEAD-83 unit-1 source-complete frozen-opener population (1,344 games including pushes; 1,311 non-push conditional-cover rows) loaded through scripts/lead83_unit2.py machinery with its input hashes verified. Upstream margin/offset/Elo/discrete-lattice models refit at every outer cutoff through Y-3. Pushes stay in upstream distributions and are excluded only from conditional cover fitting.
- Legacy masks: `market_move_available` and `market_move_toward_home` from the served per_game.parquet are the four-term inputs (unavailable rows carry move 0 and available 0), preserved identically for every arm; four terms model_logit, composition_flag_sum, move, available. No fifth term.
- Base arm four_term: ridge .001 logistic on the four terms with legacy zero substitution (repo `_fit_logit`).
- Candidate arm integrated: same four terms, same standardization (fit-row mean/scale of the legacy columns), same ridge .001 penalty on all coefficients. Observed moves stay observed. For an unavailable row the likelihood and prediction average the logistic cover probability over the empirical move law: the discrete distribution of dated matched-book median moves toward home (LEAD-83 dated quotes) among games in fit seasons 2020..Y-3 of that fold, using every game in those seasons (pre-outcome quantity), moves standardized like the legacy move column; the available flag stays at its legacy value 0. Fitting maximizes the penalized integrated log likelihood by BFGS from the four-term solution, analytic gradient. Prediction of unavailable rows: mean of the atom probabilities. The 2020-09-08 and 2020-09-13 archives are inventoried as part of the LEAD-83 quote set and add no separate arm.
- Controls: model_only, timestamp-matched market (dated Sunday matched-book median), Elo, all sharing the same lattice code; slope selected on the tuning year Brier and intercept fitted on the calibration year, identically for all five arms.
- Estimand: information-availability (what is the pick probability if the late move is unobserved). Note: legacy availability is 0 for all 2020-2022 games and 1 for all 2023-2025 games, so every fold fits on unavailable rows only and every outer row is observed; integration can only change the fitted coefficients, not outer-row inputs.
- Contrasts: candidate minus four_term/model_only/market/elo (accuracy points; loss endpoints oriented so positive favors candidate). Reporting: IS optimistic, OOS, gap OOS minus IS, decisive records first, fold coefficients, reliability, 10,000 paired season/week-block resamples seed 20260929, fixed fitted predictions.
- Primary endpoint: opener Brier improvement versus four_term, pooled OOS. No unlisted variants.

## Tried
Protocol saved before outcomes (above; copy in tests/scratch/codex/lead97_unit1/protocol.md). `.tools/uv.exe run --no-sync --no-cache python scripts/lead97_unit1.py` ran once with PYTHONPATH=scripts; degenerate-law parity with the four-term fit held (<1e-9). Ruff check and format pass. Report docs/lead97_unit1.md; rows in tests/scratch/codex/lead97_unit1/.

## Next
Orchestrator reviews the report and runs the commands below serially. A testable version needs observed moves in fit years (fold with fit seasons including 2023+), which the fixed Y-3 chronology does not offer; any such variant is a new declared row.

## Open
Three outer seasons; retrospective archives; law is the dated matched-book median move (LEAD-83) rather than the legacy leader-median move; every outer row is observed. No closure or promotion.

## Record commands
Orchestrator alone runs these 16 candidate-versus-four-term OOS cells serially.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead97_unit1_${panel}_${units}" --league nfl   --description "Averaging over unavailable late line moves versus the usual four-term calculation: ${panel}" --source docs/lead97_unit1.md --family lead97_unit1_497_looks   --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units"   --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power   --classification-evidence "Three held-out seasons; no held-out game had an unavailable move, so the mechanism was not exercised. All 497 declared looks reported."   --plain-summary "When a late line move is missing, this averages over the moves that usually happen instead of assuming none. Every recent game had its move, so it made no difference; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points 0 0 0 0.5
2023 2023 2023 233 18 log_loss_improvement -1.0747112821e-08 -2.16425116152e-08 1.16453722946e-10 0.02595
2023 2023 2023 233 18 brier_improvement -5.37391837944e-09 -1.08203237604e-08 5.67421943769e-11 0.0259
2023 2023 2023 233 18 rps_improvement -7.68269325239e-08 -1.65345197449e-07 9.24983599102e-09 0.0395
2024 2024 2024 232 18 accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss_improvement 0 0 0 0.5
2024 2024 2024 232 18 brier_improvement 0 0 0 0.5
2024 2024 2024 232 18 rps_improvement 0 0 0 0.5
2025 2025 2025 232 18 accuracy_points 0 0 0 0.5
2025 2025 2025 232 18 log_loss_improvement -5.27801271567e-08 -2.3024303317e-07 9.60966856811e-08 0.2719
2025 2025 2025 232 18 brier_improvement -2.62090662361e-08 -1.12816718945e-07 4.65791382444e-08 0.2678
2025 2025 2025 232 18 rps_improvement -4.04872262632e-08 -1.15890388602e-06 8.38208501567e-07 0.497
pooled 2023 2025 697 54 accuracy_points 0 0 0 0.5
pooled 2023 2025 697 54 log_loss_improvement -2.11607844873e-08 -9.7283615599e-08 2.68080531549e-08 0.16335
pooled 2023 2025 697 54 brier_improvement -1.0520267359e-08 -4.78423476892e-08 1.29122157103e-08 0.16025
pooled 2023 2025 697 54 rps_improvement -3.91588404177e-08 -4.46271670163e-07 2.91289536541e-07 0.33205
CELLS
```

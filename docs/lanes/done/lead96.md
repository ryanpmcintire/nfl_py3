# LEAD-96

## Goal

Execute the fixed transport-weighted older-training-games challenger (LEAD-96 unit 1); historical OPENER is the frozen pool-line proxy; research only.

## State
**Measured:** 497-look replay complete: 697 outer games; candidate 357-340, base 361-336; decisive 18-22. Brier improvement 0.001166 [-0.000223, 0.004178], probability_positive=0.93435. **Inferred:** unresolved_below_power; no serving change.

## Protocol (fixed before outcomes)

**Row (ROADMAP LEAD-96, verbatim):** Fit a training-only domain classifier on model logit, flag sum and absolute opener for 2011-2019 versus earlier available 2020+ fit seasons. Apply class-prior-corrected density-ratio weights capped at 10 to old rows in the unchanged four-term likelihood; preserve missing-move semantics and report effective independent games. D: opener Brier primary, B=6/K=4, 497 looks. Not a fifth term.

**Implementation choices fixed before outcomes:**
- Chronology, folds, roles, upstream refits, lattices, market/Elo/model-only arms, separate slope selection on Y-2 and intercept calibration on Y-1 are identical to LEAD-91 (LEAD-83 machinery); outer 2023/2024/2025, fit rows are 2020..Y-3.
- Old rows: the 2,231 population rows with season < 2020 (model_logit from the walk-forward discrete rebuild, flag sum, market_move_toward_home = 0, market_move_available = 0). All precede every fit cutoff. Old absolute opener is |proxy_open_home_spread| from the SBR era scored file joined on game_id (every old row must match, else stop); new absolute opener is |Tuesday opener spread| of the fold panel.
- Domain classifier (nuisance specification): ridge logistic (ridge 1e-3, Newton, standardized on the pooled classifier training rows) predicting old (1) versus new (0) from model_logit, composition_flag_sum, absolute opener; new = the fold's fit-role rows only. Weight for an old row = ((1-p)/p) * (n_old/n_new), capped at 10; new rows weight 1. Unweighted-equal extended training and era slopes are not run (no unlisted variants).
- Candidate four-term fit: same four terms (model_logit, composition_flag_sum, original_move, available), ridge 1e-3 (not scaled by weights), on new rows plus weighted old rows, weighted-likelihood Newton (same iteration count as the served fit), standardization from the unweighted union of training rows. Old rows carry original_move 0 and available 0, new rows carry the panel values (available 1), preserving missing-move semantics.
- Base four_term arm: the served fit on the fold's 2020..Y-3 rows only (shared.fit_probability). Arms: candidate (weighted extended), four_term, model_only, market, elo. Contrasts: candidate minus each baseline, positive = candidate better.
- Effective independent games reported per fold: Kish (sum w)^2 / sum w^2 for old rows, for all rows, and the count at the cap.
- Metrics K=4: opener accuracy, log loss, Brier (primary), RPS. Discrete lattice from the model-only arm supplies push mass, as in LEAD-91.
- 10,000 paired season/week-block resamples, seed 20260929; IS = optimistic fitting-year (2020..Y-3) predictions, gap = OOS minus IS.
- Looks: L=(27K+B+4)(F+1)+25 with K=4, B=6, F=3 = 497.

**Shared protocol declaration (verbatim from docs/lanes/lead92.md):**

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week,
observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros.
LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze
survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and
gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing;
default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap ×
K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-
game total MAE. No unlisted variants.

For 90–97 use C's chronology, five controls/arms, reporting and look formula with F=3; historical OPENER is the frozen Splash-line proxy, never pre-2026 pool captures. Shared **measured** inputs:
`artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows/107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537 including pushes),
`data/processed/game_features_weak_stack.parquet` (4,902). Reuse the upstream cutoff/refit machinery documented in `docs/lanes/lead83.md:14–19`; a downstream LOSO label alone does not certify upstream
chronology. Fixed ridge .001, no tuning grid; selection and calibration years remain separate. Each row is an independent challenger, not an eight-way winner-selection exercise. All features/quotes
are pre-deadline; historical scores/plays are training labels only. One calibrated line-conditioned PMF decides sides, with the served four-term recipe as the paired base. K=4 normally; 90/91/93/95
add local RPS/worst-season loss/tail RPS/regulation log score respectively; 92 adds nominee reward/Brier; 94 adds all-game/last-game total MAE. Tiebreaker controls follow C/LEAD-88; tiebreaker changes
keep ATS sides fixed. All population filters are outcome-blind, all IS/OOS gaps and diagnostic looks reported. Zero crossing closes nothing; RPS or tiebreaker gains alone cannot promote ATS sides
(AGENTS.md, Margins/Promotion). Freeze survivors for subsequent normal 2026 capture collection.

## Tried
Protocol saved before outcomes (copy in tests/scratch/codex/lead96_unit1/protocol.md). Ran `.tools/uv.exe run --no-sync --no-cache python scripts/lead96_unit1.py` once. Report docs/lead96_unit1.md; rows in tests/scratch/codex/lead96_unit1/.

## Record commands
Orchestrator runs these 12 candidate-versus-four-term OOS cells serially; worker did not run them.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead96_unit1_${panel}_${units}" --league nfl \
  --description "Older seasons reweighted to look like recent ones versus the usual four-term calculation: ${panel}" --source docs/lead96_unit1.md --family lead96_unit1_497_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 497 declared looks reported." \
  --plain-summary "This lets older seasons that resemble today count in the pick calculation and older ones that do not count less. The held-out comparison remains a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points 0.858369098712 -1.65992392808 3.04347826087 0.7634
2023 2023 2023 233 18 log_loss_improvement 0.000379765059343 -4.92849316003e-06 0.000748429423848 0.9739
2023 2023 2023 233 18 brier_improvement 0.000189977709693 -2.4370787089e-06 0.000374176382639 0.9739
2023 2023 2023 233 18 rps_improvement -0.00101828729527 -0.00471388435117 0.00246214548268 0.2964
2024 2024 2024 232 18 accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss_improvement -0 -0 0 0.5
2024 2024 2024 232 18 brier_improvement -0 -0 0 0.5
2024 2024 2024 232 18 rps_improvement -0 -0 0 0.5
2025 2025 2025 232 18 accuracy_points -2.58620689655 -6.25 0.440577365405 0.06175
2025 2025 2025 232 18 log_loss_improvement 0.00735996175766 -0.00301453616822 0.0192600346645 0.9098
2025 2025 2025 232 18 brier_improvement 0.00331189816672 -0.00151949133668 0.00863251325031 0.9054
2025 2025 2025 232 18 rps_improvement 0.0902865767106 0.0337811244938 0.154269097156 0.9999
pooled 2023 2025 697 54 accuracy_points -0.573888091822 -2.93255131965 1.27659574468 0.33345
pooled 2023 2025 697 54 log_loss_improvement 0.00257675234807 -0.000432096048912 0.00924959342654 0.93605
pooled 2023 2025 697 54 brier_improvement 0.00116588978628 -0.000223237512824 0.00417837396642 0.93435
pooled 2023 2025 697 54 rps_improvement 0.0297119438408 -0.00170277868542 0.0901430420198 0.79265
CELLS
```

## Next
Orchestrator reviews the report and runs the records serially.

## Open
Three outer seasons; optimistic IS; retrospective archive reuse; no pool-noon fidelity claim. No closure or promotion.

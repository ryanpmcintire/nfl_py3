# LEAD-95

## Goal

Execute the fixed regulation-label versus overtime-settlement challenger; historical OPENER is the frozen pool-line proxy. Research only; no serving change, no promotion.

## State

**Measured:** 609-look replay complete (exit 0): 697 outer games, 54 weeks, 32 held-out overtime games; candidate 359-338, four-term 361-336. Decisive vs four-term 10-12 (22 games, exact p 0.83); vs market 130-133. OOS gain vs four-term: Brier +0.000174 [-0.000127, +0.000726] pp 0.758; log loss +0.000344 [-0.000264, +0.001457] 0.753; RPS +0.00634 [-0.0046, +0.0207] 0.840; accuracy -0.29 pts [-2.03, +0.86] 0.428; regulation log score +0.109 [+0.052, +0.180] 1.0 (asymmetric lattices, disclosed). Candidate IS/OOS accuracy 58.15/51.51, gap -6.64 pts. **Inferred:** unresolved_below_power; no serving change (RPS/diagnostic gains cannot promote ATS sides).

## Protocol (fixed before outcomes)

### Fixed roadmap row

| LEAD-95 | ⬜ | Separate regulation labels from overtime settlement (rank 6/8) | **Inferred mechanism:** final-margin training can misread a regulation tie plus sudden-death score as persistent strength, obscuring real post-freeze information. Fit the same four-term lattice to regulation margin; attach an earlier-season empirical overtime increment kernel only to regulation-tie mass, then calibrate and grade FINAL opener outcomes. Keep all games; unseen rules regimes inherit the declared pooled earlier kernel, with that assumption disclosed. No overtime indicator enters pregame inputs. **Measured input:** `data/pbp/raw/20260925T202544Z/season=2020/plays.parquet`, 47,705 rows; six 2020–2025 files total 294,989 with quarter/score fields. D: opener Brier primary, B=7/K=5, **609 looks**. **Read/checked:** SIM-04/08, MOD-17/21, LEAD-66/76: changes training labels, not the simulator or older rule-shape prior; no fifth term. Rank reason: variance-reduction mechanism, but rare overtime and reconstruction cost. Units 20–25/25–30 calls. |

### Shared protocol text (verbatim, LEAD-90..97)

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

### Implementation choices fixed before outcomes (LEAD-95 specific)

- Regulation label: a game is a regulation tie if any play has qtr == 5 in data/pbp/raw/20260925T202544Z/season=YYYY/plays.parquet (2020-2025); regulation margin = 0 for those games, final result otherwise. Every OT game must be in the source-complete population or the run aborts.
- Candidate: margin ridge (alpha 10, weak_stack, market_residual target), home-side offsets, and the discrete integer lattice are all refit at each outer cutoff on training rows (fit through Y-3) with result and ats_margin replaced by the regulation versions. Elo arm, tuning, calibration and scoring use FINAL results and the FINAL cover label. Candidate raw PMF = regulation lattice with the mass at margin 0 redistributed by the overtime kernel; that final PMF gives the raw cover probability that enters the same four-term fit (model_logit, composition_flag_sum, original_move, available; ridge .001), slope selection on Y-2 Brier, intercept calibration on Y-1.
- Overtime kernel: from training-season (fit through Y-3) regulation-tie games only; fixed support of absolute final margins {0,1,2,3,6,7,8} (amended before any score: the inventory abort found 2025 overtime finals of +-1, absent from the earlier seasons), add-0.5 smoothing to each class, nonzero classes split half to each sign (sign-symmetric); no rules-regime split (2025 regular-season overtime rule change inherits the pooled earlier kernel; disclosed assumption). No overtime indicator enters any pregame input.
- Arms: candidate, four_term (final-margin base), model_only, market, elo. Contrasts: candidate minus each of the four baselines, oriented positive = candidate better.
- Endpoints (K=5): accuracy points, log loss, Brier, RPS (all on final outcome and final cover label) plus regulation log score: mean of -log(max(raw lattice mass at the realized regulation margin, 1e-4)); candidate uses its pre-kernel regulation lattice, comparators their final-margin lattice (four_term shares model_only). Asymmetry disclosed; uncalibrated raw lattices.
- Looks: F=3, K=5, B=7 (candidate, regulation-margin nuisance fit, overtime kernel, four baselines): L=(27*5+7+4)*4+25 = 609. Bootstrap: 10,000 paired season/week-block resamples, seed 20260929, fixed fitted predictions. IS = optimistic fit-year predictions; gap = OOS minus IS.
- Reuse: scripts/lead83_unit2.py machinery via scripts/lead91_unit1.py structure (hash-verified upstream inputs and quote archives, LEAD-83 cutoff checks).

## Tried

First run aborted in the inventory (before any model fit or score): 3 games in 2025 ended +-1 in overtime, outside the first kernel support; support amended (see protocol) and rerun once (exit 0). Ruff check and format pass. Report docs/lead95_unit1.md; rows in tests/scratch/codex/lead95_unit1/.

## Next

Orchestrator reviews docs/lead95_unit1.md, runs the record commands serially, assigns the next unit.

## Open

Three outer seasons, optimistic IS, retrospective archives; 2023 kernel from 7 training overtime games (2020); 2025 overtime rule change not modelled. Regulation-score comparison is candidate regulation lattice vs comparator final lattices. No closure or promotion.

## Record commands

Orchestrator alone runs these 20 candidate-versus-four-term OOS cells serially.

```bash
while read -r panel first last games blocks label units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead95_unit1_${panel}_${label}" --league nfl \
  --description "Regulation-margin fit with overtime settlement versus the usual four-term calculation: ${panel} ${label}" --source docs/lead95_unit1.md --family lead95_unit1_609_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 609 declared looks reported." \
  --plain-summary "This trains the pick calculation on the score at the end of regulation and treats overtime as a separate coin flip on tied games. The held-out comparison stays a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points accuracy_points -1.28755364807 -4.70085470085 2.13675213675 0.2322
2023 2023 2023 233 18 log_loss log_loss_improvement -3.27844502123e-05 -0.000530139301349 0.000416390488676 0.4702
2023 2023 2023 233 18 brier brier_improvement -1.61078376114e-05 -0.000264225910398 0.000208053302213 0.4711
2023 2023 2023 233 18 rps rps_improvement -0.00161799089093 -0.0126245282584 0.00961992899573 0.3902
2023 2023 2023 233 18 regulation_log_score log_loss_improvement 0.0815312298652 0.0312995641662 0.135292267365 0.9997
2024 2024 2024 232 18 accuracy_points accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss log_loss_improvement -0 -0 0 0.5
2024 2024 2024 232 18 brier brier_improvement -0 -0 0 0.5
2024 2024 2024 232 18 rps rps_improvement 0.00699440262574 -0.00619389067503 0.02108739463 0.8441
2024 2024 2024 232 18 regulation_log_score log_loss_improvement 0.158256022705 0.0574990672918 0.268251598875 0.9995
2025 2025 2025 232 18 accuracy_points accuracy_points 0.431034482759 0 1.31578947368 0.82105
2025 2025 2025 232 18 log_loss log_loss_improvement 0.00106544849016 -0.000876663867944 0.00310625002998 0.855
2025 2025 2025 232 18 brier brier_improvement 0.000539867060056 -0.000419373269161 0.00154102213021 0.8619
2025 2025 2025 232 18 rps rps_improvement 0.0136675572514 -0.00995266221057 0.0414527290233 0.8501
2025 2025 2025 232 18 regulation_log_score log_loss_improvement 0.0878946349229 0.0001100234284 0.202683028444 0.9752
pooled 2023 2025 697 54 accuracy_points accuracy_points -0.286944045911 -2.03193033382 0.860832137733 0.4283
pooled 2023 2025 697 54 log_loss log_loss_improvement 0.000343680448806 -0.000264339857128 0.00145687470087 0.75295
pooled 2023 2025 697 54 brier brier_improvement 0.00017431281459 -0.000126531156662 0.000726056111247 0.75755
pooled 2023 2025 697 54 rps rps_improvement 0.00633656070863 -0.00460845118064 0.0207411678532 0.8403
pooled 2023 2025 697 54 regulation_log_score log_loss_improvement 0.109187559725 0.0518163032268 0.179592633066 1
CELLS
```

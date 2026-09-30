# LEAD-92

## Goal

Execute the declared within-week Best Pick ordering study; research only.

## State

**Measured:** unit 1 ran once (exit 0): 697 held-out games, 54 weeks, 713 looks. Pairwise 362-335, four-term 361-336. Decisive vs four-term 3-2 (5 games, exact p 1.0); vs market 130-130. Nominee reward -0.0093 pts/week [-0.037, 0], probability_positive 0.240; nominee Brier -0.00141 [-0.00531, +0.00020], 0.215; Brier -0.000012 [-0.000144, +0.000124], 0.374 (all candidate minus base, positive = better). IS/OOS accuracy 57.32/51.94, gap -5.38. Sides nearly identical to base (slopes 0.008/0/0.46 by fold). **Inferred:** unresolved_below_power; no serving change.

## Protocol declaration (verbatim text, wrapped)

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

| LEAD-92 | ⬜ | Learn within-week Best Pick ordering directly (rank 3/8) | **Inferred mechanism:** several stale pool lines can offer similar chances, while all-game loss underweights choosing the
week's bonus candidate. Fit the same four-term score with pairwise logistic loss on within-week discordant cover labels, both team-side orientations, excluding same-game pairs; normalize to one weight
per week. Separate-year calibration produces one home probability/PMF that selects sides and legal nominee. **Measured input:** `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, 1,503
rows/107 weeks. D: nominee reward primary, B=6/K=6, **713 looks**. **Read/checked:** POL-09, LEAD-53/70/72/80/85/89, `docs/confidence_top_calibration.md:5–9`; ranking-loss training differs from
temperature, push arithmetic, uncertainty integration and timing. Not a fifth term. Rank reason: bonus-specific target, cheap pairs; pairs do not multiply independent games. Units 15–20/20–25 calls. |

## Implementation details fixed before outcomes
Use source-complete rows from the frozen opener population, requiring a dated move,
a pre-deadline Sunday market quote, and an archived Tuesday total. No minimum book-count filter.
Reconstruct moves using LEAD-73 and the existing Sunday-move function; reuse audited
composition flags including pushes. Refit margin, line-conditioned integer lattice,
home offsets, four-term comparator and Elo at each outer cutoff using LEAD-83 machinery.
Pair loss is the sum of within-week mean discordant-pair logistic losses plus
ridge .001/2 times squared coefficients, both orientations, different games only.
Keep the existing positive temperature selection on the tuning year's Brier score and
intercept calibration on the following year, identically for all five arms.
Nomination uses the existing Sunday eligibility, dispersion pool and deterministic
tie rule, expected win plus half a push; final unresolved ties average legal nominations.
Nominee reward is mean points per week; nominee Brier is conditional on a non-push.
Use 10,000 paired season/week-block resamples, seed 20260929, fixed fitted predictions;
IS means optimistic fitting-year predictions, gap is OOS minus IS. Positive contrasts
mean challenger improvements. All four baselines receive the same source-complete games.
The sixth specification counted in B is the upstream margin/discrete nuisance fit.

## Tried

Protocol saved before outcomes (above; copies docs/lead92_protocol.md, scratch protocol.md). Ran `.tools/uv.exe run --no-sync --no-cache python scripts/lead92_unit1.py`; first attempt aborted on NaN push-row flags before any score, fixed by filling the 33 push-row flags from the audited lead85 unit-1 frame (parity 1.0 on nonpush), then one full run. Ruff check/format pass. Report docs/lead92_unit1.md; rows in tests/scratch/codex/lead92_unit1/.

## Next

Orchestrator reviews report, runs the commands below serially, assigns next unit.

## Open

Three outer seasons, optimistic IS, retrospective archives. Pairwise raw logit has no intercept (pairs cannot identify it); calibration supplies it. No closure or promotion.

## Record commands

Orchestrator alone runs these 24 candidate-versus-four-term OOS cells serially.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead92_unit1_${panel}_${units}" --league nfl \
  --description "Best Pick ordering fit versus the usual four-term calculation: ${panel}" --source docs/lead92_unit1.md --family lead92_unit1_713_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "${units/nominee_brier_improvement/brier_improvement}" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 713 declared looks reported." \
  --plain-summary "This teaches the pick calculation to rank the week's games against each other, the way the bonus pick works. The held-out comparison stays a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points 0.429184549356 -0.892857142857 1.72413793103 0.71665
2023 2023 2023 233 18 log_loss_improvement -2.66709129713e-05 -0.000115219178069 6.9093149542e-05 0.2782
2023 2023 2023 233 18 brier_improvement -1.33572538364e-05 -5.75896682096e-05 3.44857497169e-05 0.2776
2023 2023 2023 233 18 rps_improvement 0.00110596696045 0.000191948601016 0.00210785461422 0.9934
2023 2023 2023 233 18 pool_points_per_week 0 0 0 0.5
2023 2023 2023 233 18 nominee_brier_improvement 2.05749299764e-05 -0.000191126991015 0.000236399776279 0.5684
2024 2024 2024 232 18 accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss_improvement -0 -0 0 0.5
2024 2024 2024 232 18 brier_improvement -0 -0 0 0.5
2024 2024 2024 232 18 rps_improvement -0 -0 0 0.5
2024 2024 2024 232 18 pool_points_per_week 0 0 0 0.5
2024 2024 2024 232 18 nominee_brier_improvement -0 -0 0 0.5
2025 2025 2025 232 18 accuracy_points 0 -1.29310344828 1.2987012987 0.4994
2025 2025 2025 232 18 log_loss_improvement -4.11532358817e-05 -0.000783182898111 0.000784049890724 0.4456
2025 2025 2025 232 18 brier_improvement -2.33649965378e-05 -0.000376162773204 0.000365568529551 0.4367
2025 2025 2025 232 18 rps_improvement -0.00257321467337 -0.00729446487043 0.00282894256162 0.1559
2025 2025 2025 232 18 pool_points_per_week -0.0277777777778 -0.0833333333333 0 0.1806
2025 2025 2025 232 18 nominee_brier_improvement -0.00408609851276 -0.0132816749496 0.00105451910854 0.0938
pooled 2023 2025 697 54 accuracy_points 0.143472022956 -0.441176470588 0.860832137733 0.65305
pooled 2023 2025 697 54 log_loss_improvement -2.261387869e-05 -0.000299395115779 0.00026909096707 0.37975
pooled 2023 2025 697 54 brier_improvement -1.22423519953e-05 -0.000144366709961 0.000124160091312 0.37435
pooled 2023 2025 697 54 rps_improvement -0.000486794121146 -0.00332846602366 0.00132538493595 0.42935
pooled 2023 2025 697 54 pool_points_per_week -0.00925925925926 -0.037037037037 0 0.2399
pooled 2023 2025 697 54 nominee_brier_improvement -0.0014091231005 -0.00531486727095 0.000201165008128 0.21525
CELLS
```

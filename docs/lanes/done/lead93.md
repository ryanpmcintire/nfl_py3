# LEAD-93

## Goal
Execute LEAD-93 unit 1: does borrowing college residual shape (not college picks) improve the NFL discrete PMF versus the served four-term recipe. Research only.

## State
**Measured:** 609-look replay complete: 697 outer games; candidate 361-336, base 361-336; decisive 3-3. Brier improvement -0.000019 [-0.000119, 0.000051], probability_positive=0.37045; tail RPS improvement -0.013600 [-0.022237, -0.002748], probability_positive=0.00000. **Inferred:** unresolved_below_power; no serving change.

## Protocol (fixed before outcomes)

**Row (ROADMAP.md, read):**

| LEAD-93 | ⬜ | Borrow college noise shape without borrowing its picks (rank 4/8) | **Inferred mechanism:** a frozen spread supplies location but not uncertainty converting late movement into cover mass. Pool absolute opener residuals after league-specific training-median centering/IQR scaling; fit one shared Student-t shape for the discretized background, retaining NFL-only centre, sign, scale and key-number atoms. Shape affects central cover mass as well as tails; tail RPS uses the fixed region beyond 1.5 training IQRs. Changes the PMF nuisance population, not a fifth term or college directional forecast. **Measured witnesses:** `data/cfb/lines/raw/20260816T143907Z/season=2023/lines.parquet` 14,943 rows; `data/cfb/schedules/raw/20260816T162105Z/season=2023/schedules.parquet` 3,734; roots contain 20/25 season files. D uses earlier seasons only: opener Brier primary, B=7/K=5, **609 looks**. **Read/checked:** XLG-09, MOD-20, LEAD-66/71/76/77/87. Rank reason: independent-game multiplier at fixed specification count; shape transport is an assumption distinct from refuted mean-error transfer. Units 15–20/25–30 calls. |

**Shared protocol for LEAD-90..97 (verbatim from docs/lanes/lead92.md):**

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

Implementation choices fixed before any outcome is computed.

Population and folds: the LEAD-83 source-complete frozen-opener population (same rows, roles, upstream refits, Elo, four-term recipe, temperature and intercept calibration as LEAD-91); five arms are candidate (shape), four_term, model_only, market, elo. The four baselines are unchanged from LEAD-91; only the candidate changes its PMF nuisance shape. B=7: candidate, shared-shape fit, NFL-only reference-shape fit, four baselines. K=5: accuracy, log loss, Brier (primary), RPS, tail RPS. Looks (27*5+7+4)*(3+1)+25=609.

College input: data/cfb lines (roots 20260816T143907Z) and schedules (roots 20260816T162105Z) read through nfl_ats.cfb_features (FBS regular completed games; home-oriented median opening spread, spread_open, per game via build_cfb_market_table, orientation repair included). College games need a home-oriented opener; residual = home margin minus spread_open. Outer fold Y uses college seasons <= Y-3 only (same cutoff as NFL fit); no college data after the cutoff, no college directional forecast.

Shape: per league, training-median centre m and IQR scale i of the signed residual (NFL: fit-role games including pushes, residual = result - opener; college: all college training games with an opener). Absolute standardized residual |r-m|/i; rounded to unit integer bins on the raw residual scale (bin [|c|-0.5,|c|+0.5], floor 0) so both leagues are discretized on the same raw grid. Half Student-t likelihood: probability of a bin = 2*(F(b)-F(a)), t with df nu in [1,200] and scale s, ML by Nelder-Mead on (log nu, log s) with nu clipped to the bounds. Shared fit pools NFL and college rows with equal per-game weight; NFL-only fit is the reference. No other variants.

Candidate PMF: take the existing tilted lattice mass for each NFL row (model_only point, key-number atoms, NFL centre m_N and scale i_N retained) and multiply each margin k by w(k)=[F_shared(b_k)-F_shared(a_k)]/[F_nfl(b_k)-F_nfl(a_k)] where (a_k,b_k) are the signed standardized bin edges (k-line-m_N-0.5, k-line-m_N+0.5)/i_N; renormalize the row to 1. Shape therefore moves central cover/loss/push mass as well as tails; sign is retained by the signed bins. Candidate raw cover probability is the row's cover/(cover+loss) mass; its logit replaces model_logit as the first of the same four terms, and the same ridge fit, slope selection on Y-2 Brier and intercept calibration on Y-1 apply. The calibrated PMF rescales cover/loss groups retaining push mass, exactly as LEAD-83 metrics. The four_term baseline keeps the original model_logit.

Tail RPS: the RPS sum over margin thresholds t on the +-100 grid restricted to |t-line-m_N| > 1.5*i_N (fold-fixed NFL training median and IQR); tail RPS uses the same calibrated PMF as ordinary RPS.

Reporting: 10,000 paired season/week-block resamples seed LEAD-83's; IS optimistic; gap OOS minus IS; positive contrasts favour the candidate; decisive-game record first; five reliability bands; shared and NFL-only Student-t parameters, m, i per fold reported. No pool or tuning grid; no closing inputs.


## Tried
Protocol saved before outcomes. Ran `.tools/uv.exe run --no-sync --no-cache python scripts/lead93_unit1.py` once, single thread. Report docs/lead93_unit1.md; rows in tests/scratch/codex/lead93_unit1/.

## Next
Orchestrator reviews the report, runs the record commands serially, assigns next.

## Open
Three outer seasons, optimistic IS, retrospective archives; college opener is the source opening_lines with sparse timestamps. No closure or promotion.

## Record commands
Orchestrator alone runs these 20 candidate-versus-four-term OOS cells serially. Other contrasts and IS/gap looks stay reported.

```bash
while read -r panel first last games blocks metric units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead93_unit1_${panel}_${metric}" --league nfl \
  --description "College-shaped uncertainty around the pool line versus the usual four-term calculation: ${panel} ${metric}" --source docs/lead93_unit1.md --family lead93_unit1_609_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "$units" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 609 declared looks reported." \
  --plain-summary "This borrows how spread misses are shaped in college football to set how wide the pick calculation's uncertainty is, without using any college picks. The held-out comparison is a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points accuracy_points 0 -2.18340611354 2.5641025641 0.48105
2023 2023 2023 233 18 log_loss log_loss_improvement 1.25088237711e-05 -0.000100658551201 0.000119998207559 0.6022
2023 2023 2023 233 18 brier brier_improvement 6.2848734384e-06 -5.01990686502e-05 5.99600630813e-05 0.6021
2023 2023 2023 233 18 rps rps_improvement 0.000311899245983 -0.0366535048257 0.0344576739283 0.5149
2023 2023 2023 233 18 tail_rps rps_improvement -0.0180107419325 -0.0274062264658 -0.00957888669759 0
2024 2024 2024 232 18 accuracy_points accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss log_loss_improvement -0 -0 0 0.5
2024 2024 2024 232 18 brier brier_improvement -0 -0 0 0.5
2024 2024 2024 232 18 rps rps_improvement -0.0518580143351 -0.0724216477946 -0.0315266232768 0
2024 2024 2024 232 18 tail_rps rps_improvement -0.0201382926025 -0.0266861692555 -0.0138418043201 0
2025 2025 2025 232 18 accuracy_points accuracy_points 0 0 0 0.5
2025 2025 2025 232 18 log_loss log_loss_improvement -0.000126231935562 -0.000583410491739 0.000310174587247 0.2919
2025 2025 2025 232 18 brier brier_improvement -6.21744941012e-05 -0.000284331944708 0.00014886240329 0.2881
2025 2025 2025 232 18 rps rps_improvement -0.00659611635152 -0.0159148892526 0.00245553552247 0.076
2025 2025 2025 232 18 tail_rps rps_improvement -0.00263275807028 -0.00355395352714 -0.00180577478676 0
pooled 2023 2025 697 54 accuracy_points accuracy_points 0 -0.860832137733 0.870827285922 0.48375
pooled 2023 2025 697 54 log_loss log_loss_improvement -3.78353703181e-05 -0.00024467226811 0.000105296489169 0.37175
pooled 2023 2025 697 54 brier brier_improvement -1.85941278628e-05 -0.000118759484509 5.13063434604e-05 0.37045
pooled 2023 2025 697 54 rps rps_improvement -0.0193524903802 -0.0498506825355 0.00843051206064 0.1016
pooled 2023 2025 697 54 tail_rps rps_improvement -0.0136002677566 -0.0222369162414 -0.00274756342712 0
CELLS
```

# move-baseline-guard

## Goal
Test two cross-book baseline guards for the served leader-median move term (defect: each leader book's own last pre-Wednesday quote is its baseline, no cross-book check; Week 8 LV at NYJ mybookieag baseline -1.0 vs consensus +1.0/+1.5) against the served term, leave-one-season-out, before any serving decision.

## State
DECLARED BEFORE OUTCOMES (2026-11-02). Scratch: C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/352ecbdc-246f-44c0-9f81-c287a6ef4436/scratchpad/agentMG (tab.py builds per-book baseline/final tables with the production filter code; reproduces 795/799 frozen 2023-25 moves, 104/104 of 2026 live moves).
- Per-book baseline = latest quote before the Wednesday anchor; final = latest quote before the cutoff (Sunday 12:45 ET for 2023-25 hourly cache; pick deadline for 2026). Served move = median over eligible leader books of final - baseline.
- Variant 1 (median baseline): move = median(leader finals) - median(leader baselines), leader books only.
- Variant 2 (outlier drop): reference R = median of baselines over all eligible weighted books; drop a leader book whose |baseline - R| > d; keep all if every leader would be dropped; move = median of kept per-book moves. d chosen per outer fold from training seasons only: candidates are the distinct training deviations (rounded to 0.25) plus no-guard, scored by inner leave-one-season-out log loss over training seasons, ties to the larger d.
- The 4 games that do not reproduce keep the incumbent move in every arm. Unexposed games stay 0.
- Looks: 2 variants, 1 chosen distance per fold (family of candidate distances counted), plus 2026 live replay. Fit and scoring harness copied from agentB/an.py (LOSO 833-670 reproduces production). Blocks for intervals: season-week. Zero crossing closes nothing.

## Tried
Ran an.py (scratchpad agentMG; out.txt). Served LOSO 833-670 reproduced exactly (pool_hook column set to 0 so the staged five-feature fit equals the four-term fit).
- V1 median baseline: 44 moves change (all 2023-25). LOSO 831-672 vs 833-670; log loss .68751 vs .68719; Brier .24720 vs .24705; dLL (served minus variant) -0.000323 [-0.000882, +0.000200], probability_positive(variant better) 0.115. Disagreement 10 games: served 6-4, variant 4-6. In-sample/OOS ll gap .00144 (served .00140).
- V2 outlier drop: chosen d per outer fold 2020-25 = inf, inf, inf, 0.0, inf, inf (inner LOSO). 35 moves change in LOSO; full-data choice is inf (0 changes). LOSO 831-672; ll .68746; dLL -0.000266 [-0.000710, +0.000131], P 0.10; disagreement 6 games served 4-2, variant 2-4. Gap .00167.
- Coefficients per fold stable (move .17-.25 all arms).
- 2026 Wk1-8 live (104 games, 89 with move): V1 changes 10 moves, 2 sides (NO_BAL wk2 served home .501 to .475, home missed; BUF_LV wk6 same shift, home covered) = 1-1 each; V2 changes 0 moves. LV_NYJ: only bovada and mybookieag eligible; V1 (median of 2) equals served 1.75; V2 reference of two baselines gives both deviation 1.0, so no guard. Neither variant fixes the defect game.
- Recorded two cells, family move_baseline_guard, unresolved_below_power.

## Next
Reading: not serve either; neither beats served on the opener grade. A fix for thin-book games needs a different reference (more books pre-Wednesday, or require leader_books>=3 as a fitted availability term) declared first.

## Open
Two-leader games cannot identify an outlier from leader books alone; variant 2 uses all weighted books for the reference, variant 1 does not.

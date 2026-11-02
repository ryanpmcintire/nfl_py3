# smooth-base-challenger-2026-11-02

## Goal
Test the AGENTS.md multimodal rule: does a four-term stack (policy four_term_pick_probability_v1) with model_logit from the SMOOTH base beat the DISCRETE base on the opener grade, paired, game by game.

## State
Done. Scripts and outputs in scratchpad agentB/ (build26.py, an.py, results.json, out.txt). Harness reproduces production LOSO exactly (833-670, max prob diff 0.0).
- 2020-25 LOSO (n=1503): discrete 833-670 LL .68719 Brier .24705; smooth 822-681 LL .68770 Brier .24731. dLL smooth improvement -0.00051 [-0.00146,+0.00045], P(smooth better)=0.146. Disagreements 103: discrete 57-46, smooth 46-57. In-sample LL disc .68579 / smooth .68620 (gap to OOS ~.0014 / .0015).
- 2026 W2-8 forward (n=104, feeds carrying both reads), fit 2020-25: served-extras variant disc 59-45 LL .6810, smooth 61-43 LL .6842; improvement -0.0033 [-0.0073,-0.0004], P=0.008; 14 disagreements disc 6-8, smooth 8-6. Chained weekly: disc 60-44, smooth 62-42, same dLL, P=0.012.
- model_logit coef per fold: disc mean .309 sd .063; smooth .244 sd .078 (2021 low: .217 / .117).
- Recorded: weak-signals smooth_base_vs_discrete_four_term_loso and _fwd (unresolved_below_power).

## Tried
Recomputed 2026 flags/move via production functions: flags all 0 and sides only 75% match the feed, so the served-extras variant (extras backed out of feed final_probability, reproduces served side 100%) is primary.

## Next
Discrete stays served; smooth is not shown to beat it on log loss. Accuracy points favour smooth only in 2026 (+2 games, P(acc>0)=0.73) and the reverse in 2020-25.

## Open
Four looks plus two chained/recomputed variants. 16 of 120 played 2026 games lack both reads in older feeds.

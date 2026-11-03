# MOD-10 unit 1: graph (Massey ridge) rating as a base-model input

## Goal
ROADMAP MOD-10 (Graph model). Add one input to the base model via the add-one harness: home-minus-away ridge/Massey rating (game margin, and EPA/play margin), exponential recency decay, home field, prior games only. Opener grade, LOSO 2020-25 plus 2014-19 proxy.

## State
DONE 2026-11-03 (18:00-18:20). scripts/mod10_graph_rating.py (modes true|proxy|true_control|proxy_control; kinds margin|epa). Grid declared before looking: tau 60/120/240/480/960 days, lambda 1/4/16/64/256, chosen per outer fold by mean held-out MSE over seasons 2011..s-1. Outputs artifacts/mod10/unit1/. 4 cells recorded (family mod10_graph_rating_base_input, unresolved_below_power).

## Tried
Baseline reproduced 802-701. 8 runs (4 arms + 4 permuted controls).
- 2020-25 margin: 800-703, flips 2-4, d_acc -0.13 [-0.28,0.00], coef ~0 and sign unstable, R2 on existing features 0.93 (duplicate).
- 2020-25 EPA: 805-698, flips 27-24, d_acc +0.20 [-0.61,1.14] P+ .60, ll gain .00054 (control .00018), coef -4.9..-7.4 per unit (stable negative), R2 0.92.
- 2014-19 margin: 769-718 = base, flips 8-8. 2014-19 EPA: 779-708 vs 769-718, flips 29-19, d_acc +0.67 [-0.07,1.48] P+ .94, ll .00042 P+ .92.
- Chosen params tau 240d lambda 4 in nearly every fold.

## Next
EPA rating coefficient is negative (corrects overshoot of existing EPA state): test it as a replacement for the diff_off/def_epa rolling features, not an add-on.

## Open
8 looks (4 arms + 4 controls); grid selection inside folds is not a look.

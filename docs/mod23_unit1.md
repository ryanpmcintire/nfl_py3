# MOD-23 unit 1: nested family and alpha selection on the base model

Script: `scripts/mod23_unit1.py`; outputs `artifacts/mod23_unit1/` (selection.json, report.json). All labels **measured** this session.

## Reproduction
The script reuses `clv.opener_pick_evaluation` with the served config (ridge, weak_stack, gaussian_median, discrete policy, home-side offset). Result 802-701 on 1,503 decisive games, row-identical to `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet`. The market family (2 columns) is pinned; the 11 other families are droppable (the served market-residual set has 90 columns).

## Declared before scoring
- Alpha grid 1, 3, 10, 30, 100. Inner criterion: mean squared error of the ats margin, fitting once per inner season on all earlier seasons, on the 4 seasons before Y.
- Greedy backward elimination, with alpha chosen jointly by inner MSE; a family is dropped only while inner MSE falls by more than 0.05% relative.
- Alpha-only arm keeps all families. Selection used only seasons before Y; season Y was then scored by the production weekly walk-forward.
- Each distinct chosen configuration was run through the full walk-forward and season Y rows taken (its home-side offset stream therefore uses that configuration's own earlier predictions).

## Results (decisive games, opener, probability rule)
| arm | record | accuracy | paired diff vs 802-701 | 95% season-week block bootstrap | probability_positive |
|---|---|---|---|---|---|
| baseline | 802-701 | 53.36% | | | |
| families + alpha | 786-717 | 52.30% | -1.06 pts | [-3.18, 0.98] | 0.15 |
| alpha only | 784-719 | 52.16% | -1.20 pts | [-2.17, -0.13] | 0.01 |

Log loss: baseline 0.6969, families+alpha 0.6928, alpha-only 0.6970, market-even 0.6931. Brier: 0.2517, 0.2498, 0.2518, even 0.25. Families+alpha is the only arm better than market-even on log loss, while worse on the pick record.

Per-season accuracy difference (pts), families+alpha: 2020 +0.5, 2021 0.0, 2022 -4.0, 2023 +3.4, 2024 -4.1, 2025 -1.9. Alpha-only: -2.7, -0.4, -1.2, -0.4, -1.5, -1.1.

## Chosen per season (greedy)
Dropped families: 2020 nine, 2021 eight, 2022 eight, 2023 eight, 2024 five, 2025 five. Context, injuries and offense were dropped in all six seasons, player_qb in five, bias and elo in four. Alpha chosen 100, 3, 1, 1, 10, 100; unstable. Alpha-only chose 100, 1, 1, 1, 3, 100.

## In-sample vs out-of-sample (MSE of the margin residual, gain over the fixed model)
Greedy inner gain 7.8, 6.7, 4.6, 4.1, 2.8, 3.5 versus outer gain 1.9, 2.2, 2.0, 1.1, 2.2, -0.1: selection gain shrinks by roughly 60 to 100 percent out of sample but stays positive in five of six seasons. Alpha-only inner gain 0.1 to 0.55, outer gain -0.94 to +0.82.

## Looks
1,810 inner (family set, alpha) evaluations for greedy, 30 for alpha-only, plus 2 outer arms; 9 distinct full walk-forward runs.

## Reading
Smaller models fit the margin better out of sample (MSE) and calibrate better than the fixed model, yet pick fewer covers; the pick record does not follow the fit gain. Neither arm beats the baseline at the opener; nothing here is a refuted mechanism, so recorded as unresolved_below_power (`mod23_unit1_nested_family_alpha_selection`, `mod23_unit1_nested_alpha_only`).

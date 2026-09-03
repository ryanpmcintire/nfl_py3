# NFL ATS predictions: 2026 Week 1

Published from synchronized model `123d60be8c80a35d` at `2026-09-03T16:00:59.053362+00:00`.

> **Early, mutable research preview.** Lines, injuries, depth charts, and model inputs may change before kickoff. Regenerate and republish this card as the week approaches.

Active model: `market_residual` with `weak_stack` features (`123d60be8c80a35d`). Its distinct close-graded chronological 2018-2025 evaluation classified **1,084 of 2,075 non-push games correctly (52.24%)**. The week-blocked 95% interval was 50.26%-54.30%. The model baseline is the separate opener-graded probability rule documented in `docs/opener_evaluation.md`.

**Production policy active:** the frozen four-member policy evaluates coach fade, division revenge, player arrests, and the spread-gap zone independently against the raw model pick, then flips once when any member fires. This week it changed 3 picks; policy `overlay_union_coach_division_revenge_player_arrests_spread_gap_v1` (`bbdd60a171238654`). Its 55.42% archive score is the best of 127 correlated subsets scored on the very games that chose it, so it is a ceiling and never an expectation. The de-inflated planning estimate for the played card is ≈55%: four real out-of-sample split-half selections average +1.30 accuracy points, and shrinking the +2.06-point archive gain by the measured 0.59-0.64 selection-shrinkage factor lands in the same place. A separate leave-one-season-out re-check of the selection step itself measured 0.00 points, so treat the estimate as an upper-middle read, not a floor. Paired prospective tracking against the prior coach-to-arrests chain begins at the Week 1 lock. Members: coach_fade, division_revenge_tilt, player_arrests_back_side_policy, spread_gap_zone_fade. See docs/overlay_subset_holdout_v2.md.

**Best Pick of the week (★):** MIA +3.5 in MIA at LV. The pool scores one Best Pick per regular-season week. This pick was nominated by calibrated probability among low-disagreement games.

| Date        | Matchup    | ATS prediction   | Decision score   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | SEA -3.5         | 51.6%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 53.8%            |
| Sun, Sep 13 | ARI at LAC | ARI +10.5        | 65.3%            |
| Sun, Sep 13 | ATL at PIT | ATL +3           | 54.2%            |
| Sun, Sep 13 | BAL at IND | IND +3.5         | 53.7%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 56.0%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 53.4%            |
| Sun, Sep 13 | CLE at JAX | CLE +7.5         | 52.8%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 52.1%            |
| Sun, Sep 13 | GB at MIN  | MIN -1.5         | 56.0%            |
| Sun, Sep 13 | MIA at LV  | ★ MIA +3.5       | 54.2%            |
| Sun, Sep 13 | NO at DET  | DET -7           | 53.8%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +3           | 50.1%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 52.5%            |
| Sun, Sep 13 | WAS at PHI | WAS +4.5         | 62.0%            |
| Mon, Sep 14 | DEN at KC  | KC -3            | 53.5%            |

`Decision score` is the raw model probability oriented to the final policy side. On a policy flip it is a mirrored decision-strength score, not a newly calibrated probability for that side; it is also not historical accuracy. This is research output, not a wagering recommendation.

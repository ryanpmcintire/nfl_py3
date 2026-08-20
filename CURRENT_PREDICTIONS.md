# NFL ATS predictions: 2026 Week 1

Published from synchronized model `3083f6cbc5e45acb` at `2026-08-20T20:22:56.890043+00:00`.

> **Early, mutable research preview.** Lines, injuries, depth charts, and model inputs may change before kickoff. Regenerate and republish this card as the week approaches.

Active model: `market_residual` with `weak_stack` features (`3083f6cbc5e45acb`). Its distinct close-graded chronological 2018-2025 evaluation classified **1,081 of 2,075 non-push games correctly (52.10%)**. The week-blocked 95% interval was 50.12%-54.24%. The model baseline is the separate opener-graded probability rule documented in `docs/opener_evaluation.md`.

**Overlay applied: 1 pick flipped** by the year-1 head-coach fade (weeks 1-8, clean case only: the model sided with a first-year coach's team against a coach the opponent KEPT). BAL at IND: BAL -> IND. See docs/coach_fade_overlay.md.

**Production policy active:** after the year-1-coach policy, back the sole team with a broad player-arrest incident dated 1-14 days before Tuesday when the incoming pick opposes it. Its frozen opener evaluation scored 53.76% versus the model baseline's 53.36% on 1,503 graded games (+0.399 accuracy points, probability_positive=0.8562). The direction was discovered on overlapping history, so it remains unresolved and both arms continue to be tracked prospectively. No game on this week's card matched the flip rule, so every side is unchanged. See docs/player_arrests_back_side_overlay.md.

**Best Pick of the week (★):** MIA +3.5 in MIA at LV. The pool scores one Best Pick per regular-season week. This pick was nominated by calibrated probability among low-disagreement games.

| Date        | Matchup    | ATS prediction   | Model estimate   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | SEA -3.5         | 51.7%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 53.6%            |
| Sun, Sep 13 | ARI at LAC | ARI +10.5        | 63.8%            |
| Sun, Sep 13 | ATL at PIT | ATL +3           | 53.4%            |
| Sun, Sep 13 | BAL at IND | IND +3.5         | 51.6%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 56.1%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 53.5%            |
| Sun, Sep 13 | CLE at JAX | JAX -7.5         | 52.1%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 50.5%            |
| Sun, Sep 13 | GB at MIN  | MIN -1.5         | 53.6%            |
| Sun, Sep 13 | MIA at LV  | ★ MIA +3.5       | 54.4%            |
| Sun, Sep 13 | NO at DET  | DET -7           | 53.8%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +3           | 50.5%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 52.6%            |
| Sun, Sep 13 | WAS at PHI | WAS +4.5         | 61.7%            |
| Mon, Sep 14 | DEN at KC  | KC -3            | 57.1%            |

`Model estimate` is the model's game-specific probability for its selected ATS side; it is not the model's historical accuracy. This is research output, not a wagering recommendation.

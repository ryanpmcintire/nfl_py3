# NFL ATS predictions: 2026 Week 1

Published from the synchronized weak stack model, 2026-09-05 21:21 UTC.

<!-- publication: model_id=ab29832a4e099766 published_at_utc=2026-09-05T21:21:28.750323+00:00 -->

> **Lines, injuries, depth charts, and model inputs may change before kickoff.** Regenerate and republish this card as the week approaches.

Active model: weak stack (market residual). Its distinct close-graded chronological 2018-2025 evaluation classified **1,087 of 2,075 non-push games correctly (52.39%)**. The 95% range was 50.41%-54.35%. The model's baseline comparison is the separate opener-graded accuracy rule documented in `docs/opener_evaluation.md`.

**Production policy active:** four situational rules run independently against the computer's first pick and flip it once when any one of them fires: coach fade, division revenge, player arrests, and the spread-gap zone. This week they changed 3 picks. Its archive score (see the This Week board's headline) is the best of 127 similar combinations scored on the very games that chose it, so treat it as a ceiling, never an expectation. The de-inflated planning estimate for the played card is ≈55%: four real out-of-sample split-half selections average +1.30 accuracy points, and shrinking the raw archive gain by the measured 0.59-0.64 selection-shrinkage factor lands in the same place. A separate re-check of the selection step itself measured 0.00 points, so treat the estimate as an upper-middle read, not a floor. Paired prospective tracking against the prior coach-to-arrests chain begins at the Week 1 lock. Rules: coach fade, division revenge tilt, player arrests back side policy, spread gap zone fade. See docs/overlay_subset_holdout_v2.md.

**Best Pick of the week (★):** MIA +3.5 in MIA at LV. The pool scores one Best Pick per regular-season week. This pick was nominated by calibrated probability among low-disagreement games.

| Date        | Matchup    | ATS prediction   | Decision score   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | SEA -3.5         | 51.4%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 54.0%            |
| Sun, Sep 13 | ARI at LAC | ARI +10.5        | 65.1%            |
| Sun, Sep 13 | ATL at PIT | ATL +3           | 54.2%            |
| Sun, Sep 13 | BAL at IND | IND +3.5         | 53.9%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 55.8%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 53.3%            |
| Sun, Sep 13 | CLE at JAX | CLE +7.5         | 52.7%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 51.8%            |
| Sun, Sep 13 | GB at MIN  | MIN -1.5         | 55.8%            |
| Sun, Sep 13 | MIA at LV  | ★ MIA +3.5       | 54.4%            |
| Sun, Sep 13 | NO at DET  | DET -7           | 53.8%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +3           | 50.3%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 52.4%            |
| Sun, Sep 13 | WAS at PHI | WAS +4.5         | 61.8%            |
| Mon, Sep 14 | DEN at KC  | KC -3            | 53.4%            |

**Tiebreaker (last game, DEN at KC):** KC 24 - DEN 20, total 44 (market total 43.5) -- consistent with the KC -3 pick.

**Source freshness: COMPLETE.** Complete: odds opener, odds refresh, injuries nflverse, injuries nflverse timestamps, projected lineups, referee assignments, player arrests, pfr transactions, airnow weather. Degraded (allowed fallback): none. Blocked: none. Not due yet: inactives. Not set up: injuries sportradar. Budgets, fallbacks and source states: `docs/source_freshness_policy.md`.

`Decision score` is the computer's own probability, oriented to the final pick. On a flip it is a mirrored decision-strength score, not a newly calibrated probability for that side; it is also not historical accuracy.

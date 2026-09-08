# NFL ATS predictions: 2026 Week 1

Published from the synchronized weak stack model, 2026-09-08 17:14 UTC.

<!-- publication: model_id=3ccf838f9a304dcd published_at_utc=2026-09-08T17:14:08.688605+00:00 -->

> **Lines, injuries, depth charts, and model inputs may change before kickoff.** Regenerate and republish this card as the week approaches.

Active model: weak stack (market residual). Its distinct close-graded chronological 2018-2025 evaluation classified **1,085 of 2,075 non-push games correctly (52.29%)**. The 95% range was 50.17%-54.34%. The model's baseline comparison is the separate opener-graded accuracy rule documented in `docs/opener_evaluation.md`.

**Production policy active:** three situational rules run independently against the computer's first pick and flip it once when any one of them fires: coach fade, division revenge, and player arrests. This week they changed 2 picks. The spread-only threshold adjustment is retired because it has no explained mechanism. Its archive comparison reuses 127 similar combinations scored on the same games; it is not independent evidence of future accuracy. The planning estimate remains ≈55%. Paired prospective tracking against the former four-adjustment card begins at the Week 1 lock. Rules: coach fade, division revenge tilt, player arrests back side policy. See docs/spread_gap_zone_retired.md.

**Best Pick of the week (★):** MIA +3.5 in MIA at LV. The pool scores one Best Pick per regular-season week. This pick was nominated by calibrated probability among low-disagreement games.

| Date        | Matchup    | ATS prediction   | Decision score   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | NE +3.5          | 50.2%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 55.6%            |
| Sun, Sep 13 | ARI at LAC | ARI +9.5         | 64.2%            |
| Sun, Sep 13 | ATL at PIT | ATL +3.5         | 55.9%            |
| Sun, Sep 13 | BAL at IND | IND +3.5         | 55.4%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 54.2%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 51.6%            |
| Sun, Sep 13 | CLE at JAX | JAX -8.5         | 53.3%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 50.2%            |
| Sun, Sep 13 | GB at MIN  | MIN -1.5         | 54.1%            |
| Sun, Sep 13 | MIA at LV  | ★ MIA +3.5       | 56.1%            |
| Sun, Sep 13 | NO at DET  | DET -6.5         | 52.3%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +1.5         | 51.6%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 50.9%            |
| Sun, Sep 13 | WAS at PHI | WAS +5.5         | 63.3%            |
| Mon, Sep 14 | DEN at KC  | KC -2.5          | 51.8%            |

**Tiebreaker (last game, DEN at KC):** KC 23 - DEN 20, total 43 (market total 43.5) -- consistent with the KC -2.5 pick.

**Source freshness: COMPLETE.** Complete: odds opener, odds refresh, injuries nflverse, injuries nflverse timestamps, inactives, projected lineups, referee assignments, player arrests, pfr transactions, airnow weather. Degraded (allowed fallback): none. Blocked: none. Not due yet: none. Not set up: injuries sportradar. Budgets, fallbacks and source states: `docs/source_freshness_policy.md`.

`Decision score` is the computer's own probability, oriented to the final pick. On a flip it is a mirrored decision-strength score, not a newly calibrated probability for that side; it is also not historical accuracy.

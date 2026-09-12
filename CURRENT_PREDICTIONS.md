# NFL ATS predictions: 2026 Week 1

Published from the synchronized weak stack model, 2026-09-12 17:48 UTC.

<!-- publication: model_id=1b9bfbddc10a39ef published_at_utc=2026-09-12T17:48:35.057548+00:00 -->

> **Lines, injuries, depth charts, and model inputs may change before kickoff.** Regenerate and republish this card as the week approaches.

Active model: weak stack (market residual). Its distinct close-graded chronological 2018-2025 evaluation classified **1,085 of 2,075 non-push games correctly (52.29%)**. The 95% range was 50.21%-54.41%. The model's baseline comparison is the separate opener-graded accuracy rule documented in `docs/opener_evaluation.md`.

**Production policy active:** three situational rules run independently against the computer's first pick and flip it once when any one of them fires: coach fade, division revenge, and player arrests. This week they changed 4 picks. The spread-only threshold adjustment is retired because it has no explained mechanism. Its archive comparison reuses 127 similar combinations scored on the same games; it is not independent evidence of future accuracy. The planning estimate remains ≈55%. Paired prospective tracking against the former four-adjustment card begins at the Week 1 lock. Rules: coach fade, division revenge tilt, player arrests back side policy, bye edge fade, forecast cold visitor tilt, pbp08 protection mismatch tilt, interim hc first game tilt, tank zone fade tilt, precip high total tilt. See docs/spread_gap_zone_retired.md.

**Best Pick of the week (★):** MIA +3.5 in MIA at LV. The pool scores one Best Pick per regular-season week. This pick was nominated by calibrated probability among low-disagreement games with a spread of six and a half or less.

| Date        | Matchup    | ATS prediction   | Decision score   |
|:------------|:-----------|:-----------------|:-----------------|
| Wed, Sep 09 | NE at SEA  | NE +3.5          | 50.2%            |
| Thu, Sep 10 | SF at LA   | SF +3.5          | 55.4%            |
| Sun, Sep 13 | ARI at LAC | ARI +9.5         | 53.4%            |
| Sun, Sep 13 | ATL at PIT | PIT -3.5         | 57.0%            |
| Sun, Sep 13 | BAL at IND | IND +3.5         | 57.1%            |
| Sun, Sep 13 | BUF at HOU | HOU +1.5         | 57.0%            |
| Sun, Sep 13 | CHI at CAR | CAR +2.5         | 57.0%            |
| Sun, Sep 13 | CLE at JAX | JAX -8.5         | 56.1%            |
| Sun, Sep 13 | DAL at NYG | DAL -2.5         | 56.9%            |
| Sun, Sep 13 | GB at MIN  | MIN -1.5         | 57.0%            |
| Sun, Sep 13 | MIA at LV  | ★ MIA +3.5       | 55.0%            |
| Sun, Sep 13 | NO at DET  | NO +6.5          | 54.9%            |
| Sun, Sep 13 | NYJ at TEN | NYJ +1.5         | 57.0%            |
| Sun, Sep 13 | TB at CIN  | CIN -3.5         | 57.0%            |
| Sun, Sep 13 | WAS at PHI | WAS +5.5         | 56.0%            |
| Mon, Sep 14 | DEN at KC  | DEN +2.5         | 57.0%            |

**Tiebreaker (last game, DEN at KC):** KC 22 - DEN 20, total 42 (market total 43.5) -- consistent with the DEN +2.5 pick.

**Source freshness: COMPLETE.** Complete: odds opener, odds refresh, injuries nflverse, injuries nflverse timestamps, inactives, projected lineups, referee assignments, player arrests, pfr transactions, airnow weather. Degraded (allowed fallback): none. Blocked: none. Not due yet: none. Not set up: injuries sportradar. Budgets, fallbacks and source states: `docs/source_freshness_policy.md`.

`Decision score` is the computer's own chance that this side covers, adjusted for how the computer has actually done on spreads this size. Big favourites and big underdogs have been its weak spot, so a very confident-looking number there is pulled back toward what it has really hit, and it is never shown below 50% on a side this card is picking. It is a per-game chance, not historical accuracy.

<!-- LATE_WEEK_REFRESH:START -->
## Late-week refresh (as of 2026-09-12T17:53:53.738014+00:00)

5 picks changed since the Tuesday card (lineups_refresh), recomputed with current data but scored at the frozen Tuesday grading line. Only games whose deadline (their own kickoff, or that week's Sunday 4:00 PM ET if earlier) had not yet passed were eligible. "Policy" is `late_week_leader_median_follow_1_0_big_spread_0_5` when the three leading books moved the line at least a full point since Tuesday -- or half a point on the biggest spreads, 10.5 or more -- and the pick followed them, `late_week_leader_median_follow_1_0_big_spread_0_5_news_veto` when they moved that far but the injury report points the other way, so Tuesday's pick stands, `handle_follow_0_70` when that rule did not fire and at least 70% of the money bet on the game sat on the other side, `rookie_crew_underdog_v1` when it did not fire and the officiating crew for that game is new this season, or `model_only` when nothing above fired (or no market evidence was available). The 1.0-point `movement_ge_1.0` consensus rule was retired from the served chain on 2026-09-10 and is recorded as the paired challenger `consensus_movement_1_0_off_incumbent` -- see docs/late_week_refresh.md's movement-policy sections. Where this table and the picks table above disagree, the side here is the one being played.

| Matchup    | Previous pick   | New pick   | Model estimate   | Policy                                            |   Market move |
|:-----------|:----------------|:-----------|:-----------------|:--------------------------------------------------|--------------:|
| ARI at LAC | AWAY            | HOME       | 36.2%            | handle_follow_0_70                                |           0   |
| ATL at PIT | AWAY            | HOME       | 47.5%            | late_week_leader_median_follow_1_0_big_spread_0_5 |           2.5 |
| CHI at CAR | HOME            | AWAY       | 48.8%            | handle_follow_0_70                                |          -0.5 |
| DEN at KC  | HOME            | AWAY       | 49.1%            | handle_follow_0_70                                |           0   |
| WAS at PHI | AWAY            | HOME       | 36.7%            | late_week_leader_median_follow_1_0_big_spread_0_5 |           1   |
<!-- LATE_WEEK_REFRESH:END -->

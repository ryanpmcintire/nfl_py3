# NFL ATS predictions: 2026 Week 2

Published from the synchronized weak stack model, 2026-09-18 16:20 UTC.

<!-- publication: model_id=0d7f451b57c46382 published_at_utc=2026-09-18T16:20:17.075115+00:00 -->

> **Lines, injuries, depth charts, and model inputs may change before kickoff.** Regenerate and republish this card as the week approaches.

Active model: weak stack (market residual). Its distinct close-graded chronological 2018-2025 evaluation classified **1,093 of 2,091 non-push games correctly (52.27%)**. The 95% range was 50.10%-54.30%. The model's baseline comparison is the separate opener-graded accuracy rule documented in `docs/opener_evaluation.md`.

**Production policy active:** three situational rules run independently against the computer's first pick and flip it once when any one of them fires: coach fade, division revenge, and player arrests. This week they changed 4 picks. The spread-only threshold adjustment is retired because it has no explained mechanism. Its archive comparison reuses 127 similar combinations scored on the same games; it is not independent evidence of future accuracy. The planning estimate remains ≈55%. Paired prospective tracking against the former four-adjustment card begins at the Week 1 lock. Rules: coach fade, division revenge tilt, player arrests back side policy, bye edge fade, forecast cold visitor tilt, pbp08 protection mismatch tilt, interim hc first game tilt, tank zone fade tilt, precip high total tilt. See docs/spread_gap_zone_retired.md.

**Best Pick of the week (★):** DEN -2.5 in JAX at DEN. The pool scores one Best Pick per regular-season week. This pick was the one this card is most confident in, among the games the books agree on with a spread of six and a half or less.

| Date        | Matchup    | ATS prediction   | Cover chance   |
|:------------|:-----------|:-----------------|:---------------|
| Thu, Sep 17 | DET at BUF | BUF -4.5         | 54.1%          |
| Sun, Sep 20 | CAR at ATL | CAR -1.5         | 52.9%          |
| Sun, Sep 20 | CIN at HOU | CIN +2.5         | 53.3%          |
| Sun, Sep 20 | CLE at TB  | TB -8.5          | 56.7%          |
| Sun, Sep 20 | GB at NYJ  | GB -3.5          | 52.6%          |
| Sun, Sep 20 | IND at KC  | IND +6.5         | 51.6%          |
| Sun, Sep 20 | JAX at DEN | ★ DEN -2.5       | 57.8%          |
| Sun, Sep 20 | LV at LAC  | LAC -7.5         | 50.3%          |
| Sun, Sep 20 | MIA at SF  | SF -13.5         | 57.9%          |
| Sun, Sep 20 | MIN at CHI | MIN +5.5         | 50.1%          |
| Sun, Sep 20 | NO at BAL  | NO +8.5          | 62.7%          |
| Sun, Sep 20 | PHI at TEN | PHI -6.5         | 57.3%          |
| Sun, Sep 20 | PIT at NE  | NE -4.5          | 56.5%          |
| Sun, Sep 20 | SEA at ARI | SEA -4.5         | 55.0%          |
| Sun, Sep 20 | WAS at DAL | WAS +3.5         | 50.5%          |
| Mon, Sep 21 | NYG at LA  | LA -7.5          | 65.8%          |

**Tiebreaker (last game, NYG at LA):** LA 31 - NYG 17, total 48 (market total 48.5) -- consistent with the LA -7.5 pick.

**Source freshness: DEGRADED.** Complete: odds opener, injuries nflverse, injuries nflverse timestamps, inactives, projected lineups, referee assignments, player arrests, pfr transactions, airnow weather. Degraded (allowed fallback): odds refresh. Blocked: none. Not due yet: none. Not set up: injuries sportradar. Budgets, fallbacks and source states: `docs/source_freshness_policy.md`.

`Cover chance` is the picked side's chance to cover. The chance beside each pick is how often picks like this one have actually landed: across 1,503 past games scored the same way, the picks this card called strong won 59% of the time and the ones it called slight won 54%.

<!-- LATE_WEEK_REFRESH:START -->
## Late-week refresh (as of 2026-09-18T17:30:31.899894+00:00)

4 picks changed since the Tuesday card (last_call_fri_13:30), recomputed with current data but scored at the frozen Tuesday grading line. Only games whose deadline (their own kickoff, or that week's Sunday 4:00 PM ET if earlier) had not yet passed were eligible. "Policy" is `late_week_leader_median_follow_1_0_big_spread_0_5` when the three leading books moved the line at least a full point since Tuesday -- or half a point on the biggest spreads, 10.5 or more -- and the pick followed them, `late_week_leader_median_follow_1_0_big_spread_0_5_news_veto` when they moved that far but the injury report points the other way, so Tuesday's pick stands, `handle_follow_0_70` when that rule did not fire and at least 70% of the money bet on the game sat on the other side, `rookie_crew_underdog_v1` when it did not fire and the officiating crew for that game is new this season, or `model_only` when nothing above fired (or no market evidence was available). The 1.0-point `movement_ge_1.0` consensus rule was retired from the served chain on 2026-09-10 and is recorded as the paired challenger `consensus_movement_1_0_off_incumbent` -- see docs/late_week_refresh.md's movement-policy sections. Where this table and the picks table above disagree, the side here is the one being played.

| Matchup    | Previous pick   | New pick   | Model estimate   | Policy     | Market move   |
|:-----------|:----------------|:-----------|:-----------------|:-----------|:--------------|
| IND at KC  | IND             | KC         | 50.0%            | model_only | n/a           |
| LV at LAC  | LAC             | LV         | 59.1%            | model_only | n/a           |
| MIN at CHI | MIN             | CHI        | 57.3%            | model_only | n/a           |
| WAS at DAL | WAS             | DAL        | 55.4%            | model_only | n/a           |
<!-- LATE_WEEK_REFRESH:END -->

# NFL ATS predictions: 2026 Week 3

Published from the synchronized weak stack model, 2026-09-25 16:17 UTC.

<!-- publication: model_id=8587951e3acc6055 published_at_utc=2026-09-25T16:17:03.761209+00:00 -->

> **Lines, injuries, depth charts, and model inputs may change before kickoff.** Regenerate and republish this card as the week approaches.

Active model: weak stack (market residual). Its distinct close-graded chronological 2018-2025 evaluation classified **1,103 of 2,107 non-push games correctly (52.35%)**. The 95% range was 50.22%-54.46%. The model's baseline comparison is the separate opener-graded accuracy rule documented in `docs/opener_evaluation.md`.

**Production policy active:** one calibrated probability combines the model, situational evidence and available line movement to choose each side.

**Best Pick of the week (★):** CAR -2.5 in CAR at CLE. The pool scores one Best Pick per regular-season week. This pick was provisionally chosen because it has the highest estimated chance to cover among eligible games. Its estimated lead over the other picks is uncertain.

| Date        | Matchup    | ATS prediction   | Cover chance   |
|:------------|:-----------|:-----------------|:---------------|
| Thu, Sep 24 | ATL at GB  | ATL +6.5         | 60.5%          |
| Sun, Sep 27 | ARI at SF  | SF -8.5          | 50.3%          |
| Sun, Sep 27 | BAL at DAL | DAL +2.5         | 54.1%          |
| Sun, Sep 27 | CAR at CLE | ★ CAR -2.5       | 60.3%          |
| Sun, Sep 27 | CIN at PIT | CIN -3.5         | 56.1%          |
| Sun, Sep 27 | HOU at IND | HOU -2.5         | 51.1%          |
| Sun, Sep 27 | KC at MIA  | KC -10.5         | 54.8%          |
| Sun, Sep 27 | LAC at BUF | LAC +7.5         | 50.7%          |
| Sun, Sep 27 | LA at DEN  | LA -2.5          | 50.8%          |
| Sun, Sep 27 | LV at NO   | NO -3.5          | 56.3%          |
| Sun, Sep 27 | MIN at TB  | TB +1.5          | 50.6%          |
| Sun, Sep 27 | NE at JAX  | JAX -2.5         | 52.5%          |
| Sun, Sep 27 | NYJ at DET | NYJ +6.5         | 50.7%          |
| Sun, Sep 27 | SEA at WAS | WAS +6.5         | 51.3%          |
| Sun, Sep 27 | TEN at NYG | TEN +3.5         | 50.2%          |
| Mon, Sep 28 | PHI at CHI | CHI +3.5         | 54.0%          |

**Tiebreaker (last game, PHI at CHI):** CHI 21 - PHI 19, total 40 (market total 41) -- consistent with the CHI +3.5 pick.

**Source freshness: COMPLETE.** Complete: odds opener, odds refresh, injuries nflverse, injuries nflverse timestamps, inactives, projected lineups, referee assignments, pfr transactions, airnow weather. Degraded (allowed fallback): none. Blocked: none. Not due yet: none. Not set up: injuries sportradar. Budgets, fallbacks and source states: `docs/source_freshness_policy.md`.

`Cover chance` is the picked side's chance to cover. Cover chance is a fitted estimate, excluding ties. Across 1,503 past games with each season held out of fitting, strong estimates won 60% and slight estimates won 55%. These broad groups do not establish a large advantage for the single highest estimate.

<!-- LATE_WEEK_REFRESH:START -->
## Late-week refresh (as of 2026-09-25T16:18:56.480800+00:00)

2 picks changed since the Tuesday card (lineups_refresh), recomputed with current data but scored at the frozen Tuesday grading line. Only games whose deadline (their own kickoff, or that week's Sunday 4:00 PM ET if earlier) had not yet passed were eligible. "Policy" identifies the probability rule recorded for that revision. `four_term_pick_probability_v1` combines the model, situational evidence and available line movement into the same calibrated chance shown on the card. Earlier revisions retain their original policy labels. Where this table and the picks table above disagree, the side here is the one being played.

| Matchup   | Previous pick   | New pick   | Model estimate   | Policy                        |
|:----------|:----------------|:-----------|:-----------------|:------------------------------|
| ARI at SF | ARI             | SF         | 50.3%            | four_term_pick_probability_v1 |
| MIN at TB | MIN             | TB         | 50.6%            | four_term_pick_probability_v1 |
<!-- LATE_WEEK_REFRESH:END -->

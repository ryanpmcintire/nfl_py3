# MOD-57 P2-A outside leads

## Goal
Leads from outside the repo that could lift picks on quiet games (line moves under a point after Tuesday; 50.4% served, model alone 52.7%), each novelty-checked. No fitting, no code.

## State
Done 2026-11-11 09:33 EST (date output). Web evidence is thin: no peer-reviewed out-of-sample spread inefficiency at the close with an effect size was found; most hits were blogs or abstracts only.
Sources (reported, abstracts only): Applied Economics Letters 30(18) 2525 (SuperContest entrants no better than a coin flip, recency bias; informahealthcare 10.1080/13504851.2022.2099517); RotoGrinders 2018 (top-5 consensus picks 200-214-11 since 2013, https://rotogrinders.com/articles/2018-supercontest-recap-three-tied-for-first-place-after-wild-week-2-2664929); J. Econ. Finance 2018 (home underdogs cover, s12197-018-9431-4); weather study 1984-2004 (bettors underrate weather, totals); QB-out betting shift +2.1/-3.1 pct of money (2003-17, bettor money not spread); a weather-added spread model 54.28% late season (chronology unverified). Vendor ML claims of 62.5% (Zoltar, swarm picks) have no disclosed chronological test: not evidence.
TESTED in repo (do not rerun): team totals (registry/weak_signals/mkt23_unit2_term.json, log-loss gain -0.0015 [-0.0036,+0.0007]); first-half line (half_line_script.json, +5.29 acc pts [-2.5,+12.6] unresolved); Kalshi/Polymarket spread ladders (docs/lanes/acc-u8-prediction-markets.md); slot residual (mkt24, -0.43); opener revision (mkt25, -0.0001); fantasy ADP (docs/ffc_adp_divergence_screen.md); TV attention (tv_attention_fade_on_production); ownership proxy (pool05_ownership_v2); injury trajectory/practice (injury_trajectory_refresh_cx14, friday_designation_momentum); weather (forecast_weather, mod38_kicker_wind); home dog/late season (mod18_home_dog_location_v1, late_season_crews); sharp-book weighted movement (sharp_book_weighted_movement.json); reverse line (midweek_steam_refresh_on_production).
Ranked ABSENT/PARTLY leads (inferred gains all small, under 0.5 pt overall):
1 Real contest consensus counts (SuperContest/Circa) ABSENT (only a Westgate line mention in docs/vegasinsider_backfill.md; pool05 is a proxy). Measure: 2020-25 quiet games, target pool-line cover, consensus share as fifth LOSO term vs four-term.
2 Sharp book (Pinnacle/Circa/Bookmaker) minus FanDuel at Tuesday noon, PARTLY (leader-book term tested; Pinnacle not in captures; history free source unverified). Measure: 2026 payload games first, quiet subset, gap sign vs cover.
3 Public rating gap (Massey composite, FPI) PARTLY (MOD-30 row unmeasured, forward only). Measure: 2020-25 Massey history if free, gap to Tuesday line as fifth term, quiet games.
4 Unsigned cross-book spread dispersion as a model-weight modifier on quiet games PARTLY (used only for best-pick ranking, docs/best_pick_composed_rule.md). Measure: served side accuracy by dispersion tercile, quiet games, permutation null.
5 Tuesday-to-Sunday wind/temperature forecast revision on unmoved lines PARTLY (weather tested; revision split on quiet games not). Measure: revision vs total-lattice read on quiet games.
6 Player-prop (QB pass yards) implied offence strength PARTLY (docs/player_props_sourcing.md pulls only; props exist from 2023-05, paid source). Measure: 2023-25 quiet games, prop vs team-total gap.
7 QB-out bettor overreaction on quiet games PARTLY (public splits tested broadly). Measure: QB-out quiet games, fade-money side vs cover.
8 MKT-26 move size as confidence modifier PARTLY (ROADMAP row unmeasured). Measure: log loss by move-size band, 2023-25.
9 Preseason win totals as early-season prior ABSENT, low gain. Measure: Weeks 1-6 residual vs total-implied strength.
10 Weekly fantasy consensus projections ABSENT (ADP tested; weekly history likely paywalled). Measure: starter-projection change vs team availability term.

## Tried
Searches: spread inefficiency, SuperContest, derivative markets, ML accuracy, injuries/weather, sharp-book price leadership. Greps: registry/weak_signals, ROADMAP, docs/lanes, docs/*.md on about 40 concept phrasings.

## Next
Start 1, 2, 3 (new information, free data). Verify the history source for each before building. 4 is in-repo and cheap.

## Open
Whether SuperContest weekly consensus archives exist back to 2020 free; whether any free Pinnacle/Circa history exists.

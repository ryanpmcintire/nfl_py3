# MOD-23 unit 8 - player-level Wikipedia pageviews in the base model

## Goal
Test whether abnormal attention to each team's starters (Wikipedia daily pageviews) adds to the base model by itself on the fair opener test (baseline 802-701, 53.36%, 1,503 decisive 2020-2025).

## State
- History check (read): team-level pageview attention was screened (attention_battery_*, docs/attention_followup.md; weak_signals families, 21 hits). No player-level pageview test found in ROADMAP.md, docs/, registry (ROADMAP has 0 hits). New here: player starters, abnormality vs own baseline, scored as a base-model feature.
- Declaration (before scoring): cutoff = Monday 00:00 UTC of the week's first game; days used are the 7 days before it (one game cycle) with a 28-day own-baseline just before that (four cycles); both are design units, not tuned. Starters = QB1, top 3 RB/WR/TE by offense snaps, top 3 defenders by defense snaps from the team's latest game with gameday before the cutoff. Score = log1p(window mean) - log1p(baseline mean), standardised with mean/sd of earlier seasons only (2016 gets 0). Assert: max day used < cutoff and source game < cutoff on every row.
- Arms (2, looks = 2 arms x 2 eras = 4 accuracy looks): A_qb (QB1 only) and B_all (QB + skill + defense), each one column home-minus-away added to the 88-input ridge, alpha 10, via mod24_u1 harness run_arm. Pre-2016 rows carry 0.
- Scripts: scripts/mod23_unit8_fetch.py (resolve titles, download), scripts/mod23_unit8_grade.py (build, grade). Raw: data/raw/wiki_pageviews/<run_id>/ with manifest sha256; processed: data/processed/wiki_pageviews/.
- Orchestrator check (read docs/attention_followup.md): prior screen = team-article views, in-season trailing-8-game z, hot/cold subset cells vs home_cover (6 battery cells, best both_cold +0.52 pts, week-blocked CI [-0.44, +1.50], P+ 0.857, season-blocked [-0.004, +1.06], P+ 0.974; plus 4 predeclared followup cells, 10 prior cells, followup recorded count per weak_stack_v3 is 3, unverified). Material differences: this unit uses player articles (QB1, skill, defenders from the last pre-cutoff game) with each player's own 28-day baseline as a lineup-news/injury-attention proxy, enters it as a fitted ridge term in the base model (one probability selects the side), and grades paired accuracy/log loss/Brier on the opener test, not subset cover gaps. Not the same construct, so continuing. Family accounting: 4 new looks added to the attention family, total 14 including the 10 prior cells.

## Tried
- Fetch done (run 20261016T221429Z, 2172 of 2174 articles; 2174 of 2611 player-positions resolved to a title). Leak check passed (max day used = cutoff - 1). Coverage 60% of all games (2016+).
- Graded (artifacts/mod23_unit8/grade.log, report.json), true 2020-2025 vs base 802-701: A_qb 797-706 (-0.33 pts, season-blocked [-1.02, +0.39], P+ 0.157, 43 flips 19-24, dLL +0.0005, dBrier +0.0002); B_all 802-701 (0.00, [-1.03, +1.11], P+ 0.461, 74 flips 37-37, dLL +0.00001). Proxy 2011-2019 mostly zero feature (pageviews start 2016): A -0.13, B -0.31.
- Recorded 4 cells, family mod23_unit8_player_pageviews, unresolved_below_power. Attention family total 14.

## Next
- Done; no pick change. Optional: a pageview-lineup-news term trained 2017+ only is untested for 2016-2019 proxy; weak.

## Open
- Title resolution by Wikipedia short description can pick a wrong namesake; unresolved players drop out (coverage reported).

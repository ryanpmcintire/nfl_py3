# madden-prior-2026-11-16

## Goal
Madden launch ratings (expert, results-free, published pre-Week-1) aggregated over each team's expected starters as a talent prior; test whether it predicts the weeks 1-6 hindsight gap the line misses (goal-65, mod57 P2-O/P2-P/Phase 3).

## Declaration (written 2026-11-16 17:43 EST, before any outcome join)
- Source: github theedgepredictor/nfl-madden-data data/madden/processed/<season>.csv (scraped from maddenratings.weebly.com launch "full_player_ratings" sheets and EA drop-api iteration 1-base; processed = nflverse join only, imputation lives in the separate dataset stage, not used). Rows with overallrating missing are unrated.
- Expected starters: weeks >= 2, players on the team's most recent prior regular-season game this season, weight = offense_pct or defense_pct in that game (nflverse snap counts, pfr id). Week 1: team's Madden launch roster ranked within Madden position by overall; slot weight = mean week-1 snap pct of that position-rank slot measured over seasons before the scored season only.
- Unit index = snap-weighted mean overall of rated players (offense, defense); team index = mean of the two; diff = home - away.
- Term madden_talent_vs_line = diff minus a leave-one-season-out OLS map of diff on spread_line (inputs only, no outcomes). Served look recomputes it against tue_open_home_spread.
- Harness: scripts/mod57_p3_gap_frame.py, family madden_talent_prior_gap_frame, primary weeks 1-6 (gap and margin labels), placebo weeks 7-12, 1000 within-season shuffles, intercept-only control, el_stack increment. Gate for ONE served look (opener 2020-25 weeks 1-6, pushes kept): primary cell above the 95th shuffle percentile on share and beats intercept-only on the decisive record.

## State
- Archive: github theedgepredictor/nfl-madden-data (no repo license; upstream EA ratings via maddenratings.weebly.com; source_policies entry madden_ratings_github_archive, amber). Raw in data/external/madden/ (gitignored): stage_<s>.csv = Madden ratings with Madden's own team; <s>.csv = nflverse registry used only for pfr ids.
- Launch proof (measured): stage files list the launch team for a player traded after launch in all 13 seasons 2013-2025 (Richardson CLE, Harvin SEA, V.Davis SF, Bradford PHI, Ajayi MIA, Mack OAK, Tunsil MIA, Ngakoue JAX, V.Miller DEN, McCaffrey CAR, C.Young WAS, D.Adams LV, Parsons DAL); 2019 Lamar Jackson 76 (launch, not his in-season 99). Wayback: Madden 16 full sheet 2015-08-26, Madden 23 sheet 2022-07-23; repo raw 2025 committed 2025-08-04 (all before Week 1).
- Pitfall fixed: processed/<s>.csv drops ratings for players whose nflverse team later differs (future roster info); use stage files. stage_2019 is an exact duplicate, deduplicated.
- Built (measured): artifacts/madden_prior/madden_prior_features.parquet, 3,389/3,407 games 2013-25 with the term; rated snap coverage 0.91 off / 0.89 def; corr(diff, close line) 0.52-0.77 by season. 2009-12 absent (snap counts start 2013), harness imputes.
- Harness DONE (measured, artifacts/madden_prior/madden_talent_table.txt; 8 looks: 2 primary, 2 placebo, 4 stack increments):
  - margin label wks 1-6: share +0.00035, shuffle percentile 0.980 (MAE 0.956, slope 0.983); slope +0.54 p+0.86; sign + in 9/11 folds, 0 negative; close record 529-470 v intercept-only 508-491; opener 2020-25 281-243; in-sample minus out-of-sample share +0.0038.
  - gap label wks 1-6: share -0.0045, percentile 0.12, ElasticNet zeroes the term in 10/11 folds; 529-470 v intercept 531-468.
  - placebo wks 7-12: margin percentile 0.505, gap 0.87; nothing.
  - inside the 295-column stack: margin wks 1-6 adds -0.0002 share (p+ 0.00), -0.0012 MAE (p+ 0.00); loses there, as the P3 best look did.
  - One look of 2 primary at the 98th percentile is what chance gives about 4% of the time (inferred).
- Recorded: `weak-signals record --batch artifacts/madden_prior/madden_talent_batch.json` recorded 2 cells (mae_improvement, unresolved_below_power, family madden_talent_prior_gap_frame); `pool --league nfl --effect-units accuracy_points` exit 0.
- Declared gate PASSED on the margin label only (above the 95th percentile and beats intercept-only 529-470 v 508-491). The ONE served look is declared, not run (tool-call cap).

## Tried
- Prior work: no Madden input ever tested (grep ROADMAP, lanes, registry); data_source_scout_v2 row 10 named it unqueued. PER-09 latent ratings and per14 unit prior were EPA-based, not expert ratings.

## Next
- Run the ONE declared served look (fresh agent): at the opener 2020-25 weeks 1-6, pushes kept, term recomputed against tue_open_home_spread (model the run on the P3-G served-stack check, artifacts/mod57_p3g). Otherwise record and close the lane as unresolved.
- Rebuild features: `scripts/madden_prior.py --first-season 2013 --last-season 2025 --pause 1.0`.

## Open
- None.

# MOD-57 P2-G - data on disk versus what the 88 inputs consume

## Goal
Read-only map for base-model rebuild units: which datasets the served ridge (alpha 10, football_weak_stack, 88 inputs) uses, at what grain, and the ten largest unused gaps. Fits and scores nothing. Parent: mod57-sixty-percent.md.

## State (2026-11-11 09:39 EST; read = file:line, measured = scratch script in tests/scratch/mod57/p2g)
Full tables: model_inputs_map.csv, dataset_join.csv, gaps_top10.csv, inputs88.csv, parquet_meta.csv, inventory_top.csv, refs.csv (same folder).

Inputs by source (measured, cols.py reproduces 88 from constants.py FEATURE_SETS["football_weak_stack"]):
- team_stats team-game, 45: offense 21 + defense 18 + results 6. Only 18 of 138 team_stats columns are read (features.py:238-320), collapsed to 13 EWMA rates, span 8, min 3 games, 0.67 offseason regression (features.py:344-420, constants.py:9).
- game/schedule level, 20: context 7, bias 9, elo 2 (K 20, HFA 55; features.py:452-490), experience 2.
- player level, 23: player_qb 5 (QB EWMA span 12), injuries 7, continuity 9, values 2 (spans 8/12/16, players.py:1631-1635).
- Play-by-play enters the 88 only through QB dropback state. The 48 pbp team inputs (success rate, explosive rate, early-down EPA, drives, pressure; game_features_pbp, 201 cols) are NOT in the 88 (pbp family closed, docs/modeling.md:420-448).

Datasets (compressed; dataset_join.csv has all 38 rows):
- BASE: schedules 4902 games; team_stats 9532 team-games; snap_counts 1.0M; injuries 68k outcomes; rosters 1.1M; depth charts 554k (QB only); player values 596k and role_actions 459k (2 terms).
- SERVED ADD-ON only: market/raw quotes (about 6.2M rows, 8919 capture dirs; 2020-22 daily, 2023-25 hourly spreads, 2026 odds-gap-private), public_betting, officials/crew, inactives, splash pool captures.
- RESEARCH only: play-by-play (828k plays x56, 2009-2026, 420 MB), participation (479k x21, 2016-2025, 2026 is HTTP 404), NGS 27k, FTN charting 205k (2022-26), PFR advstats, combine 8968, draft picks 37788, beat-reporter posts 141,891 (2023-26), injury_news 13.8M rows, prediction markets, sbr_odds 2007-21, pre-2009 pbp 898k, forecast archive, coordinators, play_probability_panel.
- CFB (pbp 7.2M plays, rosters, recruiting, portal, lines; 1.5 GB) feeds only the CFB model; never read by the NFL base.
- NEVER READ (zero grep hits in src, scripts, lanes): raw/air_quality, respiratory, bigten_availability, sec_availability, players/raw_players_master (empty).
- Not cheap to count: market/raw and injury_news have thousands of dirs; sizes from os.walk, rows from parquet metadata of first 400 files per key (inventory.csv), seasons from dir names.

## Ten gaps (rank; one-line prior attempt)
1. Play-level football under the team EWMA (828k plays; 13 rates kept): PBP bundle -0.08 pts on 1247 2013-17 games, closed (docs/modeling.md:420); MOD-23 u3 opponent-adjusted -0.80/-1.46.
2. Participation personnel/formation (479k plays): man/zone +0.47 P+ .77 (mod23 u5); 2026 source absent.
3. Player-level lineup quality (values, snaps, rosters): base keeps 2 injury terms + 9 overlaps; APM/unit ratings exist, no graded base result found.
4. Opponent-adjusted ratings: -0.80/-1.46; graph ratings never selected.
5. NGS: slightly worse (mod24 u3); MOD-40 flat, LL +0.00004.
6. FTN charting: luck +0.88 P+ .78 2023-25, 2026 forward 60-47 v 64-43.
7. Quote-level market history as a base input: add-on only; opener-input hazard.
8. Text (beat posts, injury news): all-posts -1.31 P+ .175; lead term +0.75 on 6 decisive games.
9. Pre-2009 / cross-league: A v served -0.80 P+ .187, B -2.06 P+ .067.
10. Rookie, draft, combine priors: draftcap-forward-log; result not extracted.

## Tried
- Every gap above has a prior unit except the combination: no unit rebuilt gaps 1+2+3 at play level into one unit-vs-unit model (all prior tests added EWMA team-level or add-on terms to the 88). Inferred, not measured.
- Not verified: lane search tokens are substrings (ngs, combine over-match in src counts; refs.csv).

## Next
- Rebuild unit candidate: play-level, score-state-conditioned unit ratings (gap 1+2+3) graded at the opener against 802-701, checking registry families pbp_*, mod23_unit3, sim09 first.
- Extract the draft/combine and APM results (gaps 3, 10) before ranking them.

## Open
- team_stats temp/wind in context (schedules): confirm pregame availability before any rebuild reuses it.

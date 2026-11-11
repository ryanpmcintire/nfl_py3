# MOD-57 P2-K - play-level model with who is on the field

## Goal
Give the P2-H play-level model on-field lineup inputs (participation 2016-25) and grade it at the Tuesday opener, standalone and as an added term. Parent: mod57-sixty-percent.md. Script scripts/mod57_p2k_play_model_with_lineups.py, outputs artifacts/mod57_p2k/.

## State (2026-11-11 09:43, read/measured)
- Novelty (read): no play-level model with on-field lineup inputs graded at the opener exists. MOD-22 (done/players-on-field-rating.md): game-level availability-weighted depth-chart rating sums, added to the four-term card only. MOD-23 unit 2: six MOD-22 arms on the base ridge 792-798 v 802 (-0.13 to -0.93 pts, unresolved). PER-09 apm_unit_opener_eval (artifacts/apm_unit_opener_eval, src/nfl_ats/apm_unit_feature.py, docs/apm_unit_feature_on_production.md): ridge APM per-player-in-lineup fitted on plays, but collapsed to 6 game-level team off/def unit columns added to the 88 inputs: 1,503 games, 53.0% v 53.4% (-0.40 pts [-1.86, +1.06], P+ .28). All game grain, additive ridge; none is a play-level tree with interactions. The APM artifact is game-level only (features.parquet), so it holds no per-player as-of ratings to truncation-check; the substitute below is built.
- DECLARATION (before any outcome): 
  - Plays and profiles: P2-H play_table (analysis_plays, 2009-25), joined to participation (data/players/participation/raw latest, columns offense_players/defense_players GSIS ids, ';' separated) on game_id, play_id; training uses 2016 onward.
  - Player as-of rating, by side (offense or defense): (S + k*mu)/(n + k) where S, n are the sum of EPA and count of that player's prior plays on that side from games on earlier dates (career since 2016, no decay), mu = mean EPA of all earlier-date plays on that side, k = 200 prior snaps = value_prior_snaps (src/nfl_ats/players.py:1633, reliability n/(n+k) at players.py:1461). No new constant. Defense rating is EPA allowed (higher is worse for the defense).
  - Position groups from weekly_rosters (player's modal position that season, else modal overall): offense QB, OL (OL,T,G,C), REC (WR,TE), RB (RB,FB); defense DL (DL,DE,DT,NT), LB (LB,OLB,ILB,MLB), DB (DB,CB,S,SS,FS). Group input = mean as-of rating of that group's players on the field (training: actual players that play; reference situation in a test game: expected lineup).
  - Expected lineup for a test game, nothing from that game: players of the team's previous game (previous season's last game for a first game) weighted by snap share (plays on field / team plays on that side), rating as of the test game date. No previous game with participation: group missing, HGB native NaN handling.
  - Models: HistGradientBoostingRegressor defaults, random_state SEED (per_flag_weights_eval), target epa, walk-forward test seasons 2020-25 trained on 2016..S-1. K1 = P2-H FEATURES + 7 groups; K2 = off_is_home + SITUATION + 7 groups; REF = P2-H FEATURES retrained on 2016+.
  - Roll-up and grading as P2-H (reference situations of S-1, margin_hat, standalone v Tuesday opener on 1,503 / quiet 837 beside served 802-701 / 439-398, LOSO four-term plus r at 829-674, controls intercept-only and shuffled r 50 draws). Looks: K1, K2 x 8 = 16, family mod57_play_level_matchup_model. REF graded, not recorded (it is P2-H's family on a shorter window; 8 more cells recorded under its own name only if meaningful).
  - Truncation checks: 12 player-games and 12 team-game expected lineups recomputed from data filtered to earlier dates.

## Tried (2026-11-11 09:50, measured, artifacts/mod57_p2k/results.json)
- Coverage: 328,362 scrimmage plays 2016-25, 328,353 with participation; 99.99% of slots have a position group; training rows with all 7 groups 98.3%; expected lineups non-null 99.9-100% on test team-games. Truncation: 12 player-games max diff 1e-17; 12 lineups (84 group values) max diff 2e-17.
- Standalone v opener, all 1,503 (served 802-701): REF 2016+ 758-745, K1 737-766 (-4.31 pts [-7.85,-0.92] P+ .007), K2 731-772 (-4.71 [-7.7,-1.74] P+ .001). Quiet 837 (served 439-398): REF 402-435, K1 402-435, K2 417-420.
- Added term v 829-674: REF -0.47 pts P+ .14; K1 -0.20 [-1.62,1.21] P+ .37 (log loss P+ .52); K2 -1.14 P+ .02. Quiet: K1 +1.07 [-0.96,3.09] P+ .83 (log loss P+ .63, changed 89 picks 49-40), K2 -0.36 P+ .30. K1 coefficient negative in 6 of 6 folds. Shuffled-r mean ~54.7-54.9% all.
- 16 cells recorded unresolved_below_power, family mod57_play_level_matchup_model. REF cells not recorded.
- Player ratings are on-field EPA means, so they carry team context; they did not add information beyond team form.

## Next
- Done. Next unit: play-level model whose player inputs are not raw on-field EPA (isolated player effects, e.g. APM-style ridge ratings as of date, or QB/OL/CB identity with within-team variation), or situation data not in the line.

## Open
- None.

## Status 2026-11-11 09:46
- Script written, ruff clean. Launched once in background (stdout artifacts/mod57_p2k/run.out, stderr run.err). Output: results.json, batch.json, margin_hat_{ref,k1,k2}.parquet, expected_lineups.parquet, play_table_lineups.parquet.
- Next command after it finishes: uv run --project F:\Repos\nfl_py3 --no-sync nfl-ats weak-signals record --batch F:\Repos\nfl_py3\artifacts\mod57_p2k\batch.json (from repo root). If run.err has a traceback, fix by hand and rerun.

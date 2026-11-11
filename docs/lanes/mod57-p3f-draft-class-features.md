# MOD-57 P3-F: draft-class value added (2026-11-11 12:05)

## Goal
Per-game input: expected rookie snap share added by a team's draft class, pick curve fit walk-forward. Build only; no outcomes, scores or lines read.

## State
Built (measured): artifacts/mod57_p3f/draft_class_features.parquet (REG games 2015-2026, game_id from data/processed/game_features.parquet, same as p2p), team_season_class_value.parquet, summary.json.
Columns: home_/away_/diff_ x exp_off, exp_def, exp_all (off+def), exp_all_r12 (rounds 1-2 only), n_picks; plus season, week, home_team, away_team.
Definition: rookie share = sum of offense_pct (or defense_pct) over the player's draft-season REG games / drafting team's REG games (snap_counts, pfr_player_id join). Curve = decreasing isotonic (y_min 0) on overall pick, fit separately on side O and side D picks; class sum uses side O picks with the offense curve, D picks with the defense curve; side S (59 picks) contributes 0.
Walk-forward (measured): curve for season S fit on classes 2013..S-1 only (2015 uses 2013-14); assert train.season.max() < S in code; summary.json walk_forward lists ranges. Draft-day trades: `team` column of draft_picks is the selecting team (read; picks carry the team making the pick, not the original owner); no original-team column used.
Coverage: all games have values 2015-2026 (2022: 271 games in game_features). Curves 2015 vs 2026 (off): pick1 .78/.84, 10 .68/.67, 32 .66/.56, 64 .42/.45, 100 .29/.29, 150 .15/.19, 200 .09/.09, 250 .01/.05. Def: 1 .64/.64, 10 .64/.57, 32 .44/.50, 64 .37/.39, 100 .20/.27, 150 .16/.19, 200 .12/.13, 250 .05/.05.
diff_exp_all: mean 0.0003, SD 0.754, min -2.67, max 2.67.
Join failures (measured): 3,331 picks 2013-25; 0 null pfr id; 511 have no snap row in their draft season (never played a regular-season snap that year; credited 0, a real value, not an id failure; inferred); 0 team-game mismatches. Team codes mapped to current abbreviations (TEAM_MAP).

## Tried
Single build; ruff check and format pass.

## Next
Score diff_exp_all as an input against the hindsight gap per P3-A design (weeks 1-6, walk-forward, shuffled-within-season control, placebo weeks 7-12); record via weak-signals.

## Open
Pick-1 offense curve is noisy at early seasons (isotonic on 2 classes); only 2013-14 train for 2015. 2026 rows use classes 2013-25 and picks from the 2026 draft file.

Rebuild: F:\Repos\nfl_py3\.tools\uv.exe run --project F:\Repos\nfl_py3 --no-sync python F:\Repos\nfl_py3\scripts\mod57_p3f_draft_class_features.py

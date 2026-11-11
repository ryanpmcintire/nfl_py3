# MOD-57 P2-H - play-level matchup model

## Goal
Base model that learns from plays (about 150 labels per game) instead of one margin per game; first read against the Tuesday opener. Parent: mod57-sixty-percent.md.

## State (2026-11-11 09:34)
- Novelty (read): no supervised play-level matchup model graded at the opener exists. Nearest: MOD-09 (ROADMAP:703, docs/play_level_audit.md: game-mean is a sufficient statistic, sequence-over-drives MLP P+ .27 as a term), MOD-24 u5 (EPA totals as game target), MOD-23 u3/u5 (additive opponent adjustment, coverage matchup), trees on 88 game inputs, SIM-10/11 resampler. Different: target is play EPA, offense profile x defense profile interaction learned by HistGradientBoostingRegressor.
- DECLARATION (before any outcome): plays = analysis_plays of load_pbp_snapshot(latest data/pbp/raw, regular season), 2009-2025 (pass/rush, non-null epa and wp, no kneel/spike/aborted; row counts per season printed by the script). Profiles = EWM span 8, min_periods 3, offseason retention 0.67 to league mean (pbp.py enrich_with_pbp_features span=8/min_periods=3, _build_pbp_states 720-766, constants.py:9), state from last team game strictly before the game date; metrics per offense and defense-allowed: pass EPA/play, rush EPA/play, pass success, rush success, pass explosive (>=20 yd), rush explosive (>=10 yd) (pbp.py:613-616), sack rate per pass play, turnover rate per play, early-down pass rate. Truncation check on 12 team-games.
- Play model: sklearn HistGradientBoostingRegressor, library defaults, target epa; features off profile, def profile, offense-home flag, down, ydstogo, yardline_100, half seconds, score differential. Walk-forward: season S trains on seasons < S.
- Roll-up: reference set = all scrimmage plays of S-1; margin_hat = (mean pred EPA home off v away def minus away off v home def) x S-1 mean scrimmage plays per team-game. Never the test game's own plays.
- Reads: (1) correlation/RMSE vs margin and line; (2) r = margin_hat - tue_open_home_spread, side = sign(r), all 1,503 and quiet 837 (abs(close-open)<=0.5), beside the served model (802-701, 441-396); (3) r added to the four terms (BASE) in lead98 loso fit; vs 829-674; controls intercept-only and r shuffled within season.
- Looks: 8, family mod57_play_level_matchup_model. Standalone cell effect = standalone accuracy minus served model accuracy (paired week-block bootstrap, 10000 draws, SEED from per_flag_weights_eval). Added-term cells = minus four-term.
- Code: scripts/mod57_p2h_play_level_matchup_model.py; outputs artifacts/mod57_p2h/.

## Tried (2026-11-11 09:38, measured, artifacts/mod57_p2h/results.json, run.out)
- 553,859 scrimmage plays 2009-25 (about 32-34k per season); truncation check 12 of 12 team-games identical; sklearn 1.9.0 HGB defaults (random_state set).
- Descriptive 2020-25: corr(margin_hat, margin) .358 v line .435; corr with line .767; RMSE 13.55 v line 12.97; margin_hat std 5.3.
- Standalone sign(r): 787-716 (52.4%) all; 428-409 (51.1%) quiet; served 802-701 / 439-398 (model_correct; model-only LOSO 441-396). v served: -0.97 pts [-4.4, +2.5] P+ .28 all; -1.31 [-5.9, +3.4] P+ .27 quiet.
- Added term (LOSO, same ridge): 835-668 v 829-674 all (+0.40 pts [-1.2, +2.0] P+ .67; log loss P+ .62, Brier P+ .63; changed 82-76); quiet 439-398 v 430-407 (+1.06 [-1.1, +3.3] P+ .81; log loss P+ .48, Brier P+ .48; changed 52-43). Fold coef .011-.023, 6 of 6 positive. Shuffled r (50 draws): 54.9% all (sd .5), 51.6% quiet.
- 8 cells recorded unresolved_below_power, family mod57_play_level_matchup_model.

## Next
- Profiles are the same 9 team summaries, so the model sees no new team information; next unit needs player-level or situational inputs the 88 columns lack (e.g. QB/OL/CB identity profiles), not a bigger tree.

## Open
- None.

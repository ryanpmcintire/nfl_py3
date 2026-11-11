# MOD-57 P2-C - uncorrected-error base model

## Goal
Base model aimed at what the market does not correct; quiet 837 games (|open move| <= 0.5) are the primary population (served model alone 441-396, four-term 430-407). Target about 57% there. Parent: mod57-sixty-percent.md.

## State
- 2026-11-11 09:32 novelty check done BEFORE any outcome read. READ: features.py:49 ats_margin = result - spread_line, spread_line = nflverse closing line; fit_margin_model trains on it (clv.py opener_pick_evaluation, target market_residual); only the scoring frame swaps spread_line to the Tuesday opener. So the served base is ALREADY trained on margin minus the close, not minus the opener. mod23-unit7 (done, 2026-10-16) measured opener-trained 801-702 v close-trained 801-702, P+ .43/.45. Arm K as briefed therefore equals the served model by construction; it is not a separate run (0 looks). Its test-time variant (expected margin at the test game's own close) needs the close as a feature = leakage, excluded; per_game.residual_at_open vs residual_at_close shows the line input is nearly inert (diagnostic only).
- MKT-20 (done/line-move-target.md): ridge predicting close-minus-open from opener info; refuted as a move forecast, implied pick -3.66 pts P+ 0. Different target (the move), not scored on quiet games. LEAD-86 (lead86.md): price-filtered market-regularised training toward the market lattice, weights 0/0.1/1; 721-588 v 757-552, acc -2.75 P+ .0003; trains toward the market, not on the quiet subset. Neither fitted Arm Q (train only on past games whose close was within a half point of their opener). grep of docs/lanes, done/, registry/weak_signals, ROADMAP found no quiet-trained base. Arm Q is novel.
- DECLARATION (before outcomes). Harness: clv.opener_pick_evaluation on data/processed/game_features_weak_stack.parquet, ridge alpha 10, weak_stack, script-local patch of clv.fit_margin_model only. Baseline must reproduce 802-701 on 1,503 and 441-396 on quiet 837, else stop.
  Arm Q: training rows restricted to past games with |spread_line - opener| <= 0.5; opener = SBR open (data/processed/sbr_odds.parquet) for 2011-2019, archive Tuesday opener 2020+; games with no opener (pre-2011) cannot be classified and are excluded. Control Qc: same 2011+ known-opener rows, no quiet restriction (separates the window change from the quiet restriction). Test-game close never enters features (scoring frame untouched); training windows logged per fold and checked max training gameday < week start for sampled weeks.
  Controls: intercept-only (LOSO) and line-move-only (side the line moved toward, model side when no move). Four-term LOSO with Q logit in place of model_logit, all games, against 829-674.
  Metrics: decisive record v served, accuracy pts / log loss / Brier week-block bootstrap (4000, probability_positive), per season, in-sample v out-of-sample gap, corr(residual, open_move). Looks: Q x {quiet, all} x 3 metrics = 6, plus four-term Q x 3 = 9; Arm K 0. Family mod57_uncorrected_error_model.

## Tried
- Nothing run yet.

## Next
- Write scripts/mod57_p2c_uncorrected_error_model.py, run once (background), record cells.

## Open
- None.

## Update 2026-11-11 09:34
- Script written, ruff check and format clean: scripts/mod57_p2c_uncorrected_error_model.py. Run launched once in background (tool handle b80qvzuzn), stdout tests/scratch/mod57/p2c/run.out, stderr run.err; it writes tests/scratch/mod57/p2c/results.json and cells.json (9 cells).
- Next: when results.json exists, check baseline reproduction (802-701 all, 441-396 quiet), then record cells with nfl-ats weak-signals record --project F:\Repos\nfl_py3 (batch as tests/scratch/mod57/u1c_batch.json), family mod57_uncorrected_error_model, unresolved_below_power; then report. If run.err shows a traceback, fix by hand and rerun once.

## Result 2026-11-11 09:38 (measured, tests/scratch/mod57/p2c/results.json; 9 cells recorded unresolved_below_power, family mod57_uncorrected_error_model)
- Baseline: all 1,503 reproduced (model 802-701, four-term 829-674). Quiet 837: model alone 439-398 (not the briefed 441-396; four-term 430-407 matches). Discrepancy unexplained, 2 wins.
- Arm Q (quiet-trained), quiet: 424-413 (50.7%) v model 439-398; acc -1.79 pts [-4.93,+1.56] P+ .14; log loss and Brier P+ 0.00. All: 760-743; -2.79 [-5.21,-0.34] P+ .01. Four-term with Q logit: all 811-692 v 829-674 (-1.20 [-3.01,+0.60] P+ .08); quiet 423-414 v 430-407.
- Qc control (2011+ known-opener rows, all kept): quiet 447-390 (+0.96 [-0.59,+2.52] P+ .87), all 802-701 (0.00); four-term Qc all 836-667 (+2.26 vs model alone, P+ .96 acc, LL/Brier P+ 1.00; v served four-term +7 wins, not a declared cell). Quiet restriction hurts v Qc (-2.75 quiet P+ .04). Fewer rows (training 1,226-2,059) is the mechanism candidate (inferred).
- Move correlation (residual v open_move): served .093 all/.069 quiet, Qc .076/.063, Q .037/.046. Arm K: residual_at_open v close agree on 98.5% of picks, mean abs diff 0.068.
- Leakage: 0 training fits with max gameday on or after week start; test closes untouched.
- Next: Qc four-term (836-667) is an unscored lead from the 2011+ known-opener window; needs its own declared unit and LOSO check before any claim.

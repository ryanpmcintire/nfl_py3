# MOD-57 P2-P forecast the hindsight gap

## Goal
How much of the line's measured error (hindsight rating margin minus close line) is visible before kickoff, training on that low-noise label. Parent lane mod57-sixty-percent.md. Script scripts/mod57_p2p_forecast_the_hindsight_gap.py, outputs artifacts/mod57_p2p/. Family mod57_how_wrong_is_the_line.

## State
2026-11-11 10:20:40 DECLARED before any outcome read.
- Novelty (read): grep of ROADMAP, docs, lanes, scripts, registry found hindsight only in P2-O (rating as ceiling) and the parent lane listing this unit; no model trained on an other-games hindsight label and graded on real results exists. P2-N trained on the raw residual; MOD-24 u5 used same-game totals; MKT-20 line move.
- Label L(g) = H(g) - close line, H from artifacts/mod57_p2o/per_game_ratings_LEAK_NOT_SERVABLE.parquet (own game removed). Train on seasons < test season only; print max label season per test season. Test-game H never an input or label.
- Features: the 295 columns, built as P2-N (same join, 8 tables, outcome columns home_cover/home_score/away_score removed). Truncation: rebuild game_features_pbp from pbp truncated before a cutoff date for 5 dates (>=10 games) and compare rows of games on that date; other tables not rebuildable in budget, reported untested, screened by correlation with outcome.
- Arms: EL = ElasticNetCV(TimeSeriesSplit 5) on L; GL = HistGradientBoostingRegressor defaults with early stopping on chronological inner split (last 2 training seasons), trained on L. Reference P2-N R,S,G,H,E from artifacts/mod57_p2n/oos_predictions.parquet.
- Read 1: OOS R2 of prediction on L (1 - SSE/SST around zero-mean label? reported as 1-SSE/sum(L^2) and corr^2), season-block bootstrap, overall and thirds; families by source table (EL nonzero coefficients, GL permutation importance).
- Read 2: scale = slope of real residual on L fitted on training seasons (label-to-result slope, in-training), pred = scale * prediction; MAE/RMSE of resid - pred vs line, paired diffs with season-block bootstrap and probability_positive; slope of real residual on prediction; mean abs prediction; pick accuracy vs close 2015-25 overall, by season third, by |disagreement| (<1,1-3,>3); vs Tuesday opener 2020-25 vs served 802-701.
- Looks: 2 arms x 4 = 8 primary plus accuracy cells.

## Tried
none yet

## Next
Write script, run truncation then main run, record batch.

## Open
none

2026-11-11 10:22:14 script written, ruff clean; main run launched in background (bash id b2qwdkg6g, outputs artifacts/mod57_p2p/run.out). Truncation check run separately after: script arg 'truncate'. Then record artifacts/mod57_p2p/batch.json.

2026-11-11 10:22:38 first run b2qwdkg6g failed (zip length: imputer drops all-NaN columns); fixed by using observed columns; rerun launched bg2o4b08s. Then truncate mode, record batch.json.

2026-11-11 10:24:07 second run failed at evaluation (empty gap bin in accuracy_block); folds all trained fine (max label season = test season - 1 confirmed in run.out). Fixed; third run launched in background. Then truncate mode, record batch.json.

## Result (2026-11-11 10:26:48, measured, artifacts/mod57_p2p/results.json)
Max label season in training = test season - 1 for all 11 folds (run.out). 2,895 test games.
Label fit (1 - SSE/sum L^2): EL .104 [.059,.145], GL .074 [.037,.109]; weeks 1-6 EL .002, GL -.021; 7-12 EL .129, GL .099; 13-18 EL .309, GL .261. Families: EL abs-coef share game_features .51, pbp .19, player .16, line itself .13; GL permutation MSE rise game_features 6.5, line itself 2.9, player .06, pbp .02. Caveat inferred: the label contains minus the line, so the line as input fits it mechanically.
Real results (resid - scale*pred vs line): EL MAE 9.840 vs 9.818 (diff -.022, P+ .002), sq diff -.205 (P+ .245), slope .29 [-.31,.90], mean abs pred .545, close 1403-1419 (.497), open 740-763. GL MAE 9.845 (diff -.027), sq -.002 (P+ .456), slope .50 [-.14,1.01], close 1422-1400 (.504), open 752-751 (.500) v served 802-701. P2-N arms: R .509, E .502 close.
12 cells recorded (family mod57_how_wrong_is_the_line, shard updated 10:26). Truncation (pbp rebuild, mode truncate) running: artifacts/mod57_p2p/trunc.out.

2026-11-11 10:29:55 DONE. Truncation pbp (measured): 5 cutoffs, 52 games, 103 pbp columns, max abs diff 0.0, no column fails. Other 6 tables untested (rebuilders not run). Next: leave-game-out unit rates from raw pbp, or opener-trained residual.

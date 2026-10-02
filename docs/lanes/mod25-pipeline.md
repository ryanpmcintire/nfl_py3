# mod25-pipeline

## Goal
End-to-end synthetic-training path (MOD-25 units b, c) on generator variant crp04 (scale .75) so the real test runs the moment the generator passes. ALL RESULTS PROVISIONAL (generator failed gates); nothing recorded in registry. Parent docs/lanes/mod25-synthetic-data.md.

## State
Code: scripts/mod25_pipeline.py (commands produce, features, student [--all-lambdas --workers N]). Imports generator read-only (gen sha in each chunk meta.json).
Produced (measured): 6 chunks x 6 worlds x 10 seasons = 97,920 games + test chunk_90 (6,528) in data/processed/synthetic/crp04/chunk_*/ (games, team_stats, plays, latents.npz, meta). About 4-9 min per 16k-game chunk, about 33 min total.
Features (measured): data/processed/synthetic/crp04/features.parquet, 104,448 games, 42 worlds, via production build_game_features per world, two passes (pass 1 no line -> book -> pass 2 with line so ats/bias features exist). Not producible and zero-filled: 23 player columns (qb, injuries, continuity, injury values), plus rest_diff, neutral_site, div_game, temp, wind, cpoe x3; playoff-holdover bias is 0. Producible: market, elo, experience (games in season), offense/defense (EPA, ypp, turnover, sack), results, bias prior-week. First week of each world NaN (imputed).
Book (measured, artifacts/mod25_pipeline/features_report.json): ridge on 40 producible features fitted to real SBR openers 2009-2019 (n 2816): spread R2 .78 in, .77 leave-one-season-out, resid SD 2.76; total R2 .52/.50. Constant offsets from synthetic outcomes (spread -.28, total -3.36) calibrate it to the synthetic world. Synthetic line SD 4.9 vs real 6.0; margin SD 15.2 vs 14.8; ats SD 14.5 vs 13.4.
Student: ShrinkRidge (solves ridge toward synthetic raw coefficients rescaled to real scaler), lambda grid 10..1e5 chosen per test season on inner seasons (4 prior, fit on earlier). Grade via clv.opener_pick_evaluation with patched margin.make_margin_estimator, temp recal LOSO as in mod24_u2b (reimplemented), season-block bootstrap.
Pipeline smoke test (chunk_90 only prior, MEANINGLESS as a result): base reproduced 802-701; student 794-709, acc diff -0.53 (P+ .21), temp LL gain -.0007 (P+ .006).
IN FLIGHT at cap: full run `student --all-lambdas --workers 6` (background id bkugch6oi) writing artifacts/mod25_pipeline/student.log, results.json, selection.json, sign_agreement.json, synthetic_coefs.json, scored_lambda_*.parquet. Not read yet.

## Tried
Test run on 6.5k-game prior only. Full-prior numbers not yet seen.

## Next
Read artifacts/mod25_pipeline/results.json, sign_agreement.json, selection.json (check student.log for Traceback first; if the run died rerun the student command). Report record, acc diff, recal log loss, sign agreement per family. Then rerun when generator passes gates (re-run produce, features, student).

## Open
Weaknesses: (1) generator fails gates (SD, nonstrength variance, R2 late weeks, key-number mass), so prior is of a wrong world; (2) book prices from features only, not latent strength, so synthetic line carries no hidden information a real book has; line SD 4.9 vs 6.0; (3) 23 player columns plus weather/rest/div/cpoe are zero in synthetic, so the prior says nothing there and indicator columns differ; (4) synthetic ypp level about 6.4 vs real about 5.6 (scale shift absorbed by scaler); (5) book offsets use synthetic outcomes; (6) holdout-chunk synthetic R2 not computed in the test run (needs more than one chunk; the full run computes it); (7) lambda grid fixed 8 values, looks counted 8 fixed-lambda cells plus 6 selections; fixed-lambda curve is diagnostic only, not selection; (8) first season of each world is a warmup and dropped from training; (9) rounding of lines to half points but no key-number structure in prices.

## Provisional result 2026-10-01 (measured, orchestrator read; generator not passing gates, nothing recorded)
Full student run on 104k crp04 games. Shrink strength chosen out of season
picked the grid's weakest pull (10) in 5 of 6 seasons: 798-705 vs 802-701,
-0.27 pts, recalibrated log loss -0.00009 (P+ .10). Every stronger pull is
worse: lambda 100 gives 782-721, lambda 300 gives 780-723. Sign agreement with
the real ridge: offense 7 of 18 active inputs, experience 0 of 2.
Inferred design flaw beyond generator fidelity: the synthetic book prices from
features only, so the synthetic residual (margin minus line) mostly teaches
that fake book's misspecification, not the real book's errors.
Redesign for the rerun: use the synthetic world's noiseless labels (latent
expected margin) to learn features to expected margin, then feed that estimate
to the real residual model as one fitted term or prior. Don't learn a residual
against a synthetic book.

## v2 distillation on crj @ .9 (measured 2026-10-02; done, nothing committed)
Code: scripts/mod25_pipeline.py now targets crj/.9 (data data/processed/synthetic/crj, out artifacts/mod25_pipeline_v2). Commands verify, truth, student2, diag, grade --arm ridge_nomkt|hgb_nomkt (grade = TwoStage: base ridge + ONE term w*(distilled - line - base), w from leave-one-season-out base preds in training; 2020-25 via clv opener eval, 2011-19 via mod24_u1.proxy_eval).
Production: 8 chunks, 130,560 games, 40.6 min (features 48 worlds x 10 seasons, season 1 dropped). Mapping verified by 9,600 repeated games on 96 matchups: E[margin] = 1.70 hfa + 54.6*(off diff) - 54.7*(def diff), linear (quad coef 2.9), between-matchup R2 .96. True EM SD 7.97, oracle R2 vs realized .244.
Student (holdout worlds): ridge R2 vs true .513, vs realized .128 (realized-label ridge .128: no gain from noiseless labels linearly); HGB .510/.128. Market cols excluded (stored real line is the close, leaks at opener).
Grades vs base 802-701 (recorded, family mod25_distillation, unresolved_below_power): ridge 2020-25 805-698, acc +0.20 [-0.63,1.21] P+ .58, recal LL +.00016 P+ .68, seasons+ 3/6, flips 88-85; 2011-19 1156-1075 vs 1142-1089, +0.63 [-1.08,2.86] P+ .70, LL +.00101 P+ .96. HGB 2020-25 798-705 (-0.27, P+ .33, LL +.00021 P+ .71), 2011-19 1162-1069 (+0.90, P+ .89, LL +.00111 P+ .98). Weight w fell .69 (2010) to .19 (2025).
Diagnostics: distilled corr close .84, open .85; q=(distilled-open) corr with margin-vs-open .018 (2020-25), .004 (2011-19); partial given base .026; corr(q, close-minus-open move) .09 / .26.

## Line-move test of q (measured 2026-10-02; scripts/mod25_linemove.py, artifacts/mod25_pipeline_v2/linemove/)
Move = close - opener (Tuesday opener 2020-25 n 1537; SBR open 2011-19 n 2304), LOSO by season. Verified: 88 nomkt cols, none market-named; spread_line/total_line excluded; 34 cols constant in synthetic (player, weather, rest, cpoe) so zero weight. Real features read as pregame (walk-forward z from prior seasons).
Held-out slope of move on q: positive in 9/9 (2011-19, mean b .17-.18, corr .25-.26), 6/6 (2020-25, b .05-.06, corr .11-.12), 15/15 pooled. MAE vs zero-move: no gain (2011-19 +.002 P+ .53; 2020-25 -.015 P+ 0; pooled -.015 P+ .07); adding base edge control moves it to +.007 / -.009. Slope is real but too small to beat a zero-move MAE.
Direction (sign q): mean move toward q 2011-19 +.38 pts (hgb +.41), 2020-25 +.17 (hgb +.16), pooled +.30/+.31, P+ 1.00 (6-9 season blocks), seasons+ all; moved toward q 49/43% of games, away 32/33%, zero 19/25% (toward 60/56% of moved games).
Recorded 12 cells, family mod25_distillation_linemove, unresolved_below_power (registry widened the toward_q SEs). 24 looks counted (12 unrecorded: matched A and base-edge-control B arms, in linemove.json).
Inferred: q = rating-consensus minus opener, so move toward q may be public-rating reversion rather than information the opener lacks; it says nothing yet about margin vs open (corr .02).

## Next
Whether line value converts to cover accuracy: q partial given base vs margin-vs-open is .026 (diag). Pick toward q only enters as the one fitted TwoStage term (done, unresolved). Rerun all when generator passes gates.

## Open
Weak-signal cells: 8 grade looks + 24 linemove looks. Raw (unmatched) standardisation not graded. 2011-19 SBR open is not a Tuesday opener.

## Orchestrator check 2026-10-02 (measured, linemove_frame.parquet)
Line value gained by picking each side (mean move toward the side, points).
Base model: +0.05 (2011-19) and +0.09 (2020-25). Distilled q: +0.38 and +0.17.
Where q disagrees with base (54-55% of games), q's side still gains +0.31
(n 1252) and +0.07 (n 838). Since q's direction comes from a student trained only on
synthetic games, with no real outcomes fitted, its direction carries no
real-data selection. Inferred: the market moves toward the synthetic model's
view more than toward the base model's, which is information the opener
lacks, stronger in the older era. Cover accuracy stays below power. Next: log q
forward with the 2026 weekly picks, and a calibrated fitted term combining q
with the base model (out-of-season weight).

## Forward log of q (built 2026-10-02; scripts/mod25_forward_q.py, artifacts/mod25_forward/)
Frozen (measured) in frozen/frozen.json + student_ridge_nomkt.npz (sha256 53465c34...19ba): Ridge alpha 1 on crj/.9 true expected margin, 88 nomkt cols, real inputs z-scored with the 2025-final prior-seasons mean/sd; reproduces real_distilled to 1e-14. Combination w .1848, qbar -.2824, bbar .0614, fitted through 2025 (4431 games, LOSO base preds) by capturing the TwoStage fit of a 2026 Week 1 score; base ridge refits weekly, w/qbar/bbar fixed. Freeze refuses to rerun.
Run: `.tools\uv.exe run --no-sync python scripts\mod25_forward_q.py [--week N]` writes runs/2026-week-WW-<stamp>.json (mode x); refuses outcome-present and past-deadline games. Pool line = feature-table spread_line; market_line = median HOME quote (sign flipped to spread_line convention) from latest data/market/raw capture. `--status` = coverage + pool-to-close line move toward q/base/combined side (needs kicked-off games with a capture before kickoff; no outcomes).
Week 4 logged 2026-10-02 05:11 UTC: 15 games; PIT_CLE refused past_deadline. Combined side = base side = served side on all 15.
Schedule Thu ~19:00 ET and Sun ~11:00 ET Weeks 4-18 (not Tuesday before noon freeze). Canonical row = latest before deadline. Score only after Week 18 final.
Open: combined probability uses the combined model's own discrete read (both reads True); the frozen-w capture had 2 TwoStage fits (served-helper fit plus scoring fit); took the scoring fit (max train season 2025). No Week 4 tests added.

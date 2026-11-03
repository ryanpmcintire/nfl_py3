# MOD-24 pooled sweep (2026-11-03)

## Goal
Grade every learner-side MOD-23 lever expressible on the 14 shared NFL/CFB features through scripts/pooled_screen.py at once (NFL opener 2020-25, proxy 2011-19, CFB opener-matched), pooled where heterogeneity is admissible. Base: ridge alpha 10.

## State
Family declared before any run: mod24_pooled_sweep_2026_11_03. Callables in scripts/pooled_sweep_variants.py; every parameter is chosen inside the past-only training window (inner contiguous folds or last-season holdout over a declared grid, decision metric = sign accuracy of cover where noted); no tuned constants.
Variants (13 = looks per metric; metrics accuracy, log loss, Brier; pooled = nfl_opener + cfb):
ts and cv (reference, already recorded in mod24_pooled_levers, rerun for the table); recency (exp row weights, halflife by inner holdout); window (last K seasons, K by inner holdout); huber (epsilon by inner folds); winsor (label clip quantile by inner sign accuracy); enet (ElasticNetCV); hgb; hgb_resid (ridge + HGB on residual); ix_a10 (14 + all pairwise products, alpha 10); ix_cv (same, RidgeCV); accalpha (alpha by inner sign accuracy); slope (prediction scaled by inner out-of-fold slope).
Excluded (needs NFL-only data or pinned builder): per-input EWMA spans (unit 8/12, builder), matchup products, drive/PBP blocks, block penalties, label-reliability strata needing post-game fields.
Decision: unresolved_below_power unless whole interval wrong side; nothing served.

## Tried
All 13 run via pooled_screen.py (reports artifacts/pooled_sweep/*.json); enet needed a kwarg fix (alphas=20) and was rerun. 52 cells (nfl 2020-25 and cfb 2013-25, accuracy_points and log_loss_improvement) recorded, unresolved_below_power. Pooled accuracy pts [95%] P+ / LL P+: cv +0.68 P+1.00/.95; enet +0.79 [+0.14,+1.43] .99/.95; slope +0.76 [+0.16,+1.36] .99/.91 (LL het z -0.50); accalpha +0.46 [-0.04,+0.96] .96/.84; winsor +0.07; huber +0.10; recency +0.40 (LL .05); window, hgb, hgb_resid, ix_a10, ix_cv: LL P+ <=.09, hgb/hgb_resid/ix LL heterogeneity not admissible; ts -0.08. Brier not recorded (same direction as LL). Looks: 13 variants x 3 metrics = 39 pooled.

## Next
Declare MOD-23 unit: enet, slope, accalpha (and cv) on the 88 served inputs, one look each, nested in-window selection; see report.

## Open
Screen is Gaussian-pick, 14 inputs; a pass still needs the 88-input served-path grade.

## Root adjudication 2026-11-03 ~13:10
- 13 variants, 39 looks: the best cells (enet +0.79, slope +0.76, cv +0.68, accalpha +0.46) are not findings by themselves. Three of the four recommended arms were already run today on the full 88-input served model and lost 2020-25 picks (enet -3.19 recorded wrong_sign_resolved, cv -0.73, accalpha -1.80; mod23-unit16/18 lanes), so the shared-feature shrinkage gains do not transfer to the full model and no new 88-input unit is declared for them. slope rescales predictions; on the served market-residual target it mostly cannot change sides (inferred), so it is a calibration question for the served recalibration, not a pick lever.

## ROOT CORRECTION 2026-11-03 13:55
- The pooled harness used the closing spread_line as model input while grading at the opener (scripts/mod24_more_games.py:63-73), so these results include close-minus-open line movement unavailable on Tuesday; not evidence for the served model until the opener-input rerun (see docs/lanes/mod24-more-games-2026-11-03.md).

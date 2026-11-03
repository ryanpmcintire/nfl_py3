# MOD-24 more graded games (2026-11-03)

## Goal
Grade model variants on more independent games at the opener grade; measure the effective-sample multiplier and whether the arm ranking transfers.

## State
Chosen route: CFB (cfb_game_features.parquet, spread_open) as a second graded population for ridge-style arms. Missing-opener recovery for NFL is only +4%, not worth a unit. Standing screen built: scripts/pooled_screen.py (below).

Measured results (scripts/mod24_*.py, artifacts/mod24_*/report.json; all cells recorded, unresolved_below_power; families mod24_cross_league_alpha, mod24_pooled_levers):
- Full harness (79 NFL features vs CFB features): 5.5x games (NFL 1503, CFB 8294 decisive). Log-loss ordering of alpha arms transfers (rho 1.0); accuracy ordering reverses (-0.8), het z up to -2.8: CFB rewards shrinkage NFL does not. Not poolable.
- Size-matched CFB (260 rows/season): accuracy still reordered (a1000 het z 2.5), LL admissible-ish. Shrinkage benefit halved, so training size is only part of it.
- Common 14 features: accuracy Spearman NFL vs CFB 0.92 (0.97 opener-matched, CFB sbr_multibook 2012-19, 4702 games); acc het |z| < 1.4; multiplier about 4-5x (opener-matched, a1000-a1e4). Pooling admissible for arms on those 14 inputs only, not for NFL-only features. Not proof of an edge.
- Pooled levers (base ridge alpha 10, 14 shared features): postseason rows NFL only +0.13 pts P+ .60 (CFB not buildable, needs seasonType 3 pbp rebuild). TS weights pooled -0.08 pts [-0.28,0.13] P+ .23. RidgeCV pooled +0.67 [0.17,1.17] P+ 1.0, LL +0.00030 P+ .95, mult 3.7: the one lever consistent in both leagues. Not served.

## Tried
Alpha-only full harness; size-matched subsample (5 seeds); common-feature and opener-matched; pooled levers (a postseason, b TS weights, c RidgeCV). Record CLI exits nonzero on an unrelated unit entry (margin_sq_error_improvement) but writes land.

## Next
Use scripts/pooled_screen.py for any new variant expressible on the 14 shared features. CFB postseason rows need a src rebuild. Cells are recorded by the owning unit, not by the screen.

## Open
CFB opener is a first-observed or ESPN-era quote, not a Tuesday opener (proxy discount unmeasured). NFL harness is a Gaussian-pick harness, not the served discrete read. The screen is a Gaussian ridge-family screen on 14 inputs; a passing variant still needs the served-path grade.

## Standing screen: scripts/pooled_screen.py (infrastructure, no cells recorded)
Usage (venv python, one core, ~10-40 s per variant):
`.venv/Scripts/python.exe scripts/pooled_screen.py VARIANT [--nfl-eras opener|all] [--out PATH.json]`
VARIANT is a named arm (`ts`, `cv`) or `MODULE_OR_PATH.py:CALLABLE`, callable(a, y, b, train) -> (test prediction of ats_margin, residual sd); a, b are the imputed, standardized train/test 14-feature matrices, y train ats_margin, train the REG train frame. It runs via mod24_pooled_levers.walk/grade/contrast/pool (not forked), against ridge alpha 10 REG-only. Populations: nfl_opener (2020-25 Tuesday opener), nfl_proxy (2011-19 SBR open, min train 1000 rows), nfl_all, cfb (sbr_multibook 2012-19). Prints per population base vs variant record, flips W-L, accuracy/log loss/Brier gain with 95% block interval and P+; pooled inverse-variance effect with per-metric heterogeneity z (flagged NOT ADMISSIBLE at |z| >= 1.96). Pool is nfl_opener + cfb by default (`--nfl-eras all` uses nfl_all). Report: artifacts/pooled_screen/<name>.json.
Reproduction (measured 2026-11-03): `cv` pooled acc +0.68 [0.17,1.18] P+ 1.00 mult 3.5 (recorded +0.67 [0.17,1.17]; NFL 38-24 flips, CFB 99-72 flips identical; the 0.01 gap is bootstrap draw order, the screen reseeds each contrast). `ts` pooled acc -0.08 [-0.27,0.12] P+ .23 (recorded -0.08). Proxy era cv: +0.86 pts, LL +0.00173 P+ .99.

## Era check (measured 2026-11-03, family mod24_pooled_era_check, 24 cells recorded, unresolved_below_power)
CFB espn_book_era 2023-25 (2218 decisive, walk-forward on all prior seasons) vs NFL 2020-25 opener (1503), 14 shared features, arms a1, a100, a1000, a10000, cv, slope vs base a10. Base opener accuracy: CFB 52.98, NFL 52.30. Accuracy-gain Spearman 0.88 (absolute accuracy 0.84, log-loss gain 0.53, 6 arms). Per-arm accuracy het z vs NFL: -1.87 to +0.17, all under 1.96. CFB gains are small (a10000 +0.63 P+ .87, slope +0.27, a1000 and cv +0.09, a100 -0.05, a1 +0.05) and none resolved; NFL gains larger (slope +2.00, a10000 +1.73, a1000 and cv +0.93). The ordering agrees in direction, so the 2012-19 agreement is not era-bound by this test, but CFB 2023-25 magnitudes are 3-9x smaller and slope het z -1.87 is borderline; pool only with the per-arm het check. cv equals a1000 (same selected alpha). Report artifacts/pooled_screen/mod24_era_check.json.
- Era-check scripts: scripts/mod24_pooled_era_check.py (run) and scripts/mod24_pooled_era_check_batch.py (registry batch), moved from the root scratchpad 2026-11-03 13:43.

## Limit from XLG-13 (measured 2026-11-03, family xlg13_screen_transfer)
The pooled 14-input screen is not usable to pre-screen served-model changes. On 4 candidates with both NFL 2020-25 opener gains, the screen was positive for all and the 88-input model negative for all (sign agreement 0/4); Spearman -0.40 (bootstrap P+ .25), Pearson -0.84 (P+ .21). Intervals span -1 to 1 at n=4, so unresolved_below_power, not a closed mechanism. Shrinkage and calibration levers gain on 14 inputs and lose picks on 88. Use the screen only for 14-feature questions; every served candidate needs its own 88-input grade. Table in docs/lanes/xlg-13-xlg13-screen-transfer.md.

## ROOT CORRECTION 2026-11-03 13:55 (design leak)
- Read, scripts/mod24_more_games.py:63-73: the pooled harness predicts pred_margin = spread_line + pred and grades edge = pred_margin - open, with spread_line taken from the feature table (nflverse spread_line, a closing line; inferred from the source). Every pooled pick therefore partly follows the close-minus-open move, which is not known at the Tuesday pick. XLG-11 u3 confirms it: an intercept-only arm (edge = close - open + mean) wins 815-688 and agrees with the best arm on 86% of games.
- Consequence: today's pooled results are not Tuesday-information results. Shrinkage levers looked good because more shrinkage moves the prediction toward the closing line (more line-following); that is why XLG-13 found 0/4 sign agreement with the served Tuesday model. Invalid as evidence for the served model until rerun with the opener as the input line: mod24_cross_league_alpha (commonfeat, sizematched, openermatch), mod24_pooled_levers, mod24_pooled_sweep_2026_11_03, mod24_pooled_era_check, xlg10_pooled_addone(+unit2), xlg11_cfb_centered_ridge (including the head-to-head v served), xlg13_screen_transfer. The registry cells stay as recorded measurements of a leaky design; the rerun supersedes them.
- XLG-12 forward ledgers are not affected in the same way (they take the captured pregame line as input), but its retro used the same close-as-input pattern and was already corrected.

## OPENER-INPUT RERUN (measured 2026-11-03, families mod24_pooled_era_check / xlg10_pooled_addone / xlg11_cfb_centered_ridge, ids suffixed _openinput; 58 cells, unresolved_below_power)
Fix: scripts/mod24_more_games.py to_opener/nfl_open (default on; MOD24_CLOSE_INPUT=1 reproduces the old closing-line input). Frame spread_line and ats_margin = result - open for train and test rows; NFL open = Tuesday opener 2020-25, SBR open 2009-19; CFB spread_open. Read: before the fix, spread_line was the feature-table line (features.py:49 ats_margin = result - spread_line; mean(close-open) NFL 0.10, CFB std 4.3). Walks in mod24_more_games.py, mod24_pooled_levers.py (so pooled_screen, sweep, era_check, xlg10, xlg11 all inherit). Residual leak: total_line is still the closing total (no opener total in frame).
Base accuracy at opener (Gaussian harness): NFL 2020-25 50.17% (754-749), CFB 2023-25 51.80, CFB 2012-19 50.71. Served model on the same NFL 1503: 52.83% (794). Old closing-input base was 52.30.
Accuracy pts vs base alpha 10 [95%] P+ (NFL 2020-25 | CFB 2023-25 | CFB 2012-19):
- a1: -0.07 [-0.81,0.67] .41 | -0.05 [-0.15,0.00] .00 | +0.16 [-0.06,0.40] .91
- a100: +0.20 [-0.53,0.94] .67 | +0.05 [-0.42,0.46] .54 | +0.60 [0.27,0.95] 1.00
- a1000: +0.93 [-0.26,2.14] .93 | -0.05 [-0.90,0.72] .43 | +0.45 [-0.23,1.15] .89
- a10000: +1.00 [-1.21,3.04] .81 | +0.50 [-0.94,1.82] .74 | +1.07 [0.14,2.07] .98
- cv: +0.93 [-0.26,2.14] .93 | -0.05 [-0.90,0.72] .43 | +0.41 [-0.25,1.11] .87
- slope: -1.00 [-2.55,0.07] .03 | -0.05 [-0.21,0.09] .19 | -0.06 [-0.55,0.41] .40
- enet: +0.60 [-1.47,2.47] .71 | -1.26 [-3.31,0.70] .09 | +0.66 [-0.25,1.68] .91
- intercept-only: +0.20 [-2.65,2.90] .52 | -1.44 [-3.53,0.63] .08 | +0.84 [-0.77,2.45] .86 (intercept-only NFL 757-746 = always away)
- XLG-10 bye_diff (pooled 15 features): NFL +0.07 [-0.40,0.53] .57; CFB 2013-25 +0.09 [-0.27,0.45] .67; pooled +0.08 [-0.20,0.36] .71; LL pooled -0.00007 P+ .06.
- XLG-11 u1 (CFB-centered ridge, NFL only): 2020-25 +1.13 [-0.79,3.06] .87 (771 vs 754), LL -0.00024 P+ .32; 2011-19 -1.20 [-3.21,0.63] .10, LL +0.00174 P+ .96. u3 controls NFL 2020-25: u1 771 vs away/intercept 757, fav 735, home 746.
Cross-league Spearman (8 arms incl. base): accuracy-gain NFL vs CFB 2023-25 0.31 (was 0.88), vs CFB 2012-19 0.59; absolute accuracy 0.18 / 0.70 (was 0.84); LL-gain -0.45 / 0.43 (was 0.53). Per-arm het z all under 1.4.
Reads: with the opener as input the Gaussian ridge harness is a coin flip at the opener (NFL 50.2, CFB 50.7-51.8), so the 52.3 base and most gains in the earlier pooled work were close-minus-open movement. Surviving: nothing is established; CFB 2012-19 shows a100 +0.60 P+ 1.00 and a10000 +1.07 P+ .98 (shrinkage helps in the larger CFB sample), NFL a1000/cv +0.93 P+ .93 is directionally consistent but unresolved. Not surviving: the 0.88 cross-league rank transfer and the RidgeCV/TS/postseason pooled-lever conclusions (not rerun; they used close input, treat as void until rerun), the 14-feature accuracy ranking, XLG-11 u1 and XLG-10 bye_diff positive readings (u1 NFL P+ .87 on 2020-25 is a +17-game swing and flips sign on 2011-19). Old cells keep their history and are superseded by the _openinput ids.

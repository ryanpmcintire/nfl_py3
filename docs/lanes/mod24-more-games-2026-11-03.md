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

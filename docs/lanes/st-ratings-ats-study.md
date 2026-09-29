# ST ratings ATS study

## Goal
Test one season-lagged ST differential beside the four served inputs, graded at the opener.

## State
**Measured:** single run exit 0; 1,503 games / 107 week blocks, 632 covered. On 69 differing picks: baseline 40-29, candidate 29-40. Log-loss improvement -0.00298055, 95% interval [-0.00662267, +0.00015283], probability_positive 0.03075.
**Inferred:** unresolved_below_power under AGENTS.md closing grounds; registry pending. Candidate IS/OOS log loss 0.68250836/0.68701075; IS-minus-OOS -0.00450238. ST coefficient positive 1/6 folds.

## Tried
**Measured:** recovered declaration before scoring; preserved sections 1-8. REG 2020-2025, nonpush opener target, fixed L2, season LOSO, one added ST term, 20,000 season-week draws; six OOS cells / 86 total declared looks. Full coverage, coefficients, baselines and calibration: docs/st_ratings_ats_study.md section 9.
**Measured:** ran .tools/uv.exe run --no-sync python scripts/st_ratings_ats_study.py once (UV_CACHE_DIR=.tmp/uv-cache); empty stderr. Syntax/comments/docstrings, declaration/script hashes and whitespace passed. Prediction rows: .tmp/st-ratings-ats-study.log; aggregate arithmetic agrees within 2.6e-12. Checked weak-signals record --help and PowerShell command syntax; no registry write.

## Next
Orchestrator: review report, retain temporary prediction log, run the six commands below serially, then commit the three assigned files.

## Open
Coverage remains sparse in early years; LOSO is retrospective and bootstrap conditions on fitted predictions. No refit, promotion, card change, new tests, or commit performed.

## Record commands
Exact commands, **not executed**. All six are correlated cells; none establishes a closing ground.

```powershell
$env:UV_CACHE_DIR='.tmp/uv-cache'
$stCommon = @('--league','nfl','--season-end','2025','--family','st_rating_ats_study','--classification','unresolved_below_power','--source','docs/st_ratings_ats_study.md','--category','onfield','--description','One lagged ST term added to the four-term fitted opener probability; LOSO.','--classification-evidence','No resolved wrong sign or powered positive control established.','--notes','Six correlated OOS cells; 86 total declared looks; 20000 season-week draws; unserved.')
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_all_log_loss --season-start 2020 --effect -0.002980551372153013 --effect-units log_loss_improvement --interval-low -0.0066226700518552664 --interval-high 0.0001528280301996564 --probability-positive 0.03075 --sample-games 1503 --sample-blocks 107
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_all_brier --season-start 2020 --effect -0.0014049854928483401 --effect-units brier_improvement --interval-low -0.00314813178000208 --interval-high 0.00010285268115910278 --probability-positive 0.0327 --sample-games 1503 --sample-blocks 107
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_all_accuracy --season-start 2020 --effect -0.73186959414504327 --effect-units accuracy_points --interval-low -1.6427340977175122 --interval-high 0.19841598121710599 --probability-positive 0.058924999999999998 --sample-games 1503 --sample-blocks 107
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_covered_log_loss --season-start 2022 --effect -0.0071208459120282262 --effect-units log_loss_improvement --interval-low -0.015294009109094429 --interval-high 0.00040325958872591074 --probability-positive 0.032199999999999999 --sample-games 632 --sample-blocks 54
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_covered_brier --season-start 2022 --effect -0.0033578744671419925 --effect-units brier_improvement --interval-low -0.0072722908032944204 --interval-high 0.0002569366431942202 --probability-positive 0.035499999999999997 --sample-games 632 --sample-blocks 54
.tools/uv.exe run --no-sync nfl-ats weak-signals record @stCommon --name st_rating_ats_covered_accuracy --season-start 2022 --effect -1.740506329113924 --effect-units accuracy_points --interval-low -3.8110091374007578 --interval-high 0.34784123617633561 --probability-positive 0.053925000000000001 --sample-games 632 --sample-blocks 54
```

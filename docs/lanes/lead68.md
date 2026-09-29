# LEAD-68: Model slope by weeks of information

## Goal
Execute ROADMAP.md LEAD-68 unit 1 within the assigned script and report files.

## State
Protocol frozen before outcome access. **Read:** ROADMAP.md:851.
**Measured:** unit exited 0 once; four-term reproduction error 1.67e-16.
**Inferred:** unresolved_below_power; registry writes await the orchestrator.

## Declared protocol
"Predeclared: replace the single `a*logit(model)` with three phase slopes, weeks 1-4,
5-9, 10-18 (cut points fixed by the offseason-window logic already used in LEAD-65,
never read from outcomes), same LOSO by season on 1,503 games 2020-2025,
base = served four-term model, 3 looks (one per phase slope) plus the joint fit,
metrics log loss, Brier, accuracy, reliability table, per-fold coefficients.
Not a threshold rule and it flips nothing alone."
Reuse four_term_probability_eval.py settings, opener target and population;
four looks, one joint candidate. Report train/held-out/gap, decisive W-L,
fold coefficients and reliability. Auxiliary declaration: 2,000 paired whole-season
bootstrap draws, seed 68, 95% intervals; positive = baseline-minus-candidate losses
or candidate-minus-baseline accuracy. No tuning. Zero crossing closes nothing;
one fitted probability selects the side.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead68_unit1.py` (one run;
UV_NO_CACHE=1, PYTHONDONTWRITEBYTECODE=1, BLAS/OMP threads=1). Scoped Ruff lint and
format checks passed. Cache: artifacts/four_term_probability/20260914T222345Z/per_game.csv.
**Measured:** candidate 840-663 versus 847-656; decisive 57-64 on 121 games.
Log-loss gain 0.000339 [-0.001465, 0.002021], probability_positive=0.6595;
accuracy gain -0.465735 points [-1.797603, 0.593276], probability_positive=0.25575.
Candidate train/OOS log loss 0.681508/0.683693 (gap +0.002185); all metrics,
six-fold coefficients, working intervals and reliability: docs/lead68_unit1.md.

## Next
Orchestrator reviews and runs the commands below serially; no serving change.

## Open
Historic cache and six bootstrap clusters limit inference; LOSO is retrospective.
docs/lead68_predictions.md preserves 1,503 prediction rows locally: DO NOT COMMIT
that processed-data file. Script/report/lane are reviewable deliverables.

## Record commands
Not executed. Shared arguments plus these exact PowerShell commands:
```powershell
$lead68Args = @('--source','docs/lead68_unit1.md','--description','Joint model-logit slopes for weeks 1-4, 5-9, 10-18 versus four-term LOSO baseline','--classification','unresolved_below_power','--league','nfl','--season-start','2020','--season-end','2025','--sample-games','1503','--sample-blocks','6','--family','lead68_unit1','--category','modeling','--classification-evidence','No admissible closing ground; four declared looks; 2000 paired season bootstrap draws, seed 68')
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead68Args --name lead68_joint_log_loss --effect 0.000338511692751 --effect-units log_loss_improvement --interval-low -0.001464647300345 --interval-high 0.002021299798674 --probability-positive 0.6595
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead68Args --name lead68_joint_brier --effect 0.000136239001138 --effect-units brier_improvement --interval-low -0.000749954878646 --interval-high 0.000965837098743 --probability-positive 0.6305
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead68Args --name lead68_joint_accuracy --effect -0.465735196274118 --effect-units accuracy_points --interval-low -1.797603195739015 --interval-high 0.593276203032303 --probability-positive 0.25575
```

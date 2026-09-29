# LEAD-70
## Goal
Execute LEAD-70: preserved challenger disagreement modifies the model-logit slope.
## State
**Measured:** one unit run exited 0; 135 files, 15 distinct challengers, 1,503 games.
Decisive record 8-15; Wilson interval [0.1881, 0.5511]. LL gain -0.000602733
[-0.000921038, -0.000305375], probability_positive=0; accuracy -0.465735 points
[-0.816327, -0.136986], probability_positive=0.0005. Pending serial recording.
## Protocol (declared before outcomes)
`a*logit*(1+d*z)`; LOSO 2020-2025; base=served four-term; 2 planned model looks;
log loss, Brier, accuracy, reliability by disagreement tercile with counts, IS-OOS gap.
Require >=3 six-season out-of-fold challenger artifacts. Complete immutable declaration
and implementation details: `docs/lead70_protocol.md`; no outcome-driven revisions.
## Tried
`.tools/uv.exe run --no-sync python -B scripts/lead70_unit1.py` once, with UV_NO_CACHE=1
and BLAS/OMP threads=2; scoped Ruff, closure validation and command syntax passed.
Baseline reproduced within 2.22e-16; reporting-only corrections required no refit.
IS/OOS LL 0.681778642/0.683376984; gap +0.001598342. All coefficients and nine
paired contrasts saved in results; 3 bands/12 diagnostic cells and 14 fit executions disclosed.
## Next
Orchestrator: run the nine prepared records serially, then adjudicate this tested modifier.
## Open
Whole adverse intervals support wrong_sign_resolved for three incremental cells only;
other disagreement mechanisms remain open. No untouched outer test or split-half reliability.
Prediction-level output is `docs/lead70_predictions.md`; retain locally, do not commit processed data.
## Record commands
Pending orchestrator; no registry mutation by this worker.
```powershell
$lead70Cells = @(
    @('four_term_log_loss', 'log_loss_improvement', -0.000602733258864, 0.000160787789008, -0.000921038437556, -0.000305375027671, 0),
    @('four_term_brier', 'brier_improvement', -0.000294475101834, 8.06009395718e-05, -0.000452993339221, -0.000143756666829, 0),
    @('four_term_accuracy', 'accuracy_points', -0.465735196274, 0.171681920725, -0.816326530612, -0.13698630137, 0.0005),
    @('model_only_log_loss', 'log_loss_improvement', 0.0135404170427, 0.00281723370354, 0.0083365606235, 0.0191947554204, 1),
    @('model_only_brier', 'brier_improvement', 0.00658241164341, 0.0013827145613, 0.00407627755964, 0.00931194706033, 1),
    @('model_only_accuracy', 'accuracy_points', 3.59281437126, 0.747357624538, 2.08269032922, 4.95785303958, 1),
    @('market_log_loss', 'log_loss_improvement', 0.00977019694553, 0.00128206454984, 0.00743169471083, 0.0123524375377, 1),
    @('market_brier', 'brier_improvement', 0.00486713834543, 0.000663757060289, 0.00363921662154, 0.00618701245663, 1),
    @('market_accuracy', 'accuracy_points', 7.31869594145, 0.76277409343, 5.92782273166, 8.86243386243, 1)
)
$lead70Cells | ForEach-Object {
    $lead70Verdict = @('--classification', 'unresolved_below_power')
    if ($_[0] -like 'four_term_*' -and $_[5] -lt 0) {
        $lead70Verdict = @('--classification', 'refuted_mechanism', '--closing-ground', 'wrong_sign_resolved')
    }
    .tools/uv.exe run --no-sync nfl-ats weak-signals record @lead70Verdict `
      --name ('lead70_disagreement_vs_' + $_[0]) --family lead70_disagreement_v1 `
      --description 'LOSO disagreement modifier; fixed preserved challenger panel' `
      --source docs/lead70_results.md --league nfl --season-start 2020 --season-end 2025 `
      --classification-evidence 'Incremental effects have wholly adverse intervals; other references do not isolate the modifier' `
      --effect-units $_[1] --effect $_[2] --standard-error $_[3] `
      --interval-low $_[4] --interval-high $_[5] --probability-positive $_[6] --sample-games 1503 --sample-blocks 6
}
```

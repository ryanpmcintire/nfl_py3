# Unbackfillable artifacts

RWB-09 experiment-provenance registry: `scripts/backfill_experiment_registry.py`
could not produce a registry row for these `artifacts/` run directories, because
nothing in them carries an `artifact_provenance()`-shaped block (a git revision
or a configuration hash) to lift.

Per this project's "label how you know it" rule, an approximated
`code_revision` (e.g. from `git log --before=<file mtime>`) would not actually
PIN the code that ran -- it would be a guess wearing a fact's clothes. So these
are recorded here, with a reason, rather than backfilled with an invented value.
This is the honest record `docs/closure_audit.md` S3's PageRank/HITS closure
never had: there, no artifact directory existed at all, so there was nothing
even to list. Here, the directory exists but never captured provenance.

29 run directories, as of this backfill pass:

- `artifacts/anytime/20260818T135124Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/anytime/20260818T144724Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/backtests/20260812T101321Z` -- no metadata.json or run.json in this run directory
- `artifacts/backtests/20260812T101634Z` -- no metadata.json or run.json in this run directory
- `artifacts/backtests/20260812T111305Z` -- no metadata.json or run.json in this run directory
- `artifacts/best_pick_tiebreak_cfb/20260818T212916Z` -- no metadata.json or run.json in this run directory
- `artifacts/calibration_by_regime_cfb/20260818T214613Z` -- no metadata.json or run.json in this run directory
- `artifacts/calibration_distortion/20260818T154856Z` -- no metadata.json or run.json in this run directory
- `artifacts/calibration_distortion/20260818T160920Z` -- no metadata.json or run.json in this run directory
- `artifacts/cfb_james_stein_unit/20260818T213139Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/cfb_value_weighted_continuity/20260818T211758Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/dependence/20260812T160420Z` -- no metadata.json or run.json in this run directory
- `artifacts/dependence/20260812T160522Z` -- no metadata.json or run.json in this run directory
- `artifacts/ecdf_smoothing/20260818T000600Z` -- no metadata.json or run.json in this run directory
- `artifacts/estvar_f_lever_confirmation/20260818T212440Z` -- no metadata.json or run.json in this run directory
- `artifacts/experiments/20260812T111352Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/hc_year_one_fade/20260818T003000Z` -- no metadata.json or run.json in this run directory
- `artifacts/nested_evaluations/20260812T161623Z` -- no metadata.json or run.json in this run directory
- `artifacts/novig_diagnostics/20260818T213648Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/offseason_retention_cfb_permetric/20260818T211604Z` -- no metadata.json or run.json in this run directory
- `artifacts/predictions/2025-week-18-20260812T101341Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/qb_dependence_cfb/20260818T214601Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/recency_training/20260818T211226Z` -- no metadata.json or run.json in this run directory
- `artifacts/residual_location/20260818T115234Z` -- no metadata.json or run.json in this run directory
- `artifacts/ridge_alpha_promotion/20260818T221459Z` -- no metadata.json or run.json in this run directory
- `artifacts/role_delivery_experiment/20260816T143449Z` -- no metadata.json or run.json in this run directory
- `artifacts/sensitivity_audits/20260813T141000Z` -- metadata.json exists but carries no artifact_provenance()-shaped block (no code revision or configuration hash to lift)
- `artifacts/xlg06_rookie_prior_cfb/20260818T213509Z` -- no metadata.json or run.json in this run directory
- `artifacts/xlg06_rookie_prior_cfb/20260818T215305Z` -- no metadata.json or run.json in this run directory

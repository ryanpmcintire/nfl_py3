# fit-ridge-derivation

## Goal
Audit every numeric constant in the served four-term pick-probability fit
(`src/nfl_ats/pick_probability_fit.py`, `src/nfl_ats/pick_probability.py`) for
provenance, and derive `FIT_RIDGE` out-of-sample via nested leave-one-season-out
(LOSO) selection, graded against the served constant on held-out data. Read-only:
no `src/` edits, no registry writes, no commits.

## State (complete, read-only)
Constant provenance table (all `read` from `src/nfl_ats/pick_probability*.py`;
git history via `git log --diff-filter=A -S"<name>"`):

| constant | value | file | first added | derivation evidence |
|---|---|---|---|---|
| `FIT_RIDGE` | `1e-3` | pick_probability_fit.py:47 | commit `6131b6d` (2026-09-15, "Week 1 is 9-7 everywhere...", bundles MKT-16/17/18/19) | none found — bundled default, no grid search or note in commit body or docs/lanes |
| `FIT_ITERATIONS` | `50` | pick_probability_fit.py:48 | same commit `6131b6d` | none found — fixed Newton-step count, no convergence study |
| `PROBABILITY_EPSILON` | `1e-6` | pick_probability.py:93 | same commit `6131b6d` | none found — standard numerical clip, not fit to data |
| `CONFIDENCE_BAND_EDGES` | `(0.50,0.52,0.55,0.58,0.62,1.0)` | pick_probability.py:91 | same commit `6131b6d` | none found — no data-derived quantile note |
| `STRENGTH_BAND_QUANTILES` | `(1/3, 2/3)` | pick_probability.py:89 | same commit `6131b6d` | tertile split by construction, not a fitted quantity |
| `STRENGTH_ROUNDING_PLACES` | `3` | pick_probability.py:90 | same commit `6131b6d` | display rounding, not a modeling choice |
| composition-flag weights | equal (each flag is $\{-1,0,1\}$, unweighted sum, `pick_probability.py:621-623`) | pick_probability.py | same commit | **confirmed equal-weighted** by reading `signed_composition_flags`: `flags[FLAG_SUM_COLUMN] = flags[COUNTED_FLAG_COLUMNS].sum(axis=1)`, no per-flag coefficient |

All of the above landed in one bundled commit with no logged sweep; **all are
underived constants** per AGENTS.md ("a constant derived from games it is
scored on never reaches `src/`" — these were never derived from games at all,
in-sample or out).

### FIT_RIDGE nested LOSO derivation — measured this session
Script: `scripts/fit_ridge_derivation.py` (ruff-clean). Predeclared grid
`RIDGE_GRID = np.logspace(-4, 2, 13)` (13 points, 1e-4..1e2). For each outer
LOSO season fold, ridge is chosen by inner LOSO over training seasons only
(pooled log-loss criterion), never touching the outer test season. Graded vs
the served fixed `FIT_RIDGE=1e-3` fit on the identical outer folds.

Artifacts: `artifacts/fit_ridge_derivation/20260923T210622Z/` (`report.json`,
`served_population_predictions.parquet`, `extended_population_predictions.parquet`).

**Dataset A — served population, `build_fit_population()`, 2020-2025, 1,503 games, 6 seasons** (real production call, not a fixture):
- chosen ridge per fold: 2020→31.6, 2021→10.0, 2022→31.6, 2023→31.6, 2024→31.6, 2025→10.0 (all far above 1e-3)
- nested: record 859-644, accuracy 0.57152, Brier 0.245407, log loss 0.683948
- served (1e-3): record 859-644 (**identical picks**), accuracy 0.57152, Brier 0.245388, log loss 0.683952
- gap: log loss/Brier differ by ~2e-5; served is marginally better

**Dataset B — extended population, `artifacts/extended_fit_population/20260923T205910Z/population.parquet`, 2011-2025, 3,734 games, 15 seasons**:
- chosen ridge per fold: 100.0 (grid ceiling) in every one of 15 folds — inner CV always prefers maximal shrinkage
- nested: record 1990-1744, accuracy 0.532941, Brier 0.248581, log loss 0.690386
- served (1e-3): record 1994-1740, accuracy 0.534012, Brier 0.248598, log loss 0.690453
- **measured** paired effect (nested minus served accuracy, season-block bootstrap, 20,000 reps, 15 blocks): −0.001071 accuracy points, SE 0.001095, 95% CI [−0.003415, 0.000829], probability_positive (nested beats served) = 0.134

**Conclusion (stated plainly):** `FIT_RIDGE=1e-3` is underived (no logged
search) but empirically indistinguishable from — and on the larger 2011-2025
population, slightly better than — the nested-LOSO-selected ridge. The
interval on the nested-vs-served gap crosses zero: `unresolved_below_power`,
not evidence to change the served constant. The nested inner CV pushing to
the grid ceiling (heavy shrinkage) with zero practical effect on picks is
itself informative: this 4-feature standardized logit is not ridge-sensitive
in this regime, so the underived default is doing no measurable harm. No `src/`
change is implied by this evidence.

## Tried
- `git log --diff-filter=A -S"<const>"` for all six named constants — all trace to one bundled commit, no derivation notes.
- Attempted `build_fit_population(artifacts_root, data_root)` live (real repo data, no mocks) — succeeded, 1,503 graded 2020-2025 games.
- Loaded `extended_fit_population/20260923T205910Z/population.parquet` (3,734 games 2011-2025) as the second, better-powered grading population.
- Nested LOSO ridge selection implemented by reusing served fit primitives (`_fit_logit`, `_design`, `_standardisers`, `_predict`, `_confidence_bands`) imported from `nfl_ats.pick_probability_fit` — no reimplementation drift from the served math.
- `ruff check --fix scripts/fit_ridge_derivation.py` — clean.

## Next (for the root orchestrator only — not run this session)
Record the FIT_RIDGE-sensitivity result as an unresolved modeling signal:

```
uv run nfl-ats weak-signals record \
  --name fit_ridge_nested_loso_vs_served \
  --description "Nested leave-one-season-out ridge selection for the four-term pick-probability fit vs the served fixed FIT_RIDGE=1e-3, graded on 2011-2025 extended population (3734 games) and served 2020-2025 population (1503 games)" \
  --source artifacts/fit_ridge_derivation/20260923T210622Z/report.json \
  --effect -0.001071 \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --classification-evidence "95% season-block bootstrap CI [-0.003415, 0.000829] crosses zero; probability_positive(nested beats served)=0.134; N=3734 games, 15 season blocks" \
  --league nfl \
  --season-start 2011 --season-end 2025 \
  --standard-error 0.001095 \
  --interval-low -0.003415 --interval-high 0.000829 \
  --probability-positive 0.134 \
  --sample-games 3734 --sample-blocks 15 \
  --category modeling \
  --plain-summary "We checked whether picking the ridge penalty by held-out seasons instead of using the fixed value would sharpen the model's picks; it does not — the current fixed value performs the same or slightly better." \
  --notes "FIT_RIDGE=1e-3 and its sibling constants (FIT_ITERATIONS, PROBABILITY_EPSILON, CONFIDENCE_BAND_EDGES, STRENGTH_BAND_QUANTILES, equal composition-flag weights) are all underived, all landed in commit 6131b6d with no logged search. This result does not justify changing FIT_RIDGE in src/."
```

Root should also decide whether the other five underived constants
(`FIT_ITERATIONS`, `PROBABILITY_EPSILON`, `CONFIDENCE_BAND_EDGES`,
`STRENGTH_BAND_QUANTILES`, equal flag weights) warrant their own derivation
lanes; none were touched by this session beyond provenance lookup.

## Open
- `--reliability` (split-half) was not computed for the recorded signal above;
  add via the `signal_atlas` block-bootstrap tooling referenced in
  `docs/lanes/lead59-archive-battery.md` if the root wants a terminal-quality
  record instead of `unresolved_below_power`.
- Grid ceiling (100.0) was selected in every Dataset-B fold; if the root ever
  wants a genuine "does heavier ridge help" answer, extend `RIDGE_GRID` past
  1e2 — not done here since it made zero practical difference to picks.

# RWB-12 — Drift monitoring: implementation report

Task id: `blog-rwb12-drift` · Branch: `swarm/blog-rwb12-drift` · Date: 2026-08-25

## What was asked

ROADMAP item RWB-12 (Drift monitoring): produce an implementation plan
(drift signals, weekly-pipeline hooks, storage format, alert thresholds,
required tests), then implement it **without touching evaluation semantics**.
No scoring look was authorized and none was run.

## Provenance legend (per AGENTS.md binding rule 2)

Every claim below is labelled **measured** (ran this session), **read**
(opened the file this session), **reported** (unverified), or **inferred**
(my reasoning).

## Plan (as designed before implementation)

- **Signals.** Four, per the ROADMAP row's definition of done:
  1. *Feature drift* — standardized mean shift + PSI per column against a
     reference window of the six most recent completed weeks strictly before
     the target week's earliest gameday (same leak-safe cutoff rule as
     `score_outcome_week`; **read**, `src/nfl_ats/outcomes.py:430-441`),
     columns organized by the feature-family registry
     (`FEATURE_FAMILIES`, RWB-02).
  2. *Missingness drift* — null-rate delta in percentage points; non-numeric
     garbage counts as missing; a column absent from either frame is reported
     fully-missing rather than dropped.
  3. *Probability drift* — published `home_cover_probability` distribution vs
     earlier cards of the same configuration (mean shift + share outside the
     reference central 90% band).
  4. *Calibration drift* — Brier/ECE over the most recent 4 settled weeks vs
     prior settled history, on already-published probabilities only.
- **Hook.** Optional step 13 in `weekly-run`, strictly after
  `publish-predictions` (and after POL-10 steps 9–12); failure recorded in
  `optional_failures`, never fatal; `--skip-drift` opts out.
- **Storage.** Append-only `artifacts/drift/<season>-week-NN-<run_id>/` with
  `drift_report.json` + `feature_drift.csv`, matching the margin-predict
  artifact convention (**read**: every margin-predict caller writes fresh
  timestamped directories, `src/nfl_ats/cli.py` `_cmd_margin_predict`).
- **Thresholds.** PSI 0.10/0.25 (standard tiers); mean-shift 0.5/1.0 sd;
  missingness +10/+25 pp; probability >20% out-of-band or |Δmean|>0.05;
  Δ-Brier 0.02/0.04; floors: PSI unscored below 50 current games, calibration
  below 32 recent / 200 prior settled games.

## What was implemented

All **measured** this session unless labelled otherwise:

- **`src/nfl_ats/drift.py`** (new): `psi`, `feature_drift_table`,
  `summarize_feature_drift`, `probability_drift_summary`,
  `calibration_drift_summary` (`_brier`, `_ece`), `reference_window`,
  `build_drift_report`, `write_drift_artifacts`, `worst_status`,
  `registered_feature_columns`. Read-only over inputs; writes only its own
  artifact directory via `atomic_json`/`atomic_csv`.
- **`src/nfl_ats/cli.py`**: new `drift-report` subcommand
  (`--season/--week/--features/--feature-profile/--probability-method/
  --reference-weeks/--calibration-recent-weeks`) plus `_find_drift_cards`,
  which matches candidate cards by configuration fingerprint (season, week,
  feature profile, probability method) rather than recency — the same lesson
  `prospective-record` learned because active and challenger cards share one
  `margin_predictions` namespace (**read**,
  `docs/prospective_evidence.md`). History dedupes per game keeping the FIRST
  occurrence (ledger first-write-wins convention). No matching card fails
  loudly.
- **`src/nfl_ats/weekly.py`**: step 13 `drift-report` appended to
  `plan_weekly_run` / `run_weekly` with `skip_drift: bool = False`; module
  docstring updated to describe step 13 and its not-evidence status.
- **`tests/test_drift.py`** (new, 21 tests) and **`tests/test_weekly.py`**
  (5 exact-plan assertions extended for the trailing optional step; core
  ordering assertions unchanged).
- **`docs/drift_monitoring.md`** (new): full plan, thresholds, storage
  format, hook semantics, test inventory, provenance.
- **`ROADMAP.md`**: RWB-12 row marked ✅ with summary.

### Evaluation semantics: untouched

**Measured:** no file under evaluation/backtest/scoring paths was modified —
the complete diff touches only `drift.py` (new), `cli.py` (one import block +
new handler/parser + nothing removed from existing handlers),
`weekly.py` (docstring + trailing step + two new keyword-only params defaulting
to False). `git diff --stat` confirms; prediction-safety tests all pass
unchanged (`tests/test_prediction_safety*.py` green in the full run).

## Key design decisions

1. **PSI noise floor.** Measured this session on gaussian draws: a 16-game
   week scores ~0.2 PSI at bins=5 and ~0.1 at n=100/bins=10 under a true
   null — September weeks would warn every week. So below
   `FEATURE_PSI_MIN_GAMES = 50` the value is reported but its status is
   `insufficient_history`.
2. **Telemetry, not evidence (binding rule 3 compliance).** The module
   docstring, the report JSON itself, and the docs all state that drift
   reports adjudicate no candidate, spend no rotation-registry window, and
   may never be cited about any signal — including "the feature drifted so
   the signal died", which would need the weak-signal/rotation machinery.
   Per AGENTS.md binding rule 1, an alert means "look at this", never
   "discard"; nothing here closes anything.
3. **Optional-after-publish.** Mirrors POL-10: monitoring must never be able
   to take the card down.
4. **Deterministic tests near thresholds.** Random draws at n≈16 sit within
   a hair of the warn boundaries (a calibrated-vs-calibrated Brier delta can
   exceed 0.02 by chance at n=48 — measured during development), so the
   threshold-adjacent fixtures use stratified-uniform outcome construction
   with fixed seeds instead of raw RNG draws that would flake under `-k`.

## Verification

All four gates **measured** this session, from the worktree root:

```
ruff format --check .   → 579 files already formatted
ruff check .            → All checks passed!
mypy src                → Success: no issues found in 103 source files
pytest                  → 1825 passed, 5 skipped (pre-existing skips for
                          absent local data artifacts)
```

pytest ran with `PYTHONPATH=<worktree>/src` and
`--basetemp=C:/Users/Ryan/AppData/Local/Temp/nflats-swarm-basetemp`
(outside the repo, per worker constitution).

## Not done / out of scope

- **No scoring look was executed.** The `drift-report` CLI exists but was
  never run against real data — calibration drift reads settled outcomes of
  published probabilities, and while it is telemetry by construction, running
  any scoring-shaped command against real history without a predeclaration is
  exactly what binding rule 3 forbids. First real execution belongs to the
  weekly pipeline (step 13 runs automatically on the next Tuesday run).
- No dashboard view of drift reports (could ride the RWB-05 workbench later).
- Threshold values beyond PSI's standard tiers are engineering defaults
  (**inferred**), documented as such in the docs; they should be tuned against
  observed seasons, not treated as measurements.

## Files changed

- `src/nfl_ats/drift.py` (new)
- `src/nfl_ats/cli.py` (import, handler, parser registration)
- `src/nfl_ats/weekly.py` (step 13, skip flag, docstring)
- `tests/test_drift.py` (new)
- `tests/test_weekly.py` (plan assertions extended)
- `docs/drift_monitoring.md` (new)
- `ROADMAP.md` (RWB-12 row)

Committed on `swarm/blog-rwb12-drift`; not pushed, master untouched.

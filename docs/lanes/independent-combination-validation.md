# Independent combination validation

## Goal
Complete a useful historical selection replay and preserve the untouched future
comparison of combined versus raw probabilities.

## State
**Measured:** historical replay complete on 533 decisive games in 2024-2025.
Full experimental pipeline: 269-264 versus raw 282-251; Brier improvement
-0.002283 [-0.009488,+0.005479], probability_positive 0.266050.
Post-result diagnostic before extra calibration: 300-233; +0.002057
[-0.005241,+0.010013], probability_positive 0.700550.
The added 62-game calibrator reversed the 2024 ordering; it is absent from
production. All arms, coefficients, reliability and limitations are saved in
docs/independent_combination_historical_replay.md. Thirty paired metrics are
recorded as unresolved_below_power in registry/weak_signals.json.
The untouched future study remains enrolled for 224 Week 4-18 games.
Enrollment commitment: 87d0108b14e14ad1ad530017810da1a6373a2d967e40377bb3bc03f1d64eba4e.
Local evidence: artifacts/independent_historical_validation/20260928_nested/;
future enrollment: artifacts/prospective/independent_validation/enrollment.json.

## Tried
**Measured:** both research scripts run; corrected registry-export structure;
second real replay reproduces every prediction. Independent arithmetic verifies
eight arms, 30 effects, two selection folds, source hashes and registry counts.
ruff format --check ., ruff check ., mypy src, and pytest -q --basetemp
.tmp/pytest-nested-validation all pass. Exact commands are in the report.
Future-study status still shows source_unchanged=true and awaits Week 4.

## Next
Inspect nfl-ats independent-validation status after the first Week 4 forecast.
Verify eligible games arrive before pool deadlines and keep coverage complete.
Use the historical report when assessing changes; do not introduce the
experimental small-block calibration or treat its failure as a production bug.

## Open
Earlier signal design and raw-model selection remain outside the historical
replay. Two outer seasons cannot establish stable probability improvement.
No signal is closed or served side changed. The future cohort supplies the
untouched evaluation; no retrospective backfill or interim selection.
Prior owner-decision proposal remains: remove the redundant CLI help-order
assertion; no tests were added or removed here.

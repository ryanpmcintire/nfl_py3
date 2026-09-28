# Independent combination validation

## Goal
Diagnose the three-week swing and preserve the untouched future comparison.

## State
**Measured:** diagnosis complete in docs/2026_three_week_diagnosis.md. The earlier
pressure-feed gap is repaired: all 16 Week 3 matchups have current-season coverage.
Frozen published decisions remain unchanged; its historical Week 3 win cost is unknown.
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
**Measured:** frozen published cards regraded through Sunday give Week 1 9-7,
Week 2 14-2, Week 3 5-10 with one pending: 28-19 (59.6%; descriptive Wilson 95%
45.3-72.4%). Use published cards, not the discrepant paper settlement ledger.
Readback: .tmp/2026-three-week-summary.json and matching per-game CSV.
**Measured:** Week 2 adjustments added 3 wins (14-2 versus raw 11-5);
Week 3 cost 4 (5-10 versus raw 9-6). Combined: one fewer win than raw.
The good week must count in the diagnosis. Version and per-game checks are in
docs/week2_week3_combination_comparison.md; old weights changed no traced side.

## Next
**Measured 2026-09-28:** current-season PBP ingestion and missing-pressure
publication checks are repaired; all 16 Week 3 windows now have coverage and
locked decisions are unchanged. See `done/current-week-pressure-inputs.md`.
**Measured:** all ten scheduled capture aliases ran successfully on September 28.
The former 15-minute window missed Monday behind the noon refresh; it is now
60 minutes. Strict per-game deadlines remain unchanged. The rehearsal recorded
zero rows because the active forecast is Week 3. Enrollment/source hashes match;
224 future games remain unobserved and interim scoring stays withheld.
Next verify actual Week 4 study captures after Tuesday's fresh-data lock.

## Open
Earlier signal design and raw-model selection remain outside the historical
replay. Two outer seasons cannot establish stable probability improvement.
No signal is closed or served side changed. The future cohort supplies the
untouched evaluation; no retrospective backfill or interim selection.
Prior owner-decision proposal remains: remove the redundant CLI help-order
assertion; no tests were added or removed here.

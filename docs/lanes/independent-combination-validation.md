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
**Measured 2026-10-01 17:08Z:** v1 stopped by the pinned-source guard (sources
changed in 41ae595 and 6a8335c before any capture); it stays frozen with zero
captures and is never blended. v2 enrolled explicitly:
registry/studies/combined_vs_raw_2026_v2.json, artifacts/prospective/
independent_validation_v2/, digest 42ac842e..., 224 games, cohort limited to
games kicking off after enrollment. CLI `--study` defaults to v2; v1 is
`--study independent_validation`. First capture recorded all 16 Week 4 games,
PIT at CLE included. Scheduled capture jobs now write v2 unchanged.
Next: confirm Thursday 19:55 and Sunday captures succeed; any further pinned-
source edit stops v2 the same way, so batch model-source edits after Week 18
or enroll v3 explicitly.

## Open
Earlier signal design and raw-model selection remain outside the historical
replay. Two outer seasons cannot establish stable probability improvement.
No signal is closed or served side changed. The future cohort supplies the
untouched evaluation; no retrospective backfill or interim selection.
Prior owner-decision proposal remains: remove the redundant CLI help-order
assertion; no tests were added or removed here.

**Measured 2026-10-16:** v2 stopped by the guard since 2026-10-08 18:02 (de8b5801
and 3bc81dac changed margin.py and modeling.py, scaler float-noise guard; claimed
served output unchanged on 146 refits). Last v2 capture 2026-10-08 16:28Z; Week 4
all 16 games, Week 5 15 games (latest capture Thu 10-08 only); Week 6 zero.
Every job since 10-08 19:55 failed; manual run fri_1210 failed the same. Guard
untouched. Decision for orchestrator: enroll v3 explicitly (as v2 was); no
backfill; v2 stays frozen with its 31 games.

## 2026-10-18 source-changed failures
**Read:** the guard is src/nfl_ats/independent_validation.py:90-91. It compares
_sources() (sha256 of pinned .py files: model/margin/features/overlay names,
PINNED_NAMES, cli_commands/prediction.py; lines 58-69) with enrollment
source_sha256. It never reads model_id, feature-table hash or artifact paths, so
routine same-week re-syncs cannot trip it; only edits to pinned source files do.
**Measured:** all FAIL(2) from 10-08 19:55 to the 10-16 18:09 manual run hit the v2
study (default then); v2 held 31 games (Week 4 all 16, Week 5 15), last capture
10-08. v3 (Week 6-18, 192 games) was enrolled 10-16 18:21Z-22:21Z; the 18:21 rerun,
10-17 12:10 and all 10-18 jobs are OK. v3 status: source_unchanged=true, 13
captured, 13 Week 6 cohort games (all kicking off 10-18) each captured pregame.
**Lost (v2, not recoverable, no backfill):** Week 5 Sun/Mon 10-11/10-12 refresh
captures and the Week 5 game missing from v2; Week 6 Thursday 10-15 game (kicked
off before v3 enrollment, in neither cohort); Thu 19:55, Fri 12:10 snapshots.
**Decision:** no guard change. Treating "same-method" edits as the same source
would let unverified model code mix versions, which the protocol forbids. Batch
pinned-source edits after Week 18 or enroll v4 explicitly.
**Measured 2026-10-20:** v3 (Week 6-18) failed the source guard from 10-19
after 10-18 pinned edits (inactives_refresh_overlay.py d28754e2,
transaction_flag_features.py 2d8d25df); it stays frozen with 13 Week 6
captures. v4 enrolled 22:48 UTC: registry/studies/combined_vs_raw_2026_v4.json,
artifacts/prospective/independent_validation_v4/, 179 games Weeks 7-18,
default study now v4; first capture recorded 14 Week 7 games.

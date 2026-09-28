# Current week and pressure inputs

## Goal

Update the current card without changing locked decisions, repair current-season
play-by-play ingestion, and prevent missing pressure inputs from looking like
an observed zero signal.

## State

**Measured:** Week 3 published at 5–10 with one pending, season 28–19.
All 16 frozen sides/probabilities and the locked Carolina Best Pick are preserved.
PBP snapshot `20260928T152114Z` covers 2009–2026 and every one of 47 season finals;
all 16 Week 3 matchups have usable pressure windows. Missing unlocked pressure
inputs now block publication. Tuesday refreshes PBP and consumes latest feature
inputs; its real daemon rehearsal returned `MANUAL-DRY-RUN OK weekly_lock`.
Details and exact commands: `docs/current_week_pressure_repair.md`.

## Tried

**Measured:** stale-input rejection and repaired-input acceptance both verified;
fully frozen publication passes. Ingest reuses verified historical partitions
and publishes atomically, including safe failed-download cleanup. Windows paths
and missing PyArrow type stubs corrected during review. Formatting, lint, mypy,
and `pytest -q --basetemp .tmp/pytest-pressure-repair-20260928` pass (1,645 tests).
Logs and rendered-page text diff: `.tmp/backlog-20260928-*`.

## Next

After Monday's final, settle and republish Week 3. Tuesday's normal lock fetches
fresh PBP including that game. Verify actual Week 4 independent-study captures;
the successful rehearsal did not record future predictions.

## Open

**Measured:** enrolled source hashes still match; zero captures before Week 4.
Coefficients and selection rules unchanged. No test removals proposed. Existing
policy edits, popup lane, and four earlier experiment records remain unrelated.

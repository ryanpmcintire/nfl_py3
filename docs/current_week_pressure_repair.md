# Current week and pressure-input repair

**Measured, 2026-09-28:** the published Week 3 card is 5–10 with one game left;
the season record is 28–19. These are descriptive played-card records, not
evidence for selecting a new model. The Markdown card now matches all 16 frozen
sides and cover probabilities, including Carolina as the locked Best Pick.
The site adds the Rams–Broncos final and updates the remaining-game count.

## Input failure and repair

**Read:** the prior weekly plan fetched schedules and team statistics but only
rebuilt PBP features from existing snapshots. The pressure reader selected the
newest raw directory, even if a failed download had left it without a manifest.
The latest complete snapshot stopped at 2025.

**Measured:** `pbp-ingest --reuse-history` now validates historical partitions,
downloads the active season, and publishes the completed snapshot atomically
from a separate staging directory. Legacy Windows manifest paths are accepted;
new paths are portable. Weekly ingestion includes this step, while explicit
`--skip-ingest` still skips downloads. The scheduled Tuesday lock requests the
latest feature inputs; the ordinary CLI retains its explicit pinned-data mode.

**Measured:** snapshot `20260928T152114Z` spans 2009–2026 and has all 47 current
season finals: 16 Week 1 games, 16 Week 2 games, and 15 Week 3 games. All 16
Week 3 matchups now have complete pressure windows. The base team-statistics
snapshot already includes 2026; this repair does not alter that feature recipe.

**Measured:** the publication guard rejects the old snapshot for an unlocked
card and passes with the repaired snapshot. It checks both offensive and
defensive dropbacks in the prior team-game windows, distinguishes missing
pressure fields from observed zero values, and requires saved frozen decisions
before exempting locked games. Board, Markdown, and pick-refresh entry points
share the requirement. Failed/incomplete latest inputs cannot pass unnoticed.

**Measured:** the scheduler's real `weekly_lock --dry` dispatch now performs a
safe Week 4 plan rehearsal: `MANUAL-DRY-RUN OK weekly_lock`, current inputs
enabled, no decisions written. The previous dry dispatcher did not forward a
dry flag to this recording script; the repair makes that path explicitly dry.

## Verification and continuation

All commands use `.tools/uv.exe run --no-sync`, with the writable cache at
`.tmp/uv-cache`. Evidence is in `.tmp/backlog-20260928-*.log` and the matching
coverage and card-verification JSON files.

- `nfl-ats settle --season 2026 --week 3 --write-graded`
- `nfl-ats card-ledger-check`
- `nfl-ats pbp-ingest --start-season 2009 --end-season 2026 --include-postseason --reuse-history`
- `python scripts/capture_scheduler.py --run-job weekly_lock --dry`
- `nfl-ats publish-predictions --with-board` and `nfl-ats publish-board`
- `nfl-ats independent-validation status`

**Measured:** the untouched Week 4–18 enrollment still has unchanged pinned
sources and zero captures before its first week. Coefficients, selection rules,
and stored Week 3 decisions are unchanged. No retrospective picks were added.
After Monday's final, settle and publish the remaining result. Tuesday's fresh
PBP capture must include that game before a complete Week 4 card is served.
Do not tune the combination using the recent losing week or interim cohort
performance.

**Measured:** `ruff format --check .`, `ruff check .`, `mypy src`, and
`pytest -q --basetemp .tmp/pytest-pressure-repair-20260928` all pass; pytest
reports 1,645 passed. No test files or test functions were added. Existing
weekly ordering expectations and two recorder-isolation fixtures were updated.
No test removal is proposed for these orchestration contracts.

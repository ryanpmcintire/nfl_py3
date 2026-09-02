# Repository hygiene and test-speed sweep (2026-09-02)

## Scope

[Measured: `rg --files` plus PowerShell line counts] The sweep covered 162
Python files / 88,108 lines under `src/`, 291 files / 115,326 lines under
`scripts/`, and 218 files / 67,978 lines under `tests/` at its measurement
boundary. [Measured: tracked-Python definition/reference scan plus `rg -w`]
The unused-code pass checked the whole tracked Python surface rather than only
recently edited modules.

[Measured: repository-wide Ruff and mypy runs] Import/name lint and strict
typing were also applied across the repository. [Inferred] A symbol was not
removed merely because a static name search could not see a dynamic import;
deletion required a private, unexported definition with no repository call
site and focused behavioral coverage around the owning module.

## DRY and dead-code results

[Measured: `git diff` and repository searches] Three copies of the NFL
Tuesday-through-Monday week-anchor calculation now use
`nfl_ats.nfl_week.week_cycle_sunday`; the two capture modules' duplicate byte
hash and UTC timestamp helpers now use `nfl_ats.provenance`; and the four ATS
Terminal pages now share one `_terminal_chrome` composition.

[Measured: definition/reference scan and focused tests] Four private helpers
with definition-only symbols were removed: `_regular_season_schedules`,
`_weekly_best_pick`, `_self_check_offsets`, and `_round`. Their removal deleted
40 lines and only the imports made dead by those deletions. [Measured: focused
pytest runs] The owning feature, CLV, CFB body-clock, and audit paths retained
86 passing tests after the deletion.

## Test value and runtime

[Measured: full `pytest --durations=25` run] The first combined verification
passed 3,396 tests in 289.06 seconds and exposed two dominant costs: a 55.17
second dashboard fixture setup and a 54.57 second XLG-06 frozen-artifact
reproduction.

[Measured: before/after focused timings] The dashboard fixture setup fell from
40.92 seconds to 1.41 seconds by bypassing an unrelated 8,780-file historical
quote scan with the production missing-data fallback; real artifact to view
model to HTML integration remains exercised, while quote loading and Best Pick
selection retain dedicated tests. [Measured: 127-test focused run] The affected
dashboard/market/nomination set passes in 3.49 seconds.

[Measured: before/after focused timings] Identical deterministic feature arms
now reuse one walk-forward fit, reducing the real-data identity case from 3.93
seconds to 1.55 seconds. [Measured: fixture setup profiling] Deterministic CFB
feature setup now costs 0.28 seconds once per session and 0.01--0.06 seconds for
isolated deep-copy consumers, rather than 0.30--0.44 seconds for every test.

[Measured: before/after frozen-artifact test] Batched XLG-06 correlation
bootstraps reduced the 20,000-sample reproduction from 41.77 seconds to 11.50
seconds without changing samples, seeds, or estimators. [Measured: differential
test oracle] Player- and cohort-blocked draws with ties match the original
draw-by-draw implementation within `4.44e-16` for Pearson and `2.22e-16` for
Spearman, with identical quantiles and signs.

[Measured: seeded before/after drive-simulation runs] Caching deterministic
empirical-pool options reduced the late-game behavior case from 5.49 seconds
to 0.32 seconds and the auditable-possession case from 3.42 seconds to 0.22
seconds. The unchanged 200-game run produced the same 4,106 drive rows and
identical game/drive dataframe hashes; random pool selection and drive draws
remain in their original order.

[Measured: duplicate-test review and duration profile] No test was deleted
solely for being slow or superficially similar. The slow XLG reproduction is a
scientific canary with distinct value, so its implementation was accelerated;
the dashboard fixture's unrelated market-history work was removed because
dedicated tests already cover that subsystem.

## Remaining concentration

[Measured: final PowerShell line counts] The largest Python modules remain
`src/nfl_ats/cli.py` (5,800 lines), `src/nfl_ats/public_board.py` (4,196), and
`src/nfl_ats/experiment_runner.py` (3,932). [Inferred] Their size makes them
the next DRY/ownership audit targets, but size alone is not evidence that a
split will reduce code or runtime; future extraction should require a concrete
duplicate boundary and focused equivalence tests.

## Final verification

[Measured: `ruff format --check .`, `ruff check .`, and `mypy src`] The final
format, lint, and type gates passed across 986 formatted files and 162 source
modules. [Measured: final `pytest --durations=25` run] The repository passed
3,397 tests in 156.31 seconds, including one new differential performance
regression. That is 96.01 seconds (38.1%) faster than the 252.32-second
pre-sweep baseline despite 17 additional tests, and 132.75 seconds faster than
the first profiled 289.06-second run.

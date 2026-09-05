# Verification tiers (ENG-11)

Two ways to run this repository's checks, and one rule that does not change
based on which one you ran.

## The release gate is unchanged and is NOT the fast tier

`AGENTS.md`'s "Required verification" section names four commands as
release-blocking:

```powershell
.\.tools\uv.exe run ruff format --check .
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run mypy src
.\.tools\uv.exe run pytest
```

**These four commands, run in full, remain the release gate before any push
to master.** `scripts/verify_full.py` runs exactly this sequence (unchanged
commands, no `--no-sync`, no test-marker filter) and prints per-step wall
time. Nothing below changes what gates a push — it only adds a faster
subset for the PR/edit loop.

**The fast tier described below is not sufficient for a master push.** It
exists to make the edit loop fast, not to replace the gate.

## Fast tier: `scripts/verify_fast.py`

```powershell
.\.tools\uv.exe run --no-sync python scripts\verify_fast.py
```

Runs, in order, stopping at the first failure:

1. `ruff format --check .`
2. `ruff check .`
3. `mypy src`
4. `pytest -m "not full"`

Steps 1-3 are unchanged from the release gate (whole-repo lint/format/typing
is already fast — see measured times below). Step 4 is the whole test suite
**minus** the tests tagged with the `full` pytest marker (declared in
`pyproject.toml` `[tool.pytest.ini_options]`, required because
`--strict-markers` is on). This covers safety, leakage, point-in-time-
chronology, and every other test in the suite **except** the ones tagged
`full` — see "What `full` tags, and why" below. No safety, leakage, or
chronology test is tagged `full`; they all still run in the fast tier.

Uses `--no-sync` throughout (skip `uv`'s lockfile re-resolution check) since
this tier is meant to be run repeatedly during an edit loop.

## Full tier: `scripts/verify_full.py`

```powershell
.\.tools\uv.exe run python scripts\verify_full.py
```

The same four AGENTS.md commands, unchanged, run without `--no-sync` (a
release check should run against a freshly-synced environment) and without
any `-m` filter on `pytest` — i.e. the entire suite, including everything
tagged `full`. This is the command `scripts/capture_scheduler.py`'s
disabled-by-default `verify_full_weekly` job (Monday 03:00 ET, grace 240m,
`enabled=False`) would run if enabled; see that entry's comment for why it
ships disabled (a multi-minute CPU-bound run is not something to fire
unattended without a deliberate decision to do so) and its `catch_up=True`
(a late run is still a valid one — this is a code-health check, not a
point-in-time capture).

## What `full` tags, and why

31 tests, in 24 files, are tagged `pytest.mark.full` (search `tests/` for
`ENG-11` to find every tag and its one-line reason). Each was tagged because
it does at least one of:

- **Dominates `pytest --durations`.** Measured 2026-09-04: the full suite's
  slowest single item ran 12.0s and the next three ran 6-10s, against a
  typical unit test cost of well under 100ms; the top-40 duration list is
  the primary source for this set.
- **Fits a real model** — e.g. `test_margin_hgb_and_guards` and
  `test_hgb_model_path` each fit a real `HistGradientBoosting` estimator;
  `test_nested_evaluation_selects_before_each_outer_season` and
  `test_unpaired_refits_overstate_the_refit_variance` each fit many models.
- **Reads real on-disk data or artifacts** — e.g.
  `test_feature_arm_identical_arms_measure_exactly_zero_on_real_data` and
  `test_penalty_discipline_reproduces_the_recorded_registry_entry` are each
  gated by their own `@pytest.mark.skipif(not _*_AVAILABLE, ...)` on a real
  `data/processed/game_features.parquet` / local PBP snapshot; the three
  dashboard tests (`test_board_site.py`, `test_board_terminal.py`,
  `test_board_improvements.py`) trigger the session-scoped real-artifact
  `SiteContent` build documented in `tests/conftest.py`.
- **Asserts determinism/reproduction across a full build** — e.g.
  `test_regular_season_rows_are_bit_identical_with_postseason_included`,
  `test_audit_reproduces_the_benchmark_exactly`, and
  `test_prediction_dependence_audit_is_deterministic`.

**Deliberately NOT tagged, even where slow:** any test whose job is a
safety, leakage, or point-in-time-chronology guarantee — e.g.
`test_no_wager_path.py::test_no_wager_placement_verb_outside_the_readonly_odds_client`
(the no-automated-wagering invariant), `test_cross_league_transfer.py` and
`test_xlg05_transfer.py`'s "does not depend on future rows" /
"never enters that week" tests, and `test_outcomes.py::test_walk_forward_outcomes_ignores_games_after_the_target_week`.
AGENTS.md's "Required verification" and "Pregame features must only use
information available before the prediction timestamp" both make these
release-blocking regardless of how slow they are, so the fast tier keeps
them.

**Known limitation:** the three dashboard tests share one session-scoped
fixture (`tests/conftest.py`'s `_shared_real_site_content`) with many other,
cheap dashboard tests that are *not* tagged `full`. Deselecting the one
tagged test in each file does not remove the fixture's build cost from the
fast tier — the next untagged test in that file that needs `site`/
`site_content` still pays it. Tagging every consumer was rejected as
disproportionate (it would gut fast-tier dashboard coverage for a
session-scoped cost paid once per worker); this is recorded here rather
than silently accepted.

## Measured wall times (2026-09-04, this machine, `-n auto` / 24 xdist workers)

Other agents were running their own pytest subsets on the same repository
concurrently while these were measured, so treat the specific seconds as
representative, not a clean-room benchmark; the multi-second-per-slow-test /
sub-second-per-fast-check *shape* is the reproducible part. Re-run
`scripts/verify_fast.py` / `scripts/verify_full.py` for a current number.

| Step | Wall time | Notes |
|---|---|---|
| `ruff format --check .` (whole repo) | ~0.1s | measured directly |
| `ruff check .` (whole repo) | ~0.1s | measured directly |
| `mypy src` | ~0.5-2s | measured directly; follows a handful of `scripts/*.py` modules `src/` imports verbatim (see `pyproject.toml`'s `[[tool.mypy.overrides]]` comments) |
| `pytest` (full suite, no marker filter) | 82.5s pytest-reported (85.2s wall including process startup), 3,952 passed / 1 failed of 3,953 | measured via `scripts/verify_full.py`'s own step, 2026-09-04; the one failure was in this work's own `tests/test_verification_tiers.py` (a too-strict substring check that also matched its own docstring's prose), fixed same session |
| `pytest -m "not full"` (fast tier) | 74.5s pytest-reported (77.6s for the whole fast-tier script: ruff+mypy+pytest), 3,957 passed / 0 failed | measured via `scripts/verify_fast.py`, 2026-09-04, run immediately after the full-suite measurement above |

**The fast tier's wall-time saving is small in practice on this machine**
(82.5s -> 74.5s pytest time, about 10%), not proportional to the 31/3,953
tests deselected (~0.8%). With `-n auto` spreading ~3,950 tests across 24
xdist workers, wall time is dominated by load-balancing the bulk of the
suite across workers, not by the handful of especially slow items — removing
31 tests trims some of the long tail but most workers still have plenty of
other tests queued. The tier still exists because it is a real (if modest)
win with zero coverage loss for anything release-blocking, and because
`-m full`/`-m "not full"` collection is now a fast, reliable way to answer
"is this specific slow/model-fitting/real-data test excluded from the PR
loop" without re-deriving it from `--durations` each time. Test counts
differ slightly between the two runs above (3,953 vs 3,957) because other
agents were adding test files to the same repository concurrently — both
counts are point-in-time snapshots, not evidence of a bug in either script.

Collection counts are cheap to reproduce and do not depend on wall-clock
timing noise:

```powershell
.\.tools\uv.exe run --no-sync pytest --collect-only -q -m full          # the tagged set (31 as of 2026-09-04)
.\.tools\uv.exe run --no-sync pytest --collect-only -q -m "not full"    # everything the fast tier runs
```

## Basetemp and temp-directory hygiene (ENG-30)

`pytest`'s `--basetemp` can point anywhere, and this repository does not pin
one in `pyproject.toml` (a machine-specific path there would break every
other clone). Three basetemp modes exist in practice, and only one of them
used to break the suite:

1. **No `--basetemp` at all** — pytest's own default, a numbered
   `pytest-of-<user>` directory under the OS temp dir
   (`tempfile.gettempdir()`). Always outside this repository. Safe.
2. **An explicit out-of-repo `--basetemp`** (e.g. `$env:TEMP\some_dir`, or
   the PID-keyed directories `scripts/verify_fast.py` /
   `scripts/verify_full.py` already use — see their own `_BASETEMP`
   comments). Always outside this repository. Safe.
3. **An explicit in-repo `--basetemp`** (e.g. `.agent_tmp/some_dir`, a
   pattern this repository's own agents use for scratch elsewhere). Every
   `tmp_path`-based test's temp directory then sits *inside* this
   repository's tree.

Mode 3 used to fail nine tests across `tests/test_provenance.py`,
`tests/test_odds_backfill.py`, and `tests/test_sportradar_injury_capture.py`,
plus one in `tests/test_source_policy.py`, for two distinct reasons that both
trace back to the same cause (an in-repo temp directory):

- `source_policy.py`'s `require_private_raw_destination` guard (the MKT-09
  private-raw-data policy) correctly refuses to write "raw" acquisition
  output under any in-repo path other than the sanctioned
  `data/{raw,market,cfb,players}` roots. A `tmp_path` nested under an
  in-repo `--basetemp` is exactly such a refused path, even though the test
  has nothing to do with the policy under test — it tripped
  `tests/test_odds_backfill.py::test_cli_odds_backfill_dry_run_and_execution`
  (goes through the real CLI, which enforces the guard),
  `tests/test_sportradar_injury_capture.py`'s five `tmp_path`-based tests
  (`capture()` enforces the guard before writing anything), and
  `tests/test_source_policy.py`'s own external-root assertion.
- Two `tests/test_provenance.py` tests assert "no enclosing git repository"
  (`git_state`/`git_diff_sha256` returning `None`); once `tmp_path` is
  nested inside this repository's own `.git`, git finds and reports the
  *real* enclosing repo instead.

**The fix was in the tests, not the guard** — the guard's in-repo-rejection
behaviour is correct production behaviour and is unchanged (proved by
`tests/test_source_policy.py::test_private_raw_policy_still_rejects_an_in_repo_non_sanctioned_root`,
which builds its destination from the test file's own on-disk location, not
from any temp fixture). `tests/conftest.py` adds a `private_raw_root`
fixture — a per-test directory rooted at `tempfile.gettempdir()` via
`tempfile.TemporaryDirectory`, independent of `--basetemp` — and the ten
affected tests use it in place of `tmp_path`. The suite is now robust to all
three modes (measured 2026-09-04: the affected files plus
`tests/test_source_policy.py`, and the whole-suite selection
`-k "source_policy or provenance or odds_backfill or sportradar"`, all pass
identically under an in-repo `--basetemp`, an explicit out-of-repo
`--basetemp`, and no `--basetemp` at all).

**The rule going forward: never pass an in-repo `--basetemp` for the
capture/source-policy tests; the suite is now robust to it anyway.** Any new
test that writes to a "raw" destination checked by `source_policy.py`, or
that asserts something about the *absence* of an enclosing git repository,
should take the `private_raw_root` fixture instead of `tmp_path`.

**A separate, unrelated observation from this same measurement:** mode 1 (no
`--basetemp`) intermittently fails on this machine with
`PermissionError: Access is denied` on the shared
`...\Temp\pytest-of-<user>` directory itself, reproducible while other
agents' pytest processes were concurrently active against the same shared
OS-temp path. None of those failures were `SourcePolicyError` — they occur
in pytest's own tmpdir retention/cleanup code before any test body runs, and
they also hit tests this work did not touch (e.g.
`test_provenance.py::test_hashes_are_deterministic`). This is exactly the
class of problem `scripts/verify_fast.py` / `scripts/verify_full.py` already
sidestep by using a PID-keyed basetemp under the OS temp dir instead of
pytest's shared default — further reason to prefer those scripts, or an
explicit unique `--basetemp`, over bare `pytest` on this machine when other
agents may be running concurrently.

## Pre-commit hook

`.githooks/pre-commit` (as of 2026-09-04) runs only `nfl-ats handoff` to
refresh `HANDOFF.md`/`README.md`'s generated blocks before a commit — it does
not run either verification tier, and this work leaves it unchanged. The
release gate remains a manual (agent-run) step before pushing master, per
AGENTS.md's "Automatic session handoff" section (`nfl-ats handoff --check`
before pushing, in addition to — not instead of — the four gates above).

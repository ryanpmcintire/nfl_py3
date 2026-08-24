# RWB-09 — Experiment registry: audit & verification tool

**Task:** ROADMAP item RWB-09 (Experiment registry, status 🚧). Read its row, the
`registry/*.json` structure, and the `nfl-ats experiment` CLI; sample 10 entries
and verify their linked artifacts exist (each verification labelled **measured**
or **reported-unverified**); implement fixes/gaps as code; no new experiments.

**Worker branch:** `swarm/blog-rwb09-experiment-registry`

---

## 1. RWB-09 row (read)

ROADMAP.md line 106 (read this session):

> | RWB-09 | 🚧 | Experiment registry | Config hash, code revision, source snapshot, metrics, notes |

Definition of done (DoD): every experiment run must record a **config hash**,
a **code revision**, the **source snapshot**, its **metrics**, and **notes**.
Status is 🚧 (in progress). The registry is implemented as one JSON file per run
under `registry/experiments/<command>/<stamp>.json`, written mechanically by
`write_experiment_artifact()` (a side effect of every CLI artifact write) and
by the one-time backfill script `scripts/backfill_experiment_registry.py`
(read this session: `src/nfl_ats/provenance.py`, `scripts/backfill_experiment_registry.py`).

## 2. `registry/*.json` structure (read)

`registry/` holds four git-tracked top-level JSON registries plus the
`experiments/` subdirectory (the RWB-09 deliverable):

- `rotation_registry.json` — `{families, notes, season_usage, version}` (RWB-17).
- `stadium_coordinates.json`, `stadium_elevations.json` — keyed by stadium name,
  each with a `_README` (read this session).
- `weak_signals.json` — `{notes, signals, version}` (RWB-18); schema-validated,
  narrow effect/interval/P+ triple.
- `experiments/` — the RWB-09 registry: one file per run
  (`<command>/<stamp>.json`). Count measured this session: **227 rows across 73
  command directories** (`find registry/experiments -name '*.json' | wc -l`).

Each experiment row carries (read this session from `provenance.ExperimentRecord`):
`experiment_id`, `recorded_at`, `command`, `artifact_directory`, `config_hash`,
`code_revision`, `code_dirty`, `code_diff_sha256`, `feature_table_sha256`,
`uv_lock_sha256`, `schema_version`, `metrics`, `notes`, `source`,
`weak_signal_name`, `rotation_family`, `provenance_backfilled`, `backfill_note`.
The reproducibility guarantee is the **hashes** (`config_hash`, `code_revision`,
`feature_table_sha256`, `uv_lock_sha256`) — not the artifact path, because
`artifacts/` is gitignored and local-disk-only (stated in `provenance.py`'s
module docstring; confirmed by reading it this session).

## 3. `nfl-ats experiment` CLI (read)

The `experiment` command group (read this session, `src/nfl_ats/cli.py`) has:

- `experiment compare` — feature-set walk-forward comparisons
  (`run_feature_set_experiment`); writes `artifacts/experiments/<stamp>/...` and
  records a registry row via `write_experiment_artifact`.
- `experiment run` — the declarative pipeline
  (`run_experiment_cli`): spec → screen → bootstrap → classify → registry record
  → provenance stamp.
- **`experiment verify` (added this task)** — read-only audit that checks each
  registry row's `artifact_directory` against disk and flags inconsistencies
  (see §5).

## 4. Sampling method

Measured this session: 227 rows, 73 commands. I took a **stratified-by-command
sample** — the first (alphabetically earliest) file from each of the first 10
command directories — so the sample spans backfilled rows
(`provenance_backfilled: true`, relative paths) and live rows
(`provenance_backfilled: false`, absolute machine paths). Verification was done
with the new `nfl-ats experiment verify --artifacts-root F:/Repos/nfl_py3/artifacts`
tool, which performs a real `Path.exists()` disk check this session. Each
verification below is therefore **measured** (I checked the filesystem this
session), not reported-unverified.

## 5. The 10 sampled entries

Provenance labels follow binding rule 2. "**measured**" = I ran the disk check
this session (via the `verify` tool). All 10 artifacts are **present** in the
canonical local store `F:/Repos/nfl_py3/artifacts` (which is reachable from
this environment). In *this worktree* only the 6 machine-absolute rows resolve
locally; the 4 relative-path rows are absent in the worktree's local
`artifacts/` (gitignored, nearly empty — expected, not a defect).

| # | experiment_id | path style | verification (measured) | linked artifact path | source field |
|---|---|---|---|---|---|
| 1 | `altitude-screen/20260821T182533Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\altitude_screen\20260821T182533Z` | `nfl-ats altitude-screen` |
| 2 | `attention-followup-screen/20260819T191700Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\attention_followup\20260819T191700Z` | `nfl-ats attention-followup-screen` |
| 3 | `availability-ablation/20260813T133345Z` | rel | **present** (checked this session, resolved against main-repo store; absent in worktree artifacts — expected) | `artifacts/availability_experiments/20260813T133345Z` | `artifacts/availability_experiments/20260813T133345Z/metadata.json` |
| 4 | `backtest/20260812T111822Z` | rel | **present** (checked this session, resolved against main-repo store; absent in worktree artifacts — expected) | `artifacts/backtests/20260812T111822Z` | `artifacts/backtests/20260812T111822Z/run.json` |
| 5 | `best-pick-ranker-followup/20260821T175357Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\best_pick_followup\20260821T175357Z` | `nfl-ats best-pick-ranker-followup` |
| 6 | `body-clock-night-screen/20260821T222134Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\body_clock_night_screen\20260821T222134Z` | `nfl-ats body-clock-night-screen` |
| 7 | `body-clock-screen/20260821T182157Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\body_clock_screen\20260821T182157Z` | `nfl-ats body-clock-screen` |
| 8 | `bye-overvaluation-screen/post_fix_seed20260821` | rel | **present** (checked this session, resolved against main-repo store; absent in worktree artifacts — expected) | `artifacts\bye_overvaluation_screen\post_fix_seed20260821` (note backslashes) | `nfl-ats bye-overvaluation-screen` |
| 9 | `ceiling-error-split/20260822T221223Z` | abs | **present** (checked this session at the literal absolute path) | `F:\Repos\nfl_py3\artifacts\ceiling_error_split\20260822T221223Z` | `scripts/ceiling_error_split.py` |
| 10 | `cfb-absence-separation/20260817T105651Z` | rel | **present** (checked this session, resolved against main-repo store; absent in worktree artifacts — expected) | `artifacts/cfb_role_experiments/20260817T105651Z` | `artifacts/cfb_role_experiments/20260817T105651Z/metadata.json` |

**Result:** all 10 sampled linked artifacts are **measured-present** in the
canonical local store. None are reported-unverified — I could reach the store
from this environment and confirm each by disk check this session. In a *fresh
clone* with no access to `F:/Repos/nfl_py3/artifacts`, all 10 would be
reported-unverified (the path is the only pointer and `artifacts/` is
gitignored), which is exactly the portability gap addressed in §6.

## 6. Aggregate findings (measured, this session, 227 rows)

Run with `nfl-ats experiment verify` (default, against this worktree) and again
with `--artifacts-root F:/Repos/nfl_py3/artifacts` (canonical store):

- **Link resolution:** 63/227 resolve as machine-absolute
  (`F:\Repos\nfl_py3\...`); 164/227 are repo-relative. Measured against the
  canonical store: **214 present, 13 missing**; measured against this worktree
  alone: 63 present, 164 missing (the 164 are repo-relative and the worktree's
  `artifacts/` is gitignored/empty — expected, not a defect).
- **13 genuinely-missing links even in the canonical store** (measured this
  session): 4 `qb-backup-news-visibility` smoketest runs
  (`qb_news_channel_test`, `_test2`, `_smoketest`) and 9 `vardec-noisefloor` /
  `vardec-sigma-map` dev runs (`vardec_floor/dev_run1..12`, `vardec_sigma/_smoke`).
  These are throwaway/dev runs whose artifact directories were never persisted
  to the store. The rows still carry their hashes, so they are not uninformative,
  but their linked artifact cannot be re-derived.
- **Path-style inconsistency (gap):** 63 rows store absolute, machine-specific
  paths; 164 store repo-relative paths. A registry row is only self-contained on
  the one machine that produced it. Flag `absolute_machine_path` (63 rows).
- **`source` field inconsistency (gap):** `source` takes 4 shapes — `nfl-ats
  <command>` (123), `artifacts/.../run.json|metadata.json` (74), `scripts/...py`
  (28), and 2 `docs/...md` references. For live CLI rows `source` is intentionally
  the command string (not a path); for backfilled rows it is the artifact file.
  Flag `source_not_a_path` fires on the 123 `nfl-ats <command>` + 2 `docs/...`
  rows (126 total) — informational, by design for live runs.
- **`experiment_id` vs directory slug mismatch (gap, 2 rows):** `pbp-replication-2013-2017 (spec 3.3 driver)/20260816T142340Z` and `qb-continuity-replication-2014-2017 (spec 3.2 driver)/20260816T143913Z`. The `command` field (and thus `experiment_id`) contains spaces/parens, while the file lives under the slugified directory `..._spec_3.3_driver_`. The `experiment_id` string is therefore not directly usable as a path component (a consumer must slugify it). Flag `id_not_filesystem_safe` (2 rows).
- **Separator inconsistency (gap):** `bye-overvaluation-screen/post_fix_seed20260821`
  stores `artifact_directory` with backslashes (`artifacts\bye_overvaluation_screen\...`)
  while every other row uses forward slashes.

## 7. Code implemented (the gap: no way to verify the registry)

RWB-09's DoD says "verify linked artifacts exist" is a manual expectation with
**no tool**. I implemented it as durable, read-only code:

- `nfl_ats.provenance.verify_experiment_links(registry_root=None, *, artifacts_roots=None)`
  — walks `registry/experiments/*/*.json`, resolves each `artifact_directory`
  (absolute as-is; relative stripped-and-resolved against each provided
  artifacts *directory* and the repo root), checks existence, and returns a
  frozen `ExperimentLinkVerification` per row with the flags
  `absolute_machine_path`, `id_not_filesystem_safe`, `source_not_a_path`.
- `nfl-ats experiment verify` CLI subcommand (`_cmd_experiment_verify`) — prints a
  table (rows scanned / present / missing, flag counts, missing-link detail) or
  `--json`; supports `--artifacts-root` (repeatable) and opt-in
  `--require-links` (off by default, because a missing link on a fresh clone is
  expected, not a defect — the hashes are the guarantee).
- Tests in `tests/test_experiment_registry.py`:
  `test_verify_experiment_links_finds_existing_and_flags_missing` (existing vs
  missing link, absolute-vs-relative resolution, `source_not_a_path` only on
  pathless sources) and `test_verify_experiment_links_flags_unsafe_id`
  (`id_not_filesystem_safe` on a space/paren command).

This is additive and does not alter the 227 committed rows or the
`write_experiment_artifact` contract (the existing `artifact_directory ==
str(output)` test still holds). I deliberately did **not** rewrite historical
rows to "fix" the path/source/id inconsistencies: that would alter recorded
provenance and is out of scope for an audit task; the `verify` tool now surfaces
them instead.

## 8. Quality gates (measured, this session)

```
ruff format --check .   -> OK
ruff check .            -> OK
mypy src                -> Success: no issues found in 102 source files
pytest (full suite)     -> 1806 passed, 5 skipped (local data absent), 0 failed
pytest test_experiment_registry.py + test_provenance.py -> 28 passed
```

## 9. Recommendations (not implemented — out of scope / would alter history)

1. **Make future `artifact_directory` values repo-relative and slash-normalised**
   in `write_experiment_artifact` (store `os.path.relpath(directory, repo_root)`
   with forward slashes), so new rows are portable. Existing 227 rows retain
   their values; `verify` already resolves both styles.
2. **Normalise the `source` field** to one of two shapes (command string for live
   CLI runs, artifact path for backfilled rows) and document it.
3. **Re-point or annotate the 13 missing dev/smoketest links** so they are not
   mistaken for reproducible runs; consider excluding dev/smoke runs from the
   tracked registry.
4. **Slugify `command` into `experiment_id`** for future rows so the id is
   path-safe; leave the 2 historical `(spec … driver)` rows as-is or add an
   explicit `directory` field.

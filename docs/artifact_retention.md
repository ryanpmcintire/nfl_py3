# Artifact retention policy (WP6 / ROADMAP OPS-02)

Measurement and a dry-run pruning **planner** for `artifacts/` and `data/`.
The tool is `scripts/artifact_retention.py`; this document records what it
measured on this machine, the retention policy that measurement supports,
and the safety rules the tool enforces. **There is no delete mode.** This
pass is measurement and planning only -- nothing under `artifacts/` or
`data/` was moved, renamed, or deleted to produce this document.

## Why this exists

`artifacts/` and `data/` are entirely gitignored except for
`artifacts/prospective/` and the `.gitkeep` markers under `data/raw/` and
`data/processed/` (**read**, `.gitignore:20-40`). Everything else -- every
experiment screen's timestamped output, every raw scrape snapshot, every
processed feature table -- exists only on this machine and, for `data/`,
partially on the `scripts/backup_data.py` mirror. Nobody had ever measured
how large these trees are, which runs are still load-bearing, and which are
safe to eventually compact or prune (ROADMAP OPS-02, **read**,
`ROADMAP.md:417`: "Keep manifests/ledgers, compact or prune large derived
files"). This is that measurement.

## Measured: `--report`

Command run on this machine:

```
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --report
```

**Measured** (command above, run 2026-09-01T19:01Z), by top-level tree:

| tree | runs | bytes | oldest stamp | newest stamp | protected runs | largest file |
|---|---:|---:|---|---|---:|---|
| `artifacts` | 435 | 2.2 GB | 20260812T101321Z | 20260901T190016Z | 383 | `artifacts/rehearsal_lockday/sim/data/raw/injury_news/20260819T191639Z/index.parquet` (33.0 MB) |
| `data/raw` | 175 | 749.9 MB | 20260812T101244Z | 20260901T184600Z | 175 | `data/raw/injury_news/20260819T191639Z/index.parquet` (33.0 MB) |
| `data/processed` | 39 | 147.2 MB | n/a (untimestamped) | n/a | 39 | `data/processed/game_features_weak_stack_js_prior.parquet` (9.4 MB) |
| `data/market` | 8,768 | 967.6 MB | 20200825T115500Z | 20260901T130035Z | 8,768 | `data/market/raw/20220924T145539Z-ncaaf/response.json` (525.1 KB) |
| `data/players` | 7 | 46.8 MB | 20260812T200527Z | 20260817T184920Z | 3 | `data/players/raw/20260817T184901Z/weekly_rosters.parquet` (3.3 MB) |
| `data/other` (cfb, pbp, quarterbacks) | 55 | 1.7 GB | 20260812T142851Z | 20260901T185247Z | 55 | `data/cfb/pbp/raw/20260816T162700Z/source/play_by_play_2025.parquet` (56.5 MB) |

Total measured on-disk footprint across both trees: **~5.75 GB**, of which
`artifacts/rehearsal_lockday/` alone is ~1.9 GB (see "The rehearsal_lockday
anomaly" below -- it is not really 1.9 GB of unique data).

The `artifacts` row decomposes into 145 families (full table in the
`--report` output; not reproduced here in full). Notable entries:

- Large, **mostly unprotected** families -- exactly where a retention policy
  has something to do: `margins` (9 runs, 60.9 MB, 9/9 protected right now
  only because the active model and prospective ledger cite specific runs --
  see below), `backtests` (8 runs, 28.7 MB, 5/8 protected), `experiments` (9
  runs, 26.8 MB, 8/9 protected), `player_experiments` (4 runs, 42.0 MB, 4/4
  protected today), `player_model_selection` (1 run, 42.9 MB, 0/1
  protected), `cfb_role_experiments` (4 runs, 13.3 MB, 4/4 protected).
- `.uv-cache` (6 files, 1.5 KB, 0 protected) is a stray `uv` package-manager
  cache directory that ended up under `artifacts/`, not a research artifact.
  Harmless at this size; flagged here as a finding, not a rule change.
- `rehearsal_lockday` (1 run, 1.9 GB, protected) -- see below.

## Measured: `--plan`

```
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --plan
```

**Measured** result at the default 30-day threshold: **0 candidates, 0
bytes**. The repository is young (oldest timestamped run is
2026-08-12, 20 days before this measurement) and the handful of runs old
enough to qualify are all either the newest run of their family or cited by
a doc/registry entry.

To confirm the mechanism actually finds something (not just trivially
empty), the same command at a 14-day threshold:

```
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --plan --older-than-days 14 --json
```

**Measured**: 14 candidates, 10,344,960 bytes (~9.9 MB) -- 12 from
`artifacts/` (7.1 MB: superseded `backtests`, `nested_evaluations`,
`experiments`, `weak_stack_v2`, `dependence`, `anytime`, `sec_pilot`,
`edge_audit_redteam` runs, plus the loose `decision_rule_measurements.csv`)
and 2 from `data/players` (3.2 MB). None of these are the newest run of
their family and none are cited by any doc, registry entry, or the active
model manifest.

## The `rehearsal_lockday` anomaly

`artifacts/rehearsal_lockday/sim/` (built by `scripts/lockday_rehearsal.py`)
is a full **hard-link** mirror of most of `data/`, built for a dry-run
publish rehearsal (**read**, `docs/week1_readiness.md:373`). Hard links
share the same on-disk blocks as the files under `data/` -- the reported 1.9
GB is not 1.9 GB of unique bytes, and deleting those links would not free
that much space (or, in most cases, any space, since `data/`'s copy of the
same inode would remain). `scripts/artifact_retention.py` therefore treats
`artifacts/rehearsal_lockday` as one coarse, non-decomposed node
(`COARSE_NO_DESCEND` in the script) rather than re-discovering thousands of
duplicate "runs" that are already counted under `data/`. It is separately
protected outright because `docs/week1_readiness.md` cites it wholesale.

## Backup coverage (relevant to what is safe to ever prune)

**Measured 2026-09-02**, `./.tools/uv.exe run --no-sync python
scripts/backup_data.py --status --include-artifacts`, after the owner approved
the wider off-device mirror:

```
=== E:\nfl_data_backup ===
tree                          files         size covered  pending
data                         43,046       3.6 GB  100.0%        0
artifacts                    44,047       2.3 GB  100.0%        0
TOTAL                        87,093       5.9 GB  100.0%        0
```

The widening run copied and SHA-256-verified 44,147 files (2.4 GB), including
the 100 pending `data/` files and all 44,047 artifact files (**measured**, the
preceding `backup_data.py --include-artifacts` apply-mode report).
`artifacts/prospective/` and
`artifacts/clv_ledger/` therefore no longer have a single-machine failure
mode. The weekly `backup_data` scheduler command now passes
`--include-artifacts`, so future incremental runs maintain both trees
(**read**, `scripts/capture_scheduler.py`, `backup_data` job).

## ENG-19 (2026-09-04): retention classes, disk budget, and gap closes

ROADMAP ENG-19 asked this WP6/OPS-02 pass to be audited against a concrete
definition of done: "Inventory ignored artifact growth, define retention
classes and safe pruning rules, and add a dry-run budget check that never
removes evidence needed by a registry row or published forecast." This
section records that audit and what it changed. **Still no delete mode
anywhere** -- `scripts/artifact_retention.py` and the new
`src/nfl_ats/artifact_retention_policy.py` are both measurement/planning
only; `tests/test_artifact_retention_policy.py::
test_no_delete_prune_or_apply_function_exists_anywhere` asserts neither
module defines a delete/prune/apply function.

### Refreshed inventory, measured 2026-09-04T19:08:15Z

```
python scripts/artifact_retention.py --report
```

| tree | runs | bytes | protected runs |
|---|---:|---:|---:|
| `artifacts` | 679 | 2.4 GB | 545 |
| `data/raw` | 197 | 802.2 MB | 197 |
| `data/processed` | 44 | 194.1 MB | 44 |
| `data/market` | 8,770 | 968.7 MB | 8,770 |
| `data/players` | 9 | 47.6 MB | 9 |
| `data/other` | 70 | 1.7 GB | 64 |

Total measured footprint across both trees: **~6.1 GB** (up from ~5.75 GB on
2026-09-01, three days earlier -- both dates are in this repo's short
history, so treat any growth-rate figure derived from the gap as an
**inferred**, low-confidence extrapolation, not a trend). 762 protected
reference nodes discovered (up from 753 the same session, before the
`ALWAYS_PROTECTED` additions below landed -- expected, since two of those
additions are hardcoded paths that count as discovered references the
moment the code changes, independent of any doc edit). Two concurrent
sessions' new families are already visible in the `artifacts` family
breakdown: `refresh_triggers` (2 runs, 672.4 KB, 2/2 protected) and
`prospective_scorecards` (2 runs, 120.8 KB, 2/2 protected) -- both ENG-08
and ENG-06 work landing mid-session. `artifacts/lockday_packages/` does not
exist on disk yet (ENG-01, still in flight); its hardcoded protection and
JSON-glob reference scan are wired in ahead of that, and are no-ops against
an empty/nonexistent directory in the meantime.

At the default 30-day `--plan` threshold: **0 candidates** (same as
2026-09-01 -- the repository is still young). At 14 days: **23 candidates,
~7.4 MB** (`--plan --older-than-days 14 --json`, measured), all classified
`reproducible` (20, mostly superseded `backtests`/`nested_evaluations`/
`experiments`/`weak_stack_v2` runs) or `scratch` (3, the `.uv-cache` stray
already flagged on 2026-09-01) -- zero `evidence` or `point_in_time_capture`
entries, confirming the class-based exclusion below is doing its job on the
real tree, not just the synthetic test fixtures.

### Retention classes (new: `src/nfl_ats/artifact_retention_policy.py`)

Every run the planner discovers now gets exactly one named class via
`retention_policy.classify(tree, rel, protected=...)`, and every `--plan`
candidate carries it (`retention_class` field, also a column in the text
table):

| class | prunable | min age | rule |
|---|---|---|---|
| `evidence` | never | -- | `RunNode.protected` from the existing reference scan (registry/docs/manifest/challengers.json/hardcoded paths), relabelled |
| `point_in_time_capture` | never | -- | `data/raw`, `data/market`, `data/players` wholesale, plus any run with a literal `raw` path segment, plus specific `artifacts/` families named in `POINT_IN_TIME_ARTIFACT_PREFIXES` |
| `scratch` | any age | 0 days | a `tmp`, `uv-cache`, `.uv-cache`, `__pycache__`, `.pytest_cache`, or `.agent_tmp` path segment |
| `reproducible` | yes | 30 days | everything else (default fallthrough) |

`REPRODUCIBLE_MIN_AGE_DAYS = 30` is derived, not guessed: `backup_data` runs
weekly, Sunday 22:00 America/New_York (**read**, `scripts/capture_scheduler.py`
`SCHEDULE`, the `backup_data` job), so 30 days is a ~4x multiple of that
cadence -- a reproducible run stays a candidate only after roughly four
weekly mirror windows have had the chance to run, which is slack enough for
two or three consecutive missed windows (the `MISSED`-row failure mode
AGENTS.md tells every session to check for) to still leave one completed
mirror pass before the run is old enough to plan around. See the module's
own docstring for the full derivation.

### Gap closed: `data/raw`/`data/market`/`data/players` are now hard-protected in code

Before this pass, `data/raw` and `data/market`'s protection was **entirely
incidental**: `--plan` found them non-candidates only because README.md
happens to mention those paths, and this doc's own "Safety rules" section
(rule 3, below) already flagged that as a fragility -- "the *policy* intent
... should survive even if a doc edit ever thinned out the reference." That
gap is now closed in code: `build_plan` calls
`retention_policy.is_point_in_time_capture(tree, rel)` and excludes a match
unconditionally, before the reference-protection check even matters. The
regression test that used to encode the *old*, incidental behavior --
`test_plan_data_raw_old_run_is_a_candidate_when_unreferenced` -- has been
rewritten as `test_plan_data_raw_old_run_is_never_a_candidate_even_when_unreferenced`
in `tests/test_artifact_retention.py`, and
`tests/test_artifact_retention_policy.py::test_point_in_time_capture_never_appears_in_a_prune_plan`
adds a synthetic-repo regression at the loosest possible age threshold
(`older_than_days=0`) so age can never be the only thing keeping a capture
out of a plan.

### Gap closed: also-protected paths

Four additions to `ALWAYS_PROTECTED` / the reference-scan glob sources in
`scripts/artifact_retention.py`, plus one explicit non-addition:

- `artifacts/lockday_packages/` and `artifacts/lockday_packages_rehearsal/`
  (ENG-01's independently verifiable lock-day decision packages and their
  rehearsal counterpart, `src/nfl_ats/lockday_package.py` /
  `scripts/lockday_package_verify.py`, added concurrently this session) --
  protected wholesale like `artifacts/prospective/`, and every JSON file
  under either is scanned for path references the same way
  `registry/experiments/**` is, so a package manifest citing a specific run
  elsewhere protects that run too.
- `data/scheduler_state.json` / `data/scheduler_log.txt` (the capture
  scheduler's persisted heartbeat and run log, ROADMAP ENG-03) -- already
  protected today via a `docs/capture_scheduling.md:345` mention, but that
  is exactly the "incidental via one doc sentence" pattern this pass is
  closing elsewhere, so both are now hardcoded too.
- `artifacts/refresh_triggers/` (ENG-08, added concurrently this session) is
  classified `point_in_time_capture` via a new
  `POINT_IN_TIME_ARTIFACT_PREFIXES` set in the policy module -- it has no
  literal `raw` path segment, so it needed an explicit entry rather than
  falling out of the generic heuristic. **Reported** (this session's
  coordinator instruction, not independently derived): it records exactly
  when/why a late-week pick refresh fired, which cannot be reconstructed
  after the fact.
- `artifacts/prospective_scorecards/` (ENG-06, added concurrently this
  session) is **deliberately left unprotected and unclassified specially**
  -- it holds summary scorecards derived from the evidence-protected
  `artifacts/prospective/` ledgers, so `classify` falls through to
  `reproducible` for it with no code change needed. Named in the policy
  module's comments and covered by
  `test_classify_prospective_scorecards_is_reproducible_by_default` so the
  omission reads as a decision, not a gap.

### Disk budget (`--budget-check`, dry-run, read-only)

`BUDGET_BASELINE_BYTES` in `src/nfl_ats/artifact_retention_policy.py` is the
**measured** per-tree total from the `--report --json` run below, dated
2026-09-04. `DEFAULT_BUDGET_MULTIPLIER = 5.0` turns it into a budget
(`baseline * multiplier`); `--budget-multiplier` overrides it per invocation.
`--budget-check` reports each tree's used vs. budget bytes, free disk space
(`shutil.disk_usage`), the off-device mirror's last-mirrored timestamp (a
read-only peek at `backup_manifest.json`, never a re-hash), and what a
`--plan --older-than-days 30` run would reclaim for that tree -- **without
deleting anything**. Exit code is 1 if any tree is over budget, 0 otherwise.

**Measured** (`python scripts/artifact_retention.py --budget-check`, run
2026-09-04T19:08:59Z):

```
tree                used    budget    status  reclaimable
---------------------------------------------------------
artifacts          2.4GB    11.8GB     under           0B
data/raw         802.2MB     3.9GB     under           0B
data/processed   194.1MB   970.5MB     under           0B
data/market      968.7MB     4.7GB     under           0B
data/players      47.6MB   237.8MB     under           0B
data/other         1.7GB     8.3GB     under           0B

All trees under budget (exit code 0)
```

Free space at measurement time: 977.5 GB free of 1863.0 GB total on F:
(**measured**, `shutil.disk_usage`); off-device mirror last synced
2026-09-04T17:37:37Z (**measured**, `backup_manifest.json` on `E:`). All six
trees' combined ~32 GB budget is under 4% of free space -- exhaustion is not
the near-term risk this guards against, ignored growth is; see the policy
module's docstring for the per-tree lead-time reasoning.

To confirm the check actually fires (not just trivially always-under, the
same "prove it's not vacuous" discipline the `--plan --older-than-days 14`
demo below already uses): `--budget-multiplier 0.5` on the same measurement
puts every tree over its (now-halved) budget and returns exit code 1
(**measured**, same command with `--budget-multiplier 0.5`) -- confirming
both the over/under comparison and the non-zero exit code work, without
anything being deleted, moved, or renamed (`reclaimable` stayed `0B` for
every tree in both runs, because the default `--plan --older-than-days 30`
comparison it reuses finds 0 candidates today -- see the `--plan` measurement
above).

## Retention policy

**Kept forever, regardless of age, by the tool's own protected-set logic**
(never a hardcoded family-name list except the three exceptions below --
`collect_protected_refs` discovers everything else programmatically from
file content):

- Every run (or bare family) cited by a string value anywhere in
  `registry/weak_signals.json`, `registry/rotation_registry.json`,
  `registry/experiments/**`, `registry/experiment_specs/*`, `docs/*.md`,
  `README.md`, `ROADMAP.md`, `HANDOFF.md`, or `CURRENT_PREDICTIONS.md`. A
  bare, non-timestamped mention (e.g. "outputs land under
  `artifacts/foo/`") protects every run inside that family, not just one --
  most of the repo's small, well-documented experiment families end up
  protected this way, which is why the measured `--plan` above finds
  candidates mainly in the large families that no doc happens to name in
  passing (`margins`, `backtests`, `player_experiments`, ...).
- `artifacts/active_ats_model.json` itself, plus both runs its own manifest
  points at (`historical_evaluation.artifact`, `weekly_forecast.artifact`).
- `artifacts/prospective/` in full (challenger ledgers and paper-decision
  parquet files -- carved out of `.gitignore` on purpose) and
  `artifacts/clv_ledger/` in full (the append-only paper-decision ledger
  behind `load_paper_decisions()` / `record_paper_decisions()` in
  `src/nfl_ats/clv.py`, exactly the "ledger" AGENTS.md says must never be
  lost).
- The single **newest** run within its own group (same immediate parent
  directory), independent of any reference -- a light extra safety net so a
  bug or gap in the reference scan can never strand a family with zero
  surviving output.
- Everything under `data/raw/` and `data/market/` in practice (see measured
  report: 175/175 and 8,768/8,768 protected today) -- both README.md and
  `scripts/backup_data.py`'s own docstring describe these as point-in-time
  scrapes of pages that have since changed and **cannot be re-fetched at
  their original timestamps**; losing one costs an observation permanently,
  not a re-download. Treat this as policy, not incidental: even a future
  doc edit that removed the generic mention should not make raw captures
  prunable by default.

**May be compacted** (not part of this pass, a future decision): flat,
frequently-rebuilt `data/processed/*.parquet` + `.manifest.json` pairs are
already single-current-file by construction (rebuilt in place, not
timestamped runs) -- there is nothing to compact there beyond what already
happens. The real compaction candidate is repeated near-duplicate
`predictions.parquet` / `results.json` payloads across many small
experiment runs in fast-iterating families (e.g. `graph_team_stat_*`,
`vardec_*`); a future pass could define a per-family "keep every run" vs.
"keep newest N" policy, but that is a design decision for the owner, not
this measurement.

**Prunable after N days** (`--plan --older-than-days N`, default 30): a run
that is (a) not the newest of its family/group, (b) not cited anywhere, and
(c) older than the threshold, measured from its own timestamp when the run
is timestamp-named, or filesystem mtime when it isn't. As measured above,
nothing clears the default 30-day bar today; at 14 days the tool already
finds 14 real candidates totaling ~9.9 MB, so the mechanism is confirmed
working, not just conservatively empty by construction.

## Safety rules

1. **Dry-run by default, and there is no delete mode.** `--report` measures;
   `--plan` prints candidates and their byte totals and does nothing else.
   `scripts/artifact_retention.py` has no `delete_candidates` / `prune` /
   `apply_plan` function -- `tests/test_artifact_retention.py::
   test_main_has_no_delete_flag` asserts this so a future edit cannot add
   one silently.
2. **The protected set is rebuilt from scratch on every run** by scanning
   current file content (docs, registry, the active model, the challenger
   ledger) -- it is never cached, never assumed stable across sessions, and
   never guessed. If a doc reference moves or a family gets renamed, the
   next run reflects it immediately.
3. **`data/raw/`, `data/market/`, and `data/players/`'s scraped snapshots are
   never a target, in code, unconditionally (ENG-19, 2026-09-04).** Before
   this pass the protection was entirely incidental -- a doc-reference match,
   not an exemption written into the code -- which is exactly the fragility
   this rule used to warn about. `build_plan` now calls
   `retention_policy.is_point_in_time_capture(tree, rel)` and excludes a
   match unconditionally, before the reference-protection check is even
   consulted, so a future doc edit can no longer silently make a raw capture
   prunable. See "Gap closed: `data/raw`/`data/market`/`data/players` are
   now hard-protected in code" above.
4. **Never treat something as safe to prune ahead of confirmed backup
   coverage.** `scripts/backup_data.py --status --include-artifacts` is the
   command that answers "is this mirrored." As measured 2026-09-02, both
   trees are 100.0% covered with zero pending (see "Backup coverage" above).
   Any future apply-mode design must re-run that status check and refuse to
   proceed when either tree has pending files; today's clean result is not a
   permanent assumption.
5. **Reparse-point defense.** `scan_subtree` and the run-discovery walk
   never follow a symlink or junction (checked via `is_symlink()` before
   every descent); a stray one is recorded, not traversed. No junction
   exists under `artifacts/` or `data/` today (**measured**,
   `Get-ChildItem -Force -Recurse -Directory ... | Where LinkType -in
   Junction,SymbolicLink` returned nothing), but this repo has hit the
   hazard before (see memory: "Worktree junction hazard," 2026-08-16
   incident), so the defense is unconditional rather than reactive.
6. **`--budget-check` is dry-run and read-only too (ENG-19).** It computes
   over/under-budget status from `--report`'s own byte counts and reuses
   `--plan`'s existing dry-run candidate list for the "what would this
   reclaim" figure -- there is no second code path that touches the
   filesystem differently, and its two read-only probes
   (`shutil.disk_usage`, and a peek at the mirror's own
   `backup_manifest.json`) never write, re-hash, or delete anything. Exit
   code is non-zero only to make an over-budget tree visible to a scheduled
   caller; it never gates or blocks anything on its own.

## Next step for the owner

Before any delete mode is ever written for this tool:

1. **Approve the policy above** (what's forever-kept, what's compactable,
   what's prunable after N days) -- specifically, confirm the "newest run
   always survives" safety net and the doc/registry-citation protection are
   the right bar, or say what should change.
2. **Done 2026-09-02:** the owner approved backing up the wider artifact tree;
   `scripts/backup_data.py --include-artifacts` completed and the follow-up
   status reported both trees 100.0% covered with zero pending. The scheduled
   weekly job now includes artifacts too.
3. Only after step 1 should a follow-up work package add an explicit,
   separately-reviewed `--apply` (or similar) mode -- gated on this same
   protected-set logic, requiring an interactive confirmation or an
   explicit `--yes`, and almost certainly writing to a trash/quarantine
   directory first rather than deleting outright, so a bad `--plan` run can
   never become an unrecoverable one.

## Tool reference

```
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --report [--json]
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --plan [--older-than-days N] [--json]
./.tools/uv.exe run --no-sync python scripts/artifact_retention.py --budget-check [--budget-multiplier X] [--json]
```

Tests: `tests/test_artifact_retention.py` (34 tests) exercise path-reference
extraction (including two regression cases: a CLI-usage template placeholder
that must not collapse to a bare tree root, and Windows-backslash absolute
paths from `registry/experiments/**/*.json`), the ancestor-inclusive
protection lookup, the `COARSE_NO_DESCEND` exception, byte accounting, the
newest-run guard, the age-threshold filter, and both CLI entry points -- all
against a synthetic `tmp_path` repo, never the real `artifacts/`/`data/`
trees. `tests/test_artifact_retention_policy.py` (29 tests, added 2026-09-04
for ENG-19) exercises `retention_policy.classify`/`is_scratch`/
`is_point_in_time_capture` against every class (including the
`refresh_triggers` and `prospective_scorecards` cases above), the budget-byte
math, the two read-only filesystem probes, and the `scripts/artifact_retention.py`
wiring: every `--plan` candidate's `retention_class`, the point-in-time-
capture-never-in-a-plan invariant, `--budget-check`'s exit code in both
directions (`--budget-multiplier 0` forces an over-budget synthetic repo),
and that `--report`/`--plan`/`--budget-check` stay mutually exclusive.
63 tests total across both files, all against synthetic `tmp_path` repos.

# ref-cfb-shared: extract mechanical duplication from the CFB cluster

Task: execute `scripts/swarm/tasks/ref-cfb-shared.md` on branch
`swarm/ref-cfb-shared`. The referenced upstream audit
(`reports/wave1/hyg-cfb-cluster.md`) **does not exist in this worktree**
(measured: `find reports -iname "*hyg*"` → no matches this session), so the
duplication targets were re-derived directly from the three source files and
from the audit task's own description in `scripts/swarm/tasks/hyg-cfb-cluster.md`
(read this session): "schedule loading, team-code normalization, bootstrap
plumbing that re-implements src/nfl_ats utilities."

## What changed

New module **`src/nfl_ats/cfb_common.py`** (189 lines) holding only moved,
verbatim-equivalent helpers, in three families:

### 1. Snapshot loading plumbing

- `snapshot_manifest_paths(raw_root, label)` / `latest_manifest_path(...)` —
  the sorted `glob("*/manifest.json")` + FileNotFoundError dance that was
  re-implemented inline in `cfb.py::latest_cfb_snapshot` and (as a variant)
  `cfb_features.py::cfb_season_partitions`. Error message text preserved
  exactly (`"No CFB {key} snapshots found in {root}"`).
- `manifest_payload(path)` / `require_manifest_payload(root, msg)` — manifest
  JSON parsing used by `cfb_snapshot_from_root`, `summarize_cfb_snapshots`,
  and `cfb_season_partitions`.
- `season_partition_path(root, season, filename)` — the `f"season={season}"`
  path convention previously duplicated between `CfbSnapshot.season_path`
  (cfb.py) and `cfb_features.py` (read: both sites pre-refactor).
- `load_parquet_partitions(paths, columns)` — read + `pd.concat(ignore_index=True)`
  stack used by `load_cfb_snapshot` (cfb.py) and `load_cfb_seasons`
  (cfb_features.py); each caller keeps its own existence/missing checks.

### 2. Column normalization ("code normalization") plumbing

Moved from `cfb.py` (was `_fill_missing_columns`, `_require_single_season`)
and generalized from the repeated cast loops:

- `fill_missing_columns`, `require_single_season` — moved verbatim; all ~20
  call sites in `cfb.py` renamed.
- `cast_int_columns` / `cast_nullable_int_columns` / `cast_float_columns` /
  `cast_string_columns` — replace eleven `for column in (...): result[column]
  = pd.to_numeric(...).astype(...)` loops across `cfb.py` canonicalizers plus
  one in `cfb_features._filtered_schedule`. Same operations, same order,
  same dtypes (`int64` / `Int64` / float / pandas `string`).

### 3. Bootstrap plumbing

- `week_block_indices(frame)` — moved verbatim from
  `cross_league_transfer._week_block_indices`.
- `blocked_bootstrap_positions(blocks, samples, seed)` — generator form of the
  draw loop in `cross_league_transfer._bootstrap_theta_variance`; consumes one
  `np.random.default_rng(seed)` in sample order, so resamples are bit-identical
  to the previous inline implementation.

## What deliberately did NOT change

- **The market-aggregation core is untouched**, as instructed:
  `_season_abbr_map`, `_repair_game_sides`, `_oriented_spread_rows`,
  `build_cfb_market_table`, `build_cfb_team_states`,
  `attach_cfb_team_states` in `cfb_features.py` are byte-for-byte unchanged
  apart from nothing — verified by reading the diff. Their casts (e.g.
  inside `build_cfb_market_table`) were left inline on purpose.
- `cfb_benchmark.py`, `snapshots.py`, `pbp.py`, etc. also contain similar
  manifest/bootstrap patterns but are outside this task's three-file scope;
  they can adopt `cfb_common` later.
- No contracts, error messages, dtypes, sort orders, RNG streams, or public
  signatures changed anywhere.

## Line accounting

- `git diff --stat` (measured): cfb.py −110/+40 net region rewritten;
  total 3 files changed, 77 insertions(+), 110 deletions(-) of consumer code,
  plus the new 189-line shared module. Net repo growth is +56 lines because
  the shared module carries docstrings; duplicated *statement* count dropped
  by roughly 110 lines. The task brief's "~865 lines" figure could not be
  checked against its source audit (file absent, see above) — unverified;
  I extracted every mechanically duplicated block I could find among the
  three files without touching contract-bearing logic.

## Verification (all measured this session)

```
ruff format --check .   # 637 files already formatted
ruff check .            # All checks passed!
mypy src                # Success: no issues found in 106 source files
pytest                  # 1855 passed, 5 skipped (two consecutive full runs)
```

pytest ran with `PYTHONPATH=<worktree>/src` against the main venv and
`--basetemp` outside the repo, per worker constitution. The first full run
showed one setup/teardown ERROR in
`tests/test_active_model.py::test_active_manifest_links_exact_evaluation_and_forecast`;
it did not reproduce in either subsequent full-suite run, passes standalone
and whole-module, and the clean-tree (stashed) full suite passed 1855 —
I judge it an environment flake around the shared basetemp on Windows,
not an effect of this change. Flagging it rather than hiding it.

No experiment windows were spent: this refactor runs no model/scoring look
(rule: executing a scoring look requires a predeclared window; none was
granted or needed here).

## Files

- `src/nfl_ats/cfb_common.py` (new)
- `src/nfl_ats/cfb.py` (rewired)
- `src/nfl_ats/cfb_features.py` (rewired)
- `src/nfl_ats/cross_league_transfer.py` (bootstrap rewired)

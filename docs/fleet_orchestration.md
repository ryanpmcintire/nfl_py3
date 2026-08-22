# Fleet orchestration with git worktrees

How to run many experiments in parallel against this repository without
duplicating its heavy local state (data/, artifacts/, .venv, .tools). The
enabling tool is [scripts/fabricate_worktrees.ps1](../scripts/fabricate_worktrees.ps1).

## Why worktrees plus junctions

`git worktree add` gives each worker a full checkout sharing one object store,
but a fresh checkout is inert here: everything that makes an experiment runnable
is deliberately gitignored (raw/processed data, fitted models, the uv-managed
virtual environment and toolchain). Copying gigabytes per worker is slow and
silently forks the dataset mid-wave. Windows junctions make the ignored state
appear inside each worktree while storing it once, on disk.

## Usage

```powershell
# Plan first (lists every operation, changes nothing)
.\scripts\fabricate_worktrees.ps1 -Name wave1 -Count 4 -WhatIf

# Fabricate ..\nfl_py3_wt_wave1_1 .. ..\nfl_py3_wt_wave1_4 off current HEAD
.\scripts\fabricate_worktrees.ps1 -Name wave1 -Count 4

# Tear down the whole wave
.\scripts\fabricate_worktrees.ps1 -Name wave1 -Remove
```

Worktrees land as siblings of this repo (`..\nfl_py3_wt_<name>_<i>`, i from 1),
detached at the fabricate-time HEAD so parallel workers never contend for one
branch ref. `-Name` accepts `[A-Za-z0-9_-]` only.

## What is shared vs private

| Path in worktree | Kind | Semantics |
| --- | --- | --- |
| `data\` | junction to main repo | physically shared read-write; treat as strictly READ-ONLY |
| `.venv\` | junction to main repo | shared interpreter/site-packages; read-only in practice |
| `.tools\` | junction to main repo | shared uv binary; read-only |
| `artifacts\` | real directory | fully writable, per-worktree |
| `artifacts\opener_evaluation\`, `artifacts\market_decomposition\`, `artifacts\prospective\` | junctions into main repo | read-only reference copies (only those present in the main repo at fabricate time) |
| everything git-tracked | normal checkout | per-worktree |

Design rule: artifacts must NOT be shared for writes. Each experiment writes
under its worktree's own `artifacts\<experiment>\`; the three junctioned
subdirs exist so workers can read historical evaluations without copying them.
If the main repo later gains another read-heavy artifacts subtree, add it to
`$junctionArtifactSubdirs` in the script.

Fabrication replaces only git-tracked-and-clean content that the checkout
itself created (e.g. `data/raw/.gitkeep`, `artifacts/prospective/challengers.json`)
and fails loudly rather than deleting anything untracked, ignored, or modified.
It never deletes a pre-existing non-junction path.

## Per-wave lifecycle

1. **Plan**: `-WhatIf`; confirm the op list matches expectations.
2. **Fabricate**: run from a clean master at the commit you want workers to
   start from. All members of a wave see the same HEAD.
3. **Fan out**: launch one orchestrator process per member, cwd set to that
   member. Tools run identically inside a worktree:
   `.\.tools\uv.exe run python ...`.
4. **Harvest**: collect outputs from each member's private `artifacts\`
   directory. Do not rely on the worktree surviving harvest.
5. **Remove**: `-Name <wave> -Remove`. The script strips junctions first
   (verifying each target survived), deletes only the per-worktree artifacts
   copy, then runs `git worktree remove --force` and `git worktree prune`.

## Known limits

- **Shared venv, last-writer-wins installs.** `uv run` inside a worktree
  re-resolves the project package and may rebuild/re-point the `nfl-ats`
  install inside the ONE shared `.venv` at that worktree's path (observed:
  `Built nfl-ats @ file:///F:/Repos/nfl_py3_wt_smoke_1`). Consequences: do not
  run `uv lock`, `uv sync`, or version-changing commands concurrently; after a
  wave ends, run any `.\.tools\uv.exe run python ...` once in the MAIN repo so
  the install points back here (verified to self-heal this way). Prefer
  invoking entry points through `uv run --no-sync` when several workers run
  simultaneously.
- **Do NOT run `git clean` inside a worktree.** `git clean -x` can traverse
  junctions and delete main-repo data. Deletion goes through the script's
  `-Remove`, which removes junction reparse points non-recursively
  (`[System.IO.Directory]::Delete`) and never `Remove-Item -Recurse` on a link
  (unsafe on Windows PowerShell 5.1).
- **Windows path length.** Junctioned trees keep their main-repo depth, but the
  worktree prefix adds ~15+ characters to every path. Deep data subtrees plus a
  long `-Name` can exceed MAX_PATH (260) for tools that don't support long
  paths. Keep names short; prefer enabling filesystem long paths if you hit it.
- **One object store.** Avoid `git gc`/aggressive maintenance while a fleet is
  active. Each worktree has its own index; commits in one are instantly visible
  to all.
- **Junction shadowing of tracked files.** `artifacts/prospective/challengers.json`
  is tracked; through the junction the worktree sees the MAIN repo's working-tree
  version, so `git status` there may report it modified even though the worker
  never touched it. Exclude it from worker diffs. Same logic means edits to it
  inside a worktree write to the main repo's file.
- **data/ is shared read-write physically.** Any pipeline that WRITES under
  `data/` must not run while more than zero fleets are active, or run in exactly
  one designated member.
- **Detached HEAD by default.** Committing directly on detached HEAD risks
  orphaned work; create a branch before committing (next section).

## Merge protocol

- Worker branches optional: inside a worktree, `git switch -c fleet/<name>` before
  committing. Small diffs may instead be exported as patches
  (`git diff > patch.diff`) and applied by the orchestrator, avoiding branch
  litter entirely.
- Keep diffs small and scoped: source/test changes only, no artifacts, no
  registry JSON, no data dumps. The junction-shadowed files above are excluded.
- Orchestrator merges to `master` in batches. Gates (ruff format/check, mypy,
  pytest) run ONCE per merged batch on master, not per worker; workers should
  run fast targeted tests locally, not the full gate battery, to avoid N-way
  contention on the shared caches.
- Rebase or merge master into long-lived worker branches before final harvest
  so batches stay small.
- Never push from a fleet member; the orchestrator pushes master after the
  batch gates pass.

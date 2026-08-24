# ref-cli-split — Phase 1: CLI argument-family consolidation

**Task**: ref-cli-split (executes phase 1 = audit §2, refactors #1–#6 of
`reports/wave1/hyg-cli.md` on branch `swarm/hyg-cli`, commit 7e9d762)
**Date**: 2026-08-24
**Branch**: `swarm/ref-cli-split`
**Scope**: `src/nfl_ats/cli.py` only. Behavior-preserving: every command keeps
its exact name, flags, defaults, help text, and parse results.

---

## Result

| Metric | Before | After | Delta |
|---|---|---|---|
| `src/nfl_ats/cli.py` total lines (**measured**, `wc -l`) | 5,773 | 5,706 | **−67** |
| `build_parser()` section lines (**measured**, awk span) | 1,534 | 1,335 | **−199** |
| Copy-pasted `add_argument` blocks removed (**measured**, git diff) | — | — | −340 / +273 |

All four gates pass (**measured** this session):

- `ruff format --check .` — 636 files already formatted
- `ruff check .` — all checks passed
- `mypy src` — no issues in 105 source files
- `pytest` — **1855 passed, 5 skipped** (skips are pre-existing
  local-data-absent skips; **read** at the skip sites)

### Honest note vs. the ~350-line target

The ~350-line target traces to the audit's §2 estimate (−805 for families
#1–#6), which assumed ~11 lines per `--features` occurrence and similar.
**Measured** this session: most `--features` blocks were 3–7 lines, bootstrap
pairs were 2 single-line statements, and roughly a third of each family's
"current lines" reappear as one-line helper calls plus the +141-line helper
block itself. The mechanical ceiling for pure parser dedup was therefore far
below the estimate. I stopped at the honest floor rather than forcing a
registration-table rewrite of unique help texts, which would add risk without
removing duplication.

## What changed

Eleven shared registration helpers now own the repeated argument families; each
command registers them at the original call-site position, so per-command flag
order — and therefore `--help` output — is unchanged:

| Helper | Replaces | Sites |
|---|---|---|
| `_add_features_arg` | `--features` under `data/processed/` (audit #1) | 32 |
| `_add_bootstrap_args` | `--bootstrap-samples`/`--bootstrap-seed` (#2) | 16 |
| `_add_season_range_args` | plain `--start-season`/`--end-season` pairs (#6) | 8 |
| `_add_season_week_args` | required or prospective-default `--season/--week` | 7 |
| `_add_regressor_args` | `--regressor`/`--ridge-alpha` (with/without choices) | 9 |
| `_add_feature_profile_arg` | `--feature-profile` over `MARGIN_FEATURE_PROFILES` (#4) | 10 |
| `_add_snapshot_args` | "(label) snapshot ID; defaults to latest" flags (#5) | 6 |
| `_add_include_postseason_arg` | ingest postseason flag | 4 |
| `_add_ewm_args` | EWM smoothing trio | 3 |
| `_add_board_destination_args` | duplicated board/site destination pair | 2 |
| `_add_player_feature_tuning_args` | 7 shared player-feature tuning flags | 3 |

Audit #3 (training/reg args) is partially absorbed: adjacent
`min-edge/min-train-games` and ridge pairs collapsed into the bundles above;
the ~13 remaining single-line `--min-train-games` statements are one-liners
where a helper swap saves zero lines, so they were left verbatim.

Kept inline deliberately: `refresh-picks --features` (unique `None` default),
`cfb-sensitivity-audit --seed` (not part of the bootstrap pair), all unique
help texts, and the weak-signals/rotation record blocks (long but not
copy-pasted).

## Behavior preservation — measured, not asserted

- Structural equivalence harness (`F:/tmp/parser_equiv.py`, run this session):
  walked both the HEAD parser and the new parser recursively over every action
  (option strings, dest, default reprs, required, nargs, choices, type,
  help strings, action class, subparser tree and order):
  **"PARSER TREES IDENTICAL"**.
- Parsed 33 representative command lines through both parsers and compared
  every namespace attribute (excluding `handler`, which necessarily differs
  across two module loads): **0 mismatches**.
- Full suite unchanged: 1855 passed / 5 skipped before commit-ready state,
  same counts as HEAD's expected baseline for this checkout.

One full-suite run mid-session showed `test_cli_model_workflow` failing;
**measured** cause was a stale Windows basetemp directory (FileExistsError on
`nflats-swarm-basetemp`) left by the previous run, not the refactor — two runs
with clean basetemps pass completely.

## Provenance

- **measured**: all line counts, gate outputs, parser-equivalence results and
  test counts in this report (commands shown above).
- **read**: `scripts/swarm/tasks/hyg-cli.md`; the audit report via
  `git show 7e9d762:reports/wave1/hyg-cli.md` (the file lives on
  `swarm/hyg-cli`, not in this worktree).
- **inferred**: that "phase 1" means audit §2 families #1–#6 (the task text
  names parser consolidation); the residual gap to the 350-line target is
  estimator inflation in the audit, not un-applied consolidation — I think
  remaining single-flag statements have no mechanical shrinkage left.

## Not done (deferred)

- Audit items #7–#15 (challenger-loop extraction, evaluation/feature-build
  boilerplate, metadata/print dedup) — later phases per the wave plan.
- A declarative command-registration table for whole commands: evaluated and
  rejected this phase — remaining non-uniform commands would compress ~40–60
  lines at real regression risk; helpers already give the one-line-per-family
  registration shape the task asked for.

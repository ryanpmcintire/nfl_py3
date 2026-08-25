# ref-player-clones — unify `best_pick_nomination.py` function clones

Branch: `swarm/ref-player-clones` (base `cdc7c8b`). All numbers below are
**measured** this session via `git diff --stat`, `wc -l`, and the gate runs,
unless marked otherwise.

## What the finding was

The wave-1 audit task (`scripts/swarm/tasks/hyg-player-cluster.md`, **read**
this session) flagged `best_pick_nomination.py` (954 lines pre-refactor) for
near-complete function clones. The audit report itself
(`reports/wave1/hyg-player-cluster.md`) does not exist yet in this worktree
(**measured**: file absent), so this refactor executes the finding as stated
in the task file. Three clone pairs were found (**measured**, by reading the
pre-refactor file):

| Clone pair | Duplicated lines each |
|---|---|
| `select_nominee` / `select_nominee_v3` | ~25 |
| `nominate_v2` / `nominate_v3` | ~60 |
| `record_nomination_challenger_decisions` / `record_nomination_v3_challenger_decisions` | ~150 |

Together ~370 duplicated lines, matching the task's estimate.

## What changed

### 1. `src/nfl_ats/best_pick_nomination.py` (954 → 875 lines, −79)

- **New `_select_nominee(candidates, *, rule_name, dispersion_tiebreak)`**:
  one ranking/tie-break core; `select_nominee` delegates with the dispersion
  layer enabled, `select_nominee_v3` without it. Both public names,
  signatures, error-message substrings (`"at least one candidate"`), tie-break
  labels, and sort orders are byte-identical to before.
- **New `_nominate(...)`**: one orchestration body (REG-only gate → walk-forward
  fit → missing-candidate raise → dispersion join with `one_to_one` validation →
  empty-pool raise → `select_fn`); `nominate_v2` / `nominate_v3` become thin
  wrappers that only choose `select_fn` and build their result dataclass.
- **New `_record_nomination_for_challenger(...)`**: one recording body
  (registration check → active-model/forecast validation → fingerprint pinning →
  card contract checks → `nominate_fn` refit → whole-week-pre-kickoff guards →
  single-row ledger append → result summary); both public `record_*` functions
  become thin wrappers parameterised on `challenger_id`, `nominate_fn`, and
  `tie_note_fn`.
- Docstrings that claimed the duplication was *deliberate* ("Deliberately
  duplicates ... rather than refactoring") were rewritten to describe the shared
  implementation; every behavioral guarantee they documented (side-ledger-only
  v3, untouched `NOMINATION_V2_ENABLED`, identical pool/fitting) is restated in
  place. No public name, signature, or return shape changed.

Net: −70 lines of module code while adding three documented shared helpers.

### 2. `src/nfl_ats/players.py` (+16/−7) — scattered constants

Hoisted the inline defense-disruption weights (six magic multipliers buried in a
~100-line feature loop at former line ~1400) into a module-level named constant
`_DEFENSE_DISRUPTION_WEIGHTS`. Same arithmetic order, same floats, same result.

`pick_refresh.py`: inspected for scattered constants this session and left
untouched — its tunables are already centralised as named module constants
(`MOVEMENT_POLICY_THRESHOLD = 1.0`, `SUNDAY_PICK_LOCK_LOCAL_TIME`,
`LATE_WEEK_REFRESH_START/END`, **read** lines 112–182, 992–993), so there was no
trivial fix to make.

## Verification

Characterization coverage already existed (**read**
`tests/test_best_pick_nomination.py`, 1,039 lines covering both variants'
selection rules, gates, empty-card/missing-candidate raises, lock-window /
fingerprint / inactive-registration refusals, and a v2+v3 coexistence ledger
test), so no new tests were required by the task's "add first if thin" clause;
the existing suite served as the characterization contract. No test was
weakened or edited.

Full gates, all run this session against the worktree:

```
ruff format --check .   -> clean (636 files)
ruff check .            -> All checks passed!
mypy src                -> Success: no issues found in 105 source files
pytest -q               -> 1855 passed, 5 skipped, 49 warnings (5 skips are
                           pre-existing data-absent environment skips)
```

One full-suite run initially reported 1 failed + 1 error
(`test_public_board.py::test_no_observatory_references_remain_in_generated_pages`,
`test_forecast_weather_kn_warm_team_cold_late_tilt_overlay.py::...`); both pass
on base AND on this branch when re-run (**measured**) — the failures were
Windows basetemp file-lock contamination in
`C:/Users/Ryan/AppData/Local/Temp/nflats-swarm-basetemp` from a prior crashed
run's open parquet handle, not code behavior. Clean full-suite rerun with a
fresh basetemp: 1855 passed.

## Risk

Low-to-moderate, concentrated entirely in `best_pick_nomination.py`. The
refactor is behavior-preserving by construction (parameterisation, not logic
change) and pinned by the existing dual-variant test suite; the riskiest
substitution is `_select_nominee`'s branch on `dispersion_tiebreak`, which
reproduces the two original sort orders exactly (`["spread_std","game_id"]`
with `na_position="last"` vs `"game_id"` alone). Production surfaces touched
only through unchanged public names: `publishing.py` (`nominate_v2`),
`public_board.py` (`nominate_v3`), `cli.py` (both `record_*` functions).
No experiment window was opened or scored (binding rule 3): this is a pure
refactor verified by existing tests, with zero scoring looks spent.

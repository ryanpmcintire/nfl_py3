# ref-dead-code — Dead Export & Dependency Removal (Wave 1)

**Branch**: `swarm/ref-dead-code` · **Base**: `cdc7c8b` · **Date**: 2026-08-24

## Summary

Re-verified the auditor's list from `swarm/hyg-dead-code:reports/wave1/hyg-dead-code.md`
(**read** via `git show`) this session, then swept independently. Result:
**all three of the auditor's dependency findings are false positives — none can
be dropped** — but an independent AST sweep found **5 genuinely dead functions**
the auditor's "no dead exports" conclusion missed. All 5 removed; full gates pass.

## Dependency verdicts — auditor's list REFUTED in full

The auditor reported `pyarrow`, `scikit-learn`, `tabulate` as "0 imports anywhere".
Re-verified this session (**measured**, grep across `src/ tests/ scripts/` plus
`importlib.metadata` on the locked env):

| Dependency | Auditor claim | This session | Verdict |
|---|---|---|---|
| `scikit-learn` | 0 imports | Directly imported in **13 src modules** (`backtest.py:11`, `calibration.py:12-13`, `cli.py:447`, `clv.py:52`, and 9 more) + tests + scripts | **KEEP** |
| `tabulate` | 0 imports | Never imported directly, BUT is pandas' required backend for `DataFrame.to_markdown()` (ImportError without it), called at `cli.py:3934`, `lines.py:215`, `pick_refresh.py:1021`, `pool.py:79,149`. Verified via importlib.metadata that pandas declares tabulate only as an **optional extra** (`output-formatting`) — it does NOT come transitively | **KEEP** |
| `pyarrow` | 0 imports | Never imported directly, BUT is the engine for `pd.read_parquet`/`to_parquet`, used in **30 src modules** (e.g. `io.py`, `snapshots.py`, all data ingestion). Also a pandas optional extra only (`parquet`) — not transitive | **KEEP** |

The direct-import-only methodology used by the audit misses indirect runtime
requirements. No change to `pyproject.toml` dependencies.

## Dead exports found and removed (5)

Independent sweep: AST-collected every public function/class in `src/nfl_ats`
(~123 had zero external references), then filtered to symbols with zero
references ANYWHERE — including inside their own module beyond the `def` line.
Each survivor was individually grep-verified across **every file type in the
repo** (not just `.py`: docs, toml, scripts, tests included), which also rules
out string-based/dynamic dispatch. Each appeared exactly once repo-wide: its
own definition.

1. `dashboard.findings_content.group_for` (was line 1798) — sole in-repo occurrence.
   Its dependencies `GROUPS`/`VerdictGroup` are ALIVE (imported by
   `public_board.py:92,110`; tested at `tests/test_public_board.py:451`) and stay.
2. `dashboard.viz.family_comparison_bars` (was 413)
3. `dashboard.viz.contribution_bars` (was 465) — `empty_state` keeps its two
   other callers at viz.py:191,367.
4. `decision_rule.leave_one_out_log_likelihood` (was 310) — `fit_empirical_prior`,
   `math`, `DecisionRuleError` all retain other callers.
5. `weak_signals.signals_from_iterable` (was 664) — removal orphaned the
   `Iterable` import; trimmed to `from collections.abc import Sequence`.
   `signal_from_payload`/`_require` keep other callers.

Net diff: 1 insertion(+), 178 deletion(-) across 4 files. `pyproject.toml` untouched.

## Near-misses deliberately kept (examples, not exhaustive)

~118 of the ~123 zero-external-reference symbols are dataclasses/results or
internally-used helpers (e.g. `estimation_variance.VarianceDecomposition`,
`experiment_runner.ExperimentRunOutcome`) that their own module constructs and
returns — alive code, not dead exports. Removing them would break their modules.
Also kept: `handoff.render_handoff`/`inspect_handoff`-family names referenced by
docs prose conventions even where no code calls them — flagged for a future,
more aggressive pass if desired, but not provably dead under the task criteria.

## Gates (**measured** this session)

```
ruff format --check .   → 636 files already formatted ✓
ruff check .            → All checks passed ✓
mypy src                → Success: no issues found in 105 source files ✓
pytest                  → 1855 passed, 5 skipped, 0 failed (~124 s) ✓
```

Note: one earlier full-suite run showed 2 failures
(`test_cli_model_workflow`, `test_cli_rotation_workflow`, both `SystemExit: 2`)
that reproduced neither on the clean tree nor with my changes under a fresh
basetemp. Cause (**inferred**, consistent with evidence): basetemp collision —
the task spec directs every swarm worker to the SAME `--basetemp` path, and
workers run concurrently. Full suite passes clean with a worker-unique basetemp.

## Provenance

All claims above are **measured** this session unless marked otherwise;
commands available in session transcript. No experiment windows were opened
(Binding rule 3): this task is static analysis + deletion only, no scoring runs.
Binding rule 1 not implicated (no signals evaluated, nothing closed).

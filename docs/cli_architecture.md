# CLI architecture (ENG-10)

`src/nfl_ats/cli.py` was a single ~6,500-line module holding every
`add_parser` block, every `_cmd_*` handler and every library import the CLI
touches. ENG-10 split it by domain with **no behaviour change**: the same
commands, flags, defaults, help text, handler semantics, exit codes and stdout.

## Package layout

| Module | Owns |
| --- | --- |
| `nfl_ats/cli.py` | the parser skeleton, `main()`'s dispatch, and back-compat re-exports. ~85 lines. |
| `nfl_ats/cli_common.py` | helpers used by more than one domain: `_data_root`/`_artifacts_root`/`_registry_root`, `_print_json`, `_load_features`, `_season_range`, `_repo_root_on_path`, the `_resolve_*_snapshot` family, and every reusable `_add_*_args` argument group. |
| `nfl_ats/cli_commands/__init__.py` | the ordered `REGISTRARS` tuple. |
| `nfl_ats/cli_commands/operations.py` | `doctor`, `preflight`, `handoff`, `weekly-run` |
| `nfl_ats/cli_commands/data.py` | `ingest-player-arrests`, `ingest`, `smoke-source`, `pbp-ingest`, `depth-ingest`, `depth-history-ingest`, `player-ingest`, `player-value-ingest`, `participation-ingest`, `role-actions-fetch` |
| `nfl_ats/cli_commands/publishing.py` | `publish-predictions`, `refresh-picks`, `publish-board`, and `PUBLISH_CHALLENGER_RESULT_KEYS` |
| `nfl_ats/cli_commands/cfb.py` | the nine `cfb-*` commands |
| `nfl_ats/cli_commands/market.py` | `odds-ingest`, `odds-summary`, `odds-backfill`, `market-backfill`, `market-open-close-backfill` |
| `nfl_ats/cli_commands/pool.py` | `tiebreaker`, `pool-observables`, `totals-backtest` |
| `nfl_ats/cli_commands/clv.py` | `clv-score`, `clv-ledger`, `drift-report`, `clv-pilot`, `clv-sign-test`, `opener-evaluation`, `predict-close` |
| `nfl_ats/cli_commands/prospective.py` | `prospective-record`, `prospective-score` |
| `nfl_ats/cli_commands/features.py` | the six `build-*-features` commands |
| `nfl_ats/cli_commands/evaluation.py` | `backtest`, `nested-evaluate`, `dependence-audit`, `experiment`, `margin-backtest`, the three ablations, `player-model-selection`, `anytime` |
| `nfl_ats/cli_commands/prediction.py` | `margin-predict`, `market-decomposition`, `pool-card-at-lines`, `key-number-calibration`, `predict` |
| `nfl_ats/cli_commands/registry.py` | `rotation`, `weak-signals` |

A helper used by exactly one domain lives in that domain's module. A helper
used by two or more lives in `cli_common.py`; it is never duplicated.
`cli_common` imports neither `cli` nor `cli_commands`, so there is no cycle.

## The registration-order rule

`build_parser()` is:

```python
subparsers = parser.add_subparsers(dest="command", required=True)
for register in REGISTRARS:
    register(subparsers, current_year)
```

**`REGISTRARS` order is the `nfl-ats --help` listing order**, so it is part of
the CLI contract, not an implementation detail. `tests/test_cli_contract.py`
pins it against `tests/fixtures/cli_contract.json`.

Two consequences:

- **A domain whose commands are not contiguous in the historical help order
  exposes one registrar per contiguous run**, rather than being reordered.
  `market` has `register_odds` and `register_backfill` because `pool`'s three
  commands sit between them; `clv` has `register_scoring` and
  `register_diagnostics`; `operations` has `register_health`,
  `register_handoff` and `register_weekly`. Preserving the listing was worth
  more than one-registrar-per-module tidiness.
- **Every registrar has the same signature**,
  `(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], current_year: int) -> None`.
  `current_year` is threaded through instead of each registrar calling
  `datetime.now()` so a single `build_parser()` call can never mix two calendar
  years across its clock-derived defaults (`--end-season` defaults and friends).

## The Request / Result split

The four public workflows — `weekly-run`, `publish-predictions`,
`margin-predict` and `predict` — are split into three named layers:

| Layer | Shape | Rule |
| --- | --- | --- |
| argument validation | `parse_<workflow>_request(args) -> <Workflow>Request` | Pure. Reads only `args`, touches no filesystem and no environment, returns a frozen dataclass. Raises exactly what reading a missing/ill-typed namespace attribute raised before. |
| orchestration | `orchestrate_<workflow>(request) -> <Result>` | Calls the existing library functions (`run_weekly`, `publish_active_predictions`, `score_outcome_week`, …). Contains the moved handler body verbatim, with `args.x` rewritten to `request.x`. |
| artifact writing / output | the `_cmd_*` handler | Three lines: parse, orchestrate, write. `_print_json(...)` with exactly the payload it printed before. |

```python
def _cmd_weekly_run(args: argparse.Namespace) -> None:
    result = orchestrate_weekly_run(parse_weekly_run_request(args))
    _print_json(result)
```

`weekly-run` and `publish-predictions` return `dict[str, Any]` (the JSON
document). `margin-predict` and `predict` return `PredictionArtifacts`
(`metadata` plus the `output` directory), because those two commands interleave
their artifact writes with metadata construction; splitting the writes out
would have restructured the body rather than moved it.

There is no `fit`/`train` command in this CLI; `predict` and `margin-predict`
are the commands that fit a model, so they are the ones split.

**Parse and orchestrate stay in the same module as their handler.** That is
deliberate: the library functions a handler calls are module globals of that
module, and tests monkeypatch them there. Moving orchestration into a separate
module would silently break every such patch.

## Monkeypatching, after the split

`from x import y` binds a *new name* in the importing module. So patching
`nfl_ats.cli.publish_active_predictions` no longer affects the handler — the
handler reads `nfl_ats.cli_commands.publishing.publish_active_predictions`.

**Patch the module that owns the handler under test.** `nfl_ats.cli`
deliberately does not re-export the library functions, so a stale patch target
raises `AttributeError` at `monkeypatch.setattr` instead of silently letting
the real function run. The handful of names `nfl_ats.cli` *does* still
re-export (for `scripts/` and older tests) are listed in its `__all__`.

## How to add a command

1. Pick the domain module (or add one, if the command genuinely starts a new
   domain).
2. Write `_cmd_<name>(args: argparse.Namespace) -> None` there. Import the
   library functions it needs at module level, and shared helpers from
   `nfl_ats.cli_common`.
3. Add the `add_parser` block to that module's registrar — at the END of the
   registrar if you want the command at the end of the help listing, since
   position in the registrar is position in `--help`.
4. If the command belongs at a position that splits an existing registrar,
   add a NEW registrar function and insert it in `REGISTRARS` at that point.
   Do not reorder existing entries.
5. Regenerate the contract fixture and read the diff before committing it:

   ```powershell
   .\.tools\uv.exe run --no-sync python scripts\cli_contract_snapshot.py `
       tests\fixtures\cli_contract.json --normalize-years
   ```

   The diff should show only your new command. Anything else in it is drift.

## The contract oracle

`scripts/cli_contract_snapshot.py` walks the parser and every subparser and
dumps a normalised JSON document: command path, prog, description, help, and
for every argument its option strings, dest, default, type, choices, nargs,
`required` and help — plus the subcommand ordering and each handler's module
and qualname. `--normalize-years` replaces integer defaults equal to the
current or previous year with `<CURRENT_YEAR>` / `<CURRENT_YEAR-1>` tokens so
the tracked fixture does not rot on 1 January.

It is the tool to run before and after any CLI refactor: two snapshots that
compare equal describe argparse trees that behave identically from outside.
`tests/test_cli_contract.py` runs it in-process against the tracked fixture on
every test run.

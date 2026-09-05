# Deterministic environment lock report (ENG-21)

`nfl_ats.environment_report.environment_report()` is a single, self-contained
function that answers "what interpreter, packages, and platform actually
produced this number" -- the third leg alongside `artifact_provenance()`'s
existing CODE pin (git revision/dirty flag) and DATA pin (feature-table
sha256, `uv.lock` sha256). Without it, "does this reproduce" silently assumed
numpy/pandas/scikit-learn never changed underneath a re-run.

Implementation: `src/nfl_ats/environment_report.py` (pure, fail-safe
functions, no dependency on `nfl_ats.provenance` to avoid a circular import).
Standalone CLI: `scripts/env_report.py`. Tests: `tests/test_environment_report.py`.

## Why a standalone script instead of a `nfl-ats` subcommand

`src/nfl_ats/cli.py` is one very large file that was under concurrent edit by
other in-flight backlog items (ENG-01's `lockday_package.py`, ENG-02's
`preflight.py`, both of which registered their own subcommands there) while
this module was being written, and CLI registration needs three separate edit
locations (an import, a subparser block, a handler function). This command is
read-only and has no need to live on the console-script surface to be useful,
so `python scripts/env_report.py` ships instead of `nfl-ats env-report` --
avoiding collision risk in the shared file for a command that does not
require touching it.

## Usage

```powershell
.\.tools\uv.exe run --no-sync python scripts/env_report.py
.\.tools\uv.exe run --no-sync python scripts/env_report.py --json > env_a.json
.\.tools\uv.exe run --no-sync python scripts/env_report.py --compare env_a.json
```

`--compare <path.json>` prints the current environment, diffs it against a
prior report JSON file, and classifies every differing field as
`reproducibility_affecting` or `cosmetic` (see below). Per this repository's
binding research invariant, an environment difference is reporting context,
never itself grounds to reject a result -- the script never exits nonzero on
a diff.

## What is in the report

| section | contents |
|---|---|
| `python` | full version string, major/minor/micro, implementation (CPython), executable path |
| `uv` | `uv --version` output, resolved from `.tools/uv(.exe)` then `PATH`; tolerates absence |
| `platform` | OS, release, build/version string, machine (architecture), processor |
| `packages` | resolved versions (`importlib.metadata`) of `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyarrow`, `joblib`, `nflreadpy`, `pypdf`, `tabulate` -- pyproject.toml's direct dependencies plus `scipy` (a transitive scikit-learn/numpy dependency this backlog item names explicitly) |
| `blas` | numpy's BLAS/LAPACK backend name and version, from `numpy.show_config(mode="dicts")` |
| `thread_counts` | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS` |
| `git` | revision + dirty flag (same shape as `nfl_ats.provenance.git_state`) |
| `uv_lock` | whether `uv.lock` exists and its sha256 |
| `environment_variables` | an ALLOW-LISTED dump: `PYTHONHASHSEED`, `TZ`, the `*_NUM_THREADS` family, and `NFL_ATS_*` |
| `secrets_detected` | presence-only booleans (never values) for `THE_ODDS_API_KEY`, `CFBD_API_KEY`, and any other env var whose NAME matches `KEY`/`TOKEN`/`SECRET`/`PASSWORD` |

`environment_report()` **never raises**: any exception anywhere in assembly
becomes `{"error": "<Type>: <message>"}` rather than propagating, so a broken
environment probe can never abort the run that called it. Most individual
fields already degrade on their own (missing `uv`, an uninstalled package, an
unavailable BLAS summary, a non-git working tree all produce `None`/`False`
sub-fields) -- the top-level catch is the last-resort net.

## Secrets: presence booleans only, values never recorded

Two mechanisms, so a secret cannot leak by entering through an unexpected
path:

1. The `environment_variables` allow-list is narrow by construction (see the
   table above) -- a secret-shaped name is never included in the first place
   unless it happens to start with `NFL_ATS_` (defensively redacted anyway,
   see below).
2. A final recursive redaction pass walks the **entire assembled dict**
   (`_redact` in `environment_report.py`) and collapses any surviving key
   whose name matches the secret pattern to a boolean. This is why an
   `NFL_ATS_API_KEY` override, or a test's `FAKE_API_KEY`, shows up as
   `"...": true` rather than its value: `secrets_detected` scans the whole of
   `os.environ` for the same pattern independent of the allow-list, so a
   credential's *presence* is always visible even though its *value* never
   is.

## Wiring into experiment / forecast / weekly-run metadata

Rather than edit the many separate call sites in `cli.py` and
`experiment_runner.py`, the report is wired into the ONE function both paths
already call: `nfl_ats.provenance.artifact_provenance()`. Every command that
builds `metadata["provenance"] = artifact_provenance(...)` -- which includes
`predict`, `margin-predict` (the command `weekly-run`'s card-path step
actually invokes), `predict-close`, every CFB variant, and
`nfl_ats.experiment_runner`'s experiment-registry writer -- now additionally
gets `metadata["provenance"]["environment"]` for free, computed once and
reusing the same `git_state()`/`uv.lock` hash `artifact_provenance()` already
computes rather than paying for them twice.

Two call sites also carry a **top-level convenience mirror** (a reference to
the same dict, not a recomputation) so the report is visible without digging
into `provenance`:

- `nfl_ats.experiment_runner` (`metadata["environment"] = metadata["provenance"]["environment"]`)
  -- every `registry/experiments/experiment-run/*.json` row.
- `nfl_ats.cli._cmd_margin_predict` (same pattern) -- the forecast metadata
  `weekly-run`'s `margin-predict` step writes, which is what
  `active_ats_model.json` links to and `publish-predictions` reads.

`nfl_ats.lockday_package.build_manifest` (ENG-01) additionally records
`manifest["environment"]` directly via `environment_report()`, through the
same fail-safe `_collect()` wrapper every other manifest section uses, so a
lock-day decision package always carries its own environment snapshot next to
its code/model-identity/inputs/outputs sections.

Both `provenance.py` and `lockday_package.py` edits are single additive
lines/blocks; neither file was reformatted or restructured.

## `compare_environment(a, b)`: reproducibility-affecting vs. cosmetic

A field is **reproducibility-affecting** when a difference in it CAN change
which code path runs or what numbers a run produces: interpreter minor
version, resolved package versions, the `uv.lock` hash (a different lock can
resolve different transitive versions even with the same direct pins),
BLAS/LAPACK backend identity, thread-count env vars (BLAS/sklearn parallelism
can change floating-point summation order), `PYTHONHASHSEED`, and
`platform.machine` (instruction-set-dependent code paths). A field is
**cosmetic** when it varies run-to-run or box-to-box without touching numeric
behavior: `uv`'s own patch version (the resolver, not a runtime dependency --
what matters is the separately-tracked `uv.lock` hash), the OS build/release
string, the git *dirty* boolean (it says "something is uncommitted", not
*what* -- `git.revision` itself, by contrast, is reproducibility-affecting),
timestamps, and hostnames/executable paths.

Unmatched or newly introduced fields default to **reproducibility-affecting**
rather than cosmetic: a false "this might matter" costs an extra look at a
comparison; a false "this is definitely fine" can hide a real
non-reproduction, which is the more expensive mistake for a project whose
whole premise is trusting a small measured number.

`compare_environment()` is a reporting tool, not a gate: per this
repository's binding research invariant, an environment difference is never
itself grounds to reject a result -- it is context for interpreting one.

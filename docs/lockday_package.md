# The immutable lock-day decision package (ENG-01)

Written 2026-09-04, four days before Week 1 locks.

## The problem

Week 1 2026 locks on Tuesday 2026-09-08 as one command:

```powershell
.\.tools\uv.exe run --no-sync nfl-ats weekly-run --season 2026 --week 1 --record-decisions
```

That one run downloads an nflverse snapshot, rebuilds four feature tables,
re-fits the walk-forward evaluation, scores the week, asserts the card and the
evaluation are the same model, publishes the tracked card and the public site,
and appends rows to up to seven append-only ledgers.

Afterwards, "what exactly did we decide, from what inputs, with which model" is
spread across `data/processed/*.manifest.json`, `artifacts/active_ats_model.json`,
a timestamped `artifacts/margin_predictions/` directory, a tracked Markdown
card, roughly twenty nested JSON recorder keys inside one stdout blob, and the
parquet ledgers themselves. `artifacts/` is gitignored and local-disk-only, and
`docs/closure_audit.md` records it disappearing twice in one session.

Prospective evidence is worth exactly as much as the proof that the picks
existed before the games did (`docs/prospective_evidence.md`). The package is
that proof, collected once, at the moment it is still all true.

## What gets written

`--record-decisions` now writes, as the run's **last** step:

```
artifacts/lockday_packages/<season>_wk<week>_<UTC stamp>/
    manifest.json      the package. Plain JSON, written read-only.
    manifest.sha256    SHA-256 of manifest.json, so an edit to it is detectable.
    README.md          how to read and verify it without this repository.
```

`--no-package` opts out. Nothing else about the command changes.

### Manifest sections

| key | what it pins |
|---|---|
| `code` | git revision, dirty flag, SHA-256 of `git diff HEAD`, SHA-256 of `uv.lock` |
| `model_identity` | `model_id`, method, target, feature profile, regressor, `ridge_alpha`, calibration method, probability method, status, `feature_table_sha256`, `evaluation_configuration_sha256`, and the SHA-256 of `active_ats_model.json` itself |
| `inputs.feature_tables` | every `.parquet` path an executed step named on its command line, each with SHA-256 and byte size |
| `inputs.snapshot_manifests` | each table's sibling `<stem>.manifest.json` (hashed AND embedded, so the source snapshot ids survive), plus every other `data/processed/*.manifest.json` |
| `inputs.snapshot_ids` | the `--snapshot` / `--player-snapshot` / `--player-value-snapshot` / `--pbp-snapshot` values the run actually passed |
| `outputs.forecast` | the active manifest's linked weekly-forecast directory, file by file |
| `outputs.historical_evaluation` | the linked evaluation directory, file by file |
| `outputs.cards` | `CURRENT_PREDICTIONS.md` and `docs/index.html` |
| `recorders.steps` | each `weekly-run` step's output JSON **verbatim**, with status, seconds and error |
| `recorders.by_challenger_id` | a flat index of every nested object carrying a `challenger_id` |
| `ledgers` | per ledger: path, rows before, rows after, file digest before and after, and a SHA-256 of the rows THIS run appended |
| `lockday_verify` | `scripts/lockday_verify.py`'s report for this season/week, its rendered text, and its exit code |
| `run_summary` | the whole `weekly-run` JSON summary, verbatim |
| `hashed_files` | one flat list of every hashed file — the verifier's target |
| `errors` | components that failed while assembling. See below. |

The recorder JSON is stored verbatim because the recorders are deliberately
fail-open (`{"recorded": 0, "error": ...}`, see `scripts/lockday_verify.py`), so
the JSON returned at the time is the only durable evidence that a challenger
skipped for a documented gate reason rather than breaking silently. Zero rows
looks identical either way once the run's stdout is gone.

### Why the feature tables are read off the commands

`weekly-run`'s plan passes the card's feature table as `--features` and the
learned-availability build's outputs as `--destination`/`--rates-destination`,
and that set has already changed once: the 2026-08-18 promotion moved the card
path from `player` to `weak_stack`. The package therefore takes every
parquet-shaped token off the commands that **actually ran**, rather than an
enumerated flag list that can go stale without anyone noticing.

## Fail-safe, by contract

By the time the package is assembled, the ledger rows are already appended and
the card is already published. So:

* every component is collected behind a guard that records the failure in
  `errors` and keeps going;
* the writer itself never raises — a catastrophic failure prints to stderr and
  returns `{"written": false, ...}`;
* the call site in `_cmd_weekly_run` is in a `finally`, so a `weekly-run` that
  aborts at a fatal step still gets whatever package can be built.

**A package with a non-empty `errors` list is the designed output of a
partially-broken run, not a failed lock.** Nothing here may ever abort or roll
back a lock that already happened.

## Reading it

```powershell
# human summary
.\.tools\uv.exe run --no-sync python scripts/lockday_package_verify.py `
    artifacts/lockday_packages/2026_wk01_<stamp> --summary
```

In Python:

```python
from pathlib import Path
from nfl_ats.lockday_package import load_package, summarise_package

manifest = load_package(Path("artifacts/lockday_packages/2026_wk01_<stamp>"))
print(summarise_package(manifest))
```

`load_package` accepts either the folder or the `manifest.json` inside it, and
refuses any JSON file that is not a `lockday_decision_package`.

## Verifying it independently

```powershell
.\.tools\uv.exe run --no-sync python scripts/lockday_package_verify.py `
    artifacts/lockday_packages/2026_wk01_<stamp>
```

It recomputes `manifest.sha256` against `manifest.json`, then re-hashes every
entry in `hashed_files`. Exit code 0 only when the package verifies.

* **Ledger entries are flagged `mutable`.** The ledgers are append-only and
  later in-week refresh passes legitimately add rows, so a changed ledger is
  reported under `mutable_changed` and is never a failure.
* **Missing files are reported, not fatal** by default — `artifacts/` is
  gitignored and local-disk-only, so a cleaned artifact is expected. Pass
  `--strict` to require every file that *was* hashed at write time to still
  exist.
* **Entries with no digest are never fatal.** A ledger this lock never wrote,
  or a file over `MAX_HASHED_BYTES`, is listed under `unhashed`: the package
  claimed nothing about it, so there is nothing to check.
* **A package copied to another machine still verifies**: each entry carries a
  repo-relative path alongside the absolute one, and `--repo-root` selects what
  it resolves against.

Nothing in the manifest needs this repository to interpret. Every digest names
its algorithm (`sha256`) and the exact bytes hashed, and the appended-row digest
states its own recipe in `appended_rows_digest_method`:

```python
import hashlib

import pandas as pd

frame = pd.read_parquet(row["path"]).iloc[row["rows_before"] :]
digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
assert digest == row["appended_rows_sha256"]
```

## What this is NOT

**It is not tamper-proof.** `manifest.json` gets the read-only attribute
(best effort on Windows and POSIX) and `manifest.sha256` pins its bytes. Those
stop an accidental edit and make a deliberate one visible; they do not prevent
one. Anyone who can write the manifest can rewrite the digest beside it.

**It is not a substitute for `lockday_verify`.** The package *embeds* the
verifier's report; it does not replace running it. A `MISSING` challenger is
still a lock-day failure to fix before kickoff, and the package's job is only
to make sure that fact survives the week.

**It adjudicates nothing.** It records what was decided and from what. It is
not an experiment look, it scores no signal, and it never touches
`registry/`.

## Rehearsal

`scripts/lockday_rehearsal.py --full-replay` builds a package too, from its
isolated artifacts root, tagged `"rehearsal": true` and written to
`<rehearsal root>/../lockday_packages_rehearsal/` — a deliberately different
directory name from the real `artifacts/lockday_packages/`, so a rehearsal
package can never be mistaken for a real lock's. The README inside a rehearsal
package says so on its first line.

The fast default (`scripts/lockday_rehearsal.py`, the static wiring audit) does
not build a package: it runs no recorder and writes no ledger, so there is
nothing to package.

## Where the code lives

| file | role |
|---|---|
| `src/nfl_ats/lockday_package.py` | builds, writes, loads, summarises and verifies packages |
| `scripts/lockday_package_verify.py` | the standalone verifier / summary reader |
| `src/nfl_ats/cli.py` (`_cmd_weekly_run`) | captures the before-state, then writes the package as the last step |
| `scripts/lockday_rehearsal.py` | the `--full-replay` rehearsal hook |
| `tests/test_lockday_package.py` | synthetic-tree tests; never touches the real artifacts |


## Lock-day sequence 2026-09-08

Read (`src/nfl_ats/weekly.py`, `plan_weekly_run` / `run_weekly`; CX12,
2026-09-05): after `margin-predict` and `assert-synchronized`, the weekly
chain runs `opener-evaluation --features` against the same active-profile
table used for prediction, then `overlay-composition`, before
`publish-predictions --with-board --record-decisions`. Both measurement
steps are skipped only when the manifest model id before and after prediction
is identical and a matching opener evaluation (including its per-game file)
and matching composition already exist. A changed model always recomputes
both. The last planned step is a required `publish-board`, after prospective
recording/scoring and optional drift telemetry; its failure aborts the run.
The existing lock-day package is collected afterwards by the CLI wrapper.

Measured (read from `artifacts/opener_evaluation/20260905T194919Z/metadata.json`
and `artifacts/overlay_subset_composition/20260905T200533Z/result.json`):
the evaluation records 26.91 seconds, and the composition completion timestamp
is 973.77 seconds after the evaluation's creation timestamp, a combined
16.68-minute recorded span. The composition has no duration field, so that
span includes any inter-command idle time and is not an isolated runtime.
New composition artifacts record `timing.total_seconds` for future budgeting.

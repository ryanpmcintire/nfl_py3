# Reproducible run replay (ENG-13)

A read-only command that answers "does this recorded run still reproduce" --
without refetching anything, and without rewriting a production artifact.

## What it consumes

`scripts/replay_run.py <manifest>` accepts either manifest shape this
repository already writes:

* an **ENG-01 lock-day package** `manifest.json`
  (`nfl_ats.lockday_package`, `kind == "lockday_decision_package"`); or
* a bare **forecast artifact's `metadata.json`** -- the file `margin-predict`
  writes beside `predictions.csv`, identified by carrying a `provenance`
  block (`nfl_ats.provenance.artifact_provenance`).

You can point it at the manifest file directly, or at its containing folder
(it looks for `manifest.json` then `metadata.json`).

## What it checks

| step | what | source of truth |
|---|---|---|
| 1. digests | every source/feature-table/forecast/card digest the manifest actually recorded, recomputed from disk | reuses `nfl_ats.lockday_package.verify_package` for a lock-day package (byte-identical to `scripts/lockday_package_verify.py`); reuses `hash_entry` for a bare forecast `metadata.json`, which only ever records `provenance.feature_table` and `provenance.uv_lock_sha256` |
| 2. environment | the manifest's recorded environment block vs. `environment_report()` run now | `nfl_ats.environment_report.compare_environment`, which classifies every differing field `reproducibility_affecting` or `cosmetic` |
| 3. code revision | the manifest's recorded git revision/dirty flag vs. `git rev-parse HEAD` now | `nfl_ats.provenance.git_state` |
| 4. recompute (default on) | the forecast regenerated in-process for the manifest's own season/week, compared column-by-column against the recorded `predictions.csv`, plus a small derivable metadata subset (`games`/`methods`/`game_type`) | `nfl_ats.outcomes.score_outcome_week`, driven by the `configuration` the manifest itself recorded |

Step 4 only runs when the digest that actually matters for it -- the feature
table -- verifies. A drifted `uv.lock` digest (the common case for any
manifest older than the newest `uv sync`) does **not** block recompute on its
own: it pins the dependency *environment*, not the model's *input*. It still
keeps the overall verdict `FAIL` (see below) and is reported plainly.

## Usage

```powershell
.\.tools\uv.exe run --no-sync python scripts/replay_run.py <manifest>
.\.tools\uv.exe run --no-sync python scripts/replay_run.py <manifest> --no-recompute
.\.tools\uv.exe run --no-sync python scripts/replay_run.py <manifest> --json `
    --output-root $env:TEMP\eng13_replay
```

* `--no-recompute` stops after digests/environment/revision -- useful when
  you only want to know "has anything on disk changed", not "does the model
  still produce the same numbers".
* `--output-root` is where recompute writes `regenerated_predictions.csv` and
  `regenerated_metadata.json`. It is **never** a production artifact
  directory; omit it and a fresh OS temp directory is used.
* Exit code is **0 only when**: every recorded digest verifies, no
  reproducibility-affecting environment difference exists, and (if recompute
  ran) the regenerated outputs match. A git revision mismatch is reported but
  never gates the exit code -- replaying an old run from a newer checkout is
  expected use, not an error.

Per this repository's binding research invariant, this command **reports**
differences; it never adjudicates them. Nothing here writes to
`registry/weak_signals.json` or any rotation ledger.

## Guarantees

* **Never fetches.** Every read is local disk plus a `git` subprocess call.
  No network I/O anywhere in `nfl_ats.run_replay`.
* **Never writes outside `--output-root`.** The only writes this command ever
  performs are the two regenerated files above, and only when recompute
  actually runs.
* **Never touches ledgers.** Recompute calls `score_outcome_week` directly --
  the pure scoring function `margin-predict` itself calls -- never
  `weekly-run`, never a recorder, never `--record-decisions`.

## Measured result: replaying the live Week 1 2026 forecast (2026-09-04)

Run against `artifacts/margin_predictions/2026-week-01-20260903T143253Z/metadata.json`,
a real `margin-predict` artifact from 2026-09-03 (before Week 1 locks on
2026-09-08):

**`--no-recompute`:**

```
run replay: FAIL
  digests               : FAIL  1 verified of 2 checked
    changed: 1
      uv_lock  F:\Repos\nfl_py3\uv.lock
  git revision          : MISMATCH  recorded=48034cbd9d35 current=4079db194b27
  environment           : not recorded in this manifest
```

**With recompute**, into `$env:TEMP\eng13_replay`:

```
recompute             : match  predictions match, metadata match
```

All 264 predictions-table columns across all 80 rows (16 games x 5 methods)
compared **exactly equal** (`max_abs_diff: 0.0` on every numeric column,
text columns byte-identical), and the derived metadata subset
(`games=16`, `methods`, `game_type="REG"`) matched too.

Reading the pieces together: the model's actual input (the feature table)
still hashes to what the manifest recorded, and regenerating the forecast
from that exact table -- through today's code, a different `HEAD` than the
one the manifest recorded -- reproduces the recorded predictions bit for
bit. The overall verdict is still `FAIL`, correctly: `uv.lock` has changed
since 2026-09-03 (this manifest also predates `_cmd_margin_predict`'s
`metadata["environment"]` mirror -- the artifact's `provenance.code.dirty`
was `true` at write time, so it was not produced by a clean, fully-current
tree even then). That `uv.lock` drift is a real, reportable environment
difference; it is not evidence the forecast itself fails to reproduce, and
`replay_run.py` reports the two facts separately rather than collapsing them
into one verdict.

No file under `artifacts/` was read for writing, and nothing outside
`$env:TEMP\eng13_replay` was written.

## Where the code lives

| file | role |
|---|---|
| `src/nfl_ats/run_replay.py` | `replay_manifest()`: digest verification, environment/revision comparison, recompute |
| `scripts/replay_run.py` | the CLI wrapper (`--no-recompute`, `--json`, `--output-root`) |
| `tests/test_run_replay.py` | synthetic-manifest tests: digest match/mismatch, cosmetic vs. reproducibility-affecting environment diffs, git revision mismatch, recompute match and injected drift |

## What this is NOT

**It is not a promotion or rejection gate.** A `FAIL` verdict here means "one
or more recorded facts no longer match disk", not "the underlying result is
wrong". Per this repository's binding invariant, differences are reported
for a human to interpret, never auto-adjudicated.

**It does not replay the full `margin-predict` pipeline.** Recompute calls
`score_outcome_week` only -- not `validate_outcome_prediction_card`'s full
safety audit, not `build_card_lineage`, not active-model linking -- because
those either touch registries/ledgers (out of scope for a read-only replay)
or require state (the active model manifest) replay is not given. The
metadata comparison is correspondingly narrow: only the
mechanically-derivable subset (`games`/`methods`/`game_type`), never
timestamps, `provenance`, or `environment`.

**It is not a substitute for `scripts/lockday_package_verify.py`.** For a
lock-day package, replay reuses that verifier rather than reimplementing it;
run it directly if all you need is package integrity, not a forecast
recompute.

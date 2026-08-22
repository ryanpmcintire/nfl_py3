# Batch recording: serializing weak-signal registration

`registry/weak_signals.json` is a single shared file. When many worker agents
record signals at once (mass screening waves, battery follow-ups), concurrent
direct `nfl-ats weak-signals record` calls race on that write. The fleet
convention is a queue plus one serializer:

- **Worker agents NEVER call `weak-signals record` directly during mass
  waves.** They enqueue a validated task file instead.
- **One recorder agent drains the queue**, holding an OS-level byte-range
  lock so two drains serialize rather than interleave registry writes.
- **Dry-run is the default.** Nothing touches the registry until a human or
  the recorder agent passes `--execute`.

## Queue location and naming

```
<temp>/nfl_ats_record_queue/<priority>-<timestamp>-<nonce>.json
```

- `<priority>`: zero-padded two-digit integer, lower drains first (default 5).
- `<timestamp>`: UTC enqueue time to microseconds (`YYYYmmddTHHMMSSffffffZ`),
  so filename order is priority-major, then FIFO within a priority.
- `<nonce>`: 8 hex characters, prevents collisions.

The default queue directory can be overridden per command with `--queue-dir`
or globally with the `NFL_ATS_RECORD_QUEUE` environment variable.

## Task file format

Each queue file holds one JSON object whose keys are the snake_case forms of
the real `nfl-ats weak-signals record` flags:

| Field | Required | Type / allowed values |
|---|---|---|
| `name`, `description`, `source` | yes | string |
| `effect` | yes | number |
| `effect_units` | yes | `ats_points`, `accuracy_points`, `brier`, `log_loss`, `mae` |
| `classification` | yes | `unresolved_below_power`, `refuted_mechanism`, `bounded_by_control` |
| `league` | yes | `nfl`, `cfb` |
| `season_start`, `season_end` | yes | integers, start <= end |
| `interval_low`, `interval_high` | yes | numbers, low <= high |
| `probability_positive` | yes | number in [0, 1] |
| `sample_games`, `sample_blocks` | yes | integers |
| `standard_error`, `reliability` | no | number |
| `classification_evidence`, `recorded_at`, `notes` | no | string |
| `closing_ground` | no | `wrong_sign_resolved`, `no_split_half_reliability`, `positive_control_bound` |
| `replace` | no | boolean |

Enqueue validates against these actual CLI flags and rejects unknown fields,
missing required fields, bad enums, and policy violations (a terminal
classification requires an admissible `closing_ground`;
`wrong_sign_resolved` requires the whole interval below zero) with actionable
errors before anything is dropped into the queue.

## Exact worker snippet

```powershell
# From F:\Repos\nfl_py3 -- build the arguments object, then enqueue. Do NOT run
# nfl-ats weak-signals record yourself during mass waves.

$task = @'
{
  "name": "my_family_cell_01",
  "description": "one-line description of the measured cell",
  "source": "docs/my_family.md",
  "effect": -0.12,
  "effect_units": "accuracy_points",
  "classification": "unresolved_below_power",
  "league": "nfl",
  "season_start": 2009,
  "season_end": 2025,
  "interval_low": -0.55,
  "interval_high": 0.31,
  "probability_positive": 0.62,
  "sample_games": 1200,
  "sample_blocks": 17,
  "reliability": 0.41,
  "notes": "week-blocked primary interval"
}
'@
$taskPath = Join-Path $env:TEMP "my_signal.json"
Set-Content -LiteralPath $taskPath -Value $task -Encoding UTF8

# Validate + drop into the queue; prints the queue id on success.
.\.tools\uv.exe run python scripts/batch_record.py enqueue --file $taskPath
```

To enqueue many tasks at once, pass a JSON array as the payload; every object
is validated before any file is written (all-or-nothing).

## Recorder commands

```powershell
# What would run, in order, without executing anything:
.\.tools\uv.exe run python scripts/batch_record.py status
.\.tools\uv.exe run python scripts/batch_record.py drain

# Record for real (single recorder only; the lock makes concurrent drains wait):
.\.tools\uv.exe run python scripts/batch_record.py drain --execute
```

Drain behavior:

1. Acquires an exclusive lock on `<queue>/drain.lock` (blocking up to
   `--lock-timeout` seconds, default 60; on timeout it aborts loudly instead
   of racing).
2. Lists all queued files oldest-first **inside** the lock.
3. For each task: invokes `[python -m nfl_ats.cli weak-signals record ...]`
   from the repository root with captured output.
4. On success moves the task file to `<queue>/done/`; on any failure
   (non-zero exit or validation error) moves it to `<queue>/failed/` beside a
   `<queue-id>.stderr.txt` file holding the captured stderr.
5. Prints per-task lines and a final summary
   (`queued=... executed=... recorded=... failed=...`).

## Idempotency

Done and failed files leave the queue directory root, so re-running drain
processes nothing new and never double-records. Dry-runs move nothing, so a
dry-run followed by `--execute` records exactly once.

## Failure handling policy

A record command failing means the verdict or payload is wrong, not the
validator (repository policy). Inspect `<queue>/failed/<id>.stderr.txt`, fix
the task, and re-enqueue. Never bypass the recorder or hand-edit the
registry during a wave.

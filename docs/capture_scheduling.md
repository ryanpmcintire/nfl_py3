# Capture scheduling

How the recurring point-in-time captures get run, and why the mechanism is what
it is. Written 2026-08-25, when the schedule moved out of Windows Task
Scheduler and into this repository.

## What runs

`scripts/capture_scheduler.py` holds the whole schedule in `SCHEDULE`, 15 jobs:

| Job | When (ET) | Grace | Season-guarded |
|---|---|---|---|
| `odds_tue_open` | Tue 09:00 | 180m | no |
| `odds_thu_tnf` | Thu 18:00 | 90m | no |
| `odds_sat` | Sat 12:00 | 180m | no |
| `odds_sun_close` | Sun 12:30 | **25m** | no |
| `odds_sun_late` | Sun 16:15 | 60m | no |
| `odds_mon_mnf` | Mon 19:00 | 90m | no |
| `public_betting_sat` | Sat 12:00 | 240m | no |
| `public_betting_sun` | Sun 12:00 | 45m | no |
| `injuries_wed` / `injuries_thu` / `injuries_fri` | Wed/Thu/Fri 17:30 | 240m | yes |
| `injuries_sat` | Sat 10:00 | 240m | yes |
| `refresh_thu` | Thu 15:00 | 240m | yes |
| `refresh_sat` | Sat 10:30 | 300m | yes |
| `refresh_sun` | Sun 10:00 (`--publish-card`) | 300m | yes |

Times match the retired Task Scheduler entries exactly, so the migration
changed the mechanism and not the cadence.

## Why not Windows Task Scheduler

It held eight entries pointing at the same two `.ps1` files. Everything that
mattered — when each ran, what ran last, whether anything failed — lived in
opaque per-machine config outside version control, and a failed run was
silent. Ruled out by the owner.

## Why not GitHub Actions

Considered seriously; it fails on facts specific to this repo, both measured
2026-08-25:

1. **The repository is public** (`gh repo view` → `"visibility":"PUBLIC"`) and
   the odds feed is purchased. Workflow artifacts on a public repo are
   downloadable by anyone, so captured quotes cannot land there. MKT-09, the
   provider licensing/redistribution audit, is still open, which is a further
   reason not to publish that data on a guess.
2. **`odds-ingest` is not stateless.** It calls `_load_features`
   (`src/nfl_ats/cli.py`), which raises if `game_features.parquet` is missing —
   a fresh runner does not have it.

Scheduled Actions are also documented to be delayed under load, which is
awkward against `odds_sun_close`'s 30-minute margin before 13:00 kickoffs.

Neither is a permanent verdict. A private data repo or object storage would
resolve point 1, and a raw-only capture mode would resolve point 2.

## Windows, not instants

Each job declares a target time and a **grace** period; the loop polls once a
minute and runs a job if now is inside `[target, target + grace]` and that
occurrence has not already run. This is deliberately more forgiving than cron:
a machine asleep at the target instant still captures a few minutes late
instead of losing the window entirely.

`grace` is a real per-job decision. `odds_sun_close` gets only 25 minutes
because it targets 12:30 against 13:00 kickoffs — a capture at 13:05 is not a
closing line, it is a live one, and mislabelling that would corrupt every CLV
number computed from it. `odds_tue_open` gets 180 minutes because the opener
moves slowly and a late capture is still an opener.

## Three ways a job does not run, and only one is an alarm

- **`offseason`** — the job is season-guarded and no REG game falls within ten
  days before to three days after the window. Verified at the boundaries
  (measured): off 2026-09-02, on 2026-09-08 through 2027-01-20, off by
  2027-02-15. The market captures are deliberately **not** guarded: books post
  week-1 lines months ahead and the retired tasks had been capturing them all
  summer, so guarding them would silently drop early line movement.
- **`ALREADY-CAPTURED`** — a snapshot for this job's output directory already
  exists inside the window (or within its dedupe threshold). This is what makes
  the job safe to double-schedule, which matters because the Windows tasks
  could not be removed by the agent session and still exist. Whichever runner
  fires first satisfies the other; no duplicate snapshot, no duplicate quota
  spend. Each dedupe threshold is checked to sit under the gap to that job's
  nearest sibling (measured: odds 90m against a 225m gap, injuries 300m against
  990m, public betting 90m against 1440m).
- **`MISSED`** — the window closed, nothing ran, and no snapshot exists to show
  for it. **This one is real data loss** and means the scheduler was not
  running. Restart it and say so.

That distinction is the point of the whole design: cron cannot tell you what it
failed to do. A `MISSED` row is only trustworthy because the other two states
exist to absorb the benign cases — the first draft of this marked every window
the Windows tasks had captured perfectly well as `MISSED`, which would have
trained every future reader to ignore the one alarm that matters.

## Running it

```powershell
# what is scheduled, what ran, what is missed
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status

# run whatever is due right now, then exit (idempotent)
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once

# the background loop (headless -- no console window, nothing in the taskbar)
scripts\start_capture_scheduler.cmd

# stop the background loop
scripts\stop_capture_scheduler.cmd
```

State lives in `data/scheduler_state.json`, log in `data/scheduler_log.txt`
(both gitignored). State keys are `job@occurrence-date`, so a restart cannot
re-run a window that already ran, and entries older than 60 days are pruned.

## Persistence

A shortcut in the user's Startup folder
(`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\nfl-ats capture
scheduler.lnk`) points at `scripts/start_capture_scheduler.cmd`, so the loop
starts at login. No Task Scheduler object, no service, no admin rights — the
persistence mechanism is an ordinary file.

The daemon is headless (owner request, 2026-09-01): the launcher starts uv via
`Start-Process -WindowStyle Hidden`, so its console is created hidden and
nothing appears in the taskbar. (The venv's `pythonw.exe` was tried first and
does not deliver this — uv's pythonw trampoline is a console-subsystem binary
and allocates a visible console anyway, observed 2026-09-01.) Job subprocesses
are additionally spawned with `CREATE_NO_WINDOW` so a capture can never flash a
console of its own. Because there is no visible window, stopping it goes
through `scripts/stop_capture_scheduler.cmd` (kills by command line), not by
window title.

The loop only runs while the machine is on and logged in. That is the honest
limitation, and it is covered two ways: a missed window is reported rather than
silent, and `AGENTS.md` requires every session to run `--once` and read
`--status`, so a dead scheduler surfaces within one session rather than at the
end of the season.

## Migration status (2026-08-25)

The eight Windows tasks (`\NFLATS\Odds_*`, `\PublicBetting_*`) **still exist**:
the agent session was blocked by its harness from unregistering scheduled
tasks. They are harmless in the meantime — the dedupe logic means whichever
runner fires first wins and the other records `ALREADY-CAPTURED` — but removing
them is the last step of the migration:

```powershell
Unregister-ScheduledTask -TaskPath "\NFLATS\" -TaskName Odds_TueOpen,Odds_ThuTNF,Odds_Sat,Odds_SunClose,Odds_SunLate,Odds_MonMNF -Confirm:$false
Unregister-ScheduledTask -TaskName PublicBetting_Sat,PublicBetting_Sun -Confirm:$false
```

Until that runs, both mechanisms are live and the captures are, if anything,
more redundant than before.

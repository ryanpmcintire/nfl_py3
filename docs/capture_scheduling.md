# Capture scheduling

How the recurring point-in-time captures get run, and why the mechanism is what
it is. Written 2026-08-25, when the schedule moved out of Windows Task
Scheduler and into this repository.

## What runs

`scripts/capture_scheduler.py` holds the whole schedule in `SCHEDULE`, 25 jobs:

| Job | When (ET) | Grace | Season-guarded | Catch-up |
|---|---|---|---|---|
| `odds_tue_open` | Tue 09:00 | 180m | no | no |
| `odds_thu_tnf` | Thu 18:00 | 90m | no | no |
| `odds_sat` | Sat 12:00 | 180m | no | no |
| `odds_sun_close` | Sun 12:30 | **25m** | no | no |
| `odds_sun_late` | Sun 16:15 | 60m | no | no |
| `odds_mon_mnf` | Mon 19:00 | 90m | no | no |
| `public_betting_sat` | Sat 12:00 | 240m | no | no |
| `public_betting_sun` | Sun 12:00 | 45m | no | no |
| `injuries_wed` / `injuries_thu` / `injuries_fri` | Wed/Thu/Fri 17:30 | 240m | yes | no |
| `injuries_sat` | Sat 10:00 | 240m | yes | no |
| `refresh_thu` | Thu 15:00 | 240m | yes | no |
| `refresh_sat` | Sat 10:30 | 300m | yes | no |
| `refresh_sun` | Sun 10:00 (`--publish-card`) | 300m | yes | no |
| `backup_data` | Sun 22:00 | 300m | no | **yes** |
| `player_arrests_tue` | Tue 07:00 | 90m | no | **yes** |
| `inactives_sun_early` | Sun 11:35 | 15m | yes | no |
| `inactives_sun_late` | Sun 14:40 | 15m | yes | no |
| `inactives_thu_afternoon_early` | Thu 11:35 | 15m | yes | no |
| `inactives_thu_afternoon_late` | Thu 15:05 | 15m | yes | no |
| `inactives_thu_primetime` | Thu 18:50 | 20m | yes | no |
| `inactives_sat_early` | Sat 15:30 | 15m | yes | no |
| `inactives_sat_late` | Sat 18:50 | 20m | yes | no |
| `referee_assignments_wed` | Wed 15:00 | 240m | yes | **yes** |

Times match the retired Task Scheduler entries exactly, so the migration
changed the mechanism and not the cadence. The four newest additions are
documented below: `backup_data`'s original `MISSED` incident is what
motivated `catch_up` (see "Four ways a job does not run"),
`player_arrests_tue` is the first job added under that field (see "The
player-arrests capture"), the seven `inactives_*` rows are WP17's new
capture channel (see "The official inactives capture (WP17)"), and
`referee_assignments_wed` is WP22's new capture (see "The weekly
referee-assignments capture (WP22)").

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

## Four ways a job does not run on time, and only one is an alarm

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
- **`CAUGHT_UP`** — the job's `catch_up` field is `True`, its window closed
  with nothing run and no snapshot to show for it, and the scheduler ran it
  anyway, right there, on whichever tick first noticed (the next `--once` or
  the next poll of the running daemon). This exists because not every job is a
  point-in-time capture: `backup_data` mirrors whatever is currently on disk,
  and a mirror run twelve hours late is still a correct mirror, unlike a
  "closing line" captured after kickoff. `CAUGHT_UP` is deliberately its own
  status rather than `OK` (which would read as on time, hiding that the window
  was missed) or `MISSED` (which is the one alarm meaning data is gone for
  good, and here it is not — it just arrived late). The state record also
  carries `"caught_up": true`. A job only ever gets this treatment if
  `catch_up=True` on its `Job` row; every point-in-time job defaults to
  `catch_up=False` and keeps behaving exactly as before. It cannot fire twice
  for one occurrence: `run_job` writes the state key the same way an on-time
  run does, so the `key in state["runs"]` check that already stops every other
  status from re-firing stops this one too.

  **Incident that motivated this** (2026-08-31): the machine was off overnight
  and `backup_data`'s Sunday 22:00 window (grace 300m) closed unrun, logged
  `MISSED backup_data (window 2026-08-30T22:00:00-04:00 +300m)`. But
  `scripts/backup_data.py` is fully idempotent — it mirrors `data/` to E: with
  sha256 verification, and a run by hand the same session copied the 7 pending
  files in seconds. Point-in-time captures (odds, injuries, public betting)
  genuinely cannot be caught up — a missed snapshot is gone — but writing off
  an idempotent job to the same permanent `MISSED` verdict was throwing away a
  free recovery. `catch_up` is the fix: `backup_data` and `player_arrests_tue`
  (see below) are the only two jobs with it set.
- **`MISSED`** — the window closed, nothing ran, no snapshot exists to show
  for it, and `catch_up` is `False` (the default) or was never reached (a
  catch-up attempt that itself fails is recorded with the ordinary
  `FAIL(...)` status, not `MISSED` — a failed catch-up run tried and did not
  succeed, which is a different fact from never trying). **`MISSED` is real
  data loss** and means the scheduler was not running. Restart it and say so.

That distinction is the point of the whole design: cron cannot tell you what it
failed to do. A `MISSED` row is only trustworthy because the other three states
exist to absorb the benign cases — the first draft of this marked every window
the Windows tasks had captured perfectly well as `MISSED`, which would have
trained every future reader to ignore the one alarm that matters.

## The player-arrests capture

`player_arrests_tue` (Tue 07:00 ET, grace 90m, `catch_up=True`,
`added_on="2026-09-01"`) runs `nfl-ats ingest-player-arrests`, which builds a
fresh, complete point-in-time snapshot of USA Today's public NFL
player-arrests table (`scripts/ingest_player_arrests.py`). This feeds the
PROMOTED player-arrest policy component on the live card (see HANDOFF.md,
"Current model evidence"); before this job existed, no scheduled capture
refreshed it (**measured**: `grep -n arrest scripts/capture_scheduler.py`
against the pre-WP10 file returned nothing), so the Tuesday publish depended
on whatever snapshot happened to be on disk.

- **Why 07:00, before `odds_tue_open` (09:00)**: the opener is "the grade the
  pool settles on" (`odds_tue_open`'s own `why`), and the Tuesday publish
  should read a fresh arrest snapshot, not a stale one, when it runs. 07:00
  with a 90-minute grace closes by 08:30 — a full 30 minutes of margin before
  the opener capture starts, without needing a razor-thin grace like
  `odds_sun_close`'s (there is no kickoff this job could be mislabeled
  against; running slightly later just means a slightly later, still-valid
  snapshot).
- **Why it is safe as a `catch_up` job**: read `scripts/ingest_player_arrests.py`
  — every invocation without `--snapshot` creates a brand-new UTC-timestamped
  snapshot directory (`new_snapshot_dir`) and only ever *writes once* inside it
  (`_write_once` refuses to overwrite a changed file, never mutates an old
  snapshot). A late or repeated run just produces another equally valid
  snapshot; nothing is corrupted or double-counted. It needs no API key or paid
  plan — read: `"access": {"authentication": "none", ...}` in its own
  manifest — the source is a public, unauthenticated USA Today endpoint, the
  same one `weekly-run` step 7 already calls in production
  (`src/nfl_ats/weekly.py:524`, fatal step, per `docs/week1_readiness.md`).
  Cost is measured, not guessed: the newest snapshot on disk before this work
  (`data/raw/player_arrests/20260831T163452Z/manifest.json`) shows
  `source_total_pages: 56`, `rows_cached: 1116`, `complete: true`, with a
  1.5s per-page delay — a few minutes end to end, far under the 90m grace and
  the scheduler's 1800s subprocess timeout.
- **`season_guarded=False`**: unlike the injury/refresh jobs, the arrests
  archive is not tied to a specific game week — it is a slow-moving historical
  record that can grow at any time of year. Guarding it would leave a stale
  snapshot heading into the season for a job that costs a few minutes once a
  week.
- **`dedupe_dir="data/raw/player_arrests"`, `dedupe_minutes=240`**: protects
  against the scheduler recapturing right behind a same-morning `weekly-run`
  (which calls the identical ingest as its own step 7). 240 minutes sits
  comfortably under the ~10,080-minute gap to next Tuesday's occurrence, so it
  can never be mistaken for a different week's capture.
- **`added_on="2026-09-01"`**: the date this job was written, so past Tuesdays
  are never retroactively branded `MISSED` for a job that did not exist to
  miss them (see `predates_job`, above).

## The official inactives capture (WP17)

Seven `inactives_*` jobs (`added_on="2026-09-01"`, `catch_up=False`,
`season_guarded=True`) run
`scripts/capture_inactives.py --current --slot <name>`, which calls
`nfl_ats.inactives_capture.run_capture` (`src/nfl_ats/inactives_capture.py`).
Predeclared in `docs/inactives_channel.md`'s Section 6 ("Capture-job
proposal"); see that document's "Implementation status" section for what
changed between the proposal and the build, and Section 2 for the T-90
deadline arithmetic every row's derivation comment cites.

- **Why seven jobs, not two.** Section 6 proposed Sunday-early and
  Sunday-late as concrete rows and flagged Thu/Sat as "a genuine design gap":
  measured 2026 Thu kickoffs (13:00 / 16:30 / 20:15 / 20:20 / 20:35 ET) and
  Sat kickoffs (17:00 / 20:20 ET) vary too widely for one fixed weekday+time
  job to sit near T-90 for all of them, so each historically observed cluster
  gets its own job (`inactives_thu_afternoon_early`,
  `inactives_thu_afternoon_late`, `inactives_thu_primetime`,
  `inactives_sat_early`, `inactives_sat_late`) — Option A of the two options
  Section 6 sketched, not the week-relative Option B (a scheduler capability
  the current `Job` model does not have).
- **Why the names differ from the doc's literal text in two spots.** Section
  6 wrote one name, `inactives_thu_afternoon`, against two different times
  ("two separate jobs"); `Job.name` doubles as the run-state key
  (`f"{name}@{date}"`), so two same-named jobs landing on the same Thursday
  would collide and the later one would silently no-op against the earlier
  one's already-written state entry — split into `_early`/`_late` names to
  avoid that. Section 6 also only formalized one Sat job (15:30 ET) by name,
  noting in prose that a 20:20 ET kickoff would need a second one "at sat
  18:50 ET" without writing it up — `inactives_sat_late` builds that second
  job explicitly, following the doc's own logic rather than leaving it
  unbuilt.
- **`dedupe_dir="data/players/inactives"`, not the doc's originally proposed
  `data/raw/nflcom_inactives`.** The actual capture writes snapshots under
  `data/players/inactives/<UTC ts>/` per the WP17 task specification that
  built it; the dedupe target follows where the data actually lands, matching
  how `injuries_*`/`player_arrests_tue` dedupe against their own real
  snapshot directories.
- **`dedupe_minutes=60`** for every row, as Section 6 proposed for
  `inactives_sun_early`/`inactives_sun_late`/`inactives_sat`, extended to the
  Thu cluster on the same reasoning: comfortably under the gap to each row's
  next weekly occurrence, and short enough that a genuinely new window is
  never mistaken for an old one.
- **`catch_up=False` on every row, unlike `backup_data`/`player_arrests_tue`.**
  An inactive list not captured before kickoff cannot be recovered after the
  fact — it is exactly the point-in-time case `catch_up` was deliberately
  never meant to cover (see "Four ways a job does not run", above).
- **No SNF/MNF row.** Section 2 of `docs/inactives_channel.md` measured
  SNF/MNF inactives as ALWAYS arriving after that week's Sunday 16:00 ET pick
  lock (0/17 playable in both slots in the 2026 snapshot), and Section 6
  proposes no Sunday-evening or Monday capture row for grading labels either
  — so none was added here.
- **Grace windows**: 15 minutes on every row except the two primetime-ish
  ones (`inactives_thu_primetime`, `inactives_sat_late`, 20 minutes), which
  each have to straddle a small cluster of nearby historically observed
  kickoff times rather than one fixed instant — mirrors the existing
  short-grace-for-a-tight-deadline pattern (`odds_sun_close`) and the
  wider-grace-for-more-slack pattern (`odds_tue_open`) already in this table.

## The weekly referee-assignments capture (WP22)

`referee_assignments_wed` (Wed 15:00 ET, grace 240m, `catch_up=True`,
`added_on="2026-09-01"`) runs `scripts/capture_referee_assignments.py
--current`, which calls `nfl_ats.referee_assignments_capture.run_capture`
(`src/nfl_ats/referee_assignments_capture.py`). Full source survey,
publication-timing measurements, and the join argument against the
historical crew traits (`docs/referee_battery.md`,
`docs/penalty_crew_tendencies.md`) are in `docs/referee_assignments_capture.md`.

- **Why Wednesday 15:00 ET, not Tuesday.** MEASURED across 10 sampled
  2025-season posts (that doc's Section 2): Football Zebras' own
  `article:published_time` is NEVER before Tuesday afternoon, ranges as late
  as Tue 21:27 ET, and twice landed Wed 12:42 ET (Weeks 8 and 9) — the latest
  observed within a normal (non-finale) week. Wed 15:00 ET clears that by
  over two hours. This also means the capture structurally CANNOT feed the
  Tuesday-lock/opener card for a typical week (unlike `player_arrests_tue`,
  which is timed to precede `odds_tue_open`) — it only ever reaches a
  later-week refresh pass, which is the honest, measured constraint, not a
  scheduling oversight.
- **Why `catch_up=True`, unlike the `inactives_*` rows.** A late-captured
  officiating assignment is still a valid, un-mislabelled snapshot — nothing
  about it goes stale the way a post-kickoff "closing line" or a missed T-90
  inactive list would (the exact reasoning `inactives_*` rows are
  `catch_up=False` for). This mirrors `player_arrests_tue`/`backup_data`
  instead: `src/nfl_ats/referee_assignments_capture.py` is idempotent by
  construction (every run writes a fresh UTC-stamped snapshot directory under
  `data/players/referee_assignments/` and never mutates an older one), so a
  window that closes unrun gets caught up on the next tick rather than
  recorded as permanent data loss.
- **`dedupe_dir="data/players/referee_assignments"`, `dedupe_minutes=240`**:
  matches the injuries/player-arrests convention, comfortably under the
  ~10,080-minute gap to next Wednesday's occurrence.
- **No independent fallback SOURCE, unlike `inactives_*`'s NFL.com/RotoWire
  pair.** MEASURED this session: `operations.nfl.com` carries no weekly
  officiating-assignments page at all (its own `/robots.txt` 404s — a
  client-rendered app, not a page this fetch model can read), and no
  structured third-party mirror was found either. The module instead has two
  strategies against the SAME site: Football Zebras' own reverse-chronological
  `category/assignments/` index page (primary discovery — MEASURED more
  reliable than guessing the post URL directly, since a real 2025 slug
  carried a numeric disambiguator instead of the season, e.g.
  `week-5-referee-assignments-5`) and a direct URL guess as a defensive
  fallback only when the index does not yet list the week.

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
re-run a window that already ran, and entries older than 60 days are pruned. A
record for a `catch_up` job that ran late additionally carries
`"caught_up": true` alongside `"status": "CAUGHT_UP"`.

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

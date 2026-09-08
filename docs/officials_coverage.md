# Officials Wayback coverage report (LEAD-59)

`scripts/officials_coverage_report.py` is a read-only status check over the
officiating-crew archive `scripts/officials_wayback_sweep.py` is building at
`data/raw/officials_pfr_wayback/<run-id>/`. It never fetches anything, never
writes into `data/`, and is safe to run while a sweep is actively writing to
the same files (see "Safe to run mid-sweep" below) -- it only reads what is
already on disk and writes one small JSON report under
`artifacts/research/laneN/`.

## Run it

```powershell
.\.tools\uv.exe run --no-sync python scripts/officials_coverage_report.py
```

That prints the text report below and writes
`artifacts/research/laneN/officials_coverage_report.json`. Pass `--out` to
write somewhere else, `--raw-root` to point at a different archive location,
or `--quiet` to skip the printed text and just write the JSON.

## What it answers

**How much of each season is actually usable.** For every run directory and
every season inside it, the report counts:

- **attempted** -- how many games this run has tried at all.
- **captured** -- how many of those have an archived page saved to disk,
  whether or not that page turned out to have the officiating info on it.
- **parsed with a crew** -- how many captured pages actually named at least
  one official.
- **complete crew** -- how many captured pages named all seven on-field
  positions (Referee, Umpire, Head Linesman, Line Judge, Field Judge, Side
  Judge, Back Judge). This is the number that matters for anything that
  wants the full crew, not just the head referee.
- **failed** -- games where nothing ever landed on disk at all, split into a
  network/fetch failure vs. Wayback confirming no archived page exists.
- **pending** -- games in that season's schedule the sweep hasn't gotten to
  yet (this needs the local schedule snapshot; it's skipped with `--no-pending`
  if that snapshot isn't available).

**Who officiated, and how completely.** Across every run, the report picks
one canonical page per game (see "Duplicates" below) and reports how many
different referees show up and how many games each one worked -- a quick
sanity check that the names look like real, distinct officials rather than
parsing garbage.

**Duplicates and conflicts.** If the same game ever gets captured by more
than one sweep run (this can't happen yet since no two runs cover the same
season, but the report checks for it every time in case that changes), it
says so, lists every capture found, and says which one it's treating as the
authoritative one: the newest archived copy of a captured page beats an
older one, and a captured page always beats an empty one. This mirrors the
sweep's own rule for picking between multiple archived copies of the SAME
game (it prefers the newest one taken after the game was played), just
extended to cover two separate sweep runs instead of two Wayback snapshots
inside one run.

**Whether anything downstream actually reads this archive yet.** The
report checks the source code directly and states plainly: as of this
report, nothing does. The crew-trait features used today
(`nfl_ats.officials_flag_features`, `nfl_ats.crew_tilt_refresh_overlay`)
only read the newer, separate officials feed that already covers
2015-2025; this Wayback archive is not wired into them. The plan on file
(`ROADMAP.md`, row LEAD-59) is to extend that feature set back to 2009 once
this archive is far enough along, but there is no written rule anywhere
for exactly how complete a season has to be before it's "far enough along"
-- so the report doesn't invent one. It just prints the real per-season
coverage percentage so that call can be made deliberately, with the actual
numbers in front of whoever makes it.

## Safe to run mid-sweep

The Wayback sweep writes its `manifest.json` by writing a temp file and then
swapping it into place, so a read almost always sees a clean, complete file.
On the rare occasion a read lands in the middle of that swap (a real
Windows race, seen once before), the report retries a few times in a
fraction of a second before giving up and marking that run "busy" for this
pass rather than crashing. Likewise, if a manifest names a page file that
the sweep hasn't finished writing yet, the report just notes the page isn't
there yet -- it never waits for it or assumes it will show up.

## Files

- `scripts/officials_coverage_report.py` -- the report itself.
- `tests/test_officials_coverage_report.py` -- offline tests against small,
  made-up sweep directories (never the real archive), covering a complete
  crew, a page missing the referee, a fetch that never produced a page, and
  a game captured by two different runs.
- `artifacts/research/laneN/officials_coverage_report.json` -- the most
  recent real run's output, stamped with the code revision that produced it.

# NFL.com injuries sourcing

Status: ingested and agreement-checked 2026-08-21 (measured, this session).

## Source

The league's own weekly league-wide injury report pages,
`https://www.nfl.com/injuries/league/{season}/reg{week}` — candidate rank B1 in
`docs/data_source_scout_v5.md` Section B. Each page is plain HTML: one
`section.nfl-o-injury-report__unit` per game carrying the matchup strip's two
team abbreviation tags and one `table.d3-o-reports--detailed` per team with
columns Player / Position / Injuries / Practice Status / Game Status.
Practice statuses are spelled out ("Did Not Participate In Practice",
"Limited Participation in Practice", "Full Participation in Practice"); game
statuses are Out / Doubtful / Questionable (blank when none). These are FINAL
designations published Friday/Saturday — after the pool's Tuesday grading-line
freeze but before kickoff.

## robots.txt and politeness

Measured 2026-08-21 (`www.nfl.com/robots.txt`): nothing under `/injuries/` is
disallowed for `User-agent: *`, and no Crawl-delay directive is set. The
ingester re-fetches and re-evaluates robots.txt at runtime BEFORE any page
fetch and fails closed if fetching is disallowed; it enforces a >= 2s delay
(2.5s default) between page fetches regardless.

## Snapshot

`scripts/ingest_nflcom_injuries.py`; immutable snapshot convention matching
the repo's other raw sources (nested under `data/raw/nflcom_injuries/<UTC ts>/`
so `nfl_ats.snapshots.latest_snapshot()` never mistakes it for a schedules
snapshot; `data/raw/` is gitignored):

- `<snapshot>/pages/{season}_reg{week}.html` — verbatim HTML, one sha256 per
  page recorded in the manifest.
- `<snapshot>/injuries.parquet` — tidy rows (season, week, team, player,
  position, injury, practice_status, game_status, source_url,
  fetched_at_utc).
- `<snapshot>/manifest.json` — robots result, per-page status/sha256/rows,
  every fetch failure recorded with its error, coverage summary, warnings.

Current snapshot (measured): `data/raw/nflcom_injuries/20260821T222602Z` —
54/54 pages OK (REG weeks 1-18 x seasons 2022/2023/2024), 0 failures,
17,483 parsed player-week rows (2022: 5,452; 2023: 5,439; 2024: 6,592),
all 32 team codes resolved with zero unresolved-team warnings.

Team attribution uses each table's section sub-title nickname mapped to a code
and cross-checks the matchup strip's abbreviation order; both agreed on every
section in this snapshot (no warnings recorded).

## Stage 1b agreement vs the local nflverse feed

Local feed: newest `data/players/raw/*/injuries.parquet` (measured 2026-08-19
elsewhere as structurally a Wed-Fri artifact; 2022-2024 REG rows here: 16,855,
of which 8,094 carry a non-null report status). nflverse rows have gsis_id but
no name, so names come from resolving gsis_id through the same snapshot's
`weekly_rosters.parquet` (modal full_name per id).

Documented join normalization: casefold, ASCII-fold accents, strip punctuation,
drop suffix tokens (jr/sr/ii/iii/iv/v); primary key season+week+team+
normalized full name, fallback first-initial+last-name where that key is unique
on both sides within the same season+week+team.

Measured (`artifacts/nflcom_injuries/20260821T222602Z/agreement.json`):

| Metric | Value |
|---|---|
| Match rate, NFL.com player-weeks found in nflverse | 99.63% (16,510 exact + 283 initial-last of 16,855) |
| Game-status exact agreement, jointly designated rows | 7,911 / 8,293 = 95.39% |
| Mismatch mass | nflverse never designated: 374 (289 Q, 76 Out, 9 D); `Note` rows: 6; Q<->Out flips: 2 |

Gate verdict: **the source reproduces the local feed closely enough to be its
post-2024 replacement candidate** — near-total player-level match and ~95%
status agreement, with residual disagreement concentrated in rows the local
feed simply never designated. It is also the only one of the two still
publishing. The reverse direction matters for the screen: NFL.com carries
17,483 rows against the feed's 16,855 in-scope rows, i.e. the league page is a
superset, not a lossy view.

## What this source is NOT

These are final Fri/Sat designations. They cannot reconstruct what was knowable
at the Tuesday lock (see the screen doc's cell (c) measurement: only 85 of
16,855 in-scope nflverse rows are Tuesday-dated, so essentially all final
report content is post-Tuesday information by construction).

---

## 2026-08-25: in-season live capture (the revision stream)

The historical backfill above is a one-shot archive of FINAL pages. It does
not, and cannot, capture what the league page looked like on Wednesday and
Thursday of a game week — the page is a living document that teams overwrite
as practice participation and Friday game-status designations land. Those
intermediate states are unrecoverable retroactively, so every week without a
live capture is permanently lost point-in-time data.

**New incremental mode.** `ingest_nflcom_injuries.py --current` resolves the
live (season, REG week) from the schedules snapshot — the earliest REG week
that still has an unplayed kickoff — and fetches only that week's page into a
FRESH UTC-stamped snapshot directory. Verified across dates this session
(measured): it holds week 1 through the Wednesday opener, rolls to week 2 only
once week 1 is fully played, and resolves week 5 on a week-5 Wednesday. A fresh
directory per run is what preserves each revision as its own immutable
snapshot.

**Snapshot selection had to change with it.**
`nfl_ats.prospective.latest_nflcom_injuries_snapshot` previously read the
lexicographically newest directory, full stop. Once weekly capture runs, that
newest directory holds ONE week, so the first live capture would have hidden
the entire 2022-2024 backfill from every historical read. It now takes the
(season, week) it needs and returns the newest snapshot that actually holds
that page, which also naturally prefers the FINAL revision of a week captured
several times. Verified with both snapshots on disk (measured): 2026 w1
resolves to the new capture while 2024 w5 and 2022 w12 still resolve to the
archive. Pinned by `test_a_weekly_capture_does_not_hide_the_historical_archive`
and `test_a_later_capture_of_the_same_week_wins`.

**Scheduling.** Capture runs from `scripts/capture_scheduler.py`, the in-repo
scheduler that replaced the Windows Task Scheduler entries (see
`docs/capture_scheduling.md`). Four windows per week — Wed/Thu/Fri 17:30 and
Sat 10:00 ET — with the Friday run being the one the frozen challenger rule
consumes. A job is skipped as `ALREADY-CAPTURED` when a snapshot under
`data/raw/nflcom_injuries` is less than 300 minutes old, so a manual run, a
second scheduler copy, or a `--once` invocation cannot produce duplicate
captures. Any session can bring things current with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once
```

**Observed source states.** Live-tested 2026-08-25: a historical week fetches
and parses normally (2024 week 5 → 367 rows, one page, fresh snapshot). The
not-yet-published 2026 week 1 page returns **HTTP 200 with zero rows** — an
empty report, not an error, and indistinguishable from "nobody is listed".
Mid-season a zero would instead mean the page shape changed, so it deserves a
look rather than a shrug.

**First live capture taken 2026-08-25**:
`data/raw/nflcom_injuries/20260825T191422Z` (2026 week 1, 0 rows, page not yet
published). The 2026 data gap that would otherwise have silenced this
challenger for the whole season is now closed as a repeating process rather
than a one-off.

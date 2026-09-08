# The nflverse season rollover gap (2026-09-08)

## What happened

`nflreadpy` guards every seasonal loader with its own `get_current_season()`,
which rolls over on **the Thursday following Labor Day**. Labor Day 2026 fell on
Monday 2026-09-07, so that function returned `2025` until Thursday 2026-09-10.

2026 Week 1 opened on **Wednesday 2026-09-09** (NE at SEA, 8:20 PM ET).

For two days — covering the season's first game and its pick deadline — every
call of the form `nflreadpy.load_injuries(seasons=[2026])` raised:

```
ValueError: Season must be between 2009 and 2025
```

This is the same failure shape as the missing Wednesday capture window fixed
2026-09-07: a rule that assumes the season starts on a Thursday, applied to a
season that starts on a Wednesday.

## The data was there the whole time

Measured 2026-09-08 by `HEAD` request against the release assets:

| release | status | note |
| --- | --- | --- |
| `injuries/injuries_2026` | HTTP 200, 6,405 bytes | 11 rows, exactly the NE/SEA opener |
| `weekly_rosters/roster_weekly_2026` | HTTP 200, 537,630 bytes | live |
| `snap_counts/snap_counts_2026` | HTTP 404 | genuinely unpublished — no games played |

The eleven injury rows were three New England and seven Seattle players
carrying practice designations for the very game the pool's first pick was due
on, including two who did not practise at all.

## Why nobody noticed

`scripts/build_week_lineups.py` already caught the `ValueError` and carried on,
so nothing ever crashed. It reported the cause as:

> nflverse has not published season 2026 injuries yet

Both halves of that sentence were false. The publisher had the rows; the client
refused to ask for them. The card in turn told readers no injury report existed.

**A graceful degradation that states the wrong reason is worse than a crash**,
because a crash gets investigated and a wrong reason gets believed. The
degradation itself was right — the fallback belongs there. Only its diagnosis
was wrong, and a wrong diagnosis is what kept this invisible for two days.

## The fix

`src/nfl_ats/nflverse_current_season.py`. The season guard lives in the
`load_*` wrappers, not in the transport: `load_injuries` validates the season
and then calls
`get_downloader().download("nflverse-data", f"injuries/injuries_{season}")`.
Calling that downloader directly reuses nflreadpy's own URL resolution, HTTP
session, caching and parquet parsing — everything except the date rule — so
this module hand-rolls no networking.

Two properties the implementation holds deliberately:

- **Unguarded seasons do not change path.** `load_seasons_frame` sends every
  season nflreadpy will serve through one ordinary bulk call and returns its
  frame untouched, with no concatenation in the path. When nothing is guarded —
  every season the archive is built from — the result is exactly what the
  previous code produced. That matters more here than anywhere else, because
  this feeds the model's feature table and the project's methodology rests on
  those artifacts reproducing exactly.
- **A genuine absence is still an absence.** A 404 becomes
  `SeasonReleaseNotPublished`, so `snap_counts_2026` still fails and a caller
  can honestly say the data does not exist. A real network fault propagates
  untouched: "the site is down" must never be reported as "the season does not
  exist".

## The three consumers that were affected

1. `scripts/build_week_lineups.py` — the live weekly injury feed behind the
   lineups page and the card's injury sentence. Now routed through
   `load_season_frame`.
2. `src/nfl_ats/players.py` (`fetch_player_snapshot`) — the model's feature
   pipeline. Its newest snapshot before this fix covered 2009–2025 only, with
   2026 absent from `injury_seasons` entirely. Now routed through
   `load_seasons_frame`.
3. `scripts/nflverse_injuries_ingest.py` — the bulk snapshot the refresh
   overlays read (`specialist_absence_fade_refresh_overlay
   .latest_nflverse_injuries_snapshot`). Its `SEASON_END` was set to 2025 on
   2026-08-26 as "one past the last season nflreadpy's own
   `get_current_season()` resolved to", which imported nflreadpy's rollover
   rule into this repo's data coverage. Raised to 2026 and re-run: snapshot
   `20260908T192059Z`, 90,763 rows, 11 of them 2026.

## Verified against the real card

`_fetch_current_week_injuries(2026, 1, ...)` at each scheduled refresh time:

| refresh pass | before | after |
| --- | --- | --- |
| now (Tue 2026-09-08) | 0 rows, wrong reason | 0 of 11 — leakage guard, correct |
| Wed 18:15 ET (`refresh_wed`) | 0 rows, wrong reason | **11 of 11 visible** |
| Sat 15:50 ET (`refresh_sat_inactives_early`) | 0 rows, wrong reason | 11 of 11 visible |

The Tuesday zero is not a defect. `canonicalize_injuries(timestamp_fallback=
"week_proxy")` treats a row with no `date_modified` as observed at its own
team's kickoff minus `INJURY_PROXY_HOURS_BEFORE_KICKOFF` (24), so the NE/SEA
rows only become visible at Tuesday 8:20 PM ET. That is the leakage rule doing
its job, and it is why the first pass that can actually use these rows is
Wednesday's.

## The general lesson

A third-party library's calendar heuristic is not this repo's season boundary.
Where a loader's own rule decides what data exists, name the rule and measure
it rather than inheriting it — and when a fallback fires, make it state the
reason it actually observed, not the reason it assumes.

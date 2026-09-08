# Officials archive: one crew table for 2009-2025

`src/nfl_ats/officials_archive.py` (LEAD-59). Until this module the officiating-crew
features could only see 2015-2025, because that is all the nflverse officials feed
covers. The Wayback sweep has since captured Pro-Football-Reference's own boxscores
back to 2009, and this module is what turns those captures into a table the crew
features can actually read.

## What it does

Two sources, one table:

| Source | Path | Seasons | What it carries |
| --- | --- | --- | --- |
| nflverse officials feed | `data/raw/officials/<snapshot>/officials.parquet` | 2015-2025 | full crews, jersey numbers, official ids, REG + POST |
| Wayback PFR boxscores | `data/raw/officials_pfr_wayback/<run-id>/` | 2009-2014 | full seven-person crews, REG only, no jersey numbers or ids |

Three entry points:

- **`canonical_crew_table()`** — one row per archived game, wide: `referee`, `umpire`,
  `head_linesman`, `line_judge`, `field_judge`, `side_judge`, `back_judge`, plus which
  sweep run, which Wayback capture timestamp, and which URL the row came from.
  `season`, `week`, `gameday` and the legacy `old_game_id` come from the schedule
  snapshot, which is the authority; the manifest supplies only the capture provenance.
- **`archive_officials_long()`** — the same crews in the nflverse feed's own 9-column
  long schema, keyed on the LEGACY numeric `game_id` (crosswalked through
  `schedules.parquet`'s `old_game_id`), so it is directly comparable with the feed and
  joins the same way every existing consumer already joins.
- **`load_officials()`** — the single loader every crew consumer now calls
  (`officials_flag_features`, `crew_tilt_refresh_overlay`, and both referee-trait
  builders in `experiment_runner`).

### Validation, all fail-closed

- A crew row must join to **exactly one** schedule game. Zero matches or two matches
  raises `OfficialsArchiveError` rather than silently dropping or duplicating a game.
- A capture timestamp must land **after the game's own day** — the sweep's own CDX
  bound (`from = gameday + 1 day`). A missing, malformed, same-day or earlier timestamp
  raises.
- A directory whose `manifest.json` does not declare
  `officials_pfr_wayback_manifest/1` is skipped, not crashed on. That is how the two
  `laneN_probe_*` diagnostic directories stay out of the table.
- Disk is the only truth about what was captured: a manifest row counts only when the
  `html_file` it names actually exists. A retried row can carry a `*_failed` label while
  still holding a page kept from an earlier attempt, and a live sweep can name a page it
  has not finished writing.

### Duplicate resolution

- **Across sweep runs**: the newest Wayback capture wins, ties broken on the earlier run
  id. This extends the sweep's own `capture_policy`
  (`newest_post_game_capture_with_fallback`) to the cross-run case it never has to
  resolve itself. Measured 2026-09-08: **zero** real games are captured by two runs —
  the two run directories cover disjoint season windows.
- **Within one page**: PFR occasionally lists a position twice with a variant name.
  Measured 2026-09-08, exactly two cases in the whole archive — `2014_06_DET_MIN` lists
  both "John Parry" and "John Perry" as Referee, and `2014_09_PHI_HOU` lists two Back
  Judges. The first row in page order wins and
  `discarded_duplicate_position_rows` counts what was dropped, so a duplicated Referee
  can never double-count a game downstream.

### Merge policy

`load_officials(include_archive=True)` returns the feed's rows **unchanged** and appends
archive rows for games the feed does not carry. **nflverse wins on overlap**, at game
granularity: an archive game whose legacy `game_id` is already in the feed is dropped
whole, never merged position-by-position. `tests/test_officials_archive.py` pins the
2015-2025 slice bit-for-bit against the feed. The one declared schema difference is
`jersey_number` widening from `int32` to the nullable `Int32`, because a PFR boxscore
carries no jersey numbers; every 2015-2025 jersey number stays present and identical.

### The archive is OFF by default

`INCLUDE_ARCHIVE_DEFAULT` is `False`. With it off, `load_officials()` returns the feed
bit-for-bit, so routing every consumer through it — which this change does — moves no
production number at all. Turning it on does not change any 2015-2025 *row*, but it does
widen the population every derived crew trait is computed over: a referee's
`prior_seasons_experience`, the lagged penalty-rate quartile cutpoints,
`describe_referee_left_censoring`'s 2015 censoring count. Those changes are the *point*
of extending back to 2009, and they are also card-affecting, so the flip belongs to
LEAD-59's own next step ("re-run the officials flags on production") and to a measured
read through the played card — not to a loader default. `INCLUDE_ARCHIVE_DEFAULT` is the
one place to flip it.

## Timing contract

**What the archive can prove:** that a given seven-person crew officiated a given game.

**What it cannot prove:** that the assignment was published at any particular time
before kickoff. Every Wayback capture is dated after the game — the sweep queries CDX
with `from = gameday + 1 day` on purpose, because a PFR boxscore captured before kickoff
is a placeholder page with no officials block at all.

**Is that admissible?** Yes, for historical/backtest features, and on exactly the same
footing the existing 2015-2025 family already stands on. The nflverse officials feed is
itself a post-hoc dataset with no capture timestamp of any kind. The referee battery is
admissible because (a) crew identity is public before kickoff *in reality* — the league
and Football Zebras publish assignments midweek, measured in
`docs/referee_assignments_capture.md` section 2 — and (b) every trait built on top of it
uses only the crew's own prior games. The archive meets both conditions and carries
strictly more provenance than the feed it extends: it at least records when the page
that reports the crew was captured.

So the family is labelled, honestly and in code:

```
ARCHIVE_TIMING_CLASS = "crew_identity_public_pregame_not_provably_captured_pregame"
```

- **Admissible** for historical crew-identity features and their prior-game traits, on
  the same terms as 2015-2025.
- **NOT admissible** for the prospective refresh channel. `crew_tilt_refresh_overlay`
  (fed by the `referee_assignments_wed` capture) requires a snapshot whose
  `captured_at_utc` is strictly before each game's own `min(kickoff, Sunday 16:00 ET)`
  deadline. Every archive capture is after kickoff by construction, so no archive row can
  ever satisfy that test.

That refusal is enforced in code, not prose. `crew_tilt_refresh_overlay` loads through
`load_officials_for_prospective_channel()`, which never includes the archive and then
re-checks the result with `refuse_archive_rows()` so a future edit cannot quietly widen
it. The two leakage regression tests AGENTS.md requires for a new feature family are
`test_a_capture_at_or_before_kickoff_fails_closed` and
`test_archive_rows_are_refused_by_the_prospective_channel`.

## Coverage today

Measured 2026-09-08 by running `canonical_crew_table()` against the real archive.
1,416 games, 1,415 with a complete seven-person crew, 9,911 crew rows. A full REG season
is 256 games.

| Season | Games | Complete crews | Distinct referees | Duplicate position rows discarded |
| ---: | ---: | ---: | ---: | ---: |
| 2009 | 248 | 248 | 17 | 0 |
| 2010 | 254 | 254 | 17 | 0 |
| 2011 | 254 | 254 | 17 | 0 |
| 2012 | 256 | 255 | 34 | 0 |
| 2013 | 149 | 149 | 17 | 0 |
| 2014 | 255 | 255 | 19 | 2 |

Two things to read carefully:

- **2013 is still filling.** The 2009-2013 sweep run `20260907T175420Z` was still
  running when this table was measured, and 2013 is the season it has not finished. The
  number moves upward on its own; re-run `describe_archive_coverage()` for the current
  one rather than quoting this row.
- **2012's 34 distinct referees** against 17-19 in every other season is very plausibly
  the 2012 replacement-referee lockout, a real historical event — but that is my
  reading, not a verified fact, and nothing in this repository has checked it against an
  external source.

The handful of games missing from 2009-2011 and 2014 are the sweep's own `cdx_fetch_failed`
/ `no_capture_found` rows, itemised per run and per season by
`scripts/officials_coverage_report.py` (`docs/officials_coverage.md`).

## What LEAD-33 (late-season all-star crews) can now do

**The source gap is not closed.** LEAD-33's premise is that hand-picked late-season
crews call tighter. Neither source carries an all-star, Pro-Bowl, or crew-designation
marker: the nflverse feed's nine columns have none (recorded on the LEAD-33 row in
`ROADMAP.md`), and a PFR boxscore's crew block has none either — it lists seven names and
seven positions, nothing more. No proxy is invented here, and the week-17/18-assignment
heuristic that ROADMAP already ruled out as invented rather than measured stays ruled out.

**What the archive does change** is that the full seven-person crew — not just the head
referee — is now available for 2009-2014 as well as 2015-2025. That matters for LEAD-33
because the only *measurable* handle on "this is a scrambled, hand-picked crew" is crew
COMPOSITION: within a season, each official has a modal crew (the referee they work under
most), and a game whose seven officials are drawn from several different modal crews is
compositionally scrambled relative to a normal week. That statistic is derivable from
crew membership alone, needs no designation marker, and — with this loader — is now
computable over seventeen seasons rather than eleven, across two clearly different
officiating eras.

That is a candidate, not a finding. It has to be predeclared (population, threshold,
direction — ROADMAP already fixes the direction as BACK favourites under all-star crews)
before any cover rate is looked at, and recorded through
`nfl-ats weak-signals record` whatever it shows. Nothing here says the effect exists.

## Reading the tables yourself

```powershell
.\.tools\uv.exe run --no-sync python -c @'
from nfl_ats.officials_archive import canonical_crew_table, describe_archive_coverage
table = canonical_crew_table()
print(describe_archive_coverage(table=table))
print(table.head().to_string())
'@
```

`scripts/officials_coverage_report.py` (`docs/officials_coverage.md`) remains the
per-run, per-season fetch-outcome report — attempted, captured, failed, pending. This
module is the consumer-facing table; that script is the sweep's own progress view.

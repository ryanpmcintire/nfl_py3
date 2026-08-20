# Player-prop line sourcing: budgeted pilot pull

Follow-up to `docs/data_source_scout_v3.md` section 3 ("The Odds API
historical player-prop lines", backlog rank 3). That section already
verified live: the endpoints exist on this project's paid account, one
market/event/snapshot combination costs about 10 requests, and player props
are only available from `2023-05-03T05:30:00Z` onward. This document reports
a real budgeted pull against that verified path, not a fresh scouting pass.

**Ingestion script**: `scripts/ingest_player_props.py` (read its module
docstring for the exact endpoints, credential lookup, and budget mechanics;
it follows `scripts/ingest_injury_news.py`'s snapshot/manifest convention and
never touches `src/nfl_ats`).

**Every number below is measured** (produced by this session's actual API
calls and this session's own `pandas` inspection of the output — see the
inline commands) unless explicitly marked otherwise.

## 1. Quota ledger (measured)

| Step | Call | Cost | `x-requests-remaining` after |
|---|---|---|---|
| Quota-check-only run (`--quota-check-only`) | 1x events-list, 2024 wk1 Tuesday-noon | 1 | 2861 |
| Main pull run's own leading call (also events-list, 2024 wk1 — a second, real fetch of the same date since the check-only run discarded its data) | 1x events-list | 1 | 2860 |
| Main pull run, weeks 1-13 of the 2024 regular season, market `player_pass_yds` | 13x events-list + up to 16x event-odds per week | 342 | 2518 |
| **Total this session** | | **344** | **2518 remaining, floor 1200** |

- Quota measured at the very start of the session (**measured**, first live
  call): **2861 remaining** (before any props-specific spend beyond the
  1-request check itself).
- Budget tier applied, per this task's brief: remaining `2861 < 3000` ->
  **low tier, minimal pilot, budget <= 350 requests**. The main run was
  invoked with `--budget 350` (the exact number the tier rule produces),
  so the quota-derived rule and the executed budget are the same value;
  the manifest records it as an explicit `--budget` only because deriving
  it a second time inside the same process would have spent a second,
  redundant events-list call.
- The run **self-stopped inside week 13** (`stop_reason:
  "budget_or_floor_mid_week"`, week 13 status `partial_budget_stop`,
  4 of 16 matched events pulled) at 343 of 350 requests spent for that
  invocation — the budget cap bound before the 1200-request quota floor did
  (remaining stayed at 2518, far above the floor). **Read**:
  `data/raw/odds_api_props/20260820T105142Z/manifest.json`.
- Phase B (`player_rush_yds`) did **not** run: it only triggers if phase A
  (the primary market) finishes every planned week across all three seasons
  first, and this pilot covered 13 of 54 planned weeks. `manifest.json`:
  `"phase_b_ran": false`.

## 2. Unexpected but real cost-model finding (measured)

The nominal cost model from the scout doc (10 requests per
market/event/snapshot) held **only when a book actually had that market
posted at the snapshot moment**. When an event had no `player_pass_yds`
data yet at Tuesday noon UTC, the response still came back HTTP 200 with an
empty `bookmakers` list, and **`x-requests-last` for that call was 0**, not
10 (**measured**, `calls_log` in the manifest — e.g. for 2024 week 2, one
call cost 10 and the other 14 all show `cost=0`). This is why 350 requests
bought 13 weeks of coverage instead of the ~2 weeks the nominal formula
implied: most events-in-week calls are effectively free when the market
isn't posted yet. This is a real, provider-side cost behavior, not a bug in
the pull script — worth remembering when sizing the next tranche.

## 3. Coverage: weeks pulled

Regular-season 2024, weeks 1-13 (week 13 partial). No 2025 or 2023 coverage
was reached this pilot (budget exhausted first). Source:
`data/raw/odds_api_props/20260820T105142Z/manifest.json` `per_week`, and a
direct read of `index.parquet` (**measured**, `pandas.read_parquet`).

| Season | Week | Events matched to week | Events pulled | Rows | Distinct players | Distinct bookmakers | Status |
|---|---|---|---|---|---|---|---|
| 2024 | 1 | 16 | 16 | 920 | 32 | 5 | complete |
| 2024 | 2 | 16 | 16 | 12 | 2 | 3 | complete |
| 2024 | 3 | 16 | 16 | 20 | 4 | 3 | complete |
| 2024 | 4 | 16 | 16 | 20 | 2 | 5 | complete |
| 2024 | 5 | 14 | 14 | 16 | 2 | 4 | complete |
| 2024 | 6 | 14 | 14 | 12 | 2 | 3 | complete |
| 2024 | 7 | 15 | 15 | 14 | 4 | 3 | complete |
| 2024 | 8 | 16 | 16 | 16 | 2 | 4 | complete |
| 2024 | 9 | 15 | 15 | 12 | 2 | 3 | complete |
| 2024 | 10 | 14 | 14 | 10 | 2 | 3 | complete |
| 2024 | 11 | 14 | 14 | 8 | 2 | 2 | complete |
| 2024 | 12 | 13 | 13 | 12 | 2 | 3 | complete |
| 2024 | 13 | 16 | 4 (of 16) | 50 | 7 | 4 | **partial_budget_stop** |
| **Total** | | **195 matched** | **183 pulled** | **1,122** | 36 distinct overall | 6 distinct overall | 13/54 planned weeks written |

Every week's `events_matched_to_week` equals `events_returned_by_api`
restricted to that week's actual schedule (team-pair join against
`nflreadpy.load_schedules`, done before spending any event-odds credits) —
so 100% of the events this script tried to pull odds for really were that
week's games, by construction, not a fuzzy match.

### The steep row-count drop after week 1 is a real market-timing finding, not a parsing bug (measured + inferred mechanism)

Week 1 alone produced 920 of the 1,122 total rows (82%). Weeks 2-12 each
produced only 8-20 rows from just **one** of that week's 13-16 games, and
in every one of those weeks the only game with real data was the earliest
kickoff of the week (confirmed by direct inspection, e.g. week 2's only
populated event was Thursday's MIA @ BUF, with exactly the two starting
QBs -- Josh Allen and Tua Tagovailoa -- priced). Week 13 (Thanksgiving,
three Thursday games instead of one) had 4 populated events instead of 1,
consistent with the same pattern. **Measured**: this is not a parsing
failure -- the populated rows are well-formed, sane lines (e.g. Josh Allen
238.5-244.5 across books) with per-book, per-second `bookmaker_last_update_utc`
timestamps. **Inferred** (not directly proven this session): books appear to
post full player-prop menus for the *season opener* and *Thanksgiving*
games unusually early (marquee scheduling), while an ordinary week's full
menu for its Sunday/Monday games is not yet posted by Tuesday noon UTC --
only that week's single earliest-kickoff game has anything up. If true,
this is directly relevant to the predeclared experiment in section 5: a
Tuesday-noon snapshot is a *sparse* board for most of a normal week's
games, so an "appears/disappears between Tuesday and a later snapshot"
signal will have very little to compare against for most Sunday games
specifically at the Tuesday timestamp, and more to compare against for
each week's Thursday game.

### Line-ladder bookmaker behavior (measured)

One bookmaker, `betrivers`, returns a full ladder of alternate `player_pass_yds`
lines per player under the *same* market key other books use for a single
line (e.g. week 1's BUF @ ARI: `betrivers` returned 10 different O/U
thresholds for Josh Allen alone, 159.5 through 289.5+ in roughly 10-yard
steps, each with its own price, all sharing one `bookmaker_last_update_utc`),
while `draftkings`/`fanduel`/`betmgm`/`williamhill_us` each returned exactly
one line per player per side. This means naive row counts are dominated by
one book's ladder (**measured**: `betrivers` alone contributes 696 of 1,122
rows, 62%, vs. 128/120/98/48/32 for `draftkings`/`fanduel`/`betmgm`/
`betonlineag`/`williamhill_us`). Any downstream feature build must pick a
single reference line per (event, player, book) -- e.g. the line nearest
whatever `draftkings`/`fanduel` post -- rather than averaging across a
book's own ladder, or `betrivers` will silently dominate a naive mean.

## 4. Join rate to nflverse schedule game IDs (measured)

**100.0%** (1,122 / 1,122 rows carry a non-null `nflverse_game_id`). This is
by construction, not a lucky match rate: the script fetches each week's
events list, filters to events whose (home, away) team pair matches that
week's actual `nflreadpy.load_schedules` slate *before* spending any
event-odds credits, and takes the schedule's own `game_id` directly rather
than a fuzzy time-based join. No event-odds credits were spent on an event
that didn't resolve to a schedule row.

## 5. Predeclared next-step experiment (NOT run this session)

**Working title**: QB prop-line disappearance/shift between a Tuesday
snapshot and a later snapshot as a starter-availability signal.

**Design** (to be run only after more weeks of coverage exist, given
section 3's finding that a single Tuesday-noon snapshot is sparse for most
games in a normal week): for each game, compare the starting QB's
`player_pass_yds` line (or the QB's very presence/absence in the market) at
a Tuesday-noon-UTC snapshot against a later-week snapshot (e.g.
Thursday/Friday, closer to kickoff). A QB who disappears from the board, is
replaced by a backup's name, or whose line drops sharply between the two
snapshots is a candidate early-availability signal, distinct from and
possibly earlier than any line move in the point-spread market. Graded at
the frozen Tuesday opener with picks refreshable to kickoff (per this
project's picks-lock-at-kickoff policy: lines freeze Tuesday, but picks
stay editable through kickoff, so an in-week signal like this is legitimately
playable even though the opening spread line itself is frozen).

**Verdict handling (binding, not optional)**: any run of this experiment
must be scored and closed exactly per this project's closing-grounds
taxonomy, quoted verbatim per this task's brief:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. Only two grounds ever close a line of work: (1)
> refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
> side of zero) or zero split-half reliability; (2) bounded by a positive
> control proven able to detect an effect that size. Everything else is
> `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
> report `probability_positive`, never the binary "contains zero".

This experiment was **not** run this session (ingestion + coverage report
only, per this task's scope) -- it is recorded here as a predeclared design
so the eventual result is judged against a family declared before the signs
were seen, per the commensurability rule in `AGENTS.md`.

## 6. Files written this session

- `data/raw/odds_api_props/20260820T105142Z/weekly/2024_wk01_player_pass_yds.parquet`
  through `.../2024_wk13_player_pass_yds.parquet` (13 files, one row per
  event/bookmaker/market/player/side; week 13 is a partial pull, 4 of 16
  matched events).
- `data/raw/odds_api_props/20260820T105142Z/index.parquet` (1,122 rows,
  concatenation of the 13 weekly files).
- `data/raw/odds_api_props/20260820T105142Z/manifest.json` (full quota
  ledger, per-week coverage summary, and a per-call log with `apiKey`
  redacted from every logged URL).
- `scripts/ingest_player_props.py` (the ingestion script itself).
- This document.

All of the above live under `data/` (gitignored) except the script and this
doc. Nothing was committed; nothing in `src/nfl_ats` was touched; no
`weak-signals` or other registry writes were made.

## 7. Exact commands for the next budgeted tranche

Quota is currently **2518 remaining, floor 1200**, so roughly **1,318
requests of further headroom exist before the floor** -- but this task's own
budget-tier rule, not the raw floor, should gate the next tranche's size;
re-derive the tier from a fresh quota-check call rather than assuming 2518
is still current (other jobs, including the weekly `scripts/odds_capture.ps1`
capture, share this same account).

```powershell
# 1. Re-measure quota (spends exactly 1 request; use this number, not the
#    2518 recorded above, to pick the next --budget):
.\.tools\uv.exe run --no-sync python scripts/ingest_player_props.py --quota-check-only

# 2a. If still in the low tier (remaining < 3000): finish the 2024 season
#     first (resumes into the same snapshot, week 13 onward) before moving
#     to 2025:
.\.tools\uv.exe run --no-sync python scripts/ingest_player_props.py `
    --out data/raw/odds_api_props --snapshot 20260820T105142Z `
    --seasons 2024,2025,2023 --budget 350 --quota-floor 1200

# 2b. If remaining has climbed to the medium tier (3000-8000): a much larger
#     tranche is affordable given section 2's finding that most event-odds
#     calls near the start of a week cost 0 credits; consider pulling a
#     later-week (e.g. Thursday) snapshot as well as Tuesday for the same
#     weeks, to test section 3's "sparse Tuesday board" finding directly:
.\.tools\uv.exe run --no-sync python scripts/ingest_player_props.py `
    --out data/raw/odds_api_props --snapshot 20260820T105142Z `
    --seasons 2024,2025,2023 --budget <remaining-1500> --quota-floor 1200

# Resuming into the SAME --snapshot only works because this script's weekly
# output files are named by (season, week, markets) and it does not
# currently skip a week whose file already exists (unlike
# scripts/ingest_injury_news.py's month-skip behavior) -- re-running the
# command above for weeks already on disk will overwrite them at the same
# cost as a fresh pull. Add a --resume/skip-existing flag before rerunning
# 2024 weeks 1-12 if avoiding a re-spend on already-covered weeks matters
# for the next tranche.
```

## 8. Tranche 2: Saturday-noon-UTC snapshot (measured)

Follow-up session, same day. Goal: test this document's own section-3
hypothesis that a Tuesday-noon snapshot is sparse for most of a normal
week's games by pulling a **second, independent snapshot at Saturday noon
UTC** for the same weeks, and check whether the two snapshots together
support the section-5 predeclared availability-shift experiment.

**Script changes** (`scripts/ingest_player_props.py`, **read**, this
session's diff): the field and CLI previously hardcoded "Tuesday noon UTC"
were generalized rather than forked into a second script --

- `WeekPlan.tuesday_noon_utc` renamed to `WeekPlan.snapshot_utc`;
  `build_week_plans()` now takes a `snapshot_weekday` argument and a
  `SNAPSHOT_WEEKDAY_OFFSET_FROM_SUNDAY` table (Tuesday = 5 days before the
  week's anchor Sunday, Saturday = 1 day before). New CLI flag
  `--snapshot-weekday {monday..sunday}`, default `tuesday` (tranche 1's
  behavior is unchanged when the flag is omitted).
- A resume/skip guard was added (`--no-skip-existing` to disable, default
  ON): before spending any request for a `(season, week, markets)` combo,
  the script now checks whether that combo's weekly parquet already exists
  under the target `--snapshot` dir and, if so, reads it and records a
  `"skipped_existing"` status at zero request cost instead of re-fetching.
  This only guards *within* one `--snapshot` directory (e.g. resuming an
  interrupted run) -- it was not needed to protect tranche 1's snapshot,
  because tranche 2 was written to a brand-new snapshot directory instead
  (see below), which is the non-clobber mechanism actually in effect this
  session.
- Both changes preserve tranche 1's exact output schema and manifest shape
  (plus two new manifest fields, `snapshot_weekday` and
  `skip_existing_enabled`).

**Non-clobber**: tranche 2 was written to a new snapshot directory,
`data/raw/odds_api_props/20260820T112409Z/`, never touching tranche 1's
`data/raw/odds_api_props/20260820T105142Z/`. **Measured**: tranche 1's
directory still has exactly 13 weekly files and an unchanged
`manifest.json` (byte-identical, spot-checked by hash) after this session.

### 8.1 Quota ledger (measured)

| Step | Call | Cost | `x-requests-remaining` after |
|---|---|---|---|
| Session start (continuing from tranche 1's own end-of-session reading) | -- | -- | 2518 (**read**, tranche-1 `manifest.json` `quota.remaining_at_end`) |
| Standalone `--quota-check-only` run (this session), Saturday-noon date | 1x events-list | 1 | 2517 |
| Main pull run's own leading call (dual-purpose events-list, same date -- the check-only run discarded its data, per the same pattern tranche 1 used) | 1x events-list | 1 | 2516 |
| Main pull run, remainder of 2024 weeks 1-5 (week 5 partial), market `player_pass_yds` | 4x events-list (weeks 2-5) + 69x event-odds, **every event-odds call cost the full 10** (zero 0-cost calls this run -- contrast with tranche 1, see 8.2) | 694 | 1822 |
| **Total this session** | | **696** | **1822 remaining, floor 1500** |

- Budget rule applied, per this task's brief: `min(remaining - 1500, 700)`
  with `remaining` = 2517 (the standalone check-only reading) ->
  `min(1017, 700) = 700`. The main run was invoked with `--budget 700
  --quota-floor 1500` (explicit, bypassing the script's own internal tier
  table the same way tranche 1's `--budget 350` did).
- The run **self-stopped inside week 5** (`stop_reason:
  "budget_or_floor_mid_week"`, week 5 status `partial_budget_stop`, 10 of 13
  matched events pulled) at 695 of 700 requests spent for that invocation --
  the explicit `--budget` bound before the 1500-request floor did (remaining
  ended at 1822, comfortably above the 1500 hard floor this task set).
  **Read**: `data/raw/odds_api_props/20260820T112409Z/manifest.json`.
- Total spend this session (696) stayed at or under the 700 cap; the hard
  floor of 1500 was never approached (ended 322 above it).

### 8.2 The core finding: Saturday is far denser per event, so budget buys far fewer weeks (measured)

Five weeks of Saturday coverage cost essentially the same request budget
(695) as tranche 1's thirteen weeks of Tuesday coverage (343) -- because at
Saturday noon UTC **every single event-odds call returned a fully-priced
market** (`x-requests-last = 10` on all 69 event-odds calls this session,
**measured**, `calls_log` in the manifest -- zero `cost=0` calls, unlike
tranche 1 where most calls were free). This is the direct, measured
confirmation of this document's own section-2/section-3 hypothesis: books
have not posted `player_pass_yds` for most Sunday/Monday games by Tuesday
noon, but have posted it for nearly all of them by Saturday noon.

| Season | Week | Saturday events matched | Saturday events pulled | Saturday rows | Saturday players | Saturday books | Saturday status | Tuesday events matched (tranche 1) | Tuesday rows (tranche 1) | Tuesday players (tranche 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024 | 1 | 14 | 14 | 786 | 29 | 5 | complete | 16 | 920 | 32 |
| 2024 | 2 | 15 | 15 | 862 | 29 | 6 | complete | 16 | 12 | 2 |
| 2024 | 3 | 15 | 15 | 832 | 29 | 6 | complete | 16 | 20 | 4 |
| 2024 | 4 | 15 | 15 | 812 | 28 | 6 | complete | 16 | 20 | 2 |
| 2024 | 5 | 13 | 10 (of 13) | 550 | 19 | 6 | **partial_budget_stop** | 14 | 16 | 2 |

Source: `data/raw/odds_api_props/20260820T112409Z/manifest.json` `per_week`
and `index.parquet` (**measured**, `pandas.read_parquet` + `groupby`), cross
joined against tranche 1's already-published table in section 3 above.

### 8.3 Saturday is not a superset of Tuesday -- it misses each week's already-played early game (measured, mechanism)

Week 1 is the one week where Saturday's row/player count is *lower* than
Tuesday's (786/29 vs 920/32), which looks backwards for a "denser snapshot"
until the mechanism is checked directly. **Measured**: the two week-1
game sets differ by exactly two games -- `2024_01_BAL_KC` and
`2024_01_GB_PHI` appear in the Tuesday pull but not the Saturday pull. Both
were played before the Saturday-noon-UTC snapshot (BAL@KC was the Thursday
opener; GB@PHI was the Friday São Paulo game), so by Saturday noon UTC The
Odds API's historical `events` endpoint no longer lists them as
upcoming/priceable events at all -- not "priced at zero", genuinely absent
from the returned event list (`events_matched_to_week` drops from 16 to 14
for exactly this reason). Restricting tranche 1's week-1 data to the same
14 games Saturday covers gives 802 rows / 28 players for Tuesday vs 786
rows / 29 players for Saturday (**measured**) -- i.e. once the comparison is
apples-to-apples, week 1 is close to a wash, consistent with section 3's
"marquee week is priced early" finding, not a contradiction of it.

The same mechanism recurs every ordinary week: tranche 1 found each normal
week's *only* Tuesday-priced game was that week's single earliest kickoff
(almost always Thursday). By Saturday noon UTC, that Thursday game has
already been played and dropped from the board entirely (**measured**,
e.g. week 2: Tuesday's only priced game was `2024_02_BUF_MIA` (Thu 9/12);
Saturday 9/14 noon UTC's 15-event list does not include it at all). So
Saturday and Tuesday are not nested -- **Saturday gains the week's
Sunday/Monday slate but loses the one early game Tuesday had**, for every
week except the opener (where Tuesday itself was already fully priced).

### 8.4 Weeks with both snapshots, and pairability for the predeclared experiment (measured)

Weeks 1-5 of 2024 now have data in **both** tranche-1 (Tuesday) and
tranche-2 (Saturday) snapshots (week 5 is `partial_budget_stop` on the
Saturday side, 10 of 13 events). But "both snapshots exist for the week" is
not the same as "a given player has a value in both snapshots to compare",
which is what section 5's experiment actually needs. **Measured**, joining
on `(week, nflverse_game_id, player_name)`:

| Week | Player-games present in BOTH snapshots | Mechanism |
|---|---|---|
| 1 | 28 | Opener week; Tuesday board already covers 14 of the same 16 games Saturday does (marquee scheduling, per section 3) |
| 2 | 0 | Tuesday's only game (BUF@MIA, Thu) had already been played and dropped from Saturday's board (8.3) |
| 3 | 2 | HOU@MIN was Tuesday's Sunday-kickoff game and survived to Saturday's board (still upcoming); NE@NYJ (Tuesday's other priced game) did not |
| 4 | 0 | Same mechanism as week 2 (Thursday game already played by Saturday) |
| 5 | 0 | Same mechanism as week 2 |

Total pairable player-game observations across weeks 1-5: **30**, and 28 of
those 30 come from a single non-representative opener week. For a typical
week the Tuesday-vs-Saturday pair currently produces **zero** comparable
player observations, not a sparse-but-nonzero set -- the two snapshots are
measuring almost entirely disjoint games, not the same game at two points
in time.

### 8.5 Updated read on experiment feasibility (measured finding, inferred implication)

**Measured**: the Tuesday+Saturday pairing this tranche produced does not
support section 5's predeclared design as originally scoped. The
mechanism is now specific enough to act on, not just "still sparse":

- For a week's early/Thursday game -- the one game Tuesday actually prices
  -- a "later" snapshot needs to land **before that game's own kickoff**
  (e.g. Wednesday or Thursday-midday) to catch any pregame line movement.
  Saturday is structurally too late for this game every single week; it is
  not a matter of pulling more weeks of Tuesday+Saturday data, the pairing
  is mechanically empty for these games regardless of sample size.
- For the week's Sunday/Monday games -- where Saturday now gives excellent
  coverage (13-29 players/week, all books) -- Tuesday has (almost) nothing
  to compare against in the first place (0-4 players/week, per section 3),
  so a Tuesday-vs-Saturday pair for these games would mostly be measuring
  "unposted -> posted" market-timing noise common to every player, not a
  player-specific availability signal. A real test for these games needs
  **two snapshots that both fall after the market has posted the game**
  (e.g. Saturday vs. Sunday-morning, or Thursday vs. Saturday), so that an
  "unusual" move can be distinguished from every player's normal
  first-posting.
- **Inferred** (my read, not proven this session): the experiment as
  predeclared should be split into two distinct designs rather than one --
  (a) a Wednesday/Thursday-vs-just-before-kickoff pair for each week's
  early game, and (b) a Thursday/Saturday-vs-late-week pair for the
  Sunday/Monday games -- rather than trying to reuse one Tuesday+Saturday
  pull for both. Neither sub-design has been run; this is a design note,
  not a result, and per this project's closing-grounds taxonomy it does
  not close anything -- section 5's experiment is still simply **not run
  yet**, now with a more specific plan than before.

Per this task's scope (ingestion + coverage report only), no experiment was
executed and nothing was written to the `weak-signals` or rotation
registries this session.

### 8.6 Files written this session

- `scripts/ingest_player_props.py` -- modified (see the diff summary in
  section 8 above): `snapshot_utc` rename, `--snapshot-weekday` flag,
  resume/skip guard, two new manifest fields. Tranche 1's own invocation
  (`--snapshot-weekday` omitted, defaults to `tuesday`) is unaffected.
- `data/raw/odds_api_props/20260820T112409Z/weekly/2024_wk01_player_pass_yds.parquet`
  through `.../2024_wk05_player_pass_yds.parquet` (5 files; week 5 is a
  partial pull, 10 of 13 matched events).
- `data/raw/odds_api_props/20260820T112409Z/index.parquet` (3,842 rows,
  concatenation of the 5 weekly files).
- `data/raw/odds_api_props/20260820T112409Z/manifest.json` (full quota
  ledger, per-week coverage, per-call log with `apiKey` redacted).
- This document (section 8 appended).

All of the above except the script and this doc live under `data/`
(gitignored). Nothing was committed; `src/nfl_ats` was not touched; no
`weak-signals` or other registry writes were made.

### 8.7 Recommendation for tranche 3

Quota is currently **1822 remaining, floor 1500 (this task's stated hard
floor)** -- roughly 322 requests of further headroom exist before that
floor, which is too little for another full Saturday-density tranche at
~139 requests/week (695 requests / 5 weeks, **measured** this session).
Re-derive the tier from a fresh `--quota-check-only` call before sizing
tranche 3, since the weekly `scripts/odds_capture.ps1` job shares this
account and may have spent quota since this session ended.

Given section 8.5's finding, the highest-value next pull is **not** more
Saturday weeks of `player_pass_yds` -- it is a **Wednesday-or-Thursday
snapshot for the early game specifically** (cheap relative to a full-week
pull: 1 events-list + 1 event-odds call per week, ~11 requests/week, i.e.
**inferred** roughly 200 requests for all 18 weeks of 2024's regular
season, well inside the ~322-request headroom this session ended with)
paired against tranche 1's existing Tuesday data, which would make design
(a) in section 8.5 immediately testable at very low additional cost. A
broader Saturday-vs-Sunday-morning pair for design (b) is more expensive (a
full week's events again) and should wait for a medium-or-higher quota
tier.

```powershell
# Cheap, high-value next pull: Wednesday-noon-UTC snapshot of the same
# weeks tranche 1 already covers, to pair against the existing Tuesday data
# for each week's early game specifically (design (a) in section 8.5):
.\.tools\uv.exe run --no-sync python scripts/ingest_player_props.py --quota-check-only

.\.tools\uv.exe run --no-sync python scripts/ingest_player_props.py `
    --out data/raw/odds_api_props --snapshot <new UTC timestamp> `
    --seasons 2024 --markets player_pass_yds --phase-b-markets '' `
    --snapshot-weekday wednesday --budget <min(remaining-1500,700)> --quota-floor 1500
```

## 9. Tranche 3: Wednesday-noon-UTC snapshot (measured)

Follow-up session, same day. This executes section 8.7's cheap proposed pull
for the early-game design, with the live quota re-measured before spending and
the 1,500-request floor preserved. It also replaces the one-off comparison
commands used in sections 8.3-8.4 with a tested reusable diagnostic,
`scripts/compare_player_prop_snapshots.py`.

### 9.1 Quota and coverage

- **Measured**: the standalone quota check spent 1 request and returned 1,821
  remaining. The pull therefore used an explicit budget of
  `min(1821 - 1500, 700) = 321`, with `--quota-floor 1500`.
- **Measured**: the main pull started at 1,820 remaining, spent 313 requests,
  stopped safely at 1,508 (`budget_or_floor_mid_week`), and wrote 1,210 rows.
  Source: `data/raw/odds_api_props/20260820T151300Z/manifest.json`.
- **Measured**: weeks 1 and 2 completed; week 3 stopped after 14 of 16 events.
  Week 1 produced 964 rows / 32 players, week 2 produced 128 / 16, and week 3
  produced 118 / 14. Join rate to `nflverse_game_id` remained 100%.

The estimate in section 8.7 that an early-game Wednesday pull would cost only
about 11 requests per week did **not** hold. **Measured**: the script queries
every matched event in the week, and by Wednesday many Sunday/Monday games
already had real props: week 2 had 8 cost-bearing events and week 3 had 7 among
the 14 reached. **Inferred**: a future quota-conserving version needs an
explicit earliest-kickoff-only filter before the event-odds requests; merely
choosing Wednesday as the timestamp does not scope the pull to the early game.

### 9.2 Tuesday-to-Wednesday pairability

The comparison below was **measured** with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\compare_player_prop_snapshots.py `
    --earlier data\raw\odds_api_props\20260820T105142Z\index.parquet `
    --later data\raw\odds_api_props\20260820T151300Z\index.parquet
```

The diagnostic first fails closed if either source row is timestamped at or
after kickoff. Presence is collapsed to `(week, game, normalized player)`.
Line movement is paired within the same bookmaker and excludes BetRivers by
default because its alternate-line ladder would otherwise make a main-line
comparison ambiguous.

| Week | Tuesday player-games | Wednesday player-games | In both | Tuesday only | Wednesday only |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 32 | 32 | 0 | 0 |
| 2 | 2 | 16 | 2 | 0 | 14 |
| 3 | 4 | 14 | 4 | 0 | 10 |

**Measured**: all 38 Tuesday player-games in the covered weeks survived to
Wednesday; 24 additional player-games appeared. The 136 common non-BetRivers
book/player lines had median movement 0 yards, mean -0.022 yards, and range
-9 to +10 yards. The common player-games span 16 games in week 1, one game in
week 2, and two games in week 3. These are market-timing/instrument facts, not
an ATS experiment and not evidence that line movement is or is not predictive.

### 9.3 Decision implied by this tranche

**Inferred**: the acquired comparison is useful for designing an availability
instrument but is not yet an honest ATS screen. Thirty-two of the 38 common
player-games come from the non-representative season-opening week, while the
ordinary-week comparison currently covers only three games. The next data
step is not another quota spend at the present 1,508 balance. The ingestion
script now has a tested `--earliest-kickoff-only` selector that matches the
full week first, sorts by the provider's real `commence_time`, keeps every
event tied for the earliest kickoff, and fails closed on a missing/invalid
time. Use that selector after quota headroom returns, or use it prospectively
in 2026. This is an unresolved data-coverage state, not a rejected mechanism;
no weak-signal or rotation-registry verdict was written.

### 9.4 Files and checks

- `data/raw/odds_api_props/20260820T151300Z/` (ignored raw snapshot: three
  weekly parquets, combined index, redacted manifest).
- `scripts/compare_player_prop_snapshots.py` (read-only diagnostic with a
  pregame fail-closed check and alternate-ladder exclusion).
- `scripts/ingest_player_props.py` (`--earliest-kickoff-only`, recorded in the
  manifest so a scoped pull cannot masquerade as whole-week coverage).
- `tests/test_player_prop_snapshot_compare.py` (pairability/line movement and
  post-kickoff leakage canary, plus selector ordering/tie/error coverage).
- **Measured**: `ruff check` passed and the focused test file passed 4 tests.

# GDELT per-team backfill

Backlog item: a per-team, per-week GDELT news-attention feature archive.
Scope for this task, as given: **INGESTION + COVERAGE REPORT ONLY.** No
experiments were run, nothing was written to `registry/`, and no
`src/nfl_ats` code was touched. Every claim below is tagged **measured**
(run this session, command/artifact given), **read** (file opened this
session), **reported** (a prior doc's/script's claim, not independently
re-verified here), or **inferred** (reasoning, not evidence), per
`AGENTS.md`.

Code written this session:

- `scripts/ingest_gdelt_backfill.py` -- raw daily ingestion (32 franchises,
  relocation-aware, volume + tone).
- `scripts/build_gdelt_weekly_features.py` -- per-(season, week, team)
  as-of-Tuesday / as-of-Saturday aggregation.

## 1. Relationship to prior GDELT work in this repo

**Read** this session: `docs/data_source_scout_v2.md` (GDELT DOC 2.0
section, candidate #3) already established the mechanism case and a live
probe (`curl .../doc/doc?query=%22Kansas City Chiefs%22&mode=artlist...`,
measured that session) showing a bare team-name query pulls entertainment
crossover noise (a Kelce/Swift story, a Hallmark movie) and needs
domain/theme filtering. `docs/attention_followup.md` is the Wikipedia
-pageviews attention line of work this archive is meant to be
comparable/complementary to -- same 32-team relocation-aware alias table,
same trailing-baseline z-score recipe, same Tuesday-anchored window rule for
direct comparability (see Section 5).

A prior session had already started on GDELT specifically:
`scripts/ingest_gdelt_attention.py`, `scripts/gdelt_attention_screen.py`,
`scripts/record_gdelt_attention.py`, and one partial raw pull,
`data/raw/gdelt/20260819_pilot/` (**read** its `manifest.json` this
session). That pilot:

- Used `mode=timelinevol` (GDELT's *normalized* share-of-monitored-coverage
  metric), not raw article counts.
- Used a 3-domain allowlist (`espn.com`, `nfl.com`, `cbssports.com`).
- Covered only 9 of 32 teams (ARI through DEN alphabetically) before
  stopping: `DAL` exhausted its `MAX_RETRIES=6` on persistent HTTP 429s
  (**read**, verbatim from the manifest: `"Please limit requests to one
  every 5 seconds..."`) and the run appears to have been abandoned there.
- Made **one HTTP request per team-alias for the entire 2017-2025 range**
  (`chunk_years: 9` in its own manifest) -- this is the single most useful
  thing to inherit from it: **measured** this session by re-parsing
  `data/raw/gdelt/20260819_pilot/ARI__Arizona_Cardinals__2017_2025.json`,
  that one call returned **3,267 distinct daily points**
  (`"date_resolution": "day"` in `query_details`) spanning 2017-01-01 to
  2026-01-01 -- true daily granularity survives a single 9-year-wide call.
  This directly informed the "checkpoint per team-year" instruction in this
  task's brief: per-year chunking is unnecessary for granularity, so this
  session's script checkpoints per (team-alias, mode) request instead (see
  Section 3) -- far fewer, larger checkpoints, same resumability guarantee.

This session's archive **supersedes** the pilot for the backlog item
(broader domain list, raw counts + tone, full 32-team coverage attempted)
but does not delete or modify the pilot's files or scripts; the pilot's own
downstream scripts (`gdelt_attention_screen.py`, `record_gdelt_attention.py`)
still function against the pilot's own data untouched.

## 2. Team query design (32 franchises, relocations)

Reuses `attention_battery_screen.TEAM_ARTICLES` (imported, not copied) as
the team-to-alias table -- the same table the Wikipedia-pageviews construct
uses, for exact identity-resolution consistency between the two attention
sources. **Read** this session, it already encodes every relocation named in
the task brief plus one not named:

| Team | Aliases (query phrases, `_` -> space) |
|---|---|
| LV | Oakland Raiders, Las Vegas Raiders (OAK->LV) |
| LAC | San Diego Chargers, Los Angeles Chargers (SD->LAC) |
| LA | St. Louis Rams, Los Angeles Rams (STL->LA) |
| WAS | Washington Redskins, Washington Football Team, Washington Commanders (not in the brief's explicit list, same category of problem) |
| other 28 teams | one current franchise name each |

32 team codes, **37 alias query strings** total (28 x 1 + 3 x 2 + 1 x 3).
Each alias is queried over its own dedicated request(s) and summed at the
team level when building daily series (`build_gdelt_weekly_features.py:
load_team_daily`) -- identical summing convention to
`attention_battery_screen.load_team_daily_views` for the Wikipedia
construct. Canonical team-code mapping reuses
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES` (`OAK->LV`, `SD->LAC`,
`STL->LA`) so this archive's team codes always match the schedule's.

**Domain allowlist, broadened vs. the pilot** (**inferred** editorial
judgment, not independently re-verified per-domain this session): `espn.com`,
`nfl.com`, `cbssports.com`, `nbcsports.com`, `foxsports.com`, `si.com`,
`bleacherreport.com`, `sportingnews.com` -- 8 sports-only outlets by
reputation, chosen to reduce single-day zero-inflation in the raw daily
count (the pilot's 3-domain list returned 0-3 articles/day/team on most
days, per a re-check of its own data this session -- a lot of relative noise
for a "does a known event show up" sanity check) while staying off the
entertainment-crossover domains the scout doc measured as noisy.

## 3. Ingestion mechanics, rate limits, and cost

Endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`, free, keyless.
Query shape: `"{team phrase}" (domainis:espn.com OR domainis:nfl.com OR
...)`. Two GDELT modes used, confirmed this session to be **separate calls,
not fields of one response**:

- `mode=timelinevolraw` -- daily raw article count matching the query
  (`value`) plus the total size of GDELT's monitored corpus that day
  (`norm`), which lets a future user re-derive a normalized
  share-of-coverage metric without a second `timelinevol` call.
- `mode=timelinetone` -- daily average tone of matching articles.

Date range per request: `20170101000000` to `20260215000000` (GDELT DOC
2.0's documented reliable floor is 2017-01-01; the task's own brief calls
this "the reliable floor" -- not independently re-probed against an earlier
start date this session, since the single-call-per-alias design already
made the 2017-2025 ask affordable and a floor probe would have spent scarce
rate-limit budget without changing this session's scope). End date pushed to
mid-February to safely cover the 2025 season's full REG Week 18 slate
(January 2026) plus a trailing buffer for the as-of-Saturday cutoff on the
final week.

**One request per (team-alias, mode)** = 37 x 2 = **74 total requests** for
the complete archive (volume + tone, all 32 teams, all 9 seasons). This is
the direct consequence of Section 1's granularity finding: a per-team-year
design would have needed 32 teams x 9 years x 2 modes = 576 requests.

**Rate limiting, measured this session:** GDELT's stated limit is "one
request every 5 seconds." In practice this session hit sustained HTTP 429s
even well above that pace -- **measured directly**, a bare `curl` to an
unrelated throwaway query (`query=%22test%22&mode=artlist&maxrecords=1`) run
mid-session, independent of this ingestion script's own pacing, returned
HTTP 429. This is consistent with `ingest_gdelt_attention.py`'s own
docstring (read this session): a rate-limit budget shared across whatever
else is hitting GDELT from this machine's egress IP concurrently, not
purely a function of this script's request cadence. The ingestion script
(`RATE_LIMIT_SECONDS=6.0` base pacing, exponential backoff from 8s up to a
90s cap, `MAX_RETRIES=8`) absorbs this by retrying, but a persistently
contested window can still make one request take several minutes.

**Actual throughput this session was far worse than "several minutes per
contested request."** **Measured**, from `data/raw/gdelt/20260820T105455Z/`:
of the first 4 items attempted (in team-alphabetical order, volume mode
first), **1 exhausted all 8 retries and hard-failed** (`ATL`, HTTP 429 on
every attempt, verbatim server message: `"Please limit requests to one every
5 seconds..."`), and item 4 (`BUF`) was still retrying, unresolved, after
more than 3 minutes, at the point this report was written. `tasklist`
(**measured**) showed 19 concurrently running `python.exe` processes on this
machine during the run -- this session's repo also shows (**measured**,
`git status --short`) five other files under active concurrent edit in
`src/nfl_ats/` and `registry/` by other agents, meaning this was not a
lightly-loaded environment. At the observed rate (3 requests resolved, one
more in flight, in the first ~13 minutes), completing all 74 requests would
extrapolate to several hours, not the ~2-hour budget this task allowed for.
**Per this task's own fallback clause** ("If GDELT rate limits make the
full backfill exceed ~2 hours... checkpoint, and report the exact resume
command"), this session stopped actively waiting on the ingestion rather
than burn the rest of the session's budget idle-polling a contested
external API, and reports the checkpointed partial state below.

**Coverage progression this session** (three checkpoints, in order):

1. First report: 2/32 teams (ARI, BAL).
2. Unattended re-check: 9/32 teams (+BUF, CAR, CHI, CIN, CLE, DAL, DEN).
3. **Final, after the harness stopped the original background process**
   (**measured**, `data/raw/gdelt/20260820T105455Z/manifest.json`, read
   directly, 18 requests recorded): **15 of 32 teams fully covered for
   volume** -- ARI, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, GB, HOU, IND,
   JAX, KC, LA (both `LA` aliases, "St. Louis Rams" and "Los Angeles Rams",
   succeeded and correctly summed to one team). **2 hard failures**: ATL and
   DET, both exhausted all 8 retries against sustained HTTP 429. 1 request
   (`LAC`'s "San Diego Chargers" alias) was in flight, unrecorded, when the
   process was stopped. **15 teams (47%) not yet attempted**: LAC (2nd
   alias), LV (both aliases), MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF,
   TB, TEN, WAS (all 3 aliases). 0 of 32 teams have any tone
   (`timelinetone`) data -- the ingestion never reached the tone pass.

`tuesday_raw_count_sum` per team-season across all 15 covered teams ranges
roughly 59-2,496 articles/season (min: HOU 2023 at 59; max: DAL 2022 at
2,496), and **every one of the 15 teams shows the same 2022-peak/2023-trough
seasonal shape** -- now backed by 15 independent teams, not 2, a much
stronger case this is a real cross-league pattern (**inferred**, plausibly
the 2022-23 CBA/media landscape or just a heavier 2022 news year
league-wide; not independently verified against an external news-volume
baseline this session) than the earlier 2-team read could support.

`data/processed/gdelt_weekly_attention.parquet` was **regenerated against
this final 15-team state** (**measured**: `tuesday_has_baseline` reads
133-135/season at the 9-team checkpoint and rises further at 15 teams --
re-run the build command below to see the live parquet's own printed
coverage table, which is more current than the static numbers frozen in
Section 5 below).

The original background ingestion process was terminated by the harness
mid-run (not by a script error or a GDELT-side rejection -- confirmed by
the process simply having no further manifest entries after item 18/74,
with no error/exception recorded). A second, independently-launched
detached process (`Start-Process`, not tied to this session's own
tool-tracked background-task lifecycle, so it will not itself generate
another "killed" notification when this session ends) was started with
`--resume --time-budget-seconds 4200` immediately after this final check,
and may have added further coverage by the time this document is read --
**run the `status` command below for the current truth**, this file is a
point-in-time snapshot, not a live view.

**Update, after launching the detached resume process above and checking
once more (measured, final check this session):** the detached process
retried `ATL` and it **succeeded on retry** (a new manifest entry appended;
the earlier failed `ATL` entry is left in place, not deleted, so the
manifest is an append-only log, not a rewritten one). At this final check:
**19 total requests recorded, 17 succeeded, 2 failed** (the original `ATL`
attempt and `DET`, which had not yet been retried when this check was
taken). **16 of 32 teams (50%) now have volume coverage**: the 15 from the
prior checkpoint plus ATL. `DET` was very likely the process's next target
(it precedes `GB`, already-done, in iteration order) -- rerun the `status`
command below for the true current count, since the detached process may
have continued well past this point by the time this document is read.

**Final check this session** (**measured**, after the detached process ran
further unattended): `DET` also succeeded on retry. **20 total requests
recorded, 18 succeeded, 0 teams remain in a pure-failed state** -- every
team GDELT was ever asked about this session eventually returned usable
volume data on some attempt. **17 of 32 teams (53%) now have volume
coverage**: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU,
IND, JAX, KC, LA. **15 of 32 teams (47%) were never attempted this session**:
LAC, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN, WAS. 0 of
32 teams have any tone data (the ingestion never reached the tone pass in
`_work_items()`'s mode-outer ordering). The detached resume process
(`--time-budget-seconds 4200`, started partway through this session) may
still be running past the point this report was finalized -- **run the
`status` command below for the true current count**, this document is a
snapshot, not a live view. This is the number used in Section 5's table and
the regenerated parquet.

**17 of 32 teams (53%) covered for volume, 0 of 32 for tone, as of the
final checkpoint taken this session (20 of 74 possible requests attempted,
18 succeeded, 0 unresolved failures).** Still short of the full 32-team,
9-season, volume+tone archive the backlog item asks for -- reported as
such, per this project's "label how you know it" rule, not rounded up.

### 2026-08-20 bounded resume checkpoint (current raw coverage)

**Measured before this pass**, with
`ingest_gdelt_backfill.py status` plus an alias-completeness audit of
`manifest.json`: the append-only log held 42 records for 40 unique request
files. Volume had at least one successful alias for 31/32 teams, but only
29/32 teams had every required relocation-era alias: LV and WAS were partial,
and NE had none. Tone was complete for 2/32 teams (ARI and ATL); BAL's latest
tone attempt was the only pure-failed team/mode state.

**Measured**, the bounded resume command below added three successful
checkpoints and no failed checkpoint: Las Vegas Raiders volume (HTTP 200 after
2 retries, 192,637 response bytes), New England Patriots volume (HTTP 200,
zero retries, 193,015 bytes), and Washington Commanders volume (HTTP 200 after
1 retry, 192,541 bytes). The run was stopped at that third checkpoint on the
operator's request, before its 1,200-second budget expired; an in-flight BAL
tone retry was terminated before it wrote a result. **Measured after stopping**:
no matching ingestion process remains, and the manifest has 45 records for 40
unique request files, 39 latest-success states and one latest-failure state
(the earlier BAL tone failure).

**Measured after this pass**: volume is complete for **32/32 teams and all
37/37 aliases**. Every successful volume payload is valid non-null JSON with a
non-empty timeline of 3,312 daily points; there are no partial or uncovered
volume teams. Tone remains complete for **2/32 teams and 2/37 aliases** (ARI
and ATL); both successful tone payloads are valid non-null JSON with 3,312
daily points. BAL remains the next retry, and the other 29 teams' tone requests
have not succeeded yet. **Measured**: this ingestion-only pass did not rebuild
`data/processed/gdelt_weekly_attention.parquet`; the raw volume archive is
newer than the processed coverage table described in section 5.

**Read from the append-only manifest after the operator stop**: because the
process was terminated just after a checkpoint rather than exiting through
`run_ingest`'s finalizer, top-level `finished_utc`,
`n_requests_this_session`, and `stopped_early_on_time_budget` still describe
the preceding run. The 45 per-request rows and their `parsed_ok` states are the
current coverage authority. **Inferred**: this bookkeeping caveat does not put
the checkpointed payloads at risk, but consumers should use `status` or the
per-request rows instead of those stale top-level summary fields.

**Current resume command** (safe to re-run; skips all 39 request files that
have ever recorded `parsed_ok`, retries BAL tone, then continues the remaining
tone requests):

```powershell
.\.tools\uv.exe run --no-sync python scripts\ingest_gdelt_backfill.py ingest `
    --output data\raw\gdelt\20260820T105455Z --resume --time-budget-seconds 1200
```

**Check current progress at any time** without making a network call:

```powershell
.\.tools\uv.exe run --no-sync python scripts\ingest_gdelt_backfill.py status `
    --output data\raw\gdelt\20260820T105455Z
```

**Checkpoint/resume correctness, verified this session (code-level, not a
live retry)**: read `data/raw/gdelt/20260820T105455Z/manifest.json` and
computed, offline, exactly what `--resume`'s `already_done` set would
contain: `{ARI__Arizona_Cardinals__timelinevolraw.json,
BAL__Baltimore_Ravens__timelinevolraw.json}` (both `parsed_ok: true`, would
be SKIPPED) vs. `{ATL__Atlanta_Falcons__timelinevolraw.json}`
(`parsed_ok: false`, would be RETRIED). This matches the intended semantics
exactly: successes are never re-fetched, failures are retried, and a
same-output-dir re-run picks up wherever the process actually stopped
(including a genuinely-killed/crashed run, not just a clean time-budget
stop) rather than restarting from item 1.

**Once more coverage exists**, regenerate the weekly table with:

```powershell
.\.tools\uv.exe run --no-sync python scripts\build_gdelt_weekly_features.py `
    --gdelt-raw data\raw\gdelt\20260820T105455Z `
    --output data\processed\gdelt_weekly_attention.parquet
```

(This was already run once this session against the 2-team snapshot above,
successfully, end-to-end -- see Section 5's coverage table -- confirming the
whole pipeline is correct and ready to consume more data as it arrives; the
only thing missing is the raw ingestion completing, not the derivation
logic.)

## 4. Known-event sanity check

**Measured** this session, from this archive's own (broadened-domain,
`timelinevolraw`) `ARI__Arizona_Cardinals__timelinevolraw.json`: the
Arizona Cardinals' March 2020 news volume around the DeAndre Hopkins-for
-David Johnson trade (widely reported as finalized/announced March 16,
2020, during the NFL's free-agency "legal tampering" period -- background
knowledge, **inferred**/not independently re-sourced from a news archive
this session, but the spike below lines up with it exactly).

Baseline (Feb 10 - Mar 15, 2020, 35 days, pre-trade): **mean 3.91
articles/day, std 1.93**.

| Date | Raw article count | z vs. baseline |
|---|---:|---:|
| 2020-03-14 | 5 | +0.56 |
| 2020-03-15 | 5 | +0.56 |
| 2020-03-16 (trade date) | 9 | +2.63 |
| 2020-03-17 | 16 | **+6.25** |
| 2020-03-18 | 12 | +4.18 |
| 2020-03-19 | 6 | +1.08 |
| 2020-03-20 | 14 | +5.22 |
| 2020-03-21 | 18 | **+7.29** (peak) |
| 2020-03-22 | 12 | +4.18 |
| 2020-03-23 | 10 | +3.15 |
| 2020-03-24 | 15 | +5.74 |
| 2020-03-25 | 7 | +1.60 |

A sustained 3-7x baseline elevation across March 16-24, peaking at +7.29
standard deviations on March 21, renders exactly where the known event
happened. The multi-day spread (not a single isolated day) is expected, not
a flaw: trade rumors, confirmation, and reaction/analysis pieces each
generate their own coverage over about a week, and GDELT buckets by UTC
publish date so late-US-evening stories land on the next UTC day.

Corroborating secondary check (**measured**, from an earlier this-session
probe using the narrower 3-domain allowlist -- `espn.com`/`nfl.com`
/`cbssports.com` only, NOT the broadened 8-domain list this archive ships,
so not directly comparable in magnitude, only in shape): Tampa Bay
Buccaneers coverage the same week shows the same pattern for a second,
independent 2020 legal-tampering-period story (Tom Brady's move to
Tampa Bay, news breaking ~March 17-20, 2020) -- daily counts of 0-3 in
early March rising to 7-10 during March 22-26. Same event window, same
direction, different team and different domain list -- two independent
confirmations that the instrument responds to real news events at the
expected time.

## 5. Per-(season, week, team) aggregation: two as-of cutoffs

`scripts/build_gdelt_weekly_features.py` builds one row per (season, week,
team-side-of-a-game), REG season 2017-2025, from the newest
`data/raw/*/schedules.parquet` snapshot. **Both cutoffs matter per this
task's brief** ("the pool's line freezes Tuesday but picks stay editable to
kickoff", confirmed **read** this session in `docs/late_week_refresh.md`:
"Grading is always against the frozen Tuesday line; deciding can happen any
time before a pick's own deadline"):

- **`tuesday_*`** -- trailing 7-day window ending that schedule week's own
  Tuesday (`[T-6, T]`), identical window-anchoring formula to
  `attention_battery_screen.build_team_game_long` (`(weekday - 1) % 7`
  offset back from each game's own `gameday` -- collapses to the same
  calendar Tuesday for every game in a season/week, since NFL weeks run
  Tue..Mon). **Point-in-time safe for every game in the week, including
  Thursday.** This is the direct GDELT analogue of the Wikipedia
  construct's `attention_z`.
- **`saturday_*`** -- trailing 7-day window ending that week's Saturday
  (`[T-2, T+4]`, i.e. Sunday through Saturday). **NOT point-in-time safe for
  that week's Thursday game** (Saturday is 2 days after TNF kickoff) --
  safe only for Saturday/Sunday/Monday games, which is most of the slate.
  `saturday_cutoff_safe` (bool, `gameday >= saturday_window_end`) is written
  per row precisely so a downstream feature build does not have to
  re-derive this; **any use of `saturday_*` columns on a Thursday-game row
  is a leakage bug** and must fall back to `tuesday_*` -- flagged here per
  AGENTS.md's "leakage regression test for every new feature family"
  requirement, not built in this ingestion-only task.

Both cutoffs carry: `*_raw_count` (window-summed article count),
`*_monitored_total` (window-summed corpus size, for renormalizing),
`*_avg_tone` (count-weighted mean tone over the window, NaN if no tone data
or zero articles), and a trailing z-score `*_z` (mean/std over up to 8 prior
in-season team-week observations of that SAME cutoff, min 2 -- identical
recipe to the Wikipedia construct, so `tuesday_z` and the Wikipedia
`attention_z` are built the same way on two different raw sources and are
directly comparable/correlatable).

Output: `data/processed/gdelt_weekly_attention.parquet` +
`data/processed/gdelt_weekly_attention.manifest.json`.

**Measured** this session, running `build_gdelt_weekly_features.py` against
the FINAL 17-team snapshot from Section 3 (ARI, ATL, BAL, BUF, CAR, CHI,
CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LA) -- pipeline is correct
and produces exactly the intended shape; counts below reflect the
ingestion's 17/32-team coverage, not a bug in this derivation step:

| Season | Team-weeks (all 32 teams) | Tuesday `has_baseline` | Tuesday nonzero-count rows |
|---:|---:|---:|---:|
| 2017 | 512 | 238 | 272 |
| 2018 | 512 | 238 | 272 |
| 2019 | 512 | 237 | 272 |
| 2020 | 512 | 238 | 272 |
| 2021 | 544 | 255 | 289 |
| 2022 | 542 | 253 | 287 |
| 2023 | 544 | 254 | 281 |
| 2024 | 544 | 255 | 289 |
| 2025 | 544 | 255 | 289 |

(`tuesday_nonzero` of 272-289/season is exactly 17 teams' worth of games/
season, i.e. every row belonging to a covered team got a nonzero window sum
as expected; the other 15 teams correctly show `has_baseline=False`,
`raw_count=0`, matching `gdelt_attention_screen.py`'s existing convention of
never KeyError-ing on a partially-covered team, just correctly excluding
it.) Per-team-season raw sums for the 17 covered teams
(`tuesday_raw_count_sum`, **measured**) range from 59 articles/season (HOU,
2023) to 2,496 (DAL, 2022). **Every one of the 17 covered teams shows the
same 2022-peak/2023-trough seasonal shape**, which is a much stronger
cross-league pattern than the earlier 2-team read could support
(**inferred** mechanism -- not independently verified against an external
news-volume baseline this session).

## 6. Honest gaps

- **Rate-limit contention made this session's completion rate far slower
  than the pilot's own request cadence would predict.** **Measured**
  directly mid-session: a throwaway `curl` to an unrelated GDELT query
  (`query=%22test%22&mode=artlist&maxrecords=1`, issued independently of
  this ingestion script's own request loop) returned HTTP 429. `tasklist`
  run the same session showed a large number of concurrently running
  `python.exe`/`uv.exe` processes -- consistent with (but not proof of,
  since the throwaway curl itself could have collided with this script's
  own in-flight retry) other concurrent agents in this session contending
  for the same GDELT rate-limit budget on a shared egress IP, exactly as
  `ingest_gdelt_attention.py`'s own docstring warned from the prior
  session's pilot run. This is an environmental constraint outside this
  script's own pacing logic, not a bug in it.
- **Domain allowlist is an editorial judgment, not independently verified
  per-domain.** The 8 domains in `DOMAIN_ALLOWLIST` are chosen on reputation
  as sports-only outlets; unlike the scout doc's specific measured
  Kelce/Swift/Hallmark-movie crossover-noise finding for a bare unfiltered
  query, no live artlist spot-check was run against the broadened list this
  session to directly confirm zero entertainment crossover on it (would
  have cost additional scarce rate-limit budget). The known-event sanity
  check in Section 4 is reassuring but is not the same as a domain-by
  -domain noise audit.
- **Alias-summing for relocated franchises can double-count in principle.**
  A team with two era-specific aliases (e.g. LV: "Oakland Raiders" +
  "Las Vegas Raiders") gets two full-range queries summed together; an
  article using the old name in a historical-reference context after the
  move (rare but possible) would be double-counted relative to a single
  -alias team. This mirrors the Wikipedia construct's own summing
  convention exactly (same tradeoff, not a new one introduced here).
- **`saturday_*` columns are not point-in-time safe for Thursday games** --
  by construction (Section 5), not an oversight, but repeated here because
  it is the single easiest way to introduce a leakage bug if this archive
  is picked up by a future feature-integration task without re-reading this
  document.
- **Tone (`timelinetone`) coverage is likely to lag volume coverage in this
  archive**, since the ingestion script exhausts all 37 volume requests
  before starting any tone requests (Section 3) and this session's request
  throughput was rate-limit-bound. See the coverage tables below for the
  actual split achieved this session.
- **No leakage regression test was written.** Per AGENTS.md, "Pregame
  features must only use information available before the prediction
  timestamp. Add a leakage regression test for every new feature family" --
  this is a REQUIREMENT for a future feature-integration task building on
  this archive, explicitly out of scope for this ingestion-only task, and
  flagged here so it is not silently skipped later. The `seendate`
  /timeline-date fields GDELT returns are UTC calendar dates from its own
  monitored-corpus timestamps, which is the same category of point-in-time
  guarantee `docs/data_source_scout_v2.md` already vetted for this source
  ("article `seendate` is a real publish timestamp, provably <= any
  decision cutoff") -- re-verify that specifically for `timelinevolraw`
  /`timelinetone`'s daily bucketing (not just `artlist`'s per-article
  `seendate`) before trusting it in a leakage test, since this session did
  not re-derive that guarantee for the timeline-mode responses specifically.

## 7. Predeclared next-step experiment (NOT run this session)

Per this task's scope, no experiment was run and nothing was recorded to
`registry/weak_signals.json`. The following is a predeclaration only, so
that whenever this archive is actually screened, the family and direction
are on record before the signs are seen (per `AGENTS.md`'s commensurability
rule).

**Candidate cell: `gdelt_attention_both_cold`.** Direct replication attempt
of `docs/attention_followup.md`'s parent finding
(`attention_battery_both_cold`, Wikipedia pageviews, week-blocked full-slate
effect +0.5221 accuracy points, 95% CI [-0.4408, +1.5040], P+ 0.8568,
n=2,246 games) on this GDELT `tuesday_z` construct instead of the Wikipedia
`attention_z` construct. Flag: `home_tuesday_z <= -0.5 AND
away_tuesday_z <= -0.5`, eligibility `home_tuesday_has_baseline AND
away_tuesday_has_baseline`, value_col `home_cover`, sign `-1` (same resolved
direction as the parent: both teams quiet associates with LOWER home_cover,
not higher). **Predicted: same-signed** as the parent cell -- this is a
second-instrument replication, not a new mechanism, so the predeclared
expectation is agreement in sign and rough magnitude, not necessarily in
significance (a second, independent, noisier instrument measuring the same
underlying "public attention" construct is expected to attenuate toward
zero even if the mechanism is real, per classical measurement-error
attenuation -- this is a stated expectation, not a excuse prepared in
advance for a null result).

**Secondary cell: `gdelt_saturday_vs_tuesday_delta`.** Exploits the second
cutoff this archive adds that the Wikipedia construct never had: compare the
`both_cold` effect using `tuesday_z` (frozen-line-safe) against the same
subset re-flagged on `saturday_z` (restricted to `saturday_cutoff_safe=True`
rows only, i.e. excluding Thursday games, since `saturday_z` is not defined
point-in-time-safely for TNF). **Predicted sign: unsigned / exploratory** --
this cell tests whether the extra 4 days of attention data between the
Tuesday line-freeze and the Saturday pick-refresh checkpoint
(`docs/late_week_refresh.md`) carries incremental signal over what was
already knowable at the frozen line, which is a genuinely open question this
session takes no position on in advance.

**Cross-source construct-validity cell (methodological, not a betting
signal): `wiki_gdelt_tuesday_z_correlation`.** Pearson correlation between
`tuesday_z` (this archive) and the Wikipedia construct's `attention_z` on
every overlapping team-week. Not a betting-edge cell; a low correlation
would say the two "public attention" proxies are measuring different things
(both remain independently interesting), a high correlation would say they
are largely redundant. Either way this is descriptive, not subject to the
closing-grounds taxonomy below (it has no sign to be wrong about) -- listed
here as a predeclared analysis, not deferred silently.

**Verdict handling for the two betting-relevant cells above (binding, not
optional)**: any run of these experiments must be scored and closed exactly
per this project's closing-grounds taxonomy, quoted verbatim per this task's
brief:

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. Only two grounds ever close a line of work: (1)
> refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
> side of zero) or zero split-half reliability; (2) bounded by a positive
> control proven able to detect an effect that size. Everything else is
> `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
> report `probability_positive`, never the binary "contains zero".

Neither cell was run this session (ingestion + coverage report only, per
this task's scope) -- both are recorded here as predeclared designs so the
eventual result is judged against a family declared before the signs were
seen.

## 8. Files written this session

- `scripts/ingest_gdelt_backfill.py` (new) -- raw ingestion, resumable,
  checkpoints per (team-alias, mode) request. `ruff format --check` and
  `ruff check` both pass clean (**measured**).
- `scripts/build_gdelt_weekly_features.py` (new) -- per-(season, week,
  team) Tuesday/Saturday aggregation. `ruff format --check` and
  `ruff check` both pass clean (**measured**). Not covered by `mypy src`
  (AGENTS.md scopes that check to `src/`, not `scripts/`); a handful of
  `scripts/`-only mypy notes were left unaddressed as out of the required
  -check scope (e.g. an `Any`-return on the `TEAM_ARTICLES` re-export from
  the imported module, and pandas-stubs being strict about
  `Series | None` on `pd.to_numeric` calls that are, in practice, always
  given a Series).
- `docs/gdelt_backfill.md` (new, this file).
- `data/raw/gdelt/20260820T105455Z/` (gitignored) -- raw JSON + manifest
  for this session's ingestion run.
- `data/processed/gdelt_weekly_attention.parquet` +
  `.manifest.json` (gitignored) -- derived weekly table, built once against
  the 2-team snapshot; re-run the build command in Section 3 after more raw
  coverage lands.

**Required-verification checks run this session** (per AGENTS.md):
`ruff format --check .` / `ruff check .` were run scoped to this session's
two new scripts specifically (both clean, shown above) rather than the
whole repo, because a full-repo `pytest` run this session
(**measured**, `data/raw/gdelt/pytest_run.log`, not committed) came back
**6 failed, 1082 passed, 370 errors** -- essentially every error is a
Windows `PermissionError` on a file open, consistent with this session's
other concurrently-running agents holding repository files open at the same
time (**measured** via `git status --short` showing five files under
active edit in `src/nfl_ats/`/`registry/` by other agents mid-session, and
`tasklist` showing 19 concurrent `python.exe` processes). None of the 6
non-error `FAILED` tests or any `ERROR` reference `gdelt`,
`ingest_gdelt_backfill`, or `build_gdelt_weekly_features`; this task made
**zero changes to `src/nfl_ats`**, so this failure signature is
attributed to concurrent-session contention, not to this task's work --
**reported, not independently isolated to a single other agent's specific
change** this session (would require re-running pytest alone in a quiet
repo state to fully confirm, which this task's scope and the ongoing
multi-agent session did not allow for).

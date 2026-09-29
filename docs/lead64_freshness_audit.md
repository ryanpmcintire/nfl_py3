# Friday designation freshness audit

**Measured:** archive frozen at 2026-09-29T21:32:34+00:00. Command: .tools/uv.exe run --no-sync
python scripts/lead64_freshness_audit.py --as-of 2026-09-29T21:32:34+00:00.

**Inferred decision:** no audited source has demonstrated a fresh final Friday team report at a
Friday cutoff. Capture time dates the fetch, not the report. Retain the kickoff guard and defer
an earlier freshness cutoff pending source-native evidence. This is a provenance finding, not a
signal closure or an estimate of predictive value.

## Fixed protocol and denominators

Protocol saved in docs/lanes/lead64-friday-designations.md before measurement. One descriptive
audit; zero outcome looks, fits, selected parameters, or folds. In-sample/out-of-sample gaps and
probability_positive do not apply to this timestamp/identity census. No signal is rejected for
an interval crossing zero.

All available captures through the freeze are inventoried. Tables use report season 2026; bulk
historical seasons are counted separately. Weekday is Eastern capture weekday. Weekly key:
season/week/team/player. Headline key: URL/player, with no official report week. Inactive key:
team/player; the parser's caller-assigned week is not source-native evidence. Rows include blank
game statuses. All weeks in a cumulative release remain separate keys.

First means a nonblank designation without one in the preceding capture, including the initial
local baseline and blank-to-status changes; it is not publication time. Changed and Same require
a nonblank designation in consecutive captures. Cleared is status-to-blank. Practice separately
counts practice-status changes on matched rows. Reappearance after an absent row is separate in
scratch output. Same/(Changed+Same) has a descriptive 95% Wilson interval; repeated captures are
correlated, so these are not independent-game intervals. Unchanged does not itself mean stale.

## Source and capture coverage

| Source | Captures/pages | First capture UTC | Last capture UTC | Empty | Rows with native update field | Official report-to-capture lag |
| --- | --- | --- | --- | --- | --- | --- |
| NBC Sports headlines | 147 | 2026-09-10T02:52:33.878893+00:00 | 2026-09-29T20:01:18.378017+00:00 | 2 | 2362 | unknown; n=0 official pairs |
| NFL.com inactives | 57 | 2026-09-06T15:35:20+00:00 | 2026-09-27T18:40:30+00:00 | 57 | 0 | unknown; n=0 official pairs |
| NFL.com weekly | 3 | 2026-08-21T22:34:01+00:00 | 2026-08-25T19:42:02+00:00 | 3 | 0 | unknown; n=0 official pairs |
| RotoWire inactives | 7 | 2026-09-24T18:46:17+00:00 | 2026-09-27T18:40:30+00:00 | 2 | 0 | unknown; n=0 official pairs |
| nflverse | 64 | 2026-08-26T12:28:53+00:00 | 2026-09-27T13:00:22+00:00 | 2 | 0 | unknown; n=0 official pairs |
| Sportradar | 0 | -- | -- | -- | -- | no local snapshot |

**Measured:** other-season weekly row observations: NFL.com weekly: 17,483; nflverse: 5,808,128.
NFL.com has historical backfill and empty current-season attempts. Historical date_modified and
fetch times cannot measure current Friday latency.

## Designation transitions by source and weekday

**Measured:** capture-row observations, not unique players or games. Missing weekdays have no
archived capture for that source.

| Source | ET weekday | Captures | Rows | First | Changed | Same | Cleared | Practice | Same share, 95% Wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NBC Sports headlines | Monday | 18 | 500 | 12 | 0 | 488 | 0 | 0 | 488/488 (100.0%; 95% 99.2-100.0%) |
| NBC Sports headlines | Tuesday | 5 | 141 | 9 | 0 | 132 | 0 | 0 | 132/132 (100.0%; 95% 97.2-100.0%) |
| NBC Sports headlines | Wednesday | 14 | 292 | 12 | 0 | 280 | 0 | 0 | 280/280 (100.0%; 95% 98.6-100.0%) |
| NBC Sports headlines | Thursday | 52 | 260 | 3 | 0 | 257 | 0 | 0 | 257/257 (100.0%; 95% 98.5-100.0%) |
| NBC Sports headlines | Friday | 18 | 238 | 36 | 0 | 200 | 0 | 0 | 200/200 (100.0%; 95% 98.1-100.0%) |
| NBC Sports headlines | Saturday | 18 | 380 | 18 | 0 | 362 | 0 | 0 | 362/362 (100.0%; 95% 98.9-100.0%) |
| NBC Sports headlines | Sunday | 22 | 551 | 6 | 0 | 545 | 0 | 0 | 545/545 (100.0%; 95% 99.3-100.0%) |
| NFL.com inactives | Monday | 6 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Wednesday | 29 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Thursday | 11 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Saturday | 4 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Sunday | 7 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com weekly | Tuesday | 2 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com weekly | Friday | 1 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| RotoWire inactives | Thursday | 3 | 11 | 11 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| RotoWire inactives | Saturday | 2 | 22 | 0 | 0 | 22 | 0 | 0 | 22/22 (100.0%; 95% 85.1-100.0%) |
| RotoWire inactives | Sunday | 2 | 186 | 120 | 0 | 66 | 0 | 0 | 66/66 (100.0%; 95% 94.5-100.0%) |
| nflverse | Tuesday | 6 | 66 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| nflverse | Wednesday | 17 | 2205 | 5 | 0 | 721 | 0 | 0 | 721/721 (100.0%; 95% 99.5-100.0%) |
| nflverse | Thursday | 10 | 3650 | 17 | 0 | 735 | 0 | 18 | 735/735 (100.0%; 95% 99.5-100.0%) |
| nflverse | Friday | 8 | 3711 | 4 | 0 | 747 | 0 | 141 | 747/747 (100.0%; 95% 99.5-100.0%) |
| nflverse | Saturday | 16 | 4845 | 243 | 0 | 1396 | 0 | 173 | 1396/1396 (100.0%; 95% 99.7-100.0%) |
| nflverse | Sunday | 7 | 3131 | 16 | 4 | 1168 | 0 | 13 | 1168/1172 (99.7%; 95% 99.1-99.9%) |

**Measured:** nflverse has 17,608 current-season row observations, 0 nonnull date_modified
values. New-week repeated designations among keys with an observed designated immediately
preceding week: 40/71 (56.3%; 95% 44.8-67.3%). Recurrence across report weeks does not prove the
previous report was reused.

## Friday nflverse capture detail

**Measured:** counts below include each week separately; blank-status practice rows are not
final game designations. Current-week Friday counts may be lower than cumulative totals. These
fetch clocks do not measure official publication lag.

| Capture ET | Game designations by report week |
| --- | --- |
| 2026-09-11T06:00:42-04:00 | W1: 8 |
| 2026-09-11T16:30:33-04:00 | W1: 8 |
| 2026-09-18T06:00:25-04:00 | W1: 61; W2: 6 |
| 2026-09-18T09:00:05-04:00 | W1: 61; W2: 6 |
| 2026-09-18T16:31:38-04:00 | W1: 61; W2: 6 |
| 2026-09-25T06:00:39-04:00 | W1: 61; W2: 105; W3: 12 |
| 2026-09-25T09:00:53-04:00 | W1: 61; W2: 105; W3: 12 |
| 2026-09-25T16:30:49-04:00 | W1: 61; W2: 105; W3: 12 |

| Report week | Last Friday capture ET | Friday / latest designation counts |
| --- | --- | --- |
| 1 | 2026-09-11T16:30:33-04:00 | 8 / 61 |
| 2 | 2026-09-18T16:31:38-04:00 | 6 / 105 |
| 3 | 2026-09-25T16:30:49-04:00 | 12 / 119 |

**Measured:** the last Friday run is around 16:30 ET in each observed week. Later Friday
coverage is unmeasured. This comparison uses the latest weekly count as a reference, not
independently verified official completeness.

## Source-native clock evidence and Friday trust

| HTML source | Archived pages | Pages with timestamp markers |
| --- | --- | --- |
| NFL.com inactives | 57 | 0 |
| NFL.com weekly | 56 | 0 |
| RotoWire inactives | 7 | 0 |

**Measured:** marker scan checks datePublished/dateModified, article publish/modify metadata,
time elements, last-updated labels, report date/time and publication time. Matches retain
bounded context in scratch output. Generic navigation dates do not date a team report. This scan
does not establish the absence of every possible date-like string.

| Source | Trusted fresh at Friday cutoff? | Limitation |
| --- | --- | --- |
| nflverse weekly | Not demonstrated | Native season/week identity, but no verified official report time or final-report revision identity |
| NFL.com weekly | Not demonstrated | Week/season URL; current attempts empty; backfill cannot measure Friday arrival |
| NFL.com inactives | No Friday-designation evidence | A different release; archived primary pages lack usable current inactive rows |
| RotoWire inactives | Not demonstrated | No native list date/week/game identity; parser assigns season/week |
| Headline archive | Provisional news only | lastmod is editorial modification, not team-report publication or a complete designation list |
| Sportradar | Unmeasured | No local injury snapshots |

**Read:** scripts/nflverse_injuries_ingest.py:117 stores fetch time and row schema;
scripts/ingest_nflcom_injuries.py:188 stores requested week and fetch time;
src/nfl_ats/inactives_capture.py:231 injects caller season/week. None establishes an official
team publication clock. Official-report-to-capture lag has n=0 verified pairs for every source:
median/range/interval are not estimable. No fetch/lastmod proxy is substituted.

## Scheduler corroboration

**Measured:** 77 distinct snapshot stamps in relevant completion/skip lines; 83/278 audited
source captures match a stamp. Events span 2026-09-06 15:35:20+00:00 to 2026-09-29
20:01:18+00:00. These include manual reruns, not unique publications. Missing log matches do not
invalidate manifests; successful fetches do not prove freshness.

| Job family | ET weekday | OK | Failures | SKIP |
| --- | --- | --- | --- | --- |
| inactives | Monday | 6 | 0 | 0 |
| inactives | Saturday | 4 | 0 | 0 |
| inactives | Sunday | 7 | 0 | 0 |
| inactives | Thursday | 10 | 0 | 0 |
| inactives | Wednesday | 3 | 0 | 0 |
| injury_news | Friday | 17 | 0 | 0 |
| injury_news | Monday | 18 | 0 | 0 |
| injury_news | Saturday | 15 | 0 | 0 |
| injury_news | Sunday | 19 | 0 | 0 |
| injury_news | Thursday | 52 | 0 | 0 |
| injury_news | Tuesday | 5 | 0 | 0 |
| injury_news | Wednesday | 12 | 0 | 0 |
| nflverse_injuries | Friday | 7 | 0 | 0 |
| nflverse_injuries | Saturday | 12 | 0 | 0 |
| nflverse_injuries | Sunday | 6 | 0 | 0 |
| nflverse_injuries | Thursday | 10 | 0 | 0 |
| nflverse_injuries | Tuesday | 5 | 0 | 0 |
| nflverse_injuries | Wednesday | 7 | 0 | 0 |

## Season-ending IR question

**Measured:** injury archive data/raw/nflverse_injuries/20260927T130019Z/injuries.parquet joined
to roster archive data/players/raw/20260927T131540Z/weekly_rosters.parquet by
season/week/team/GSIS ID, regular season only. Denominator is limited to season/weeks present in
the injury archive; 0 missing roster IDs excluded; LA/JAC/WSH aliases normalized. RES is a
reserve code, not an explicit season-ending-IR flag.

| Week | Roster code | Player-weeks | In weekly injury report, 95% Wilson |
| --- | --- | --- | --- |
| 1 | RES | 266 | 7/266 (2.6%; 95% 1.3-5.3%) |
| 1 | ACT | 1536 | 122/1536 (7.9%; 95% 6.7-9.4%) |
| 2 | RES | 253 | 6/253 (2.4%; 95% 1.1-5.1%) |
| 2 | ACT | 1536 | 175/1536 (11.4%; 95% 9.9-13.1%) |
| 3 | RES | 279 | 19/279 (6.8%; 95% 4.4-10.4%) |
| 3 | ACT | 1724 | 276/1724 (16.0%; 95% 14.4-17.8%) |

**Measured:** all overlapping archived seasons' reserve-coded report inclusion: 6478/64923
(10.0%; 95% 9.7-10.2%). This is same-week co-occurrence, not an exclusion policy or timestamped
IR transition.

| Season-ending headline match | First locally seen UTC | Roster / injury-row presence |
| --- | --- | --- |
| Jake Tonges | 2026-09-16T20:01:15.821511+00:00 | W1 ACT: absent; W2 RES: absent; W3 RES: absent |
| Avonte Maddox | 2026-09-21T16:22:53.389818+00:00 | W1 ACT: absent; W2 ACT: absent; W3 RES: absent |
| Zach Bako-Bewele | 2026-09-22T20:01:22.203536+00:00 | W1 ACT: present; W2 ACT: present; W3 INA: present |
| Jaxson Dart | 2026-09-23T16:21:22.186971+00:00 | W1 ACT: absent; W2 ACT: absent; W3 RES: present |

**Measured:** examples require an explicit season-ending headline phrase, exact normalized
roster-name match and local first capture before the roster snapshot. Headlines are **reported**
news, not independently verified medical/transaction evidence. Earlier weeks are context, not
evidence of when IR began. **Inferred:** intentional nflverse exclusion of season-ending IR
remains unresolved. RES does not identify that population, roster snapshots do not timestamp its
transitions, and omission cannot establish publisher intent. Do not treat absence as
healthy/available or necessarily a data defect.

## Proposed cutoff rule (not implemented)

For configured Friday cutoff F in Eastern time, accept a Sunday-game final designation only when
an archived source-native record names the intended team and game/week, identifies the final
report/revision, and supplies a timezone-qualified official report time R such that R <= capture
C <= F < kickoff. The report date must be the intended game's final-report day; use the actual
reporting calendar for other game days. An authoritative final-report identity without a
publication clock needs separate validation, absent from this archive.

Missing/inconsistent identity or time means freshness unknown; retain previously authorized
behavior. Never silently reuse last week's status or turn missing into available. An unchanged
payload may be valid when independently dated for this report. First appearance only bounds
local availability. No fixed maximum age in hours is supported by n=0 official-time pairs. Keep
the kickoff guard; do not impose a narrower cutoff or serve headline-only statuses.

## Reproduction and limits

The script reads local manifests, captured payloads, weekly rosters and scheduler logs only.
Inactive replay is diagnostic; no capture-store or served-card writes. Input hashes, capture
coverage, clock candidates, transition counters, headline evidence and scheduler counts are
saved to tests/scratch/codex/lead64_freshness_audit.json. No prediction rows or outcomes read.
Scratch output is not a committed data artifact.

**Measured:** parser/input warnings: 2. data\raw\injury_news\20260819T191639Z\manifest.json: no
current.parquet; data\raw\injury_news\20260917T080042Z\manifest.json: no current.parquet

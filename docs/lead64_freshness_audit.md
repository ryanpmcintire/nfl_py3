# Friday designation freshness audit

**Measured:** archive frozen at 2026-11-03T19:26:04+00:00. Command: .tools/uv.exe run --no-sync
python scripts/lead64_freshness_audit.py --as-of 2026-11-03T19:26:04+00:00.

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
| NBC Sports headlines | 340 | 2026-09-10T02:52:33.878893+00:00 | 2026-11-03T01:01:09.949543+00:00 | 2 | 23887 | unknown; n=0 official pairs |
| NFL.com inactives | 108 | 2026-09-06T15:35:20+00:00 | 2026-11-01T20:20:05+00:00 | 108 | 0 | unknown; n=0 official pairs |
| NFL.com weekly | 3 | 2026-08-21T22:34:01+00:00 | 2026-08-25T19:42:02+00:00 | 3 | 0 | unknown; n=0 official pairs |
| RotoWire inactives | 58 | 2026-09-24T18:46:17+00:00 | 2026-11-01T20:20:05+00:00 | 17 | 0 | unknown; n=0 official pairs |
| nflverse | 137 | 2026-08-26T12:28:53+00:00 | 2026-11-03T19:26:03+00:00 | 2 | 0 | unknown; n=0 official pairs |
| Sportradar | 0 | -- | -- | -- | -- | no local snapshot |

**Measured:** other-season weekly row observations: NFL.com weekly: 17,483; nflverse:
12,433,024. NFL.com has historical backfill and empty current-season attempts. Historical
date_modified and fetch times cannot measure current Friday latency.

## Designation transitions by source and weekday

**Measured:** capture-row observations, not unique players or games. Missing weekdays have no
archived capture for that source.

| Source | ET weekday | Captures | Rows | First | Changed | Same | Cleared | Practice | Same share, 95% Wilson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NBC Sports headlines | Monday | 48 | 4246 | 26 | 0 | 4220 | 0 | 0 | 4220/4220 (100.0%; 95% 99.9-100.0%) |
| NBC Sports headlines | Tuesday | 14 | 1153 | 24 | 0 | 1129 | 0 | 0 | 1129/1129 (100.0%; 95% 99.7-100.0%) |
| NBC Sports headlines | Wednesday | 44 | 3272 | 23 | 0 | 3249 | 0 | 0 | 3249/3249 (100.0%; 95% 99.9-100.0%) |
| NBC Sports headlines | Thursday | 82 | 3242 | 9 | 0 | 3233 | 0 | 0 | 3233/3233 (100.0%; 95% 99.9-100.0%) |
| NBC Sports headlines | Friday | 48 | 3393 | 116 | 0 | 3275 | 0 | 0 | 3275/3275 (100.0%; 95% 99.9-100.0%) |
| NBC Sports headlines | Saturday | 47 | 3747 | 45 | 0 | 3702 | 0 | 0 | 3702/3702 (100.0%; 95% 99.9-100.0%) |
| NBC Sports headlines | Sunday | 57 | 4834 | 18 | 0 | 4816 | 0 | 0 | 4816/4816 (100.0%; 95% 99.9-100.0%) |
| NFL.com inactives | Monday | 6 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Wednesday | 34 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Thursday | 26 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Saturday | 14 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com inactives | Sunday | 28 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com weekly | Tuesday | 2 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| NFL.com weekly | Friday | 1 | 0 | 0 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| RotoWire inactives | Wednesday | 5 | 3 | 3 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| RotoWire inactives | Thursday | 18 | 50 | 28 | 0 | 0 | 0 | 0 | not estimable (n=0) |
| RotoWire inactives | Saturday | 12 | 152 | 8 | 0 | 126 | 0 | 0 | 126/126 (100.0%; 95% 97.0-100.0%) |
| RotoWire inactives | Sunday | 23 | 2559 | 413 | 0 | 1762 | 0 | 0 | 1762/1762 (100.0%; 95% 99.8-100.0%) |
| nflverse | Monday | 1 | 734 | 0 | 0 | 285 | 0 | 0 | 285/285 (100.0%; 95% 98.7-100.0%) |
| nflverse | Tuesday | 7 | 2525 | 0 | 0 | 1067 | 0 | 0 | 1067/1067 (100.0%; 95% 99.6-100.0%) |
| nflverse | Wednesday | 32 | 23835 | 5 | 0 | 9799 | 0 | 0 | 9799/9799 (100.0%; 95% 100.0-100.0%) |
| nflverse | Thursday | 25 | 28744 | 59 | 0 | 9897 | 0 | 28 | 9897/9897 (100.0%; 95% 100.0-100.0%) |
| nflverse | Friday | 23 | 29904 | 13 | 0 | 9969 | 0 | 472 | 9969/9969 (100.0%; 95% 100.0-100.0%) |
| nflverse | Saturday | 32 | 32412 | 912 | 0 | 12380 | 0 | 624 | 12380/12380 (100.0%; 95% 100.0-100.0%) |
| nflverse | Sunday | 17 | 20835 | 83 | 13 | 8708 | 5 | 56 | 8708/8721 (99.9%; 95% 99.7-99.9%) |

**Measured:** nflverse has 138,989 current-season row observations, 0 nonnull date_modified
values. New-week repeated designations among keys with an observed designated immediately
preceding week: 256/390 (65.6%; 95% 60.8-70.2%). Recurrence across report weeks does not prove
the previous report was reused.

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
| 2026-10-02T06:00:53-04:00 | W1: 61; W2: 105; W3: 119; W4: 9 |
| 2026-10-02T09:00:28-04:00 | W1: 61; W2: 105; W3: 119; W4: 9 |
| 2026-10-02T16:30:50-04:00 | W1: 61; W2: 105; W3: 119; W4: 9 |
| 2026-10-09T06:00:53-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 9 |
| 2026-10-09T09:00:06-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 9 |
| 2026-10-09T16:30:05-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 9 |
| 2026-10-16T06:00:58-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 6 |
| 2026-10-16T09:00:11-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 6 |
| 2026-10-16T16:30:19-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 6 |
| 2026-10-23T06:00:55-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 15 |
| 2026-10-23T09:00:22-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 15 |
| 2026-10-23T16:30:39-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 15 |
| 2026-10-30T06:00:50-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 153; W8: 12 |
| 2026-10-30T09:00:34-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 153; W8: 12 |
| 2026-10-30T16:30:33-04:00 | W1: 61; W2: 105; W3: 119; W4: 138; W5: 178; W6: 181; W7: 153; W8: 12 |

| Report week | Last Friday capture ET | Friday / latest designation counts |
| --- | --- | --- |
| 1 | 2026-09-11T16:30:33-04:00 | 8 / 61 |
| 2 | 2026-09-18T16:31:38-04:00 | 6 / 105 |
| 3 | 2026-09-25T16:30:49-04:00 | 12 / 119 |
| 4 | 2026-10-02T16:30:50-04:00 | 9 / 138 |
| 5 | 2026-10-09T16:30:05-04:00 | 9 / 178 |
| 6 | 2026-10-16T16:30:19-04:00 | 6 / 181 |
| 7 | 2026-10-23T16:30:39-04:00 | 15 / 153 |
| 8 | 2026-10-30T16:30:33-04:00 | 12 / 132 |

**Measured:** the last Friday run is around 16:30 ET in each observed week. Later Friday
coverage is unmeasured. This comparison uses the latest weekly count as a reference, not
independently verified official completeness.

## Source-native clock evidence and Friday trust

| HTML source | Archived pages | Pages with timestamp markers |
| --- | --- | --- |
| NFL.com inactives | 108 | 0 |
| NFL.com weekly | 56 | 0 |
| RotoWire inactives | 58 | 0 |

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

**Measured:** 194 distinct snapshot stamps in relevant completion/skip lines; 246/646 audited
source captures match a stamp. Events span 2026-09-06 15:35:20+00:00 to 2026-11-03
01:01:10+00:00. These include manual reruns, not unique publications. Missing log matches do not
invalidate manifests; successful fetches do not prove freshness.

| Job family | ET weekday | OK | Failures | SKIP |
| --- | --- | --- | --- | --- |
| inactives | Monday | 6 | 0 | 0 |
| inactives | Saturday | 14 | 0 | 0 |
| inactives | Sunday | 23 | 0 | 0 |
| inactives | Thursday | 25 | 0 | 0 |
| inactives | Wednesday | 8 | 0 | 0 |
| injury_news | Friday | 47 | 0 | 0 |
| injury_news | Monday | 48 | 0 | 0 |
| injury_news | Saturday | 44 | 0 | 0 |
| injury_news | Sunday | 54 | 0 | 0 |
| injury_news | Thursday | 82 | 0 | 0 |
| injury_news | Tuesday | 14 | 0 | 0 |
| injury_news | Wednesday | 41 | 0 | 0 |
| nflverse_injuries | Friday | 22 | 0 | 0 |
| nflverse_injuries | Saturday | 28 | 0 | 0 |
| nflverse_injuries | Sunday | 16 | 0 | 0 |
| nflverse_injuries | Thursday | 25 | 0 | 0 |
| nflverse_injuries | Tuesday | 5 | 0 | 0 |
| nflverse_injuries | Wednesday | 22 | 0 | 0 |

## Season-ending IR question

**Measured:** injury archive data/raw/nflverse_injuries/20261103T192549Z/injuries.parquet joined
to roster archive data/players/raw/20261101T141534Z/weekly_rosters.parquet by
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
| 3 | ACT | 1536 | 206/1536 (13.4%; 95% 11.8-15.2%) |
| 4 | RES | 300 | 16/300 (5.3%; 95% 3.3-8.5%) |
| 4 | ACT | 1536 | 216/1536 (14.1%; 95% 12.4-15.9%) |
| 5 | RES | 289 | 32/289 (11.1%; 95% 8.0-15.2%) |
| 5 | ACT | 1445 | 235/1445 (16.3%; 95% 14.5-18.3%) |
| 6 | RES | 273 | 26/273 (9.5%; 95% 6.6-13.6%) |
| 6 | ACT | 1344 | 262/1344 (19.5%; 95% 17.5-21.7%) |
| 7 | RES | 271 | 25/271 (9.2%; 95% 6.3-13.3%) |
| 7 | ACT | 1344 | 239/1344 (17.8%; 95% 15.8-19.9%) |
| 8 | RES | 254 | 20/254 (7.9%; 95% 5.2-11.8%) |
| 8 | ACT | 1507 | 292/1507 (19.4%; 95% 17.5-21.4%) |

**Measured:** all overlapping archived seasons' reserve-coded report inclusion: 6597/66310
(9.9%; 95% 9.7-10.2%). This is same-week co-occurrence, not an exclusion policy or timestamped
IR transition.

| Season-ending headline match | First locally seen UTC | Roster / injury-row presence |
| --- | --- | --- |
| Jake Tonges | 2026-09-16T20:01:15.821511+00:00 | W1 ACT: absent; W2 RES: absent; W3 RES: absent; W4 RES: absent; W5 RES: absent; W6 RES: absent; W7 RES: absent |
| Avonte Maddox | 2026-09-21T16:22:53.389818+00:00 | W1 ACT: absent; W2 ACT: absent; W3 RES: absent; W4 RES: absent; W5 RES: absent; W7 RES: absent; W8 RES: absent |
| Zach Bako-Bewele | 2026-09-22T20:01:22.203536+00:00 | W1 ACT: present; W2 ACT: present; W3 INA: present; W4 RES: absent; W5 RES: absent; W6 RES: absent; W7 RES: absent; W8 RES: absent |
| Jaxson Dart | 2026-09-23T16:21:22.186971+00:00 | W1 ACT: absent; W2 ACT: absent; W3 RES: present; W4 RES: absent; W5 RES: absent; W6 RES: absent; W7 RES: absent |
| Leo Chenal | 2026-09-28T16:24:38.013648+00:00 | W1 ACT: absent; W2 ACT: absent; W3 ACT: present; W4 RES: absent; W5 RES: absent; W6 RES: absent; W8 RES: absent |
| Durham Smythe | 2026-10-06T20:01:11.696495+00:00 | W1 ACT: absent; W2 ACT: absent; W3 ACT: absent; W4 ACT: present; W5 RES: present; W6 RES: absent; W7 RES: absent; W8 RES: absent |
| Braxton Berrios | 2026-10-07T04:01:07.978539+00:00 | W1 ACT: absent; W2 ACT: absent; W3 ACT: absent; W4 ACT: absent; W5 RES: absent; W6 RES: absent; W7 RES: absent |
| Tom Kennedy | 2026-10-12T20:01:25.995883+00:00 | W1 ACT: absent; W2 ACT: absent; W3 ACT: absent; W4 ACT: absent; W5 ACT: absent; W7 RES: absent; W8 RES: absent |
| Tremaine Edmunds | 2026-10-24T00:01:59.962173+00:00 | W1 ACT: absent; W2 ACT: absent; W3 ACT: absent; W4 ACT: absent; W5 ACT: absent; W6 ACT: absent; W7 RES: present |
| Daron Payne | 2026-10-31T17:31:21.385522+00:00 | W1 ACT: present; W2 ACT: present; W3 ACT: absent; W4 ACT: absent; W5 ACT: absent; W6 ACT: absent; W8 ACT: present |

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

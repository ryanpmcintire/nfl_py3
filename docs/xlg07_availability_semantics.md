# XLG-07 cross-league availability semantics

Audited 2026-09-02. This is a source-feasibility decision only. No ATS
experiment was run, no model feature was built, and no weak-signal or rotation
verdict was recorded.

## Decision

**Measured — fail closed for historical learning.** The locally retained CFB
availability artifacts do not provide enough seasons with artifact-level
pregame timestamps and interpretable missingness to train or evaluate an
availability model. Recent Big Ten and SEC mandates make prospective collection
useful, but they do not repair the historical instrument already on disk.

**Read — this completes the XLG-07 question rather than rejecting a measured
signal.** `ROADMAP.md` asks whether historical reports are genuinely pregame
and complete enough to learn availability, with a fail-closed result when
timestamps or missingness are ambiguous. No ATS outcome was inspected in this
audit, so the weak-signal terminal taxonomy is not applicable.

## Admission contract

**Read — `docs/data_feasibility.md` lines 8-18.** A source with fewer than five
usable seasons or without a defensible historical timestamp is Low/blocked for
retrospective estimation.

**Inferred — operational interpretation for this audit.** A historical CFB
availability source is admissible only when all of the following are true:

1. **Inferred:** the retained artifact, not merely the conference policy, proves
   that each team report existed before its game's kickoff;
2. **Inferred:** a missing player or team can be distinguished from an
   unavailable report, a bye, a non-covered game, and an explicitly healthy
   report;
3. **Read:** coverage reaches the five-season floor in
   `docs/data_feasibility.md`, with source regimes separated; and
4. **Inferred:** the raw artifact and its capture/issuance timestamp can be
   retained so later parsing cannot silently rewrite the historical state.

## Legacy all-FBS sources

**Read — `docs/cfb_data.md` lines 66-80 and `src/nfl_ats/cfb.py` lines
429-443.** The `espn_cfb_injuries` release has zero assets and CFBD v5 has no
injuries endpoint. ESPN game-roster Active/Inactive fields are scrape-time
attributes rather than game-day designations; `did_not_play`, `starter`, and
`valid` are unusable defaults and are quarantined by the canonical loader.

**Read — `docs/data_feasibility.md` lines 73-76.** The 2024 roster audit found
zero of 27,471 players changing Active/Inactive state and all-false
`did_not_play`/`starter` fields. An absent injury record in those sources means
the source is unavailable, not that the player was healthy.

**Inferred:** no all-FBS historical availability target can be learned from the
legacy sources without inventing labels from postgame participation.

## Big Ten, 2023

### What is genuine

**Measured — official page fetched 2026-09-02:** the Big Ten's 2023 policy page
returned HTTP 200 and says institutions must submit a gameday availability
report before every contest, no later than two hours before scheduled kickoff;
the conference then distributes it on its report page and social account.
Source: `https://bigten.org/fb/article/blt2856785fb75ee868/`.

**Read — `data/raw/bigten_availability/20260822T125805Z/manifest.json`:** the
2023 hub exposed 14 weekly PDF links. The pilot retained Weeks 1-6, and each of
the six hashes in the manifest still matches a file with a valid PDF header.

**Measured — pypdf extraction over the six retained files:** all six PDFs parse,
each has 14 team pages, for 84 pages total. Active-game pages contain explicit
`OUT` and `QUESTIONABLE` sections and sometimes the literal value `None`.

### Why the archive is not a historical pregame instrument

**Measured — pypdf metadata joined to
`data/cfb/schedules/raw/20260816T162105Z/season=2023/schedules.parquet`:** 71 of
the 84 team pages matched exactly one local team/week game; the other 13 were
byes or schedule-key ambiguities. Of the 71 matched pages, **53 were inside an
aggregate PDF whose CreationDate was after that team's kickoff**.

**Measured — the clearest example:** Week 3's PDF CreationDate is
2023-09-16 21:32:26 UTC, while its pages include eleven teams whose games had
already kicked off. Every audited file has at least one matched page whose game
started before aggregate creation.

**Measured — extracted text across all 84 pages:** zero page-level strings
identify an issue, submission, publication, or update time. The only machine
timestamp is the PDF-level CreationDate/ModDate.

**Inferred:** the conference policy establishes when schools were supposed to
submit, but the retained aggregate establishes only when the combined file was
created. Without immutable per-team snapshots or page-level issuance times, the
audit cannot prove whether a page preserves the pregame submission or a later
revision.

**Inferred:** explicit `None` makes within-page negative-list semantics clearer
than the legacy roster source, but it does not make an absent page/report safe.
The local snapshot is also only six weeks of one season, below the admission
floor even if every page timestamp were repaired.

## SEC, 2024-2025 archive surface

**Read — `artifacts/sec_pilot/20260822T140443Z/pilot_status.json` and
`coverage_stats.json`:** the parsed pilot contains 137 rows from only Weeks 2,
6, and 15 of 2024, spanning 14 team-games. Team names all normalized, and the
observed statuses are `out`, `probable`, `questionable`, `doubtful`, and
`game time decision`.

**Read — the same artifacts:** Weeks 2 and 6 have contemporaneous Wayback
capture timestamps and are defensible pregame states. Week 15 was fetched live
long after the 2024-12-07 game and is explicitly classified PIT-B, not a
contemporaneous snapshot.

**Measured — official pages fetched 2026-09-02:** both
`https://www.secsports.com/fbreports` and
`https://www.secsports.com/fbreports-archive` returned HTTP 200, embedded the
HD Intelligence application, and exposed zero direct PDF links. The archive
page identifies the 2025 football season but does not expose anonymous report
files in its server-rendered HTML.

**Read — `docs/sec_availability_pilot.md`:** the 2024 public Google Sheet
updated in place, Wayback retained only two distinct table states, the 2025
archive API returned empty to unauthenticated callers, and school workbooks
returned HTTP 401. Only the last written week remained on the live public tabs.

**Inferred:** an omitted SEC team/week cannot safely mean “no unavailable
players” because most weeks were never recovered. Fourteen team-games from one
season are not enough to estimate availability semantics, regardless of the
number of player rows.

## Final disposition

**Measured:** neither conference supplies five usable historical seasons in
the retained local evidence. Big Ten's six-week snapshot lacks per-team
pregame artifact timestamps, and SEC supplies only two contemporaneously
captured historical weeks plus one retrospective state.

**Inferred — fail-closed rule:** do not join either source to XLG-03, do not
train a CFB availability model from it, and do not interpret absent rows as
healthy. Realized participation remains postgame evidence and may affect only
later games under the existing lag contract.

**Inferred — prospective path:** immutable per-report capture can make future
Big Ten and SEC rows admissible. Each capture must retain the raw payload,
capture time, source-provided issue time when present, scheduled kickoff,
conference/team/week coverage universe, explicit-empty reports, and a source
regime label. That is collection work for a future prospective dataset, not an
unfinished deliverable of this historical feasibility audit.

## Reproduction notes

**Measured:** the local Big Ten audit used the six manifest-pinned PDFs,
`pypdf` 6.16.2 from the locked environment, and the pinned 2023 schedule
parquet above. It counted status headings and issue-time markers in extracted
page text, then compared each unambiguous team/week match with the PDF metadata
CreationDate.

**Measured:** the SEC counts were read directly from the final, non-superseded
pilot artifact `artifacts/sec_pilot/20260822T140443Z/`; the earlier
`20260822T135810Z` run is excluded because its parser double-counted Wayback
wrappers, as documented in `docs/sec_availability_pilot.md`.

**Inferred — no new audit program:** `scripts/pilot_bigten_availability.py` and
`scripts/pilot_sec_availability.py` already provide the source-delivery and
manifest machinery. Adding a third overlapping command solely to restate this
decision would create unused duplication; the evidence gap is historical
capture provenance, not parsing code.

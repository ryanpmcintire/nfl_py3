# ENV-02 roof-decision source feasibility

Date: 2026-09-02

## Decision

**Measured (2026-09-02 primary-source browser audit):** no documented,
stable, public machine interface or complete point-in-time archive was found
for the T-90 open/closed decision across ARI, ATL, DAL, HOU, and IND.

**Inferred:** ENV-02 therefore remains open, and adding a scheduler or parser
now would turn a heterogeneous set of editorial pages and social posts into an
undocumented availability assumption.

**Read (`docs/roof_decision_screen.md`, lines 1-7 and 325-354):** the
existing screen uses nflverse's realized `roof` field to study a mechanism; it
does not establish when a roof decision became knowable and it explicitly does
not provide a live T-90 ingestion path.

**Measured (repository diff and commands in this audit):** no ATS experiment,
weak-signal record, model-profile change, capture-scheduler change, or roof
state inferred from a realized nflverse value was produced by this work.

## Point-in-time contract

**Read ([NFL Football Operations, Game and Stadium
Prep](https://operations.nfl.com/game-operations-logistics/preparation-safety/game-and-stadium-prep),
live page opened 2026-09-02, lines 248-264):** at 90 minutes before kickoff the
home club tells the referee or on-site NFL Football Operations representative
whether the roof or wall will be open or closed, and the designated position
must be reached by 60 minutes before kickoff.

**Read (same NFL source, lines 261-264):** an initially open roof may later be
closed for precipitation or hazardous conditions, so a capture must preserve
transitions rather than overwrite the T-90 state with a final state.

**Inferred:** an admissible decision feature must come from a source observed
no later than the model decision timestamp, carry its own retrieval timestamp,
and fail closed to `unknown` when the source is absent, ambiguous, changed, or
published too late.

## Primary-source audit

| Venue / authority | What the official source currently exposes | Point-in-time finding |
|---|---|---|
| NFL | **Read ([NFL Football Operations](https://operations.nfl.com/game-operations-logistics/preparation-safety/game-and-stadium-prep), opened 2026-09-02, lines 248-264):** the league documents the internal T-90 notification and T-60 positioning process. | **Measured (page inspection 2026-09-02):** the page contains no public per-game roof-status feed, archive, event identifier, or publication timestamp. |
| IND / Lucas Oil Stadium | **Read ([Colts Gameday A-Z Guide](https://www.colts.com/game-day/a-z-guide), opened 2026-09-02, lines 497-502):** roof updates are shared exclusively on `@ColtsLife` and at the stadium. | **Measured (page inspection 2026-09-02):** the official web page provides policy and routes the actual status to a social account; it does not expose the decision itself or a documented archive/API. |
| HOU / NRG Stadium | **Read ([Texans A-Z Guide](https://www.houstontexans.com/stadium/a-z-guide), opened 2026-09-02, lines 298-300):** the club says the decision is made two hours before kickoff and directs fans to HoustonTexans.com or a text number for the result. | **Measured (official-site search and page inspection 2026-09-02):** dated roof articles exist, but the guide does not name a stable per-game endpoint, publication schema, completeness guarantee, or archive contract. |
| ATL / Mercedes-Benz Stadium | **Read ([2026 Falcons-Ravens event page](https://www.mercedesbenzstadium.com/falcons-gameday/baltimore-ravens), opened 2026-09-02, lines 152-180):** the official event page has a `Roof Status` field and currently reports `TBD`. | **Measured (page inspection 2026-09-02):** the event-specific page is useful prospective evidence, but it supplies no status-publication timestamp, update history, archive guarantee, or documented API. |
| ARI / State Farm Stadium | **Read ([Cardinals article dated 2024-01-06](https://www.azcardinals.com/news/before-talk-of-offseason-first-comes-cardinals-finale), opened 2026-09-02, lines 43-46 and 87-91):** a team news article announced that the roof would be closed for the next day's finale. | **Measured (official-site search 2026-09-02):** roof decisions can appear inside dated editorial articles, but no stable current status page, per-game schema, complete archive, or documented API was found. |
| DAL / AT&T Stadium | **Read ([Cowboys Know Before You Go](https://www.dallascowboys.com/stadium/know-before-you-go), opened 2026-09-02, lines 28-57):** the current event page supplies game and entry times but no roof-status field. | **Measured (official-site search and page inspection 2026-09-02):** no current official per-game roof-status endpoint, timestamped archive, or documented API was found. |

**Measured (the source matrix above):** the publication surfaces differ by
venue: social-only routing at IND, a generic website/text instruction at HOU,
an event field at ATL, occasional news articles at ARI, and no roof field on
the inspected DAL event page.

**Inferred:** a single HTML selector cannot provide defensible five-venue
coverage, and silently treating a missing item as `closed` would confound
source failure with venue state.

**Inferred:** public readability alone is not a grant of automated reuse; a
production capture should use an official documented API/feed, a licensed
archive, or explicit permission whose terms cover the intended collection and
retention.

## Required capture shape

**Inferred:** if a defensible source becomes available, each observation should
be append-only and contain at least:

- **Inferred:** `game_id`, scheduled kickoff, home team, and venue identity.
- **Inferred:** source URL/provider, immutable raw-payload digest, retrieval UTC
  timestamp, and any source-authored publication/update timestamp.
- **Inferred:** normalized state in exactly `open`, `closed`, or `unknown`, plus
  the unmodified source text/value and parser version.
- **Inferred:** seconds relative to kickoff and a flag showing whether the
  observation satisfied the decision cutoff.
- **Inferred:** a supersession link rather than mutation when weather or safety
  closes a roof after the initial designation.

**Inferred:** the parser must return `unknown` for `TBD`, missing elements,
contradictory values, an unparseable timestamp, an event/game mismatch, HTTP
failure, or a retrieval later than the permitted decision timestamp.

**Inferred:** before model eligibility, source coverage must be measured by
venue and season, and a leakage regression must prove that later page changes,
postgame realized roof values, results, and future rows cannot alter an earlier
decision row.

## Concrete blocker and next evidence

**Measured (2026-09-02 audit above):** ATL is the only inspected official
surface with a directly parseable event-page field, while IND explicitly sends
the status off-site and the other inspected club pages do not expose a uniform
equivalent.

**Inferred:** scheduler integration is not operationally justified until every
target venue has a permitted, stable source and the publication latency is
measured against the actual decision cutoff; scheduling a broken or partial
source would create authoritative-looking `unknown`/missingness without solving
the feature.

**Inferred:** the next acceptable evidence is either (a) a league/club feed or
licensed archive with game identifiers, timestamps, and documented coverage,
or (b) a prospective one-season source manifest that proves all five official
venue adapters publish in time and preserves immutable raw captures.

**Inferred:** until one of those conditions is met, ENV-02 should remain open;
the existing realized-roof screen remains mechanism evidence only and must not
be used to reconstruct historical T-90 decisions.

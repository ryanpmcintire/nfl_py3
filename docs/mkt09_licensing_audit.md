# MKT-09: provider licensing / quota audit

Status: **audit delivered 2026-08-24** (read-only session; this file is the only artifact).
Scope: every external data source the project consumes, its governing terms,
what personal research vs redistribution is allowed, retention limits on raw
snapshots, and a RAG verdict with the specific risk. ROADMAP row: MKT-09
("Terms, redistribution limits, cost, retention, failure policy").

## Provenance legend (binding, per AGENTS.md)

- **measured** — the page/artifact was fetched or opened THIS session
  (2026-08-24); URL/path given inline.
- **read** — a repo file was opened this session; path given.
- **reported** — a repo doc (or subagent-era measurement recorded in it) says
  so and was NOT independently re-executed here. Treat as unverified.
- **inferred** — reasoning or judgment, explicitly not evidence.

All quoted clauses were captured verbatim from the governing pages during this
session unless labeled otherwise.

## What the public output actually is (the redistribution question)

The project's public surface is (a) the tracked GitHub files
`README.md` / `CURRENT_PREDICTIONS.md` and (b) the GitHub Pages dashboard built
into `docs/*.html`. Both publish **model-derived picks, probabilities, and at
most ONE consensus market line per game plus our own fair line** — never raw
quote tables. The code already enforces this boundary:

> "LICENSING (MKT-09 provider licensing/quota audit, ROADMAP.md): the public
> site plots ONLY the one consensus market line this card already publishes
> and our own fair line. The internal dashboard also plots an archive-derived
> opener consensus and a predicted close; both stay off the public site until
> that audit clears redistribution."
> — **read** this session: `src/nfl_ats/public_board.py:1070-1074`.

Raw snapshots of every source live only in gitignored local directories
(`data/raw/**`, `data/market/raw/**`, `data/processed/**`) — **reported**
(README "Repository layout", `docs/data.md` "Storage"; the same convention was
verified by `git status` in the SBR session, `docs/sbr_odds_archive.md:36-41`),
so nothing in Git redistributes source bytes. This audit's job is to check
whether that posture satisfies each provider's actual terms.

## Master compliance matrix

| # | Source | We take | Access path | Governing terms (URL, access date) | Personal research OK? | Raw-snapshot retention limit | RAG |
|---|--------|---------|-------------|-------------------------------------|----------------------|------------------------------|-----|
| 1 | The Odds API — paid historical snapshots (the "purchased 2020–2025 archive") | Point-in-time multi-book odds boards at 6 decision labels, 2020–2025; hourly 2023–2025; player-prop tranches (MKT-13) | Paid API key; `nfl_ats.odds_backfill` → `api.the-odds-api.com/v4/historical/...`; credits metered (**read**: `src/nfl_ats/odds_backfill.py:1-9,39`) | Terms & Conditions, `https://the-odds-api.com/terms-and-conditions.html`, accessed 2026-08-24 (**measured**) | Yes — analytical tools expressly encouraged, incl. commercial use | No expiry/deletion clause found; prohibition is on redistributing raw data as a standalone product, not on retaining it | **GREEN** |
| 2 | The Odds API — free-tier live captures | Current spread/moneyline/total observations; ~six weekly scheduled captures, 11 books (**reported**, ROADMAP MKT-02; `docs/ops_runbook.md:52`) | Same account/key, free tier (500 credits/mo, **measured** pricing page 2026-08-24); `scripts/odds_capture.ps1` → `nfl-ats odds-ingest` | Same terms page as row 1 (**measured** 2026-08-24) | Yes | Same as row 1 | **GREEN** |
| 3 | Internet Archive Wayback Machine | Archived copies of VegasInsider NFL boards 2005–2016; archived copies of the SEC availability sheet | `web.archive.org` CDX + snapshot fetches (`scripts/backfill_vegasinsider.py`; `docs/vegasinsider_backfill.md`) | IA Terms of Use (31 Dec 2014), `https://archive.org/about/terms.php` — JS-gated; text captured this session from IA's own hosted copy (see §3) (**measured, with caveat**) + Wayback help FAQ, `https://help.archive.org/help/using-the-wayback-machine/` (**measured**) | Yes — "scholarship and research purposes" is the granted purpose | None stated; but IA can exclude/remove captures at an owner's request, so IA is not a durable backup | **YELLOW** |
| 4 | VegasInsider board content (underlying site mined via row 3) | Per-(capture, game, named book) spread/total lines 2005–2016 | Never touches vegasinsider.com servers; consumed as third-party Wayback copies | VegasInsider Terms of Use, `https://www.vegasinsider.com/terms-of-use/`, accessed 2026-08-24 (**measured**) | Download-to-personal-device is permitted; scraping/data-mining of their Service is prohibited; republication prohibited absent written consent | No explicit retention clause; the binding restriction is on republication/display, which stays local-only here | **YELLOW** |
| 5 | NFL.com league-wide injury-report pages | Final Fri/Sat game-status designations, seasons 2022–2024 (54 pages, 17,483 rows) + prospective weekly fetches planned | Direct HTTPS scrape with runtime robots check, ≥2s delay (`scripts/ingest_nflcom_injuries.py`; `docs/nflcom_injuries_sourcing.md`) | NFL.com Terms & Conditions, `https://www.nfl.com/legal/terms/`, accessed 2026-08-24 (**measured**); robots.txt re-checked this session (**measured**) | **Ambiguous-to-no**: §1.3 bars "systematic retrieval … to create or compile … a collection, compilation, database" absent express written consent, with NO commercial-purpose qualifier | No retention clause; the operative problem is the collection act itself, not storage | **RED** |
| 6 | nflverse maintained datasets (schedules, results, closes, PBP, participation) | Game/results backbone, closing spreads/prices, play-by-play, participation | `nflreadpy` downloads from nflverse releases (**read**: `docs/data.md:11-23`) | nflverse-data repo is CC-BY-4.0 (`https://github.com/nflverse/nflverse-data`, accessed 2026-08-24, **measured**); upstream datasets may differ (**read**: `docs/data.md:109-115`) | Yes — attribution is the only condition | Redistribution allowed WITH attribution; repo keeps raw out of Git anyway | **GREEN** |
| 7 | SEC student-athlete availability reports (published Google Sheet) | Tidy Wed/Thu/Fri/gameday availability states (137 pilot rows, 3 PIT states) | Unauthenticated `/pub?gid=…&output=csv` endpoints of a sheet embedded on secsports.com (`docs/sec_availability_pilot.md:33-42`) | Google Terms of Service, `https://policies.google.com/terms` (eff. 2026-07-30), accessed 2026-08-24 (**measured**); the SEC-side rights statement was NOT located this session (**reported**: none found) | Yes for reading published CSVs; Google's "Other content" clause defers to the content owner's rights | None stated; sheet updates in place, so local timestamped snapshots are the only durability | **GREEN** |

Adjacent sources already inside the repo that share the same analysis (compact;
all **read** from `docs/`):

| Source | License posture (per repo doc) | RAG |
|---|---|---|
| sportsbookreviewsonline.com odds archive 2007–2021 (MKT-10) | Free static HTML; robots.txt permits the archive path; cached locally only; no republication (`docs/sbr_odds_archive.md:11-41`) | GREEN (private research), YELLOW if ever republished |
| Spreadspoke/Kaggle `tobycrabtree` close archive | CC BY-NC-SA 4.0 — **non-commercial**; raw ZIP + license preserved together (`docs/data.md:134-146`) | GREEN (research use), RED if commercialized |
| 2025 opener / nine-book close Kaggle sample | CC BY-NC 4.0 — **non-commercial** (`docs/data.md:147-169`) | GREEN (research use), RED if commercialized |

---

## Row 1+2 — The Odds API (paid historical archive + free-tier captures)

**What we take and how.** The "purchased point-in-time snapshot archive" is
not a separate vendor product: it is The Odds API Pty Ltd's *historical odds
endpoint*, queried under this project's own paid plan and stored locally with a
`historical_backfill` marker — **read**: `src/nfl_ats/odds_backfill.py:1-9`
(module docstring: "Historical point-in-time snapshot backfill from The Odds
API historical endpoint"), line 39 (endpoint URL), and `docs/novig_diagnostics.md:60`
("a purchased, point-in-time, multi-book odds archive"). Archive size
(8,746 snapshots, verified backups on two drives) and the six-weekly-capture
cadence are **reported** from ROADMAP MKT-02, not re-counted here. Player-prop
tranches (MKT-13) ran on the same paid account with a hard budget floor;
1,508 requests remained after the August tranche — **reported**
(`docs/player_props_sourcing.md`, ROADMAP MKT-13).

**Governing terms — measured 2026-08-24** at
`https://the-odds-api.com/terms-and-conditions.html` (operator identity:
"The Odds API Pty Ltd, ACN: 627461947"). Operative clause, verbatim:

> "**Restrictions** — Do not resell, repackage, or redistribute our data as a
> standalone data product. This includes, but is not limited to, offering our
> data through your own API, data feed, downloadable files, or any other
> format intended to serve as a source of raw data for others.
>
> We support and encourage the use of our data in websites, mobile apps,
> dashboards, analytical tools, and other user-facing applications, including
> commercial use, provided our data is not the primary product being sold or
> redistributed."

Cost mechanics — **measured** the same day on
`https://the-odds-api.com/historical-odds-data/`: "Historical data is only
available on paid usage plans"; featured-market snapshots exist from
2020-06-06 at 10-minute intervals (5-minute from September 2022); usage cost
is "10 per region per market" per historical call; free plans carry
"500 credits per month" (pricing section of `https://the-odds-api.com/`).

**Analysis.**

- *Personal research*: unambiguously permitted; analytics/backtesting is the
  encouraged case.
- *License scope*: there is **no expiry clause** and no termination-deletion
  clause in the fetched terms — cancellation ends billing and (per the abuse
  clause) access can be revoked, but nothing requires destroying already
  downloaded snapshots. That reading is **inferred** from the absence of any
  such clause in the measured text, not from an affirmative grant.
- *Derived features in public docs*: covered — the dashboard/card is a
  "user-facing application" and the odds data is not the primary product sold
  or redistributed (nothing is sold). Publishing one consensus line per game
  is comfortably inside the encouraged category. Publishing **raw quote
  archives** (bulk downloadable files of snapshots) would fall in the literal
  words of the prohibition — the existing public-board guard
  (`public_board.py:1070-1074`) already prevents this.
- *Vendor-confusion caution*: a distinct company trading as "TheOddsAPI"
  (theoddsapi.com) operates separately; the genuine operator's own footer
  carries an "Impersonator Warning" link. All URLs above are on the
  `the-odds-api.com` domain matching the endpoint hardcoded in
  `src/nfl_ats/odds_backfill.py:39` — **measured**.

**Verdict GREEN.** Conditions to keep it green: raw responses/snapshots never
leave the local gitignored tree; any future public artifact ships derived
numbers only; quota ledger discipline continues.

---

## Row 3 — Internet Archive Wayback Machine

**What we take.** Third-party archived copies of VegasInsider NFL odds boards
2005–2016 (12 REG seasons, ~18k tidy rows — **read**:
`docs/vegasinsider_backfill.md`) and three point-in-time states of the SEC
availability Google Sheet (`docs/sec_availability_pilot.md`). Fetched via CDX
listing plus snapshot GETs, batched (≤2 seasons/run, ≥3s delay, wall-clock cap
— **read**, same doc, "Reproduction / resumption").

**Governing terms.** The canonical page `https://archive.org/about/terms.php`
renders a JavaScript shell this session (**measured**: two fetch attempts
returned "Javascript is required"), so the verbatim text below was captured
from the Internet Archive's own hosted copy of its Terms of Use
(`https://ia801507.us.archive.org/11/items/archive.org-terms-and-conditions/Archive.org%20Terms%20and%20Conditions.txt`,
surfaced via web search 2026-08-24) and cross-checked against the Archive's
announcement of the same text (`https://blog.archive.org/2014/12/30/update-to-terms-of-use`,
**measured** via search result). Label: **measured, with the caveat that the
canonical page itself was JS-gated**. Operative clauses, verbatim:

> "Access to the Archive's Collections is provided at no cost to you and is
> granted for scholarship and research purposes only."

> "You agree to abide by all applicable laws and regulations, including
> intellectual property laws, in connection with your use of the Archive. In
> particular, you certify that your use of any part of the Archive's
> Collections will be limited to noninfringing or fair use under copyright law."

(The pre-2014 version additionally required "noncommercial" use and barred
copying Collections offsite without permission; both were removed on
2014-12-30/31 — **measured** from the update post.)

Robots/exclusion posture — **measured** 2026-08-24 from the Wayback help FAQ
(`https://help.archive.org/help/using-the-wayback-machine/`): sites may be
missing because they were "blocked by robots.txt" or excluded, and owners may
request exclusion ("If you would like to submit a request for archives of your
site or account to be excluded from web.archive.org, send us a request to
info@archive.org…"). The Archive also disclaims accuracy of the Collections.

**Analysis.**

- *Personal research*: squarely inside the granted purpose
  ("scholarship and research"); the December 2017-era debate about robots.txt
  retroactivity concerns the Archive's crawler, not our replay access, and the
  current FAQ states exclusions happen at crawl time or by owner request —
  **inferred** from the measured FAQ text.
- *Republication risk*: the IA grant is to USE the Collections for research;
  it does not launder the rights of the underlying site (row 4). Republishing
  Wayback-captured board data publicly would put BOTH rows 3 and 4 in
  conflict. Current posture (local-only, derived-only publication) avoids it.
- *Durability*: because owners can retroactively exclude domains, the local
  immutable snapshots are the real archive; Wayback is an access method, not a
  backup — **inferred**.

**Verdict YELLOW.** Specific risk: (i) the "scholarship and research purposes"
grant is broad but not defined, and bulk scripted CDX harvesting of one
commercial domain sits at the aggressive end of ordinary scholarly practice;
(ii) captures can vanish retroactively, so reproducibility depends entirely on
our local copies; (iii) IA's copyright disclaimer shifts the underlying-rights
question entirely onto row 4.

---

## Row 4 — VegasInsider board content (mined via row 3)

**Governing terms — measured 2026-08-24**, full page fetch of
`https://www.vegasinsider.com/terms-of-use/` (Better Collective USA, Inc.).
Operative clauses, verbatim:

> §2: "The Website, the Services, and the Content are provided for your
> non-commercial entertainment and enjoyment. Under the TOS, you may download
> certain Content and Services available on the Website to a single personal
> computing device for your use and entertainment. However, you may not
> distribute, modify, republish, or publicly display any of the Content or
> Services unless you have the prior written permission of VegasInsider.com…
>
> The Service contains proprietary information, statistics, and projections,
> both original and from other third-party sources. Users of the Service may
> not engage in unauthorized spidering, 'scraping,' data mining or harvesting
> of Content, or use any other unauthorized automated means to gather data
> from or about the Service."

> §16: "Services and Content are the property of VegasInsider.com and are
> licensed to VegasInsider.com and may not be reproduced without the prior
> written consent of VegasInsider.com… you agree not to reproduce, republish,
> upload, post, transmit, distribute, copy, publicly display or otherwise use
> any Content or any derivative works based on the Website, Services, Content
> or the Software, in whole or in part."

**Analysis.**

- The 2005–2016 backfill never contacted vegasinsider.com; it read the
  Internet Archive's copies. A browsewrap contract binds users of *their*
  service, so the anti-scraping sentence arguably never attached to us —
  **inferred** (and consistent with the general logged-out-scraping contract
  logic discussed in X Corp. v. Bright Data; that precedent is context, not
  a guarantee).
- What survives regardless of the contract question is the proprietary-rights
  claim over the CONTENT (§16): the historical board numbers are VI's
  compilation. Private research use matches §2's own picture of permitted
  use (personal download/use); public display of the book-by-board history
  would contradict §2 and §16 outright.
- The 2011-era layouts being mined are historical pages whose then-publisher
  terms may have differed; no archived 2005-era ToS was retrieved this
  session — **reported/inferred gap, flagged honestly**.

**Verdict YELLOW.** Specific risk: public republication of the historical
book-by-book lines (or shipping them as a downloadable dataset) would be a
direct textual violation of §2/§16. Private research caching is consistent
with §2's permitted use. Mitigation already in place: parsed parquet lives in
gitignored local directories; research docs publish aggregates and coverage
statistics, not board cells.

---

## Row 5 — NFL.com injury pages (RED)

**What we take.** League-wide final designations from
`https://www.nfl.com/injuries/league/{season}/reg{week}` — a one-time 54-page
retro snapshot (2022–2024 REG, 17,483 player-week rows) plus intended weekly
prospective fetches — **read**: `docs/nflcom_injuries_sourcing.md`.

**Governing terms — measured 2026-08-24**, full fetch of
`https://www.nfl.com/legal/terms/` (updated May 16, 2024). Operative clauses,
verbatim:

> §1.3 Permitted Uses: "You may use the Services solely for your own
> individual non-commercial and informational purposes only. Any other use,
> including for any commercial purposes, is strictly prohibited without our
> express prior written consent. **Systematic retrieval of data or other
> content from the Services, whether to create or compile, directly or
> indirectly, a collection, compilation, database, or directory, is
> prohibited absent our express prior written consent.**"

> §11(f): "use or attempt to use any engine, software, tool, agent or other
> device or mechanism (including, browsers, spiders, robots, avatars or
> intelligent agents) to navigate or search the Services to harvest or
> otherwise collect information from the Services to be used for any
> commercial purpose"

robots.txt — **measured** 2026-08-24, `https://www.nfl.com/robots.txt`: the
`User-agent: *` block disallows `/_ctv/`, `/_fantasy-app/`, `/_libraries/`,
`/_mobile-app/`, `/_mobileview/`, `/_phs/`, `/_sponsors/`, `/account/`,
`/nfl-films-beta/`, `/search/`. **`/injuries/` is NOT disallowed** and there
is no Crawl-delay directive. This re-confirms the repo doc's 2026-08-21
measurement (`docs/nflcom_injuries_sourcing.md:19-25`, which also documents
that the ingester re-checks robots at runtime and fails closed, ≥2s delays).

**Analysis — why RED despite robots-permitted access.**

1. §1.3's systematic-retrieval prohibition has **no commercial-purpose
   qualifier** (contrast §11(f), which is commercial-only). Compiling 54
   pages into one tidy 17,483-row database is literally "systematic retrieval
   … to create or compile … a collection, compilation, database". On the
   plain text, the retro snapshot and any recurring scheduled harvest require
   "express prior written consent" we do not have. This is the one row where
   the operative terms plausibly prohibit the collection activity itself —
   i.e., even the personal-research cache — and therefore the row the audit
   brief asked to flag.
2. robots.txt permission is good-faith evidence but does not waive the
   posted contract terms; the two signals coexist uneasily, and courts have
   gone both ways on browsewrap reach for logged-out scrapers — **inferred**.
3. Mitigating facts, stated plainly: non-commercial private research;
   54 requests total at ≥2s spacing; robots-compliant; fail-closed politeness;
   nothing published. Enforcement against this profile is uncommon. RED here
   grades the TEXTUAL conflict and the fact that the roadmap intends to make
   this a RECURRING production dependency ("post-2024 replacement candidate",
   `docs/nflcom_injuries_sourcing.md:72-78`), which scales the exact activity
   the clause names.

**Verdict RED.** See remediation item 1. Nothing about this verdict requires
deleting existing data; it requires an owner decision before scaling the
pattern.

---

## Row 6 — nflverse maintained datasets

**Governing terms.** The automation repo that serves nflverse data releases
(`https://github.com/nflverse/nflverse-data`, fetched 2026-08-24) presents a
CC-BY-4.0 license — **measured** (repo sidebar/license link). The repo's own
governance doc already warns that "Most datasets are CC-BY 4.0; individual
upstream datasets can have different terms. Review the source-specific license
before redistributing a snapshot or using it commercially" — **read**:
`docs/data.md:109-115`. The pipeline consumes schedules/results/spreads/prices,
PBP, and participation through `nflreadpy` — **read**: `docs/data.md:11-23`.

**Analysis.** CC-BY 4.0 permits any use including commercial and
redistribution, conditioned on attribution. The public card publishes model
picks derived from these data rather than the data itself, which is well
inside the grant; adding an explicit attribution footer to the public pages
closes the last gap. Raw snapshots remain local-only per repo policy, stricter
than the license requires.

**Verdict GREEN.** Residual: upstream-dataset variance (e.g., NFL.com-derived
injury feeds historically distributed through nflverse carried the league's
underlying posture; that feed ended after 2024 — **reported**,
`docs/nflcom_injuries_sourcing.md:75-78`), which is exactly why row 5 exists.

---

## Row 7 — SEC availability reports (published Google Sheet)

**What we take.** Structured CSV exports of the availability-report sheet
embedded on secsports.com — sheet ID
`1m9NvaYU1N4ViI4MLrXLoTp5SdYYAp2tlWgxS5t9triM`, tabs readable unauthenticated
via `/pub?gid=<gid>&single=true&output=csv`; 137 tidy pilot rows across 3
point-in-time states; ≥2s polite delays; SHA-256 manifests — **read**:
`docs/sec_availability_pilot.md` (lines 33-42, 44-58, 129-134). The gated
third-party app behind the same page was probed and left alone when it
required accounts (401s respected) — **read**, same doc, "Structure found".

**Governing terms — measured 2026-08-24**, `https://policies.google.com/terms`
(effective July 30, 2026). Operative clauses, verbatim:

> "using automated means to access content from any of our services in
> violation of the machine-readable instructions on our web pages (for
> example, robots.txt files that disallow crawling, training, or other
> activities)" [listed under "Don't abuse our services"]

> "Other content: … some of our services give you access to content that
> belongs to other people or organizations … You may not use this content
> without that person or organization's permission, or as otherwise allowed
> by law."

**Analysis.** Reading a deliberately published CSV view is normal use of
Google Sheets' publish feature; nothing in the measured Google terms forbids
it, and the machine-readable-instructions clause only bites where robots-type
directives forbid it. The residual question is the content owner's (SEC's)
rights under the "Other content" clause; the sheet is published for public
consumption by the conference's own site, which is strong practical evidence
of intended public reading, but no SEC-side terms page was located and read
this session — **reported: none found; treat owner-permission as presumed-
from-publication, not verified**. Volume is trivial (~15 min/week at the
politeness floor if prospective collection proceeds — **read**, pilot doc).

**Verdict GREEN**, conditional on staying read-only-consumer of published
tabs and on attributing/asking before any published CFB availability product
quotes the sheet's contents directly.

---

## Answers to the specific flags requested

1. **Sources whose terms prohibit even personal research caching.** One:
   **NFL.com** — §1.3 bars systematic retrieval to compile a database "absent
   our express prior written consent," with no personal/non-commercial carve
   (verbatim above; **measured** 2026-08-24). Every other source either
   grants research use expressly (Odds API, nflverse, IA) or permits personal
   download/use while barring republication (VegasInsider §2).
2. **Where the PUBLIC dashboard could constitute republication.** Three lines
   not to cross: (i) shipping raw Odds-API quotes as downloadable archives
   (their Restrictions clause names "downloadable files" literally);
   (ii) displaying VegasInsider book-by-book historical boards or NFL.com
   report tables on the public site (VI §2/§16 "republish … publicly
   display"; NFL §1.1 copying/display prohibition). The current dashboard
   already confines itself to derived picks + one consensus line + our fair
   line (`public_board.py:1070-1074`, **read**), which keeps every row green
   on the republication axis.
3. **Purchased archive license scope.** Vendor = The Odds API Pty Ltd via its
   historical endpoint (not a bespoke signed license — the terms are
   site-posted browsewrap). No expiry clause; no deletion-on-termination
   clause (both **measured absences**, with the inference flagged);
   commercial use and dashboards/analytical tools expressly encouraged
   provided the data is not resold as the primary product; derived features
   in public docs are covered. Practical expiry risks are operational, not
   legal: subscription lapse ends *access* (new pulls), not possession.
4. **Wayback robots/ToS posture.** Grant is "for scholarship and research
   purposes only" plus a user certification of noninfringing/fair use; the
   2014 removal of the "noncommercial" word means commercial-adjacent private
   research is not excluded by that word anymore (**measured** via IA's own
   update post). Exclusions are applied at crawl time and retroactively at
   owner request (**measured** FAQ), so treat Wayback as ephemeral access,
   not storage.
5. **NFL.com terms on automated access.** robots.txt: `/injuries/` permitted,
   no crawl-delay (**measured** 2026-08-24). ToS: §1.3 systematic-retrieval
   ban (no consent held), §11(f) commercial-harvest ban, §1.1 copying/display
   prohibition, NY law + AAA arbitration — **measured** 2026-08-24. The
   ingester's fail-closed robots gate and ≥2s delay satisfy robots etiquette
   but do not address §1.3.

---

## Prioritized remediation list

1. **[RED — do first] NFL.com injuries: obtain consent or shrink the pattern
   before it becomes production.** Concretely: (a) pause any scheduled/
   recurring NFL.com fetching until the owner chooses among (i) requesting
   express written consent from NFL Enterprises legal for a low-volume,
   robots-compliant weekly fetch, (ii) narrowing to a single weekly
   Friday-designation page fetch with a documented personal-research rationale,
   or (iii) substituting a differently-licensed source for the post-2024
   injury feed; (b) keep the existing 2022–2024 snapshot local-only and out of
   every published artifact (already true); (c) never render NFL.com report
   rows/tables on the public dashboard. Owner decision required; this audit
   does not authorize or perform any change.
2. **[YELLOW] VegasInsider-derived history: write down its permanent
   constraints.** Add a retention note (local-only, no republication of board
   cells, cite capture timestamps when describing the dataset) next to
   `data/raw/vegasinsider/` documentation or in `docs/vegasinsider_backfill.md`.
   If a future publication needs 2005–2019 historical lines, prefer sources
   with explicit licenses (e.g., the SBR archive's posture, or a licensed
   feed) over VI-derived cells.
3. **[YELLOW] Wayback dependence: stop treating it as durable.** Preserve the
   existing immutable local snapshot dirs (already the pattern); for any NEW
   Wayback mining, keep the ≤2-season batching and ≥3s delay conventions and
   record CDX query + capture timestamps in the manifest (the VegasInsider
   script already does this — replicate everywhere).
4. **[GREEN housekeeping] The Odds API: pin the terms to the repo record.**
   Add the terms URL + 2026-08-24 access date (and the historical-endpoint
   pricing facts) to `docs/data.md`'s "Optional live odds" section; keep the
   quota ledger; note the impostor-domain warning; reaffirm that any future
   export/share feature must ship derived numbers, never raw quote archives.
5. **[GREEN housekeeping] nflverse attribution on public pages.** Add a short
   data-attribution line (nflverse, CC-BY 4.0, with link) to the GitHub Pages
   footer/template so the public card carries the license notice.
6. **[GREEN housekeeping] SEC sheet: verify before publishing.** Before any
   public CFB-availability output quotes the sheet's contents, locate and read
   the SEC-side terms (or simply attribute and link the public sheet); keep
   the ≥2s floor and the read-only-consumer posture; prefer prospective
   collection over deeper historical probing of the gated app.
7. **[process] Make this audit repeatable.** Re-run on vendor change or
   annually: refetch each terms page, hash/diff against the quotes recorded
   here, and re-grade RAG. Store fetched terms snapshots alongside the raw
   data manifests so clause drift is detectable. When done, the owner may mark
   ROADMAP MKT-09 ✅ with this document as the deliverable.

*RAG legend: GREEN = current use consistent with operative terms; YELLOW =
tension/ambiguity manageable under the current private-research posture;
RED = plain-text conflict with an operative clause that gates planned work.*

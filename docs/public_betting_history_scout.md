# Public-betting history scout: pre-2018 sources

Scope, per this task's package (WP20): **scout only** -- a source inventory
and a coverage report for what already exists, plus per-candidate-source
measurements of what a NEW source could add before it is built. No
experiment is run, no `registry/` file is written, no `src/nfl_ats` code is
touched. Every claim below is tagged **measured** (fetched or computed this
session, exact command/URL/path given), **read** (a file or artifact opened
this session), **reported** (a claim from an existing doc or this model's
own prior knowledge, unverified this session -- said out loud), or
**inferred** (reasoning, not evidence), per the binding `AGENTS.md` labeling
rule.

**Binding closing-grounds taxonomy, restated verbatim per `AGENTS.md` /
`CLAUDE.md`, for any subagent or later session that scores what this scout
finds:**

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

Nothing in this document adjudicates anything -- there is no effect to
close or keep here, only a source scout -- but the taxonomy is restated per
this task's own instruction, verbatim, for whatever later session runs
section 5's predeclared experiment.

---

## 0. The gap, precisely (read)

**Read**, `docs/public_betting_sourcing.md` (2026-08-20 session) and
`registry/weak_signals.json`: this project already has a working,
Wayback-backed historical public-betting archive --
`data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet`
-- built from `actionnetwork.com/nfl/public-betting`, 153 day-collapsed
Wayback captures, 2018 through 2026-08, 1,658 parsed game rows, 800 matched
to a REG-season schedule game within 72h. Five `public_betting_battery_*`
cells and six `movement_attribution_pop_*_public_*` cells are already
recorded from it (all `unresolved_below_power`, section 1 below).

**The gap is bounded from BOTH sides, measured this session:**

- **Lower bound: `game_features.parquet` (read, `data/processed/`)** covers
  seasons **2009-2026** (`season.min()==2009`, `season.max()==2026`,
  measured via `pandas.read_parquet(..., columns=["season"])`). A
  close-graded public-betting cell could in principle run on any of those
  seasons if a source existed.
- **actionnetwork's own archive starts 2018** (**measured**, section 1:
  earliest CDX capture `2018-09-02`, `manifest.json`). **Inferred**: this
  reads as a site-existence gap (Action Network's public-betting product did
  not exist yet), not a capture-density gap, since the very next capture is
  already in the site's mature template shape rather than a thin early
  version -- not independently confirmed against an Action Network company
  history this session. **2009-2017 is the one span no existing archive in
  this repository covers at all**, regardless of which explanation is
  right.
- **The opener-quote archive (`data/market/raw`, read via directory
  listing) starts 2020-08-25** (earliest directory:
  `20200825T115500Z-futures`) -- this bounds the OPENER grade specifically,
  independent of public-betting source coverage. A pre-2018 public-betting
  source cannot extend `movement_attribution_pop_*_public_*` (which needs a
  Tuesday-open quote, 2020+) further back than 2020 no matter what is
  found -- **restated so a later session does not expect otherwise**. It
  CAN extend a CLOSE-graded cell (like `public_betting_battery_*_close`,
  which only needs `game_features.parquet`'s own `spread_line`/`ats_margin`,
  present back to 2009) into 2009-2017, which is the honest target of this
  scout.

---

## 1. Inventory: what public-betting history already exists locally (measured)

### 1.1 `data/raw/public_betting/20260820T111148Z/` -- actionnetwork.com backfill

Read directly this session from `manifest.json` and `coverage_report.json`
(reproducing `docs/public_betting_sourcing.md`'s own numbers from the
artifact itself, not the prose):

| Field | Value |
|---|---|
| Source | `actionnetwork.com/nfl/public-betting`, via Wayback CDX (`web.archive.org`), never the origin site directly |
| CDX captures found (day-collapsed) | 153 |
| Parsed OK | 147 / 153 (96.1%) |
| Game rows parsed | 1,658 |
| Rows w/ >=1 non-null bet%/money% | 1,348 |
| REG-season rows matched to schedule (<=72h kickoff) | 800 |
| Seasons present | 2018-2026 (2026 preseason only) |
| Fields, era1 (2018-Oct/Nov 2022) | `spread_{side}_bet_pct`, `ml_{side}_bet_pct`, `total_{over,under}_bet_pct` -- **bet% (ticket) only, no money%** |
| Fields, era2 (Nov 2022-) | same bet% fields **plus** `spread_{side}_money_pct`, `ml_{side}_money_pct`, `total_{over,under}_money_pct` |
| Book | "Consensus" (dynamically resolved id, has been id 15 in every snapshot sampled) |
| Known gap | 2022-10-16 through 2023-10-16: 27 captures parsed OK but every game's `markets` object is empty (site-side, not this repo's parser) |
| Best-season coverage ceiling | 92/271 REG games (34%, 2022) with any pregame reading at all |
| Staleness | median 45.2h before kickoff (n=416 collapsed games), most readings land AFTER that week's Tuesday-noon line freeze but are still playable per this project's picks-lock-at-kickoff convention |

Per-season capture/row detail (**measured**, `coverage_report.json`
`per_season`, read directly this session):

| Season | Distinct captures | REG games in schedule | Games w/ >=1 pregame reading | ...before Tue noon ET | ...after Tue noon ET only |
|---:|---:|---:|---:|---:|---:|
| 2018 | 1 | 256 | 3 | 0 | 3 |
| 2019 | 5 | 256 | 31 | 0 | 31 |
| 2020 | 12 | 256 | 45 | 16 | 29 |
| 2021 | 14 | 272 | 81 | 10 | 71 |
| 2022 | 19 | 271 | 92 | 8 | 84 |
| 2023 | 9 | 272 | 51 | 5 | 46 |
| 2024 | 27 | 272 | 77 | 12 | 65 |
| 2025 | 6 | 272 | 36 | 12 | 24 |
| 2026 (through Aug, preseason only) | 2 | 272 | 12 | 12 | 0 |

### 1.2 `data/raw/public_betting_live/` -- prospective weekly capture

**Measured**, `capture_log.txt` (read in full this session): the
`public_betting_sat`/`public_betting_sun` scheduler jobs
(`scripts/capture_scheduler.py`, read) have run **7 times** since
2026-08-20, all `OK`, all `era2_scoreboard_response`, all 16/16 rows with
data. This is the "live captures" the task brief refers to as the
boundary -- everything before this cutover is backfill-only.

### 1.3 `data/raw/public_betting/20260820T111148Z/covers/` -- confirmed dead end

**Read**, `verification_summary.json` and `docs/public_betting_sourcing.md`
section 3: `covers.com/picks/nfl` is a handicapper-picks page (win-rate
badges, not a bet%/money% consensus), and the real consensus data at
`contests.covers.com/consensus/topconsensus/nfl/overall` is populated by
client-side AJAX the Wayback crawler never executed. Zero genuine
percentages found across 5 samples spanning 2016-2026 and both URLs. This
source is **not re-scouted below** -- re-fetching it a second time would
not change a structural (client-side-AJAX) finding. Its 331-capture CDX
inventory (2023-2026) is already cached at
`data/raw/public_betting/20260820T111148Z/covers/cdx_index.parquet` if a
future session ever finds the real JSON endpoint the page calls.

### 1.4 What the existing recorded cells actually used (read, `registry/weak_signals.json`)

| Cell family | Source | Seasons | Grade |
|---|---|---|---|
| `public_betting_battery_*` (5 cells) | actionnetwork index.parquet | 2018-2025 (opener cells 2020-2025) | close + opener |
| `movement_attribution_pop_*_public_*` (6 cells) | actionnetwork index.parquet, joined to `observed_movement_channel` | **2020-2025** | close-vs-open movement disagreement |

The `movement_attribution_pop_*_public_*` cells' season floor is **2020**,
not 2018 -- inherited from `observed_movement_channel`'s own dependency on
the opener-quote archive (section 0), not from actionnetwork's own 2018
start. A pre-2018 public-betting source cannot move that floor; only a
pre-2020 opener-quote archive could, which is out of this package's scope.

---

## 2. A sustained Wayback Machine outage blocked live scouting this session (measured)

**Measured**, repeated `curl` probes across roughly 17 continuous minutes
(two independent background poll loops, ~10 min then ~7 min, plus manual
checks interleaved between them, zero successes throughout) starting
2026-09-01 ~19:05 UTC: `https://web.archive.org/` and every
`https://web.archive.org/cdx/search/cdx?...` query attempted this session
returned `curl` exit 28 (connect-phase timeout, ~21s, never completing a TCP
handshake) against `web.archive.org`'s resolved IP, `207.241.237.3`
(`nslookup web.archive.org`, measured). This is a **full connect-level
failure**, not a slow HTTP response or a `503` -- distinct from, and more
severe than, the intermittent "Internet Archive: Temporarily Offline" banner
page `docs/sagarin_backfill.md` section 1 documented recovering from within
seconds in a prior session.

**Differential measured this same session, to rule out a purely local
network block**: `https://archive.org/` (a different IP, `207.241.224.2`,
same organization) and `https://archive.org/wayback/available?...` both
returned HTTP 200 normally throughout. `curl --resolve
web.archive.org:443:207.241.224.2 https://web.archive.org/cdx/search/cdx?...`
(forcing the Wayback hostname's TLS/Host onto the working `archive.org` IP)
completed the TCP/TLS handshake successfully but the shared frontend
answered with a bare **HTTP 500** for the CDX path -- i.e. the front door
that serves `archive.org` is up and recognizes the `web.archive.org`
hostname, but cannot reach whatever backend actually serves Wayback CDX/page
content right now. This is consistent with an Internet-Archive-side
incident affecting the `web.archive.org` cluster specifically, not a block
on this environment's egress (general internet access and `archive.org`
itself both work throughout -- **measured**, `https://www.google.com` and
`https://archive.org` both 200 at the same timestamps the CDX calls were
failing). `WebFetch` was also tried as a second network path and is
**hard-blocked for this host regardless of the outage** ("Claude Code is
unable to fetch from web.archive.org") -- not a usable fallback.

**Consequence for this task**: sections 2-5 below could not be filled with
this session's own live CDX enumeration and page fetches, which the task
explicitly requires ("measured, not just read"). Rather than substitute
priors dressed up as measurements, this section documents what IS known
(from this repository's own prior sessions, labeled **reported** and not
independently re-verified today) and gives the exact, ready-to-run commands
for the first session that finds this host reachable again. **No claim
below is mislabeled as measured** -- where this repo has no prior finding at
all for a candidate source (sportsbookreview.com, pregame.com,
teamrankings.com), this document says so plainly rather than filling the gap
with unlabeled priors.

---

## 3. Per-source verdict (reported/inferred only -- live scouting blocked, section 2)

| Source | Status this session | Prior finding (label) | What a follow-up must run first |
|---|---|---|---|
| `actionnetwork.com/nfl/public-betting` | Already fully backfilled 2018-2026 (section 1.1) | **measured**, prior session (`docs/public_betting_sourcing.md`) | Confirm no pre-2018 captures exist: `curl --compressed -L "http://web.archive.org/cdx/search/cdx?url=actionnetwork.com/nfl/public-betting&output=json&from=2005&to=2018&limit=20"` -- expected empty/near-empty, since the site itself launched ~2017-18, but not verified this session |
| `covers.com/picks/nfl` + `contests.covers.com/consensus/topconsensus/nfl/overall` | Confirmed dead end (client-side AJAX; static Wayback captures never contain real percentages) | **measured**, prior session, re-verified against the scout doc's own cited snapshot (section 1.3) | Not worth re-fetching -- the failure mode (client-rendered content) is structural, not a density problem. A real path would need the JSON endpoint the page's own JS calls, found via a live browser network trace, then a CDX check of whether THAT endpoint URL has ever been crawled |
| `vegasinsider.com` | **Not re-verified this session** (outage) | **reported**, `docs/archive/data_source_scout_v3.md` line ~106 (that document's own session labeled this "measured," but this session did not independently confirm it): "old URL scheme back to 2005, hundreds of captures/year through 2012" -- deep archive -- but "a measured spot check of a Jan-2019 snapshot found no percentage data at all (just layout CSS and a promo banner)." **Important gap in that prior check**: it sampled only ONE snapshot, on the NEWER (2019) URL scheme, not the OLD pre-2012 scheme the same note says is denser. The old scheme was never actually opened for content. This is the single highest-value re-check for a follow-up session -- the archive depth (2005+) would, if it actually contained percentages, be the only candidate source in this list that could reach the full 2009-2017 gap in one place. | `curl --compressed -L "http://web.archive.org/cdx/search/cdx?url=vegasinsider.com&matchType=domain&filter=original:.*(matchup\|consensus\|trend\|public).*nfl.*&output=json&limit=100&collapse=urlkey"` to find the actual old-scheme path, then fetch ONE capture from 2009-2012 specifically (not just 2019) and grep the body for `%` near team names |
| `oddsshark.com/nfl/consensus-picks` | **Not re-verified this session** (outage) | **reported**, `docs/archive/data_source_scout_v3.md` line 113: named as a "current-season-only free tool," found via search, "not fetched" -- i.e. that document itself never measured it, only repeated a search snippet. Site (OddsShark) did not launch until ~2011, so it cannot reach 2009-2010 regardless. | `curl --compressed -L "http://web.archive.org/cdx/search/cdx?url=oddsshark.com/nfl/consensus-picks&output=json&limit=200&collapse=timestamp:8"` -- first real measurement of this source, still unmade |
| `sportsbookreview.com` consensus/public-betting page | **No prior finding in this repository at all** | none -- not `docs/archive/data_source_scout_v3.md`, not `v4.md`, not `docs/public_betting_sourcing.md` name this source | Exact URL path unknown; needs a domain-wide CDX discovery query before a targeted enumeration is possible: `curl --compressed -L "http://web.archive.org/cdx/search/cdx?url=sportsbookreview.com&matchType=domain&filter=original:.*(consensus\|percent\|public).*&output=json&limit=100&collapse=urlkey"`. **Inferred, low confidence, not from any document**: this model's own general knowledge associates bet%/money% archives more with SBR's sister site `sportsinsights.com` (later "Sports Insights" / acquired) than with `sportsbookreview.com` itself, which is primarily a line-shopping/odds-comparison site -- worth trying both domains in the discovery query, but this whole paragraph is a guess, not evidence |
| `pregame.com` | **No prior finding in this repository at all** | none | Same domain-wide discovery pattern: `curl --compressed -L "http://web.archive.org/cdx/search/cdx?url=pregame.com&matchType=domain&filter=original:.*(public\|betting\|percent).*&output=json&limit=100&collapse=urlkey"` |
| `teamrankings.com` | **No prior finding in this repository at all** | none | Same pattern. **Inferred, low confidence**: TeamRankings' known public product is power ratings / against-the-spread performance stats, not a bet%/money% consensus tool -- this is the weakest-prior candidate on the list and should be tried last if fetch budget is limited |

**What this table is not**: a verdict. Per the task brief and the binding
taxonomy restated at the top of this document, nothing here is "closed" --
the sources with no prior finding are simply unmeasured, not negative, and
`vegasinsider.com`'s prior negative check is itself thin (one snapshot, one
era) and worth redoing properly before being treated as settled.

---

## 4. Recommended ingest design (predeclared script layout, NOT built this session)

Written so the first session that finds `web.archive.org` reachable can go
straight to fetching without a design pass. Mirrors
`scripts/ingest_sagarin_ratings.py` (**read** in full this session) --
the newest, most battle-tested Wayback ingestion script in this repo, with
three archive.org quirks already fixed and documented (gzip bodies needing
`--compressed`, same-timestamp redirects needing `-L`, a 0-byte cache
poisoning the resume check) that a new script should inherit rather than
rediscover:

- **File**: `scripts/ingest_public_betting_history.py` (new; does not touch
  `scripts/ingest_public_betting.py`, `scripts/ingest_sagarin_ratings.py`,
  or anything under `src/nfl_ats`).
- **`RateLimiter`/`_fetch`**: copy `ingest_sagarin_ratings.py`'s
  implementation verbatim (curl subprocess, `--compressed -L`, retry with
  backoff, `TRANSIENT_MARKERS = (b"Temporarily Offline", ...)` detection,
  empty-body-is-a-retryable-failure) rather than re-deriving it --
  **measured this session** (section 2) that the SAME failure category
  (connect-level timeout, not just a slow/banner response) can also occur
  and needs its own bounded-retry ceiling so a resumable run fails closed
  rather than hanging.
- **CDX enumeration**: `cdx_query()` pattern from `ingest_sagarin_ratings.py`
  -- fetch the full uncollapsed capture list per URL and collapse
  consecutive same-digest rows client-side (measured in that script:
  server-side `collapse=digest` caused timeouts on low-traffic URLs; this is
  the same CDX backend, so the same server-side cost is expected to apply
  here).
- **Cache location**: `data/market/public_betting_history/<UTC
  timestamp>/<source>/` (per this task's own instruction -- note this
  deviates from `ingest_public_betting.py`'s `data/raw/public_betting/`
  convention; both are gitignored, `data/raw/**` and `data/market/**`
  **read**, `.gitignore` lines 21/23). Layout per source:
  `cdx_index.parquet`, `raw_html/<timestamp>.html`, `index.parquet` (parsed
  rows), `manifest.json`. `resolve_snapshot_dir()` (copy
  `ingest_sagarin_ratings.py`'s version) makes `--snapshot` optional and
  auto-resumes the most recent existing snapshot directory.
- **Dedupe key**: `(source, capture_ts, site_game_id_or_team_pair)` --
  matching `ingest_public_betting.py`'s existing `index.parquet` row
  identity so a future merge of the two archives (2009-2017 new source +
  2018-2026 actionnetwork) is a straight `pd.concat`, not a reconciliation
  project. `html_path.exists()` is the resume/skip check per capture,
  exactly as both existing scripts already do.
- **Alignment rule to the per-game deadline**: reuse
  `scripts/injury_tuesday_cutoff_experiment.py`'s `team_week_tuesday_noon`
  convention verbatim (**read** this session, cited already in
  `docs/public_betting_sourcing.md` section 5) for the Tuesday-noon marker,
  but grade playability against the project's actual binding rule -- `MEMORY:
  picks lock at kickoff, min(own kickoff, Sunday 16:00 ET)`, not Tuesday --
  matching `docs/public_betting_sourcing.md` section 5's own two-column
  report (`before_tuesday_noon` / `after_tuesday_noon_only`, both counted as
  playable, neither discarded). For each `(season, week)` schedule row,
  join the single latest capture with `capture_ts` strictly before that
  game's own deadline.
- **CLI**: `--dry-run` (CDX enumeration + a written plan -- per-source,
  per-season capture counts and an estimated total fetch count -- no HTML
  fetched); `--fetch --max N` (fetch up to N new captures this run,
  resumable, sha256-cached per capture so a byte-identical re-fetch is
  detected and skipped even across snapshot directories --
  `hashlib.sha256(body).hexdigest()` stored per row in `index.parquet`,
  matching this task's own instruction and going one step further than
  either existing script, neither of which currently records a body hash);
  `--source {vegasinsider,oddsshark,sportsbookreview,pregame,teamrankings}`
  (repeatable, default: whichever section 3 recommends once measured);
  `--start-season`/`--end-season` (default 2009/2017, the actual gap).
- **Test**: `tests/test_public_betting_history_ingest.py`, mirroring
  `tests/test_sagarin_ingest.py`'s fixture pattern exactly -- real, trimmed
  HTML excerpts saved verbatim from genuine fetched captures under
  `tests/fixtures/public_betting_history/*.html` (header/table rows only,
  boilerplate dropped), parser functions tested directly with zero network
  calls, one fixture per era/template shape the real fetch turns up.

---

## 5. Predeclaration for the follow-up experiment (NOT run this session)

**Binding closing-grounds taxonomy, restated verbatim per `AGENTS.md` /
`CLAUDE.md`, per this task's own instruction, for whatever session actually
runs this:**

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

**Correction to this task's own framing, stated plainly (section 0):** the
task brief frames this gap as extending `movement_attribution_pop_*_public_*`
"to more seasons." **That specific family cannot be extended by a pre-2018
public-betting source** -- its population is bounded below by
`observed_movement_channel`'s own dependency on the Tuesday-opener quote
archive (`data/market/raw`, starts 2020-08-25, measured directory listing,
section 0), independent of how far back any public-betting source reaches.
The correct, extensible target is the **close-graded** cell family
(`public_betting_battery_*_close`), which only needs
`game_features.parquet`'s own `spread_line`/`ats_margin` -- present back to
2009 (measured, section 0) -- not a market-archive opener quote.

**Cells (frozen in construction now, before any new source's field
availability is known -- adapted per source once section 3 resolves)**:

1. **Source**: whichever candidate section 3 confirms usable (fields
   available -- at minimum `spread_{side}_bet_pct`; money% only if the
   source's era supports it, same era-conditional design as
   `ingest_public_betting.py`'s `era1`/`era2` split).
2. **Population**: new-source rows, seasons 2009-2017 (the actual gap),
   matched to `game_features.parquet`'s REG-season schedule on team-pair
   within 72h of kickoff (identical join method to
   `docs/public_betting_sourcing.md` section 5 and
   `docs/public_betting_battery_predeclaration.md` item 2 -- reused, not
   re-derived), one row per game (latest capture strictly before kickoff).
3. **Cell A -- fade-heavy-public, close-graded, 2009-2017**:
   `public_betting_history_fade_heavy_public_close`. Condition:
   `max(home_bet_pct, away_bet_pct) >= 70.0` (same 70% bar as
   `public_betting_battery_predeclaration.md` cell A, reused so the two
   eras are comparable, not re-chosen). Metric: forced-pick accuracy of
   "fade side covers" minus 0.50, week-blocked and season-blocked
   bootstrap, 20,000 resamples, **same seed `20260818`** this project's
   existing public-betting battery already used (reused, not re-picked, so
   this is directly comparable rather than a new arbitrary seed).
4. **Cell B -- union replication**: once cell A is scored on the new
   2009-2017 population alone, a SEPARATE pooled cell
   (`public_betting_history_fade_heavy_public_close_pooled_2009_2025`)
   combining the new rows with the existing 2018-2025 actionnetwork rows
   already used in `public_betting_battery_fade_heavy_public_close` --
   explicitly flagged in `--notes` as **not independently poolable** with
   the original 2018-2025 cell (same mechanism, overlapping construction,
   extended population) per `AGENTS.md`'s commensurability rule: same
   units (accuracy points), same population type (REG-season games, close
   grade, 70%-bet-share fade rule), family declared (`public_betting_history`)
   before any number from the new source is seen.
5. **Recording**: every cell via `nfl-ats weak-signals record
   --family public_betting_history --effect-units accuracy_points
   --closing-ground <omit unless BOTH week- and season-blocked intervals sit
   entirely below zero>`, `--probability-positive` from the week-blocked
   bootstrap (this project's primary block), `--season-start 2009
   --season-end 2017` for cell A, `2009`/`2025` for cell B. Default
   classification `unresolved_below_power` unless a cell actually clears one
   of the two admissible closing grounds -- not assumed, checked.
6. **What this is not**: not a rotation-registry confirmation look (a mined,
   backfill-quality battery on a source not yet even confirmed to exist, per
   this project's own convention for `public_betting_battery_*` above); does
   not call `nfl_ats.rotation.assign_window`/`record_look`.

No cell above is computed in this document. Section 3's outage means the
source itself is unconfirmed; this predeclaration exists so the FIRST
session that measures a usable source does not also have to design the
experiment from a blank page, and so the family is genuinely declared before
any sign is seen, per the project's own binding commensurability rule.

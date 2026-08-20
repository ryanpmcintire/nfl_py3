# USA Today NFL player-arrests sourcing

## Outcome

**Measured** (`scripts/ingest_player_arrests.py --snapshot
20260820T153000Z`): the anonymous paginated ingestion completed all 56 source
pages and cached 1,116 unique records dated 2000-01-24 through 2026-06-23.

**Measured** (`data/raw/player_arrests/20260820T153000Z/manifest.json`): the
source reported 1,116 total results at 20 rows per page, every page from 1
through 56 is cached, `complete=true`, and `resume_command=null`.

**Measured** (`git check-ignore -v
data/raw/player_arrests/20260820T153000Z/index.parquet`): the entire raw
snapshot is excluded by the repository's existing `data/raw/**` rule.

**Inferred**: this source is now ready for data-feasibility and entity-matching
work, but it is not an ATS signal and no model, pick, or prospective challenger
changed in this task.

## Anonymous access contract

**Measured** (live landing-page fetch on 2026-08-20):
`https://databases.usatoday.com/nfl-arrests/` returned HTTP 200 and embedded a
`sitedata` JSON object containing the AJAX URL, current nonce, page ID 10,
`sortBy=Date`, and `sortOrder=desc`.

**Measured** (live POST on 2026-08-20): the public
`wp-admin/admin-ajax.php` endpoint accepted `action=cspFetchTable`, the current
landing-page nonce, `pageID=10`, `blogID=''`, `sortBy=Date`, `sortOrder=desc`,
`page=1`, `searches={}`, and `heads=true` without authentication.

**Measured** (same response): the envelope returned `success=true`, 20 result
rows, field metadata, `totalResults=1116`, page size 20, and 56 pages.

**Read** (`scripts/ingest_player_arrests.py`): every run fetches the landing
page first, validates page ID and sort order, and uses the current nonce only
in memory for table POSTs.

**Read** (same script): the user agent is the generic project identifier
`nfl-ats-research/0.1 (private research ingestion)` and the default delay is
1.5 seconds between uncached page requests.

**Read** (same script): a cached raw page is parsed and skipped on resume; it
is never overwritten with a different response.

## Snapshot structure and integrity

**Measured** (`data/raw/player_arrests/20260820T153000Z/`): the snapshot contains
56 immutable `pages/page-NNNN.json` responses, a full archival `index.parquet`,
a restricted `incidents_point_in_time.parquet`, landing access checks, and
`manifest.json`.

**Measured** (`manifest.json`): SHA-256 hashes are recorded separately for all
56 raw JSON pages and both Parquet indexes.

**Measured** (`index.parquet`): all 1,116 `record_id` values are unique, every
row has an incident date and team, and 38 distinct source team strings occur.

**Measured** (`index.parquet`): source case labels comprise 938 `Arrested`, 80
`Charged`, 41 `Cited`, 20 `Surrendered`, 17 `Indicted`, 11 `Warrant`, four
`Summoned`, three `Detained`, one `Died`, and one `Jailed` rows.

**Read** (USA Today landing-page description, captured in the snapshot): the
database says it is assembled from media reports and public records and cannot
be considered fully complete.

## Point-in-time and leakage contract

**Read** (`scripts/ingest_player_arrests.py`): `incident_date` is the only
source field assigned availability-date semantics.

**Read** (same script): the full archival index retains source narrative,
outcome, and links under names ending in `_archive_only` so their status is
visible at the schema boundary.

**Read** (same script): `incidents_point_in_time.parquet` contains only record
ID, incident date, player name, team, position, case type, and category; it has
no outcome, resolution, description, or link column.

**Measured** (`tests/test_player_arrests_ingest.py`): mutating Outcome,
Description, and Links leaves the point-in-time view bit-identical.

**Inferred**: because USA Today exposes no historical revision timestamps,
outcome/resolution text is retrospective and forbidden as a model feature;
description and links are also excluded from the safe view rather than assumed
to have existed on the incident date.

**Inferred**: a future feature may use the incident event only after matching
the player/team to the schedule and proving the incident was publicly known
before that game's prediction timestamp; the date-only source does not prove
an intra-day publication time.

## Nonce handling correction

**Measured** (`tests/test_player_arrests_ingest.py`): new landing captures
replace the parsed nonce with `[REDACTED_EPHEMERAL_NONCE]` before writing HTML.

**Measured** (`manifest.json`): `access.nonce_stored=true` for this particular
snapshot because its first two landing checks were captured before sanitizing
was added; the final landing check is sanitized and its hash is recorded.

**Read** (`manifest.json`): the stored note identifies those earlier values as
anonymous and ephemeral and states that the nonce is never copied into the
manifest.

**Inferred**: retaining the two pre-correction landing checks preserves the
immutable raw audit trail while making the manifest truthful; all newly created
snapshots sanitize from their first request.

## Resume and reproduction

```powershell
$env:UV_CACHE_DIR='F:\Repos\nfl_py3\.uv-cache'

# New immutable snapshot; fetches a current landing nonce, then all pages.
.\.tools\uv.exe run --no-sync python scripts\ingest_player_arrests.py

# Resume a partial snapshot; valid cached pages are parsed and skipped.
.\.tools\uv.exe run --no-sync python scripts\ingest_player_arrests.py `
  --snapshot 20260820T153000Z

# Bounded access smoke test for a new snapshot.
.\.tools\uv.exe run --no-sync python scripts\ingest_player_arrests.py `
  --max-pages 2

# Parser, access-contract, resume, hash, nonce, and leakage tests.
.\.tools\uv.exe run --no-sync pytest tests\test_player_arrests_ingest.py -q
```

**Measured** (`tests/test_player_arrests_ingest.py`): the test suite covers
landing configuration parsing, exact POST fields, changed-sort refusal,
response-page validation, outcome leakage exclusion, nonce sanitization, and a
two-stage partial/resume run that fetches only the missing page.

# SEC availability pilot (scout v5 Section D #2) — ingest + parse

Run 2026-08-22. Scope per instruction: fetch the 2024 archive page, enumerate
weekly reports, download up to 6 weekly reports (polite delay >=2s, SHA-256
manifest), parse to tidy rows (season/week/date/team/player/status), normalize
team names to CFB codes with an unmapped report, coverage stats per week.
No registry writes, no ATS evaluation.

Canonical artifacts: `artifacts/sec_pilot/20260822T140443Z/` (final run) and
`data/raw/sec_availability/20260822T140443Z/`. An earlier same-day run
(`...135810Z`) is superseded — its parser double-counted two Wayback wrappers
carrying the same table; raw snapshots in it are still valid.

## Structure found

**HTML, not PDFs** — and not a page of weekly links either (all **measured**
this session):

1. `secsports.com/fbreports` and `/fbreports-archive` are Inertia/Vue CMS
   pages containing **zero** `.pdf` hrefs. Both saved to the run's raw dir.
2. `/fbreports` embeds two carriers:
   - an iframe to `confinjrepxyz.hdintelligence-app.com?source=SECreports`
     (the archive page uses `?source=SECarchive`). This is a third-party React
     app that gates rendering on `document.referrer`, resolves an organization
     via `POST /api/V$3rs1 {userEmail}` (returns `404 {"organization":null}`
     for any address not registered — measured), and serves per-league JSON
     endpoints whose obfuscated paths are embedded in its JS chunks. Every
     league-archive endpoint we could reach returned empty (`[]` / `{}`)
     unauthenticated, including `GET /api/archive` and `POST /api/S3C$v8W!s$i0N
     {"organization":"SEC"}` with the API key shipped in the client bundle.
     The school-level workbooks it links are all HTTP 401 without login.
   - a published Google Sheet iframe:
     `docs.google.com/spreadsheets/d/1m9NvaYU1N4ViI4MLrXLoTp5SdYYAp2tlWgxS5t9triM`.
3. That master sheet has 8 tabs (measured from the pubhtml TOC):
   InitialReport, ThursdayUpdate, FridayUpdate, GamedayUpdate,
   Formatted-Data, FORMATTED-PUBLIC, "2024 Football Schedule",
   SchoolWorkbooks. **Every tab is readable as CSV unauthenticated** via
   `/pub?gid=<gid>&single=true&output=csv` (measured; the `/pubhtml` HTML view
   now renders as an empty JS shell).

So the ingestible surface is structured CSV plus Wayback point-in-time
captures — no PDF parser was needed at any step.

## Weeks captured: 3 distinct point-in-time states (target was up to 6)

| State | Source capture | Window | Week | Game date | Matchup |
|---|---|---|---|---|---|
| 1 | Wayback `20240905205904` (pubhtml) | initial report | 2 | 2024-09-07 | South Carolina @ Kentucky |
| 2 | Wayback `20241005020434` (gid=1425434111) | Friday update | 6 | 2024-10-05 | 5 SEC games |
| 3 | Live sheet tabs, fetched today | initial/thu/fri/gameday | 15 | 2024-12-07 | Georgia vs Texas (SEC CG) |

The sheet updates **in place**: each window tab holds only the most recent
week written, and the sheet was never touched after 2024-12-07 (the 2025+
season moved into the gated app). The only recoverable history is therefore
Wayback captures of the sheet URL itself — CDX lists exactly 3 captures for
the whole sheet prefix, one of which duplicates another's table content after
wrapper-stripping (deduped by parsed-content hash). 6 weeks do not exist on
this channel; stopping there honestly rather than widening scope.

## Rows parsed and coverage (all measured, `coverage_stats.json`)

137 tidy rows total (`tidy_rows.parquet` / `.csv`; schema: source,
captured_at_utc, season, week, game_date, side, window, updated_stamp_raw,
team_raw, team_code, player, position, status_raw, status_norm).

- **Week 2** — 19 players, SCAR+UK; statuses: out 7, probable 5,
  questionable 4, doubtful 3; stamp "Initial Report: Updated as of 9/04 at
  7:10 PM CT".
- **Week 6** — 72 players across 10 teams (ALA ARK AUB MISS MIZ SCAR TAMU
  TENN UGA VAN), 5 games; statuses: out 38, probable 23, questionable 11;
  Friday-update state.
- **Week 15** — 46 rows, TEX+UGA across four window tabs plus the formatted
  gameday grid; statuses: out 25, questionable 15, game-time-decision 6;
  stamp "Gameday Update: Updated as of 12/07 at 1:45 PM CT".

Unmapped team names: **none** (empty unmapped report). Normalization covers
all 16 current SEC members; e.g. Ole Miss→MISS, Mississippi State→MSST,
Texas A&M→TAMU. Codes are stable short forms, chosen to match common SEC
abbreviations; the local canonical table
(`data/processed/cfb_game_features.parquet`) carries CFBD display names
("Alabama", "Mississippi State", ...) — read this session — so any future join
maps code/display-name explicitly rather than assuming.

Season/week/date were resolved against the sheet's own **"2024 Football
Schedule" tab** (66 rows, weeks 2-15 incl. the championship), i.e. the
mapping is the source's own, not inferred from calendar arithmetic. Week 1
(non-conference) is absent from that tab by construction.

Statuses observed: out / probable / questionable / doubtful / Game Time
Decision — normalized lowercase, raw preserved. The scout doc's claim of
Wed-initial → Thu/Fri updates → final ~90min pregame is **consistent with**
the tab structure and stamps (**read** off artifacts above), but only the
championship week has all four windows in hand.

## Point-in-time assessment

- **measured**: every row carries `captured_at_utc` (Wayback timestamp or live
  fetch time) and, where present, the sheet's own update stamp
  (`updated_stamp_raw`). Wayback timestamps make states 1-2 PIT-A (archived
  before kickoff); the live week-15 state is a retrospective copy of what was
  published before the 2024-12-07 kickoff, but its capture is today — treat it
  as PIT-B until a contemporaneous capture confirms it.
- **inferred, low risk**: the in-sheet stamps ("Updated as of ...") are
  produced by the SEC's reporting workflow, so window ordering within a week
  should be reliable; this has not been cross-verified against an independent
  record.

## What would unlock full coverage

1. A registered email accepted by the hdintelligence org lookup (gives the
   archive API) — requires an account decision by the owner, not scraping.
2. Prospective collection from the public CSV tabs each Wed/Thu/Fri/Sat going
   forward (cheap, fully PIT-A, ~15 min of requests/week at the >=2s floor).
3. Periodic Wayback/archivetoday saves as belt-and-braces if prospective
   collection lapses.

## Gates (scoped)

- `ruff format --check scripts\pilot_sec_availability.py docs`: pass (repo-wide
  `ruff format --check .` crashes on unrelated permission-denied directories,
  reproducible without my changes).
- `ruff check .`: pass. `mypy src` (MYPYPATH=src): pass, 101 files.
- `pytest --basetemp ...\pt_sec`: **1707 passed, 1 failed** — the failure is
  `test_every_script_writing_artifacts_json_uses_the_provenance_helper`
  flagging `arctic_shift_gate.py` and `recurrence_hazard_features.py`, two
  pre-existing scripts owned by other workstreams; my script is not flagged
  and no file I own is implicated.

## Provenance legend

- **measured** — fetched/parsed this session; path given above.
- **read** — opened the artifact just now.
- **reported** — external/doc claim not verified locally.
- **inferred** — reasoning or design, explicitly not evidence.

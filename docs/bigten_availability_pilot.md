# Big Ten availability pilot (XLG-07 unblocking) — snapshot delivery

Run 2026-08-22. Scope per instruction: fetch the 2023 hub page, enumerate PDF
links, download up to 6 weekly PDFs (polite delay >=2s, SHA-256 manifest),
then parse text **only if** a PDF parser was already installed. No registry
writes, no ATS evaluation, nothing added to `pyproject.toml`/`uv.lock`.

## Outcome at a glance

| Step | Result | Provenance |
|---|---|---|
| Hub page fetched live | HTTP 200, 1,647,965 chars | measured (`scripts/pilot_bigten_availability.py`, this session) |
| Weekly PDF links enumerated | **14 weekly links** (Weeks 1-14); hub also lists 3 bowl links (`Bowls_ALL`, `Bowls_MICH_CFP` x2) out of 17 total `.pdf` hrefs | measured (regex over saved `hub_2023.html`) |
| PDFs downloaded | **6 of 6 requested** (Weeks 1-6, 212-218 KB each), all carry `%PDF-` headers, SHA-256 recorded | measured (`data/raw/bigten_availability/20260822T125805Z/manifest.json`) |
| Text parsing | **NOT attempted — stopped at snapshot delivery.** No `pdfminer`/`pypdf`/`PyPDF2`/`PyMuPDF`/`pdfplumber` present in the locked env (measured via `importlib.util.find_spec`); per scope no dependency was added | measured |
| Tidy rows / XLG-03 join / coverage stats | **not computed** — blocked one step upstream by parsing; designed only (below) | — |

Artifacts:

- `data/raw/bigten_availability/20260822T125805Z/` — gitignored raw snapshot:
  `hub_2023.html`, `2023_week_01.pdf` ... `2023_week_06.pdf`, `manifest.json`
  (per-file source URL, sha256, size, `%PDF-` check, fetch timestamp).
- `artifacts/bigten_pilot/20260822T125805Z/pilot_status.json` — machine status
  with provenance stamp, outcome `snapshot_only_no_pdf_parser_installed`.
- `scripts/pilot_bigten_availability.py` — reproducible fetcher
  (`--max-pdfs`, `--delay-seconds` floor 2.0s, resumable manifest layout).

## Source pages (all fetched this session unless tagged)

- 2023 hub: `https://bigten.org/fb/article/blt19caa3aea8cf4525/` — "2023
  Football Availability Reports", Weeks 1-14 + FCG/Bowls + CFP entries.
  Measured live.
- Policy announcement:
  `https://bigten.org/fb/article/blt2856785fb75ee868/` — states schools submit
  gameday availability "no later than two hours before scheduled kickoff
  times," distributed on BigTen.org/FBReports. Measured live this session.

## Point-in-time assessment (evidence in hand)

- **measured**: the conference's own policy text commits to filing >=2h before
  scheduled kickoff. That makes the archive point-in-time-A *by policy*.
- **inferred, unverified against artifact content**: whether each individual
  PDF carries its own publication timestamp (a `date_on_pdf` field) is unknown
  until parsing runs. If the PDFs are undated, PIT rests on the policy claim
  alone plus the hub's weekly grouping — weaker, and the tidy schema below
  should record `date_on_pdf` as nullable so the gap stays visible.
- **reported, unverified**: the scout doc's claim that these are game-level
  rows with statuses available/probable/questionable/doubtful/out
  (`docs/data_source_scout_v5.md` Section D #1). Not yet checked against any
  downloaded file's actual content.

## Join attempt — NOT run (designed)

Blocked by missing parser. Design recorded so the next session can execute
without re-deciding anything:

- Source rows (intended): `(season=2023, week, date_on_pdf?, away, home,
  player, position, status)` per PDF row.
- Target: `data/processed/cfb_game_features.parquet` (12,500 games, 2006-2025;
  measured this session: columns `season, week, home_team, away_team,
  home_conference, away_conference, ...`; 2023 has all 14 Big Ten members with
  short names, e.g. "Penn State", "Ohio State").
- Join key: normalized team name on both sides of each game row
  (`str.casefold()`, strip mascot suffixes); match counted separately for
  home-side, away-side, and game-level. Match rate will be reported honestly
  including unmapped-name lists — no silent drops.
- Coverage stats intended: games covered per week, status distribution per
  team-game, share of clean-core (2012-2019, 2021-2025) B1G-vs-B1G games with
  >=1 flagged player.
- Honest caveat now, before any number exists: 6 weekly PDFs cover at most ~60
  B1G games (~120 team-games) — far below any single-cell detection power in
  this project; the pilot measures *ingest feasibility*, never an effect.

## Cheapest-experiment statement (designed, NOT run)

Once parsing works and the join rate is known: compare line movement /
spread error between **flagged** (>=1 `out`/`doubtful` designated starter on a
team) vs **unflagged** B1G games — using the existing CFB close-proxy spread in
the XLG-03 table, week-blocked bootstrap, candidate arm vs market-only
baseline, identical windows, per-season stability reported. This screen has
NOT been executed and no effect estimate, interval, or sign exists yet. Per
AGENTS.md, whatever it returns (including an interval crossing zero) would be
recorded via `nfl-ats weak-signals record`, not judged by significance.

## Provenance legend

- **measured** — run/fetched this session, command or path given above.
- **read** — opened a file/page just now.
- **reported** — scout-doc or search-snippet claim, unverified locally.
- **inferred** — design/reasoning, explicitly not evidence.

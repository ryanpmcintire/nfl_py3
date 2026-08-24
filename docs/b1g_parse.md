# Big Ten availability reports — PDF ingest + parse (`b1g_parse`)

Run 2026-08-24. Scope per instruction: add the PDF parser dependency, implement
the parser for the already-scouted snapshot, validate yield with a hand
spot-check, document the schema/join key. **Source ingestion + validation only:
no model features, no ATS screen, no experiment row, no tracked-registry write.**

Precedent: the SEC pilot ingests via a public Google Sheet
(`docs/sec_availability_pilot.md`) — no parser needed there. The Big Ten channel
is genuinely PDF-based and was blocked only on parsing ("one uv add away",
`docs/bigten_availability_pilot.md`). This session removed that block.

## Source (read this session)

- Snapshot: `data/raw/bigten_availability/20260822T125805Z/` — 6 weekly PDFs
  (2023 Weeks 1–6) + `hub_2023.html` + SHA-256 `manifest.json`, fetched by
  `scripts/pilot_bigten_availability.py` on 2026-08-22.
- Origin hub: `https://bigten.org/fb/article/blt19caa3aea8cf4525/` lists 14
  weekly report links (Weeks 1–14) plus 3 bowl links — only Weeks 1–6 are in
  the local snapshot.
- Policy context (reported by `docs/bigten_availability_pilot.md`, quoting the
  conference's announcement): schools submit gameday availability no later than
  two hours before scheduled kickoff; reports are distributed on
  BigTen.org/FBReports. Point-in-time-A rests on that policy claim plus the
  hub's weekly grouping.

## Dependency added (measured)

- `pypdf>=6.16.2` via `uv add pypdf` → installed **pypdf 6.16.2**;
  `pyproject.toml` + `uv.lock` updated. No other dependency changes.

## Parser

`scripts/b1g_parse.py`. Input: a pilot snapshot dir (default: newest under
`data/raw/bigten_availability/`); verifies every file's SHA-256 against the raw
manifest before parsing (mismatch = loud failure). Unit tests:
`tests/test_b1g_parse.py` (13 tests over synthetic pages replicating the real
layouts and extraction artifacts).

## Layout notes (all measured against the six snapshot PDFs)

Two layout generations, both handled:

1. **Standard game page** (Weeks 1–6, one page per B1G school): header line,
   `Week N: <date range>` line, all-caps matchup line (`ILLINOIS vs. Toledo`,
   `NEBRASKA at Minnesota` — the B1G school is always listed first), a
   kickoff/broadcast line, then designation sections of `<number> <name>`
   player lines under `OUT` / `QUESTIONABLE`; an empty section reads `None`.
   From Week 4 on, some names carry a trailing `(Season)`/`(season)`
   annotation (25 occurrences across the six files).
2. **Bye page** (Weeks 5–6): school name alone, a `BYE` line, and both
   designation headers collapsed onto one empty `OUT QUESTIONABLE` line
   (5 bye pages in the snapshot).

Only OUT and QUESTIONABLE appear as designation headers anywhere in the six
files (measured across all 84 pages); no DOUBTFUL/PROBABLE rows exist here.

Extraction artifacts handled explicitly:

- pypdf splits a capital **T** (53×) or **Y** (3×) from the rest of the word
  ("T yson Rooks"); repaired conservatively — only those two measured letters,
  and only when followed by a lowercase letter or period. The raw extracted
  name is always preserved in `player_raw`.
- Curly apostrophes (U+2019) normalized to `'` in `player` (raw preserved).

Fail-loud contract: any page not matching a known layout — bad header, missing/
mismatched Week line, unknown section header, unparseable player line, unmapped
team, missing designation section — raises `PageParseError`, is recorded in
`parse_report.json` → `failures`, and makes the script exit nonzero. Nothing is
silently dropped.

## Schema / join readiness

One row per (week, B1G team, listed player, designation). Columns
(`tidy_rows.csv`/`.parquet`):

| Column | Meaning |
|---|---|
| `season` | 2023 (snapshot's season; the week lines carry no year) |
| `week` | Big Ten week number, cross-checked against the manifest |
| `team_code` | **stable join key** — see table below |
| `cfb_display_name` | canonical CFB display name matching `data/processed/cfb_game_features.parquet`'s short names (e.g. "Penn State"), so a future join maps explicitly rather than assuming |
| `team_raw` | all-caps name exactly as printed |
| `opponent_raw` | opponent as printed (None on bye pages) |
| `venue_side` | inferred from the vs./at convention (`vs.`→home, `at`→away) — **inferred**, unverified against actual home/away records |
| `player_number` | jersey number as printed |
| `player_raw` / `player` | extracted text / artifact-repaired name |
| `annotation` | `season` where printed, else null |
| `designation_raw` / `designation_norm` | `OUT`/`QUESTIONABLE` / lowercase |

Team crosswalk (14 members of the 2023 Big Ten; codes are stable short forms
chosen like the SEC pilot's):

ILL ILLINOIS · IND INDIANA · IOWA IOWA · MD MARYLAND · MICH MICHIGAN ·
MSU MICHIGAN STATE · MINN MINNESOTA · NEB NEBRASKA · NU NORTHWESTERN ·
OSU OHIO STATE · PSU PENN STATE · PUR PURDUE · RUTG RUTGERS · WISC WISCONSIN

Unmapped team names encountered: none (measured — all 84 pages mapped).

## Yield (measured, run `artifacts/b1g_parse/20260824T111154Z/`)

- Files: **6 parsed / 6 total, 0 failed**; every SHA-256 matched the raw manifest.
- Pages: **84/84 parsed** (79 game pages + 5 bye pages), 0 parse failures.
- Rows: **607** player designations — out **473**, questionable **134**;
  season annotations **25**.
- Per week (rows / distinct teams): W1 120/14 · W2 112/13 · W3 108/14 ·
  W4 99/14 · W5 77/12 · W6 91/11 (W5/W6 dip = bye teams emit no rows).
- Spot-check (measured — each row re-derived against its PDF page text):
  W1 ILL #9 "Tyson Rooks" OUT (raw "T yson Rooks", repair verified);
  W1 WISC #26 Grady O'Neill QUESTIONABLE (apostrophe normalization verified);
  W4 ILL #2 Matthew Bailey OUT + `(Season)` annotation lifted;
  W6 IOWA #12 Cade McNamara OUT + `(season)` lifted;
  W5 MINN page 7 OUT list (#1 Darius Taylor [repaired], #7 Chris Autman-Bell,
  #34 Jack Tinnen, #36 Jackson Powers, #47 Hayden Schwartz, #68 Jackson
  Ruschmeyer) all match verbatim. **10 designations checked, all correct.**

## Known gaps

- **Coverage**: 6 of the hub's 17 linked reports (2023 Weeks 1–6 only).
  Extending is a fetch problem (`scripts/pilot_bigten_availability.py
  --max-pdfs`), not a parser problem. 2024+ seasons need their own hub URLs.
- **No publication timestamp on the PDFs** (measured — no per-report stamp
  beyond the week-range line): point-in-time rests on the ≥2h-before-kickoff
  policy claim alone. Any future feature must treat these as gameday states.
- **Designation poverty**: only OUT/QUESTIONABLE; no probable/doubtful tier and
  no position/starter information whatsoever, so "flagged *starter*" definitions
  would need an external importance source.
- `venue_side` is convention-inferred (see schema note above).
- The T/Y split-letter repair is evidence-bounded to this snapshot's fonts; a
  future season could introduce other split letters (they fail nothing — raw is
  preserved and the clean column just keeps the space until the letter set is
  re-measured).

## What a future availability-feature lane would need (predeclared, NOT run)

Nothing below has been executed; it is the recorded prerequisite list.

1. Complete the archive: fetch Weeks 7–14 + bowls of 2023, then discover 2024+
   hubs; re-run this parser per snapshot (it takes any run directory).
2. Predeclare the experiment family BEFORE any screen: candidate definition
   (e.g. team flagged with ≥1 `out`), window, blocking, baseline (market-only),
   and per-season stability reporting — per the designed-but-not-run statement
   in `docs/bigten_availability_pilot.md`. Whatever the result — including an
   interval crossing zero — goes through `nfl-ats weak-signals record`;
   probability_positive is the reported quantity, never "contains zero".
3. Join mechanics: match `cfb_display_name` against both sides of
   `cfb_game_features.parquet` games; report home-side/away-side/game-level
   match rates honestly including unmapped-name lists (no silent drops).
4. Pooling with SEC rows only after declaring commensurability (same units,
   scale, population) and the family before signs are seen.
5. Power honesty first: ~60 team-games/week max means the historical arm alone
   stays far below single-cell detection power; the pool/sign accumulation path
   is the reason the ingest is worth completing despite that.

## Gates (scoped)

- `ruff format --check` + `ruff check` scoped to owned files: pass.
- `mypy src`: pass. Full `pytest --basetemp ...\pt_b1g`: pass (see final report).
- Provenance: `metadata.json` stamped via `artifact_provenance()` +
  `write_experiment_artifact()`; the experiment row is redirected inside the
  gitignored artifact snapshot (`experiment_registry/experiments/b1g_parse/`)
  — no tracked `registry/` write (verified via git status this session).

## Provenance legend

- **measured** — produced/verified this session; command or path given above.
- **read** — opened the artifact/file just now.
- **reported** — doc claim not independently re-verified this session.
- **inferred** — reasoning/design, explicitly not evidence.

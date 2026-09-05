# VegasInsider half-line archive (LEAD-60 build)

**ENG-40 correction (2026-09-05):** the FULL-GAME `season_<year>.parquet`
tidy table (not this document's half-line table) had a parser bug: a total
(O/U) value was misfiled into `spread_line` for 155 rows, all season 2009
(measured: `spread_line > 0` or `abs(spread_line) > 30` across every
`season_<year>.parquet` in `artifacts/vegasinsider_backfill/20260822T033952Z/`).
Root cause and fix live in `scripts/backfill_vegasinsider.py::classify_line_tokens`
(a board-cell token-order/sign-convention bug, fixed with a sign rule, not a
magnitude filter — see that function's docstring). All 12 seasons were
rebuilt from the same cached HTML (network hard-blocked throughout); measured
2026-09-05, zero rows across the archive now violate the favorite-side/
magnitude convention. **This document's own tables are for the HALF-LINE
archive (`half_lines_<year>.parquet`), which was never affected** — the
ENG-40 bug and fix live entirely in `build_tidy`/`classify_line_tokens`, a
different code path from `build_half_lines`/`extract_book_half_lines`; a
full-table diff (`half_lines_<year>.parquet`, all 12 seasons) confirms 0
rows changed, and the coverage numbers in this document (rows,
half1/half2-with-spread counts, join rate) are therefore unchanged and were
re-verified against the rebuilt `coverage_<year>.json` files. See
`docs/lead02_half_line_script.md` for the downstream LEAD-02 screen's
updated results (both legs join against the now-fixed full-game table).

Follow-up to `docs/vi_first_half_probe.md` (2026-09-04, GO). That probe found
the cached 2005–2016 VegasInsider backfill snapshot
(`data/raw/vegasinsider/20260822T033952Z/`) carries `1st Half`/`2nd Half`
material and recommended parser work only, no new fetches. This document
covers the build: what exists, its real (measured) schema, coverage, and how
to load it. **No ATS screen has been run against this archive.** It exists to
unblock LEAD-02 (1H/full-game script disagreement) and an unproposed 2H
sibling; both remain `🔬`/unscreened.

## Correction to the 2026-09-04 probe (read the source before trusting the summary)

The probe's coverage section said the movement pages carry "Fav/Dog spreads
plus Over/Under totals" for both halves. **Measured 2026-09-05** (byte
inspection of all 165 cached `line_movement/*.html` files): that is wrong for
the half columns specifically. Every one of the 160 files with a real
half-column header has this exact 12-column layout:

```
Date | Time | ML-Fav | ML-Dog | Spread-Fav | Spread-Dog | Total-Over | Total-Under | 1H-Fav | 1H-Dog | 2H-Fav | 2H-Dog
```

The full game gets a spread AND a total. **Each half gets a spread only** —
there is no half total column anywhere in this cache, and no price/juice
suffix on the half cells either (full-game cells read `IND-18.5 -110`; half
cells read `IND-11.5`, `GNB -11`, `PHI +.5`, or a withdrawn-line placeholder
`XX`/`PK`, never a trailing price). The probe's "Over/Under totals" line was
an unverified inference from the nav bar's label text, not from reading a
populated cell. Treat this document's numbers as the corrected reference.

A second correction: the "board snapshot" files
(`data/raw/vegasinsider/<run-id>/snapshots/*.html`, the multi-book grid pages)
do **not** embed half data at all. Their `1st Half` text (176/189 files) is
only a nav-bar **link** to a separate VegasInsider URL
(`/nfl/odds/las-vegas/first-half/`) that this backfill never fetched (0 CDX
entries for that path — checked, see below). All actual half-line data comes
from the per-book **line-movement** pages
(`data/raw/vegasinsider/<run-id>/line_movement/*.html`), which is a
completely different set of cached files, already fetched for the existing
book-anchor-name resolution step.

## What was built

`scripts/backfill_vegasinsider.py::build_half_lines` re-parses the SAME
cached `line_movement/*.html` files already used for book-name resolution
(`fetch_book_map`/`extract_anchor_names`) — no new fetch, no new cache. One
line-movement page covers **every book's** movement history for one game (not
one book per page): each book's section is anchored by
`<a name="X">BOOK NAME LINE MOVEMENTS</a>` followed by its own Date/Time/…/2H
table. The parser:

1. Reads the page's own title (`<font size=4>Away Team @ Home Team</font>`),
   `Game Date:`, and `Game Time:` fields for matchup/date identity —
   independent of the board-page rotation codes, since these pages spell out
   full franchise names ("Indianapolis Colts", not "IND"). A new
   `FULL_TEAM_NAME_TO_CODE` table maps all 33 distinct strings measured across
   the 165 files (32 franchises; `"N.Y. Giants Giants"` is a VegasInsider
   page-title concatenation artifact for the Giants, not a 33rd team) onto
   the same codes the full-game tidy table already uses (`LAR` covers both
   St. Louis- and Los Angeles-era Rams).
2. Splits the page into per-book sections and walks each book's movement rows
   in document order (oldest → newest, as rendered). For each half
   independently, the **last row with a usable value wins** — mirroring how
   the full-game board cell already reports only the book's current line, not
   its history. A cell reading `XX` (withdrawn/never posted) is `None`; `PK`
   is `0.0`; a signed number (`-11.5`, `+.5`, `-11`) parses directly. When the
   Fav cell is unusable, the Dog cell's value is negated as a fallback.
3. Emits one row per **capture × matchup × book × half**, `half` in `{1, 2}`,
   matching the full-game tidy table's join keys
   (`capture_ts, game_date, away, home, book`) so the two tables join
   directly.

Full-game output is untouched: `build_tidy`, `parse_board`, and everything
that feeds `season_<year>.parquet` was not modified. The half-line build runs
strictly after `tidy.to_parquet(...)` and writes to a new, separate file.

**Proof of no regression (measured 2026-09-05):** ran the pre-LEAD-60 script
(`git show HEAD:scripts/backfill_vegasinsider.py` at the commit before this
change) and the current script over the identical cached inputs for season
2006, into two isolated output directories, with network calls hard-blocked
in both runs (`fetch_via_curl` monkeypatched to raise before any subprocess
call — this build reads local files only, per the task constraint). Result:
`season_2006.parquet` is byte-identical between the two runs
(sha256 `5a3bc186...df120d` both sides, `cmp` reports no differences); the
`coverage_2006.json` files are identical except for the new, additive
`half_lines` key. Every other season's full-game parquet was regenerated by
the same unmodified code path, so this is a structural guarantee (no shared
function was edited), not just a one-season spot-check.

### Provenance

`half_lines_<year>.parquet` is stamped via
`nfl_ats.provenance.stamp_sidecar()` — a `<path>.provenance.json` sidecar
beside each parquet (code revision, dirty flag, row count, season), the same
pattern `docs/script_contracts.md` sanctions for tabular writers. The
existing `season_<year>.parquet`/`coverage_<year>.json` write pattern and the
run-level `metadata.json` (`write_experiment_artifact`) are unchanged.

### Network discipline for the real run

The 12-season "real" build below was executed by calling `process_season()`
directly for each season with `fetch_via_curl` monkeypatched to raise
immediately — this guarantees zero subprocess/socket calls, never merely
relying on a fetch failing or timing out. This matters because, empirically,
the **vanilla, unmodified** script also attempts network fetches when
book-map resolution needs a `(capture_ts, line-movement-link)` pair that
wasn't cached under that exact key during the original 2026-08-22 backfill
(a normal, pre-existing fallback path — those captures already fall back to
`header_text_columns`/`cross_capture_fallback` book-name resolution in
production, see `coverage_<year>.json["book_identity"]`). Blocking the fetch
outright reproduces the same fallback outcome deterministically (proven
byte-identical above) without ever opening a network path.

## Schema

`artifacts/vegasinsider_backfill/<run-id>/half_lines_<year>.parquet`:

| column | type | notes |
|---|---|---|
| `capture_ts` | str (14-digit) | Wayback capture timestamp of the **board** snapshot this line-movement page was linked from — identical join domain to the full-game tidy table's `capture_ts`. Always the snapshot's own filename prefix, never a row-internal history date (leakage/point-in-time test: `tests/test_vegasinsider_half_lines.py::test_capture_ts_is_the_filename_timestamp_not_a_row_date`). |
| `game_date` | str (ISO date) | From the page's own "Game Date:" field. |
| `away`, `home` | str | Franchise code, from the page title via `FULL_TEAM_NAME_TO_CODE`. |
| `kickoff_time` | str or null | From "Game Time:"; informational only, not a join key. |
| `book` | str | Book name from the section anchor (e.g. `CAESARS`, `HILTON`) — same naming as the full-game tidy table's `book` column. |
| `half` | int | `1` or `2`. |
| `spread_line` | float or null | Signed half spread (favorite convention: negative = favored), or null if withdrawn/never posted. |
| `total_line` | float or null | **Always null in this cache** — no half total market exists in the source data (see correction above). Column kept for forward compatibility / future sources. |
| `spread_price`, `total_price` | float or null | **Always null in this cache** — no price/juice is ever quoted on a half cell in this source. Kept for the same reason. |

## Coverage (measured 2026-09-05, all 12 seasons, `run-id 20260822T033952Z`)

Command: `process_season()` called directly per season 2005–2016 (network
blocked as described above); source: `nfl-ats` repo,
`artifacts/vegasinsider_backfill/20260822T033952Z/coverage_<year>.json["half_lines"]`.

| season | half rows | half1 w/ spread | half2 w/ spread | distinct (capture,book) keys | join rate vs full-game tidy |
|---|---|---|---|---|---|
| 2005 | 0 | 0 | 0 | 0 | n/a (pre-dates the half-line feature on VI's pages) |
| 2006 | 120 | 53 | 36 | 60 | 55.0% |
| 2007 | 350 | 48 | 10 | 175 | 45.7% |
| 2008 | 406 | 87 | 54 | 203 | 38.9% |
| 2009 | 256 | 71 | 82 | 128 | 29.7% |
| 2010 | 508 | 234 | 139 | 254 | 38.2% |
| 2011 | 464 | 139 | 77 | 232 | 50.9% |
| 2012 | 418 | 160 | 120 | 209 | 54.5% |
| 2013 | 162 | 77 | 56 | 81 | 51.9% |
| 2014 | 480 | 183 | 66 | 240 | 43.8% |
| 2015 | 620 | 144 | 3 | 310 | 43.6% |
| 2016 | 606 | 289 | 162 | 303 | 38.6% |
| **total** | **4,390** | **1,485** | **805** | **2,195** | **43.6% (958/2,195)** |

Notes:

- `half rows` always splits evenly into half1/half2 counts (2,195 each) — the
  builder emits a row for both halves per (capture, book) found in a movement
  page, with `spread_line = null` when that half was never usable, so the
  count reflects "books observed at all", not "books with a usable line".
  `rows_with_half1_spread`/`rows_with_half2_spread` above are the usable
  subset.
- 1H is populated roughly 2x as often as 2H (1,485 vs 805 of 2,195) — 2H
  markets are quoted/retained less consistently by these books across this
  window. This is descriptive, not a finding; no direction is claimed.
- **2005 has zero half rows.** This is a real absence, not a parser miss:
  none of 2005's line-movement pages have a half-column header at all (the
  feature didn't exist on VI's pages yet that season) — consistent with all
  13 "no `1st Half` nav link" board snapshots also being 2005/early-2006
  dated (see below).
- **Join rate against the full-game tidy rows (43.6% overall, 958/2,195):** a
  half-line (capture, matchup, book) key is present in the full-game tidy
  table's own key set 43.6% of the time. This is expected, not a defect —
  many books quote a full-game line without ever quoting (or retaining) a
  half line, so the half archive is necessarily a subset of the full-game
  book set, not a 1:1 mirror.
- **5 of 165 line-movement files parse as `unparsed_line_movement_files`**
  (all 2011-dated) — these are mis-fetched VegasInsider homepages (a
  different Wayback redirect target than the requested line-movement URL),
  not real movement pages; `parse_line_movement_page` returns `None` for
  them and they contribute zero rows, which is the correct behavior (not a
  parser bug — verified by reading the raw HTML, it's literally a college
  basketball betting-trends widget).

### The 13 board-snapshot files with no `1st Half` nav link (deliverable #2)

Classified via `classify_missing_half_nav_boards`: checked all 189 cached
board-snapshot files across the 12 seasons; **13 have no `1st Half` text**,
and **all 13 (13/13) classify as `layout_variant_legacy_board`** — every one
predates VI's addition of the half-odds nav link to their board template
(capture timestamps span 2005-10-01 through 2006-09-05; the `oddsText`
CSS-class grid the modern board layout uses is absent from all 13, confirming
they use the older "legacy" board template `parse_board_legacy` already
handles). **Zero classify as `genuinely_absent`** — there is no modern-layout
board snapshot in this cache missing the nav link. This is purely a fact
about the board pages' own nav bar, though: as established above, that nav
link never carried embedded half data anyway, so it does not gate anything in
this archive — the actual half-line coverage comes entirely from the
line-movement pages tabulated above.

## How to load it

```python
import pandas as pd

RUN_ID = "20260822T033952Z"
half = pd.concat(
    pd.read_parquet(f"artifacts/vegasinsider_backfill/{RUN_ID}/half_lines_{year}.parquet")
    for year in range(2006, 2017)  # 2005 has no rows
)
full = pd.concat(
    pd.read_parquet(f"artifacts/vegasinsider_backfill/{RUN_ID}/season_{year}.parquet")
    for year in range(2005, 2017)
)

# Join a half spread onto its full-game row for the same capture/matchup/book:
joined = full.merge(
    half[half["half"] == 1][
        ["capture_ts", "game_date", "away", "home", "book", "spread_line"]
    ].rename(columns={"spread_line": "half1_spread_line"}),
    on=["capture_ts", "game_date", "away", "home", "book"],
    how="left",
)
```

## What this unblocks

- **LEAD-02** (1H/full-game script disagreement): the gating half-line
  archive now exists. LEAD-02 remains `🔬` (unscreened) — this document does
  not run or claim any ATS result.
- **The unproposed 2H sibling** the original probe flagged (a 2H-line
  cousin of LEAD-27's 3rd-quarter-adjustment mechanism): same archive, `half
  == 2` rows, also unscreened.
- Any future half-line work should budget for the join-rate ceiling above
  (43.6% overall, book- and season-dependent) rather than assuming full
  full-game coverage.

# VegasInsider Wayback backfill, 2005-2016 REG seasons

Status: COMPLETE (all 12 seasons). Run id `20260822T033952Z`. No registry
writes; raw snapshots are local-only under `data/raw/vegasinsider/20260822T033952Z/`
(gitignored). Tool: `scripts/backfill_vegasinsider.py`, adapted from the 2011
pilot (`scripts/pilot_vegasinsider_wayback.py`) with its CDX query, fetch,
manifest, parser core, and team normalization reused.

## What was produced

Per season, under `artifacts/vegasinsider_backfill/20260822T033952Z/`:

- `season_<year>.parquet` — tidy rows, columns exactly:
  `capture_ts, game_date, away, home, kickoff_time, book, spread_line, total_line`.
  One row per (capture, game, named book cell). Anchor-less open-line and
  offshore-comparison affiliate columns are excluded (counted per season in
  coverage JSON as `unanchored_cells_excluded`).
- `coverage_<year>.json` — parse/cell/book-identity/schedule-match stats.
- `status.json` — consolidated completion state.

## Measured coverage by season

`sp`/`tt` = share of tidy rows carrying a spread / total. `own` = captures whose
book names came from their own line-movement page; `hdr` = captures named from
board header text (2005-era layout); `fb` = captures using cross-capture anchor
fallback. `match` = board game instances matched to the local nflverse schedule
within +/-1 day by matchup codes; `Tue/Wed` = share of REG schedule games with at
least one Tuesday/Wednesday-dated capture containing them. Seasons 2005-2008
predate the local schedules snapshot (starts 2009), so match stats are
unavailable there.

| season | tidy rows | sp | tt | own | hdr | fb | no-map | match | Tue/Wed |
|--------|-----------|------|------|-----|-----|----|--------|-------|---------|
| 2005 | 1,084 | 0.742 | 0.953 | 0 | 12 | 0 | 0 | n/a | n/a |
| 2006 | 1,311 | 0.893 | 0.893 | 5 | 0 | 9 | 2 | n/a | n/a |
| 2007 | 1,524 | 0.868 | 0.879 | 16 | 0 | 4 | 0 | n/a | n/a |
| 2008 | 1,529 | 0.887 | 0.893 | 18 | 0 | 0 | 0 | n/a | n/a |
| 2009 | 770 | 0.934 | 0.664 | 10 | 0 | 0 | 0 | 0.705 | 0.090 |
| 2010 | 1,343 | 0.923 | 0.918 | 15 | 0 | 0 | 0 | 0.944 | 0.188 |
| 2011 | 1,877 | 0.985 | 0.963 | 19 | 0 | 1 | 0 | 0.987 | 0.613 |
| 2012 | 1,697 | 0.899 | 0.938 | 17 | 0 | 0 | 0 | 1.000 | 0.141 |
| 2013 | 561 | 0.893 | 0.945 | 6 | 0 | 0 | 0 | 1.000 | 0.086 |
| 2014 | 1,727 | 0.885 | 0.962 | 15 | 0 | 1 | 0 | 0.973 | 0.648 |
| 2015 | 2,050 | 0.890 | 0.949 | 20 | 0 | 0 | 0 | 0.971 | 0.602 |
| 2016 | 2,175 | 0.912 | 0.976 | 19 | 0 | 0 | 0 | 0.925 | 0.352 |

Books detected per season (distinct named books): 8 in 2005 (Stardust,
Caesars-Hilton, Stations, MGM-Mirage, Leroy's, Imperial Palace, Harrah's-Rio,
Palms-LV); 6-7 through 2010 (Caesars, Hilton, Leroy's, Mirage, Stations, Wynn,
plus LVSC/LVSC-consensus variants); 6-8 from 2011 on (Harrah's, LVH/LV Hilton,
Mirage-MGM, Stations, Wynn, VI Consensus, William Hill, Cantor/CG Technology,
Westgate Superbook).

## Structural notes (measured)

Three distinct board layouts appear across the window, all handled by
`scripts/backfill_vegasinsider.py`:

1. **Legacy header layout (2005 through Sep 2006):** book columns carry TEXT
   headers in `<td class='odds'><b>...</b></td>` cells; each game row's book
   `<td>`s map positionally to those headers. Cells hold total first
   (`46½o/u`) then spread (`-3.0`, vig optional). No line-movement pages exist;
   identity comes from page headers, not fallback. The Sept 2006 capture in the
   CDX is an unparseable variant (0 games) and one Sept 2012 capture likewise;
   both are recorded as parse failures rather than silently dropped.
2. **oddsText layout (Oct 2006 - 2011):** the pilot's structure verbatim
   (`<tr class='oddsText_odd|even'>`, image book headers, movement-page anchor
   fragments `#J`, `#BT`, ...). Two adaptations: line-movement links may be
   UNQUOTED (`href=/nfl/...`), so link extraction no longer depends on quoting;
   and from 2006-2010 the per-week movement URL differs per game, so up to 8
   distinct paths are tried per capture before falling back to the cross-
   capture anchor union.
3. **viCell layout (Sep 2012 onward):** `<td class="viCellBg1|2 ...">` game
   blocks with rotation-number + bold-link team cells and `width="56"` book
   cells holding total then spread tokens (`47u-10` / `-1&nbsp;-10`); anchors
   and movement pages work exactly as in the pilot.

Token classification is layout-aware: signed numbers become spreads, bare or
`o/u`-suffixed numbers >= 15 become totals (2006-era boards print the total
unsigned FIRST, which defeats the pilot's greedy spread-first tokenizer).
Spread tokens with `.0` decimals (`-18.0 -110`) are accepted.

Game-date year rollover (December boards captured in January) is resolved by
choosing the candidate year closest to the capture date.

## Reduced-confidence flag

The only season flagged under the >20% fallback rule is **2006** (fallback rate
0.643: 9 of 14 modern-layout captures needed cross-capture anchor fallback
because most early per-game line-movement pages were never archived). 2007 is
under the threshold (4 fallback captures, 16 own). All later seasons rely on
their own movement pages almost exclusively. Book NAMES for flagged seasons are
still anchored-derived; the flag marks lower per-capture independence, not
guessed identities.

## Dispersion feasibility

Book-disagreement sd per game IS computable from this backfill. Requirements
are >=2 named book quotes for the same (game, line type) in one capture; every
season provides that:

- Named books per season: minimum 5 (2009), maximum 8 (2005, 2013, 2014).
- Typical books-per-game distribution on well-captured seasons: 6-8 books for
  the large majority of game instances (e.g., 2011: 134 instances with 7 books,
  7 with 6, plus a small tail of early-week thin boards).
- Caveats: 2009 totals are thinner (total coverage 0.664) and 2009/2013 have
  few archived captures overall (10 and 6 unique-digest captures exist in the
  CDX for those windows — all were used), so within-week dispersion estimates
  there rest on fewer snapshots. 2005 spreads are absent on ~26% of rows
  (cells often carried a total with no posted spread yet).

Feasibility statement: compute per-(capture, game) sd across named books for
spread_line and total_line separately, on any season 2005-2016; expect usable
sample sizes everywhere except early-week cells, with 2009 and 2013 the thinnest
and 2011/2014/2015 the strongest windows (many Tue/Wed captures, high line
coverage).

## Reproduction / resumption

All seasons are complete; nothing remains. To re-run or extend (idempotent per
season — existing `season_<year>.parquet` files are skipped, cached raw
snapshots are reused):

```powershell
.\.tools\uv.exe run --no-sync python scripts/backfill_vegasinsider.py `
    --run-id 20260822T033952Z --seasons <years>
```

Constraints honored per invocation: <=2 seasons, >=3s polite delay between
archive fetches, 35-minute wall-clock cap with clean stop, status JSON, exit 0.

Historical accuracy caveat: this dataset is pre-game market lines, not picks;
nothing here speaks to ATS edge.

## 2026-09-05 - Rerun after point-in-time fix 42d78f6

**Decision (inferred):** I think the corrected 1H and 2H directions remain challenger leads; the close-graded rerun does not establish an opener-card change.

**Measured:** both cached archives were rebuilt offline into versioned directories under `artifacts/vegasinsider/cx7_42d78f6_main/` and `cx7_42d78f6_pilot/`, preserving the old tables; total rows stay 4,706, 1,392 spread values change, 15,421 future movements are rejected, 0 movements have unparseable timestamps, and 640 rows are marked in-play (`artifacts/vegasinsider/cx7_42d78f6_audit/cache_comparison.json`). **Measured:** main-archive spread availability changes from 2,290 to 1,223 of 4,390 rows; pilot availability changes from 137 to 79 of 316 rows, with the pilot's old values reconstructed in memory using the pre-fix unfiltered movement selection (`cache_comparison.json`). **Measured:** no tidy rows are dropped, unavailable spread values remain null, and 8 non-movement files are unparsed (5 main, 3 pilot; `cache_comparison.json`).

**Measured:** after excluding in-play observations, the unchanged LEAD-02 screen reads +5.2928 accuracy points, 95% [-2.5132, +12.6042], probability_positive=0.90925 on 48 games for 1H; its 2H sibling reads +16.6667, [0.0000, +25.0000], probability_positive=0.92352 on 4 games (`artifacts/vegasinsider/cx7_42d78f6_lead02/results.json`). **Read:** these replace the earlier ENG-40 rescreen's +1.4778/70-game and +7.8098/38-game headline reads (`artifacts/vegasinsider/cx7_42d78f6_audit/superseded_registry_entries.json`); the full old/new era table and unavailable 2H-era estimates are in `docs/lead02_half_line_script.md`'s dated rerun section. **Read:** split-half reliability was not estimated by that screen; it is not recorded as zero.

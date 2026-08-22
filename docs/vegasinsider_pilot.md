# VegasInsider Wayback pilot — feasibility report

Written 2026-08-22. Pilot of `scripts/pilot_vegasinsider_wayback.py` against the
Section F lead 1 source (`docs/data_source_scout_v5.md`). Provenance tags inline
per the project rule: **measured** (run this session, command/artifact given),
**read** (file opened this session), **inferred** (reasoning), **reported**
(unverified). No ATS evaluation was run and no tracked registry was written.

## Verdict up front

**GO for the full 2005-2016 backfill. Effort estimate M** (measured basis
below; one bounded pre-2010 layout probe recommended first, see risks). The
2011 boards parse at 100% page success with named multi-book spreads AND totals,
zero silent team mappings, and honest null handling.

## Method

1. CDX query `vegasinsider.com/nfl/odds/las-vegas/`, window 20110901-20111231,
   statuscode 200 filter, dedup by digest (**measured**: 53 unique-digest
   captures in the window; artifact `data/raw/vegasinsider/20260822T015704Z/cdx.json`).
2. Selected 14 evenly-spaced captures, preferring Tuesdays/Wednesdays within
   each slot (measured selection in the manifest below).
3. Fetched each via the Wayback raw endpoint (`web/{ts}id_/{original}`) through
   curl with a 3.0 s self-imposed delay and retry-once (**measured**: 14/14
   succeeded, 98-118 KB each).
4. Per capture, fetched one line-movement page to recover the anchor-to-book
   map (see structure below); retry across up to 5 distinct movement links.
5. Parsed boards to tidy rows; wrote sha256 manifest, tidy CSV, normalization
   report, feasibility JSON, and provenance-stamped metadata into
   `artifacts/vegasinsider_pilot/20260822T015704Z/`.

Run ID: `20260822T015704Z`. Re-run any phase offline with
`--run-id 20260822T015704Z --skip-fetch`.

## Measured results

Captures selected (all fetched HTTP 200, sha256-manifested):

```
20110921183627  20111005193511  20111005204835  20111012194937
20111019235815  20111026112359  20111102063341  20111116130921
20111123132337  20111129024621  20111206192307  20111213131021
20111220114653  20111228051432
```

| Metric | Value | Source |
|---|---|---|
| Pages fetched / failed | 14 / 0 | measured |
| Pages parsed (games extracted) | 14 / 14 (parse_success_rate 1.0) | measured |
| Tidy rows (game x book-cell) | 1,798 (~112-144/page) | measured |
| Games parsed per page | 13-17 | measured |
| Named books detected | 7: VI CONSENSUS, LV HILTON, MIRAGE-MGM, LEROYS, WYNN, HARRAH'S, STATIONS (~11-17 cells/book/page) | measured |
| Spread coverage over anchored book cells | 99.32% | measured |
| Total coverage over anchored book cells | 96.75% | measured |
| Books consistent across pages | y (same 7-name set on all 14 pages after recorded fallback fill) | measured |
| Team slots mapped / total | 3,596 / 3,596 (unmapped values: none) | measured |
| Sample verbatim row | `20110921183627, 2011-09-25, San Francisco @ Cincinnati, LV HILTON, spread -2.5 vig -110, total 40.5u-110, raw "40.5u-110 \| -2.5 -110"` | measured |

Cross-validation: the 2011-11-06 Atlanta @ Indianapolis VI CONSENSUS cell parses
to spread -7 / total 45, matching the raw HTML read directly this session
(**read**, snapshot 20111102063341). Kickoff dates/times and rotation numbers
(away = lower rotation listed first) are carried verbatim; the visitor-first
convention is confirmed on every sampled row but is layout convention rather
than a documented spec (**inferred** from 14/14 consistent orderings).

## Board structure (what made parsing possible)

Measured on snapshot 20111102063341 (**read**):

- Each game is `<tr class='oddsText_odd|even'>`: a nested info table
  (`<b>MM/DD  H:MM PM</b>`, then `<b>ROT&nbsp;<a>Team</a></b>` twice) followed
  by one `<td width=N class='oddsText...'>` per sportsbook column.
- Cell tokens: spread `-7-110`, total `45u-110` (o/u marks the juiced side);
  half-points render as `&frac12;` entities; missing cells are genuinely blank
  or carry only one market.
- **Surprise:** book names are NOT text — the header is an image
  (`/images/odds_vegas_*.png`) plus an imagemap naming only open/consensus/
  Sportsbook.com. Recoverable anyway: every book cell links to
  `/nfl/odds/las-vegas/line-movement/<game>#CODE`, and those pages define
  `<a name="CODE">` immediately ahead of `<BOOK NAME> LINE MOVEMENTS`
  headings. One movement fetch per capture therefore names all columns
  (BT=VI CONSENSUS, J=LV HILTON, N=MGM/MIRAGE-MGM, L=LEROYS, AA=WYNN,
  E=HARRAH'S, X=STATIONS).
- Columns WITHOUT anchors (an open-line cell and an affiliate/"Picks" cell)
  keep `book_name=null`; no guessed labels anywhere.

## Structural surprises and honest caveats

- The hover-note span sits BEFORE the odds cells inside the info `<td>`
  (**read**); a naive truncation dropped most rows on the first parser attempt.
  Fixed by selecting book cells directly by their `width=N class=oddsText`
  signature. This is the kind of trap that makes a per-era smoke test mandatory.
- Early-season boards name fewer books: the Sept 21 board's own movement page
  defined only 3 anchors; the rest were filled from the cross-capture union map
  (anchor codes stable across all 14 captures — **measured**; fill recorded in
  `feasibility.json` as `book_map_source`).
- Two mid-holiday captures (20111220, 20111228) had ALL their line-movement
  pages missing from Wayback (HTTP 404 / non-board shells — **measured**); they
  run entirely on the cross-capture fallback. Inferred risk: some captures may
  have anchor codes absent from every other capture; those would stay null-
  named rather than guessed.
- `spread_line` is the displayed FAVORITE-side spread; which team it applies to
  is not stated in the cell and is not inferred here. Side attribution needs a
  money-line/LM join in a later step (**read** of token format; attribution
  deliberately deferred).
- Pre-2010 layouts are UNVERIFIED (**reported** density only; not probed this
  session). The parser targets the 2010+ layout shape; a 2005-2009 probe could
  reveal different markup, which is why backfill effort is M not S.

## Go/no-go and cost reasoning

GO, effort **M**. Basis: full ingest cost is ~1 CDX query + ~1.2 fetches per
capture (boards + movement maps) at >=2.5 s politeness; 2011 alone held 53
unique-digest captures (**measured**), so 2005-2016 plausibly yields several
hundred captures -> a few hours of polite fetching plus disk well under 100 MB
(**inferred** from measured per-file sizes of ~100 KB). Parsing is already
working end-to-end at the measured rates above. Recommended sequence:

1. Bounded probe: fetch + parse ONE capture each from 2005-2009 seasons before
   any bulk run (layout-drift check; cheap).
2. Bulk backfill season-by-season with the same manifest/provenance pattern;
   keep per-capture `book_map_source` recording.
3. Only after ingestion: opener-vs-consensus dispersion features, evaluated
   chronologically against the SBR opener/close baseline. Nothing here yet
   speaks to predictive value; this session establishes access and parsing
   feasibility only (**inferred** planning, no evaluation run).

## Gates run this session

- `ruff format --check` + `ruff check` on the script: clean (**measured**)
- `mypy src`: Success, no issues in 101 source files (**measured**)
- `pytest --basetemp C:\Users\Ryan\AppData\Local\Temp\opencode\pt_vi`:
  1692 passed (**measured**)

## Artifacts

- Raw snapshots + manifest: `data/raw/vegasinsider/20260822T015704Z/` (gitignored)
- Feasibility JSON, tidy CSV, normalization report, metadata:
  `artifacts/vegasinsider_pilot/20260822T015704Z/` (gitignored)

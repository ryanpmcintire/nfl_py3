# FantasyFootballCalculator ADP — sourcing and feasibility

Written 2026-08-22. Ingest of `docs/data_source_scout_v5.md` Section C rank 2
(FantasyFootballCalculator historical ADP REST API). Scope here is ingestion +
feasibility only; no ATS screen was run.

## Source and access

- Endpoint (verified live 2026-08-22, HTTP 200):
  `https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams=12&year=YYYY&position=all`
  with `scoring` ∈ {`ppr`, `standard`}. Free, keyless. The vendor requests
  attribution; the string recorded in every manifest is: *"ADP data provided by
  FantasyFootballCalculator.com ... free public API; the vendor requests
  attribution for use of its data."* (**measured**, live fetch this session)
- Each response is JSON: a `meta` block (`type`, `teams`, `rounds`,
  `total_drafts`, `start_date`, `end_date`) plus a `players` list (`name`,
  `position`, `team`, `adp`, `times_drafted`, plus high/low/stdev/bye).
- Politeness: ≥1.5 s self-imposed delay between requests (no published
  Crawl-delay found for the host this session). Fetches via curl subprocess,
  matching the repo's Sagarin-ingest precedent.
- Ingest script: `scripts/ingest_ffc_adp.py`. Resume-capable: rerun with
  `--snapshot <ts>` skips any response already on disk (**measured**: second run
  reported all 32 combinations "cached", zero network).

## Measured coverage

Snapshot `data/raw/ffc_adp/20260822T004750Z/` (**measured** from its
`manifest.json`; per-response sha256 recorded there):

| Year | ppr drafts | ppr window | standard drafts | standard window |
|---|---|---|---|---|
| 2010 | 898 | 09-03 → 09-08 | 1,535 | 09-06 → 09-08 |
| 2011 | 600 | 09-06 → 09-09 | 1,144 | 09-07 → 09-09 |
| 2012 | 303 | 09-04 → 09-05 | 507 | 09-04 → 09-05 |
| 2013 | 764 | 09-02 → 09-04 | 992 | 09-02 → 09-04 |
| 2014 | 635 | 08-31 → 09-01 | 752 | 08-31 → 09-01 |
| 2015 | 844 | 09-06 → 09-09 | 822 | 09-06 → 09-09 |
| 2016 | 956 | 09-01 → 09-02 | 701 | 09-01 → 09-02 |
| 2017 | 685 | 09-03 → 09-04 | 1,384 | 09-01 → 09-04 |
| 2018 | 2,494 | 09-01 → 09-04 | 1,698 | 08-28 → 09-04 |
| 2019 | 2,167 | 09-02 → 09-04 | 696 | 09-02 → 09-04 |
| 2020 | 2,403 | 08-30 → 09-01 | 2,667 | 08-25 → 09-01 |
| 2021 | 1,709 | 08-31 → 09-01 | 2,656 | 08-28 → 09-01 |
| 2022 | 1,633 | 09-03 → 09-04 | 2,112 | 08-31 → 09-04 |
| 2023 | 3,146 | 08-30 → 09-01 | 1,104 | 08-30 → 09-01 |
| 2024 | 1,371 | 08-31 → 09-01 | 742 | 08-30 → 09-01 |
| 2025 | 8,470 | 08-25 → 09-01 | 2,017 | 08-25 → 09-01 |

All 32 requested combinations (16 years × 2 formats) returned data; zero
failures. Total mock drafts captured across both formats: **50,607**
(**measured**, sum of `meta.total_drafts`). Player rows in the tidy table:
6,193. Draft volume is thinnest 2011-2017 (~300-1,400/format/year) and densest
2023+.

## Point-in-time assessment

What the meta stamps prove (**read**, from the captured responses):

- `start_date`/`end_date` are the EXACT window over which the returned ADP was
  computed — e.g. 2020 ppr = mocks run 2020-08-30 through 2020-09-01. This is a
  self-consistent provenance stamp on each aggregate, which is what earns the
  scout's grade A: we know precisely when the crowd expectation was formed, and
  it is always pre-Week-1 or Week-1 (latest window end observed: 2011-09-09).
- Every window falls in late August / early September of its season, i.e. all
  snapshots are draft-season baselines formed BEFORE most NFL games are played.
- What the stamps do NOT prove: that archive values are frozen. The API returns
  whatever the vendor currently holds for a given (scoring, year); if the
  vendor ever recomputed history, a later refetch could differ. Mitigation:
  this snapshot is immutable on disk with per-response sha256 in
  `manifest.json`; downstream work must cite the snapshot dir, not re-fetch.

## Feasibility transforms (artifacts/ffc_adp/20260822T004750Z/)

- `adp_tidy.parquet`: one row per player-year-format — (year, scoring, player,
  position, team, adp, times_drafted, window_start, window_end) plus
  normalized `team_code`.
- `team_top8_feasibility.parquet`: per (year, scoring, franchise_code): count,
  mean ADP and min ADP of the roster's top-8 players by ADP (times_drafted > 0
  only), plus mean times_drafted. 1,013 rows. Honest caveat: only 231/1,013
  rows have the full 8 players (mean n_top8 = 5.83) because the returned list
  depth varies (~180-250 players) and weak fantasy rosters legitimately have
  fewer drafted players; min_adp_top8 remains well-defined throughout.
- `normalization_report.json`: mapping audit.

### Team-name → franchise normalization (honest accounting)

FFC already emits CURRENT franchise codes even in archived seasons — measured:
2010 rows carry `LAC`/`LAR`/`LV`, not SD/STL/OAK — so mapping is a 32-code
passthrough plus a small alias table (`JAC→JAX`, `ARZ→ARI`, `OAK→LV`, `SD→LAC`,
`STL→LA`, `WSH→WAS`) that fired zero times in this snapshot.

Row-level ambiguity (**measured**): 100/6,193 rows unmapped = **1.61%**, all of
them either `"FA"` (59 rows, mostly 2025) or null team (41 rows).

Franchise-level gaps are worse than the row rate suggests and are listed
explicitly:

- The null-team rows concentrate on **Buffalo Bills players 2010-2015**
  (C.J. Spiller ×3 years, Steve Johnson 2011, Fred Jackson 2012 — all
  verifiably Bills stars with top-60 ADPs). Consequence: **BUF is entirely
  absent from the team-level aggregate for 2010, 2011, 2013 and 2014** (both
  formats), despite having elite-ADP players those years.
- 2012 ppr additionally lacks LAR, NYJ and SF aggregates (29 franchises).
- These gaps are NOT fixable from the API as-is; they would need a hand-coded
  patch list (a handful of known players) if BUF's early-decade baseline ever
  matters to a screen. Recorded as a limitation, not patched silently.

## Weeks 1-4 covariate design sketch (predeclared, screen-ready, NOT run)

Mechanism framing — divergence-from-market: ADP is a crowd expectation of each
roster's fantasy talent formed at a dated pre-Week-1 moment. The project's
Weeks-1-4 cells are where betting lines are historically stalest (small sample,
regression-to-the-mean errors). Candidate covariate family, all derivable from
`team_top8_feasibility.parquet` alone or joined to existing schedules:

1. **Roster-quality level**: franchise `mean_adp_top8` (lower = more fantasy
   talent) as a preseason strength prior for early-season games where in-season
   form data does not yet exist.
2. **Divergence feature**: for each Weeks-1-4 game, the gap between the two
   rosters' `min_adp_top8` (star-power asymmetry) and/or `mean_adp_top8`
   (depth asymmetry), expressed as a z-score against that year's league
   distribution — i.e. how far the crowd's fantasy expectation diverges from
   what the market line implies.
3. **Format robustness check**: compute the same quantity under `standard`
   scoring; a signal that survives both formats is less likely to be a
   positional-scrub artifact.

Predeclared evaluation shape (to be registered before running, per repo rules):
chronological split, comparison against the same market and simple-model
baselines, prediction-level output, uncertainty and season stability reported;
an interval crossing zero is NOT grounds for rejection (AGENTS.md binding rule)
— outcomes route through the category taxonomy and `nfl-ats weak-signals
record`. No such screen has been run for this source yet.

## Limitations

- **Windows are August-September only.** FFC stamps one aggregate per season
  per format (the accumulated mock pool at fetch time), not weekly snapshots.
  There is no way to recover a mid-season (e.g. Week 6) crowd state from this
  API; the covariate family above is structurally a pre-Week-1 prior only.
- **12-team archive only** was pulled. FFC hosts other sizes (8/10/14) but the
  scout verified the 2007+ claim for 12-team formats; mixing sizes would change
  the ADP scale.
- Draft counts vary ~10× across years (303 minimum: 2012 ppr); early-year
  estimates carry wider sampling noise. Report year-stratified, never pooled
  naively.
- Team-mapping gaps listed above (BUF 2010/2011/2013/2014 absent; LAR/NYJ/SF
  missing in 2012 ppr; FA-labeled rows esp. 2025).
- Archive immutability over time is unproven (see PIT section); cite snapshot
  hashes.

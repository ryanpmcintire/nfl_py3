# Can SBR's "Open" column be trusted as an opener? A provenance audit

This is an **audit and measurement**, not a promotion. It does not change the
active model, the evaluation window, or any published number. Every claim is
tagged **measured** (run this session, command/path given), **read** (file
opened this session), **reported** (another doc's claim, not reverified
here), or **inferred** (reasoning, not evidence), per `AGENTS.md`'s "label
how you know it" rule.

**No closing-grounds taxonomy applies here.** That taxonomy (an interval
crossing zero is never grounds to close a line of work; only a resolved wrong
sign or a positive-control bound closes anything) governs ATS-edge
experiments with a directional hypothesis and a `probability_positive`. This
document has neither -- it characterizes data quality (does source A
reproduce source B's number?), not an effect on accuracy. Nothing here is
recorded via `nfl-ats weak-signals record`; that command is for edge signals,
and this finding is neither positive nor negative in that sense.

## The question

The project's headline opener-graded evaluations run on 1,537 paired
2020-2025 games (**reported**, `AGENTS.md`'s own MOD-07 promotion note; not
independently recounted this session) from a purchased point-in-time
snapshot archive (The Odds API), because that archive is the only source
with a genuine capture timestamp proving "this line was on the board at time
T." SBR's `open_home_spread` (**measured**,
`data/processed/sbr_odds.parquet`: 4,025 games total, 15 seasons 2007-2021;
`data/processed/sbr_odds.manifest.json`'s own `matched_to_game_features_rows`
field puts 3,491 of those in the 2009-2021 span this project's own
`game_features.parquet` also covers) could meaningfully expand the
opener-graded population for 2009-2021 -- over double the 1,537-game
headline count, though the two are not a clean apples-to-apples ratio, since
the headline figure also reflects the production model's own walk-forward
warm-up-floor eligibility, not just spread-quote availability -- **if** its
"Open" column can be trusted as an honest opener. This document measures how
much of that trust can be earned.

---

## 1. What SBR's columns actually contain

**Read** this session, `data/raw/sbr_odds/20260819T192226Z/` (raw HTML) and
`scripts/ingest_sbr_odds.py` (parser/ingest code, also read in full).

- **Raw source columns** (one HTML `<table>` per season, header row
  read verbatim off the live page): `Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th,
  Final, Open, Close, ML, 2H`. That is the entire schema SBR publishes for a
  game: **no timestamp column, no book-attribution column, and no revision
  history of any kind** -- measured directly from the raw HTML header row,
  not inferred from documentation (SBR publishes none). `Open` and `Close`
  are each a single number per team-row; there is nothing else to query.
- **This project's processed columns** (`data/processed/sbr_odds.manifest.json`,
  read this session): `season, game_date, sbr_date_raw, neutral_site,
  away_team_raw, home_team_raw, away_team, home_team, away_rot, home_rot,
  away_score, home_score, open_home_spread, close_home_spread, open_total,
  close_total, open_ambiguous, close_ambiguous, away_moneyline,
  home_moneyline, sbr_season_slug, game_id, week,
  week_match_date_diff_days` -- 24 columns total. `open_ambiguous` /
  `close_ambiguous` and `week_match_date_diff_days` are all **derived by this
  project's own ingest code**, not sourced from SBR; they flag internal
  parsing uncertainty (a spread/total split heuristic, and a schedule-join
  date mismatch), not anything about SBR's own capture process. None of the
  24 columns carries a timestamp or book identity either -- confirmed by
  reading the manifest's own column list, not assumed from the raw-HTML
  finding above.
- **What "Open" means**: **inferred**, not established. The module docstring
  in `scripts/ingest_sbr_odds.py` (read this session) states the spread/total
  disambiguation convention is "the classic, widely-documented SBR parsing
  convention -- inferred, not independently re-derived from a site FAQ this
  session." No SBR page (index, season page, or the site's about/FAQ content,
  none of which was scraped further this session) states whose line "Open"
  is, when it was captured, or whether it is a single book, a consensus, or
  a first-print snapshot that a human editor later transcribed. This is the
  central gap the rest of this document works around empirically rather than
  resolving directly: **SBR's own site gives no mechanism to verify "this
  was on the board at time T."**
- The ingest pipeline itself is solid on the parts it CAN verify directly
  from the HTML structure: team-to-favorite attribution comes from the raw
  table's own row structure (`VH` literal V/H tags plus each team's own
  printed number), a **deterministic, non-statistical** read, not an
  inference -- unlike the spread-vs-total magnitude split, which the
  `open_ambiguous`/`close_ambiguous` flags exist to police. In the matched
  population used in Section 3 below, `open_ambiguous` is **measured** False
  for all 824 rows, so that particular parsing uncertainty is not a
  contributor to any of the disagreement reported there.

**Bottom line for item 1**: SBR's "Open" is a bare, un-timestamped,
un-attributed retrospective number. Nothing in the source or this project's
own ingest code can establish provenance directly. Sections 3-4 test it
empirically instead.

---

## 2. Independent, timestamped second sources on the same games

### 2a. The purchased archive (2020-2025) -- already the project's baseline

`data/market/raw`, `capture_kind="historical_backfill"` (**read**,
`src/nfl_ats/odds_backfill.py`): a `tue_open` decision-label snapshot exists
per game, captured via a documented request protocol
(`DECISION_TIMES`, Tuesday 09:00 America/New_York, -5 days from the week's
anchor Sunday) against The Odds API's historical endpoint. This already has
genuine point-in-time provenance -- it is the archive this project's headline
opener metric already trusts. Overlap with SBR (whose archive tops out at
season 2021-22) is **seasons 2020-2021 only**.

### 2b. VegasInsider via the Wayback Machine (2005-2016) -- genuinely new for this question

**Read** this session, `docs/vegasinsider_backfill.md` and
`docs/vi_dispersion_screen.md` (both already in the repo from an earlier,
unrelated lead-generation session; **not** built for this task, but directly
reusable). `scripts/backfill_vegasinsider.py` CDX-enumerates
web.archive.org captures of vegasinsider.com's NFL Las Vegas odds board and
fetches each distinct-content snapshot, producing
`artifacts/vegasinsider_backfill/20260822T033952Z/season_<year>.parquet` for
12 seasons, 2005-2016. The 2005-2008 portion of that predates this project's
own local schedule data (which starts at season 2009) and so cannot be
matched/scored against anything here, even though the raw board data exists.
Columns: `capture_ts, game_date, away, home, kickoff_time, book,
spread_line, total_line` -- one row per (capture, game, named book). This
**is** a genuinely timestamped source: `capture_ts` is the Wayback Machine's
own crawl timestamp (verifiable independently at web.archive.org), and each
row carries a named book (Caesars, Hilton, Mirage, Stations, Wynn, and
others depending on era -- **read**, `docs/vegasinsider_backfill.md` section
"Books detected per season").

**Two measured limitations, both already established by the prior session
(read, not re-derived here) and both carried into Section 3 below:**

- **No home/away orientation.** The VI board stores the displayed
  favorite-side spread with no team attribution beyond which GAME it belongs
  to -- **measured** (prior session): 97.17% of 11,170 non-null spread values
  are negative regardless of which team actually was favored (verified
  against realized home margins). A magnitude-only comparison is the most
  this source can support; it cannot confirm which team SBR says is favored.
- **Coverage is uneven and thin in several seasons.** Tuesday/Wednesday
  capture coverage of the REG schedule ranges from 8.6% (2013) to 64.8%
  (2014) -- **read**, `docs/vegasinsider_backfill.md`'s own coverage table.

### 2c. Sagarin ratings -- checked, not relevant to this question

**Read** this session, `docs/sagarin_backfill.md`: `data/raw/sagarin/` holds
Jeff Sagarin's team POWER RATINGS via Wayback captures, not betting lines.
It is a candidate for a Sagarin-implied-spread-vs-market divergence signal
(a different research question, predeclared but not run in that document),
not a second source for SBR's own opening LINE. Not used further here.

---

## 3. The decisive empirical test: does SBR's Open reproduce a timestamped opener?

Both arms below were run this session via a new script,
`scripts/sbr_opener_provenance_check.py`, and written to
`artifacts/sbr_opener_provenance/<run-id>/results.json` through
`nfl_ats.provenance.write_experiment_artifact` (not printed numbers copied by
hand). No model was scored and no registry was written.

### Arm 1 -- SBR Open vs the purchased archive's `tue_open` (2020-2021), signed

**Measured this session**, re-running `scripts/ingest_sbr_odds.py`'s existing
`opener_check()` -- reused, with one small additive change: the function
already computed per-season median/exact-share but never pooled them across
the two overlap seasons, so this session added `overall_median_abs_diff` /
`overall_share_exact` fields (new keys only; every existing key and value is
untouched, verified by diff). This reproduces
`docs/sbr_odds_archive.md`'s already-published 2026-08-19 per-season numbers
exactly, confirming they are stable and not a one-off fluke of that run.
**Correction to that prior doc**: its published table's "Overall" row
carried a `18.0%` exact-share figure this session cannot reproduce from the
pooled join (measured here: **17.7%**, `(87 of 491)`) -- close to, but not
exactly, that number; the prior doc did not show its arithmetic for that
cell and this session's is now the traceable one (`overall_share_exact` in
`opener_check()`'s return, stamped in this run's artifact JSON).

| Season | SBR games | tue_open available | Matched | Mean \|diff\| | Median \|diff\| | Share ≤0.5pt | Share exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 269 | 239 | 239 | 1.403 | 1.0 | 43.5% | 17.6% |
| 2021 | 285 | 252 | 252 | 1.313 | 1.0 | 46.0% | 17.9% |
| **Overall** | | | **491** | **1.357** | **1.0** | **44.8%** | **17.7%** |

This is a **signed** comparison (`open_home_spread - tue_open_home_spread`,
so it tests both magnitude AND side simultaneously) against the exact
Tuesday-morning quote the pool's own Tuesday lock is meant to represent.
**Reported, not reverified this session**: the prior doc also computed a
Pearson correlation of 0.949 between the two series (signed values) on this
same population -- this session's script does not recompute that number, but
Arm 2's independently-run Pearson-CI code below is the same estimator
applied to the new 2009-2016 arm.

### Arm 2 -- SBR Open vs a Wayback-timestamped VI board (2009-2016), magnitude only

**New this session.** Per-game feature construction (Tuesday/Wednesday
pre-kickoff captures only, earliest qualifying capture per matched game,
cross-book median spread) is reused **by import, unmodified**, from
`scripts/vi_dispersion_screen.py` (`load_board_instances` /
`join_schedule` / `game_level_features`) -- the same "Wayback-derived
opener" construction that document already uses for its own scoring, not a
fresh definition invented for this audit.

**A data-quality defect was found and excluded, not glossed over.** One
Wayback capture, `20091216095259` (VI's Week 15 2009 board), has a
spread/total token-parsing bug: its named books agree tightly with each
other (cross-book range <1pt, so it passed `vi_dispersion_screen.py`'s own
`spread_range>10` disagreement cap) but on numbers in the 37-54pt range --
that is a TOTAL misread as a spread, not a real NFL spread. That prior
document's own predeclaration had already flagged "2 instances" from this
exact capture via the disagreement cap; among games that also matched to
SBR, **7** from that same capture share the defect by implausible magnitude
(37-54pt "spreads"), invisible to a disagreement-based filter because the
mislabeling is consistent across books. The whole capture -- **15 games**
across the full VI-matched population, not just the ones that happen to
overlap SBR -- was dropped by `capture_ts`, a targeted exclusion -- **not**
a blanket magnitude cap. A magnitude cap would have been
the wrong fix: a genuinely large, legitimate spread exists in this same
window (`2013_06_JAX_DEN`, VI median 27.5 vs SBR open 24.0 / close 26.5 --
close three-way agreement, the real 2013 Jaguars, not an artifact) that a
naive "no spread over 26" rule would have wrongly discarded alongside the
defective capture.

**Measured**, after that exclusion:

| Season | SBR games | Matched | Mean \|diff\| | Median \|diff\| | Share ≤0.5pt | Share ≤1.0pt | Share exact | Pearson r |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2009 | 267 | 13 | 0.865 | 0.5 | 53.8% | 84.6% | 15.4% | 0.955 |
| 2010 | 267 | 52 | 0.644 | 0.5 | 69.2% | 78.8% | 36.5% | 0.951 |
| 2011 | 267 | 192 | 0.621 | 0.5 | 73.4% | 84.9% | 32.8% | 0.970 |
| 2012 | 267 | 43 | 0.727 | 0.5 | 62.8% | 86.0% | 20.9% | 0.942 |
| 2013 | 267 | 27 | 1.463 | 1.5 | 33.3% | 44.4% | 7.4% | 0.956 |
| 2014 | 267 | 198 | 0.577 | 0.5 | 68.7% | 86.4% | 34.3% | 0.962 |
| 2015 | 267 | 189 | 0.753 | 0.5 | 65.1% | 78.8% | 27.0% | 0.928 |
| 2016 | 267 | 102 | 0.627 | 0.5 | 74.5% | 86.3% | 34.3% | 0.926 |
| **Overall** | **2,136** | **824 (816 w/ both values)** | **0.680** | **0.5** | **68.0%** | **82.4%** | **30.5%** | **0.952 [0.940, 0.961]** |

(Pearson CI: paired-index bootstrap, 20,000 samples, seed 20260826,
`nfl_ats.provenance`-stamped run. Robustness restricted to matched games with
≥3 named books feeding the VI median: n=752, mean \|diff\| 0.640, Pearson
0.959 -- tighter, not looser, than the unrestricted set.) Match rate against
the full SBR population is uneven by season (from 4.9% in 2009 to 74.2% in
2014), tracking `docs/vegasinsider_backfill.md`'s own Tuesday/Wednesday
coverage table directly -- thin years are thin because the archive itself is
thin there, not because of any new join failure.

**What this does and does not show.** This magnitude-only comparison is
**tighter** than Arm 1's fully-signed 2020-2021 comparison against the
purchased archive (mean \|diff\| 0.68 vs 1.36; Pearson 0.95 vs a reported
0.949 on the signed series) on eight seasons the purchased archive cannot
reach at all. It is real evidence that SBR's Open is not fabricated or
random noise for 2009-2016: an independently-captured, Wayback-timestamped
board agrees with SBR's number to within a point on 82.4% of matched games.
**It does not verify side** -- VI's instrument
structurally cannot support that check (Section 2b) -- so this result speaks
to magnitude fidelity only. SBR's own team-to-favorite attribution (Section
1) comes from a separate, non-statistical mechanism (the raw table's row
structure), which is why this gap is judged less concerning than it would be
if SBR's whole Open value were a black box.

---

## 4. What a defensible expansion would require

**This is a measurement, not a recommendation to change the evaluation
window; that decision belongs to the project owner, per this task's
constraints, and any expansion would need its own predeclaration before
scoring, per this project's standing rules on predeclared evaluations.**

What the numbers say, stated plainly:

- **SBR's Open is not a black box.** It correlates strongly with two
  independently-timestamped sources it was never derived from: r ranges
  0.926-0.970 across **every one** of the eight matched 2009-2016 seasons
  (including the thinnest-coverage ones, 2009 and 2013 -- thin match COUNT
  does not mean weak agreement), plus a reported 0.949 signed correlation for
  2020-2021. That correlation is evidence of a real, non-fabricated market
  reading, not proof of point-identity.
- **It is not point-identical to a genuine Tuesday-morning snapshot either.**
  Even in the best-measured arm (2009-2016 vs VI, magnitude only), only 30.5%
  of matched games agree exactly and 17.6% fall outside a full point. The
  2020-2021 signed comparison is looser still (**measured** this session,
  `overall_share_within_1.0pt` added to `opener_check()`: 17.7% exact, 60.9%
  within a full point -- so **39.1% outside a full point**). **Any expanded
  evaluation that treats SBR's Open as interchangeable with a true
  timestamped Tuesday opener, rather than a correlated proxy with a stated
  error band, would be overstating its precision.**
- **The honest characterization is a discount, not a ban.** SBR's Open
  behaves like a real early-week market reading with roughly **0.6-1.5
  points of typical noise** relative to a genuinely timestamped snapshot,
  by season mean \|diff\| in the tables above: tightest in 2010/2011/2014/2016
  (0.62-0.64pts), middling in 2012/2015 (0.73-0.75pts), loosest in 2009 and
  2013 within the VI arm (0.87 and 1.46pts) and in both years of the
  `tue_open` arm, 2020-2021 (1.31-1.40pts). 2021 is also the one
  measured-outlier season for SBR's *Close* against this project's own
  closing line (mean \|diff\| 0.595 vs 0.19-0.38 every other season 2009-2020
  -- **reported**, `docs/sbr_odds_archive.md` section 3a, read this session,
  not independently re-run here). An expansion built on SBR's Open should
  carry that noise forward explicitly (e.g., as a season- or era-varying
  error band on the settlement line, or by restricting to seasons/games
  where a cross-validated proxy exists), not silently substitute it for the
  purchased archive's point-in-time truth.
- **Residual risk that is NOT resolved here**: what SBR's Open actually
  represents (a single book's opener, a multi-book consensus, a delayed
  transcription of an early-week number) remains unknown -- this document
  bounds the disagreement empirically without ever answering that question
  directly, because SBR itself publishes no mechanism to answer it (Section
  1). Side agreement for the 2009-2016 arm specifically is **not verified at
  all** (Section 2b/3) -- only magnitude. A future predeclared evaluation
  that leans on this range should disclose that gap rather than treat the
  magnitude-only correlation as a full validation.
- **A related, already-established fact, reconfirmed rather than re-derived
  this session**: `docs/sbr_odds_archive.md` already showed SBR's *Close*
  column agrees tightly with this project's own trusted close
  (`game_features.parquet`'s `spread_line`); **measured this session**
  (`.\.tools\uv.exe run --no-sync python scripts/ingest_sbr_odds.py
  --skip-fetch --validate`, CLOSE check) reproduces that table exactly:
  mean \|diff\| 0.19-0.38pts and share-within-half-a-point ≥88.4% every
  season 2009-2020 except 2013 (83.1%), with 2021 the one clear outlier
  (0.595pts mean, 67.0% within half a point). That is a separate,
  already-decided question (this project's close-graded work already draws
  on a range this wide); this document adds the analogous OPEN-side answer
  for the first time.

**If the owner chooses to expand the opener-graded window using SBR's Open**,
the defensible form based on what is measured here would predeclare, before
any accuracy number is seen: (a) which seasons are in-scope (2009-2016 has
the strongest cross-validation; 2007-2008 has none at all, since no
independent timestamped source overlaps them, and 2017-2019 similarly has no
independent timestamped cross-check performed in this document); (b) whether
the settlement line carries a stated per-era error band rather than being
treated as exact; and (c) that any resulting accuracy read is reported
alongside, not as a replacement for, the existing 1,537-game purchased-
archive headline. **If the owner instead judges the residual side-agreement
gap and the un-established 2007-2008/2017-2019 stretches too costly, that is
also a fully defensible reading of the same numbers** -- this document closes
nothing about the underlying question either way.

---

## Files

- `scripts/sbr_opener_provenance_check.py` -- this session's new
  measurement (Arm 1 rerun + Arm 2 new), writes
  `artifacts/sbr_opener_provenance/<run-id>/results.json` via
  `write_experiment_artifact`.
- `scripts/ingest_sbr_odds.py` -- SBR fetch/parse/validate (pre-existing;
  its `opener_check()` is imported by Arm 1 above, with three new pooled-
  overall keys added this session -- `overall_median_abs_diff`,
  `overall_share_within_1.0pt`, `overall_share_exact` -- every pre-existing
  key/value verified unchanged).
- `scripts/vi_dispersion_screen.py` -- VI board feature construction
  (pre-existing; `load_board_instances`/`join_schedule`/`game_level_features`
  are imported unmodified by Arm 2 above).
- `docs/sbr_odds_archive.md`, `docs/sbr_opener_evaluation.md` -- prior
  sessions' SBR ingest validation and era-stratified grading (read, not
  modified, this session).
- `docs/vegasinsider_backfill.md`, `docs/vi_dispersion_screen.md` -- prior
  session's VI Wayback backfill and dispersion screen (read, not modified,
  this session; both predate and are independent of this task).
- `data/processed/sbr_odds.parquet` / `.manifest.json` -- gitignored,
  regenerated this session from the existing raw snapshot (no re-fetch;
  `--skip-fetch`); the manifest's `built_at_utc` naturally differs per run,
  but the parsed row counts, seasons, and columns reproduce
  `docs/sbr_odds_archive.md`'s original ingest exactly (**measured**, this
  session's CLOSE/OPENER/COVERAGE check output above).

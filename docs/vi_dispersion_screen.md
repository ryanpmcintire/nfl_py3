# VegasInsider multi-book dispersion screen — predeclaration

Written **before** `scripts/vi_dispersion_screen.py` computed any cover rate,
gap, interval, or probability_positive. Only FEATURE distributions (SD
histograms, tercile cut points, book counts, parse-artifact scans, match
coverage) were examined beforehand — never an ATS outcome. Source boards:
`artifacts/vegasinsider_backfill/20260822T033952Z/season_*.parquet`
(2005–2016, 7 Las Vegas books, Wayback Machine captures of
vegasinsider.com NFL Las Vegas odds boards).

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every scored cell below is recorded regardless of interval shape.

## Mechanism (stated before freezing directions)

Book-to-book disagreement on a Tuesday/Wednesday board is residual
uncertainty about team strength that the consensus has not yet resolved into
a single quote. Unresolved spots are where quotes go stale: the resolving
side of the late information tends to move lines toward the underdog while
slow books keep the favorite priced short, so the AVERAGE POSTED LINE in a
high-dispersion game systematically overstates the favorite relative to the
eventual consensus. Backing the UNDERDOG side of high-disagreement games
therefore buys a better-than-consensus price.

## Frozen directions (locked before any outcome was seen)

- Cell A (`vi_dispersion_top_tercile_underdog`): top-tercile spread-SD
  games, UNDERDOG side, **+1** — the mechanism above.
- Cell B (`vi_dispersion_bottom_tercile_underdog`): bottom-tercile
  spread-SD games (sharp-consensus control), UNDERDOG side, **+1** — same
  sign as A by symmetry; the mechanism predicts B sits near null relative to
  A, which is what makes it a control.
- Cell C: era split of the strongest cell (A): **2009–2012** vs **2013–2016**,
  same flag and +1 direction, week-blocked.
- Cell D (`vi_dispersion_top_tercile_favorite`): top-tercile spread-SD
  games, FAVORITE side, **−1** — the opposite-side interaction implied by the
  same mechanism (the stale premium is ON the favorite). Sign fixed by
  construction, not chosen after seeing outcomes.

## Feature construction (frozen)

Per board game INSTANCE (capture_ts × game_date × away × home):

1. Keep only captures whose UTC date falls on a Tuesday or Wednesday.
2. Drop season 2006 entirely (reduced-confidence: book-identity fallback
   rate 0.64 > 0.20, reported by the backfill's own status.json).
3. Books = DISTINCT NAMED books with a non-null `spread_line`
   (resp. `total_line`). Require ≥3 books (`MIN_BOOKS = 3`) or the instance
   carries no dispersion feature.
4. Features: `spread_sd`, `total_sd` (sample SD, ddof=1 across books),
   `spread_range`, `total_range` (max−min), `n_books`.
5. Parse-artifact cap (frozen after inspecting feature distributions ONLY):
   drop instances with `spread_range > 10` or `total_range > 10`. Measured
   pre-freeze: exactly 2 instances exceed either cap, both from one capture
   (20091216095259, max posted "spreads" 41.5/43.5 — impossible quotes).
6. Leakage guard (hard assert in code): capture datetime must precede
   kickoff datetime (game_date + board kickoff time, Eastern → UTC).
   Measured pre-freeze: the 18 instances whose capture DATE equals
   game_date+1 are all early-Tuesday-UTC snapshots of Monday-night games
   taken pre-kickoff, so the timestamp comparison (not date equality) is the
   correct guard.
7. Schedule join: map board codes to schedule codes
   (LAR→STL / LAC→SD / LV→OAK, LAR→LA in 2016), match REG schedules on
   (away, home) with gameday within ±1 day; require EXACTLY ONE candidate;
   game-level feature row = EARLIEST qualifying instance per matched game.

## Population (measured pre-freeze)

- Boards span 2005–2016, but the newest local schedules snapshot
  (`data/raw/20260817T235649Z/schedules.parquet`, read this session) starts
  at season 2009 (**measured**: min(season)=2009), so ATS scoring covers
  **REG 2009–2016 only**. The 2005–2008 board seasons build features but
  cannot be scored; this is a coverage boundary, not a window retirement.
- Scoreable population pre-cap/clean: 843 unique matched games with ≥2-book
  spreads; 767 with ≥3 books (the frozen MIN_BOOKS=3 scoring frame),
  median 7 books; n_books distribution: 2:34, 3:42, 4:35, 5:70, 6:122,
  7:498.
- Spread-SD distribution on the ≥3-book frame (measured pre-freeze):
  median 0.267, mean 0.376 (inflated by the 2 artifacts capped in step 5),
  quartiles 0.189 / 0.267 / 0.450; 18.6% of games have IDENTICAL quotes
  across all books (SD = 0). Tercile cuts on the cleaned frame are computed
  inside the script and printed with results.
- Total-SD distribution similar scale (median 0.274, quartiles
  0.204 / 0.450). Totals feed coverage/diagnostics and the reliability
  estimate only — NO total-based cell is declared (multiplicity discipline).

## Method (reused verbatim from `scripts/divisional_rematch_screen.py` convention)

- Population: one row per TEAM-GAME (long table, both sides of every matched
  scored game), pushes/missing spreads dropped via
  `nfl_ats.features.add_ats_outcomes` (`home_cover` notna).
- Side classification uses the PREGAME board itself: home is the favorite in
  a flagged game iff that game's `median_spread` < 0; PK games (median == 0)
  contribute neither a dog nor a favorite flag but stay in complements.
- Value = `team_covered`; subset-vs-complement full-slate-scaled effect:
  `(subset_cover − complement_cover) × 100 × fraction_of_slate`.
- Week-blocked joint bootstrap primary (block = season×100+week);
  season-blocked secondary (block = season); same
  `block_bootstrap_two_group` algorithm.
- **20,000 samples, seed 20260823** (mandated).
- `probability_positive` = fraction of draws favouring the PREDECLARED
  direction (sign applied before the >0 test), reported continuously.
- Split-half reliability of the spread-SD trait: within each qualifying
  game's capture, repeatedly (250 draws, seed 20260823) split the named
  books into two random halves (≥3 each when possible), correlate
  half-SDs across games, Spearman-Brown-correct the mean correlation.
  Reported with the results and carried into the record lines.

## Mined-multiplicity disclosure

All four cells mine ONE feature family (cross-book spread dispersion) from
ONE 8-season scored window with ONE join. The tercile cut was selected after
inspecting feature distributions (never outcomes), which is itself a
researcher degree of freedom. Cells A/B/D partition overlapping populations
and are strongly correlated with each other; **none may be pooled with the
others as independent evidence**, and the family-wide read is the pool of
recorded entries, which already contains correlated decompositions. The
bottom-tercile control (B) and the favorite-side mirror (D) double the
effective number of looks at the same underlying contrast.

## Recording commitment

Every scored cell records via `nfl-ats weak-signals record` as
`unresolved_below_power`, league=nfl, effect-units=accuracy_points,
seasons 2009-2016, unless the whole interval sits below zero against the
frozen direction (wrong_sign_resolved) or the trait's split-half reliability
is zero (no_split_half_reliability). Exact command lines are RETURNED, not
run — this task writes neither registry JSON.

## Results

All numbers below are **measured** this session from
`artifacts/vi_dispersion/20260822T233521Z/results.json` (run log stamped to
`registry/experiments/vi-dispersion-screen/20260822T233521Z.json`), produced by
`scripts/vi_dispersion_screen.py`, 20,000 samples, seed 20260823,
week-blocked primary. Pipeline: 1,456 clean T/W instances (15 dropped by the
kickoff leakage guard, 0 by the range cap once unnamed-book columns are
excluded — the two gross >40-point artifacts pre-freeze both lived in
unnamed fallback columns), 839 unique matched games, **739 scored games**
(REG 2009–2016, ≥3 named books). Tercile cuts on scored `spread_sd`:
0.2041 / 0.3780 points. 58 week blocks.

Coverage table (**measured**):

| season | clean T/W instances | matched games | scored games |
|---|---|---|---|
| 2005 | 72 | 0 | 0 |
| 2006 | 0 (excluded, reduced confidence) | 0 | 0 |
| 2007 | 219 | 0 | 0 |
| 2008 | 183 | 0 | 0 |
| 2009 | 41 | 28 | 24 |
| 2010 | 58 | 58 | 48 |
| 2011 | 238 | 192 | 166 |
| 2012 | 59 | 44 | 36 |
| 2013 | 28 | 28 | 25 |
| 2014 | 215 | 198 | 180 |
| 2015 | 226 | 189 | 164 |
| 2016 | 117 | 102 | 96 |

Split-half reliability of the spread-SD trait (**measured**, 618 ≥6-book
games, 250 random half-splits, seed 20260823): mean split correlation
**−0.0206**, Spearman-Brown **−0.0421**. Robustness re-checks (**measured**
this session): mean Spearman variant +0.0655 (SB ≈ +0.12); total-SD trait
−0.0025. At a ≤7-book instrument the disagreement reading is essentially
noise: which books you sample decides which games look "high dispersion".

| cell | n_flag | full-slate effect | week-blocked 95% | P+ | season-secondary P+ |
|---|---|---|---|---|---|
| A `vi_dispersion_top_tercile_underdog` (+1) | 252 | **−0.6525 pts** | [−1.8744, +0.5158] | 0.1348 | 0.0427 |
| B `vi_dispersion_bottom_tercile_underdog` control (+1) | 259 | **+1.1075 pts** | [−0.0849, +2.3027] | 0.9620 | 0.9346 |
| D `vi_dispersion_top_tercile_favorite` (−1) | 246 | **−0.4870 pts** | [−1.6682, +0.6406] | 0.1960 | 0.0795 |

Cell C era splits of A (**measured**): 2009–2012 n_flag=27, −0.4798 pts,
P+ = 0.1383; 2013–2016 n_flag=225, −0.7801 pts, P+ = 0.2077.

Read-through per the binding taxonomy:

- Cells A and D point AGAINST their frozen directions (P+ 0.13 / 0.20) but
  neither interval sits wholly below zero, so `wrong_sign_resolved` is
  inadmissible and nothing is closed on sign.
- Cell B — the sharp-consensus CONTROL — outperformed the flagged tercile
  descriptively (P+ 0.962 for low-dispersion dogs vs 0.135 for
  high-dispersion dogs), the opposite ordering of the mechanism. This is a
  descriptive contrast between correlated cells, not an adjudication.
- All three cells record as `unresolved_below_power` via returned record
  lines (this task writes neither registry JSON), carrying the measured
  reliability. The near-zero split-half reliability is disclosed inside the
  records as the live closure route (`no_split_half_reliability`) for any
  future predeclared adjudication of this instrument; it was not treated as
  grounds to skip or discard the cells.

Multiplicity disclosure stands as predeclared: one feature family, one
window, three correlated flags — never pool A/B/D as independent evidence.

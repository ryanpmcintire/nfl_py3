# VegasInsider multi-book dispersion screen — predeclaration

Batch 1 (2026-08-22): written **before** `scripts/vi_dispersion_screen.py`
computed any cover rate, gap, interval, or probability_positive. Only FEATURE
distributions (SD histograms, tercile cut points, book counts, parse-artifact
scans, match coverage) were examined beforehand — never an ATS outcome.
Source boards: `artifacts/vegasinsider_backfill/20260822T033952Z/season_*.parquet`
(2005–2016, 7 Las Vegas books, Wayback Machine captures of
vegasinsider.com NFL Las Vegas odds boards).

Batch 2 (2026-08-24) below adds THREE further cells from the SAME feature
family; its predeclaration was written before this session computed any of
those cells' outcomes, but NOT before batch-1 results were seen — that
dependence is disclosed inline and in the multiplicity section.

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

---

# Batch 2 (2026-08-24): orientation-free cells — predeclaration

## Instrument finding discovered this session, before any batch-2 outcome was computed (**measured**)

The VI backfill parquet **cannot encode which SIDE (home/away) a spread line
favors**. Measured this session over all 2009–2016 tidy rows (n=11,170
non-null `spread_line` values): 97.17% negative, 1.39% positive, 1.44%
exactly zero — the parser stores the displayed favorite-side quote verbatim
(`scripts/backfill_vegasinsider.py` `classify_line_tokens` keeps the signed
token as printed; no home/away reorientation). Cross-checked against the
nflverse closing line (`data/raw/20260824T110229Z/schedules.parquet`, whose
own convention is + = home favorite, verified against realized home margins:
mean margin +5.93 when `spread_line>0` vs −4.63 when `<0`) on 1,170 matched
game instances: raw sign agreement 0.3402, sign-flipped agreement 0.6521
(≈ the base rate of home favorites), |corr(VI median, close)| = 0.11. A
variable that is negative ~always cannot carry side information.

**Consequence for batch 1:** its flags were derived from
`median_spread < 0 ⇒ home_is_favorite`, which fires on essentially every
game. Cells A/D therefore actually scored the AWAY-side and HOME-side cover
rates in top-tercile games, not underdog/favorite sides. The batch-1 NUMBERS
stand as side contrasts; their mechanism paragraphs ("stale premium on the
favorite", dog/favorite framing) are INVALIDATED as interpretations. No
reorientation is possible from the stored schema. Batch-2 cells below are
chosen to be orientation-free by construction.

## Frozen cells and directions (written before any batch-2 outcome)

All three cells use WITHIN-SEASON tercile cuts of the dispersion feature
(top vs bottom third of each season's scored games), removing era/board-
generation confounds from flag assignment. Population: REG seasons 2009–2016
(the board archive ends 2016; local schedules start 2009 — measured in batch
1), ≥3 named books per frozen batch-1 construction, earliest pre-kickoff
T/W capture per game, same leakage guard, range cap, and join.

- **Cell P (primary)** `vi_disp_homecover_top_vs_bottom_tercile`: opener
  cross-book spread-SD within-season terciles, TOP vs BOTTOM arm,
  outcome = `home_cover` (`nfl_ats.features.add_ats_outcomes`). Effect =
  (cover_top − cover_bottom) in percentage points × fraction of the scored
  slate in the two arms (full-slate-scaled accuracy points, batch-1
  convention). Week-blocked bootstrap primary, season-blocked secondary.
  Direction **+1** (top tercile home-cover ≥ bottom), PREDECLARED and
  explicitly informed by batch-1's measured home-side contrast (cell D,
  actually the HOME side: P+ 0.196 against its −1 label ⇒ descriptive
  top-home advantage). This is a second look at one family/window — not
  independent evidence.
- **Cell S2** `vi_disp_movement_top_vs_bottom_tercile`: same spread-SD
  terciles; outcome = absolute subsequent movement
  `||close| − |opener||` where close = |nflverse `spread_line`| and opener =
  |VI board median spread| (both are favorite-side magnitudes; orientation
  not needed). Direction **+1**: disagreement measures unresolved
  information, and resolution requires repricing, so high-dispersion openers
  precede larger absolute moves.
- **Cell S3** `vi_totaldisp_overshoot_top_vs_bottom_tercile`: cross-book
  TOTAL-SD within-season terciles (same MIN_BOOKS=3 gate on named total
  quotes); outcome = actual total points − opener mean posted total.
  Direction **+1** (high total-disagreement games overshoot their opener
  number more). Grounding is honestly weak (inferred): the top-minus-bottom
  differencing cancels era-wide scoring drift, but the residual claim that
  disagreement correlates with upward rather than downward revision is not
  independently established. Reported regardless of direction found.

Classification rule, declared up front: every scored cell records via
`nfl-ats weak-signals record` as `unresolved_below_power` unless a terminal
ground from the binding taxonomy applies (`wrong_sign_resolved` requires the
whole interval below zero against the frozen direction;
`no_split_half_reliability` requires a measured reliability consistent with
zero). An interval containing zero is NEVER a ground. Record lines are
RETURNED as text; nothing here writes `registry/weak_signals.json`.

## Method (batch-2 specifics)

- Bootstrap: block resampling of season×100+week blocks (primary) and season
  blocks (secondary), 20,000 draws, seed 20260823 (same mandated seed family
  as batch 1); two-arm gap = mean_top − mean_bottom recomputed inside each
  resample; invalid draws (either arm empty) dropped and counted.
- Per-season stability: per-season arm sizes and raw gaps reported for every
  cell alongside the pooled intervals.
- Split-half reliability recomputed this session with batch-1's estimator
  (250 random book-half splits, ≥6-book games): spread-SD trait feeds P/S2,
  total-SD trait feeds S3.
- Coverage honesty: counts of matched games with <2 named books (dispersion
  undefined), the 2-book band dropped by MIN_BOOKS=3, games lost to missing
  closing lines (S2), and games failing the totals gate (S3) are reported
  with the coverage table.

## Mined-multiplicity disclosure (cumulative)

One feature family (cross-book VI board disagreement), one archive window
(2009–2016 scoreable), one join. Scored looks now total seven: batch-1 A/B/C
(era splits)/D plus batch-2 P/S2/S3. NO multiplicity correction is applied;
every cell speaks category-3 (`unresolved_below_power`) unless terminal
grounds are met; no pair of these cells may be pooled as independent
evidence — they share games, features, and (for P) direction information
from batch-1 outputs.

## Batch-2 results (all **measured** this session)

From `artifacts/vi_dispersion/20260824T112204Z/results.json` (run log stamped
to `registry/experiments/vi-dispersion-screen/20260824T112204Z.json`), 20,000
draws, seed 20260823. Batch-1 cells re-ran deterministically and reproduced
their published numbers exactly (same seed); schedules snapshot is now
`data/raw/20260824T110229Z`. Pipeline unchanged: 1,456 clean T/W instances,
839 matched games, **739 spread-scored games** (REG 2009–2016, ≥3 named
books).

Coverage honesty (**measured**): dispersion undefined (<2 named book spreads)
on 42 of 839 matched games — those drop; the 2-book band drops another 34 to
the MIN_BOOKS=3 gate; 0 further games lost to missing closing lines (S2);
33 spread-scored games fail the ≥3-named-total-books gate for S3 (S3 frame
n=744). Split-half reliability (**measured**, same estimator as batch 1):
spread-SD trait −0.0421 (618 ≥6-book games; Spearman variant ≈ +0.12 per
batch-1 robustness re-check), total-SD trait −0.0110 (686 games) — at a ≤7-
book instrument both disagreement readings are close to noise, disclosed as
the live `no_split_half_reliability` route for any FUTURE predeclared
adjudication; not treated as grounds to skip recording.

| cell | direction | n (top/bottom) | effect | week-blocked 95% | P+ | season-blocked P+ |
|---|---|---|---|---|---|---|
| P `vi_disp_homecover_top_vs_bottom_tercile` | +1 | 254/262 | **+5.3637 acc pts** (full-slate-scaled; raw gap +7.68 pp) | [−0.3686, +11.1612] | **0.9671** | 0.9238 |
| S2 `vi_disp_movement_top_vs_bottom_tercile` | +1 | 254/262 | **−0.2179 pts** mean |close−open\| | [−1.0290, +0.2471] | 0.3592 | 0.3302 |
| S3 `vi_totaldisp_overshoot_top_vs_bottom_tercile` | +1 | 271/279 | **−0.6045 pts** overshoot gap | [−2.9300, +1.6967] | 0.3039 | 0.3069 |

Per-season raw gaps (top − bottom, native outcome units; thin-season noise
expected):

- P (home-cover prob): 2009 +0.403, 2010 +0.038, 2011 +0.238, 2012 −0.231,
  2013 +0.111, 2014 −0.042, 2015 +0.018, 2016 +0.156 — sign-stable positive
  in 6 of 8 seasons, magnitude carried mostly by 2011.
- S2 (points): 2009 −10.198 (single-thin-season outlier, 41 T/W instances),
  2010 +0.174, 2011 +0.013, 2012 −0.077, 2013 −0.722, 2014 +0.283, 2015
  +0.355, 2016 +0.031 — no consistent ordering.
- S3 (points): +3.867, −10.761, −0.707, +4.917, −8.968, +1.738, −1.776,
  +3.086 — unstable, alternating signs.

Read-through per the binding taxonomy:

- Cell P favours its frozen +1 direction (P+ 0.967 week-blocked, 0.924
  season-blocked) but does NOT clear the project's promotion-grade bar, and
  this is a second, direction-informed look at a family whose first look
  pointed the other way on the away side — category 3,
  `unresolved_below_power`, never "confirmed" off one mined family.
- Cells S2 and S3 point against their frozen directions (P+ 0.36 / 0.30):
  high-disagreement openers moved slightly LESS afterwards, and high
  total-disagreement games undershot their opener totals slightly more;
  neither interval sits wholly below zero, so nothing closes on sign.
- All three cells record via returned record lines (this session executes
  none); every factual claim above is measured from the stamped artifact.

Batch-2 multiplicity disclosure stands as written before scoring: seven
correlated looks at one family/window across two batches; no correction; no
pair pooled as independent evidence.

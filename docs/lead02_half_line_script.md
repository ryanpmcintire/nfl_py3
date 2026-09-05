# LEAD-02: first-half/full-game script disagreement — predeclaration + results

Gated on LEAD-60 (`docs/vi_half_lines.md`), now built. This document is the
predeclaration (frozen before any outcome was read — see the ratio
distribution and frozen cut section) followed by the results.
`scripts/lead02_half_line_script_screen.py` implements exactly what is
predeclared here; nothing in the script diverges from this document.

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

## Mechanism and predeclared direction

A first-half (1H) spread that is SHORT relative to the full-game spread
implies the market expects the favorite to pull away in the second half
rather than dominate from the opening whistle — a "slow starter" script. A
team the market doesn't expect to be ahead by much at halftime, relative to
how much it's expected to win by overall, wins more of its margin late, when
backup snaps, clock-killing play-calling, and garbage-time defense make the
final score a worse proxy for game control. **Predeclared direction: FADE
the full-game favorite** (equivalently, back the underdog) when this
disagreement is present.

A cheap 2H sibling cell uses the same mechanism on the second-half leg
instead: a 2H spread that is short relative to the full game implies the
favorite is expected to have built most of its margin already by halftime,
leaving a closer second half — again, **BACK the underdog** (the team whose
2H line is more favorable than the full-game line implies it "should" be).

## Data and population

- Source: `artifacts/vegasinsider_backfill/20260822T033952Z/half_lines_<year>.parquet`
  (LEAD-60, docs/vi_half_lines.md) joined to the SAME run's
  `season_<year>.parquet` on `(capture_ts, game_date, away, home, book)` —
  CLOSE-graded lines, 2005–2016 REG.
- Season 2006 excluded entirely (board book-identity fallback rate 0.643,
  docs/vegasinsider_backfill.md's "Reduced-confidence flag" — both the
  full-game and half legs derive from the same board/movement pages, so this
  exclusion applies to both, consistent with `scripts/vi_dispersion_screen.py`
  precedent).
- Outcomes: the newest local `data/raw/<stamp>/schedules.parquet` snapshot
  (**read** 2026-09-05: `F:\Repos\nfl_py3\data\raw\20260824T115346Z\schedules.parquet`),
  matched on season (the archive's own file-year) + team codes (normalized
  through `scripts/vi_dispersion_screen.py::vi_to_sched`, the LAR/LAC/LV
  relocation-era alias table already established for this archive) + game
  date within ±1 day. **That local schedules snapshot only starts in 2009**
  (docs/vegasinsider_backfill.md: "Seasons 2005-2008 predate the local
  schedules snapshot") — seasons 2005–2008 are therefore market-only in this
  screen and never scored. This is documented here, not discovered as a
  surprise in the results.
- Side attribution (which team is the full-game favorite) always comes from
  the SCHEDULE's own signed `spread_line`, never from the VI archive: the VI
  archive's own `spread_line` is a favorite-side-only quote with no home/away
  orientation (docs/vegasinsider_pilot.md), so it cannot answer "who is
  favored" by itself.
- Population unit is one GAME (not one capture/book row): after joining, the
  favorites-of-3+ subset is deduplicated to one row per game, keeping the
  LATEST capture_ts among usable (capture, book) instances — the best
  available proxy for a close read this archive offers (no true book-level
  closing timestamp is recorded, only Wayback capture time).

### Data-quality guard (measured 2026-09-05, before any scoring)

155 of 14,727 non-null `spread_line` values in the underlying (pre-existing)
`season_<year>.parquet` tidy table are POSITIVE, in the 40–54 range — e.g.
away=LAC home=TEN, capture `20091226095259`, `spread_line=53.5` at
Caesars/Mirage. This is a total (O/U) value misfiled into the spread column
by the archive's existing board parser (not touched by this lane), and it
violates the archive's own documented convention that `spread_line` is a
favorite-side quote and therefore always <= 0 (docs/vegasinsider_pilot.md).
`filter_plausible()` drops any row where either leg is positive or exceeds
30 points in magnitude. Cost: 7 rows (half1) / 5 rows (half2) out of several
hundred joined rows; the largest genuine full-game favorite left in the
archive afterward is -19.0, so the guard costs no real market data.

## Encoding, frozen before any outcome was read

`disagreement = half_spread / full_spread` (both stored in the archive's
favorite-side convention, both <= 0 in the plausible range, so the ratio is
normally positive — a shorter half line relative to the full-game line gives
a SMALLER ratio). Restricted to the eligible population: full-game favorites
of 3+ points (`full_spread <= -3.0`), one row per game (see above).

The cut is **measured** from this ratio's own empirical distribution and
frozen at its 20th percentile — market data only; `freeze_cut()` never reads
an outcome column, and the deduplication/eligibility/ratio/cut sequence in
`run_half_cell()` runs strictly before the schedule join that first
introduces `home_cover`/`result`.

**Measured ratio distributions (market-only, before any outcome join):**

| leg | n (games) | mean | median | min | max | n sign-flip (ratio<0) | **frozen 20th-pct cut** |
|---|---|---|---|---|---|---|---|
| 1H (LEAD-02) | 87 | 0.6144 | 0.6154 | -0.0 | 1.0 | 0 | **0.5000** |
| 2H (sibling) | 43 | 0.5213 | 0.6000 | -0.0 | 1.0 | 0 | **0.1524** |

The median (~0.60-0.62) matches the mechanism note's expectation that "a
normal 1H line is ~50-55% of the full line" reasonably closely; the frozen
1H cut of 0.50 is close to the illustrative example the gating brief used
(0.40) but is the ACTUAL measured 20th percentile, not the illustrative
number. Zero sign-flip cases (a half spread on the opposite side from the
full-game favorite) occurred in either leg's eligible population.

`flag = ratio < frozen_cut`, computed once and never recomputed after the
schedule/outcome join — `tests/test_lead02_half_line_script.py::
test_flag_is_invariant_to_the_games_outcome` checks this directly (two rows
identical on market columns but assigned opposite outcomes must retain an
identical flag).

## Method (mirrors `scripts/nfl_bias_battery_screen.py`)

Subset-vs-complement week-blocked bootstrap (20,000 draws, seed 20260905),
raw gap scaled by the flagged subset's fraction of the scored slate
(`full_slate_effect_pts`, in accuracy points), plus:

- **Era split** 2005-2010 / 2011-2016 (the split requested for this lead;
  because the local schedule snapshot starts in 2009, the "2005-2010" bucket
  is, in practice, only 2009-2010 — reported explicitly per cell below, not
  silently substituted).
- **Positive control**: a flag planted directly from each game's own
  REALIZED ats margin (the `n_flag` games — matched to the real cell's own
  flagged count — where the underdog beat the closing spread by the MOST),
  scored with the identical bootstrap. This is an oracle by construction (it
  reads the outcome to build the flag); its purpose is to measure what THIS
  population size and week-block structure CAN show for a real effect at
  least this large — a power check, not a claim about the market-only flag.
- **200-draw within-week permutation null** (seed 20260905): permute which
  rows carry the flag WITHIN each week block, recompute the raw gap each
  draw. **Measured, not a bug:** this population's week blocks are almost
  all size 1 (a game roughly every 1.1-1.9 weeks per flagged/unflagged
  pairing — see `n_blocks_multi_row` below), so a within-WEEK permutation is
  near-degenerate: a block of exactly one row cannot be permuted at all, so
  most/every draw reproduces the observed gap exactly. This is reported
  as-is (the task specified a within-week null), plus a labeled
  **within-season permutation null, supplementary** (bigger blocks, real
  swap room) alongside it for an informative read — never in place of the
  primary one.

## Results (measured 2026-09-05)

Command: `.\.tools\uv.exe run --no-sync python scripts\lead02_half_line_script_screen.py`
Artifact: `artifacts/lead02_half_line_script/20260905T170817Z/results.json`
(plus `scored_half1.parquet`/`scored_half2.parquet` in the same directory).

### Population funnel

| leg | raw joined rows (both spreads) | implausible dropped | favorites 3+ | deduped games (market-only) | matched to schedule | schedule match rate | pushes dropped | **scored games** | **flagged** |
|---|---|---|---|---|---|---|---|---|---|
| 1H (LEAD-02) | 593 | 7 | 465 | 87 | 70 | 80.46% | 2 | **68** | **10** (14.7%) |
| 2H (sibling) | 332 | 5 | 264 | 43 | 38 | 88.37% | 1 | **37** | **7** (18.9%) |

The archive's own documented half-vs-full join rate (43.6%, 958/2,195,
docs/vi_half_lines.md) is a **reported**, whole-archive figure; this
screen's own schedule-match rate (80.46% / 88.37% above) is a narrower,
**measured** figure specific to the deduped favorites-3+ population.

### 1H (LEAD-02 primary): FADE the full-game favorite

Effect below is `full_slate_effect_pts` — the flagged-subset-minus-
complement dog-cover-rate gap, scaled by the flagged fraction of the scored
slate, in accuracy points (percentage points). Positive favors the
predeclared direction.

| period | n (scored) | n flagged | effect (pts) | week-blocked 95% CI (pts) | P+ |
|---|---|---|---|---|---|
| full period (2009–2016 actual data) | 68 | 10 | -0.71 | [-5.59, +4.22] | 0.376 |
| era 2005-2010 (actual: 2009-2010 only) | 14 | **1** | -2.75 | [-4.55, -1.10] | 0.000 |
| era 2011-2016 | 54 | 9 | -0.37 | [-6.32, +5.56] | 0.441 |
| **positive control** (oracle, realized-margin-planted, n_flag matched) | 68 | 10 | +9.63 | [+7.88, +11.33] | 1.000 |

**A flat, near-null read.** The full-period point estimate is small and
negative (opposite the predeclared direction), the interval straddles zero,
and P+ is below 0.5 — this is a genuine, close-to-coin-flip measurement, not
a "wrong sign resolved" finding: the whole-interval-negative era-2005-2010
row above is **NOT interpretable as a resolved sign** — it rests on
`n_flag=1` (one single flagged game), so the week-blocked bootstrap has
mechanically only one possible flagged draw value to resample from
(zero-variance CI by construction, not evidence of a real, stable negative
effect at that era). This is flagged explicitly rather than quoted as a
finding, and is NOT used as a `wrong_sign_resolved` closing ground anywhere
in this write-up or the registry entries below.

The positive control (oracle: the 10 games with the largest realized dog
margin, matched exactly to the real cell's flagged count) shows a CI cleanly
excluding zero (+7.88 to +11.33 pts, P+ 1.0) — proving the bootstrap/
population COULD show a real effect of that size if one were present. The
real cell's flat reading is therefore a genuine null-ish measurement at this
power, not an artifact of an underpowered or broken instrument.

Permutation nulls (200 draws, full period): the required within-week null is
near-degenerate (9 of 59 week blocks hold 2+ rows; 18 of 68 games sit in a
multi-row block; observed -4.83pts sits inside a null with essentially zero
spread by construction). The supplementary within-season null (8 blocks, all
multi-row, all 68 games swappable) gives a null mean +6.90pts, sd 14.73pts,
95% [-16.84, +30.34]; the observed -4.83pts sits well inside that band —
unremarkable relative to season-shuffle noise.

### 2H sibling: BACK the underdog on a favorable 2H line

| period | n (scored) | n flagged | effect (pts) | week-blocked 95% CI (pts) | P+ |
|---|---|---|---|---|---|
| full period (2009–2016 actual data) | 37 | 7 | **+7.84** | **[+0.17, +14.47]** | **0.976** |
| era 2005-2010 (actual: 2009-2010 only) | 12 | 2 | +3.33 | [-7.69, +14.81] | 0.647 |
| era 2011-2016 | 25 | 5 | **+10.00** | **[+0.95, +16.47]** | **0.981** |
| **positive control** (oracle, realized-margin-planted, n_flag matched) | 37 | 7 | +14.50 | [+11.60, +17.09] | 1.000 |

**The more promising of the two cells, still `unresolved_below_power`.** The
full-period and 2011-2016-era intervals both exclude zero on the predeclared
(favorable) side, with P+ above 0.97 in both. Per the binding taxonomy, an
interval EXCLUDING zero on a sample this small (n=37, 7 flagged games) is
**not itself a closing ground either** — the taxonomy's only two closing
grounds are a refuted mechanism (wrong sign) or a positive-control bound,
and neither applies here (the sign is right, and the point estimate is well
above what a null/noise process alone would explain, not bounded away from
a real effect by the control). This stays `unresolved_below_power` and is
recorded as such, flagged as the more promising lead of the two for a future
opener-graded confirmation pass — no play/publish decision follows from a
CLOSE-graded lead-generation screen at n=37.

Permutation nulls (200 draws, full period): the required within-week null is
again near-degenerate (4 of 33 week blocks hold 2+ rows). The supplementary
within-season null (7 blocks, all multi-row, all 37 games swappable) gives a
null mean +6.37pts, sd 20.75pts, 95% [-29.05, +41.43]; the observed
+41.43pts (raw, pre-slate-scaling gap) sits at the null distribution's own
maximum — only 12% of season-shuffle draws are at least as extreme in
absolute value as what was observed. This is suggestive, not a closing
ground (the taxonomy has none for "small p-value"), and is reported as a
number, not a verdict.

## Registry

Every scoreable cell above is recorded via `nfl-ats weak-signals record` as
`unresolved_below_power`, `--effect-units accuracy_points`, `--category
market`, `--league nfl`. Families: `half_line_script` (1H, LEAD-02) and
`half_line_script_2h` (2H sibling), each with a full-period entry plus one
entry per era (6 entries total). `--season-start`/`--season-end` use the
requested archive-scope boundaries (2005/2016 full period, 2005/2010 and
2011/2016 per era); the actual scored season range (2009-2016, and 2009-2010
within the first era bucket) is stated in each entry's `--notes` field per
the population-funnel and era tables above — never silently substituted for
the requested boundary labels.

## What this does NOT establish

- No play, publish, or promotion decision — this is a CLOSE-graded
  lead-generation screen (docs/rotation_registry.md rule 8), not an
  opener-graded rotation window.
- The 2H sibling's zero-excluding intervals are a lead worth a future
  opener-graded confirmation pass, not a finding — n=37 with 7 flagged games
  is far below the sample size this project treats as informative on its
  own, and the taxonomy above explicitly does not treat "excludes zero" as
  a closing ground in either direction.
- The 1H (LEAD-02) primary cell is a clean, flat null-ish read at this
  power, validated by a positive control that proves the instrument is not
  simply blind — also not closed (per the taxonomy, "contains zero" is
  never grounds for closure either).

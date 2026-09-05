# LEAD-02: first-half/full-game script disagreement — predeclaration + results

Gated on LEAD-60 (`docs/vi_half_lines.md`), now built. This document is the
predeclaration (frozen before any outcome was read — see the ratio
distribution and frozen cut section) followed by the results.
`scripts/lead02_half_line_script_screen.py` implements exactly what is
predeclared here; nothing in the script diverges from this document.

**ENG-40 update (2026-09-05):** the results below were rescreened after
`scripts/backfill_vegasinsider.py`'s spread/total-misfiling parser bug
(ENG-40) was fixed at the source and all 12 seasons of
`season_<year>.parquet` were rebuilt from the same cached HTML — see
`docs/vi_half_lines.md`'s ENG-40 note and this document's "Data-quality
guard" section below. The numbers on this page are the POST-fix numbers,
measured 2026-09-05 from `artifacts/lead02_half_line_script/20260905T174147Z/results.json`,
and supersede the original `20260905T170817Z` run (registry entries were
re-recorded with `nfl-ats weak-signals record --replace`, same six names).

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

### Data-quality guard (ENG-40, FIXED 2026-09-05 at the source)

155 of 14,727 non-null `spread_line` values in the underlying
`season_<year>.parquet` tidy table were POSITIVE, in the 40–54 range — e.g.
away=LAC home=TEN, capture `20091226095259`, `spread_line=53.5` at
Caesars/Mirage. This was a total (O/U) value misfiled into the spread column
by a board-layout ambiguity in `classify_line_tokens`: VegasInsider renders
a cell's spread and total lines in either vertical order (the away team's
row shows its own line only when it is the favorite; the underdog's row
shows the total instead), and when a total carried no o/u price marker it
was rendered as a bare "+"-signed number (e.g. `"+54"`) that satisfied the
spread-token regex before the real (later, negative) spread token was ever
read. **This is now fixed at the source** (`scripts/backfill_vegasinsider.py::
classify_line_tokens`, 2026-09-05): a `"+"`-prefixed token can never be this
archive's favorite-only spread, so it is routed to total classification
instead — a sign-convention rule read from the token text, not a
magnitude/range filter on the resulting value. All 12 seasons were rebuilt
from the same cached HTML with the fixed parser (network hard-blocked
throughout, cache-only): **measured 2026-09-05, zero rows across the full
2005-2016 archive now have a positive or >30-magnitude `spread_line`.** The
rebuild also recovered 54 previously-lost `total_line` values (same root
cause: a `"+"`-signed total with no o/u marker used to be silently dropped
when the correct spread token had already been captured first) — every
other row in the archive (11 of 12 seasons entirely, plus every row outside
these 209 in season 2009) is byte-identical to the pre-fix table, proven by
a full-table diff against a scratch rebuild before the real tables were
regenerated. See `scripts/backfill_vegasinsider.py::classify_line_tokens`'s
own docstring and `tests/test_vegasinsider_backfill_layout.py` for the fixture
test.

`filter_plausible()`'s full-game-leg guard (dropping a row where the
full-game spread was positive or exceeded 30 points) is now a **structural
no-op** and was removed. A narrower, UNRELATED half-leg-only guard remains:
one row (1H) still carries an implausible HALF spread even after the
ENG-40 fix — away=IND home=TEN, capture `20111026112359`, book HARRAH'S,
`full_spread=-9.0` but `half1_spread=-31.5` (a half spread numerically
LARGER than the full-game spread, impossible for a real quote). Read
directly from the cached line-movement HTML
(`data/raw/vegasinsider/20260822T033952Z/line_movement/20111026112359_bbcbd8fd.html`):
VegasInsider's own page shows `"TEN-31.5"` / `"IND+31.5"` verbatim in the raw
1H-Fav/1H-Dog cells of its last three movement rows for that book — this is
an anomalous/erroneous quote baked into the archive's own source HTML on
the LEAD-60 half-line code path (`build_half_lines`/`extract_book_half_lines`),
a different function from the one ENG-40 fixed, and out of that fix's scope.
`filter_plausible()` keeps a narrow half-leg-only magnitude guard for this
one confirmed case rather than silently passing an impossible value into
the bootstrap; it costs exactly 1 of 593 joined 1H rows and 0 of 332 joined
2H rows.

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
| 1H (LEAD-02) | 89 | 0.5935 | 0.6000 | -0.0 | 1.0 | 0 | **0.5000** |
| 2H (sibling) | 44 | 0.5352 | 0.6077 | -0.0 | 1.0 | 0 | **0.1571** |

(Post-ENG-40 fix, measured 2026-09-05; `n` rose from 87/43 to 89/44 because
2 more games now clear the favorites-3+ eligibility screen with a correctly
classified spread. `n sign-flip (ratio<0)` was 0 before the fix too — this
row was never affected by ENG-40, since a positive `spread_line` fails the
`full_spread <= -3.0` eligibility filter outright rather than producing a
negative ratio.)

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

## Results (measured 2026-09-05, post-ENG-40 fix)

Command: `.\.tools\uv.exe run --no-sync python scripts\lead02_half_line_script_screen.py`
Artifact: `artifacts/lead02_half_line_script/20260905T174147Z/results.json`
(plus `scored_half1.parquet`/`scored_half2.parquet` in the same directory).
Supersedes the pre-fix `20260905T170817Z` run.

### Population funnel

| leg | raw joined rows (both spreads) | implausible dropped | favorites 3+ | deduped games (market-only) | matched to schedule | schedule match rate | pushes dropped | **scored games** | **flagged** |
|---|---|---|---|---|---|---|---|---|---|
| 1H (LEAD-02) | 593 | 1 | 467 | 89 | 72 | 80.90% | 2 | **70** | **12** (17.1%) |
| 2H (sibling) | 332 | 0 | 265 | 44 | 39 | 88.64% | 1 | **38** | **7** (18.4%) |

`implausible dropped` is now the residual half-leg-only anomaly guard
described above (1 row, 1H; 0 rows, 2H) — ENG-40's own contribution to this
column (7 rows 1H / 5 rows 2H, pre-fix) is gone.

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
| full period (2009–2016 actual data) | 70 | 12 | +1.48 | [-3.79, +6.67] | 0.710 |
| era 2005-2010 (actual: 2009-2010 only) | 16 | 2 | -4.46 | [-7.50, -1.67] | 0.000 |
| era 2011-2016 | 54 | 10 | +3.11 | [-3.32, +9.05] | 0.839 |
| **positive control** (oracle, realized-margin-planted, n_flag matched) | 70 | 12 | +11.82 | [+9.80, +13.77] | 1.000 |

**A mild, positive-leaning read — no longer flat, still not a finding.**
Post-ENG-40-fix, the full-period point estimate flips sign relative to the
pre-fix reading (+1.48pts vs the pre-fix -0.71pts) and P+ crosses 0.5 (0.710
vs 0.376), but the interval still straddles zero comfortably and n=70/12
flagged remains a CLOSE-graded lead-generation screen, not a decision. The
whole-interval-negative era-2005-2010 row above is **NOT interpretable as a
resolved wrong sign** — `n_flag=2` in that bucket (up from `n_flag=1`
pre-fix, so no longer literally zero-variance by construction, but still
tiny), and the family's own primary full-period reading is positive; a
resolved-wrong-sign closure requires a well-powered read, not a 16-game era
subgroup pointing the opposite way from its own parent population. This is
flagged explicitly rather than quoted as a finding, and is NOT used as a
`wrong_sign_resolved` closing ground anywhere in this write-up or the
registry entries below.

The positive control (oracle: the 12 games with the largest realized dog
margin, matched exactly to the real cell's flagged count) shows a CI cleanly
excluding zero (+9.80 to +13.77 pts, P+ 1.0) — proving the bootstrap/
population COULD show a real effect of that size if one were present. The
real cell's now-positive-leaning reading sits meaningfully below that
control, at this power a genuine unresolved measurement, not an artifact of
an underpowered or broken instrument.

Permutation nulls (200 draws, full period): the required within-week null is
near-degenerate (9 of 61 week blocks hold 2+ rows; 18 of 70 games sit in a
multi-row block; observed +8.62pts sits inside a null with essentially zero
spread by construction, share-at-or-beyond 0.44). The supplementary
within-season null (8 blocks, all multi-row, all 70 games swappable) gives a
null mean +4.55pts, sd 14.37pts, 95% [-21.55, +28.74]; the observed +8.62pts
sits well inside that band — unremarkable relative to season-shuffle noise
(74.5% of draws at least as extreme).

### 2H sibling: BACK the underdog on a favorable 2H line

| period | n (scored) | n flagged | effect (pts) | week-blocked 95% CI (pts) | P+ |
|---|---|---|---|---|---|
| full period (2009–2016 actual data) | 38 | 7 | **+7.81** | **[+0.36, +14.09]** | **0.978** |
| era 2005-2010 (actual: 2009-2010 only) | 13 | 2 | +3.50 | [-6.59, +13.99] | 0.667 |
| era 2011-2016 | 25 | 5 | **+10.00** | **[+0.95, +16.47]** | **0.981** |
| **positive control** (oracle, realized-margin-planted, n_flag matched) | 38 | 7 | +14.26 | [+11.43, +16.69] | 1.000 |

**Still the more promising of the two cells, still `unresolved_below_power`,
essentially unchanged by the ENG-40 fix** (this leg lost only its 5
pre-existing implausible-drop rows, which the earlier funnel had already
absorbed without materially shifting the reading; the numbers move by
~0.03pts on the full-period effect and stay inside the pre-fix interval).
The full-period and 2011-2016-era intervals both exclude zero on the
predeclared (favorable) side, with P+ above 0.97 in both. Per the binding
taxonomy, an interval EXCLUDING zero on a sample this small (n=38, 7 flagged
games) is **not itself a closing ground either** — the taxonomy's only two
closing grounds are a refuted mechanism (wrong sign) or a positive-control
bound, and neither applies here (the sign is right, and the point estimate
is well above what a null/noise process alone would explain, not bounded
away from a real effect by the control). This stays `unresolved_below_power`
and is recorded as such, flagged as the more promising lead of the two for a
future opener-graded confirmation pass — no play/publish decision follows
from a CLOSE-graded lead-generation screen at n=38.

Permutation nulls (200 draws, full period): the required within-week null is
again near-degenerate (4 of 34 week blocks hold 2+ rows). The supplementary
within-season null (7 blocks, all multi-row, all 38 games swappable) gives a
null mean +7.20pts, sd 19.86pts, 95% [-27.65, +42.40]; the observed
+42.40pts (raw, pre-slate-scaling gap) sits at the null distribution's own
maximum — only 8% of season-shuffle draws are at least as extreme in
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
  opener-graded confirmation pass, not a finding — n=38 with 7 flagged games
  is far below the sample size this project treats as informative on its
  own, and the taxonomy above explicitly does not treat "excludes zero" as
  a closing ground in either direction.
- The 1H (LEAD-02) primary cell is a mild, positive-leaning read at this
  power (post-ENG-40 fix: +1.48pts, P+0.710, interval still crossing zero),
  validated by a positive control that proves the instrument is not simply
  blind — also not closed (per the taxonomy, "contains zero" is never
  grounds for closure either, and P+ crossing 0.5 is not itself a closing
  ground in the other direction).

## 2026-09-05 - Rerun after point-in-time fix 42d78f6

**Decision (inferred):** I think both half-line directions remain challenger leads: the corrected full-period reads favour backing the underdog, while these close-graded screens do not establish an opener-card change.

**Measured:** the offline rebuild and pregame-filtered screen are `artifacts/vegasinsider/cx7_42d78f6_main/rebuild_summary.json` and `artifacts/vegasinsider/cx7_42d78f6_lead02/results.json`; all predeclared seasons, era splits, arms, empirical 20th-percentile encoding, 20,000 bootstrap draws, seeds and 200 null draws were retained.

**Read:** old figures below are the existing post-ENG-40 records, preserved in `artifacts/vegasinsider/cx7_42d78f6_audit/superseded_registry_entries.json`; **measured:** new figures come from the corrected screen above.

| Cell | Old effect / 95% interval (accuracy points; read) | New effect / 95% interval (measured) | probability_positive old -> new | Games old -> new | Split-half reliability old / new |
|---|---|---|---|---|---|
| lead02_half_line_1h | +1.4778 / [-3.7914, +6.6745] | +5.2928 / [-2.5132, +12.6042] | 0.70960 -> 0.90925 | 70 -> 48 | Unmeasured / unmeasured |
| lead02_half_line_1h_era_2005_2010 | -4.4643 / [-7.5000, -1.6667] | -1.1364 / [-16.3636, +19.0909] | 0.00000 -> 0.41038 | 16 -> 11 | Unmeasured / unmeasured |
| lead02_half_line_1h_era_2011_2016 | +3.1145 / [-3.3223, +9.0535] | +7.2693 / [-0.9828, +14.2093] | 0.83930 -> 0.95680 | 54 -> 37 | Unmeasured / unmeasured |
| lead02_half_line_2h | +7.8098 / [+0.3637, +14.0867] | +16.6667 / [+0.0000, +25.0000] | 0.97830 -> 0.92352 | 38 -> 4 | Unmeasured / unmeasured |
| lead02_half_line_2h_era_2005_2010 | +3.4965 / [-6.5934, +13.9860] | Not estimable | 0.66680 -> not estimable | 13 -> 3 | Unmeasured / unmeasured |
| lead02_half_line_2h_era_2011_2016 | +10.0000 / [+0.9524, +16.4706] | Not estimable | 0.98150 -> not estimable | 25 -> 1 | Unmeasured / unmeasured |

**Measured:** 1H has 11 flagged games out of 48; 2H has 1 out of 4. The 2H era cells have 0 flagged/3 complement games and 1 flagged/0 complement games, respectively, so the predeclared subset-versus-complement estimator returns no effect, interval or probability for either era; these are not measured zeros (`results.json`).

**Measured:** the 1H positive control is +16.1036 accuracy points, 95% [12.6437, 19.2204], probability_positive=1.00000; the 2H control is +16.6667, [0.0000, 25.0000], probability_positive=0.92348, with 6,344 unusable resamples out of 20,000 (`results.json`). **Inferred:** I think this 2H control does not establish a candidate-sized absence bound; neither direction has an admissible closing ground.

**Measured:** four estimable cells were replaced via `nfl-ats weak-signals record --replace` as `unresolved_below_power` (`artifacts/vegasinsider/cx7_42d78f6_audit/half_record_commands.json`). **Read:** none of the six old half-line records was a terminal closure (`superseded_registry_entries.json`); the earlier negative 1H-era interval no longer supports even that historical wrong-sign reading. **Read:** the registry requires a finite effect (`src/nfl_ats/weak_signals.py:471`), whereas both corrected 2H-era estimates are absent; their old numerical records must not be treated as corrected rerun evidence.

**Read:** the screen does not estimate split-half trait reliability (`scripts/lead02_half_line_script_screen.py:480`); **inferred:** I think an unmeasured reliability must remain unmeasured rather than be replaced by zero.

**Measured:** the rebuilt main table retains 4,390 rows, changes 1,313 spread values (1,067 become unavailable; 246 change while remaining quoted), rejects 14,246 future movements, and marks 612 rows in-play; 0 capture timestamps change in these cached files (`artifacts/vegasinsider/cx7_42d78f6_audit/cache_comparison.json`). **Measured:** excluding in-play records leaves 818 quoted 1H rows and 54 quoted 2H rows before the screen joins and exclusions (`cache_comparison.json`).

**Read:** the earlier half/full join-rate constant in the screen output describes the old archive, not the rerun; **measured:** the corrected screen joins 351 usable 1H and 27 usable 2H quote rows (`results.json`). The full-game `season_<year>.parquet` files remain the ENG-40-corrected comparator; old half tables remain preserved as evidence.

**Measured registry limitation:** both unestimable 2H-era rows were re-recorded with an explicit INVALIDATED description and notes via the record CLI; their old historical point estimates remain, with interval/probability removed (`artifacts/vegasinsider/cx7_42d78f6_audit/half_invalidation_commands.json`). **Read:** `poolable_signals` still selects every unresolved row (`src/nfl_ats/weak_signals.py:1029`), so these two invalidated rows must be excluded explicitly from any new pool or sign test; the permitted CLI has no missing-estimate/removal state (`nfl-ats weak-signals --help`, read this session). **Inferred:** I think changing the historical point estimates to invented zeros or assigning an inadmissible closure would be worse than exposing this schema limitation.

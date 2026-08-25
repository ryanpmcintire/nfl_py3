# MKT-03: no-vig market probability diagnostics

Written 2026-08-18. Companion module: `src/nfl_ats/novig.py`. Screen script:
`scripts/novig_diagnostics_screen.py`. Tests: `tests/test_novig.py` plus
extensions to `tests/test_clv.py`.

## What this is and is not

**This is a diagnostic, not a predeclared confirmation.** It produces no
pick, scores no candidate model, and gates no modeling decision, so it is
not an NFL "confirmation look" in the sense `docs/rotation_registry.md`
governs. Rule 8 of that registry states this plainly: *"CFB and non-reserved
seasons stay free. The registry governs NFL confirmation looks only."*
`docs/rotation_registry.md` does not contain a separate, explicit "NFL
diagnostics are free" carve-out distinct from that CFB-focused rule 8 — this
document's judgment call, adjudicated by the orchestrating session rather
than found in writing, is that the same logic extends to purely descriptive
NFL market measurement that declares no family, draws no confirmation
window, and produces no accept/reject verdict. Nothing here calls
`nfl_ats.rotation.assign_window`/`record_look`, and no `registry/*.json` file
is touched by this work. If that judgment turns out to be wrong, the fix is
cheap: nothing here has been treated as evidence for a pick.

**Consuming any output of this diagnostic as a model feature is explicitly
out of scope.** If a later session wants to feed
`no_vig_home_cover_probability`, `spread_hold`, or the calibration-bucket
findings below into the `market_residual` model (or any other model), that
is a new candidate signal and must go through the project's normal
look-spending process — a declared family, an assigned confirmation window,
`nfl-ats weak-signals record` for anything that lands as category-3 —
exactly like any other candidate. Nothing in this document should later be
cited as having already cleared that bar. For the same reason, this document
does not itself call `nfl-ats weak-signals record` on the calibration
findings in the "Results" section below: they are not being proposed as a
signal here, only measured.

**No accuracy claim.** Per project memory ("Edge means beating 50%"), the
only claim that matters for the primary goal is forced-pick ATS accuracy
against the coin flip at the opener, and nothing in this diagnostic computes
that. This measures market microstructure — is the archive's price
information redundant with the line, and is the no-vig probability derived
from it well-calibrated — not accuracy.

## Question

`src/nfl_ats/odds.py` already has `implied_probability`, `no_vig_probabilities`,
and `market_hold` — audited, unit-tested (`tests/test_odds.py`), general
math. **[read]** grepping the codebase for their call sites shows they are
already used elsewhere: `nfl_ats.margin`'s `target="market"` cover
probability, and `nfl_ats.backtest`'s `market_home_no_vig_probability`/
`market_hold` columns (cross-checked by `nfl_ats.prediction_safety`) — but
every one of those call sites reads a single, already-existing per-game
`home_spread_odds`/`away_spread_odds` feature column, sourced upstream of
`nfl_ats.clv` (traced to `nfl_ats.cfb_features`/nflverse-derived schedule
fields — a single fixed price per game, not point-in-time, not per-book).
**This corrects an earlier scoping note that claimed these functions were
unused outside `odds.py`/`test_odds.py` — that claim was wrong; the
correct, narrower gap is what follows.**

This project also runs a purchased, point-in-time, multi-book odds archive
(`data/market/raw/`, via `nfl_ats.clv`) with a `price` column on every
`spreads`-market quote, at every decision label, for every book. Inside
`nfl_ats.clv.decision_market_consensus`, a per-(game, decision label, side)
median `consensus_price` is already computed for the `spreads` market — but
`build_pairing_table` never carries it through; only the spread's LINE
survives (`home_spread`/`spread_min`/`spread_max`/`spread_std`). The PRICE
is computed and silently dropped every time a pairing table is built. The
questions this diagnostic answers:

1. Is that dropped price actually informative — does it vary enough, across
   the archive, to not just be a restatement of the half-point line movement
   `spread_line` already captures?
2. If a no-vig probability is computed from it, is it well-calibrated (the
   textbook favourite-longshot check), or does it show a bucket-level bias?
3. As a secondary, same treatment for moneyline (`h2h`) prices, which the
   pairing table already carries but nothing has applied vig-removal to.

## Method (not frozen — this is descriptive measurement, re-runnable)

### New code, and what it deliberately does not touch

- **`nfl_ats.clv.spread_price_consensus_table`** (new function, added in
  `clv.py` section 2, immediately after `close_reference_table`): a sibling
  to `build_pairing_table` that surfaces the same `decision_market_consensus`
  output's `consensus_price` for the `spreads` market, home and away sides,
  at the same join keys (`game_id`, `season`, `week`, `decision_label`,
  `capture_kind`). It duplicates `build_pairing_table`'s own quote-loading,
  true-week-correction, and monotone-timeline steps rather than factoring
  them into a shared helper, specifically so `build_pairing_table` has zero
  code-path overlap with this addition.
  - **[measured]** `git diff` on `src/nfl_ats/clv.py` shows the new function
    is a pure insertion (`@@ -554,0 +555,94 @@`); the diff touches zero lines
    inside `build_pairing_table` itself.
  - **[measured]** empirically as well: `build_pairing_table` was run against
    the real local archive (2023 season, `tue_open`/`sun_late_close`/
    `sun_early_close`, 623 rows) before this work started and again after
    every edit in this task was complete;
    `pandas.testing.assert_frame_equal(before, after, check_exact=True)`
    passes. The frozen-inputs invariant (`docs/archive/opus_execution_specs.md`,
    "the frozen model's inputs are sacred... new COLUMNS are fine; changed
    VALUES in existing columns are not") holds by the strongest available
    proof: nothing changed at all.
- **`src/nfl_ats/novig.py`** (new module): pure functions, no I/O.
  `spread_novig_probabilities`/`moneyline_novig_probabilities` call
  `nfl_ats.odds.no_vig_probabilities`/`market_hold` row-by-row (not
  re-derived as a vectorized formula — one audited definition of the math,
  reused exactly the way `nfl_ats.margin`/`nfl_ats.backtest` already do).
  `favourite_longshot_calibration` buckets a probability column against a
  realized 0/1 outcome; `calibration_gap_metric_fn`/`bootstrap_calibration_gaps`
  wrap `nfl_ats.clv.week_blocked_bootstrap` with a `metric_fn` shaped exactly
  like `clv_summary`/`opener_evaluation_metrics` — no new bootstrap code.
- **`scripts/novig_diagnostics_screen.py`**: reads only committed local
  snapshots via `nfl_ats.clv`'s loaders — never the Odds API. Restricted to
  the 6 decision labels the archive carries `h2h` (moneyline) coverage at
  (`tue_open`, `thu_pre_tnf`, `sat_midday`, `sun_early_close`,
  `sun_late_close`, `mon_pre_mnf`) rather than `labels=None`, which would
  also load the archive's 6,966 `intraday_hourly` snapshots that no
  diagnostic here needs.
- **Not touched:** `src/nfl_ats/odds.py`, `nfl_ats.clv.build_pairing_table`,
  `nfl_ats.clv.decision_market_consensus`, `src/nfl_ats/public_board.py`,
  any `docs/*.html`, `registry/rotation_registry.json`,
  `registry/weak_signals.json`.

### A correctness point beyond what was scoped: which line the outcome is graded against

The spread no-vig probability at a given decision label is a probability of
covering **that label's own spread line**, not necessarily the schedule's
closing `spread_line`. `nfl_ats.features.add_ats_outcomes`'s `home_cover`
column is defined relative to the schedule's close, so using it directly to
grade a `tue_open`-priced probability would silently mismatch the line the
probability was priced against whenever the line moved between open and
close (routine). The screen script instead recomputes the realized outcome
at each row's own decision-label line
(`home_cover_at_label = sign(result - home_spread_at_that_label)`, same
push convention as `add_ats_outcomes`/`nfl_ats.clv.pick_correct` — a push is
excluded, not scored as 0). **[inferred]** this is a refinement beyond the
original design note, made here because grading against the wrong line
would silently invalidate the calibration read below; it does not change
anything about the sibling-extraction or public-pages scoping decisions.

### Realized-outcome definitions

- ATS arm: `home_cover_at_label`, defined above (graded against the
  `tue_open` line specifically, matching the probability it is bucketed
  against).
- SU arm: `home_win = sign(result)`, from `regular_season_rows` of
  `data/processed/game_features_player.parquet`'s `result` column (a tie is
  excluded, the same way a push is, rather than scored).

### Bucketing and bootstrap

Bucket edges are quantile-based (`buckets=5` requested), computed **once**
on the full available sample and then reused identically for the
point-estimate table and every bootstrap resample — recomputing edges per
resample would make the buckets themselves move. `week_blocked_bootstrap`
(2,000 samples, seed 20260818, both `block="week"` and `block="season"`) is
reused unmodified from `nfl_ats.clv`.

**A read on `mean_abs_calibration_gap`'s `probability_positive`:** this
summary metric averages the *absolute value* of each bucket's signed gap,
so by construction it is virtually always positive — its
`probability_positive` is not evidence of anything and should not be read
that way; only its estimate/CI (a magnitude range) is informative. The real
directional evidence is each bucket's own **signed** `calibration_gap` and
that bucket's own `probability_positive`, reported below.

## Results

**[measured]**, run `20260818T213648Z`
(`artifacts/novig_diagnostics/20260818T213648Z/`), git revision
`cfa3ecc4f5f873cdfa350234d1b9b8918bbb8fb9`, 2,000 bootstrap samples, seed
20260818, `data/market/raw` as of this session.

### 3a. Is the dropped spread price actually informative?

Loading raw `spreads`-market quotes at the 6 labels this screen uses:
**438,424** quote rows with a non-null price; **55.50%** are NOT exactly
`-110` (span `-105, -115, -112, -120, +100, -108, ...`). An earlier scoping
pass **[reported, unverified by this session beyond the count above]**
measured 52.8% across *every* historical decision label including
`intraday_hourly` (a different, wider denominator than the 6-label scope
used here) — the two numbers are not directly comparable, but both say the
same thing: a majority of archived spread quotes carry price information a
half-point line cannot. Since a spread line only moves in half-point
increments but price can encode finer directional lean between line moves,
a no-vig probability built from price is not a mechanical restatement of
`spread_line`.

### 3b/3c. Favourite-longshot calibration at the Tuesday opener

**ATS arm** (`no_vig_home_cover_probability`, graded against the `tue_open`
line, 1,503 of 1,537 `tue_open`-paired games 2020–2025; **[measured]** all 34
excluded rows are exactly-zero `ats_margin` at the `tue_open` line, i.e.
pushes against that line):

With 5 quantile buckets requested (6 quantile edge points at 0/20/40/60/80/
100%), the sample collapsed to **2 realized buckets** — **[measured]** the
raw quantile values are `[0.0349, 0.5, 0.5, 0.5, 0.5, 0.9909]`: the 20th,
40th, 60th, and 80th percentiles of `no_vig_home_cover_probability` are all
*exactly* 0.5 (only 3 of the 6 values are distinct once duplicates
collapse, before the outer two are widened to ±inf), so the no-vig
probability derived from spread price is tightly clustered at pick'em,
unlike the moneyline arm below. This is itself informative: price variation
around the spread line is real (3a) but mostly *fine-grained* rather than
*large*, so it does not spread the no-vig probability across the full [0,1]
range the way moneyline does.

| bucket | range | n | mean predicted p | mean observed freq | gap | week 95% CI | week P(gap>0) | season 95% CI | season P(gap>0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (−∞, 0.500] | 1,215 | 0.4963 | 0.4947 | −0.0016 | [−0.0278, +0.0254] | 44.9% | [−0.0205, +0.0190] | 42.2% |
| 1 | (0.500, ∞) | 288 | 0.5209 | 0.5035 | −0.0175 | [−0.0769, +0.0391] | 27.6% | [−0.0706, +0.0339] | 30.5% |

`mean_abs_calibration_gap`: week-blocked 0.0096, 95% [0.0028, 0.0452];
season-blocked 0.0096, 95% [0.0027, 0.0432] (magnitude only —
`probability_positive` not informative for this metric, see above).

Reading: near pick'em (bucket 0), the market is essentially perfectly
calibrated (gap −0.16 percentage points, wide interval straddling zero). Above the
median (bucket 1, mean predicted 52.1%), the observed cover rate (50.3%)
sits below the no-vig price, a small apparent overpricing of the favored
side — but both blockings' 95% intervals cross zero (**not a rejection,
per the project's binding interval rule**; this is unresolved at this
sample size, not a finding). No claim is made here in either direction.

**SU arm** (`no_vig_home_win_probability`, secondary-goal-only, 1,532 of
1,537 `tue_open` games with resolved moneyline; **[measured]** all 5
excluded rows are the archive's actual NFL ties — `2020_03_CIN_PHI`,
`2021_10_DET_PIT`, `2022_01_IND_HOU`, `2022_13_WAS_NYG`, `2025_04_GB_DAL`,
each `result == 0.0`):

| bucket | range | n | mean predicted p | mean observed freq | gap | week 95% CI | week P(gap>0) | season 95% CI | season P(gap>0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (−∞, 0.368] | 307 | 0.2806 | 0.2834 | +0.0028 | [−0.0484, +0.0546] | 52.8% | [−0.0504, +0.0492] | 58.3% |
| 1 | (0.368, 0.499] | 306 | 0.4260 | 0.3856 | −0.0404 | [−0.0899, +0.0089] | 6.1% | **[−0.0982, −0.0076]** | **0.0%** |
| 2 | (0.499, 0.606] | 307 | 0.5593 | 0.5472 | −0.0120 | [−0.0692, +0.0437] | 33.0% | [−0.0606, +0.0318] | 29.0% |
| 3 | (0.606, 0.717] | 305 | 0.6595 | 0.6623 | +0.0028 | [−0.0534, +0.0621] | 54.8% | [−0.0706, +0.0927] | 49.0% |
| 4 | (0.717, ∞) | 307 | 0.7867 | 0.7883 | +0.0016 | [−0.0483, +0.0460] | 53.8% | [−0.0439, +0.0351] | 56.2% |

`mean_abs_calibration_gap`: week-blocked 0.0119, 95% [0.0117, 0.0443];
season-blocked 0.0119, 95% [0.0114, 0.0578] (magnitude only, see above).

Reading: bucket 1 (moderate home underdogs, no-vig win probability
37–50%) shows a season-blocked interval that excludes zero — home teams in
this range won straight-up 38.6% of the time against a no-vig-implied
42.6%, a 4.0-point gap. The week-blocked interval for the same bucket does
**not** exclude zero (upper bound +0.89pp). **This is reported as
continuous evidence, not a finding**, for two reasons stated plainly: (1)
this is the SU arm, secondary-goal-only — the pool grades ATS, not straight
up; (2) seven buckets were read across both arms with no multiple-comparison
correction, and one crossing at nominal 95% is within chance expectation
even under a true null. Per section "What this is and is not" above, this
number is not recorded to `registry/weak_signals.json` here because it is
not being proposed as a candidate signal in this document.

### 3d. Hold trended by decision label

| decision_label | market | mean hold | n |
|---|---|---|---|
| tue_open | spread | 4.25% | 1,537 |
| sun_early_close | spread | 4.16% | 1,396 |
| sun_late_close | spread | 4.06% | 473 |
| tue_open | moneyline | 4.16% | 1,537 |
| sat_midday | moneyline | 4.14% | 750 |
| sun_early_close | moneyline | 3.93% | 1,396 |
| sun_late_close | moneyline | 4.00% | 244 |
| thu_pre_tnf | moneyline | 3.91% | 806 |
| mon_pre_mnf | moneyline | 3.40% | 61 |

Hold is stable (3.4–4.3%) across labels and slightly higher at the opener
than at the close, consistent with thinner opener liquidity — descriptive,
not accuracy-moving by itself, and not chased further here.

**Scoped out, flagged (decision needing review):** the design note that
grounded this work also proposed a per-side price-dispersion column
(`price_std`, alongside the line's existing `line_std`) as a second
liquidity proxy. Producing it would require adding a column inside
`nfl_ats.clv.decision_market_consensus`'s own aggregation — a function
`build_pairing_table` also depends on. Even though a new, unselected column
would be consistent with the "new columns are fine" half of the
frozen-inputs invariant, touching shared code inside `decision_market_consensus`
was judged higher-risk than the sibling-function pattern used for price
itself, for a diagnostic that is not required to ship in this task. Left
undone; `books` counts (`home_spread_price_books`/`away_spread_price_books`,
already in `spread_price_consensus_table`'s output) stand in as a coarser
liquidity proxy for now.

## An observation, not chased (per the orchestrating session's adjudication)

**[measured]** `nfl_ats.clv.load_snapshot_manifest_index` on the local
archive shows exactly **2** `capture_kind="live"` manifests, and
`load_decision_quotes(..., capture_kind="live")` shows **10** distinct
`bookmaker_key` values (`betmgm, betonlineag, betrivers, betus, bovada,
draftkings, fanduel, lowvig, mybookieag, williamhill_us`) across 9,154 live
quote rows. `ROADMAP.md`'s MKT-02 entry describes "six weekly scheduled live
captures running (11 books)." This is noted here as an observation, not
chased further: today is 2026-08-18, three weeks before the 2026-09-08
season lock, so it may simply reflect preseason timing (weekly captures not
yet at full cadence, or this checkout not having synced every live snapshot)
rather than a real regression in capture coverage. **Unverified beyond the
counts above** — a reviewer relying on live-capture cadence for production
work should check current coverage directly rather than trusting the
ROADMAP figure or this note.

## Explicit non-claims

- No accuracy claim. This measures market microstructure, not forced-pick
  ATS accuracy.
- The ATS arm's bucket-1 apparent overpricing and the SU arm's bucket-1
  underpricing are both reported with intervals that either cross zero
  (ATS; SU week-blocked) or exclude it only under one of two blockings and
  only one arm of secondary-goal relevance (SU season-blocked) — neither is
  a finding, and neither is recorded as a signal.
- Nothing here has been fed into any model, feature table, or the picks
  pipeline. `data/processed/game_features_*.parquet` is unchanged.
  `registry/rotation_registry.json` and `registry/weak_signals.json` are
  unchanged.
- Consuming any of this as a candidate feature requires a separate,
  future, properly look-spent pass — see "What this is and is not" above.

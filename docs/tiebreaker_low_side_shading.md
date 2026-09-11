# Tiebreaker low-side shading (LEAD-54)

Predeclared design read from `ROADMAP.md` (LEAD-54 row): the pool breaks
ties on the final score of the week's last game; public tiebreak entries are
believed to cluster over the market total because the public loves overs, so
the predeclared direction is to shade the guessed total to the LOW side of
the market total. The row's own history (`docs/prospective_bestpick_tiebreaker.md`)
already fixed the shade magnitude for the live, forward-looking prospective
arm at exactly 1.0 point under market total. This document is the
retrospective, full-history build: it measures the under-rate on 17 seasons
of completed games, simulates the tie-break itself, and reports where a
1-4 point shade helps or hurts, plus a chronological out-of-sample check.

`POL-12` (read in full) states plainly that **the pool's actual tie-break
metric has never been captured** -- the row exists specifically because
nobody recorded whether the pool judges "closest total" or "closest exact
score." `docs/score_lattice.md` (SS1.5-1.6, SS2.2) already established
**closest-total absolute error** as the working proxy for exactly this
reason, so that is the primary metric here; a secondary **closest exact
score** (Manhattan distance on the two teams' points) reading is also run
so the shade is not tuned to a metric assumption the project cannot verify.

## Data used

**Market total, 2009-2025 (primary population).** `nfl_ats.tiebreaker`'s own
convention -- `snapshot_consensus`'s fallback path and every historical
neighborhood computation in `build_report`/`lined_finals` -- treats
`schedules.parquet`'s `total_line` column as "the market total." Read
(`data/raw/20260908T162105Z/schedules.parquet`): `total_line` is populated
for all 17 REG seasons, 256 games/season 2009-2020, 271-272 games/season
2021-2025 (17-game schedule). That is the total used for the full-history
under-rate, shade comparison, sensitivity sweep and out-of-sample check
below.

**Checked what a true opening total would add, per season, before using
it.** `data/market/raw` (The Odds API archive) and
`nfl_ats.schedule_flag_features.default_opener_lines` (built from
`build_pairing_table`'s `tue_open` label) only resolve a genuine
Tuesday-opener total for **2020-2025** -- measured
(`default_opener_lines(schedules)` against the 2009-2025 REG schedule):
`tue_open_total_line` populated for 216-272 games/season from 2020 onward,
**zero** for every season 2009-2019. `data/market/historical/open_close`
(a Kaggle multi-book opener/closer sample) covers a single season, 2025,
17,100 quote rows -- also too narrow to extend the opener series backward.
Using a true Tuesday opener for the full 2009-2025 window is not possible
with data on disk; the closing/consensus `total_line` is the only total
available across all 17 seasons and is what production already calls "the
market total" for this exact tiebreaker. A same-population **robustness
check restricted to 2020-2025** (n=101 last-games with both totals) is
reported below.

**Model lean.** `artifacts/totals_backtest/20260901T184010Z/predictions.parquet`
(the POL-12 walk-forward totals regime, min-500 warm-up) supplies
`predicted_residual` per game, 2010-2025; joined to the last-game-of-week
population by `game_id` for the model-lean-proportional shade arm (261 of
294 rows match).

## Method

`scripts/tiebreaker_low_side_shading.py` builds one row per REG season/week
2009-2025 for that week's LAST game (`nfl_ats.tiebreaker.last_game_of_week`,
reused not reimplemented), keeping rows with a market total, a market
spread and both final scores. 294 games, 17 seasons (17 weeks/season
2009-2020, 18 weeks/season 2021-2025, one game per week).

**Public field, declared before any shade result was read** (per the lane
brief): a simulated public entrant's total guess is `market_total +
Normal(mean=+1.5, sd=6)` -- 20,000 Monte Carlo draws per game, common random
numbers shared across every offset and the served guess so a null shade
(offset 0) scores an exact, non-noisy zero against itself. A sensitivity
sweep varies the mean offset over {0, +1, +1.5, +2, +3} and the sd over
{4, 6, 8, 10}.

**Closest-total win rule:** whichever guess (served or a candidate shade)
has the smaller `|guess_total - actual_total|` wins that draw; an exact tie
splits 0.5/0.5. **Closest-exact-score win rule (secondary):** both the
served/shaded guess and the public draw are converted to an integer score
using the market's own spread margin (`nfl_ats.tiebreaker.market_implied_scores`,
rounded) combined with the (possibly shaded) guessed total; the win rule is
the smaller Manhattan distance `|home error| + |away error|` to the actual
final score. This deliberately does NOT reuse the production lattice-
consistency machinery (`nfl_ats.score_lattice`), because that machinery
needs an active-model pick side that only exists from ~2020 onward; building
a full historical lattice-consistent counterfactual for every shaded total
back to 2009 is a separate, heavier build and is named as a follow-up below
rather than approximated.

Every comparison is a paired bootstrap over 4,000 draws, block = **season**
(each observation here already IS one week, so the natural correlated unit
one level up is the season, e.g. weather/rule-era effects shared across a
season's weeks), fixed seed 20260910, using
`nfl_ats.clv.week_blocked_bootstrap(..., block="season")` and
`nfl_ats.evidence_conventions.probability_positive_from_draws`.

Closing-grounds taxonomy in force for every verdict below (binding,
restated verbatim per AGENTS.md): an interval or CI that contains zero is
never grounds to reject, fail or close a line of work. Only two grounds ever
close one: (1) a refuted mechanism -- a RESOLVED wrong sign, the whole
interval on the wrong side of zero, or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`, recorded with
`nfl-ats weak-signals record`, reporting `probability_positive`, never the
binary "contains zero."

## 1. Under-rate

**Measured** (`artifacts/tiebreaker_low_side_shading/20260911T023009Z/summary.json`,
`under_rate_by_season.parquet`, `under_rate_by_bucket.parquet`): across all
294 last-games-of-week 2009-2025, the actual combined score landed UNDER the
market total in **53.06%** of games, 95% season-blocked CI **[48.30%,
57.97%]**, `probability_positive` (that the true under-rate exceeds the
50/50 coin flip) **0.887**. The interval crosses 50% -- per the binding rule
above, that is the EXPECTED shape for a real small signal at this
resolution, not grounds to reject the predeclared "public loves overs"
mechanism.

By season (games, under-rate, push-rate, over-rate):

| Season | Games | Under | Push | Over |
|---|---:|---:|---:|---:|
| 2009 | 17 | 41.2% | 0.0% | 58.8% |
| 2010 | 17 | 47.1% | 0.0% | 52.9% |
| 2011 | 17 | 47.1% | 0.0% | 52.9% |
| 2012 | 17 | 64.7% | 0.0% | 35.3% |
| 2013 | 17 | 58.8% | 0.0% | 41.2% |
| 2014 | 17 | 35.3% | 0.0% | 64.7% |
| 2015 | 17 | 70.6% | 0.0% | 29.4% |
| 2016 | 17 | 52.9% | 0.0% | 47.1% |
| 2017 | 17 | 47.1% | 0.0% | 52.9% |
| 2018 | 17 | 52.9% | 0.0% | 47.1% |
| 2019 | 17 | 64.7% | 5.9% | 29.4% |
| 2020 | 17 | 47.1% | 5.9% | 47.1% |
| 2021 | 18 | 50.0% | 0.0% | 50.0% |
| 2022 | 18 | 61.1% | 5.6% | 33.3% |
| 2023 | 18 | 72.2% | 0.0% | 27.8% |
| 2024 | 18 | 44.4% | 0.0% | 55.6% |
| 2025 | 18 | 44.4% | 0.0% | 55.6% |

Season-level under-rate swings widely (35.3% to 72.2%) with no visible
trend; the pooled 53.06% is not one season carrying the whole result (2012,
2013, 2015, 2019, 2022, 2023 all sit above 55%, 2009, 2014, 2024, 2025 all
sit below 45%, and both groups span the full 2009-2025 range).

By market-total quartile bucket (data-driven `pd.qcut`, duplicates
collapsed):

| Bucket (market total) | Games | Under | Push | Over |
|---|---:|---:|---:|---:|
| 33.5-42.5 | 82 | 43.9% | 0.0% | 56.1% |
| 42.5-45.5 | 74 | 63.5% | 1.4% | 35.1% |
| 45.5-48.5 | 74 | 47.3% | 2.7% | 50.0% |
| 48.5-63.5 | 64 | 59.4% | 0.0% | 40.6% |

Not monotonic in the market total: the lowest-total bucket actually leans
OVER (56.1%), the next bucket up leans UNDER hardest (63.5%), and the
highest-total bucket leans under again (59.4%). A single fixed-point shade
cannot fit this shape uniformly; it is reported as a real, unresolved
pattern rather than smoothed into a single number.

**Robustness: true Tuesday-opener total vs. the closing total it stands in
for, 2020-2025 only** (`n=101` last-games with both totals; opener total
from `nfl_ats.schedule_flag_features.default_opener_lines`, closing total
from `schedules.total_line`): under-rate on the closing total **53.47%**,
under-rate on the true opener total **55.45%** (opener runs +0.19 points
above closing on average, giving slightly more room to land under); the two
totals agree on the under/over classification for **94.1%** of games. The
finding is not an artifact of using the closing line as a stand-in for the
market total.

## 2. Reliability (odd vs even seasons)

**Measured**: pairing each odd season with the following even season
(2009-2010, 2011-2012, ..., 2023-2024; 2025 left over, 8 pairs) and
correlating the two halves' season-level under-rates (season-pair bootstrap,
4,000 draws): Pearson r = **-0.384**, 95% **[-0.881, +0.195]**,
`probability_positive` **0.068**. Read plainly: this leans NEGATIVE and the
interval sits mostly below zero, but its upper bound (+0.195) is still above
zero, so this is NOT a `no_split_half_reliability` closing ground under the
binding rule (which requires the reliability to be resolved at/near zero,
not merely negative-leaning on 8 pairs) -- reported honestly rather than
either oversold or auto-closed.

Pooling each half wholesale (not pair-by-pair) tells a different, and more
reassuring, story: odd-numbered seasons (2009, 2011, ..., 2025; 156 games)
under-rate **55.13%**, 95% **[48.08%, 62.66%]**; even-numbered seasons
(2010, 2012, ..., 2024; 138 games) under-rate **50.72%**, 95% **[44.85%,
56.83%]**. Both halves lean the same direction (above 50%), consistent with
the pooled 53.06% headline. The season-pair correlation and the half-pooled
means are different estimands: WHICH season is high or low does not predict
its calendar-adjacent partner well (the negative-leaning pair correlation),
but the population-wide lean itself shows up in both halves independently.
Precedent for reading these two together this way:
`docs/coaching_leads.md`'s Denver Q4 finding, where a negative season-pair
correlation coexisted with both halves' pooled means agreeing in sign and
neither result closed the other.

## 3. Shade comparison, closest-total metric (primary)

**Measured** (`shade_comparison_total_metric.parquet`; served = raw market
total, offset 0; positive AE-improvement and positive win-rate-improvement
both favour the shade; public field mean +1.5, sd 6):

| Shade | Mean |guess-actual| | AE improvement vs served | 95% CI | P+ | Win rate vs public | Win-rate improvement (pp) vs served | 95% CI | P+ |
|---|---:|---:|---|---:|---:|---:|---|---:|
| 0 (served) | 10.1854 | -- | -- | -- | 58.46% | -- | -- | -- |
| -1 | 10.1378 | **+0.0476** | [-0.0509, +0.1502] | **0.816** | 58.32% | -0.1368 | [-0.9352, +0.6614] | 0.362 |
| -2 | 10.1514 | +0.0340 | [-0.1741, +0.2483] | 0.610 | 57.61% | -0.8428 | [-2.5239, +0.8308] | 0.159 |
| -3 | 10.2330 | -0.0476 | [-0.3584, +0.2770] | 0.376 | 56.32% | -2.1378 | [-4.5606, +0.2311] | 0.037 |
| -4 | 10.3690 | -0.1837 | [-0.5993, +0.2467] | 0.195 | 54.68% | **-3.7737** | **[-6.7823, -0.8128]** | **0.005** |

**What this implies for the decision, before what is wrong with it.** On
the absolute-error metric, the -1 shade (LEAD-54's predeclared magnitude)
is the best-performing offset tested: it is the only one with
`probability_positive` above 0.8, and -2 is second-best; -3 and -4 both
flip to hurting AE on average. On expected value alone (AGENTS.md: a
promotion bar is not a decision bar; `probability_positive` above 0.5
favours playing it), the -1 shade is worth taking on this metric, not
worth waiting on.

The win-rate metric (a coarser, threshold win/lose outcome against an
ALREADY-biased public field) is far less sensitive to a small shade: -1
and -2 both cross zero and lean slightly negative, because the served
(unshaded) guess already wins most of its edge simply from being unbiased
against a public that is not, leaving little marginal win-rate for a
further nudge to capture. **The -4 shade is the one RESOLVED result in
this table**: its whole 95% win-rate interval sits below zero
(`refuted_mechanism`, closing ground `wrong_sign_resolved`) -- shading a
full 4 points low measurably costs tiebreakers against this declared
public field. This closes the -4 arm only; it says nothing about -1/-2/-3,
which are separate cells and remain `unresolved_below_power`.

## 4. Sensitivity sweep (-1 shade, win-rate metric)

**Measured** (`sensitivity_sweep.parquet`; -1 shade only, the predeclared
direction), win-rate improvement (pp) vs served across public-field
parameters:

| Public mean offset | sd=4 | sd=6 | sd=8 | sd=10 |
|---|---:|---:|---:|---:|
| 0.0 | +0.273 (P+ 0.667) | +0.145 (P+ 0.622) | +0.091 (P+ 0.594) | +0.069 (P+ 0.583) |
| 1.0 | -0.042 (P+ 0.459) | -0.039 (P+ 0.451) | -0.039 (P+ 0.443) | -0.022 (P+ 0.458) |
| 1.5 | -0.198 (P+ 0.355) | -0.137 (P+ 0.362) | -0.105 (P+ 0.365) | -0.065 (P+ 0.391) |
| 2.0 | -0.348 (P+ 0.252) | -0.209 (P+ 0.297) | -0.171 (P+ 0.295) | -0.124 (P+ 0.316) |
| 3.0 | -0.621 (P+ 0.088) | -0.412 (P+ 0.133) | -0.300 (P+ 0.163) | -0.235 (P+ 0.179) |

Every cell's 95% interval crosses zero (not shown in the table above for
space; see `sensitivity_sweep.parquet` for the full bounds) -- none of this
sweep is resolved either way. The pattern makes mechanical sense and is
reported as read, not oversold: when the assumed public bias is smaller than
or similar to the measured true under-lean (mean offset 0), a small low
shade adds win-rate on top of the served guess; as the assumed public bias
grows past that (mean offset 1.5+), the served guess alone already captures
most of the "beat an over-biased public" edge, and the extra shade adds
little or trends slightly negative. The declared base case (mean +1.5, sd
6) sits inside this less-favourable region for the win-rate metric even
though the AE metric already favours -1 there.

## 5. Model-lean-proportional shade

Rather than a fixed offset, this arm shades the total by the ALREADY-SERVED
production blend `0.1 x predicted_residual`
(`nfl_ats.tiebreaker.TOTALS_RESIDUAL_WEIGHT`, reused from `docs/totals_model.md`
rather than re-derived), i.e. a shade sized to the model's own per-game
under/over lean, on the 261 last-games-of-week with a walk-forward
`predicted_residual` (2010-2025).

**Measured**: AE improvement vs raw market **-0.0142**, 95% **[-0.0446,
+0.0170]**, P+ **0.190**; win-rate improvement **-0.0994pp**, 95%
**[-0.3405, +0.1507]**, P+ **0.215**. Both cross zero;
`unresolved_below_power`. This matches the pre-existing finding in
`docs/totals_model.md` (the same 0.1 blend measured `probability_positive`
0.583 on the full walk-forward archive) -- restricted to just the
last-game-of-week subset, the blend is, if anything, a hair weaker, not a
free source of extra edge beyond what a FIXED low shade already provides.

## 6. Secondary metric: closest exact score

**Measured** (`exact_score_check`, -1 shade only; scores built from the
market spread margin plus the (shaded) total, rounded, no lattice
consistency machinery -- see Method): Manhattan-distance improvement vs
served **-0.0578** points, 95% **[-0.2389, +0.1199]**, P+ **0.257**;
win-rate improvement **-2.555pp**, 95% **[-6.785, +2.010]**, P+ **0.121**.
Both cross zero and both lean the OTHER way from the closest-total reading.
`unresolved_below_power` on both. Read plainly: **the -1 shade's benefit is
specific to the closest-total reading of the pool's rule.** Since POL-12
never captured which rule the pool actually uses, this divergence is exactly
the sort of thing that should be captured, not glossed over -- if the pool
in fact judges the exact score, this analysis gives no support for shading
low.

## 7. Chronological out-of-sample check

Trained (shade selection only) on 2009-2017 (153 games, 9 seasons), tested
strictly on 2018-2025 (141 games, 8 seasons), no re-selection on the test
half.

- By **AE-improvement**, the training half itself already picks **-1** as
  the best offset (matching LEAD-54's predeclared direction) -- the
  training data did not need a different answer coaxed out of it.
- By **win-rate-improvement** (declared public field), the training half
  picks **0** (no shade) as best; scoring that OOS is a degenerate 0-vs-0
  comparison (identical to the served arm by construction), consistent
  with SS3-4 above (win-rate is the less sensitive of the two metrics at
  this public-field parameterization).
- **Scoring the AE-selected -1 shade on the 2018-2025 held-out seasons**:
  AE improvement **+0.0780**, 95% **[-0.0638, +0.2254]**, P+ **0.843** --
  the same direction and a similar magnitude to the full-sample and
  training-only reads, i.e. the -1 shade's AE edge replicates
  out-of-sample rather than evaporating. Win-rate improvement on the same
  held-out games: **-0.1162pp**, 95% **[-1.2792, +1.0992]**, P+ **0.428**
  -- flat, consistent with SS3-4.

## Decision

Per AGENTS.md ("a promotion bar is not a decision bar" and "decide on
expected value"): the -1 point shade (LEAD-54's predeclared, already-served
prospective direction) has `probability_positive` 0.816 on the closest-total
AE metric full-sample, 0.843 out-of-sample on the held-out 2018-2025 seasons,
and its win-rate read against the declared public field is flat (crosses
zero, mildly unfavourable at the declared base parameters, favourable at
lower assumed public bias). No result here refutes the -1 shade or bounds
it with a positive control; nothing in this analysis is grounds to change
what is already served (`tiebreaker.json`'s existing prospective shade,
`docs/prospective_bestpick_tiebreaker.md`). The one RESOLVED result is
negative and about a DIFFERENT, larger magnitude: shading a full 4 points
low measurably loses win-rate ground against the declared public field, so
this analysis argues against ever widening the served shade to -4, while
saying nothing against the already-served -1.

## Follow-ups (named, not fabricated)

1. A full lattice-consistent (not market-margin-approximated) exact-score
   simulation, back through 2009, requires either a historical active-model
   pick side (unavailable before ~2020) or a defined proxy pick rule; this
   was named rather than approximated (Method, SS6).
2. The bucket-level under-rate (SS1) is non-monotonic in the market total;
   a bucket-conditional shade was out of scope for this lane and is a
   candidate follow-up if the closest-total reading is later confirmed as
   the pool's actual rule.
3. POL-12's underlying question -- closest total vs. closest score -- is
   still uncaptured. Nothing in this document resolves it; SS6 exists
   precisely because it cannot be assumed away.

## Files

- `scripts/tiebreaker_low_side_shading.py` -- builds the population, the
  shade comparison, the sensitivity sweep, the model-lean arm, the
  exact-score secondary check, the reliability split and the
  out-of-sample check; writes
  `artifacts/tiebreaker_low_side_shading/<UTC stamp>/`
  (`summary.json`, `last_games.parquet`, `shade_comparison_total_metric.parquet`,
  `sensitivity_sweep.parquet`, `under_rate_by_season.parquet`,
  `under_rate_by_bucket.parquet`).
- This document.
- `ROADMAP.md` LEAD-54 row (updated with these numbers; no other row
  touched).
- 14 `nfl-ats weak-signals record` entries, family
  `tiebreaker_low_side_shading`, category `market` (see the record commands
  in the session report; registry `registry/weak_signals.json`).

At the time this document was first written, `tiebreaker.json` and the served
tiebreaker code (`src/nfl_ats/tiebreaker.py`) were NOT modified -- it only
proposed a shade; nothing in it changed what was served. That changed the
same day (below).

## Wired into the served guess (2026-09-11)

**Measured** (this session, live checkout): the EV case above -- the
predeclared -1 shade at `probability_positive` 0.816 full-sample and 0.843
out-of-sample, no result above resolved against it -- is exactly the
"promotion bar is not a decision bar" situation AGENTS.md describes, so the
shade is now wired into what the pool actually reads, not left as a
retrospective-only proposal. `src/nfl_ats/tiebreaker.py` adds
`TOTAL_LOW_SIDE_SHADE_POINTS = -1.0` (source-tagged
`TOTAL_LOW_SIDE_SHADE_SOURCE = "docs/tiebreaker_low_side_shading.md"`) and
applies it to `build_report`'s `guess_total_line` immediately after the
existing market+model blend is computed -- additive, after the blend, to the
total only. The margin blend (`MODEL_RESIDUAL_WEIGHT` = 0.2) and the totals
blend that feeds the pre-shade total (`TOTALS_RESIDUAL_WEIGHT` / the
joint-residual method) are untouched; only the already-blended total is
shifted, and the score split between the two teams still preserves the
projected margin. Because `guess_total_line` is the single value the module's
neighborhood, its pick-consistent score lattice, and its fallback
median-based guess all key off (`docs/tiebreaker.md`: "one lattice, one
margin, one total"), shading it once keeps all three internally consistent
at the new, one-point-lower number rather than introducing a second,
diverging "total" for some of them. The pick-side-strictly and
lattice-consistency guard
(`nfl_ats.score_lattice.pick_consistent_top_score`, `TiebreakerConsistencyError`
on failure, publishing fails closed on a contradiction) is untouched and
still enforces both the strict pick-side inequality and the score-lattice
proximity/hard-guard checks -- it is simply handed the shaded total as its
target instead of the unshaded one.

The shade is recorded, not narrated: `TiebreakerReport` carries two new
fields, `low_side_shade_points` and `low_side_shade_source`, surfaced in
`tiebreaker.json` as `total_low_side_shade_points` /
`total_low_side_shade_source`, and as a new `total_low_side_shade` entry from
`tiebreaker_lineage_sources()` (its `unknown_source_reason` states plainly
that this is a policy constant, not a data capture, naming this document).
Reader-facing text (`format_report`, the published card's tiebreaker line)
carries only the already-shaded score and total -- no shade language was
added there.

**Measured** (rehearsal against the live checkout, read-only
`nfl_ats.tiebreaker.tiebreaker_report()`, nothing written to `tiebreaker.json`
or `CURRENT_PREDICTIONS.md`): the week's last game, DEN at KC
(2026-09-14, Monday), market total 43.5. Before this change (shade forced to
0.0 for the comparison) the served total was **43.72** and the published
score **KC 23 - DEN 20 (total 43)**; after (shade -1.0, as now wired) the
served total is **42.72** and the published score is **KC 23 - DEN 20 (total
43)** -- unchanged this specific week. The reason is disclosed rather than
hidden: at this game's margin (KC -2.5), the only admissible pick-consistent
cell near the centre has margin exactly 3 (KC 23, DEN 20), and the feasible
totals at that margin are spaced two points apart (odd sums only); 43 is the
closer of the two neighbours to both 43.72 and 42.72, so the discrete
published score does not move even though the continuous served total moved
a full point. This is a real, checked case of the shade changing the
underlying number without changing the printed score, not a sign the wiring
is inert -- the earlier fallback-branch (no active model pick) and
model-blend unit tests in `tests/test_tiebreaker.py` and
`tests/test_totals.py` do show the shade changing the printed guess when the
game's own lattice is less coarsely spaced.

Existing tests were edited to match the new served total, per the test
moratorium (no new test files or functions): `tests/test_tiebreaker.py` (six
assertions plus one comparison block rewritten against the shaded
neighborhood), `tests/test_served_total.py` and `tests/test_totals.py` (two
assertions each, importing the new constant instead of a bare literal), and
`tests/test_played_card_lineage.py` (its two exact-list lineage-source
assertions widened for the new `total_low_side_shade` entry, and its tiny
3-row `_finals()` fixture broadened to a 17-score dense set after the shade
pushed that fixture's lattice construction to exactly zero feasible mass --
an artifact of the fixture's size, not of production data, where the
equivalent rehearsal above completed without incident). Gates measured this
session: `ruff format --check .` and `ruff check .` are clean on every file
this lane touched (two pre-existing, unrelated failures remain in
`scripts/playcaller_change_screen.py` and
`scripts/sharp_book_weighted_movement.py`, neither touched here); `mypy src`
reports "Success: no issues found in 218 source files"; `pytest tests -k
tiebreak -n 2` reports 75 passed.

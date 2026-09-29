# LEAD-69 referenced metrics

Read: LEAD-69 incorporates LEAD-67 metrics. Reference copied before outcomes.

| LEAD-67 | ⬜ | Price the late move in key-number probability, not raw points (rank 2 of 8) | **Added 2026-09-29
(unmeasured).** Mechanism: the served four-term probability enters the leader-median late move as one linear
coefficient in points (MKT-18/19), but half a point across 3 or 7 moves cover probability by roughly 4 to 7
points while half a point between 4.5 and 5 moves it by about 1 to 2 (row 693 measured push rate 10.1% at line
3), so the linear term mixes informative and uninformative moves and under-weights crossings. Replace `c*move`
with `c*dP`, where dP is the change in the discrete-lattice cover probability of the frozen-line side implied by
moving the line from the frozen opener to the pre-deadline leader-median line; an additive arm keeps both.
Predeclared: 799 opener-graded games 2023-2025 (the move-available population), LOSO by season, the served
four-term base as comparator, 2 looks, metrics log loss, Brier, accuracy and line-move-toward-pick with
`probability_positive`, coefficients per fold with stability, decisive-game record first. The move remains a
fitted term in one probability; no flip rule. Checked, not duplicated: MKT-16/17/18/19 (raw points), MOD-20 unit
5 (flag-sum and availability interactions), `sunday-market-probability` lane (which cutoff, not the move's
units), LEAD-66 (grades the PMF; this row changes a served-model input). About 15 tool calls. |

## Implementation declaration, before task outcomes

- Population: the active artifact frozen at `artifacts/pick_probability/20260929T192747Z/per_game.parquet`,
  seasons 2023-2025, nonpush opener targets, move available. Require the declared starting count of 799.
  Structurally exclude Monday and Sunday kickoffs at or after 19:00 Eastern (SNF); report attrition.
- Source: `artifacts/sharp_book_weighted_movement/spread_quotes.parquet`, `intraday_hourly`, leader books,
  historical seasons only. Read predictor columns first. Compute each week's Sunday anchor from quote kickoff
  timestamps; freeze is Tuesday noon Eastern. Use `pick_refresh.pick_deadline` for min(kickoff, Sunday 16:00).
  Use the earliest kickoff supported by pre-deadline captures, a conservative cutoff when times are revised.
  Captures, snapshot timestamps, and bookmaker updates must
  be at or before the deadline; updates must not follow capture. Require positive freeze-to-deadline hours.
- Rebuild the production Wednesday-onward median leader-book move with `include_sunday=True` and an explicit
  deadline cutoff. Apply the same rebuilt move to every fitted arm. Missing eligible source coverage stops
  at inventory without reading outcomes; never substitute close lines or unverified cached moves.
- Primary terms: intercept, model_logit, composition_flag_sum, market_move_toward_home, market_move_available,
  plus `market_move_toward_home * log(hours_to_deadline)`. The added parameter is c1, natural logarithm.
- Descriptive second look: replace that interaction with move-by-slot interactions; fixed categories Thursday,
  Friday, Saturday, Sunday before 16:00 (reference), Sunday at/after 16:00, other weekdays. No category selection.
- Fit: unchanged production ridge 0.001 and Newton solver; standardize using training rows only. Three LOSO
  folds, 2023/2024/2025; also fit all eligible games for the required in-sample diagnostic. No tuning.
- Metrics: opener accuracy, log loss, Brier, and signed opener-to-close line movement toward the selected side
  (evaluation only). Home side iff the single fitted home probability is at least 0.5. Show decisive records first,
  IS/OOS gaps, per-season metrics, natural-unit coefficients per fold, and coefficient ranges/sign stability.
- Comparators: refitted four-term, unchanged discrete model-only probability, neutral opener market p=0.5.
  Calibration: five fixed equal-width probability bins, including empty cells. No thresholds are selected.
- Uncertainty: exact paired season-cluster bootstrap, all 27 ordered resamples of three seasons, percentile 95%
  intervals; probability_positive = positive fraction + half the zero fraction. Report gains with positive good:
  challenger-minus-base accuracy/movement and base-minus-challenger log loss/Brier. Only three independent seasons.
- Looks: exactly two declared research comparisons. Separately disclose every fitted arm, numerical fit, fixed
  calibration band, and reported metric cell as descriptive diagnostics; none is a new selection search.
- A zero crossing never closes this signal. Classification remains `unresolved_below_power`; no served change.
  Required registry commands will be written to the lane for serial execution by the orchestrator.
- Output: only the assigned script and `docs/lead69_*.md`; prediction-level evidence is a Markdown appendix.
  One process, numerical and Arrow threads capped at two, no network, no registry writes, no pipeline rebuild.

## Pre-outcome source correction

**Measured:** the initial inventory reproduced 799 starting games and 685 after 114 structural exclusions,
then stopped before outcome loading on 456 games with revised kickoff timestamps. Examples in
`lead69_source_diagnostics.md` show 13:00/13:01 and 20:20/20:25 revisions. Source absence was not established.
The initial implementation's unanimity requirement exceeded the row. Before any fit or score, it was replaced
with the conservative earliest supported kickoff and a repeated deadline filter. Population, terms, folds,
metrics, ridge, exclusions, uncertainty, and the two declared research looks remain fixed.

## Correctness correction after the initial fit

The saved-prediction review found 20 initially assigned slot labels that had not been updated after conservative
deadline contraction. The descriptive arm was recomputed with the same declared categories; no terms or
parameters were selected. The complete verification command was rerun; all primary horizon results remained
exactly unchanged. The final report discloses the inventory halt and both scoring passes (24 numerical fits total).

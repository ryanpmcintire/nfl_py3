# Modeling and evaluation

## Target and sign convention

The model estimates `P(home team covers)`.

```text
result      = home_score - away_score
ats_margin  = result - spread_line
home_cover  = 1 if ats_margin > 0
              0 if ats_margin < 0
           null if ats_margin = 0 (push)
```

Positive nflverse spread values favor the home team. Pushes are not binary
training examples and are not counted in classification metrics. A selected
bet that pushes returns its stake and contributes zero profit.

## Features

The initial model uses only pregame values:

- closing spread, total, home-away rest difference, neutral/divisional context,
  weather, and cyclical week;
- pregame Elo difference and implied home win probability;
- exponentially weighted offense, defense, turnover, sack, scoring, and prior
  ATS residual states for both teams and their differences.

Every accepted predictor is named in `MODEL_FEATURE_COLUMNS`. Missing early
season histories are imputed inside the fitted pipeline, so imputation learns
only from the training window.

### Temporal opponent adjustment

The optional graph workbench revives the legacy project's schedule-network
idea without its initial Week 11 leakage, off-by-one window, or conversion of
continuous centrality into ordinal ranks. A loser-to-winner graph produces a
continuous PageRank strength score. A separate defense-to-offense scoring graph
uses HITS to retain distinct offensive strength and defensive vulnerability.
Edges decay with an eight-week half-life and regress toward a symmetric prior
between seasons.

A conventional time-weighted ridge/SRS rating is built from the identical game
history. This is the required comparator: graph centrality is not credited for
schedule adjustment unless it beats or complements the simpler rating. Every
game in a week receives ratings before that week's outcomes update either
method. Graph, schedule-only, combined, and baseline feature sets are retained
separately for matched ablations.

The 2018–2025 comparison did not validate PageRank/HITS as an ATS improvement.
Adding graph features to `market_context` worsened Brier from 0.250420 to
0.250606. Season-blocked paired improvement was -0.000186 with a 95% interval
of [-0.000376, 0.000005]. In nested selection, graph candidates were chosen in
zero of eight outer seasons. The ridge/SRS candidate was chosen three times,
but its pooled Brier improvement was only 0.000013 with a season-blocked
interval of [-0.000193, 0.000209]. Neither is promoted to a default feature set.

The graph profile also failed the 2018–2025 outcome-model comparison. Relative
to the corrected base features, fair-margin cover Brier worsened from 0.25745
to 0.25970 and MAE from 10.148 to 10.158 points. Market-residual cover Brier
worsened from 0.25208 to 0.25398 and MAE from 9.905 to 9.937. Cover Brier was
worse in seven of eight fair-margin seasons and six of eight residual seasons.
This rules out default promotion for the current graph formulation.

### Opponent-adjusted play-by-play

The PBP v2 feature layer adds a conventional, interpretable schedule
adjustment rather than another graph score. At every weekly cutoff, six
team-game outcomes—EPA/play, early-down EPA, success rate, explosive rate,
pressure allowed, and sacks allowed—are each decomposed into an offensive-team
effect plus an opposing-defense effect with ridge shrinkage. Prior observations
decay with a 16-week half-life. The actual home-offense/away-defense and
away-offense/home-defense pairings yield 18 expectation and difference fields.
All games in the current week are excluded, even if one was played earlier in
the week.

The 2018–2025 direct-ATS ablation showed a small unresolved improvement over
raw full PBP: Brier fell from 0.250803 to 0.250738, while its season-blocked 95%
improvement interval was [-0.000050, 0.000199]. Accuracy remained below 50%,
and `market_context` remained substantially better at 0.250420 Brier.

The outcome-model comparison rejected promotion more clearly. Adjusted versus
raw PBP worsened fair-margin MAE from 10.175 to 10.194 and cover Brier from
0.25944 to 0.25993; it worsened market-residual MAE from 9.935 to 9.957 and
cover Brier from 0.25351 to 0.25416. Straight-up Brier worsened from 0.22262 to
0.22358. The feature contract and `pbp_adjusted` profile remain reproducible
research tools, but neither replaces the current defaults.

## Models

`logistic` is the default: median imputation, missingness indicators,
standardization, and regularized logistic regression. It is intentionally hard
to beat as a stable, interpretable probability baseline.

`hgb` is a compact histogram gradient boosting classifier for nonlinear
comparisons. It is an experiment, not the default merely because it is more
complex.

### Player availability and value

Player features are deliberately separated into quarterback state, injury
burden, lineup/roster continuity, and value-weighted injury burden. All snap
shares and weekly production are outcomes: a game updates those states only
after its prediction row is emitted. Injury rows are filtered to the latest
observation available at the 24-hour decision cutoff.

The v2 value proxy uses a span-16 exponentially weighted state. Non-QB rushing
plus receiving EPA supplies a low-dimensional offensive skill value; tackles
for loss, forced fumbles, sacks, hits, interceptions, and passes defended form
a defensive disruption proxy. Both are expressed per 100 prior snaps and
shrunk by `career_snaps / (career_snaps + 200)`, then multiplied by current
injury unavailability and the player's lagged role share. These are pragmatic
public-data proxies, not causal WAR estimates.

The first fixed 2018–2025 screen reached 52.14% for the value-extended full
player profile versus 52.05% for the original player profile and 51.08% for
base. The value increment was 0.10 percentage points with intervals crossing
zero. A two-season rolling profile selector reached 52.47% over 2020–2025
versus 50.88% for base; its blocked accuracy interval was borderline while
Brier score worsened and 2025 accuracy was 49.82%. The next player experiment
must therefore freeze regularization and calibration budgets before scoring,
and use participation-based ratings rather than hand-tuning these weights.

When enough data exists, probability calibration uses the chronological tail
of the training window. A temporary model is fit on the earlier portion, its
later out-of-time probabilities train a Platt calibrator, and the base model is
then refit on all available training games.

### Winner, fair-margin, and market-residual models

The outcome workbench deliberately fits separate questions:

- `straight_up` estimates the probability that the home team wins without
  reusing a cover label;
- `fair_margin` uses football-only inputs to estimate the home scoring margin,
  which is the model's fair home-oriented spread;
- `market_residual` predicts `actual margin - market spread`, treating the
  closing spread as a strong prior rather than trying to rediscover it;
- `direct_ats` remains the direct cover classifier for comparison.

Ridge regression is the default margin model and bounded histogram boosting is
the nonlinear comparator. A chronological 20% tail of each training window is
used to form an empirical residual distribution. Winner/cover probabilities
and 50%/80% margin intervals are derived from that distribution, whose coverage
is scored out of time. The market baseline uses the line as its mean forecast
and actual two-way prices for its no-vig cover probability.

The fair spread uses the nflverse sign convention: positive means the home team
is favored. `fair_spread - market_spread` is therefore the model's proposed
market correction.

In the first 2018–2025 comparison, histogram boosting was worse than Ridge on
winner Brier, margin MAE, cover Brier, and paper ROI for both fair-margin and
market-residual targets. Ridge remains the default on evidence and simplicity,
not because the nonlinear candidate was omitted.

## Team state and offseasons

The default team state is an exponentially weighted mean with span 8,
equivalent to an update weight of `2 / 9` for the latest game. It has a roughly
2.8-game half-life; it is not a hard eight-game window. At least three observed
games are required for the initial state.

At a season boundary, each team metric retains 67% of its difference from the
previous season's league mean. Current-season games played resets to zero for
Week 1. Week 2 therefore combines the regressed preseason state with Week 1,
rather than relying on Week 1 alone or carrying the prior season unchanged.

## Backtesting

Evaluation expands one NFL week at a time. For every test week:

1. choose completed, non-push training games strictly before the week's first
   game date;
2. fit preprocessing, classifier, and optional time-ordered calibration;
3. score that week's games once;
4. retain the maximum training date beside every prediction for auditing.

The primary research metric is chronological out-of-sample ATS classification
accuracy against a 50% coin-flip reference. Week- and season-blocked intervals,
paired baseline comparisons, and season stability determine how much confidence
to place in an observed improvement. Brier score, log loss, and expected
calibration error diagnose probability magnitudes; they do not redefine whether
the selected ATS side was correct. Margin MAE/RMSE answer the separate fair-line
question. Bet coverage, profit units, ROI, and Kelly sizing belong only to the
optional paper-betting simulation.

Backtest artifacts also contain deterministic 95% block-bootstrap intervals.
One interval resamples whole NFL weeks and another resamples whole seasons, so
games that share a schedule period are not treated as fully independent. These
intervals describe uncertainty in the historical sample; only frozen future
forecasts can provide prospective confirmation.

The optional `dependence-audit` expands each game into home- and away-team
probability errors, measures pooled and team-level lag-one correlation, and
compares the pooled result with a season-preserving permutation null. This
tests, rather than assumes, whether team-specific forecast errors persist.
Regardless of its result, week- and season-blocked intervals remain available.

### Validation and outer tests

Validation is not leakage. It is the period intentionally used to choose a
configuration. Leakage means training data or features contain information that
was unavailable at the prediction cutoff. Repeatedly examining an alleged final
test while adapting the model is a third issue: adaptive test-set reuse.

`nested-evaluate` separates these roles with rolling-origin outer folds. For an
outer test season `Y`, the frozen candidate budget is compared only on the
preceding configurable validation seasons. The winning model/feature set is
then fixed before season `Y` is scored. During `Y`, weekly refits may use results
from earlier weeks, matching deployment without allowing any `Y` result to
choose the configuration that scores `Y`.

The candidate validation table, selected configuration, outer-fold metrics,
prediction rows, and source/code hashes are retained. This is historical
out-of-time evidence; a separately frozen prospective ledger remains the
strongest confirmation of future performance.

The implementation materializes one chronological prediction stream for each
candidate across the complete validation/test range. Overlapping validation
windows then consume season slices from that stream. A candidate's prediction
for a week is unchanged by the later fold that reads it, so this removes
duplicate fits without changing any training cutoff or selection decision.

## Decision policy

Home and away probabilities are compared separately with the break-even
probability implied by their American prices. The side with the larger edge is
selected only when it clears `--min-edge`; otherwise the output is `PASS`.
The default missing price is -110 and is visible in the recommendation output.
Two-way prices are also normalized into no-vig market probabilities and their
implied hold is retained for diagnostics. Bet selection still compares the
model with the actual price-specific break-even probability, because removing
vig does not remove the cost a wager would have to overcome.

### Prediction integrity gate

Model uncertainty is expected; internally inconsistent output is not. Before a
weekly direct-ATS or outcome card can be returned or written, a separate safety
module verifies its schema and independently recomputes the wager decision from
the stored probability, home/away prices, and configured edge threshold. It
also enforces probability bounds, American-odds and NFL-spread plausibility,
no-vig/hold math, one prediction per game and method, nested margin intervals,
fair-spread/residual identities, and a training date strictly earlier than the
game date.

Prospective cards additionally require known future kickoffs, both prices, and
no populated outcomes. Missing quote timestamps are recorded as a visible
warning because kickoff safety does not prove line freshness. The audit prevents
software and data-contract mistakes; it cannot certify that a statistically
valid model forecast will be correct.

Every backtest also produces a paper-only fractional-Kelly ledger. The default
uses quarter Kelly, caps any one stake at 2% of bankroll, and caps total risk in
one NFL week at 10%. All games in a week are sized from the same starting
bankroll so Sunday outcomes cannot finance other Sunday stakes. These defaults
are risk controls for research, not evidence that the probabilities are good
enough to wager.

## Feature experiments

Named feature sets make ablations reproducible: `market`, `market_context`,
`market_elo`, `football`, `full_without_ats`, `full`, graph-only and
schedule-rating variants, plus `football_pbp` and `full_pbp`. The experiment runner
evaluates each set on identical walk-forward weeks and retains both aggregate
metrics and prediction-level rows. A feature family earns promotion only when
its calibration improvement is stable across seasons and later prospective
data.

Experiment artifacts also pair every candidate with the designated baseline on
the exact same games and bootstrap accuracy, Brier, and log-loss improvements
by whole week and whole season. ATS accuracy is the headline comparison for the
classification objective. Probability scores remain useful diagnostics and
training objectives. Positive values favor the candidate; intervals that cross
zero are reported as unresolved evidence rather than discarded results.

Model selection and edge thresholds must be chosen using older development or
validation seasons. Repeatedly tuning against final reported seasons makes them
de facto validation data and biases the apparent final-test result upward. It
does not literally place their labels in an estimator's training matrix.

Each new backtest writes JSON and Markdown model cards containing intended use,
out-of-scope uses, the evaluation period, season-level calibration history,
known limitations, and exact provenance.

## Play-by-play and quarterback promotion rules

The PBP v1 filter keeps regular-season pass/rush plays with offense, defense,
EPA, and win probability; removes kneels, spikes, aborted plays and no-plays;
and computes efficiency only inside a 5%–95% win-probability band. The default
PBP state uses the same span-8, minimum-three-game, 67% offseason regression as
the base team state. It adds 48 explicitly named pregame inputs.

In the 2018-2025 six-candidate nested comparison, a PBP candidate was selected
for four outer seasons. Matched validation Brier improvements were small for
logistic models (about 0.0002 on average) and negligible for histogram boosting.
The selected outer stream produced 50.36% ATS accuracy, 0.25084 Brier, and
-1.66% paper ROI, with both accuracy and ROI intervals spanning no edge. The
result supports testing opponent adjustment and feature compression; it does
not support promoting the current PBP family as a profitable model.

Opponent adjustment was subsequently implemented and evaluated rather than
left as a proposed explanation. Its direct-ATS probability change was small
and unresolved, while fair-margin, market-residual, and straight-up error all
worsened. The next PBP work therefore focuses on drive/possession state and
feature compression, not tuning this result against the same test seasons.

The completed drive layer adds 12 offense/defense possession states (36 model
columns): points, yards, plays, elapsed seconds, scoring rate, and turnover
rate, plus the corresponding opponent-allowed values. These are calculated
from the same versioned play filter and joined only from earlier games. In the
fixed 2018–2025 direct-ATS screen, adding them to full PBP worsened Brier from
0.250803 to 0.250891; the season-blocked improvement interval was
[-0.000481, 0.000217]. The layer remains available as `full_drive` and
`football_drive`, but is not promoted or tuned against the same evaluation
period.

Quarterback performance is derived from earlier PBP appearances with at least
five dropbacks, a span-12 EWM, 50-dropback minimum history, and 75% offseason
retention. Starter identity is accepted only from a depth-chart observation at
or before the configured decision time (default: 24 hours before kickoff) and
no more than 14 days old. Actual historical starters are outcomes and are not a
fallback. Current timestamped depth data provides too little history for a
credible model comparison, so QB columns remain prospective enrichment rather
than promoted model inputs.

## Paper sizing and simulation

Fractional Kelly can apply an absolute probability haircut before sizing. For
example, a 58% selected-side probability with a 1-point haircut sizes as 57%,
never below 50%. Historical ledgers still settle against realized outcomes.
Monte Carlo paths instead sample outcomes from the (haircut) model
probabilities and report terminal-bankroll and drawdown distributions. Those
paths describe risk conditional on the model being calibrated; they do not
validate the model.

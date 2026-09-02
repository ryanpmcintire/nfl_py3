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

> **Provenance warning (added 2026-08-22).** The artifact behind the
> PageRank/HITS numbers above does not survive: no graph or PageRank output
> exists under `artifacts/` or in any commit, and no dedicated doc holds the
> underlying per-season results — these figures are prose-only and are not
> recomputable under current window constraints (no re-run available for
> 2018–2025 without a frozen predeclaration). They are recorded as
> `graph_schedule_rating_brier` in `registry/weak_signals.json` with
> classification `unresolved_below_power` and `probability_positive` ≈ 0.028:
> a consistent directional lean across replications, not a resolved
> refutation, and not resolved evidence of either sign. See
> `docs/closure_audit.md` §3 and the open-defect section of
> `docs/revisit_list.md`. Cite this entry, not the bare numbers.

The graph profile also failed the 2018–2025 outcome-model comparison. Relative
to the corrected base features, fair-margin cover Brier worsened from 0.25745
to 0.25970 and MAE from 10.148 to 10.158 points. Market-residual cover Brier
worsened from 0.25208 to 0.25398 and MAE from 9.905 to 9.937. Cover Brier was
worse in seven of eight fair-margin seasons and six of eight residual seasons.
On the surviving prose record this rules out default promotion for the
current graph formulation; per `docs/closure_audit.md` §3 that record rests
on a consistent lean under selection and season sign counts, not on a
resolved interval, and its underlying artifact is unrecoverable.

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

Roster continuity has two point-in-time shapes. The established
`player_continuity` family compares the two latest completed snap lineups and
the two latest strictly earlier weekly rosters. The isolated
`roster_returning_snaps` family measures the prior season's offense, defense,
and special-teams snap mass carried by players on the latest safely observable
current-season roster. Weekly roster rows have no observation timestamps, so
the builder delays them one week and emits missing values in Week 1 rather
than using hindsight. The returning-snap family is registered but is absent
from every existing feature profile; see `docs/roster_continuity.md`.

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
Brier score worsened and 2025 accuracy was 49.82%. That result required the
next player experiment to freeze regularization and calibration budgets before
scoring rather than hand-tune the value weights.

The follow-up budget was frozen before scoring: `base`, original `player`,
`player_qb_continuity`, and `player_value`; Ridge alpha 1, 10, and 100; and
none, Platt, isotonic, and beta calibration. Twelve raw expanding-week streams
begin in 2016. Calibrators for a target week use only completed out-of-sample
prediction rows before that week's first kickoff, with at least 400 eligible
games. A second nested layer uses the two preceding seasons to select one of 48
configurations for each 2020–2025 outer season.

That policy reached 50.70% ATS over 1,582 outer games versus 50.88% for fixed
base/alpha-10. The -0.19-point paired change had a week-blocked 95% interval of
[-2.48, +2.21] points; Brier improvement was 0.00042 with an interval spanning
zero. It failed the promotion gate. The best pooled classification row,
QB+continuity/alpha-1/uncalibrated, reached 52.63% but worsened Brier and was
chosen only after all 48 rows were visible. Full-player/alpha-1/beta reached
52.34% with Brier 0.24965, but is subject to the same selection caveat. These
are next-test hypotheses, not replacements for the active model. New player
signal should now come from participation-based ratings rather than more
hyperparameter searching on these seasons.

The first participation specification was declared before its ATS rows were
generated. It joined nflverse participation to the canonical competitive PBP
filter, retained only valid 11-on-11 plays, clipped EPA to ±5, and fit sparse
offense-player and defense-player effects alongside scaled team effects. Each
target season used at most the preceding three seasons, Ridge alpha 1,000, and
an additional `plays / (plays + 500)` reliability shrink. A target season's
participation and EPA cannot change its own ratings; a regression canary
mutates the full target season and proves that only later ratings move.

Reported unavailability × lagged role share × the offense or defense rating
created two new matchup inputs. They were tested once against the exact parent
`player_value` profile with Ridge alpha 10 and no calibration. The extension
classified 1,073 of 2,075 non-push games correctly (51.71%), versus 1,082
(52.14%) for the parent. Its paired change was -0.43 percentage points with a
week-blocked 95% interval of [-1.53, +0.63] points. Brier score worsened by
0.00083; the season-blocked interval excluded zero in the wrong direction.
It improved accuracy in only 2022 and 2024, tied 2019, and worsened the other
five seasons. This formulation is not promoted or retuned on the same years.
The more direct next availability question is whether observed report and
practice states can estimate actual play probability better than the current
fixed status weights.

That probability experiment uses an expanding prior-season table. Report ×
practice is the base group, shrunk by 20 player-games toward the overall rate;
position-group observations shrink by 100 toward that base group. The target
is whether the player logged any offense, defense, or special-teams snap. Only
seasons with an actual snap-count partition are admitted—2013–2024—and a
coverage canary prevents a missing source season from becoming false inactive
labels. No ATS outcome selects the groups or priors.

On 57,294 out-of-season player-games, learned availability Brier was 0.09056
versus 0.09500 for the fixed weights, and binary accuracy was 87.88% versus
87.11%. Historical rates are materially more faithful: questionable/full,
questionable/limited, and questionable/DNP imply roughly 20%, 32%, and 60%
unavailability, while the old code assigned 35% to all three. Doubtful/DNP was
about 98%, versus the old 85%.

The fixed ATS replacement then changed the values of the existing QB expected
EPA/start probability, injury burden, and box-score value fields; it did not
append duplicate rate features. On matched 2018–2025 rows the candidate reached
52.24% (1,084/2,075) versus 52.14% (1,082/2,075). The +0.10-point change had a
week-blocked 95% interval of [-0.63, +0.78] points. Brier improved by 0.000063,
ECE by 0.00064, and margin MAE by 0.004 points; all are too small to support
promotion. Expected role delivery conditional on status is the next distinct
hypothesis because “logged one snap” does not distinguish a full workload from
a token or special-teams appearance.

The margin-model calibration suite is separate from its residual distribution.
Platt, isotonic, and beta calibrators consume only prior walk-forward
predictions and their completed ATS outcomes. Stored columns expose the raw
probability, calibration method, history size, and maximum history date. A
release canary requires that date to precede the target kickoff. The `none`
policy must reproduce the raw probability exactly. Calibration can improve
confidence quality and paper sizing; ATS side accuracy remains the headline
classification metric.

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

### Positive controls and statistical sensitivity

Blocked intervals quantify uncertainty; they are not a binary feature-value
oracle. A deterministic positive-control audit runs the exact active
market-residual profile and introduces independent synthetic pregame variables
with known counterfactual effects on ATS margin. The audit must first reproduce
the saved active predictions, probabilities, and classification count or fail.

The August 2026 audit passed that contract: 2,127 residual predictions matched
within `6e-14`, cover probabilities matched exactly, and non-push
classification reproduced 1,080/2,075. Eight independent replicas at each
effect size produced these averages:

| Known effect per feature SD | Candidate accuracy | Lift over matched no-signal model | Week interval clears | Season interval clears |
|---:|---:|---:|---:|---:|
| 0.0 points | 52.13% | +0.08 points | 0/8 | 0/8 |
| 0.5 points | 52.80% | +0.78 points | 3/8 | 2/8 |
| 1.0 point | 53.48% | +1.42 points | 2/8 | 4/8 |
| 2.0 points | 55.62% | +3.96 points | 7/8 | 7/8 |

Permuted controls cleared neither blocked interval at any effect size. This
shows the evaluator recovers material signal and rejects unrelated variables,
but the NFL sample often cannot resolve real 0.5–1-point mechanisms at 95%
confidence. Sparse effects applying to only a fraction of games are harder
still. Candidate decisions therefore combine point estimate, blocked
uncertainty, season direction, an independently validated transformation
target, football rationale, and multiple-testing exposure. A wide interval no
longer means "no effect"; a negative estimate with worse probability error is
still negative evidence.

College football is the highest-priority external replication and transfer
domain. CFB may estimate shared position, role-loss, and replacement mechanisms
with more games, but CFB and NFL ATS rows are never treated as exchangeable.
Final comparisons remain NFL-only: NFL-only versus naïvely pooled control,
CFB-pretrained, and hierarchical partially pooled models on identical outer
weeks.

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

The August 2026 sensitivity-aware review formalized this with a multiplicity
ledger: on the order of 130–150 candidate prediction streams have been scored
against the 2018–2025 outcomes, so the best pooled numbers there are what
selection on noise plus a possibly small real effect would produce. Untouched
pre-2018 windows were then spent on frozen, predeclared replications: the
raw-PBP market-residual bundle scored −0.08 points against base on 1,247
never-selected-on 2013–2017 games (its post-hoc 2018–2025 comparison had shown
+1.69), and the declared QB-plus-continuity alpha-1 candidate scored exactly
+0.00 points on 997 games in 2014–2017 with all probability diagnostics worse.
Both families are closed, both windows are declared spent, and new evidence for
any existing family must come from prospective 2026 outcomes or cross-league
replication rather than another 2018–2025 screen. Replication artifacts,
including predeclaration copies, live under `artifacts/pbp_replication/` and
`artifacts/qb_continuity_replication/`.

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
worsened. The 2013–2017 market-residual replication then closed the raw PBP
bundle outright (−0.08 points versus base with margin error resolved worse),
so remaining PBP value, if any, lies in compressed low-dimensional mechanisms
chosen on CFB data or in joint score/pace distributions — not in re-screening
the 48-column bundle.

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

PER-02's completed construction path now retains both named QB1 and QB2 from
that same depth observation, applies the existing fixed or season-lagged
starter-availability probability, and mixes their strictly prior EPA/dropback
and CPOE states. It exposes the named backup's state and QB2-minus-QB1
adjustment instead of substituting a generic replacement value. Uncovered
injury seasons, stale depth, and missing player histories stay null and carry
auditable source/timestamp fields. Its `depth_qb_*` namespace keeps these
depth-derived semantics distinct from `player_qb`'s prior-appearance
projection, while both reuse the same availability resolver. The new
`quarterback_depth` family is registered but absent from every model profile; see
`docs/quarterback_state_features.md` for the complete contract and build path.

## Paper sizing and simulation

Fractional Kelly can apply an absolute probability haircut before sizing. For
example, a 58% selected-side probability with a 1-point haircut sizes as 57%,
never below 50%. Historical ledgers still settle against realized outcomes.
Monte Carlo paths instead sample outcomes from the (haircut) model
probabilities and report terminal-bankroll and drawdown distributions. Those
paths describe risk conditional on the model being calibrated; they do not
validate the model.

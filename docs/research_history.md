# Research history (superseded results)

This section was moved out of README.md on 2026-09-12; every result here is superseded by the active model described in README.md.


Everything in this section is a frozen record of research that predates, or
was superseded by, the model described under "Active model and grades" above.
It is retained rather than deleted because `AGENTS.md` requires negative
results to stay visible; read it as chronological history, not as a
description of what is live now.

The primary research target throughout is **binary ATS classification**:
select the team that covers more accurately than a 50% coin flip in
chronological out-of-sample games. The point spread is required to define
that label. Betting prices, vig, ROI, and Kelly sizing are not part of this
definition; they remain an optional paper-betting analysis. Brier score and
log loss diagnose probability calibration, but poor calibration does not erase
an observed classification advantage.

The first full player-profile result was, at the time, the project's headline
lead. Across 2,075 non-push games from 2018 through 2025, the market-residual
model with the `player` feature profile selected the covering team **1,080
times (52.05%)**, 2.05 percentage points above chance. Its 95% accuracy
interval was 49.85%–54.25% when resampling whole NFL weeks and 50.19%–54.14%
when resampling whole seasons. Against the exact paired base model, accuracy
improved from 51.08% to 52.05%; that 0.96-point incremental improvement
remained unresolved under week blocking. The 0.2532 Brier score said the
confidence magnitudes were poorly calibrated, not that the 52.05% binary
classification result disappeared. This was the strongest lead as of
2026-08-17 and motivated the decomposition, ablations, and challenger research
below; `player` was superseded as the active profile by `weak_stack` on
2026-08-18 (see Active model and grades above for what is live now).

The player layer was then decomposed instead of treating that 52.05% as one
indivisible result. In a fixed 2018–2025 comparison, the coarse injury-only
profile reached 51.28%, lineup continuity reached 51.95%, QB plus continuity
reached 52.34%, and the original full bundle reached 52.05%. The original gain
therefore came mostly from continuity/QB state, not the first injury-severity
index. None of the paired week- or season-blocked intervals resolved a fixed
profile improvement over base.

A second leak-safe layer archived 291,747 weekly player-stat rows from
2009–2025 and weighted reported absences by reliability-shrunk, strictly
lagged offensive EPA and defensive disruption per snap. Adding those two value
fields to the full player profile classified 52.14% correctly, 0.10
percentage points above the original player model; its paired interval
crossed zero. A nested policy selecting among base/injury/player/value
profiles on the prior two seasons reached 52.47% over 1,582 games from
2020–2025, versus 50.88% for the fixed base profile. Its 1.58-point paired
accuracy improvement had a week-blocked interval of 0.06–3.10 points, but
Brier score was worse and the latest outer season (2025) was only 49.82%.
This was the strongest refinement lead at the time, not a promoted model. It
required a frozen nested regularization and calibration gate before any
promotion decision—not more tuning of feature definitions against the same
seasons.

That frozen gate ran to completion. The budget was declared before scoring:
four profiles (`base`, `player`, `player_qb_continuity`, `player_value`),
Ridge alpha values 1/10/100, and no/Platt/isotonic/beta calibration. Each
calibrator learned only from earlier out-of-sample weekly predictions
beginning in 2016; every outer season from 2020 through 2025 selected one of
48 configurations using only the prior two seasons. The resulting selector
classified 50.70% of 1,582 games, versus 50.88% for fixed base/alpha-10. Its
paired change was -0.19 percentage points with a week-blocked 95% interval of
-2.48 to +2.21 points. **It was not promoted.**

The pooled table still produced useful hypotheses. QB plus continuity with
alpha 1 and no calibration led classification at 52.63%; its +1.54-point
change over base had a week-blocked interval of -0.29 to +3.26 points and its
Brier score was worse. Full-player alpha 1 with beta calibration reached
52.34% and improved Brier from 0.25208 to 0.24965. Both were identified after
comparing 48 rows, so they were development leads rather than independent
confirmation. The then-active 52.05% `player` model remained unchanged.
Participation-based player ratings were the next attempt to add new signal
instead of further selecting among these same rows.

That participation experiment ran to completion and was negative. An
immutable 2016–2025 snapshot contains 478,989 source rows. For each target
season, one regularized adjusted-plus/minus fit used only the preceding three
seasons of competitive, valid 11-on-11 plays; EPA was clipped at five points,
team effects absorbed broad team strength, Ridge alpha was fixed at 1,000, and
player coefficients received a 500-play reliability prior. The resulting
offense and defense ratings added only two injury-value contrasts to the
prior player-value profile. On the exact same 2,075 non-push games, ATS
classification fell from 52.14% to **51.71%**. The -0.43-point change had a
week-blocked 95% interval of -1.53 to +0.63 points, and Brier error worsened
by 0.00083. The then-active model was unchanged. The code and result remain
available as a failed fixed hypothesis; the next availability work learned
actual play probabilities from historical report status and snaps rather than
retuning these player ratings.

That follow-up also ran to completion. Across 57,294 player-games scored
strictly with earlier-season rates, replacing the hand weights with report ×
practice rates plus a strongly shrunk position refinement improved the
availability target's Brier score from 0.09500 to 0.09056 and classification
from 87.11% to 87.88%. When those learned rates replaced the old weights
throughout the same QB, injury-burden, and player-value feature columns,
matched ATS classification moved from 52.14% to **52.24%**—1,084 rather than
1,082 correct games. Brier, ECE, margin error, and straight-up scores also
improved slightly. The ATS gain was only 0.10 points with a week-blocked
interval of -0.63 to +0.78 points, and 2025 scored 49.08%, so this was
promising but unresolved and was not promoted at the time. It later fed
`weak_stack`, the profile that is active today. A replacement live injury
source is still required because the historical nflverse injury feed ends in
2024.

The first expanding-window development benchmark is intentionally recorded
even though it is not a winning strategy. Because its 2018 through 2025
results were examined during model development, it is not described as an
untouched final test. Retraining the default logistic model before every week
produced:

- 2,075 evaluated non-push games and 50.2% side accuracy;
- Brier score 0.2508 and log loss 0.6947;
- 538 selected bets at a 2% minimum edge, returning -1.60 units (-0.3% ROI);
- quarter-Kelly paper bankroll 100 → 94.97 with an 18.1% maximum drawdown.

Week-blocked 95% intervals are 48.1%–52.2% accuracy, 0.2496–0.2521 Brier,
and -8.7%–8.4% ROI. Those intervals make the conclusion clearer: the current
result is statistically compatible with no classification advantage.

The research bar is out-of-time ATS accuracy above 50% that remains stable by
week and season. Probability quality is a secondary diagnostic and paper
returns are reported separately; neither defines the classification target.

A 2022–2025 feature-set ablation also found that the nine-feature
`market_context` model outscored the full 58-feature model on Brier score
(0.25018 vs 0.25051). This is an exploratory comparison, not a newly selected
production model; it indicates that team-form features have not shown stable
incremental value on their own.

The next outcome-model benchmark (2018–2025, 2,127 games) separated winner,
margin, and ATS questions. The independent fair-margin ridge model predicted
the winner correctly 65.2% of the time, but the closing market was better on
winner probability and absolute margin error (9.84 vs 10.15 points). A model
that predicted a correction to the market reached 67.2% winner accuracy,
versus 66.4% for the market favorite, but had worse Brier score and margin
error; its week-blocked winner-accuracy improvement interval crossed zero.
Its +2.9% paper ROI also had a 95% week-block interval of roughly -2.1% to
+8.3%, so it was not evidence of an ATS edge.

The versioned PBP experiment added 48 early-down EPA, success, explosive-play,
pressure, pass-rate/PROE, drive, and field-position states (106 total inputs).
It did not improve the outcome models: fair-margin winner accuracy fell from
65.2% to 64.7%, and direct-ATS Brier was essentially unchanged. This negative
result is retained as an experiment artifact instead of being tuned away.

A stricter six-candidate nested comparison subsequently selected a PBP variant
in four of eight outer seasons. PBP modestly improved the logistic candidates'
two-season validation Brier score in five or six folds, depending on the base
feature set, but was effectively tied for histogram boosting. The selected
outer predictions reached 50.36% ATS accuracy, 0.25084 Brier, and -1.66% paper
ROI; week-blocked 95% intervals were 48.40%-52.56% accuracy and -9.30%-6.79%
ROI. PBP was, at that stage, a weak lead for refinement, not demonstrated
edge -- it later fed `weak_stack`.

The legacy PageRank/HITS idea has also been rebuilt with strict weekly
cutoffs, continuous scores, temporal decay, and a ridge/SRS
opponent-adjustment comparator. Across 2018–2025, graph candidates were
selected in zero of eight nested outer seasons. The simpler schedule rating
was selected three times but did not produce a statistically resolved
improvement. Both remain available as research feature sets and neither is a
default.

Adding those ratings to the fair-margin and market-residual models also made
both probability and margin error worse. The graph experiment is therefore a
completed negative result, not an unfinished candidate awaiting promotion.

The next PBP iteration separated six observed offensive efficiencies into
time-decayed, ridge-shrunk offense and opposing-defense effects before each
NFL week. Its 18 matchup fields modestly improved full-PBP direct-ATS Brier
from 0.250803 to 0.250738, but the season-blocked 95% improvement interval
crossed zero (-0.000050 to 0.000199). It worsened fair-margin MAE (10.175 to
10.194), market-residual MAE (9.935 to 9.957), and straight-up Brier (0.22262
to 0.22358). Opponent adjustment is therefore implemented and available for
research, but is not a default feature family.

The drive layer is also complete: points, yards, plays, duration, scoring, and
turnover rates per possession are carried forward for both offenses and
opponent-allowed defenses. Adding its 36 fields to raw PBP raised full-model
ATS Brier from 0.250803 to 0.250891. The season-blocked improvement was
-0.000088 with a 95% interval of [-0.000481, 0.000217], so drive state remains
available as a research family and did not advance to the more expensive
outcome-model stage.

The nonlinear histogram-boosting margin challenger was also worse than Ridge:
fair-margin MAE was 10.30 vs 10.15 points, and market-residual MAE was 10.00
vs 9.91. Its residual-model paper ROI was -2.1% rather than Ridge's
exploratory +2.9%. Ridge therefore remains the simpler research default;
neither model has demonstrated an ATS edge on its own.

A nested rolling-origin evaluation chose among nine logistic/HGB and
feature-set candidates using only the preceding two seasons before scoring
each outer season from 2018 through 2025. Across 2,075 non-push outer-test
games it produced 49.7% accuracy, 0.25064 Brier error, and -3.0%
selected-bet paper ROI. Six different configurations won the eight validation
folds, so that model-selection signal was unstable at the time. This
corrected the evaluation protocol and left substantial feature/model research
open; it did not imply historical research was exhausted.

Nested evaluation computes one leak-safe chronological prediction stream per
candidate and reuses immutable season slices for overlapping validation folds.
This is equivalent to repeating the same weekly fits inside every fold, while
removing redundant computation and making larger frozen candidate budgets
practical.

An explicit team-error dependence audit found pooled lag-one correlation of
-0.018, inside the season-preserving shuffle null range (-0.031 to 0.032;
two-sided p=0.238). The residuals examined did not show detectable team-level
serial correlation, but week/season blocked intervals remain the conservative
reporting default. Independence affects uncertainty; it does not prohibit
validation and test partitions.

The [historical data-feasibility audit](docs/data_feasibility.md) gates the
research backlog. It verifies actual nonempty releases, source/schema changes,
availability semantics, and effective game-level sample size before a lead is
implemented. The historical player layer uses timestamp-filtered 2009–2024
injury reports, lagged 2013–2025 snap counts, and strictly prior-week rosters.
Its value extension uses 2009–2025 weekly player production only after each
game is complete. Player participation and NGS provide ten seasons for
restrained unit-level work. The one-season opener/close sample and one season
of precisely timestamped depth history are not sufficient for retrospective
edge claims on their own.


# MOD-06 dynamic arm: a Bayesian team-strength posterior screened on production

Predeclared 2026-09-11, before any candidate was scored. Results are appended
below the "Results" heading and nothing above it was edited afterwards.

## Which arm of MOD-06 this is

Read, `ROADMAP.md:785`. MOD-06's title is "partial pooling, uncertainty,
explicit offseason evolution". Three of its arms are already settled there:

- the **coefficient-level** arm (ridge/empirical-Bayes shrinkage of the
  regression coefficients) is closed on its own measurement, 2026-08-17, and
  the row says in terms not to reopen it;
- the **unit-level** arm (James-Stein shrinkage of a thin player's value
  toward a position prior instead of toward zero) was screened on NFL
  2026-08-19: -0.048 accuracy points, week-blocked 95% [-0.771, +0.672],
  `probability_positive` 0.4148, `unresolved_below_power`;
- the **residual-offset** arm (modelling the trailing-holdout location that
  the pick boundary sits on) ran 2026-09-07 as lanes A/D/G and ended in the
  `gaussian_median` promotion that is production today.

What has never been built is the thing the row is actually named after: a
**dynamic** team-strength state with **explicit offseason evolution** and a
**posterior standard deviation**. Every arm above is a shrinkage or a
location question about a static weekly refit. This lane builds the dynamic
posterior and screens it on production two ways.

## Expected size, stated before scoring

Read, `docs/pool_edge_plan.md:124-147`: a feature whose contribution is "we
now know how strong these teams are, more precisely" is refining the one
quantity the spread is unambiguously good at, and its achievable gain is
bounded near zero. The measured ceiling on that family is +0.0129 margin-MAE
points from a deliberate full-sample leak. Screen (a) below is squarely in
that family and is expected to be small; it is run anyway because the pool is
forced picks and the decision is expected value, not significance.

Screen (b) is **not** in that family. The posterior SD is not a better
estimate of team quality; it is a statement about how much the model does not
know, and it is scored against the reliability of production's existing picks.
A result there is a displayed-confidence input and a page-level finding even
with zero pick changes.

## The posterior (frozen construction)

State-space model, Kalman-filtered in calendar order, closed form, no sampler
(the ROADMAP row disqualifies a sampler on reproducibility grounds).

State: a 33-vector, one latent strength per team plus a league-wide
home-field advantage. Observation for game `g` in week `t`:

    result_g = theta_t[home] - theta_t[away] + hfa_t * (1 - neutral_g) + eps,
    eps ~ Normal(0, sigma^2)

`result` is the realised home margin (`data/processed/game_features.parquet`).

Evolution, week to week inside a season (partial pooling toward the league
mean, which is zero by construction):

    theta <- phi_week * theta,   Var += w_week^2 * I
    hfa   <- hfa,                Var += w_hfa^2

Evolution across the offseason (the explicit offseason term):

    theta <- phi_off * theta,    Var += w_off^2 * I

`phi_week` and `phi_off` strictly inside (0, 1) are the partial pooling: every
team is pulled toward the league mean each week and again, harder, every
offseason, by an amount the data chooses rather than a hand-set constant.

Filtering is by week batch: the state is evolved to the start of a week, every
game in that week is predicted from that state, and only then is the week's
result used to update. A game is therefore never predicted using its own week.

Eight hyperparameters (`sigma`, `w_week`, `w_off`, `w_hfa`, `phi_week`,
`phi_off`, and the two prior SDs at the 2009 cold start) are fitted by
maximising the one-step-ahead predictive log-likelihood on seasons
**2009-2019 only**, with 2009-2010 excluded from the likelihood as filter
warm-up. They are then frozen and never refitted. The 2020-2025 evaluation
window never touches the hyperparameter fit.

Two quantities come out of the filter for each game, both computed strictly
before it:

- `posterior_margin` = the posterior mean of the home margin above,
- `posterior_sd` = the posterior SD of that same linear combination, carrying
  parameter uncertainty only (the observation noise `sigma` is excluded, since
  screen (b) asks what the model does not know, not what football is).

## Screen (a): the posterior as a residual feature on production's pick

Production's served opener pick is `pick_home_at_open_probability_rule` in the
pinned archive `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`
(active model `d49194e04945a5e5`, `gaussian_median`, home-side offset served,
1,537 games, 2020-2025). The incumbent is the saved pick; nothing is refit.

The candidate adds one column, in points:

    bayes_residual = posterior_margin - tue_open_home_spread

Production's own signal is read back through its published probability:
`z = Phi^-1(home_cover_probability_at_open)`, so production's pick is exactly
`z >= 0`. The stacked candidate pick is

    z + k * bayes_residual >= 0

with a single scalar `k`, derived on seasons **2012-2019** (post-warm-up,
pre-window) as `w / s`, where `w` is the OLS slope of the realised market
residual on `bayes_residual` and `s` is the SD of the realised market residual
over the same seasons. `k` is a derived constant from outside the evaluation
window, not a tuned one. It is a deliberate upper bound on the marginal weight
because production's own signal is not partialled out of it pre-2020; the
walk-forward stack below does that partialling properly and is reported beside
it. A sensitivity sweep over multiples of `k` is reported as a disclosed
diagnostic and is never used to select the played weight.

Secondary, single-variable isolation: a walk-forward stack inside the archive.
For each week, an OLS of the realised market residual on `[s * z]` (control)
or on `[s * z, bayes_residual]` (candidate) is fit on archive games strictly
before that week, minimum 500 training games (production's own
`DEFAULT_MIN_TRAIN_GAMES`), and the pick is the sign of the fitted value. The
control and the candidate differ by exactly one column.

Grading: forced picks at the saved Tuesday opener, pushes excluded by the
existing correctness nulls. Paired accuracy points = 100 x the mean of
`candidate_correct - incumbent_correct` on identical non-push rows.
Uncertainty: `nfl_ats.clv.week_blocked_bootstrap`, 20,000 draws, seed
20260817, 95% percentile interval, `probability_positive` via the shared
`nfl_ats.evidence_conventions.probability_positive_from_draws` half-credit
convention. Within-week correlation is ZERO by owner mandate; no design
effect, no padding. Season-blocked is secondary; per-season deltas disclose
stability. A frozen-pick null (200 seeded within-week permutations of the
realised cover labels with both arms' picks held fixed) reports the null mean,
interval and the observed percentile.

Rotation: family `bayesian_team_model_on_production`, grade opener, assigned
window **[2020, 2021]** (measured, `nfl-ats rotation assign`, 2026-09-11). The
full 2020-2025 archive read is an explicitly reused descriptive look, not an
untouched confirmation window, and is disclosed as such. The assigned window's
own result is recorded separately through `nfl-ats rotation record`.

## Screen (b): the posterior SD as an uncertainty signal

No pick changes. Production's own served opener picks are split into terciles
of `posterior_sd` (cut on the evaluation sample's own tercile boundaries, a
descriptive partition, disclosed) and scored:

- accuracy of production's opener pick within each tercile, week-blocked 95%,
- the low-SD minus high-SD accuracy difference with its interval and
  `probability_positive`,
- the Brier score of production's published opener probability within each
  tercile, as a calibration read that does not depend on the pick.

`posterior_sd` is mechanically larger early in a season and after the
offseason variance injection, so week-of-season is a confound and is reported
beside the terciles rather than silently absorbed. Measured before any
accuracy was scored, on the filter run at the starting hyperparameters: the
Spearman correlation between `posterior_sd` and week-of-season on the 1,537
archive games is **-0.969**. Every team plays the same number of games, so a
homogeneous random walk makes the posterior SD almost a deterministic function
of the calendar. The raw tercile split is therefore reported as what it is --
a season-calendar split -- and a second arm is scored beside it on
`posterior_sd` residualised against week-of-season (subtract the archive's own
per-week mean SD, a non-parametric residual). That second arm is the one that
isolates team-specific uncertainty: which teams the filter is genuinely less
sure about, holding the calendar fixed. Both arms are reported; neither is
selected on its result.

## Reliability

Split-half reliability of the team posterior, two ways:

- **odd vs even seasons**: each team's mean pre-game posterior strength within
  odd seasons against the same within even seasons, Spearman across the 32
  teams. This measures persistence of team identity, not measurement error.
- **odd vs even weeks within a season**: the same Spearman computed inside
  each season and averaged, which is the cleaner measurement-error read, with
  the Spearman-Brown adjustment for a full-length split.

A trait with zero split-half reliability is an admissible closing ground. A
non-zero one is not a promotion argument by itself.

## Binding taxonomy (verbatim from the lane brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Decide on expected value: `probability_positive`
above 0.5 favours playing it; 0.90 and 95% govern only what these docs may
CLAIM, never which card is played. Grade at the OPENER. Margins are discrete
and multimodal, so no cover probability is served off a Gaussian margin: this
lane's posterior is over team strength and its only output into the pick is a
points-valued residual, never a served cover probability.

Nothing in this lane changes the served model, the card, or any published
number.

## Results

All numbers below are measured, from one run of
`.\.tools\uv.exe run --no-sync python scripts/mod06_bayesian_team_model_on_production.py`,
recorded in `artifacts/bayesian_team_model_on_production/20260911T043121Z/summary.json`
unless another artifact is named. Population checks from that file: 1,537
archive games, 0 missing posterior, 34 pushes excluded, **1,503 scored games**,
107 weeks, 2020-2025. Production's pick was reconstructed from its published
probability with **0 boundary mismatches** and **0 correctness mismatches**
against the saved columns, so the incumbent arm is production's own card, not a
replay of it.

### The fitted posterior

Hyperparameters, maximum one-step predictive likelihood on 2009-2019 with
2009-2010 as warm-up, then frozen:

| Quantity | Fitted |
| --- | --- |
| Observation SD (points of margin) | 12.393 |
| Weekly innovation SD | 0.903 |
| Offseason innovation SD | 3.121 |
| Home-field innovation SD | 0.091 |
| Weekly pooling factor | 0.99849 |
| **Offseason pooling factor** | **0.607** |
| Cold-start strength prior SD | 1.78e5 (diffuse) |
| Cold-start home-field prior SD | 2.82e3 (diffuse) |

The offseason pooling factor is the interesting one and it was fitted, not
chosen: a team carries about **61% of its strength across the offseason** and
gives up the rest to the league mean, with roughly 3.1 points of fresh
uncertainty added on top. The two cold-start priors ran away to effectively
flat, which is the filter saying it wants no opinion at all about 2009 before
it sees a game. In-season drift is slow: 0.9 points of innovation a week
against a 12.4-point observation SD.

### Reliability

- Odd vs even **seasons**, each team's mean pre-game posterior strength,
  Spearman across 32 teams: **+0.947** (p = 2.5e-16).
- Odd vs even **weeks within a season**, mean across seasons: **+0.992**,
  Spearman-Brown **+0.996**.

The trait is about as reliable as a trait in this project gets. Nothing here
can be closed on a reliability ground.

### Screen (a): the posterior as a residual feature on production's pick

The out-of-window stacking weight, derived on 2012-2019 (2,136 games): the OLS
slope of the realised market residual on the posterior's disagreement with the
line is **+0.1107 points per point**, correlation **+0.0233**, realised
residual SD 13.14 points, so **k = 0.008424**. The sign is the one the
mechanism predicts and the magnitude is tiny: a full point of disagreement with
the market buys a tenth of a point of expected residual.

Primary, full archive 2020-2025 (a reused descriptive look, see Rotation
above), 1,503 non-push games, 107 weeks:

| | Value |
| --- | --- |
| Incumbent (production) accuracy | 54.558% |
| Candidate accuracy | 53.626% |
| **Accuracy points** | **-0.931** |
| Week-blocked 95% | **[-1.978, +0.066]** |
| `probability_positive` | **0.0328** |
| Season-blocked 95% | [-1.858, +0.071] |
| Picks changed | 68 (4.52%) |
| Candidate's record on the changed picks | 39.7% |

The interval's upper edge is **+0.066**, above zero, so this is **not** a
resolved wrong sign and the arm is recorded `unresolved_below_power`. It is
also a clear lean against: `probability_positive` 0.0328 means the EV rule
points at keeping production, and the 68 changed picks lost outright
(27 of 68 correct).

Per season: 2020 **+0.91**, 2021 **+0.85**, 2022 -1.61, 2023 -2.63, 2024
-1.13, 2025 -1.50. The two positive seasons are the first two, which is also
the assigned rotation window, so the window read below is flattered by exactly
the instability it is meant to guard against; both are reported.

Assigned window [2020, 2021], 456 non-push games, 35 weeks: candidate 54.167%
vs incumbent 53.289%, **+0.877 accuracy points**, week-blocked 95%
**[-0.665, +2.428]**, `probability_positive` **0.8607**, 14 picks changed, of
which the candidate wins 9 (64.3%). Recorded through `nfl-ats rotation record`
as `unresolved`. This window disagrees in sign with the full-archive primary
and does not overturn it; the predeclared primary is the full archive.

Frozen-pick null (200 within-week permutations, both arms' picks held fixed):
null mean **-0.042**, 95% [-1.198, +1.068], observed at percentile **0.075**.
The candidate sits in the low tail of its own null, which is the same lean the
bootstrap reports, not a second independent finding.

Weight sweep (disclosed diagnostic, never used to select): the damage scales
monotonically with the weight -- 0.25k -0.27 points (16 picks changed), 0.5k
-0.47 (31), k -0.93 (68), 2k -2.33 (133), 4k -2.33 (225). There is no
multiple of this feature's weight that helps, which is a cleaner statement than
the single point estimate.

**Walk-forward stack, replacing production's decision rule** (the secondary
arm), 999 covered games from 2022 on: 51.952% vs production's 55.155%,
**-3.203 accuracy points**, week-blocked 95% **[-5.419, -0.994]**,
`probability_positive` **0.00275**. The whole interval sits below zero, so this
one **is** a resolved wrong sign and is recorded `refuted_mechanism` with
closing ground `wrong_sign_resolved`. What is refuted is the *machinery*:
replacing production's fitted probability mapping with a weekly two-column OLS
on the same signal costs about 2.8 points on its own.

**Single-variable isolation inside that stack** -- control (production's signal
only) 52.352% vs candidate (production's signal plus the posterior residual)
51.952%: **-0.400 accuracy points**, week-blocked 95% **[-1.529, +0.703]**,
`probability_positive` **0.2419**, 38 picks changed.
`unresolved_below_power`. Properly partialling production's own signal out
shrinks the feature's cost from -0.93 to -0.40, which is the expected direction
and confirms the primary's `k` is the upper bound it was declared to be.

**Leak diagnostic.** Team strengths fitted once over the whole 2020-2025
evaluation sample -- perfect foreknowledge of how good every team turned out to
be -- stacked on production at the same `k`: **-0.532 accuracy points**,
week-blocked 95% [-1.894, +0.802], `probability_positive` 0.2182, 106 picks
changed. Foreknowledge of team quality does not beat the opening line either.
This is the `docs/pool_edge_plan.md` ceiling reproduced in-house on NFL
openers. It is reported as a diagnostic and **not** used as a
`positive_control_bound` closing ground: a control that detects nothing proves
nothing about the instrument's sensitivity.

### Screen (b): the posterior SD as an uncertainty signal

The raw posterior SD is a calendar variable on this sample (Spearman with
week-of-season **-0.910**), so the raw tercile split is a season-calendar
split and is reported as one. Production's own served opener picks, by tercile:

| Tercile | Mean week | Games | Production accuracy | 95% | Brier |
| --- | --- | --- | --- | --- | --- |
| High SD (early season) | 3.6 | 501 | **55.69%** | [50.91, 60.36] | 0.2487 |
| Mid | 9.8 | 501 | 56.89% | [52.41, 61.35] | 0.2494 |
| Low SD (late season) | 14.7 | 501 | **51.10%** | [46.96, 55.20] | 0.2567 |

High minus low: **+4.591 accuracy points**, week-blocked 95%
**[-1.628, +10.758]**, `probability_positive` **0.9245**; season-blocked
[-0.034, +8.400], `probability_positive` 0.9746. The Brier gap runs the same
way: **+0.00798** in favour of the high-SD tercile, week-blocked
[-0.00136, +0.01714], `probability_positive` **0.9548**, season-blocked
[+0.00305, +0.01241], `probability_positive` 0.9994. Production's stated
confidence is flat across all three terciles (mean stated probability
0.5566 / 0.5581 / 0.5580) while its realised accuracy moves 4.6 points, which
is a displayed-confidence defect, not a pick defect.

Stated in football words, from the same artifact: production's opener picks go
**56.01% in weeks 1-9 (766 games)** and **53.05% in weeks 10 and later (737
games)**, and early beats late in five of the six seasons (2023 is the
exception, 55.97% early against 56.82% late).

**Team-specific uncertainty**, the posterior SD with the week-of-season mean
subtracted (Spearman with week now +0.021): low 55.89%, mid 52.50%, high
55.29%; low minus high **+0.599 accuracy points**, week-blocked 95%
**[-5.343, +6.677]**, `probability_positive` **0.5673**. The ordering is
non-monotone -- the middle tercile is the worst -- so there is no usable
ordering here. `unresolved_below_power`; the mechanism is not refuted, it is
simply not visible at this resolution.

### What this implies for the decision

- **Do not stack the posterior residual on the played card.**
  `probability_positive` 0.0328 on the predeclared primary is an EV call
  against it, the changed picks lost 27-41, and every multiple of the weight is
  worse than the last. The served model is unchanged.
- **The posterior SD tercile ordering is worth showing a reader.** A 4.6-point
  accuracy spread inside production's own picks, against a stated confidence
  that does not move at all, is exactly the Model page's weak-spots material
  (`probability_positive` 0.9245 on accuracy, 0.9548 on Brier). It is a
  calendar effect -- early season better than late -- and must be labelled as
  one, not as a team-uncertainty effect.
- **Nothing here closes MOD-06's dynamic arm as a mechanism.** The posterior is
  highly reliable (+0.947 across odd and even seasons) and its disagreement
  with the line carries the predicted sign out of window (+0.111 points per
  point). What is closed is the walk-forward-OLS *replacement* of production's
  decision rule, on a resolved wrong sign.
- The one thing the lane produces that the project did not have is the
  **offseason pooling factor, 0.607, fitted rather than assumed** -- a measured
  answer to "how much of a team carries over" that any future carry-over or
  regression-to-mean feature can use instead of a hand-set constant.

### Registry entries

Eight entries, all `--league nfl --family bayesian_team_model --category
modeling`, all recorded 2026-09-11 against the artifact above:

| Name | Classification | Effect | `probability_positive` |
| --- | --- | --- | --- |
| `bayesian_team_posterior_residual_on_production_opener` | `unresolved_below_power` | -0.931 accuracy points | 0.0328 |
| `bayesian_team_posterior_residual_on_production_opener_2020_2021` | `unresolved_below_power` | +0.877 | 0.8607 |
| `bayesian_team_posterior_walk_forward_stack_replaces_production_opener` | `refuted_mechanism` (`wrong_sign_resolved`) | -3.203 | 0.00275 |
| `bayesian_team_posterior_column_in_walk_forward_stack_opener` | `unresolved_below_power` | -0.400 | 0.2419 |
| `bayesian_team_leaked_strength_on_production_opener` | `unresolved_below_power` | -0.532 | 0.2182 |
| `production_opener_accuracy_by_bayesian_posterior_sd_tercile` | `unresolved_below_power` | +4.591 | 0.9245 |
| `production_opener_brier_by_bayesian_posterior_sd_tercile` | `unresolved_below_power` | +0.00798 Brier | 0.9548 |
| `production_opener_accuracy_by_team_specific_posterior_sd` | `unresolved_below_power` | +0.599 | 0.5673 |

Rotation: `nfl-ats rotation record --name bayesian_team_model_on_production
--verdict unresolved --probability-positive 0.8607`, window [2020, 2021] spent
2026-09-11. `nfl-ats rotation validate` reports 455 families, 0 errors.

All eight carry the posterior's split-half reliability +0.947 (odd vs even
seasons) in the `--reliability` field. The entries are heavily correlated --
they are the same posterior over overlapping games -- and are not independent
votes.

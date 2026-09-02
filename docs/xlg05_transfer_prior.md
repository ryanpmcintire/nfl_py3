# XLG-05: hierarchical CFB→NFL transfer, four arms on one shared feature space

Written **before any NFL ATS outcome number is produced by this comparison**,
per the same rule that governs `docs/graph_team_stat_def_ypp_on_production.md`
and `docs/graph_team_stat_on_production.md`. **Sections 1-6 are the
predeclaration** and contain no accuracy, cover-rate, or `probability_positive`
number against NFL outcomes. **Section 7 was added after the look** and reports
what it found; it changes nothing above it.

Work package WP25, ROADMAP row XLG-05 ("Hierarchical CFB→NFL transfer: compare
matched NFL-only, naively pooled control, CFB-pretrained, and partially pooled
models on NFL outcomes").

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". A promotion threshold governs only what the docs may
CLAIM; it never governs which card is PLAYED, which is expected value.

## 1. What this asks, and why it is a MODEL change rather than a team-quality feature

The project's standing bound, recorded in `docs/pool_edge_plan.md` and in the
owner's own memory note ("team quality is already priced"), says that features
which only *estimate team quality better* are bounded near zero — the
opponent-adjustment leak probe put perfect foreknowledge of team quality at
about +0.013 margin-MAE points, and `docs/scaling_and_transfer.md` independently
put the data needed to close that gap at ~35,000 games, about 2.8x the entire
CFB corpus. Any XLG-05 design that answers "does CFB give us a better team-quality
estimate" is answering a question already bounded near zero.

**This design deliberately answers a different question.** All four arms are fit
on the **identical feature subset S** (section 2). No arm sees a column another
arm does not see; no arm computes a team-state estimate differently; the feature
table is byte-identical across arms. The only thing that varies between arms is
**where the ridge estimator shrinks its coefficients toward** — zero (the
project's standing convention), the pooled two-league fit, or the CFB corpus's
own fitted coefficient vector. That is a change to the ESTIMATOR — a prior on the
residual model's coefficients — not a change to what is measured about the teams.

That distinction is enforced by construction, not by argument: because the design
matrix is the same in every arm, a difference between arms cannot be a
team-quality-measurement difference. It can only be a difference in how the
~500-4,000 NFL training rows available at each week are regularised.

The mechanism being tested is the one `docs/scaling_and_transfer.md` §"Part 2"
built and validated CFB-internally (Power-Five auxiliary → Group-of-Five target,
free under rotation rule 8) and then explicitly predeclared for a future NFL
session it was not permitted to run. This document is that NFL session. It
departs from that earlier predeclaration in one respect, disclosed here rather
than glossed: the earlier document proposed the family name
`cross_league_hierarchical_transfer` with the DerSimonian-Laird empirical-Bayes
`hierarchical` arm as the headline. WP25's arm set is different — its partially
pooled arm selects a *prior strength* by leave-one-season-out rather than deriving
per-coefficient empirical-Bayes weights — so it is declared under its own name,
`xlg05_transfer_prior`, and does not claim that earlier document's window.

Section 7 additionally reports the explicit team-quality bound check this design
makes possible (section 3, arm `prior_market_only`, and section 6's coefficient
decomposition): how much of any measured gain survives when the CFB prior is
forbidden from touching the team-quality coefficients at all.

## 2. The shared feature subset S

S is the **measured** intersection of the two leagues' frozen feature contracts —
columns carrying the same name and the same football semantics in both
`nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns) and the NFL
production feature table `data/processed/game_features_weak_stack.parquet`. The
intersection was measured this session, not assumed, and comes to **14 columns**,
matching the already-frozen `nfl_ats.cross_league_transfer.
ALIGNED_TRANSFER_FEATURE_COLUMNS` contract (whose own subset property is a
release-blocking test,
`tests/test_cross_league_transfer.py::test_aligned_columns_are_a_true_subset_of_both_leagues_contracts`):

| block | columns |
|---|---|
| market | `spread_line`, `total_line` |
| context | `rest_diff`, `neutral_site`, `week_sin`, `week_cos` |
| experience | `home_team_games`, `away_team_games` |
| team quality (Q) | `home_off_epa_per_play`, `away_off_epa_per_play`, `diff_off_epa_per_play` |
| team quality (Q) | `home_def_epa_per_play`, `away_def_epa_per_play`, `diff_def_epa_per_play` |

The six-column **team-quality block Q** is named here, before the look, because
section 6's bound check partitions on it.

The 21 CFB contract columns outside S (success rate, explosive rate, pace, the
power-conference flags, `conference_game`) have no NFL counterpart at this
contract level; the NFL production profile's ~270 columns (pass/rush-split EPA,
CPOE, Elo, injury value, weather, graph ratings, …) have no CFB counterpart. S
is the only space in which two leagues' coefficient vectors are even
comparable, let alone blendable.

**Disclosed semantic mismatches within S.** Two columns share a name and a
concept but not a scale, and this is stated up front rather than discovered
later: `week_sin`/`week_cos` are built on a 16-week period in CFB
(`CFB_WEEK_PERIOD = 16.0`) against the NFL's own season length, and
`home_team_games`/`away_team_games` count games within a shorter CFB season. The
pooled preprocessing pipeline (section 3) standardises both leagues on one
imputer and one scaler fit on the union of their rows, which puts them on a
common scale but cannot make a "week 12 of 16" identical to a "week 12 of 17".
A transfer arm that gains only on those two blocks would be suspect; section 7
reports the coefficient decomposition that would reveal it.

## 3. The four arms, the reference line, and the bound-check diagnostic

All four arms share: feature subset S; `ridge_alpha = 10.0` verbatim from the
active model (`artifacts/active_ats_model.json`; **read**: `ridge_alpha: 10.0`,
`regressor: "ridge"`, `method: "market_residual"`) — **nothing new is derived or
tuned**; target `market_residual` (`ats_margin`); one pooled preprocessor (a
single median imputer plus a single `StandardScaler`, fit on the union of both
leagues' rows drawn strictly before the test window's first game, so no scaling
moment is computed from a scored week); and an augmented design whose trailing
constant column carries the intercept, so an intercept is shrunk by exactly the
same machinery as a slope. Each arm's fitted coefficient vector is wrapped in
the existing `nfl_ats.margin.MarginModel` so cover probabilities and the 80/20
out-of-time residual-distribution recipe are the project's tested ones, not a
fourth implementation.

| arm | estimator | prior mean for the NFL coefficients |
|---|---|---|
| **(a) `nfl_only`** — the BASELINE | plain ridge, alpha 10, NFL rows only | zero (the project's standing convention) |
| **(b) `naive_pooled`** — the control | one ridge on NFL + CFB rows stacked, plus a binary league indicator; scored with the indicator fixed at the NFL value | n/a — one coefficient vector for both leagues, only an additive league offset is league-specific |
| **(c) `cfb_prior`** | ridge on `y − X·θ_cfb`, then `θ_cfb` added back | `θ_cfb`, the CFB-only ridge fit on S at the same alpha |
| **(d) `partial_pooled`** | as (c) but the prior mean is `κ·θ_cfb`, `κ` chosen by leave-one-season-out on TRAINING seasons only | `κ·θ_cfb` |

**Why (a) and not full production is the baseline.** The point of this
experiment is the ESTIMATOR. A comparison of a transfer arm on S against the
~270-column production `weak_stack` chain would confound two entirely different
questions — "does borrowing CFB's coefficients help?" and "do we already have
richer features than S?" — and would answer neither. Arm (a) is production's
exact recipe (ridge, alpha 10, `market_residual`, same imputer/scaler discipline)
restricted to S, so arms (b)/(c)/(d) minus (a) isolates the estimator change and
nothing else. This is the same reasoning the earlier `docs/scaling_and_transfer.md`
predeclaration froze ("the `target_only` baseline must be fit on the identical 14
columns … comparing a transfer arm to the full production model would confound
the two questions").

**Reference line (reported, not a comparison arm).** Full production `weak_stack`
— `nfl_ats.margin.fit_margin_model(feature_profile="weak_stack", target=
"market_residual", model_name="ridge", ridge_alpha=10.0)` — is walked forward on
**the same games, the same weeks, the same training cutoffs**, and its accuracy
level is reported in section 7 beside the four arms. This exists so the reader
can see what the pool actually plays next to a 14-column research space, and so
nobody mistakes arm (a) for the production card. It is a level, not a paired
comparison, and it is not recorded to the registry by this document.

**Bound-check diagnostic arm `prior_market_only`.** Arm (d), refit with the prior
mean forced to **zero on the six Q coefficients** and left at `κ·θ_cfb` on the
other nine augmented components (market, context, experience, intercept). CFB is
thereby forbidden from informing any team-quality coefficient. Its paired delta
vs (a) is reported in section 7 alongside (d)'s. **Predeclared reading**: if
`prior_market_only` ≈ `partial_pooled`, whatever the transfer buys is NOT a
better team-quality estimate and the owner's "team quality is already priced"
bound does not apply to it; if the gain lives entirely in Q, the bound does
apply and should be stated plainly. This arm is a **diagnostic only** — it is a
correlated decomposition of the same window, and AGENTS.md explicitly warns that
such decompositions overstate precision when pooled, so it gets **no**
`weak-signals record` entry of its own.

### Arm (d) in detail: the prior-strength grid and its LOSO selection

Arm (d) minimises `‖y − Xθ‖² + α‖θ − κ·θ_cfb‖²` with `α = 10.0` fixed (nothing
new is derived; the *alpha* is the project's frozen constant and stays frozen).
Only `κ ∈ [0, 1]`, the **prior strength**, is selected. This parameterisation is
chosen because it makes the family a genuine interpolation containing both
neighbouring arms exactly: **`κ = 0` reproduces arm (a) coefficient-for-coefficient**
(plain ridge, prior mean zero) and **`κ = 1` reproduces arm (c)**. Arm (d) can
therefore never be a different mechanism from its neighbours — only a different
point on the same line.

`θ(κ)` is **exactly linear in κ**: the normal equations give
`(XᵀX + αI)θ = Xᵀy + ακθ_cfb`, so `θ(κ) = θ(0) + κ·(θ(1) − θ(0))`. Two fits per
training set therefore price the whole grid exactly, with no approximation.

**Grid (frozen):** `κ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`.

**Selection (frozen):** leave-one-season-out over the seasons present in that
week's TRAINING frame only. For each held-out training season `s`: fit on
(training frame minus season `s`), predict season `s`'s market residual, and
accumulate squared error. The selected `κ` minimises the pooled out-of-fold
**mean squared error of the predicted market residual**; ties break toward the
**smallest** `κ` (i.e. toward arm (a), the status quo).

MSE and not accuracy is the declared selection criterion, and the reason is
stated before the result rather than after: an NFL season is ~256 games, so a
held-out-season forced-pick accuracy carries roughly ±3 points of noise, and
selecting a 5-point grid on it would mostly select noise. MSE is the loss the
ridge itself minimises, so the selection is internally consistent with the
estimator. The selected `κ` per week is written to the artifact so the choice is
auditable rather than asserted.

**The test week is never in any fold.** Folds are drawn only from the training
frame, which is by construction every completed NFL game strictly before the
scored week's earliest kickoff. `θ_cfb` is likewise fit only on CFB games
strictly before the same cutoff, and is held fixed across that week's folds
(CFB games already played are legitimately available at prediction time; the
fold structure is an NFL-side device and does not restrict them). Both
properties are release-blocking regression tests
(`tests/test_xlg05_transfer.py`).

**Fallback (frozen):** if a week's training frame contains fewer than 2 seasons,
or any fold would leave fewer than `MIN_FITTABLE_TRAIN_GAMES` (50) rows, LOSO
cannot run and arm (d) falls back to `κ = 1.0` — arm (c)'s setting, so that a
week without selection collapses (d) onto the fixed-prior arm rather than onto
the baseline. The count of fallback weeks is reported in section 7.

## 4. The comparison and the metric

**Primary quantity:** the **paired candidate-minus-baseline** forced-pick
accuracy delta of each of (b), (c), (d) against (a), in `accuracy_points`
(percentage points), `pick_correct` against `home_cover_probability >= 0.5` —
the same probability rule production plays — via `nfl_ats.clv.pick_correct`.

**Walk-forward:** weekly, forward-chaining. Each scored week's models (all
arms, plus the production reference line) are fit strictly on games that kicked
off before that week's earliest kickoff, for both leagues. `min_train_games` =
`nfl_ats.constants.MIN_FITTABLE_TRAIN_GAMES` = **50**, production's own
constant, applied to the NFL training frame and to the CFB auxiliary frame
alike. Weeks that cannot clear the floor are skipped, not imputed.

## 5. Grade, window, and family

**Grade: close.** This is a screen of an estimator change, not a play/no-play
decision. `docs/graph_team_stat_on_production.md` §5 states the project's reason
a screen is close-graded, and the binding "grade the decision at the opener" rule
means nothing measured here may settle a promotion or a card change: a
close-graded look settles no play decision **regardless of sign**, and every arm
is therefore recorded `unresolved_below_power` at this grade (section
"Recording").

**Family: `xlg05_transfer_prior`, declared with NO inheritance.** This is a new
mechanism hypothesis — a prior on the residual model's coefficients — not a
variant of any feature family already in the ledger, so it inherits no spent
windows. `--acknowledge-mined` is **not** passed, so the assignment algorithm
must hand it a block outside the mined 2018-2025 seasons.

**Window: whatever `nfl-ats rotation assign --name xlg05_transfer_prior`
returns.** It is not hand-picked here and it is not predicted here. The CLI
computes the earliest eligible contiguous block for a fresh close-graded family
at declaration time (`DEFAULT_WINDOW_SIZE["close"] = 3` seasons); section 7
records what it actually assigned, quoting the CLI. The driver script reads the
assigned window back out of `registry/rotation_registry.json` by family name
rather than taking a hardcoded default, so a hand-picked window is not
expressible.

The family name is registered **only** in the CLI declare call and in this
document. It is deliberately not added to `src/nfl_ats/constants.py`.

## 6. Uncertainty and instrument checks

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`: **week-blocked is the
primary** reference (within-week game correlation is zero by owner mandate),
**season-blocked is a secondary read** reported beside it and never averaged with
it. A 3-season window gives the season-blocked bootstrap only three blocks, so
its interval carries very little combinatorial diversity — the same caution
`docs/graph_team_stat_def_ypp_on_production.md` §7 applies to its own
season-blocked secondary. `BOOTSTRAP_SAMPLES = 1000`, `SEED = 20260826`, the same
constants `scripts/graph_input_screen.py` already uses, for comparability with
the on-production screens.

**Within-week permutation null**, 200 permutations. Both arms' models are fit
ONCE per week on the REAL `ats_margin`; only the grading margin is shuffled
within week, so 200 draws cost no extra model fits. This null is **not** centred
on zero by design — it preserves each week's realized home-cover rate, and the
arms may carry different home-pick rates — and is reported ALONGSIDE the
bootstrap-vs-zero interval, never instead of it. `--mode null` runs it as a
standalone harness check before the window is spent.

**Positive control** (`--mode positive-control`), run BEFORE the real screen:
in arm (a) only, the S column `diff_off_epa_per_play` is REPLACED by the realized
`ats_margin` — a deliberate, large leak — and arm (a)'s pooled preprocessor is
refit on the leaked frame so the standardisation stays honest. The instrument
must then show an obvious, large effect. Two properties are stated before it
runs so neither can be spun afterwards: (i) the effect's **sign is negative by
construction**, because the leak is in the REFERENCE arm and the metric is
candidate-minus-baseline, so every candidate arm should read a large negative
delta with `probability_positive` near 0; (ii) what this proves is
**magnitude-detection**, not direction — it establishes that a real
between-arm difference of meaningful size is visible to this paired evaluator on
this window, so that a subsequent "no effect" reading cannot be dismissed as a
blind instrument. `diff_off_epa_per_play` is the leaked column specifically
because it is a Q (team-quality) column, so the control doubles as a
demonstration that this harness CAN see a team-quality effect when a real one is
present.

**Prior split-half stability** (CFB-only, free under rotation rule 8, spends no
NFL window). AGENTS.md makes split-half reliability the decisive field for
adjudicating a signal later, but this family has no *trait* to split — the arms
differ in the estimator, not in a feature column, so no NFL-side trait
reliability exists and **`no_split_half_reliability` can therefore never be an
admissible closing ground for this family**. The honest analogue is the
stability of the transferred object itself: `θ_cfb` is fit twice on the CFB
corpus, once on odd-numbered seasons and once on even-numbered seasons, on one
shared pooled preprocessor, and the Pearson correlation between the two
coefficient vectors is reported with its Spearman-Brown full-length correction.
That corrected value is what is recorded in `--reliability`, labelled in this
document and in the registry notes as **prior-vector stability, not trait
reliability**. Per `docs/scaling_and_transfer.md`'s own caveat, several S columns
are exactly collinear by construction (`diff = home − away`), so the WHOLE-VECTOR
agreement is the robust read and individual component signs are not.

### 6a. Amendment: tie-aware `probability_positive` (added between the null check and the screen)

Disclosed as an amendment rather than folded silently into section 6, because it
was written after an instrument run rather than before it. The `--mode null`
instrument check (which fits the arms and reports a permutation distribution)
revealed a degenerate case sections 1-6 did not anticipate: at the frozen
`ridge_alpha = 10.0`, arms (c) and (d) move `home_cover_probability` by so
little that they can make **the same picks as arm (a) on every game**. Two arms
that make identical picks produce a paired delta of exactly zero in every
bootstrap resample, and `nfl_ats.clv.week_blocked_bootstrap` computes
`probability_positive` as `mean(draws > 0)` — so it reports a dead heat as
**0.000**, which is indistinguishable from a certain loss and is materially
wrong for an EV decision, where a tie is neither better nor worse.

Three reporting additions, none of which changes any arm, fit, window, metric,
or grade:

1. `probability_positive` continues to be reported exactly as the existing tool
   computes it, unchanged.
2. Beside it, the **tie share** (fraction of resamples where the delta is
   exactly zero) and a **tie-aware** `probability_positive` =
   `P(better) + 0.5 * P(tie)`, the standard sign-test treatment of ties. An
   exactly tied arm therefore reads 0.500, the EV-neutral value, instead of
   0.000. Both numbers are reported in section 7 and the tie-aware one is what
   the registry entries carry, with the raw value in the notes.
3. Deterministic sign-test counts that need no resampling at all: how many of
   the window's picks the candidate arm made differently from the baseline, and
   of those, how many it got right and wrong. For near-identical arms these
   counts are the honest read and the bootstrap interval is nearly a point mass.

The permutation null gains the analogous field (`fraction_of_null_tied_with_
observed`), because a percentile is meaningless against a point mass and a
"0.0th percentile" would otherwise read as an extreme tail when it is a tie.

**Coefficient decomposition** (reported in section 7, no registry entry). The
fitted coefficient shift `θ_d − θ_a`, averaged over scored weeks, split into its
Q block and its non-Q block by L2 norm, so section 7 can say where the estimator
change actually moved the model rather than inferring it.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The decision
is expected value — `probability_positive` above 0.5 favours the candidate over
the baseline — never a 0.90 or 95% threshold; predeclared thresholds govern only
what a doc may CLAIM. This run is close-graded, so it settles no play/no-play
decision by itself. What it DOES settle is whether a CFB-anchored prior on the
NFL residual model's coefficients looks worth an eventual opener-graded
confirmation, and whether any gain it shows is or is not the already-bounded
team-quality gain.

## Recording

**Three `nfl-ats weak-signals record` entries**, `--league nfl`,
`--effect-units accuracy_points`, `--family xlg05_transfer_prior`,
`--category modeling`, one per arm compared against arm (a):
`xlg05_transfer_naive_pooled`, `xlg05_transfer_cfb_prior`,
`xlg05_transfer_partial_pooled`. Every one is `unresolved_below_power` **at this
close grade regardless of sign**, mirroring
`scripts/graph_team_stat_record.py::classify` — a resolved wrong sign at this
grade is reported as continuous evidence, never as a `refuted_mechanism`
closure, because that closure is reserved for the opener grade. The three
entries share one window and one baseline arm and are therefore correlated, not
three independent votes; that is what `--family` is for and it is stated here so
a later pooling run can see it.

**One `nfl-ats rotation record --name xlg05_transfer_prior --verdict unresolved`**
call spends the assigned window, carrying **arm (d)'s** paired effect, interval,
and `probability_positive`. That choice is declared here, before the numbers
exist: the rotation ledger records one verdict per window, and arm (d) is the
family's headline mechanism — it is the only arm that adapts its prior strength
to the data, it strictly contains arms (a) and (c) as the endpoints of its own
grid, and it is the arm whose result would drive any follow-up. Arms (b) and (c)
are recorded as weak signals but do not each spend a window.

## 7. Results (added after the look, 2026-09-01)

Rotation family `xlg05_transfer_prior` was declared close-graded with no
inheritance and no `--acknowledge-mined`, then assigned by
`nfl-ats rotation assign` (never hand-picked). The CLI returned the earliest
eligible block: **`"seasons": [2011, 2013]`, `"state": "assigned"`,
`"assigned_at": "2026-09-01"`** — quoted from the assign command's own JSON, and
read back out of the ledger by the driver script rather than hardcoded. 51
weeks were fitted, 765 games scored, **746** paired non-push games graded (19
pushes drop out of `pick_correct`), 3 seasons. NFL training grew 512 → 1,264
games across the walk; the CFB auxiliary pool grew 2,311 → 4,198 games.

Artifacts: null `artifacts/xlg05_transfer_prior/20260901T194740Z/results.json`,
positive control `.../20260901T195115Z/results.json`, screen
`.../20260901T195150Z/results.json`.

### Instrument checks

**Null check** (`--mode null`, 200 within-week permutations, no leak). Sane,
finite, and — exactly as section 6 warned — not centred on zero: `naive_pooled`
nulls at **−1.219** accuracy points (sd 1.620, 95% [−4.427, +1.749]), which is
the home-tilt artifact doing its job, since that arm picks home 55.6% of the
time against the baseline's 51.8%. `cfb_prior` nulls at **+0.076** (sd 0.347).
The two arms that make identical picks to the baseline null at exactly 0.000
with a tie share of 1.000 — a point mass, not a distribution, and the reason
section 6a exists.

**Positive control** (`--mode positive-control`, the BASELINE arm's
`diff_off_epa_per_play` replaced by the realized `ats_margin`). The leaked
baseline scores **100.00%**, and every candidate arm reads a paired delta of
**−49.062 / −48.928 / −49.062** accuracy points (`naive_pooled` / `cfb_prior` /
`partial_pooled`), week-blocked 95% CIs [−52.32, −45.64], [−52.36, −45.82],
[−52.52, −45.97], `probability_positive` 0.000, with **365-366 of 746 picks
differing and every one of them going the leaked baseline's way**. Negative by
construction and enormous, precisely as predeclared. The paired evaluator on
this 746-game window is demonstrably not blind to a real between-arm
difference.

### The screen

Accuracy levels on the 746 paired games:

| arm | accuracy | home-pick rate |
|---|---|---|
| **(a) `nfl_only`** (baseline) | **50.938%** | 0.518 |
| (b) `naive_pooled` | 50.938% | 0.556 |
| (c) `cfb_prior` | 51.072% | 0.512 |
| (d) `partial_pooled` | 50.938% | 0.518 |
| `prior_market_only` (diagnostic) | 50.938% | 0.518 |
| `production` `weak_stack` (reference line) | 49.866% | 0.536 |

Paired candidate-minus-baseline, accuracy points, week-blocked primary:

| arm | delta | week 95% CI | P+ raw / tie-aware | tie share | picks differing (better/worse) | null pctile |
|---|---|---|---|---|---|---|
| (b) `naive_pooled` | **+0.000** | [−2.957, +3.200] | 0.509 / **0.522** | 0.027 | 220 (110 / 110) | 74.5th |
| (c) `cfb_prior` | **+0.134** | [−0.669, +0.926] | 0.596 / **0.652** | 0.113 | 7 (4 / 3) | 41.0th |
| (d) `partial_pooled` | **+0.000** | [0.000, 0.000] | 0.000 / **0.500** | 1.000 | 0 (0 / 0) | n/a (point mass) |
| `prior_market_only` | +0.000 | [0.000, 0.000] | 0.000 / 0.500 | 1.000 | 0 (0 / 0) | n/a (point mass) |

Season-blocked secondary, reported beside the week-blocked primary and never
averaged with it (3 blocks only, so very little combinatorial diversity —
the same caution `docs/graph_team_stat_def_ypp_on_production.md` §7 gives its
own): (b) [−1.600, +1.195], P+ 0.422 raw / 0.532 tie-aware; (c) [0.000,
+0.408], P+ 0.720 raw / **0.860** tie-aware; (d) [0.000, 0.000], 0.500
tie-aware.

**Selected prior strength.** `kappa` = {**0.00 in 34 weeks**, 0.25 in 1, 0.50 in
1, 0.75 in 1, **1.00 in 14**}; **zero** LOSO fallbacks (every week had 2-5
usable training seasons, so the frozen `kappa = 1.0` fallback never fired).
The selection is genuinely exercised — it is not pinned at one end of the grid.

**Where the prior actually moved the model.** The fitted coefficient shift
`theta_d − theta_a` is non-zero in **17 of 51** weeks (the weeks where LOSO
chose `kappa > 0`). Among those, its L2 norm is **3.107 in the six team-quality
(Q) components against 0.620 in the other nine** — the CFB prior spends
**96.2%** of its squared coefficient movement on the team-quality block. And it
still changes **zero** picks.

**Prior-vector stability** (CFB-only, no NFL window spent, pre-window seasons
2006-2010, 1,128 odd-season and 1,183 even-season rows): Pearson **r =
−0.5333**, cosine similarity **−0.4986**. The transferred coefficient vector is
*anti*-correlated between two halves of its own corpus. (The Spearman-Brown
correction returns −2.286, which is why it is quoted here and NOT recorded: the
correction is only meaningful for a positive `r`.)

### What this implies for the decision, before what is wrong with it

**Arm (d), the headline mechanism, plays the literally identical card to the
baseline: 0 of 746 picks differ.** There is therefore no EV decision here to
get wrong in either direction — the honest `probability_positive` is **0.500**,
EV-neutral, and the raw bootstrap's 0.000 is an artifact of `mean(draws > 0)`
scoring a dead heat as a loss (section 6a). Arm (b) is a genuine coin flip that
moves a lot of paper: it differs on **220 of 746 picks and splits them exactly
110/110**, tie-aware P+ 0.522. Arm (c) is the only arm that leans, and it leans
**toward the candidate**: +0.134 points, 4 of its 7 differing picks right,
tie-aware P+ **0.652** week-blocked and **0.860** season-blocked, with a
season-blocked lower bound sitting at exactly 0.000. On EV grounds — the only
decision rule this project uses — **nothing here argues for the status quo over
the CFB prior; arm (c) mildly argues the other way, and arms (b) and (d) argue
for neither.** What none of them do is change enough picks to matter: this is
close-graded, so per the binding "grade at the opener" rule it settles no play
decision regardless, and even if it could, arm (d) has no different card to
play.

**The reference line is the most decision-relevant number in the run, and it
comes with a measured caveat that defuses it.** Full production `weak_stack`
scores **49.866%** on these 746 games against the 14-column `nfl_only` arm's
**50.938%** — the ~90-column production chain is **1.07 accuracy points worse
than a 14-column ridge** on this window. Before anyone reads that as an
indictment of production: **measured this session**, 7 of production's 90 model
columns (the lineup-continuity family) have **0% coverage in 2009-2012 and
87.5% in 2013**, so across the whole assigned window production is running with
part of its feature set median-imputed to a constant and contributing nothing;
2011-2013 also sits entirely outside the seasons on which the production chain
was built and selected. This is a close-graded, era-mismatched read on 746
games and it may not veto or move anything. It is flagged rather than buried
because it is exactly the kind of number a future opener-graded look should
want to chase.

**The "team quality is already priced" bound: checked, and the check is
informative.** The bound could not apply by construction — all four arms are fit
on the identical 14-column design, so no arm estimates team quality better than
another (section 1). The decomposition says something sharper than that,
though: **96.2% of the CFB prior's squared coefficient movement lands in the
six team-quality columns**, the exact block the owner's bound says is already
priced — and the `prior_market_only` diagnostic, which forbids the prior from
touching a single one of those coefficients, changes **zero** picks too. So the
gain is zero on *both* sides of the Q partition. There is no gain to attribute
to team quality, and there is none outside it either; the transfer is not being
throttled by the bound, it is simply not moving the card.

**Why it does not move the card, stated as mechanism rather than as a verdict.**
Two measured facts, and one of them contradicts a written expectation this
project was carrying. (1) At the frozen `ridge_alpha = 10.0`, the prior's pull
on `home_cover_probability` maxes out at **0.016** (measured on the 2011 slice;
about 0.55 points of predicted margin), which flips a forced pick only for games
sitting within about half a point of the fence — hence 7 flips for arm (c) and 0
for arm (d). `docs/scaling_and_transfer.md` measured the same near-no-op
CFB-internally and predicted it would NOT hold on NFL ("NFL's per-window
training sizes are typically much smaller than Group-of-Five's here, so the same
alpha would bind proportionally harder there"). **That prediction does not hold
up**: at 512-1,264 NFL training rows the prior is, if anything, even more inert
on the pick than it was at Group-of-Five scale. Retuning alpha would be new
tuning this predeclaration does not authorize, and is the obvious next
predeclaration. (2) The transferred object is not stable: `theta_cfb` fit on odd
CFB seasons is anti-correlated (r = −0.53) with `theta_cfb` fit on even CFB
seasons. Per `docs/scaling_and_transfer.md`'s own caveat, S contains three exact
collinearities by construction (`diff = home − away` for the EPA triples), so
ridge is free to split weight arbitrarily among collinear columns and a small
data change can swing that split hard — which makes an anti-correlated
whole-vector reading much more plausibly a **design defect in S than a fact
about football**. Dropping the redundant `diff_*` columns and re-measuring is a
concrete, cheap, CFB-only follow-up.

**What this does NOT close, and why.** Every entry is
`unresolved_below_power`. `refuted_mechanism` is unavailable on both of its
grounds: no arm's interval sits entirely on the wrong side of zero (arm (c)'s
week-blocked interval [−0.669, +0.926] straddles it and its point estimate is
*positive*; arms (b) and (d) are exact ties, which is not a wrong sign), and
`no_split_half_reliability` was ruled out **in the predeclaration, before the
look** (section 6) — this family's arms differ in the ESTIMATOR, not in a
feature column, so there is no trait to split, and the −0.53 prior-vector
stability figure is deliberately NOT recorded in the registry's `reliability`
field to stop a future session mistaking it for a trait reliability and reading
a closure out of it. `bounded_by_control` is unavailable too: the positive
control proved the harness detects a **49-point** effect, and nothing here
proved it would have detected the fractions-of-a-point effect actually at issue
— a control that large bounds nothing at this scale. The family stays **open**
with 2 eligible windows remaining, and the two follow-ups it points to (a
declared prior-strength/alpha sweep; S with its exact collinearities removed)
are new predeclarations, not reruns of this one.

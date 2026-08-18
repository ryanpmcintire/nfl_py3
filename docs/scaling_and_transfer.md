# Scaling and cross-league transfer

Written 2026-08-18 (US). Two questions the project had never put a number
on: are we data-limited or model-limited, and does the CFB corpus (~12,500
games) have unexploited value as a **pretraining** source rather than only
a screening substrate for ~4,431 usable NFL games? Code lives in
`src/nfl_ats/cross_league_transfer.py`, tests in
`tests/test_cross_league_transfer.py`, drivers in
`scripts/scaling_learning_curve.py` and
`scripts/scaling_cross_league_transfer.py`. Nothing here touched
`registry/rotation_registry.json`, changed a pick, or spent an NFL
confirmation window -- everything that requires one is predeclared at the
end of this document for a future session.

## Headline verdicts

- **Data-limited or model-limited: principally model/feature-limited**, for
  the metric the pool is actually judged on. Forced-pick accuracy is flat
  across a 100-fold-plus range of CFB training-set size, flat across the
  frozen NFL backtest's own (narrower) range, and flat in the
  pre-existing, independent `docs/rotation_registry.md` sweep -- three
  convergent measurements. Continuous error metrics (margin MAE, Brier) DO
  keep improving with more data, but with sharply diminishing returns
  already ~90% realized by ~800 CFB training games (~8% of the available
  corpus), and closing the residual gap to match the opponent-adjustment
  leak probe's own +0.0129-point ceiling would need roughly **35,000
  games -- about 2.8x the entire 12,500-game CFB corpus this project has
  access to**. Not a discrepancy with that ceiling -- the same fact from a
  different angle.
- **Does borrowing strength from CFB help, hurt, or nothing: helps, with a
  textured verdict.** On a free, real, CFB-internal large-to-small
  distribution-shift proxy (Power-Five auxiliary, Group-of-Five target),
  joint fitting resolves a positive margin-MAE/RMSE/Brier/log-loss
  improvement over the target-only fit under both week- and season-blocking
  (`probability_positive` 0.94-0.99+). No arm shows resolved HARM anywhere.
  Forced-pick accuracy itself does not resolve for any arm, but
  hierarchical shrinkage is the only arm with a positive accuracy lean
  (`probability_positive` 0.89 week-blocked, 0.85 season-blocked) --
  exactly the kind of "unresolved, not negative" signal AGENTS.md says must
  be reported, not discarded. Hierarchical shrinkage earns the NFL
  predeclaration below; prior-mean ridge does not, in this form (see
  caveat).

---

## Part 1 -- the learning curve

### Method

**CFB (primary, well-powered).** Fixed test window: every completed CFB
game, 2023-2025 (2,244 games, 123 weeks). For a doubling grid of training
sizes `K in {100, 200, 400, 800, 1600, 3200, 6400, full (~10,256)}`,
walked forward week by week, truncating each week's training frame to the
`K` most recent completed games before that week's cutoff (point-in-time
identical to the frozen XLG-03 recipe -- only the amount of history differs)
and refit `fit_cfb_residual_model` (Ridge alpha 10, verbatim) at each `K`.
Every `K` arm sees the exact same 2,244 test games.

**NFL (secondary, read-only, no fresh window).** Re-aggregated the
ALREADY-FROZEN weak_stack backtest
(`artifacts/margins/20260818T012407Z/predictions.parquet`, the active
model's own historical evaluation, 2018-2025, 2,127 `market_residual` rows)
by its already-recorded `train_rows` column into 8 quantile bins
(2,304-4,415 training rows, a 1.77x range). No new fit; pure re-slicing of
predictions that were made and scored before this session started.

### CFB results

| K | games | accuracy | Brier | margin MAE |
|---|---|---|---|---|
| 100 | 2,244 | 0.5107 | 0.2976 | 14.288 |
| 200 | 2,244 | 0.4993 | 0.2906 | 13.340 |
| 400 | 2,244 | 0.5071 | 0.2672 | 12.725 |
| 800 | 2,244 | 0.5157 | 0.2555 | 12.251 |
| 1,600 | 2,244 | 0.5080 | 0.2530 | 12.155 |
| 3,200 | 2,244 | 0.5116 | 0.2522 | 12.146 |
| 6,400 | 2,244 | 0.5071 | 0.2513 | 12.100 |
| full (~10,256) | 2,244 | 0.5162 | 0.2508 | 12.087 |
| market | 2,244 | 0.5016 | 0.2500 | 12.035 |

Paired, week-blocked, 2,000 samples, seed 20260818 (`probability_positive`
positive = more data wins):

| comparison | margin MAE improvement | P+ | accuracy improvement | P+ |
|---|---|---|---|---|
| 100 -> 800 | +2.037 [+1.381, +2.964] | 1.000 | +0.005 [-0.022, +0.032] | 0.640 |
| 100 -> full | +2.201 [+1.526, +3.123] | 1.000 | +0.005 [-0.020, +0.030] | 0.655 |
| 800 -> full | +0.164 [+0.044, +0.279] | 0.996 | +0.000 [-0.024, +0.023] | 0.518 |
| 1,600 -> full | +0.068 [-0.016, +0.153] | 0.947 | +0.008 [-0.015, +0.032] | 0.751 |

93% of the entire 100->full margin-MAE gain (2.037 of 2.201 points) is
already captured by 800 games -- about 8% of the available CFB corpus.
Brier and log-loss keep resolving positive even 800->full
(`probability_positive` 0.99), but at a magnitude an order smaller
(+0.0047 Brier points) than the 100->800 step (+0.042). Accuracy never
resolves at any comparison (`probability_positive` 0.52-0.75 throughout) --
consistent with, not contradicting, the margin-error picture: accuracy is a
coarse ~2-point-resolution instrument (AGENTS.md), and these are
fractions-of-a-point effects.

A power law `metric(n) = a + b * n^-c` fits the margin-MAE curve tightly
(`a` = 12.011, `b` = 135.72, `c` = 0.885, R² = 0.993). Extrapolating: closing
a further 0.05-point gap needs ~7,600 games (close to what `full` already
has, at a realized mean of 11,351 training games across the walk-forward);
closing a 0.0129-point gap -- the exact size of the deliberate-leak
positive control's ceiling in `docs/cfb_opponent_adjustment.md` -- needs
**~35,165 games: roughly 3.1x the training pool this screen actually used
and ~2.8x the entire 12,500-game CFB corpus this project has access to**.
That is not a contradiction of the leak-probe ceiling; it is the same wall
measured a second way. The leak probe showed perfect foreknowledge of team
quality is worth only 0.013 points; this curve shows that even an
enormous, currently unobtainable increase in real data would buy about
that same amount. Both readings say the team-quality family is close to
exhausted, not that one contradicts the other.

### NFL results (read-only)

| train_rows bin | games | accuracy | Brier | margin MAE |
|---|---|---|---|---|
| 2,304-2,560 | 272 | 0.477 | 0.257 | 10.25 |
| 2,576-2,832 | 272 | 0.502 | 0.254 | 9.92 |
| 2,848-3,088 | 256 | 0.551 | 0.254 | 10.33 |
| 3,104-3,360 | 272 | 0.521 | 0.252 | 10.68 |
| 3,376-3,631 | 271 | 0.523 | 0.253 | 8.80 |
| 3,647-3,887 | 256 | 0.537 | 0.254 | 10.15 |
| 3,903-4,159 | 272 | 0.522 | 0.251 | 9.47 |
| 4,175-4,415 | 256 | 0.494 | 0.254 | 10.07 |

Pearson correlation of `train_rows` against each metric: accuracy r=0.226
(p=0.59), Brier r=-0.518 (p=0.19), margin MAE r=-0.296 (p=0.48). None
resolve at this narrow, 1.77x range and n=8 bins -- underpowered, not
contradictory. Brier's negative (improving) direction is consistent with
the CFB finding.

This independently corroborates the pre-existing `docs/rotation_registry.md`
measurement (real NFL walk-forward, 2012-2025, 3,573 games, training
truncated to N=50/500/full): forced-pick accuracy flat at .509/.499/.508;
Brier and margin MAE "degrade smoothly and monotonically... no cliff at 500
or anywhere else." Three independent measurements (CFB doubling grid, NFL
frozen re-slice, and the prior NFL N-sweep) now agree.

### Verdict

For the metric the pool is graded on -- forced-pick accuracy -- this
project is **model/feature-limited**, not data-limited, and has been for a
while: no measurement anywhere (CFB, NFL, at any sample size from 50 to the
full corpus) resolves an accuracy gain from more rows. For continuous error
metrics there is a real, resolvable, but now nearly-saturated data effect:
worth having captured, not worth further investment relative to feature
work. This is consistent with, and gives a second, independent measurement
of, the `docs/pool_edge_plan.md` finding that team-quality-measurement
features are bounded near zero.

---

## Part 2 -- CFB as pretraining, not just a test bed

### The aligned feature contract

CFB and NFL feature tables diverge past a 14-column intersection: CFB's
base contract lacks pass/rush-split EPA, CPOE, Elo, and yards/turnover/sack
rate; NFL's base profile lacks success rate, explosive rate, and pace at
this contract level. `ALIGNED_TRANSFER_FEATURE_COLUMNS` is the literal
column-name intersection, verified in
`tests/test_cross_league_transfer.py::test_aligned_columns_are_a_true_subset_of_both_leagues_contracts`
against the real `CFB_MODEL_FEATURE_COLUMNS` and NFL `FEATURE_SETS["full"]`
constants: `spread_line`, `total_line`, `rest_diff`, `neutral_site`,
`week_sin`, `week_cos`, `home_team_games`, `away_team_games`, and the
home/away/diff triples of `off_epa_per_play` and `def_epa_per_play`.

### Mechanisms (`src/nfl_ats/cross_league_transfer.py`)

- **`fit_joint_league_model`** -- pools both leagues' rows plus a binary
  league indicator into one ridge fit (alpha 10, verbatim), so the 14
  shared coefficients are estimated on the pooled sample.
- **`fit_hierarchical_shrinkage_model`** / **`derive_shrinkage_weights`** --
  partial pooling. The target league's own ridge estimate is shrunk toward
  the auxiliary league's, per augmented coefficient (14 features + league
  intercept), by an empirical-Bayes weight `w_j = tau^2 / (tau^2 +
  sigma_j^2)`: `sigma_j^2` from a week-blocked bootstrap of the target-only
  fit, `tau^2` from the DerSimonian-Laird closed-form moment estimator with
  known per-coefficient prior means (the auxiliary fit). Derived once from
  the pool strictly before the test window, then held fixed while the two
  anchor estimates keep updating walking forward.
- **`fit_prior_mean_ridge_model`** -- generalized ridge, `||y - X*theta||^2
  + alpha*||theta - theta_0||^2`, `theta_0` the auxiliary-only fit. Solved
  by residualizing against the prior and running an ordinary
  `Ridge(fit_intercept=False)` on the residual (verified algebraically
  identical to the closed form in
  `test_prior_mean_closed_form_matches_direct_linear_algebra`).

All three wrap their fitted coefficients in a `_FixedLinearRegressor` that
plugs into `nfl_ats.margin.MarginModel` unchanged, so cover probabilities,
the residual-distribution recipe (80/20 out-of-time split, refit on 100%,
identical to `fit_cfb_residual_model`), and every downstream bootstrap
reuse the project's existing, tested machinery rather than a fourth
implementation. Leak-safety (perturbing rows at or after each scored week
must not change earlier predictions, and training must never reach the
scored week) is a release-blocking regression test, per AGENTS.md, in
`tests/test_cross_league_transfer.py`.

### Measuring the mismatch first (full-history, static diagnostic)

Before trusting any shrinkage, the two leagues' fitted coefficients were
compared on a POOLED preprocessing pipeline (one imputer, one scaler, fit on
the union of both leagues) so they live in a genuinely common space --
`measure_league_mismatch`, never scored against held-out outcomes.

Proxy leagues for this measurement (and the benchmark below): CFB
Power-Five-vs-Power-Five games as the large "auxiliary" pool (5,684 games,
2006-2025) and Group-of-Five-vs-Group-of-Five games as the smaller
"target" pool (4,914 games) -- a real, measured talent and market-depth
split inside one sport, entirely free under rotation-registry rule 8.

| | value |
|---|---|
| cosine similarity (feature coefficients) | **-0.398** |
| Pearson correlation | -0.411 |
| residual std, group5 / power5 | 15.507 / 15.439 (ratio 0.996) |
| features agreeing in sign | 6 / 14 |

The mismatch is real and not small: the two tiers' full-history fits are
mildly ANTI-correlated overall. Market and defense-EPA columns mostly agree
in sign; context (`neutral_site`, `week_sin/cos`) and the offense-EPA /
experience columns mostly do not (`diff_off_epa_per_play`: +2.11 for
Group-of-Five vs -0.59 for Power-Five). Residual scale is nearly identical
(ratio 0.996) -- the outcome noise itself does not differ between tiers,
only how the predictors relate to it. This is the honest empirical basis
the shrinkage below needs: a naive full-pooling or heavy prior-mean-toward-
auxiliary scheme would risk real harm on the mismatched half of the
contract, which is exactly why the derived weights (below) land at a mean
of 0.64 favoring the target's own data rather than close to 0 or 1.

Caveat: several columns are exactly collinear by construction (`diff = home
- away`), so individual coefficient SIGNS are not fully stable estimates on
their own (`tests/test_cross_league_transfer.py` documents this explicitly
and checks whole-vector cosine similarity rather than asserting
component-wise sign agreement for exactly this reason). The cosine/
correlation summary is the robust reading; the per-feature table
(`mismatch_per_feature.csv`) is diagnostic detail, not a claim about any
single column in isolation.

### Derived shrinkage strength

From the pre-2018 pool (2,570 Group-of-Five games, 3,222 Power-Five games),
200-sample week-blocked bootstrap: `tau^2 = 0.751`; weights (fraction trust
on the target's own data) range **0.343 to 0.919** across the 15 augmented
components, mean **0.640**. Genuinely derived and genuinely partial --
neither collapsing to "ignore CFB" (w=1) nor "ignore the target league"
(w=0) for any component.

### Transfer benchmark: Group-of-Five walked forward, 2018-2025

2,344 target-league games, 123 weeks, all scored (no games dropped to the
training floor). Paired against `target_only` (the proxy's "current
NFL-only-style fit"), 2,000-sample week- and season-blocked bootstrap, seed
20260818:

| arm | metric | week-blocked | P+ | season-blocked | P+ |
|---|---|---|---|---|---|
| joint | margin MAE | +0.0437 [+0.0104, +0.0774] | **0.9945** | +0.0437 [+0.0018, +0.0816] | 0.9785 |
| joint | margin RMSE | +0.0533 [+0.0136, +0.0903] | 0.9955 | +0.0533 [+0.0058, +0.0980] | 0.9910 |
| joint | Brier | +0.00124 [+0.00026, +0.00220] | 0.9940 | +0.00124 [+0.00008, +0.00229] | 0.9855 |
| joint | accuracy | -0.0013 [-0.0171, +0.0146] | 0.4250 | -0.0013 [-0.0163, +0.0110] | 0.3985 |
| hierarchical | margin MAE | +0.0361 [-0.0029, +0.0743] | 0.9665 | +0.0361 [-0.0160, +0.0912] | 0.8955 |
| hierarchical | margin RMSE | +0.0524 [+0.0121, +0.0941] | 0.9940 | +0.0524 [+0.0030, +0.1033] | 0.9805 |
| hierarchical | Brier | +0.00099 [-0.00013, +0.00206] | 0.9580 | +0.00099 [-0.00056, +0.00258] | 0.8825 |
| hierarchical | accuracy | **+0.0113 [-0.0066, +0.0287]** | **0.8885** | +0.0113 [-0.0076, +0.0347] | 0.8470 |
| prior_mean | margin MAE | +0.0002 [-0.0002, +0.0006] | 0.8805 | +0.0002 [~0, +0.0005] | 0.9700 |
| prior_mean | accuracy | +0.0009 [-0.0005, +0.0026] | 0.7930 | +0.0009 [-0.0009, +0.0025] | 0.7965 |

(Positive = the candidate beats `target_only`; full table in
`arm_vs_target_only.csv`.)

Raw accuracy levels: target_only 49.10%, joint 48.98%, hierarchical
**50.21%**, prior_mean 49.19%, market 49.23%. Levels vs market (delta,
week-blocked, not resolved for any arm): target_only -0.041 margin-MAE
points worse than market [-0.010, +0.089]; joint +0.002 points better
[-0.045, +0.038]; hierarchical +0.005 points worse [-0.035, +0.043].

Reading this honestly: **joint fitting resolves a real, if modest,
improvement on every continuous metric under both blockings** --
`probability_positive` from 0.94 to 0.99+ -- but does not move (and leans
mildly against) forced-pick accuracy. **Hierarchical shrinkage is weaker on
the continuous metrics** (margin MAE crosses zero on the tighter
week-blocked interval, though RMSE still resolves) **but is the only arm
with a positive accuracy lean** -- unresolved (`probability_positive` 0.89
week-blocked, 0.85 season-blocked), not negative, and per AGENTS.md's
binding rule that is not grounds to set it aside. **Prior-mean ridge is
close to a no-op here**: at the project's frozen, unretuned `ridge_alpha =
10`, the prior's pull is governed by the same small penalty that barely
regularizes an already-multi-thousand-row fit, so `theta` moves almost
nothing from `theta_target_only`. That is a real property of this specific
alpha at this specific sample size, not evidence the mechanism is
worthless -- NFL's per-window training sizes are typically much smaller
than Group-of-Five's here, so the same alpha would bind proportionally
harder there. Retuning it would be new tuning, not something this
predeclaration authorizes.

### Why this is a genuine validity check, not a curiosity

Two things make the Power-Five/Group-of-Five split a reasonable stand-in
for "does borrowing strength from a larger, differently-distributed
football corpus help a smaller one," rather than a toy: it is a real
measured mismatch (cosine similarity -0.40, not the near-1.0 a same-
distribution split would show), and the residual scale is nearly identical
between the two pools (ratio 0.996), so any measured effect is coming from
the coefficient relationship, not from one pool simply being noisier. It
is understated relative to the real CFB->NFL case in one respect: the
size ratio here (5,684 vs 4,914, 1.16x) is far gentler than CFB's actual
12,500 vs NFL's ~4,431 (2.8x), so a real CFB-auxiliary transfer has access
to a proportionally larger pretraining pool than this proxy tested with.

It remains a proxy, not a CFB->NFL result: the two CFB tiers share a market
(same books, same season, similar line-setting process) in a way CFB and
NFL do not, so **market efficiency differences -- the dimension most likely
to break a naive transfer -- are not exercised by this screen at all**.
A positive result here is a green light to run the real screen, not a
substitute for it.

### Verdict

**Helps, textured.** No arm shows resolved harm on any metric. Joint
fitting resolves positive on every continuous metric with high confidence.
Hierarchical shrinkage is the only arm leaning positive on the metric that
actually matters for the pool (forced-pick accuracy), unresolved but not
negative. Prior-mean ridge at the frozen alpha is a near no-op here for a
specific, understood, sample-size reason rather than a mechanism failure.
This closes nothing and clears the screen threshold this project has used
elsewhere (`probability_positive >= 0.75`) for hierarchical shrinkage on
accuracy specifically -- earning the predeclaration below.

---

## Predeclaration for a future NFL session

**Not run this session. Requires `registry/rotation_registry.json` writes
this agent is not permitted to make.** Framed here so a future session can
execute mechanically.

- **Family name (proposed):** `cross_league_hierarchical_transfer`.
  **Grade:** `nflverse_spread` (close) for the initial screen, matching this
  project's established two-stage pattern (`best_pick_ranker` ->
  `best_pick_ranker_opener`); an opener-graded confirmation arm follows only
  if the screen clears, per AGENTS.md's binding "grade the decision at the
  opener" rule for anything that could move a pick.
- **Inherits:** none. This is a genuinely new mechanism hypothesis (a
  modeling technique, not a feature-family variant of `pbp_drive_bundle` or
  `player_qb_continuity`), so it does not inherit their spent windows.
- **Auxiliary league for the real run:** the FULL CFB corpus (all
  conferences, not Power-Five-only), 2006 through the season strictly
  before the assigned window -- Power-Five-only was this session's proxy-
  construction choice to create a matched target/auxiliary split, not a
  recommendation for the real confirmation's auxiliary pool. The actual
  asset being tested is "all ~12,500 CFB games."
- **Target:** NFL, the same 14-column `ALIGNED_TRANSFER_FEATURE_COLUMNS`
  contract -- NOT the full production (`weak_stack`) feature set. The
  `target_only` baseline must be fit on the identical 14 columns so the
  comparison isolates "does transfer help" from "do we already have richer
  features"; comparing a transfer arm to the full production model would
  confound the two questions.
- **Frozen parameters:** `ridge_alpha = 10.0` verbatim for target-only,
  joint, and hierarchical arms (no retuning). Shrinkage weights derived
  once, before the window, by the same DerSimonian-Laird week-blocked-
  bootstrap recipe used here (>=150 samples), held fixed through the walk.
  Prior-mean ridge runs at `alpha = 10` as the fair, no-retuning arm; a
  SEPARATELY predeclared arm may explore a derived `alpha_prior` (e.g. sized
  to the auxiliary pool's information content) only as new, disclosed
  tuning -- not pre-authorized here.
- **Primary metric:** forced-pick accuracy (this session used margin MAE as
  primary because CFB favors continuous metrics at this n; the real NFL
  confirmation must flip the primary back to accuracy, the project's stated
  goal). Margin MAE/RMSE, Brier, and log-loss report as coherence
  secondaries, exactly as `docs/cfb_opponent_adjustment.md` did.
- **Window:** assign via `nfl-ats rotation assign` for a fresh
  `nflverse_spread` family -- do not hand-pick a block in a document. Per
  the ledger read this session (`2013`:2, `2014`:3, `2015`:3, `2016`:2,
  `2017`:2, `2020`:2, `2021`:2 looks logged; no family has spent seasons
  earlier than 2013), the earliest eligible 3-season block for a
  non-inheriting family is expected to land at `[2011, 2013]` under the
  current assignment algorithm, but the CLI computes this at declaration
  time, not this document.
- **Decision rule:** report `probability_positive` for every arm and metric,
  never a bare pass/fail. Screen threshold for a further confirmation:
  `probability_positive >= 0.75` on accuracy specifically (this project's
  established screen bar), for hierarchical shrinkage first given this
  session's lean; joint fitting reports alongside as a secondary candidate
  given its strong continuous-metric evidence despite the accuracy miss
  here.

## Artifacts

- `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\scaling\learning_curve\` --
  CFB learning-curve predictions, summary, week-blocked intervals, power-law
  fits, and the NFL frozen re-slice.
- `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\scaling\cross_league_transfer\` --
  mismatch report, shrinkage derivation diagnostics, transfer-benchmark
  predictions, delta-vs-market, and each arm's paired evidence vs
  `target_only`.

Both scratchpad directories are session-local and not committed; rerun the
two driver scripts to regenerate them.

## Goal

Read-only research question: does season-recency sample weighting on the
served base margin model (`artifacts/active_ats_model.json`: `market_residual`,
`weak_stack`, ridge alpha 10) beat the unweighted served model, standalone at
the opener and as `model_logit` inside the four-term pick-probability fit
(`src/nfl_ats/pick_probability_fit.py`)? Mechanism: rules/scoring environment
drift, so older seasons may mislead the residual model. No change to the
active model or served path.

## Predeclared grid (before any number was read)

Half-life candidates: `{2, 4, 8 seasons, baseline (unweighted, all history)}`.
Weight formula (season granularity, matches the registered `era_weighted_
half_life_8` challenger, `src/nfl_ats/era_weighted_half_life_8_overlay.py`):
`weight = 0.5 ** (max(0, predict_season - row_season) / half_life)`.

Half-life is chosen **per outer test season** (2020-2025, walk-forward) by
inner leave-one-season-out validation restricted to the up-to-3 seasons
immediately preceding that outer season (`INNER_LOOKBACK=3`), never using the
outer season or any later season. Argmax mean inner accuracy; ties keep the
list order `[2, 4, 8, baseline]`.

Disclosed scope restriction (per task's own allowance): full weekly refit
across a 4-candidate x 6-fold x ~3-inner-season grid was too slow for this
session's budget, so both inner and outer fits are refit once per **season
boundary** (cutoff = that season's first game) and reused for every week in
that season, not refit weekly like the production card. The inner half-life
selection scores accuracy at the **close** line (`spread_line`, no opener
archive exists pre-2020); the outer 2020-2025 evaluation grades at the
**opener** (`docs/opener_evaluation.md`'s archive, `tue_open_home_spread`),
matching AGENTS.md's "grade the decision at the opener" rule for the actual
decision-grade read. `min_train_games=500`, `ridge_alpha=10.0`,
`feature_profile="weak_stack"`, `probability_method` read from
`nfl_ats.clv.resolve_active_model_config` (the served config, `gaussian_median`
as of 2026-09-23), discrete push read via `nfl_ats.mass_preserving_lattice`
(the actual served discrete mechanism), all imported unmodified.

Metrics: standalone (opener discrete read) accuracy/Brier/log-loss/margin MAE,
paired vs the served archive's own numbers; and the same four metrics with
each side's `model_logit` fed into the four-term fit
(`model_logit, flag_sum, move, move_available`) refit LOSO across 2020-2025,
reusing `nfl_ats.pick_probability_fit`'s private `_design/_fit_logit/_predict/
_standardisers/build_fit_population` unmodified. Week-blocked paired
bootstrap, 2000 resamples, seed 20260923 (lighter than this repo's usual
20000 for session-budget reasons -- disclosed, not hidden).

Prior art this does not re-litigate: `docs/era_weighting_screen.md` (MOD-14)
already screened this exact half-life grid two other ways (CFB walk-forward,
NFL close-grade) and found `half_life_8` leaning positive on both, but
**negative** on its own NFL **opener**-grade read (Section 8, P+ 0.299) --
unresolved throughout, never promoted. This lane's contribution is a fresh,
independently-selected (nested, not best-of-six) half-life per fold, evaluated
both standalone and inside the four-term fit, which MOD-14 did not do.

## State -- DONE, measured, not recorded to the registry yet

Ran twice. First run (`artifacts/base_model_recency/20260923T211059Z/`) had a
column-semantics bug in the served margin-MAE comparison (treated
`residual_at_open`, which is the served model's **predicted** market
residual per `nfl_ats.clv` line 2128, as an error term instead of subtracting
it from `margin_vs_open`) -- disclosed, not hidden; superseded. Second run
(fixed, trusted) artifact: `artifacts/base_model_recency/20260923T211631Z/
metadata.json`. Accuracy/Brier/log-loss numbers are identical across both
runs (unaffected by the bug); only margin MAE changed.

**Chosen half-life per outer fold** (inner LOSO on the up-to-3 preceding
seasons, close-grade, argmax mean inner accuracy): 2020 -> `baseline`
(unweighted), 2021 -> `half_life_2`, 2022 -> `baseline`, 2023 -> `baseline`,
2024 -> `half_life_4`, 2025 -> `half_life_4`. **No half-life wins
consistently** -- 3 of 6 folds select the unweighted baseline itself, unlike
`docs/era_weighting_screen.md` (MOD-14), which picked one arm
(`half_life_8`) by comparing all six candidates' OUTER performance (a form
of in-sample selection this lane's nested design avoids).

**Standalone, opener discrete read, 2020-2025 (n=1503 graded, pushes
excluded), week-blocked bootstrap, 2000 samples, seed 20260923:**
| Metric | Recency | Served | Effect (recency - served) | 95% CI | P+ |
|---|---:|---:|---:|---:|---:|
| Accuracy | 52.10% | 53.36% | -1.264 pts | [-3.065, +0.466] | 0.071 |
| Brier | 0.25251 | 0.25172 | -0.000788 (improvement) | [-0.00293, +0.00114] | 0.229 |
| Log-loss | 0.69852 | 0.69692 | -0.001602 (improvement) | [-0.00603, +0.00235] | 0.2335 |
| Margin MAE | 11.269 | 10.190 | **-1.079 (improvement), RESOLVED WORSE** | **[-1.357, -0.805]** | **0.0** |

Margin MAE is the only metric whose whole interval sits on one side of zero
here: recency-weighting is resolved worse at the opener on this continuous
metric (matches the exact "accuracy unresolved, continuous metrics resolved
worse" shape `docs/era_weighting_screen.md` already found for `rolling_6`
and `half_life_2` -- not a contradiction, a repeated pattern). Accuracy
leans negative but is `unresolved_below_power` (P+ 0.071, interval crosses
zero) -- this also matches MOD-14 Section 8's own opener-grade finding for
`half_life_8` (P+ 0.299, also negative-leaning, also unresolved).

**Fed as `model_logit` inside the four-term fit
(`model_logit, flag_sum, move, move_available`), LOSO-refit 2020-2025,
n=1503, same bootstrap config:**
| Metric | Recency4 | Served4 | Effect | 95% CI | P+ |
|---|---:|---:|---:|---:|---:|
| Accuracy | 57.22% (860-643) | 57.15% (859-644) | +0.0665 pts | [-0.997, +1.126] | 0.5385 |
| Brier | 0.24522 | 0.24539 | +0.000165 (improvement) | [-0.00039, +0.00070] | 0.722 |
| Log-loss | 0.68364 | 0.68395 | +0.000310 (improvement) | [-0.00083, +0.00142] | 0.700 |
| Decisive games (picks differ) | 29-28 | 28-29 | n=57 | -- | -- |

Inside the combined fit the standalone opener-margin penalty washes out to
essentially flat/coin-flip on every metric -- nothing here resolves either
direction.

**Disclosed restrictions** (stated in the grid section above, restated
here): season-boundary refit not weekly; inner half-life selection graded at
the close (no pre-2020 opener archive exists), outer 2020-2025 evaluation
graded at the true opener; 2000 bootstrap samples not this repo's usual
20000, for session-budget reasons.

## Tried

`/f/Repos/nfl_py3/.tools/uv.exe run python scripts/base_model_recency.py`
ran clean twice (exit 0) -- run 1 (`20260923T211059Z`, margin-MAE bug) and
run 2 (`20260923T211631Z`, fixed, trusted; **all numbers in this lane are
read from run 2**). All accuracy/Brier/log-loss/fold-selection numbers below
are unaffected by the margin-MAE bug and by the ruff cleanup that followed
(they only touched string formatting, not arithmetic).

`ruff check` on the first working version of the script found 17 style
errors (line length, one `C416` dict-comprehension); `--fix` handled 4, the
rest were fixed by hand across ~10 `Edit` calls. **BUG INTRODUCED BY THAT
CLEANUP, NOT YET FIXED**: the post-cleanup script now raises
`TypeError: 'list' object is not callable` at
`scripts/base_model_recency.py:201` inside `_week_blocked_bootstrap`, on the
line `grouped = dict(df.groupby(["season", "week"]))` -- some edit in that
same pass rebound the name `dict` to a list somewhere in reachable scope
(not yet isolated; a `grep -n "\bdict\s*="` of the file, excluding `dict[`
type hints, was queued but did not run before the tool-call cap hit). The
**results below are still trustworthy** (produced by run 2, before this
regression existed) but the **script on disk right now will not reproduce
them if re-run** -- it must be fixed first. Likely one-line fix: replace
`grouped = dict(df.groupby(["season", "week"]))` with a dict/comprehension
that never calls a bare `dict(...)`, e.g.
`grouped = {key: group for key, group in df.groupby(["season", "week"])}`
(ruff flagged that exact original form as `C416` and auto-"fixed" it to the
`dict(...)` call that now breaks -- the auto-fix is the suspected culprit;
if reverting it alone does not clear the `TypeError`, grep the whole file
for a stray `dict =` / `dict:` binding before assuming anything else).

## Next (for the root orchestrator)

1. Fix the `_week_blocked_bootstrap` regression above (see exact line and
   suspected cause), rerun
   `/f/Repos/nfl_py3/.tools/uv.exe run python scripts/base_model_recency.py`,
   confirm exit 0 and that the new run's `standalone_summary` /
   `combined_fourterm_summary` numbers match run 2's (already reproduced once
   before the regression, so a mismatch means a second bug, not noise), then
   `/f/Repos/nfl_py3/.tools/uv.exe run ruff check scripts/base_model_recency.py`
   clean before anything else touches this file.
2. Record two registry entries (classifications are mechanical per AGENTS.md's
   binding taxonomy: the standalone accuracy interval crosses zero so it is
   `unresolved_below_power`; the standalone margin-MAE interval is entirely
   negative, so if recorded as its own entry it is the one place
   `wrong_sign_resolved` legitimately applies here, closing-ground
   `wrong_sign_resolved`; the combined four-term entries are both
   `unresolved_below_power`, intervals cross zero):
   `nfl-ats weak-signals record --name base_model_recency_standalone_opener_accuracy --league nfl --category model_recipe --effect-units accuracy_points --effect -1.2641 --ci-low -3.0648 --ci-high 0.4664 --probability-positive 0.071 --classification unresolved_below_power --plain-summary "Season-recency-weighted base model (half-life chosen out-of-season per fold: 2020/2022/2023 baseline, 2021 half-life 2, 2024-2025 half-life 4) vs the served unweighted base model, standalone opener discrete read, 2020-2025, n=1503."`
   `nfl-ats weak-signals record --name base_model_recency_standalone_opener_margin_mae --league nfl --category model_recipe --effect-units margin_mae_points --effect -1.0789 --ci-low -1.3571 --ci-high -0.8055 --probability-positive 0.0 --classification wrong_sign_resolved --closing-ground wrong_sign_resolved --plain-summary "Same standalone comparison, margin MAE: recency-weighted model resolves WORSE (11.27 vs 10.19 pts MAE), whole week-blocked interval negative; accuracy itself stays unresolved, matching docs/era_weighting_screen.md's rolling_6/half_life_2 continuous-metric-vs-accuracy divergence pattern."`
   `nfl-ats weak-signals record --name base_model_recency_fourterm_combined_accuracy --league nfl --category model_recipe --effect-units accuracy_points --effect 0.0665 --ci-low -0.9974 --ci-high 1.1258 --probability-positive 0.5385 --classification unresolved_below_power --plain-summary "Same challenger as model_logit inside the served four-term pick-probability fit, LOSO-refit 2020-2025 (n=1503): near-flat, 860-643 vs 859-644, decisive games 29-28 vs 28-29 on 57 games."`
3. Do not promote or change the served model from this -- margin MAE
   resolving worse at the opener, plus accuracy leaning (unresolved)
   negative at the opener while going flat once wrapped in the four-term
   fit, is not evidence FOR season-recency weighting; it is a second,
   independently-selected data point in the same direction as MOD-14's own
   opener-grade finding. A promotion bar is not a decision bar either way,
   and this is a below-power screen on 6 outer folds.

## Open

None -- this is a bounded research question; no scenario producer, no
dashboard change, no operational job.

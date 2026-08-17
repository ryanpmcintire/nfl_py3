# MOD-16 conditional margin variance — CFB screen predeclaration

Predeclared: 2026-08-17 (US), before any run in which the conditional
distribution below was scored against game outcomes. Frozen constants are
mirrored in `src/nfl_ats/margin_variance.py`; the runner records
`hypothesis_frozen_before_scoring: true`. Results are appended below the
predeclaration and never edit it.

## Hypothesis

The active margin models center one pooled out-of-time residual sample on
every game's predicted margin, giving every game the same distribution
shape. MOD-16 hypothesizes that pregame context predicts per-game residual
*scale* — bigger mismatches, higher totals, faster teams, and thin
early-season team states produce wider margin distributions — and that
conditioning on it yields better-calibrated cover probabilities. Per the
roadmap, the screen runs on the CFB benchmark first: its 8,933-game clean
core resolves calibration differences the 2,075-game NFL evaluation cannot,
and CFB outcomes are not burdened by the NFL 2018–2025 look ledger.

The forced picks cannot change: the mean model is byte-identical to the
frozen XLG-03 `market_residual` arm, and a symmetric rescaling of the
residual sample around the same center leaves which side is more likely
untouched. This experiment is purely about probability calibration — which
is what MOD-16's definition of done demands ("accepted only if held-out
cover/push/loss calibration beats the pooled baseline").

## Frozen recipe

- **Mean/center**: `fit_cfb_residual_model`, unchanged (Ridge alpha 10,
  frozen XLG-03 feature contract, trailing-20% out-of-time residual pool).
- **Variance model**: on the same chronological trailing-20% holdout
  (identical sort keys, split fraction, estimator recipe, and seed 42), a
  Ridge (alpha 10, standardize+impute pipeline) predicts
  `log(|out-of-time residual| + 1)` from eight frozen columns:
  `abs_spread_line` (derived |spread_line|), `total_line`,
  `home/away_off_plays_per_game`, `home/away_team_games`,
  `week_sin`, `week_cos`.
- **Per-game scale**: ratio `r = (exp(prediction) − 1) / s_bar`, where
  `s_bar = exp(mean(log(|residual|+1))) − 1` on the holdout, clipped to the
  frozen band **[2/3, 3/2]** (symmetric in log space). Predictive sample =
  `center + pooled_residuals × r`; `r = 1` recovers the pooled arm exactly.
- **Walk-forward**: the XLG-03 protocol verbatim (2006–2025, ≥500
  strictly-earlier training games, identical weeks across arms), three
  arms: `market`, `market_residual` (pooled), `market_residual_variance`.

## Frozen decision rule

Primary metric: paired per-game **cover log-loss improvement** of the
variance arm over the pooled arm on the clean core (2012–2019, 2021–2025),
via `paired_feature_comparisons` (2,000 samples, seed 20260817). The
candidate **clears** only if the week-blocked 95% interval excludes zero
from below (lower bound > 0). Brier improvement, season-blocked intervals,
ECE, and 50/80% interval coverage are reported as coherence checks and do
not override the rule in either direction. One run; no feature, clip-band,
target, or split retuning after seeing results — any variant is a new
predeclaration. Consequences:

- **Clears** → an NFL MOD-16 candidate may be predeclared (a NEW frozen
  document that explicitly acknowledges the ~130–150-look 2018–2025
  ledger), and conditional variance becomes admissible input for BET-04
  probability haircuts and the MOD-05 distribution work.
- **Does not clear** → recorded as-is; the pooled distribution stands and
  successors (e.g. distributional boosting, MOD-08) need new
  predeclarations.

## Declared limitations

1. The scale model conditions the *width* only; the residual shape (skew,
   key-number mass) stays pooled. A true conditional density is MOD-08's
   territory.
2. `abs_spread_line` and `total_line` are close-proxy market values; the
   variance features inherit the XLG-03 market table's timing semantics.
3. The clip band bounds how wrong the scale model can be, at the cost of
   capping genuine extreme-dispersion games.
4. Push probabilities in CFB are near zero for both arms (continuous
   centers), so "push calibration" carries almost no weight in this screen.

---

## Results (run 2026-08-17, artifact `artifacts/cfb_variance_experiments/20260817T112146Z`)

The scale model behaved as intended mechanically: per-game ratios averaged
1.008 with real spread (p10 0.905, p90 1.115) and fewer than 1% of games at
either clip bound, and the forced picks and margin errors were identical to
the pooled arm game-for-game. The calibration verdict, clean core, 8,933
paired games, candidate minus baseline (positive = candidate better):

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% |
|---|---|---|---|
| Cover log-loss improvement (primary) | **−0.000335** | [−0.000558, −0.000122] | [−0.000607, −0.000075] |
| Brier improvement | −0.000148 | [−0.000239, −0.000058] | [−0.000263, −0.000037] |
| Accuracy improvement | +0.000336 | [−0.0011, +0.0018] | [−0.0008, +0.0014] |

80% interval coverage also drifted slightly further from nominal (0.7968 vs
the pooled 0.7996 on the clean core); 50% coverage was equivalent.

### Verdict: **not cleared** (frozen rule), recorded as a real negative

Conditioning the residual scale on mismatch size, total, pace, and
early-season sample thinness made cover probabilities resolvably *worse*,
not better — the CFB sample is large enough to resolve even this
3-thousandths-of-a-nat degradation, which is exactly why the screen ran
here first. The honest mechanistic reading: the pooled out-of-time residual
distribution is already close to correctly calibrated (its 50%/80%
intervals cover at almost exactly nominal rates), so a scale model fit on
only the trailing-20% holdout adds estimation noise with no exploitable
heteroskedasticity signal at these features' resolution.

Consequences, per the predeclaration:

- The pooled residual distribution **stands** for CFB and, by the declared
  screening logic, an NFL MOD-16 candidate built from these same
  market/pace/experience features is not admissible. An NFL variant would
  need genuinely NFL-only variance features (QB experience/backup status,
  weather) and a new frozen predeclaration acknowledging the 2018–2025
  look ledger — recorded as possible but deprioritized by this result.
- No retuning of the feature list, clip band, target, or split on these
  outcomes. Distributional successors (MOD-08 quantile/NGBoost-style, or
  MOD-05's joint score/total) are new predeclarations.

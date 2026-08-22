# Shrunk overlay weights: ridge-logistic over flip indicators

Structural de-overfit of the discrete overlay-subset selection
(`scripts/overlay_subset_composition.py`, 127 correlated subsets; best subset
+2.0625 pts full-slate, selection-inflated). The split-half holdout measured
the selection shrinkage at OLS slope 0.6356, Spearman rho 0.7207
(artifacts/overlay_selection_holdout/20260821T195512Z/result.json — read).
This experiment replaces max-hunting with structure: each game's seven binary
flip indicators (six prospective overlays + reconstructed player-arrests
back-side policy) become FEATURES in an L2-regularized logistic model of
baseline pick correctness; the weights are continuous, shrunk, and
cross-validated rather than chosen by max.

Artifact: `artifacts/shrunk_overlay_weights/20260822T035633Z/result.json`
(all numbers below measured from it this session). Script:
`scripts/shrunk_overlay_weights.py`. Baseline: the frozen opener archive
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet` (sha256
f2b439ba…), 1,503 scored games 2020-2025 under the production probability
rule; unflipped baseline accuracy 53.3599%. No rotation-registry window is
spent: attribution and cross-validation only on already-scored data.

## Method

- Design matrix per scored game: seven binary overlay-fire indicators;
  intercept via sklearn's unpenalized `fit_intercept`. Outcome: baseline pick
  correct (production probability rule).
- Ridge-logistic (`LogisticRegression(C=1/alpha, solver=lbfgs)`).
- **Alpha selected by leave-one-season-out CV on POOLED held-out LOG LOSS,
  declared before any accuracy number existed** (predeclared in the script
  docstring). Grid {3, 10, 30, 100, 300}:

| alpha | pooled LOSO log loss |
|------:|---------------------:|
| 3     | 0.692127             |
| 10    | 0.691631             |
| 30    | 0.691171             |
| **100** | **0.691003 (chosen)** |
| 300   | 0.691032             |

- Deployed policy: flip when predicted P(baseline correct) < 0.5 given the
  fired indicators (equivalently a weighted vote crossing threshold).

## Weights (full-sample fit, alpha = 100)

| indicator | weight (logit units) |
|---|---:|
| intercept | +0.1742 |
| spread_gap_zone_fade | -0.0871 |
| division_revenge_tilt | -0.0788 |
| coach_fade | -0.0696 |
| injury_value_lost_tilt | -0.0424 |
| player_arrests_back_side | -0.0354 |
| surface_switch_tilt | +0.0064 |
| backup_qb_fade | +0.00002 |

Negative weights mean "indicator firing predicts baseline pick wrong" — the
direction the discrete overlays assume. Every magnitude is small and shrunk;
the largest is less than a tenth of a logit unit.

## Two estimates: attribution upper bound vs nested deployable

| arm | games | flips | delta vs baseline (pts) | week-blocked 95% interval | P+ (week) |
|---|---:|---:|---:|---|---:|
| Best discrete subset (selection-inflated reference) | 1,503 | 427 | +2.0625 | [-0.6597, +4.7494] | 0.9272 |
| Incumbent production chain (coach fade -> arrests) | 1,503 | 107+24 | +0.7984 | [-0.6694, +2.3057] | 0.8421 |
| Shrunk weights — IN-SAMPLE attribution (UPPER BOUND) | 1,503 | 15 | +0.0665 | [-0.4587, +0.5925] | 0.5508 |
| Shrunk weights — walk-forward NESTED (DEPLOYABLE) | 1,283 | 3 | -0.0779 | [-0.3808, +0.1575] | 0.1797 |

Season-blocked reads (same arms, order as above): P+ = 0.9189 / 0.7918 /
0.5916 / 0.0000; the nested season-blocked interval is [-0.2308, 0.0000].
Bootstrap: week- and season-blocked paired, 20,000 samples, seed 20260822.
The fast blocked bootstrap was re-verified exactly equivalent to
`nfl_ats.clv.week_blocked_bootstrap` on the walk-forward column (measured,
`equivalence_check_vs_nfl_ats_week_blocked_bootstrap: true`).

**The nested figure is the honest one.** The attribution row refits on the
same 1,503 games it is scored on, so it is an upper bound; even that bound
(+0.07 pts, P+ 0.55) sits far below the discrete best subset's +2.06 pts —
the regularized structure cannot reproduce max-hunting's selection inflation,
which is the point. The deployable expectation is the expanding-window
walk-forward (train strictly prior seasons, predict next; season 2020 has no
prior training data and is excluded): out-of-fold picks flip only 3 of 1,283
games (all in the 2024 fold), scoring 53.4684% vs the unflipped baseline's
53.5464% on the same covered games — -0.0779 pts, week-blocked P+ 0.1797.

Reading the walk-forward honestly, per the binding taxonomy:

- The week-blocked interval [-0.3808, +0.1575] crosses zero. That is the
  EXPECTED shape for a real-but-small signal at ~2-point resolution and is
  NOT grounds for rejection. Report `probability_positive` = 0.180, never
  "contains zero".
- It is not `refuted_mechanism`: the sign is not resolved (the interval's
  upper end is above zero, so `wrong_sign_resolved` is inadmissible), and no
  split-half reliability of the shrunk-policy trait was measured here.
- No positive control was run. Classification: **category 3,
  `unresolved_below_power`**, recorded below.

What IS settled by construction (inferred from the mechanism, not an
interval): at alpha = 100 with these seven indicators, the fitted decision
boundary almost never crosses 0.5 out-of-fold — 3 flips in five seasons. The
shrunk policy is operationally "keep the baseline card" far more often than
it is a competing pick rule; its deployable expectation is statistically
indistinguishable from the unflipped baseline on this archive.

## Record line

Run (do NOT paste into registry JSONs by hand):

```powershell
nfl-ats weak-signals record `
  --name shrunk_overlay_policy_walkforward `
  --description "Ridge-logistic (alpha=100, LOSO log-loss CV) over seven overlay flip indicators; flip when predicted P(baseline correct)<0.5; expanding-window walk-forward out-of-fold picks vs unflipped opener baseline" `
  --source artifacts/shrunk_overlay_weights/20260822T035633Z/result.json `
  --effect -0.07794 --effect-units accuracy_points `
  --classification unresolved_below_power --league nfl `
  --season-start 2021 --season-end 2025 `
  --standard-error 0.13434 --interval-low -0.38081 --interval-high 0.15748 `
  --probability-positive 0.17975 --sample-games 1283 --sample-blocks 90 `
  --classification-evidence "week-blocked interval crosses zero (P+ 0.180); not refuted_mechanism because the whole interval does not sit below zero and no split-half reliability was measured; no positive control run; category 3 unresolved" `
  --notes "Nested walk-forward is the deployable expectation; in-sample attribution upper bound +0.0665 pts (P+ 0.551); season-blocked [-0.2308, 0.0000] P+ 0.0000; bootstrap seed 20260822, 20k samples"
```

The same line flattened to one argument-per-line form is in
`registry/experiments/shrunk-overlay-weights/record_lines.txt`.

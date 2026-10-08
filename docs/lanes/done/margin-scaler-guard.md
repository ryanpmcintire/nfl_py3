# margin-scaler-guard

## Goal
Stop a float-noise column (std ~1e-18) from being standardised to ~1e16 in margin.fit_margin_model's 80/20 split (2020 Week 6 blowup, mod23 Unit 5b).

## State
- Guard added: FloatNoiseGuardedScaler in src/nfl_ats/margin.py replaces StandardScaler in make_margin_estimator (both ridge pipelines). A column whose fit std <= np.finfo(float).eps * max|finite fit value| (matrix scale) gets scale 1, as sklearn does for exact zeros.
- Matrix scale is used because a noise column's own magnitude is the noise.
- Audit and equality scripts: scratchpad audit.py (before.pkl pre-guard, after.pkl post-guard).

## Tried
- Audit, measured (146 weak_stack market_residual refits 2018 W1..2026 W4 plus 2026 W5, 292 scaler fits): min nonzero column std 0.00725; noise-level columns (0<std<1e-10) in 0 fits; held-out/in-fit max abs standardised ratio max 1.20, 0 fits >100; max held-out |z| 15.0. Opener evaluation artifacts/margins/20261008T210951Z market_residual: 2191 rows, 0 with p<=.01 or >=.99 (cover, win, loss). Recompute before vs after guard: all 2206 rows x all numeric columns identical (max abs diff 0.0).
- ruff format/check clean, mypy src clean, tests/test_margin.py + test_discrete_margin_mapping.py 34 passed.
- Recompute of 2026 W5 differs from the stored served csv by up to 0.056 in cover p (sorted comparison); same before and after guard, so a data or feature-table drift, not the guard; not investigated.

## Next
- None for this lane; orchestrator may move it to done/ after commit.

## Open
- None.

Orchestrator 2026-10-08 (measured): synthetic 1e-18 noise column, held-out 0.05 -> |z| 2.3e17 plain vs 0.05 guarded; guard fires. Week 5 recompute vs margin_predictions/2026-week-05-20261008T211138Z mismatch: that artifact postdates the 17:07 ET feature write, so a feature change does not explain it; CORRECTED: it is the served forecast; the gap is the home-side offset (7.5-10 bucket, 2 games) and the discrete conditional read that the audit recompute omitted (measured). Open: compare its config to the audit recompute; extend guard to modeling.py:55 and totals.py:110.

Unit 3 (measured 2026-10-08): W5 mismatch explained, no defect. Served artifact 2026-week-05-20261008T211138Z IS active_ats_model weekly_forecast (same created_at); its config (weak_stack, market_residual, ridge 10, calibration none, gaussian_median base, min_train 500, model 7fc3c6be355d224d) equals the active model. Audit recompute omits two served layers: home_side_offset (margin diff max 0.7798 = bucket 7.5-10 offset, 2 games; vs served predicted_margin_uncorrected diff 0.0) and the discrete conditional read (cover p diff 0.0558 vs uncorrected, base_probability_policy discrete_conditional_non_push_v1). Guard moved to modeling.py (shared), used by margin, modeling.make_estimator (logistic), totals.make_totals_estimator. Totals/logistic inputs have 0 zero-variance and 0 noise columns on current data, so guard is precautionary. Before/after W5 totals (wave1+wave2, 15 games) and logistic cover probs identical (json equal). ruff, mypy src clean; 34 + test_served_total/test_tiebreaker pass.

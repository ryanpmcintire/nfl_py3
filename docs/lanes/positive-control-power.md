# Positive-control evaluator power

## Goal

Measure whether the LEAD59 evaluator can recover predeclared synthetic effects on its served population,
including the two fixed LEAD59 flag vectors.

## State

- The harness supports IID controls, fixed DPI and holding vectors, LOSO fits, null companions, and
  persisted per-fold optimizer diagnostics.
- **Measured:** the served 2020-2025 population has 1,503 eligible games, six seasons, and 107 weeks.
  The fixed vectors contain 38 DPI and 54 holding games.
- **Read:** harness and evaluator use the same population builder, four base terms, ridge, fold-local
  standardization, separately fitted fifth flag term, LOSO scoring, and `p >= 0.5` side rule.
- **Measured:** explicit `evaluator_season_week` resampling now matches the evaluator's hierarchy: sample
  seasons with replacement, then weeks with replacement within each sampled season.
- **Read:** generic flat season and week modes remain available with their existing defaults. The matched
  method requires the evaluator's 50 Newton updates and labels artifacts `season_then_week`.
- **Read:** detection covers the accuracy-difference interval only. The evaluator also reports Brier,
  log loss, reliability, season stability, and a decisive-game exact null.

## Predeclared matched design

- Family: the exact DPI-uncertain and holding-trait vectors defined by the LEAD59 evaluator.
- Population: the active-model matched-opener population from 2020-2025; pushes and ungraded games drop.
- Injection: add a coefficient times the globally standardized fixed flag to the full-population fitted
  base logit, draw synthetic outcomes, and refit base and augmented probabilities LOSO.
- Looks: two controls across coefficient zero and the seven-point grid
  `0.10,0.25,0.45,0.70,1.00,1.40,1.80`; 200 simulations and 400 bootstrap draws per cell.
- Detection rule: the clustered 95% interval for augmented-minus-base accuracy is wholly above zero.
  A zero-crossing interval never closes a signal, and one fitted probability selects every side.

## Tried

- **Measured:** the full matched study completed in 444.8 seconds. At coefficient 1.80, DPI detection
  was 38.5% [32.0, 45.4] at a mean +2.350 accuracy points; holding was 78.0% [71.8, 83.2] at +3.126.
  Intervals describe detection-rate Monte Carlo uncertainty, not uncertainty in the mean effect.
- **Measured:** zero-coefficient false-positive rates were 0.5% [0.1, 2.8] for both controls. All 16 cells
  completed 200 simulations; neither reached 80% detection, so neither produced an 80% MDE.
- **Measured:** 44,800 fits completed with no solver failure or nonfinite prediction. Maximum final
  gradient norm was `6.15e-14`; maximum last step norm was `1.03e-12`.
- **Measured:** a 37-draw fixed-seed comparison exactly matched the evaluator's hierarchical bootstrap.
  Locked Ruff format and focused Ruff check passed.
- Full artifact: `artifacts/positive_control_power/20260927T034508Z/results.json`.
  [Study results and earlier experiments](done/positive-control-power-study-history.md) preserve all
  16 cells and limitations; the [original record](done/positive-control-power-history.md) is byte-exact.

## Next

Historical input pinning and the full replay are complete. Before using this power estimate to close
a research line, resolve the 400-versus-20,000 bootstrap-draw precision difference for the relevant
effect size. Keep the synthetic accuracy endpoint and sparse-control limitations explicit.

## Open

- The synthetic generator uses a full-population fitted base and globally standardized flags. It does
  not estimate a fold-local natural effect. Flags are sparse and season-concentrated.
- Finite precision remains: 200 simulations and 400 draws per cell. Diagnostics cannot prove general
  convergence or identification. Power covers accuracy only, not calibration or all atlas looks.
- **Inferred:** the earlier flat-bootstrap MDEs do not establish power for this hierarchical evaluator.
  No result promotes or closes a signal. AGENTS.md requires an admissible closing ground; these results
  do not establish one, and unresolved signals remain open.

**Measured:** Historical-input audit matched all 1,503 keys, outcomes, three non-model fit inputs, and
both fixed flags. Base model logits differ on 1,484 rows (maximum absolute difference 0.0183285361):
historical model `d5da2c0670e17eba`, current model `284a38bf00c29c53`. The completed study is a
current-active-model analogue. **Read:** LEAD59 declares 20,000 bootstrap draws; 5,000 is atlas-only.
The 400-draw run does not match that precision. Full comparison is retained in the study history.

**Measured:** The historical-input replay completed all 16 cells and 44,800 fits without numerical
failures in 462.9 seconds. At coefficient 1.80, DPI detection was 38.5% [32.0, 45.4] at +2.349
accuracy points; holding was 78.0% [71.8, 83.2] at +3.127. Both null false-positive rates were
0.5% [0.1, 2.8]. Neither reached 80% detection in the grid. Full results and pinned hashes:
`artifacts/positive_control_power/20260927T042645Z/results.json` and the study history above.

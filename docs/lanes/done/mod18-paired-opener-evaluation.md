# MOD-18 paired opener evaluation

## Goal
Compare FanDuel and consensus opener probabilities on identical eligible games with completed-prior-season fitting, calibration, and joint uncertainty estimates.

## State
**Measured:** the authenticated producer completed with 1,153 matched rows per arm across 2021-2025, ten arm/fold records and 287 exclusion rows. Root verified every outcome against result minus line, every probability bound, all output hashes and prior-season training/lattice/offset provenance. FanDuel has 17 pushes; consensus has none in the input population. These are input counts, not final evaluation counts.
Artifacts: `artifacts/paired_opener_inputs/20260927-frozen-config`. **Measured:** the declared evaluation completed on 963 paired / 950 jointly decisive games. FanDuel 494-456 versus consensus 502-448; difference -0.8421 percentage points, season-bootstrap interval [-2.1921, +0.2137], probability_positive 0.10115. Recorded as unresolved_below_power via weak-signals record. Full baselines, calibration and look accounting: `docs/mod18_paired_opener_evaluation.md`. No card change.

## Tried
- Implemented `paired-opener-inputs` with authenticated feature/raw-source/line bindings, exact line-bound prices, discrete probabilities and frozen prior-season fits. Supported feature profiles are base and weak_stack.
- Repaired grading sign, target-season invariance to requested output subsets, and both-arm support validation before fitting. Root and independent review passed.
- **Measured:** the real configuration is weak_stack, market_residual ridge alpha 10, gaussian_median, minimum 500 training games. The initial default-ECDF preparation was stopped before evaluation and produced no eligible report.
- **Measured:** Ruff, mypy, all 1,645 repository tests and the existing CLI contract passed. No new tests were added.
- Evidence: `.tmp/resume-paired-inputs-frozen.log`, `.tmp/resume-paired-real-verification.json`; controlled runner `.tmp/mod18_run_evaluation.py` authenticates output hashes and preserves all six evaluation frames.

## Next
This evaluation unit is complete; the signal remains unresolved in the registry. Any new mechanism or quote-preparation optimization belongs in a fresh bounded lane. **Measured:** all 20,000 accuracy draws/order and probability_positive values match the saved reference exactly; maximum proper-score drift is 4.44e-16. Cached recomputation took 0.50 seconds. Final Ruff, mypy and all 1,645 tests passed; publication and card-ledger verification passed.

## Open
No pre-2020 authenticated lattice exists; 2021 supplies the first calibration history and evaluation uses 2022-2025. Zero-crossing intervals do not close a signal; no serving promotion is implied. **Measured:** preparation took about 13 minutes. **Read:** repeated timestamp parsing/hashing and per-game full-frame scans are the bounded performance follow-up; this producer has no declared FND-13 runtime budget.

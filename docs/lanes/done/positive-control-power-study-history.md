# Positive-control power study history

This record preserves the detailed IID, fixed-vector, null, and diagnostic results moved from the active
lane. The original byte-exact development record remains in
[`positive-control-power-history.md`](positive-control-power-history.md).

## IID controls

- The completed artifact is `artifacts/positive_control_power/20260927T021328Z/results.json`; the log is
  `.tmp/positive-control-full.log`. It used 1,503 games from 2020-2025, 107 week blocks, 200 simulations,
  400 bootstrap draws, 20 fit iterations, and grid `0.10,0.25,0.45,0.70,1.00,1.40,1.80`.
- Detection rates by injected coefficient were `0.090, 0.340, 0.525, 0.700, 0.805, 0.910, 0.970` for
  binary prevalence 0.03; `0.060, 0.420, 0.860, 0.990, 1.000, 1.000, 1.000` for prevalence 0.10; and
  `0.025, 0.425, 0.965, 1.000, 1.000, 1.000, 1.000` for prevalence 0.50.
- Interpolated 80% minimum detectable effects were 2.123, 3.276, and 5.121 accuracy points for
  prevalence 0.03, 0.10, and 0.50 respectively.
- The null artifact is `artifacts/positive_control_power/20260927T021558Z/results.json`. False-positive
  rates were 0.010, 0.010, and 0.015 for prevalence 0.03, 0.10, and 0.50 respectively.

## Fixed LEAD59 controls

- The completed artifact is `artifacts/positive_control_power/20260927T024008Z/results.json`; the log is
  `.tmp/positive-control-fixed-full.log`. Runtime was 184.1 seconds with the same population, seasons,
  blocks, simulations, draws, fit iterations, and seven-point grid as the IID study.
- DPI uncertain type had detection rates `0.085, 0.370, 0.620, 0.745, 0.825, 0.880, 0.945`; its
  interpolated 80% MDE was 2.024 accuracy points.
- Holding trait had detection rates `0.065, 0.240, 0.585, 0.750, 0.885, 0.930, 1.000`; its extrapolated
  80% MDE was 2.333 accuracy points. The MDE has no simulation interval.
- The null companion is `artifacts/positive_control_power/20260927T024051Z/results.json`; its log is
  `.tmp/positive-control-fixed-null.log`, and runtime was 30.5 seconds. False-positive rates were 0.005
  for DPI uncertain type and 0.020 for holding trait.
- The fixed vectors contained 38 DPI and 54 holding games among 1,503 eligible games. Their prevalence
  was 0.0253 and 0.0359. The artifact persists the population-relative vector hashes.

## Optimizer diagnostics

- The bounded diagnostic smoke artifact is
  `artifacts/positive_control_power/20260927T030218Z/results.json`; its log is
  `.tmp/positive-control-diagnostics-postchange.log`.
- All 112 injected fold fits and seven null fold fits were finite and successful. No fit used a fallback,
  and none reported a failure or nonfinite result. Maximum final gradient norm was `2.76e-14`; maximum
  final step norm was `1.88e-13`.
- Removing only the newly persisted diagnostic fields made the post-change smoke artifact exactly equal
  to the pre-change artifact. The comparison is saved in `.tmp/positive-control-diagnostics-compare.log`.
- These figures establish stationarity for the smoke inputs after 20 Newton updates. They do not prove
  general convergence, optimizer identification, or equality to the evaluator's 50-update calculation.

## Evaluator-matched fixed-control study

**Measured:** `artifacts/positive_control_power/20260927T034508Z/results.json` completed in 444.8 seconds.
The declared 16 cells used 200 synthetic simulations, 400 bootstrap draws, and 50 Newton updates.
Each bootstrap samples seasons, then weeks within sampled seasons. Both probabilities are refitted LOSO.
Population: 1,503 eligible games, six seasons, 107 season-week blocks; 38 DPI and 54 holding flags.

| Control | Injected coefficient | Mean accuracy effect (points) | Detection rate, Wilson 95% interval |
| --- | ---: | ---: | ---: |
| DPI | 0.00 | -0.094 | 0.5% [0.1, 2.8] |
| DPI | 0.10 | +0.206 | 2.5% [1.1, 5.7] |
| DPI | 0.25 | +1.024 | 5.5% [3.1, 9.6] |
| DPI | 0.45 | +1.688 | 22.0% [16.8, 28.2] |
| DPI | 0.70 | +1.932 | 19.5% [14.6, 25.5] |
| DPI | 1.00 | +2.066 | 20.5% [15.5, 26.6] |
| DPI | 1.40 | +2.182 | 23.5% [18.2, 29.8] |
| DPI | 1.80 | +2.350 | 38.5% [32.0, 45.4] |
| Holding | 0.00 | -0.077 | 0.5% [0.1, 2.8] |
| Holding | 0.10 | +0.285 | 1.0% [0.3, 3.6] |
| Holding | 0.25 | +0.948 | 11.0% [7.4, 16.1] |
| Holding | 0.45 | +1.674 | 28.0% [22.2, 34.6] |
| Holding | 0.70 | +2.194 | 37.5% [31.1, 44.4] |
| Holding | 1.00 | +2.570 | 54.0% [47.1, 60.8] |
| Holding | 1.40 | +2.832 | 63.5% [56.6, 69.9] |
| Holding | 1.80 | +3.126 | 78.0% [71.8, 83.2] |

**Measured:** neither control reaches 80% detection on this grid; both stored minimum detectable effects
are null. The zero-coefficient false-positive rate is 0.5% [0.1, 2.8] for both controls.
All 44,800 fits across 224 diagnostic groups completed with finite predictions and no solver failure.
Maximum final gradient norm is `6.15e-14`; maximum final step norm is `1.03e-12`.

**Inferred:** the prior flat-bootstrap MDEs do not describe this hierarchical evaluator. These matched
results do not establish sufficient power to close the actual LEAD59 signals under the AGENTS.md
positive-control closing rule. They also do not promote either signal.

Limits: 200 simulations and 400 draws provide finite Monte Carlo precision; the synthetic generator uses
a full-population fitted base and global flag standardization. The study measures accuracy detection,
not calibration, chronological retraining, or the atlas deletion family. Sparse, season-concentrated flags
and complete fold coefficients remain explicit in the artifact. No registry verdict was changed.


## Historical population identity audit

﻿
Scope: compare the current positive-control population with the saved LEAD59 fifth-term evaluator input at
`artifacts/lead59_type_trait_bins/20260923T205414Z/per_game.parquet`. The signal-atlas artifact is an
active-lineage cross-check only because its delete-one estimand is different.

## Exact matches

- **Measured:** Both inputs contain the same 1,503 unique graded game keys, with no key mismatch or
  duplicate. The sorted key SHA-256 is
  `e8930d4cf27fd00db846f2608b9ae0f18d5d3b61422b039b0baf0edfcf6e747c`.
- **Measured:** Season counts match exactly: 2020 220, 2021 236, 2022 248, 2023 266, 2024 266,
  and 2025 267. Both have 34 pushes and no ungraded rows before the 1,503-game scored population.
- **Measured:** `season`, `week`, and `home_covered` match for every key. Their canonical population hash
  is `2d404aa236b43c37d21bbad2f4e6510e64ee0f729cb4decc0b4e3dd1184a0db9` in both inputs.
- **Measured:** The three non-model `FIT_FEATURES` values and both fixed-control vectors match at tolerance
  1e-12. Their normalized canonical hash is
  `fdaf071cf61b38cd610f62c344e8e69d4ce81e4a55b59b47d568f6e2b6b51c2c` in both inputs.
- **Measured:** DPI has 38 positives and game-ID hash
  `14e73db1fd13c158fd124e86413b8cc5c4d86f3277be95cedd39282cd225ac05`; holding has 54 positives
  and hash `9926a30798ef048ba501102d47e3cd5a82691f99b61e677b3a3f70a949774607` in both inputs.

The population is built by `build_fit_population` in `scripts/lead59_type_trait_bins.py:203` and the current
harness in `scripts/positive_control_power.py:570`. The historical fixed vectors are formed at
`scripts/lead59_type_trait_bins.py:226-236`; `FIT_FEATURES` is declared at
`src/nfl_ats/pick_probability_fit.py:46`.

## Remaining lineage difference

- **Measured:** `model_logit` differs on 1,484 of 1,503 games, maximum absolute difference 0.0183285361.
  The complete selected-row hashes therefore differ: saved
  `06002e59b3a4a979e97129b526f6d5756d2cbef87b4068198c7109cbe09ee6d6`, current
  `02b6ea8f80e225e148a681e6b660e7813972cf0b9d8a95a348e5e3bc7f9b8418`.
- **Read:** The historical evaluator used active model `d5da2c0670e17eba` and opener evaluation
  `opener_evaluation/20260923T172849Z`; the harness builder resolves current active model
  `284a38bf00c29c53` and `opener_evaluation/20260926T174414Z`.
- **Measured:** The active atlas has the same keys, seasons, weeks, and outcomes as the historical input,
  but its model probability differs on 1,484 games, maximum absolute difference 0.0045375142. It sources
  `pick_probability/20260926T174417Z` under model `284a38bf00c29c53`.

**Inferred:** The completed power result is an exact population/outcome/control-vector match using the
current active model's base logit. It is not a byte-identical replay of the historical fifth-term evaluator.

## Bootstrap precision

- **Read:** The LEAD59 fifth-term evaluator declares 20,000 season-then-week bootstrap draws at
  `scripts/lead59_type_trait_bins.py:21,244-266`. The artifact points to the predeclaration at
  `docs/lanes/done/lead59-archive-battery.md:96-104`; the measured run record repeats 20,000 at lines 149-153.
- **Read:** The 5,000-draw value belongs to the separate active signal-atlas report at
  `artifacts/signal_atlas/20260926T174435138851Z/report.json:12`.

## Verification and next action

Command (exit 0):

`$env:UV_CACHE_DIR = Join-Path (Get-Location) '.tmp\uv-cache'; .\.tools\uv.exe run --no-sync python `
` .tmp\compare_positive_control_population.py *> .tmp\positive-control-population-comparison.log`

Machine-readable results are in `.tmp/positive-control-population-comparison.json`; the bounded script is
`.tmp/compare_positive_control_population.py`.

Next action: label the completed matched power study as the current-active-model analogue of the historical
LEAD59 fifth-term input; if historical calibration parity is decision-critical, add a pinned source-artifact
mode and verify it before another full study. No signal is closed or promoted by this audit.


## Historical-input evaluator replay, 2026-09-27

**Measured:** The new `--lead59-historical-input` option replayed the archived LEAD59
per-game inputs from `artifacts/lead59_type_trait_bins/20260923T205414Z` with the
season-then-week bootstrap and 50 fit iterations. The full command completed in
462.9 seconds with exit 0. Results are in
`artifacts/positive_control_power/20260927T042645Z/results.json`.

The command used `--population served_2020_2025 --lead59-fixed-controls
--lead59-historical-input artifacts/lead59_type_trait_bins/20260923T205414Z
--bootstrap-block evaluator_season_week --fit-iterations 50 --sims 200
--draws 400 --grid 0,.10,.25,.45,.70,1,1.4,1.8` with the locked environment.

**Measured:** All 1,503 archived observations, outcomes, base fit terms and the two
fixed flags were loaded from the pinned artifact. Input-file and canonical row
hashes are retained in `population_input`. The loader checks model provenance,
unique game keys, seasons, finite values, binary outcomes/flags, and consistency
between stored base logits and probabilities. A separate one-simulation replay,
Ruff format and focused Ruff check passed before the full run.

**Measured:** Every cell completed 200 simulations. All 44,800 fits completed;
solver failures, nonfinite predictions, nonfinite diagnostic values and linear
solve fallbacks were zero. Maximum final gradient norm was `5.93e-14`; maximum
last step norm was `1.45e-12`. The compact diagnostic aggregation is saved at
`.tmp/positive-control-historical-diagnostics.json`.

**Measured:** Detection rates below have Wilson 95% intervals for Monte Carlo
uncertainty across simulations. They are not intervals on the mean accuracy effect.

| Control | Injected coefficient | Mean effect (accuracy points) | Detection rate, Wilson 95% interval |
| --- | ---: | ---: | ---: |
| dpi_tilt_pass_heavy_favorite | 0.00 | -0.090818 | 0.5% [0.1, 2.8] |
| dpi_tilt_pass_heavy_favorite | 0.10 | +0.200599 | 2.5% [1.1, 5.7] |
| dpi_tilt_pass_heavy_favorite | 0.25 | +1.021623 | 5.5% [3.1, 9.6] |
| dpi_tilt_pass_heavy_favorite | 0.45 | +1.688623 | 22.0% [16.8, 28.2] |
| dpi_tilt_pass_heavy_favorite | 0.70 | +1.931138 | 19.5% [14.6, 25.5] |
| dpi_tilt_pass_heavy_favorite | 1.00 | +2.068530 | 20.5% [15.5, 26.6] |
| dpi_tilt_pass_heavy_favorite | 1.40 | +2.186959 | 24.0% [18.6, 30.4] |
| dpi_tilt_pass_heavy_favorite | 1.80 | +2.348636 | 38.5% [32.0, 45.4] |
| holding_tilt_run_heavy | 0.00 | -0.072522 | 0.5% [0.1, 2.8] |
| holding_tilt_run_heavy | 0.10 | +0.290419 | 1.0% [0.3, 3.6] |
| holding_tilt_run_heavy | 0.25 | +0.943114 | 11.5% [7.8, 16.7] |
| holding_tilt_run_heavy | 0.45 | +1.670659 | 28.0% [22.2, 34.6] |
| holding_tilt_run_heavy | 0.70 | +2.190286 | 37.5% [31.1, 44.4] |
| holding_tilt_run_heavy | 1.00 | +2.568862 | 53.5% [46.6, 60.3] |
| holding_tilt_run_heavy | 1.40 | +2.832668 | 64.0% [57.1, 70.3] |
| holding_tilt_run_heavy | 1.80 | +3.126747 | 78.0% [71.8, 83.2] |

**Inferred:** Neither control reached 80% detection anywhere in the tested grid;
the reported 80% MDE remains undefined, with no extrapolation. These results
closely track the current-active-model analogue, but use the original base logits.
The replay still uses 400 bootstrap draws per simulation rather than LEAD59's
20,000; it establishes historical input parity, not identical bootstrap precision.
The synthetic generator, sparse flags and accuracy-only endpoint retain the
limitations above. Under AGENTS.md's admissible-closing-ground rule, this run
closes or promotes no signal. The recorded weak signals remain unresolved.

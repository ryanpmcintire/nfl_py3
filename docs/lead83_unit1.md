# LEAD-83 unit 1: empirical-Bayes move features

**Measured:** the real command completed the roadmap's variance-feature unit. No game-outcome columns were loaded; no probability fit or score was computed.

## Population and clocks

**Measured:** 1,311 non-push games across 2020-2025, plus 33 source-complete pushes retained separately. The four-term source has 1,503 rows; its opener parent has 1,537, including 34 pushes. Declared population match: True.

| Season | Non-push games | Extra pushes | Median common books | Zero jackknife variance |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 191 | 7 | 12.0 | 162 |
| 2021 | 213 | 3 | 16.0 | 191 |
| 2022 | 210 | 6 | 19.0 | 192 |
| 2023 | 233 | 6 | 15.0 | 203 |
| 2024 | 232 | 6 | 9.0 | 184 |
| 2025 | 232 | 5 | 11.0 | 185 |

**Measured:** target season/week, game/team identity, archive hashes, Tuesday/Sunday calendar dates, paired book outcomes, observed/book/market timestamps, and both archived and schedule kickoff bounds are checked. Tuesday captures precede noon; they establish available anchors, not noon-capture evidence. The later snapshot is before Sunday 12:30. Final closing quotes and game-outcome columns are never loaded.

- **Measured:** tue_open: 131 files, 145,340 quote rows.
- **Measured:** sun_early_close: 131 files, 171,190 quote rows.
- **Measured:** 256 target-population spread rows excluded by clock/finite-value checks.

## Frozen construction and training-only variance

Each book contributes its Sunday minus Tuesday home spread; positive means movement toward home. The game measurement is the median. With n books and leave-one-book-out medians m_i, v=(n-1)/n * sum((m_i-mean(m_i))^2). Training tau^2=max(0, sample variance of game medians - mean(v)); shrink weight=tau^2/(tau^2+v). If both terms are zero, weight is zero. The original move remains available beside the shrunk move. The variance estimate retains pushes; conditional cover fitting excludes them. No ranking, bands, or outcomes choose books or shrinkage. Books are measurement replicates; games remain the sample units.

| Outer | Fit through | Fit / tune / calibrate / outer games | Variance fit including pushes | Move variance | Mean v | tau^2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2023 | 2020 | 191 / 213 / 210 / 233 | 198 | 2.37080353 | 0.05603766 | 2.31476587 |
| 2024 | 2021 | 404 / 210 / 233 / 232 | 414 | 2.76646957 | 0.05336014 | 2.71310943 |
| 2025 | 2022 | 614 / 233 / 232 / 232 | 630 | 2.42831292 | 0.05151967 | 2.37679326 |

## Replay handoff

**Measured:** 0 outcome looks executed; 501 remain predeclared for the cached replay. Decisive-game record, IS/OOS scores and gap, probability coefficients, season/week-block intervals, reliability, and probability_positive are not estimated in this feature unit. The variance estimates above are nuisance feature parameters, not cover-probability coefficients.

**Measured:** the parent opener artifact exposes 0 per-row training/cutoff columns. The frozen cache does not certify Protocol C cutoffs. Protocol C fit or score. Cached replay must preserve fit through Y-3, tuning on Y-2, calibration on Y-1, and outer Y; do not substitute unrestricted LOSO or missing-move zeros.

**Read:** src/nfl_ats/clv.py:2177-2189 fits the opener parent weekly, using all completed games before the target week. **Inferred:** those upstream fits do not respect a fixed training cutoff through outer season Y-3. A fold-specific upstream probability cache is required before this protocol can be scored. The next unit must reconstruct those folds from cached inputs without a full-history rebuild.

**Inferred:** this feature unit says nothing about predictive gain. No closure, promotion, or serving decision is made. Zero crossing cannot close a signal, and one fitted calibrated discrete probability must select the side.

**Measured:** feature rows, matched-book rows, fold roles, input hashes, and inventory are saved under tests/scratch/codex/lead83_unit1/. No prediction-row dump is stored in docs.

Command: .tools/uv.exe run --no-sync --no-cache python scripts/lead83_unit1.py. No-cache avoids the inaccessible shared uv cache; no dependencies are installed.

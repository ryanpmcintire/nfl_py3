# LEAD-70 protocol saved before outcomes

## Protocol (verbatim ROADMAP row; whitespace wrapped)

| LEAD-70 | ⬜ | Cross-model disagreement as a confidence modifier (rank 5 of 8) | **Added 2026-09-29 (unmeasured).** Mechanism: the served model is one ridge fit; where
independently built challengers with preserved predictions (105 artifacts keep `predictions.parquet`, ENG-46) disagree in margin, the served margin is estimated less reliably and
its logit is more likely to be the noise MKT-17 found in its slope; where they agree, the logit carries more signal. Feature: standard deviation of the preserved challengers'
predicted margins for the game, from models whose fold cutoff precedes the game. Predeclared: the slope of the model logit interacts with one standardised disagreement term
(`a*logit*(1 + d*z)`), LOSO by season 2020-2025, base = served four-term model, 2 looks, metrics log loss, Brier, accuracy, reliability by disagreement tercile with counts, IS-OOS
gap. The disagreement never selects a side; it scales the served probability. Inventory step first (which artifacts cover all six seasons out-of-fold); if fewer than three qualify
the row records a data gap, not a result. Checked, not duplicated: MOD-20 (adds signal families as terms, no ensemble-dispersion term), MKT-17..19 (single model),
`independent-combination-validation` lane (three-week swing diagnosis). About 20 tool calls. |

## Tried
Read the assigned row and ideation context; no outcomes computed.
Before outcomes: use the active discrete four-term per-game population; exclude pushes.
Use margin-valued `fair_margin`/`market_residual` NFL streams, grouped by declared model
configuration; take the newest six-season, strictly pregame-cutoff stream per configuration.
Exclude the served configuration and exact duplicate predictions; use their common games.
Fit the existing four-term form and its one logit-by-disagreement interaction with the
existing fixed ridge, training-only standardisers and terciles, LOSO 2020-2025; full-fit IS
is descriptive. No tuning or outcome-based source selection. Fixed 4,000 season bootstrap
draws (seed 20260929); positive means improvement, bootstrap ties contribute 0.5 to P+.
Report the two fitted model forms, both fixed market/model-only references, three diagnostic
terciles, six folds, all three metrics, paired intervals, decisive record, and all reporting cells.


The declaration above was copied into the lane before any outcome was computed.
Moved here at handoff to keep the lane under one page; the protocol was not revised.

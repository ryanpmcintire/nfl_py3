# ENG-60

## Goal
Neutral-site and international orientation: home flag and home-field term for games without a true home team (ROADMAP ENG-60).

## State
Unit 1 (2026-11-03): scripts/eng60_neutral_audit.py, artifacts/eng60/audit.log. Orientation, grader and ledger clean; Elo home edge zeroed at neutral (features.py:481-482); neutral_site built at features.py:700. The BUF_SF / JAX_CHI sign question is closed by ENG-59 (line moves, not sign errors).

Unit 2, 2026 check on 2026_10_NE_DET, 2026-11-14T21:16Z. Scripts tests/scratch/eng60/neutral_2026.py and neutral_2026_sensitivity.py; output artifacts/eng60/neutral_2026.json. Nothing served was changed; nothing recorded in the registry.
- Trace (measured unless marked): schedule snapshot 20261111T010519Z has location Neutral, home DET, away NE, spread_line +2.5 (DET favoured), rest 7 and 7, Munich. Splash 2026_week10_20261110_2005 has home DET, home_spread 2.5, NE +2.5. Market quotes: 704 rows, three providers, all home DET, last home spread 2.5. Feature row: neutral_site 1, rest_diff 0, elo_diff 38.4 with no home edge added. The served 90-feature set has no travel, time-zone or venue input. Home-side offset for this game 0.0 (bucket 0-3). Card: NE +2.5, 53.5%.
- Served path reproduced exactly: model residual -4.256 points for DET, model cover chance DET 0.4096; calibrated DET 0.4649, NE 0.5351 with line move -0.5 toward NE, half-point term +0.096 toward DET, no flags.
- Answer: no served input gives DET a home edge or treats the game as in Detroit. The opposite: the fitted neutral_site term is -1.51 points against the designated home team. With that term off for this game NE is 52.1%. A model made blind to the home label gives NE 51.2%. Calibration home constants are worth 0.13 points. NE stays the side in every case.
- The -1.51 comes mostly from seven 2026 neutral games (designated home 0-7 against the closing line, mean -9.2, exact p 0.016). Refit without them: term -0.50, NE 52.4%. 2009-2025: 61 games, 30-31, mean -0.30 (SE 1.88). All 68: 30-38, exact p 0.40.
- History: 79 neutral games 2009-2025 (61 REG, 17 SB, 1 WC). All 61 REG are kept in training with the flag set, none dropped; 68 of 4567 training rows. Four Toronto games are labelled Home. Opener record on 28 neutral games 2020-25: 14-14 (exact p 1.0), probability rule 11-17 (p 0.34); non-neutral 780-695.
- Latent defect, not served: graph_ratings.py:241-243 adds the fitted home-field intercept to schedule_predicted_margin at neutral rows (88 rows, mean 1.95; 0.22 here), and :140-143 fits neutral games as home games. The column is not in the weak_stack served set; removing it moves the served probability by 0.

## Tried
Novelty check; unit 1 audit; unit 2 trace, three counterfactuals, two sensitivities. Nearest: MKT-24, ENG-55.

## Next
1. Decide whether the neutral_site term should be fitted on international games separately from Super Bowls, and whether seven same-season games should carry two thirds of it; test leave-one-season-out at the opener before any change (features.py is pinned).
2. Repair graph_ratings.py:241-243 and :140-143 only if a profile that uses schedule_predicted_margin is ever served; it changes the feature table hash.
3. Optional registry row for designated-home ATS at neutral sites (unresolved_below_power). Check 2026 neutral weeks 1, 3, 5, 6, 7, 9 against the served board archive.

## Open
Audit only, no research verdict. The pin status of graph_ratings.py is inferred from the naming rule, not read from the guard. The 0-7 run is graded at the closing line, not the opener.

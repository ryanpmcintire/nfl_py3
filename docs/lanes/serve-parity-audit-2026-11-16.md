# Serve parity audit, 2026 Weeks 1-10 (2026-11-16)

## Goal
Compare each 2026 game's served input row (last margin-predict forecast before Sunday 17:00 UTC of its week) with the row the training builder now holds in data/processed/game_features_weak_stack.parquet; size per-column mismatches and their effect on the model logit and side; patch proven defects (not applied).

## State
Scripts and outputs in tests/scratch/serve_parity/: audit.py (column_diffs.csv, logit_effects.csv), attribute.py (group_attribution.csv), weather_patch_effect.py (weather_patch_effect.csv), weather_drop.patch (git apply --check OK; patched copy under weather/).
Scoring: score_outcome_week, market_residual, weak_stack, ridge alpha 10, current table for training rows, no home-side offset or discrete read (so served-row probabilities do not reproduce the card, max gap 0.146; deltas only). Served-stage delta = 0.308 x model-logit delta (model_logit coefficient, pick_probability/20261116T171555Z).
Measured, 150 games, 90 model columns, 31 columns differ:
- temp/wind: 45 games, Weeks 1-4; served NaN, rebuilt holds observed game weather. Weeks 5-11 NaN in both (feed stopped). Training 2023-25 has weather on 158-190 of 285 games a season; at serve every game is NaN, which the median imputer plus missing indicator reads like a dome game. Group swap: mean |logit| 0.029, max 0.071, 4 model-side changes (W1 CHI@CAR, CLE@JAX, DEN@KC, TB@CIN, all within 1.3 pts of 50%).
- Patch (drop temp, wind from SCHEDULE_FEATURES and context family): W1-10 mean |logit| 0.018, max 0.038, 3 model-side changes (W1 CLE@JAX, DEN@KC, W8 IND@JAX); served stage max 0.012 logit. Week 11: 1 of 13 rows has a spread, 0 changes.
- total_line: 94 games, served forecast-time total vs table total, mean |diff| 1.34; mean |logit| 0.0067, max 0.029, 0 sides. Not a builder defect (line moved).
- QB (diff_qb_*): 40 games, mean |logit| 0.0085, max 0.049, 0 sides. Out-starter and debut defects covered by tests/scratch/qb_input_repair/repair_ab.patch (not re-sized here).
- neutral_site/elo: 2026_05_PHI_JAX served Home, rebuilt Neutral (upstream schedule corrected later, inferred); logit 0.114, 0 sides.
- rest_diff: 0 mismatches in 150 games; training rows agree with gameday gaps on 4597 of 4613 (16 off: 2019 6, 2021 8, 2022 2). No serve defect.
- Other team-state columns: 12 games, max |logit| 0.0007.

## Tried
Full-row swap per week, group swap, weather-drop arm with monkeypatched FEATURE_SETS.

## Next
1. Orchestrator decides the weather fix: weather_drop.patch (parity by removal), or a pregame forecast weather feed for serve and training alike. Apply at a pinned change with refit and the usual identity check.
2. Optional: rebuild rest from gamedays for the 16 historical mismatches (training-only).

## Open
- Served-card side changes were not measured; card logits for 2026 are not stored per game in the forecast artifact. Served-stage shifts are at most 0.012 logit, so only a card within about 0.3 pts of 50% could change (inferred).
- Whether durable tests pin the context family columns was not checked.

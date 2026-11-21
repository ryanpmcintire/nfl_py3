# goal-55-backwards-2026-11-21

## Goal
Owner 2026-11-21: get to 55%+, starting backwards: what would it take. Done when each yardstick has its gap in wins and the levers that could close it are named and started.

## State (measured 2026-11-21 13:35 EST)
- Inputs: artifacts/opener_evaluation/20261121T171727Z/per_game.parquet (model a8ea1dadc06c2405, 1,503 non-push opener games 2020-25); `scripts/week_attribution.py --season 2026 --weeks 1-10`. Script: scratchpad backwards.py (not tracked).
- Model alone at the opener: 802-701 (53.36%). 55% = 827: +25 wins, about 4 a season. Seasons 49.6% (2023) to 55.6% (2024); 2021 and 2024 already at 55%.
- Confidence quintiles 52.5 / 55.7 / 49.2 / 56.3 / 53.2: the model's confident games are not its better games, so reweighting what it has cannot supply the 25 wins.
- Closing-line ceiling: following the opener-to-close move where the line moved (75% of games) and the model elsewhere gives 54.76% (823); moves of 1+ point followed go 376-277 (57.6%). Model agrees with the eventual move on 57.2% of moved games (371-277); where it fights the move it goes 232-253. Perfect hindsight of the close is 4 wins short of 55% at the opener.
- Live 2026 card: 77-73 (51.3%) through Week 10. Weeks 3-10 card 54-64 v current formula replay 62-56 v calibrated model 65-53; the 16 games where the card differed from its formula went 4-12 (zero such games in Week 10; fidelity check live). Flag set was chosen partly on 2026 results, so the replay flatters (inferred).
- Rest of season about 122 games: 55% for the season needs 73-49 (60%); 55% from here needs 68-54.
- Pool-line replay held out (goal-65 lane): 836-667 (55.6%); pooled stack 852-651 but its gain is the coach flag, which went 11-21 live.

## Tried
- Gaussian correlation mapping of residual to accuracy predicts 51.8% against 53.4% measured; not usable for sizing, dropped.

## Next
1. Sunday 11-22: confirm zero card-v-formula mismatches after the 12:45 refresh (backlog-2026-11-21.md Next 2).
2. Model alone: 55% needs information the closing market lacks. Target the 485 games where the model fights the eventual move (232-253): what, knowable Tuesday, separates its winning fights from losing ones. One declared look, matched null, LOSO.

## Open
- None.

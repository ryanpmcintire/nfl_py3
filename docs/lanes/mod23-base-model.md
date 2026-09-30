# MOD-23 - improve the model by itself

## Goal
Raise the model-alone record against the opener on the fair test (each season
predicted only from earlier seasons). Baseline: 802-701, 53.36% on 1,503
decisive 2020-2025 games (`artifacts/opener_evaluation/20260929T192743Z`).
Done = a change that beats that baseline out of sample, or each unit recorded.

## State
- 2026-09-30: row MOD-23 added to ROADMAP.md; lane opened.
- Base model: ridge (alpha 10) on 88 `football_weak_stack` inputs, target
  margin minus line. Families: offense 21, defense 18, continuity 9, bias 9,
  context 7, injuries 7, results 6, player_qb 5, elo 2, experience 2,
  player_values 2.

## Tried (earlier, on the model alone or older profiles)
- Recency weighting flat; 2011-2025 training no better; trees worse; market
  ratings (MOD-21) wrong sign; graph ratings never selected.
- August 2026 nested family selection on an older profile: 52.47% vs 50.88%
  fixed, Brier worse, 2025 49.8%. Not redone on the current base.
- MOD-22 players-on-field was graded only as an add-on to the four-term card.

## Next
- Unit 1: nested walk-forward selection of feature families and ridge alpha
  (choose with seasons before Y, score Y), graded against the baseline above.
- Unit 2: re-grade MOD-22 constructions on the model-alone test.
- Unit 3: opponent-adjusted efficiency inputs.

## Open
- None.

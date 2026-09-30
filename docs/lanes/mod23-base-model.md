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

- Unit 1 DONE (docs/mod23_unit1.md, scripts/mod23_unit1.py, artifacts/mod23_unit1/): baseline reproduced 802-701. Nested families+alpha 786-717 (-1.06 pts, [-3.18, 0.98], P>0 0.15) but log loss 0.6928 vs 0.6969 and margin MSE better in 5/6 seasons; alpha-only 784-719 (-1.20, [-2.17, -0.13]). Recorded unresolved_below_power (two cells).

- Unit 2 DONE (docs/mod23_unit2.md, scripts/mod23_unit2.py, artifacts/mod23_unit2/): six MOD-22 arms on the base model alone, clock kickoff-24h evidenced reports. 792-798 wins vs 802; diffs -0.13 to -0.93 pts, all intervals cross zero, none beats baseline on LL/Brier. Recorded unresolved_below_power x6 (blocks 107 counted).

## Tried (earlier, on the model alone or older profiles)
- Recency weighting flat; 2011-2025 training no better; trees worse; market
  ratings (MOD-21) wrong sign; graph ratings never selected.
- August 2026 nested family selection on an older profile: 52.47% vs 50.88%
  fixed, Brier worse, 2025 49.8%. Not redone on the current base.
- MOD-22 players-on-field was graded only as an add-on to the four-term card.

## Next
- Unit 3: opponent-adjusted efficiency inputs.

## Open
- Fit gain (MSE, log loss) does not convert to picks; consider grading on a smaller-model probability, not a pick swap.
- Unit 2 QB arms train only on 2020+ (terms zero before); pre-2020 QB terms not built.
- Unit 1 sample-blocks in the registry cell (102) was estimated, not counted.

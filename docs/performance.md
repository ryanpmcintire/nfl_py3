# Performance contract

Research integrity includes computational integrity. An evaluator that cannot
finish inside a known budget encourages interrupted runs, missing uncertainty
reports, and ad hoc reductions in the evaluation protocol.

## Reference budgets

These are regression budgets on the local August 2026 development machine,
not promises for every computer:

| Workflow | Reference workload | Budget | August 2026 measurement |
|---|---|---:|---:|
| Outcome bootstrap | 2,075 games, five methods, 1,000 week draws plus 1,000 season draws | 5 seconds | 0.38 seconds inside the CLI run |
| Full outcome evaluator | 2018–2025, five methods, weekly refits, player-QB profile, 2,000 total bootstrap draws | 120 seconds | 53.25 seconds |
| Player feature build | 4,703 games, 310,475 snap rows, 76,784 injury rows | 120 seconds | 68 seconds |
| Evaluator sensitivity audit | 2018–2025 active profile, eight signal/permutation replicas, four effect sizes, 2,000 blocked draws | 180 seconds | 127.9 seconds |

Every `margin-backtest` artifact records `modeling_seconds`,
`uncertainty_seconds`, and `total_seconds` in `metadata.json`.

## Required design rules

1. Aggregate resampling statistics once per week or season. Never materialize a
   new pandas frame and rerun grouped metrics for every bootstrap draw.
2. Compile repeated player-lineup structures once. Do not perform tiny joins
   or dataframe scans for every feature and team-game.
3. Profile a representative season before launching a full-history evaluator.
   Set the command timeout from the measured runtime plus a reasonable buffer.
4. Preserve numerical equivalence when optimizing research code. Tests compare
   the vectorized bootstrap with the original resampling definition.
5. Add a structural regression test for the failure mode. The bootstrap test
   asserts that metric-summary calls depend on the number of methods, not the
   number of resamples.
6. A workflow that exceeds its reference budget must be profiled and fixed or
   explicitly documented before more experiments use it.
7. Fit each sensitivity signal and permutation stream once. Recover the
   declared 0/0.5/1/2-point counterfactuals algebraically from the joint Ridge
   targets and empirical residual components; never rerun the weekly evaluator
   independently for every effect size.

The sensitivity audit additionally fails unless it reconstructs the active
point predictions, empirical cover probabilities, and 1,080/2,075
classification count. Each artifact records `timing.total_seconds`; its runtime
is not evidence if those canaries fail.

## Why the optimized bootstrap is equivalent

Accuracy, Brier score, log loss, MAE, RMSE, and ROI can all be represented by
per-block sums and counts. Resampling complete blocks and summing those stored
components produces the same metric draws as concatenating the corresponding
game rows. The optimized implementation uses the same random block selections;
saved 500-draw intervals from the former implementation agree to floating-point
precision.

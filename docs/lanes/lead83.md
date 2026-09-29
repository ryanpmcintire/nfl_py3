# LEAD-83 — shrink noisy move measurements

## Goal
Replay noisy-book shrinkage beside the original move. Historical 2020–2025 OPENER
is the frozen pool-line proxy; the refitted served four-term recipe is the base.

## State
**Measured:** unit 2 complete: 697 outer games, candidate 362–335, base 361–336;
decisive 24–23. Brier improvement −0.000170 [−0.000690,+0.000176],
probability_positive=0.24715; accuracy +0.143 points [−0.996,+1.702], 0.4990.
**Inferred:** unresolved_below_power, pending serial records; no serving change.

## Tried
The pre-outcome amendment is preserved verbatim in docs/lead83_unit2.md and
tests/scratch/codex/lead83_unit2/protocol.md: fit through Y−3, tune Y−2, calibrate
Y−1; outer 2023/24/25; five arms; 501 looks. **Measured:** upstream fixed-cutoff
margin/discrete/four-term refits resolved the weekly-cache blocker. One calibrated
probability selected sides. All declared non-push rows were retained; pushes
remained in upstream distributions. Real replay exited 0 once; scoped Ruff and diff checks passed. All 16 command expansions parsed without calling handlers.
Candidate IS/OOS accuracy 58.81%/51.94%, gap −6.87 points; 2024 tune slope zero.
Only this packet's script/report/lane and scratch outputs changed; no registry,
Git mutation, publication, or new test. LEAD-84 is outside this packet.

## Record commands
Bash loop expands to 16 exact candidate-versus-base OOS commands (season panels
and pooled; no other baseline contrasts). Orchestrator runs serially; not run here.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --name "lead83_unit2_${panel}_${units}" --league nfl \
  --description "Noisy movement shrinkage versus refitted four-term base: ${panel}" \
  --source docs/lead83_unit2.md --family lead83_unit2_501_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" \
  --effect-units "$units" --effect "$effect" --interval-low "$low" --interval-high "$high" \
  --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Pooled primary Brier remains unresolved; no closure is claimed for this cell." \
  --plain-summary "When books disagree, this softens their line move before making the pick. It has not shown a dependable improvement over the usual calculation."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points 0.858369098712 -2.53164556962 4.31034482759 0.684300000000
2023 2023 2023 233 18 log_loss_improvement 0.0000951070833658 -0.000889153855018 0.000967703784076 0.600600000000
2023 2023 2023 233 18 brier_improvement 0.0000480297250444 -0.000443335290368 0.000483585087652 0.601300000000
2023 2023 2023 233 18 rps_improvement -0.0122876398807 -0.0221377685270 -0.00348538815843 0.00110000000000
2024 2024 2024 232 18 accuracy_points 0.00000000000 0.00000000000 0.00000000000 0.500000000000
2024 2024 2024 232 18 log_loss_improvement 0.00000000000 0.00000000000 0.00000000000 0.500000000000
2024 2024 2024 232 18 brier_improvement 0.00000000000 0.00000000000 0.00000000000 0.500000000000
2024 2024 2024 232 18 rps_improvement 0.00000000000 0.00000000000 0.00000000000 0.500000000000
2025 2025 2025 232 18 accuracy_points -0.431034482759 -1.33333333333 0.00000000000 0.179400000000
2025 2025 2025 232 18 log_loss_improvement -0.00114714424479 -0.00298553450760 0.000416050392457 0.0843000000000
2025 2025 2025 232 18 brier_improvement -0.000558503982151 -0.00145016975323 0.000199523774716 0.0849000000000
2025 2025 2025 232 18 rps_improvement -0.00689360774707 -0.0169840578496 0.00220299728483 0.0758000000000
pooled 2023 2025 697 54 accuracy_points 0.143472022956 -0.995732574680 1.70212765957 0.499000000000
pooled 2023 2025 697 54 log_loss_improvement -0.000350039475421 -0.00141863156756 0.000353458946242 0.245150000000
pooled 2023 2025 697 54 brier_improvement -0.000169845047236 -0.000690244164975 0.000176408547523 0.247150000000
pooled 2023 2025 697 54 rps_improvement -0.00640220529344 -0.0140197048332 0.00000000000 0.0296500000000
CELLS
```

## Next
Orchestrator reviews docs/lead83_unit2.md and executes the commands above serially;
then chooses the next bounded task. Rows and lineage: tests/scratch/codex/lead83_unit2/.

## Open
Three outer seasons, optimistic upstream training reads, and retrospective source
reuse limit inference. Three Tuesday totals are missing; no outcome-selected
repair. No closure or promotion; 2023 secondary RPS worsened despite unresolved
pooled primary Brier. Zero tuning slope is not a split-half reliability result.

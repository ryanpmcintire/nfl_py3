# LEAD-91 — season-stable fitting

## Goal
Execute the fixed four-term minimax-season challenger; historical OPENER is the frozen pool-line proxy. Research only.

## State
**Measured:** 601-look replay complete: 697 outer games; candidate 351-346, base 361-336; decisive 7-17. Brier improvement 0.000471 [-0.001087, 0.002823], probability_positive=0.63930. **Inferred:** unresolved_below_power; no serving change.

## Tried
Protocol saved before outcomes (verbatim in docs/lead91_unit1.md and scratch protocol.md); fixed ridge .001, four terms, minimax season mean loss; outer 2023/24/25, fit Y-3, select Y-2, calibrate Y-1. Refit upstream models, preserve pushes in discrete distributions. Full IS/OOS/gaps, five arms, coefficients and 601 looks saved in report. Single-thread real replay. Verified 2026-09-30 from saved predictions.parquet/summary.json: numbers reproduce; 601 = (27*5+5+4)*(3+1)+25; upstream fit through Y-3 enforced in code; 2023 (one training season, calibration slope .008) and 2024 (calibration slope 0) candidate equals base by construction. Script scripts/lead91_unit1.py, report docs/lead91_unit1.md.

## Record commands
Orchestrator runs these 20 exact candidate-versus-base OOS expansions serially; worker did not run them. Other contrasts and all IS/gap looks remain reported.

```bash
while read -r panel first last games blocks units effect low high pp; do
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name "lead91_unit1_$panel-$units" --league nfl \
  --description "Fit for the hardest season versus the usual four-term calculation: $panel" --source docs/lead91_unit1.md --family lead91_unit1_601_looks \
  --season-start "$first" --season-end "$last" --sample-games "$games" --sample-blocks "$blocks" --effect-units "${units/worst_season_log_loss_improvement/log_loss_improvement}" \
  --effect="$effect" --interval-low="$low" --interval-high="$high" --probability-positive "$pp" --classification unresolved_below_power \
  --classification-evidence "Three held-out seasons; no mechanism closure or serving claim. All 601 declared looks reported." \
  --plain-summary "This gives the hardest training season more say in the pick calculation. The held-out comparison remains a research result; pool picks have not changed."
done <<'CELLS'
2023 2023 2023 233 18 accuracy_points 0 0 0 0.5
2023 2023 2023 233 18 log_loss_improvement -4.72426542331e-11 -8.37916394603e-10 7.02703029098e-10 0.449
2023 2023 2023 233 18 brier_improvement -2.36919095453e-11 -4.18835537086e-10 3.50934095672e-10 0.44875
2023 2023 2023 233 18 rps_improvement 7.49254169818e-09 2.20031737275e-09 1.3111652053e-08 0.9975
2023 2023 2023 233 18 worst_season_log_loss_improvement -4.72425432108e-11 -8.37916394603e-10 7.02703029098e-10 0.449
2024 2024 2024 232 18 accuracy_points 0 0 0 0.5
2024 2024 2024 232 18 log_loss_improvement -0 -0 0 0.5
2024 2024 2024 232 18 brier_improvement -0 -0 0 0.5
2024 2024 2024 232 18 rps_improvement -0 -0 0 0.5
2024 2024 2024 232 18 worst_season_log_loss_improvement -0 -0 0 0.5
2025 2025 2025 232 18 accuracy_points -4.31034482759 -8.62068965517 -0.420168067227 0.0166
2025 2025 2025 232 18 log_loss_improvement 0.00342529073766 -0.00716005824717 0.0157739375672 0.713
2025 2025 2025 232 18 brier_improvement 0.0014138669508 -0.00345650356064 0.00689891148475 0.6942
2025 2025 2025 232 18 rps_improvement 0.0418007460124 -0.0150022092958 0.105158141516 0.9202
2025 2025 2025 232 18 worst_season_log_loss_improvement 0.00342529073766 -0.00716005824717 0.0157739375672 0.713
pooled 2023 2025 697 54 accuracy_points -1.43472022956 -4.55882352941 0 0.152
pooled 2023 2025 697 54 log_loss_improvement 0.00114012545212 -0.00220455627288 0.00644580034753 0.6537
pooled 2023 2025 697 54 brier_improvement 0.000470612807843 -0.00108703142412 0.00282260689032 0.6393
pooled 2023 2025 697 54 rps_improvement 0.0139135937168 -0.00250124484781 0.0506974388317 0.93985
pooled 2023 2025 697 54 worst_season_log_loss_improvement -0 -0.00640725893792 0.00358508040566 0.48215
CELLS
```

## Next
Orchestrator reviews report and executes serial records, then assigns the next bounded task. Rows/lineage: tests/scratch/codex/lead91_unit1/.

## Open
Three outer seasons; optimistic IS; retrospective archive reuse; missing Tuesday totals; no pool-noon fidelity claim. No closure or promotion. No unrelated edits, Git mutations or publication.

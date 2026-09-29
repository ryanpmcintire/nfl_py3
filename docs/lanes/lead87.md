# LEAD-87

## Goal
Refit the four-term base with whole-season exclusion and execute the postseason joint-likelihood replay once.

## State
**Measured:** replay completed once (exit 0): 686 outer REG games, 54 week blocks; candidate/served both 385-301, decisive 1-1. Brier gain -0.000003570 [-0.000125589, 0.000118521], probability_positive 0.4865. Provisional unresolved_below_power; not recorded or closed.

## Amendment declared before scoring
The immutable mechanism/family remain `docs/lead87_protocol.md`: one auxiliary POST likelihood shares only the move slope with all four REG terms, plus its own intercept. Historical opener is the frozen pool-line proxy; no pre-2026 Splash captures. Population: original 2020–2025 REG opener rows with admissible leader-book pairs. Missing moves are excluded, never imputed as no movement. Target: home cover at opener conditional on no push; POST pushes stay in evidence, not fitting.

Replace cached model logits: for each 2020–2025 season refit the configured weak-stack ridge margin model on completed earlier seasons only; refit residual distribution, discrete line-conditional margin reader, and home-side offset using earlier seasons only. Score at the historical opener. Pregame rolling features may use earlier games as observed information; no fitted parameter, residual pool, or offset uses held-out-season outcomes. Retain source features/flags; do not rebuild history. Save training IDs/cutoffs, hashes, and prediction rows in scratch.

Retain chronological outer 2023/2024/2025: joint fit through Y−3, Y−2 reserved for tuning (no selection: one fixed specification), REG-only Platt calibration on Y−1, score Y. POST training also ends Y−3. Fixed served ridge 0.001, margin ridge 10; no search. Independently refit identical four-term REG-only comparator. One calibrated probability selects the side. Baselines: season-excluded discrete model-only, timestamp-matched discrete market (opener plus paired move), opener/Elo logistic; calibration excludes Y. Report optimistic fit-set IS, outer OOS, OOS−IS gaps.

Primary endpoint opener Brier; also log loss, accuracy points, conditional binary RPS (= Brier). Decisive side-disagreement records precede effects. Report natural fold coefficients/calibration slopes, five fixed equal-width reliability bands, 10,000 paired week-block bootstrap draws stratified by outer season, seed 20260929, 95% percentile intervals, half-credit ties in probability_positive. Intervals condition on fitted models, not refit uncertainty. Family remains 497 looks (F=3, B=6, K=4), no added arms or outcome-driven revision. Zero crossing closes nothing; provisional unresolved_below_power pending orchestrator recording.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit2.py` ran once. Verified 20,893 training memberships exclude target/later seasons; 1,503 fresh base rows; 686 unique outer predictions; mass and metric arithmetic agree. Report: `docs/lead87_unit2.md`; evidence/log/amendment: `tests/scratch/codex/lead87_unit2/`.

## Record commands
Not run; orchestrator only. Positive favors candidate; RPS duplicates Brier, so no duplicate record.

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead87_unit2_postseason_brier --description 'Playoff training added to the regular-season four-term recipe: brier' --source docs/lead87_unit2.md --effect=-3.5700307536579157e-06 --effect-units brier_improvement --standard-error 6.2386612579113773e-05 --interval-low=-0.00012558885904129236 --interval-high 0.00011852125055055381 --probability-positive 0.48649999999999999 --sample-games 686 --sample-blocks 54 --classification unresolved_below_power --league nfl --season-start 2023 --season-end 2025 --family lead87_postseason_joint_likelihood --category market --classification-evidence 'No admissible closing ground established; fixed 497-look family.' --plain-summary 'Use playoff games to help judge how a moving line should change a regular-season pick. The added games left the score nearly unchanged; keep this idea under study.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead87_unit2_postseason_log_loss --description 'Playoff training added to the regular-season four-term recipe: log_loss' --source docs/lead87_unit2.md --effect=-1.2695663796913337e-05 --effect-units log_loss_improvement --standard-error 0.00013249659747407836 --interval-low=-0.00027160633303257296 --interval-high 0.0002480422463890741 --probability-positive 0.47049999999999997 --sample-games 686 --sample-blocks 54 --classification unresolved_below_power --league nfl --season-start 2023 --season-end 2025 --family lead87_postseason_joint_likelihood --category market --classification-evidence 'No admissible closing ground established; fixed 497-look family.' --plain-summary 'Use playoff games to help judge how a moving line should change a regular-season pick. The added games left the score nearly unchanged; keep this idea under study.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead87_unit2_postseason_accuracy_points --description 'Playoff training added to the regular-season four-term recipe: accuracy_points' --source docs/lead87_unit2.md --effect 0 --effect-units accuracy_points --standard-error 0.19884527856331308 --interval-low=-0.43165467625898674 --interval-high 0.43415340086830412 --probability-positive 0.49359999999999998 --sample-games 686 --sample-blocks 54 --classification unresolved_below_power --league nfl --season-start 2023 --season-end 2025 --family lead87_postseason_joint_likelihood --category market --classification-evidence 'No admissible closing ground established; fixed 497-look family.' --plain-summary 'Use playoff games to help judge how a moving line should change a regular-season pick. The added games left the score nearly unchanged; keep this idea under study.'
```

## Next
Orchestrator reviews the report, runs the three record commands serially, and commits this packet. No additional scored arm authorized.

## Open
Inherited feature/flag clocks were not re-audited; offsets cold-start in 2020; bootstrap conditions on fitted models. Only 8/17/26 POST labels enter chronological fits. No served-card or Git changes.

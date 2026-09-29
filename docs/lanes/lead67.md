# LEAD-67

## Goal
Price the late move in key-number probability under the unchanged ROADMAP.md LEAD-67 protocol.

## State
**Measured:** one successful unit run; 799 games, two arm looks. Decisive records vs base: replacement 40-51; additive 9-14. OOS accuracy gains in percentage points: -1.3767, 95% [-3.0075, +0.3745], probability_positive=0.0360; -0.6258, [-1.8797, 0.0000], probability_positive=0.1489. IS/OOS accuracy: base 57.3217/57.3217%; replacement 56.0701/55.9449% (gap 0.1252 pp); additive 57.1965/56.6959% (gap 0.5006 pp). Full losses, movement, reliability and coefficients: docs/lead67_report.md. No serving or closure decision.

## Declaration
Copied before outcomes; full original row and implementation declaration remain in docs/lead67_protocol.md: "Predeclared: 799 opener-graded games 2023-2025 (the move-available population), LOSO by season, the served four-term base as comparator, 2 looks, metrics log loss, Brier, accuracy and line-move-toward-pick with `probability_positive`, coefficients per fold with stability, decisive-game record first. The move remains a fitted term in one probability; no flip rule."

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead67_unit1.py` exited 0 once (UV_NO_CACHE=1, two numerical threads). Scoped Ruff and `mypy src` pass; isolated-temp `pytest -q --basetemp <temporary directory>` passes 1,645 tests. Repository-wide format/lint failures concern other files; initial pytest temp access failed. Logs: `%TEMP%/nfl-lead67-*.log`. No tests added or registry writes. **Read:** AGENTS.md research rules before fitting.

## Next
Orchestrator: run the two commands below serially; review endpoint/basis confounding before declaring another unit. Keep docs/lead67_predictions.md local and uncommitted.

## Open
**Inferred:** unresolved_below_power pending registration. Three-season intervals are nominal percentile ranges, with unvalidated coverage. The absolute late median differs from the base's within-book net move, mixing key-number pricing with pool/leader basis. Zero crossing never closes a signal; one fitted probability chooses the side. Global Ruff failures remain outside this packet.

## Record commands
Orchestrator only; accuracy_points uses percentage points per docs/weak_signal_registry.md:57. Reliability was not measured; no closing ground claimed.

```powershell
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead67_key_number_move_replacement --description "LOSO replacement of raw move by discrete frozen-opener versus late-leader price; 40-51 decisive" --source docs/lead67_report.md --effect -1.376720901126 --effect-units accuracy_points --interval-low -3.007518796992 --interval-high 0.374531835206 --standard-error 0.794514448847 --probability-positive 0.036 --sample-games 799 --sample-blocks 3 --season-start 2023 --season-end 2025 --league nfl --family lead67_key_number_move --classification unresolved_below_power --classification-evidence "Three-season paired bootstrap; no admissible closure established" --category market --notes "Two declared arm looks; endpoint and pool-leader basis are inseparable in this unit" --recorded-at 2026-09-29
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name lead67_key_number_move_additive --description "LOSO discrete frozen-opener versus late-leader price added to four-term base; 9-14 decisive" --source docs/lead67_report.md --effect -0.625782227785 --effect-units accuracy_points --interval-low -1.879699248120 --interval-high 0 --standard-error 0.507915253872 --probability-positive 0.1489 --sample-games 799 --sample-blocks 3 --season-start 2023 --season-end 2025 --league nfl --family lead67_key_number_move --classification unresolved_below_power --classification-evidence "Three-season paired bootstrap; no admissible closure established" --category market --notes "Two declared arm looks; endpoint and pool-leader basis are inseparable in this unit" --recorded-at 2026-09-29
```

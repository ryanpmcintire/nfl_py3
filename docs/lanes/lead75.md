# LEAD-75 — prospective paired wind forecasts

## Goal
Collect comparable Tuesday/deadline forecasts. Own `scripts/capture_paired_wind_forecast.py`, new Job entries in `scripts/capture_scheduler.py`, and this lane.

## State
**Measured:** collector and 15 scheduler entries implemented; all 272 scheduled 2026 REG games have a deadline slot. Live source check was blocked by sandbox socket permissions (WinError 10013): 0 receipts, 0 saved pairs. No outcomes read, fits, scores, registry writes, served-card changes, daemon restart, or commit. **Read:** historical source gate remains `data_gap`; see `docs/lead75_inventory.md`.

## Protocol — declared before collection
Population: upcoming REG games with mapped stadium stations in each Tuesday-through-Monday cycle; explicitly record unmapped/missing sources. Capture GFS (the deadline weather product) and extended-range MEX separately at both checkpoints; never substitute across products. Select the nearest valid time to kickoff on Tuesday and require the identical game/product/station/kickoff/valid-time key at deadline. Tuesday window: 12:05–12:20 ET. Deadline window: final 90 minutes before `pool_decision_cutoff`, including the Sunday 16:00 ET cap. Save actual post-response UTC receipt time, issuance, product, raw response, and checksum; exclude late receipts and reversed issuance order. No catch-up for missed windows. Terms/folds/metrics: none for collection; outcome looks: 0. Historical replay remains source-gated, not rejected; zero-crossing cannot close a signal, and any future replay must use one fitted calibrated probability to select the side.

## Tried
- **Measured:** `.tools/uv.exe run --no-sync python scripts/capture_paired_wind_forecast.py --phase tuesday --dry --game-id 2026_04_PIT_CLE` exited 1: source access blocked before any receipt; second product was not attempted after the transport failure.
- **Measured:** `.tools/uv.exe run --no-sync python scripts/capture_scheduler.py --run-job <name> --dry` exercised all 15 `paired_wind_*` jobs: 15 OK, 0 failures; all skipped outside their windows. This scheduler's `--dry` does not forward collector `--dry`; in-window invocations can save raw captures. Direct collector `--dry` above verified its no-write source path.
- **Measured:** Ruff check and format checks passed for both Python files. `.tools/uv.exe run --no-sync pytest -q -k scheduler -n 2 --basetemp <private-temp>` passed 64/64 in 16.99s. Used a private temporary uv cache and pytest base directory after default-cache access failures. Verification logs: `%TEMP%/lead75-scheduler-manual.log`, `lead75-pytest-scheduler-two-workers.log`.

## Record commands
None: collection produces no fitted/scored effect or statistical verdict to register. Counts have no sampling interval; `probability_positive` is not estimated. Orchestrator owns any later registry writes.

## Next
Orchestrator loads the updated schedule through its normal deployment process, verifies network access, then checks the first eligible Tuesday capture (2026-10-06 12:05 ET) and subsequent deadline pairs under `data/raw/paired_wind_forecast/<Tuesday>/<phase>/<receipt-run>/`.

## Open
**Measured:** network restrictions prevented live payload/write-path verification. Today's Tuesday window was already past. **Inferred:** the running daemon needs to load the new entries; activation was not verified. International stations can remain unmapped; schedule changes outside these kickoff clusters need new slots. Historical source gaps remain unresolved.

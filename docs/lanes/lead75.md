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

## Interim look 1 (2026-11-03, measured)
Look budget: declared protocol was 291 counted looks (ROADMAP) for the historical replay; this forward interim is 1 look, 1 term (MEX revision x home-minus-away pass-rate diff), no bands. Pairs: Weeks 4-6 had no usable Tuesday receipts (0 pairs); Week 7: 14 paired rows (2 dome, 8 outdoor incl. 1 GFS); Week 8: 8 paired rows (6 outdoor MEX + 1 GFS), unsettled. Scored: 7 settled outdoor MEX pairs; innovation mean 0.0, sd 1.3 mph (Week 8 sd 3.0). Slope of home ATS margin on term 46.3 pts/unit, bootstrap 95% [-497, 184], P+ 0.64 (declared sign negative); sign-consistent 2 of 5 nonzero. Recorded as lead75_paired_wind_interim_wk7, unresolved_below_power. Power (inferred): slope SE needs ~160 games for SE 20; at ~7 outdoor pairs a week, ~23 weeks. 11-01 11:45 FAIL: IEM MOS read TimeoutError aborted the run, 13 Week 8 rows not attempted (not_attempted_after_source_error); no retry exists.

## Tuesday-leg audit and capture fix (2026-11-03)
**Read/measured:** job `paired_wind_tue` (tue 12:05, 15 min) exists in scripts/capture_scheduler.py:683. Week 4 (cycle 2026-09-29): job added that evening, scheduler log MISSED, no capture. Weeks 5-6 (10-06, 10-13): RUN then FAIL(1) at 12:05, `atomic_bytes` argument-order AttributeError at the first receipt write, leaving an empty Tuesday dir (no forecasts.parquet); fixed in e7dacf95 before 10-20. 10-20 and 10-27 captured fully (14 games, 28 rows, 66-77 s). **Recoverability:** none. No other timestamped Tuesday 2026 forecast capture exists (data/raw/forecast_archive ends 2025); re-fetching old IEM runtimes now would be a later capture, forbidden. Forward-log audit should count a week healthy only if its Tuesday dir has forecasts.parquet.
**Fix (scripts/capture_paired_wind_forecast.py):** `read_with_retry`: READ_ATTEMPTS=3, backoff 15 s then 30 s (= request timeout, doubled), retrying timeouts, URL errors and HTTP 5xx; each attempt timestamp stored per receipt as `read_attempts_utc`. Basis (inferred): one measured IEM read timeout in the log (11-01 11:45) against ~450 receipts otherwise ok, so no failure-length distribution exists; worst added delay 45 s sits inside the 15-min Tuesday window. Tuesday leg already iterates every game in the week (no subset); the per-week limit is GFS/MEX horizon (12-13 of 28 rows `target_outside_product_horizon`) and 2 unmapped international stations, a protocol matter, not changed. No scheduler Job change needed.
**Before next look:** the pass-rate input of the interim look still needs a pregame (no leakage) check.

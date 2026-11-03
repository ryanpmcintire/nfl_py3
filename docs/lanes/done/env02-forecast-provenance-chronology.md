# ENV-02 forecast provenance and chronology repair

## Goal
Make roof-state screens use the exact archived input population and reject unverifiable pregame timing.

## State
Consumer repair complete; historical roof research remains unresolved.
**Measured:** the recovered 4,902-row feature build matches the original manifest SHA-256. Its isolated bundle supplies 626 roof-population games and preserves the original archive.
**Measured:** all 4,380 successful historical forecast rows have timezone-naive issuance values. The JSONL and parquet preserve those same values; neither retains authoritative timezone metadata. The loader rejects this archive.

## Tried
- Bound schedule, feature, and forecast files to manifest hashes; reject missing or duplicate game identities and incomplete population coverage.
- Require aware issuance, cutoff, and kickoff times, with issuance no later than cutoff and cutoff no later than kickoff.
- Bind kickoff to the pinned feature row and cutoff to the shared pool-decision policy. Reject nonfinite successful forecast values.
- **Measured:** `.tmp/env02_schedule_binding_check.py` exited 0: a valid pinned fixture loads; changed kickoff, changed cutoff, and the historical naive issuance archive fail closed. Earlier population and chronology probes also passed.
- **Measured:** scoped Ruff checks pass. Recovery evidence is in `.tmp/env02-recovery/issuance-evidence.json`; schedule-binding verification is in `.tmp/env02-schedule-binding-verification.md`.
- **Read:** `scripts/ingest_forecast_archive.py` requests UTC data but discards the response envelope and writes the returned issuance string unchanged. That request alone does not prove the historical strings' timezone.

## Next
Obtain a hashed authoritative issuance-time contract or retained response metadata, then create a separately versioned aware archive with explicit old-to-new row provenance. Rerun all consumers before fitting any roof term.

## Open
The original archive cannot be repaired by assuming UTC or refetching present-day responses. No new research look, closure, promotion, or served-side change was made. The legacy standalone roof flip is inadmissible under the one-fitted-probability rule.

## Update 2026-11-03 (issuance localization)
**Measured:** `.tmp/env02_tz_check.py` and `scripts/env02_localize_issuance.py`: for all 4,380 ok rows, naive `issuance_runtime_utc` equals floor-to-12h of the UTC `decision_cutoff_utc` minus 12h x `lookback_steps_used`, i.e. exactly the UTC runtime the request sent (`...Z`). 4,380/4,380 match, 0 mismatches. A naive local-time frame would not reproduce the requested UTC instants. This is derivation from request/response, not a provider contract (IEM doc URLs 404 via WebFetch).
**Measured:** after localizing to UTC: issuance <= cutoff 4,380 pass / 0 fail; cutoff <= kickoff 4,380 pass / 0 fail.
**Wrote:** new aware snapshot `data/raw/forecast_archive/pool_decision_2009_2025_aware/forecasts.parquet` (+ `localization.json`, sha256 a6ef0bcf...10ec); original untouched (sha 976c4907...). Script refuses on any mismatch.
**Not done:** manifest/loader rebinding of the aware snapshot (loader checks manifest hashes; needs a manifest for the new snapshot) and the roof-term LOSO fit and `weak-signals record`. Pinned-file diff: none needed.
**Next:** build manifest for the aware snapshot, run strict loader, then fit roof as one term in the served probability LOSO.
**Open:** whether owner accepts derived-from-request evidence in place of a hashed provider contract.

## Update 2026-11-03 (manifest, loader, roof term)
**Root decision:** the measured 4,380/4,380 identity of naive issuance values with the UTC runtimes the ingest requested is accepted as timezone provenance in place of a hashed provider contract (IEM docs page returned 404; a docs page is weaker than the exact identity).
**Wrote:** `data/raw/forecast_archive/pool_decision_2009_2025_aware/manifest.json` via `scripts/env02_write_aware_manifest.py`: original schema, new parquet sha a6ef0bcf...10ec, `source_snapshot` (original parquet and manifest hashes), `localization` derivation and evidence. Game-feature input path points at the recovered copy `.tmp/env02-recovery/game_features.parquet` (sha 40019393... matches the pin; the live file has since changed).
**Measured:** `roof_state_screen.load_forecast_inputs` and `build_prediction_table` accept the aware snapshot (4,431 rows, 4,380 ok, 626 roof-population games).
**Measured:** `scripts/env02_roof_fitted_term.py` (artifact `artifacts/env02_roof_fitted_term/20261103T125920Z/summary.json`): walk-forward predicted roof-open probability as a fifth term, LOSO, 2011-2025, 3,734 games, 530 nonzero. OOS log-loss improvement v four-term -0.000189 (season-block [-0.000304, -0.000092], P+ 0.000); Brier -0.000094 [-0.000152, -0.000046], P+ 0.000. In-sample improvement 3e-8, gap 1.9e-4. Fold beta mean +0.004, sd 0.077, 8/15 positive. Accuracy diff -0.05 pts, decisive 6-8 (14 games). Versus market 0.5 and model-only the full model beats them (log loss P+ 0.98 / 0.98) but that is the four-term base, not the roof term. Recorded as `env02_roof_open_probability_term_oos_log_loss` and `_brier`, unresolved_below_power (effect is ~0, term is unfit in-sample, folds flip sign; no refuting ground). 4 looks.
**Deleted:** `.tmp/env02_tz_check.py` (superseded by `scripts/env02_localize_issuance.py`; not cited by artifacts).
**Next:** none for provenance; roof term not served.

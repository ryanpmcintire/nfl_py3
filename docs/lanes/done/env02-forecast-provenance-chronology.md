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

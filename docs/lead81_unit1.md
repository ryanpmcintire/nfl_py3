# LEAD-81 unit 1: arrival-source inventory

**Measured:** decisive-game record is unavailable: 0 games scored; no win/loss/push record was computed.
Effect, 95% interval and `probability_positive` are unestimated, not zero or 0.5.

**Read:** the protocol was copied into `docs/lanes/lead81.md` before this run from
`ROADMAP.md:864` and `docs/lanes/ideation-2026-09-29b.md:11-27`.
It requires 2020-2025 REG games, verified planned/actual arrivals for both teams,
explicit on-time controls and public issuance/ingestion before the pick deadline.

## Inventory result

**Measured:** `source_gap`; 0 complete-schema candidates;
0 eligible game pairs established by this inventory.
This is a source-availability finding; the mechanism remains untested.

| Metadata check | Measured result |
|---|---:|
| Local data filenames, including ignored files | 95805 |
| Local artifact filenames, including ignored files | 53551 |
| Data table schemas inspected | 41639 |
| Travel-named artifact table schemas inspected | 0 |
| Source-like filenames across both roots | 3 |
| Tables with source-like filenames or columns | 25 |
| Tables with arrival/flight/reroute/itinerary columns | 0 |
| Registered source policies | 21 |
| Source-policy names matching travel/arrival terms | 0 |
| Metadata read errors | 5 |

**Measured:** Parquet reads use schemas only; CSV/TSV reads use headers only.
Data files in other formats and unrelated artifact files were catalogued by filename only.
No outcome rows, archives, model fits, network requests or source acquisitions were used.
Split-table sources, generic JSON/news archives and symlink targets are not certified absent.
A complete schema still requires row-level timing and control-coverage review before a replay.
Missing roots and read errors are recorded; neither proves source absence.

**Read:** `docs/travel_geometry_features.md:21-32` describes scheduled distance,
venue timezone and prior scheduled travel. Those inputs do not establish realized arrival delays.
**Inferred:** no qualifying arrival archive was identified in this inventory; retain the source gap.
Remembered cancellations or distance proxies cannot substitute for the declared population.

## Evaluation availability

**Measured:** IS/OOS accuracy, Brier, log loss, margin MAE and their gaps: unavailable.
Combined-model, model-only, market-only and Elo comparisons and five-band reliability: unavailable.
For each scheduled fold (2020, 2021, 2022, 2023, 2024, 2025), coefficients and season
intervals are unavailable because no arrival dataset was certified. No weights were fitted.
Planned looks: B=2, F=6, E=0; (2 + 36)(6 + 1) + 25 = 291. Executed statistical looks: 0.
**Read:** `AGENTS.md:65-83` permits no closure from insufficient evidence; no terminal verdict
or research effect is asserted here. There is no numeric result to record yet.

## Prospective collection specification

**Inferred collection requirements, without changing the registered historical protocol:**

- Declare the entire team-game roster before collecting reports, including home teams.
  Keep missing reports as missing; collect explicit on-time confirmations for controls.
- Preserve game/team/opponent/venue, season/type, frozen Tuesday-noon opener and its source,
  pick deadline, kickoff and observed market movement with issuance and ingestion times.
- Preserve separate planned and actual arrival times with timezone, report URLs, public
  publication times, local first-ingestion times and immutable evidence references.
  Retain original reports and revisions; never overwrite the as-of record.
- Capture explicit arrival status and documented cancellation/reroute cause, interruption
  time and report provenance. Confirm interruption occurred after the frozen opener.
- Require both teams' timing and evidence before the pick deadline and actual arrival
  no later than its report. Reject retrospective and outcome-selected reports.
- Derive arrival delay in hours and the away-minus-home term only from certified reports.
  Missing evidence never becomes zero. Reconstruct authentic frozen openers; never use closes.
- Before replay, tabulate both-team/control coverage by scheduled season without scoring.
  Use certified predictions and earlier-only training, selection and calibration folds.
  One calibrated discrete-margin distribution fits delay conditional on market movement.
  Preserve the declared metrics, baselines, season intervals, coefficients and 291-look budget.
- New prospective seasons require a separate predeclared protocol before outcomes; they
  cannot silently replace the fixed 2020-2025 historical population.

## Reproduction and handoff

Command: `.tools/uv.exe run --no-sync python scripts/lead81_unit1.py`.
Metadata details: `tests/scratch/codex/lead81_unit1_inventory.json` (local, ignored).
Next unit: identify a qualifying existing archive or arrange authorized prospective collection.
No registry command was run; there is no estimable effect or valid numeric record command yet.

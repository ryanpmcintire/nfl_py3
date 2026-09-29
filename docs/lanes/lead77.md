# LEAD-77 — quote-freshness reliability

## Goal
Test whether quote age improves the four-term probability at the historical opener.

## State
**Measured:** completed on 533 held-out 2024-2025 games / 36 week blocks; decisive 20-19.
Accuracy gain +0.187617 points [-2.044705, 2.247191], probability_positive 0.572050.
Log-loss gain -0.001454 [-0.004871, 0.001946], probability_positive 0.203200;
Brier gain -0.000788 [-0.002443, 0.000854], probability_positive 0.175400.
**Inferred:** unresolved_below_power; no serving or closure decision. Registry pending.

## Amendment saved before outcomes
**Read:** the full pre-outcome amendment is preserved verbatim in
`docs/lead77_unit2.md` under Declaration, with its SHA-256 and scratch declaration copy.
Historical OPENER proxies the frozen pool line; use the quoted 2020-2025 subset,
served four-term base, and provider observed_at_utc as assumed availability.
Every result is conditional on that provenance assumption. Two response fits retain
quote distance and age-times-distance; only pooled NFL+CFB supplies the fitted ATS
correction. Chronology-purged LOSO excludes held/later seasons in both leagues;
2023 trains only, 2024/2025 score. Features/labels precede each source cutoff.
The 217-look budget was reduced before scoring to 162; 158 consumed, including
conservatively counted derived gaps. No outcome-driven specification changes.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead77_unit2.py` completed
once, exit 0, with UV_NO_CACHE=1. Earlier attempts stopped before fitting at the
label-deadline guard and then a locale-decoding error; both were fixed without
changing the protocol. Scoped Ruff format/check passes; no new tests.
Validated 1,792 capped snapshots; saved rows/logs under `tests/scratch/codex/lead77_unit2/`.
Report includes IS/OOS gaps, every fold coefficient, response MAE and reliability.
Verified declaration hash, unchanged prediction rows, no code comments/docstrings,
and all nine registry payloads with the read-only validator. No registry writes.

## Record commands
Orchestrator runs serially; batch contains the nine measured paired contrasts,
per-cell sample counts, source and pool-player summaries. This command was NOT run.
```bash
UV_NO_CACHE=1 .tools/uv.exe run --no-sync nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead77_unit2/registry_batch.json \
  --source docs/lead77_unit2.md \
  --plain-summary "Older sportsbook prices were compared with recent prices before the deadline. This check does not establish that quote age should change the current pool picks."
```

## Next
Orchestrator: review `docs/lead77_unit2.md`, run the record command, then commit owned files.

## Open
Availability is assumed; only two seasons score. Same-line responses miss line changes.
No publication, served change, commit or push by this worker. Broader research remains open.

**Recorded 2026-09-29 (root):** the three freshness-vs-four_term cells only, unresolved_below_power.

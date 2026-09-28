# OPS-02 artifact retention inventory

## Goal

Measure current artifact and data ownership, then identify bounded retention work without deleting or
classifying files from age alone.

## State

The 2026-09-27 read-only snapshot found 3.3 GiB under `artifacts`, 14.4 GiB under `data`, and 3.9 GiB
under `.tmp` excluding its live `uv-cache`. Sizes are logical. The active pointer protects the current
historical evaluation and Week 3 forecast. Current contracts also protect feature, forecast, card, decision,
revision, and lockday lineage. Raw market, player, injury, and other point-in-time captures need retention and
source-retrievability evidence before pruning.

## Tried

Measured aggregate counts, logical sizes, useful date bounds, and hardlink identities. The 2.1 GiB rehearsal
tree has 2.0 GiB hardlinked outside it, leaving only 45.8 MiB without another link. Traced the two
`artifacts/cache` families to content-derived keys in `clv.py`; they total 85.5 MiB. Traced scheduled backup
ownership, but did not contact backup destinations, so current coverage remains unverified. Full evidence is
in `.tmp/ops02-retention-inventory.md` and its two JSON inputs.

**Measured:** The reference report checked 67 ledgers, 57 inactive manifests, and
826 registry JSON files. The archive-review audit found 6,258 files totaling
528.6 MiB, with no reported external hardlinks. Five generated subdirectories
account for 527.9 MiB; 41 root evidence files remain distinct historical evidence.
Reports: `.tmp/ops02-reference-report.md` and
`.tmp/ops02-archive-review-eligibility.md`. No files were deleted.

## Next

**Measured 2026-09-28:** approved read-only process inspection read all 16 candidate
command lines; none referenced `.tmp/archive-review`. The five named generated
directories still total 553,507,289 bytes and contain no reparse entries. This
resolves the unavailable command-line check, not every possible open handle.
The exact deletion scope is awaiting the owner's required approval; no deletion
has occurred. Retain all root screenshots, source snapshots, logs, and diffs.
Evidence: `.tmp/backlog-execution-20260928/process-ownership.json` and
`retention-candidates.json`. Do not restore the removed retention planner.

## Open

Which `.tmp` review outputs remain needed by active work? What regeneration cost is acceptable for evaluator
caches? Which superseded runs are referenced by registries or published evidence? Do current data and
artifact backups cover every candidate before any future mutation?

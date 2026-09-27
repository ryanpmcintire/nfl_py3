# Retention tooling availability

## Goal
Correct OPS-02 references that present removed retention commands as available.

## State
Completed documentation correction on 2026-09-26. OPS-02 itself remains open.

## Tried
**Measured:** `git diff-tree --no-commit-id --name-status -r b7ed31d` scoped to
`scripts/artifact_retention.py`, `src/nfl_ats/artifact_retention_policy.py`, and
their two former test files reports all four deleted. Current `Test-Path` checks
confirm all four remain absent. Scoped filename searches found no replacement
under scripts, src/nfl_ats, or tests; tracked retention filenames are documents.
Updated `docs/artifact_retention.md` and the OPS-02 roadmap row to distinguish
historical commands and measurements from available tooling. No files were
pruned, quarantined, or restored, and no operational jobs ran for this correction.

## Next
Assess the current need for read-only inventory and budget tooling before a
bounded restoration or replacement. Preserve historical evidence and policy.

## Open
Current disk use and backup coverage were not measured by this task. Historical
measurements do not establish either. A deletion mode remains outside this task.

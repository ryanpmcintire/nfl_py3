# Roadmap sync — 2026-09-29

## Goal
Reconcile only ROADMAP.md rows LEAD-66–89, MOD-22 and XLG-09 with their lanes and today's registry cells; write docs/research_digest_2026-09-29.md.

## State
Complete: only the 26 authorized ROADMAP.md rows, docs/research_digest_2026-09-29.md and this lane changed. **Measured:** 19 rows complete, 2 with declared work remaining, 5 source-gated without scoring; digest has 32 unit/inventory entries and three inferred paragraphs. No experiments, registry writes, Git mutations, publication or served-card changes.

## Tried
Read research policy, scoped lanes/reports with cr.ps1 and searches with cs.ps1; reconciled 84 cells recorded today. **Measured:** all 54 digest effect/interval/probability_positive triples match registry rounding, including newly recorded LEAD-88 unit 3. Inline `.tools/uv.exe run --no-sync python -` verified all 26 summaries, 32 table entries, three inferred paragraphs and 34 local links. `git diff --check -- ROADMAP.md docs/research_digest_2026-09-29.md docs/lanes/roadmap-sync-2026-09-29.md` passed; non-target roadmap text has the same SHA-256 as before editing. No Python files changed, so Ruff is not applicable.

## Next
Orchestrator reviews and commits this packet, then refreshes LEAD-86 when unit 2 finishes. Completion marks research work, including unresolved results, rather than mechanism closure or serving approval.

## Open
**Read:** LEAD-66 lacks older-period secondary predictions; LEAD-86 unit 2 is pending. LEAD-75/76/80/81/89 lack scored units. LEAD-88 unit 3 finished and was registered by the orchestrator during verification; it is included. LEAD-76's old pre-2026 pool-capture gate is superseded by the owner's opener standard; older histories/provenance remain missing. Registry classifications supersede stale lane proposals; later results need fresh reconciliation.

## Record commands
None: this bookkeeping task creates no new research looks or registry decisions.

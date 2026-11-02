# Backlog triage G1 (market/pools/ops), 2026-11-02

## Goal
Classify MKT-08 POL-04 POL-05 POL-08 POL-10 POL-11 POL-12 POOL-01 OPS-02 OPS-06 UI-20. Read-only; nothing run.

## State
- MKT-08 FORWARD-ONLY: retrospective term recorded unresolved; prospective bridge live (scheduler refresh_trigger_log_sun OK 2026-11-01 12:43). Scorable once news-event decision rows accrue.
- POL-04 CLOSE-NOW (already status X; premise note about earlier-game unlock flagged in row, not an open unit).
- POL-05 FORWARD-ONLY: waits on field size/prize structure observables (`nfl-ats pool-observables`); owner must supply from the pool page.
- POL-08 NEXT-UNIT: row is one line, no design; fold into POOL-01.
- POL-10 FORWARD-ONLY: scorecard `scripts/prospective_scorecard_2026.py`; runs weekly.
- POL-11 NEXT-UNIT: repair unrun (`--record-decisions --replace-week --no-board`) + read lane; evidence base unresolved (P+ 0.935).
- POL-12 NEXT-UNIT: totals model execution frozen in `docs/totals_model.md`.
- POOL-01 NEXT-UNIT: Unit 1 recorded; Unit 2 needs real pool picks.
- OPS-02 NEXT-UNIT / OPS-06 OWNER-PHYSICAL / UI-20 standing lane.

## Tried
Read row tails, `docs/lanes/pol10-prospective-2026.md`, scheduler log tail.

## Next
Root applies verdicts from the final report.

## Open
POL-12 and POL-11 row middles not fully read; verdicts rest on row tails.

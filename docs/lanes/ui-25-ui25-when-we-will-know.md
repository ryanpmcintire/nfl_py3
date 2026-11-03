# UI-25 Findings

## Goal
Findings: a 'when we will know' table of games still needed for each open question (ROADMAP UI-25).

## State
Added 2026-11-03, unmeasured. Pool players see claims but not when they can be settled. Each open forward log has a games-to-resolution date; the page should say it.

## Tried
Nothing yet. Novelty: UI-20 (standing lane) and UI-24 (luck band). Grep found no resolution-date table.

## Next
First unit: Add a table to the Findings page listing each open question, the games logged and the earliest date it can be read, in plain words.

## Open
Data: Output of ENG-48 and the per-log declared effect sizes. Zero crossing closes nothing; record unresolved results with a closing ground.

## Unit 1 (2026-11-03)
Built: "When we will know" table on Findings, right after the signal atlas (board_terminal._forward_tests_section_html, board_site_content._load_forward_tests). Rows read only from artifacts/forward_monitor/monitor.json (games, sign of mean gain, final e vs threshold, inferred games to threshold; "Unknown" when null). Existing classes only. Rendered at 390 and 1440 in tests/scratch/ui-audit-2026-11-03/site11, looked fine. Not published to docs/, not committed.
Next: orchestrator publishes; add MOD-35/SKY/ENV-02 rows when the monitor covers them.

# Weekly navigation and dashboard fixes

## Goal
Ship the current-week selector and useful parallel backlog fixes while preserving original published picks.

## State
Complete and verified on 2026-09-22; publication generated successfully. Current Week 3 defaults and clearly waits for authentic pool lines; archived Weeks 1 and 2 show original recorded picks. Findings redesign is outside this batch.

## Tried
- Measured in Edge: Weeks 1/2 each show 16 original picks; URL/history navigation, current-week return, 390px mobile layout, game-room Enter/Escape, lineup and formation selection, column-guide keyboard operation, and unique ticker links work without JavaScript exceptions.
- Measured required checks: Ruff format/check passed, mypy passed (237 source files), pytest passed (4529 passed, 9 skipped, 162 warnings). Logs: `.tmp/week-navigation/*-release-final.log`.
- Measured `nfl-ats publish-board`: success. Rendered diff: index +162/-12 lines; model/history/findings each +36/-9 from shared presentation assets.
- Completed parallel work: keyboard controls, initial selection states, plain-language column guide, ticker keyboard duplicates, injury-name/designation presentation, and portable source-policy backup documentation.
- Post-review residue sweep: no tests added and no research-only runtime assertions introduced. Removed a proposed market-readiness gate that conflicted with fitted missing-data handling.

## Next
Week 3 acquisition continues in `../week3-lines-2026-09-22.md`; capture authentic FTPL text, validate it, then exercise the existing weekly-lock scheduler command. Remaining backlog priorities live in ROADMAP.md.

## Open
- Week 3 picks require authentic pool spreads. Paid odds returned HTTP 401; Bovada returned empty JSON; OddsGap captured six quotes for one matchup. Book quotes do not replace FTPL lines.
- Scheduler status measured a healthy daemon (PID 25656) and missed Tuesday lock. The start script refused a duplicate; no restart is claimed. No scheduler jobs were added or changed.
- Preserve preexisting changes: CURRENT_PREDICTIONS.md, tiebreaker.json, and six untracked experiment registry records.

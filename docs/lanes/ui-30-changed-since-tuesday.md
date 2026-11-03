# UI-30

## Goal
Changed-since-Tuesday strip: games whose side or number moved since the Tuesday card, in plain words (ROADMAP UI-30).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Pool players want to know what changed before they enter picks. Fixes the card and board in place from the pick-revision ledger, no new layout. No fitted term; display only.

## Tried
Novelty check only. Nearest: UI-22 (why the line moved), UI-26; grep 'changed since' found none. Differs: Pick changes, not line moves.

## Next
First unit: Add a short strip above the picks listing each changed game, previous side, current side and the plain reason; check at 1440 and 390 widths.

## Open
Data: Pick-revision ledger (local). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03T21:25Z
Strip at top of the board column (board_terminal.py _changed_strip_html, called from _board_section); data from content.week_changes. Archived weeks now get their own panel from the pick-revision ledger (board_archive.py _archived_changes; artifacts_root threaded via board_site/board_week_navigation). Week 9: "No changes since Tuesday."; Week 8: 5 games changed. No cover-chance-change line (no derivable threshold wired; quote_step in board_move_reason.py exists). Archive reasons are generic. 390px not rendered, by CSS read only. Not published/committed.
Next: cover-chance changes beyond quote_step; render at 390.

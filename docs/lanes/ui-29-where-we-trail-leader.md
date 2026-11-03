# UI-29

## Goal
Where we trail the leader: the games that separated our card from the top entry, shown without entrant names (ROADMAP UI-29).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: The pool has one winner; the reader should see which games gave the leader the points and whether those were identifiable beforehand. Fix in place on History, with entrants shown by rank only.

## Tried
Novelty check only. Nearest: pool-leader-romana.md (research on one named entrant), UI-21, UI-24. Grep of docs/*.html and board code for a leader comparison found none. Differs: a reader panel with ranks only.

## Next
First unit: List games where our side differed from rank 1, with net points each way, and render on History at both widths.

## Open
Changes: Page: History. Data: data/splash/field picks (Weeks 1-8) and our settled ledger. Rank-labelled panel from stored picks. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03T21:01Z
History Season-so-far: details fold "Where we split from the leader" + table (week, game, our side, their side, right). Source data/splash/field/2026_romana_empire_w01-07.csv joined to ledger by game_id; Weeks 1-7 only (leader picks not captured for 8+). Code: board_site_content.py _leader_split + SeasonSoFar.leader_split_*; board_terminal.py _leader_split_html. 309 board tests pass; built in memory, not published. Next: view at 390 width; capture week 8+ leader picks; net-points view.

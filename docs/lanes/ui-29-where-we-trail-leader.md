# UI-29

## Goal
Where we trail the leader: the games that separated our card from the pool's rank-1 entry, shown by rank only, on History (ROADMAP UI-29).

## State
Unit 2 (2026-11-21, stopped at the subagent tool cap; not committed, not published).
- Done in src/nfl_ats/board_site_content.py: `_leader_split` now reads `data/splash/field/{season}_leader_rank1_picks_w*.txt` (new `_leader_picks`, `_LEADER_PICK`, `_SPLASH_TEAM_CODES` JAC->JAX, LA/LAR), joins week + team to the served ledger rows (`served_paper_decisions`), grades both sides at the pool line in the file, returns 4-tuple; new `SeasonSoFar.leader_split_best_text`. board_terminal.py `_leader_split_html` prints the Best Pick paragraph; headers now "Leader's side", "Point went to". Old romana CSV is no longer read by board code; scripts/pool_observed_replay.py still reads it directly (untouched).
- Measured (tests/scratch/ui29/build.log, first build): 151/151 leader picks matched; 0 disagreements between ledger grade and pool-line grade; 57 split games, leader 34, us 23, net 11 to the leader; weeks us-them W1 3-3, W2 4-2, W3 1-2, W4 3-4, W5 3-5, W6 1-5, W7 1-3, W8 1-3, W9 3-4, W10 2-3, W11 1-0 (Thursday only); Best Picks over 11 weeks leader 7-4, ours 6-5, same game in 2 weeks (W4 PIT-CLE, W11 IND-HOU), opposite sides both times. Leader's 89 points = 89 correct picks, so Best Pick adds no points.
- After that build the Best Pick sentence was revised to say same/opposite sides; not yet rendered (rebuild was killed at the cap).

Unit 3 (2026-11-21): three mypy fixes applied; mypy src 14 errors in 5 files, 0 in board_site_content.py/board_terminal.py; ruff check and format clean on both; build rerun OK (151/151 matched, 0 disagreements); Best Pick sentence now renders "7-4 vs 6-5, same game in 2 weeks, opposite sides". Reader text checked: rank only, no handles/ids.

## Tried
mypy src (before revisions): 20 errors in 7 files; in owned files: board_site_content.py `_half_point_week_texts` int(week) on groupby key (2), board_terminal.py evidence `max(... probability_positive)` (3) and `_season_timing_points_text` Any return (2). No headless browser in repo scripts (grep scripts for --headless/playwright: none).

## Next
1. (done unit 3) mypy fixes, build, ruff. pytest -k board run in unit 3; see root report.
3. Root: 390 px width check of tests/scratch/ui29/site/history.html, then publish.

## Open
ROADMAP UI-29 update: Unit 2 switched the History panel to the current rank-1 entry (Weeks 1-10 plus Week 11 Thursday), with net points each way (57 opposite-side games, leader 34, us 23, net 11) and Best Picks each way (7-4 vs 6-5); mypy fixes done (0 errors in both files), ruff clean; pending width check, publish.
Weeks 11+ need a fresh read of the rank-1 entry's picks after each slate.

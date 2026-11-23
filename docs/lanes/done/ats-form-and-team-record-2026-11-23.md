# ats-form-and-team-record-2026-11-23

## Goal
Owner 2026-11-23: backtest hot/cold ATS streaks and season ATS record (home team especially) as a lagging-market edge; per-team record of the model's picks; a board column with the model's record picking each team THIS SEASON (owner narrowed it to this season only). Done = all three, recorded and published.

## State (done 2026-11-23)
- Scripts: scripts/ats_form_and_team_record.py, ats_form_early_away.py, ats_form_streak_early.py; outputs artifacts/ats_form/ (summary.json, early_away.json, streak_early.json, team_pick_record_2020_2025.csv). Declarations were stamped in this lane before each read (14:45:27Z, 14:51:01Z, 14:52:21Z). 23 cells in registry families ats_form_2026_11_23, ats_form_early_2026_11_23 and ats_form_streak_early_2026_11_23; pool exit 0.
- Momentum / lagging line REFUTED vs the opener 2011-25 (wrong_sign_resolved): following better ATS form went 1420-1550; season margin diff -2.18 [-3.78, -0.66]; weeks 2-8 -4.31 [-6.78, -1.88]; streak weeks 2-8 follow 561-619. Weeks 9+ nothing. The effect comes from the AWAY team's form (home team's own record ~0). In points the opener does not lag (lag-1 r .002 [-.024, .028]).
- The fade direction does not add to the model, 2020-25 LOSO at opener (base 810-693): full-season terms -0.7 to -1.7; early-only margin 806-697, early away 809-694, early streak 814-689 (null 87th), Sept 3-straight flag early 808-695. Log loss improves slightly (.6913 v .6922), fold signs stable. Same answer as mod57-p2r (2026-11-11, D2 x weeks 2-6: 828-675 v 829-674), so this is a replication, not new.
- Per team, model alone at opener 2020-25: NYJ 14-24 picked / 34-24 against, HOU 16-32, KC 18-27, NO 14-22; CIN 39-18, SF 30-15. Spread a bit wider than luck (95th pct), split-half r .30 (95th pct), but every usable form loses: fade-below-.500 rule -8 games (718-638 v 726-630); pick-rate term -1.06; model-bias term -1.73.
- 2026 card by team through Week 11 (83-79): TEN 0-4, PHI 0-3, ARI 1-5, DAL 2-5; SF 5-0, DEN 7-1, JAX 5-1, NYJ 4-5.
- Board: "Model on this team" column (src/nfl_ats/board_team_record.py; GameRow.team_record_text; _with_team_records in board_content.py; 8th column in board_terminal.py _board_section; CSS widths; tests/test_board_flip_line.py colspan 8). Shows this season's card record picking that team, weeks before the current one; archived weeks omit it. Owner follow-up 2026-11-23: the cell shows both teams, picked team first (e.g. "NYJ 3-5 · LAC 1-3"); the other team's record is our record when we picked them. Rendered 13/13 Week 11 rows; 302 board tests pass; publish-board exit 0.

## Tried
- The Sept ats_streak_regress test (2020-21, 456 games) could only detect about 2.2-point effects.
- A subagent's repo-wide `ruff format .` rewrote 250 tracked files; reverted byte-exact (hash check: only the 4 intended files differ). Format named files only.

## Next
- None in this lane. The early-season fade lead stays with mod57-p2r (2027 weeks 2-8 forward).

## Open
- None.

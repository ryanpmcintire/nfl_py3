# lead65-protection-window-split

## Goal

Answer the owner's 2026-09-13 question about the served pass-protection
mismatch tilt: its four-game pressure window uses the previous four
regular-season games (playoffs excluded) and rolls across the offseason, so
weeks 1-4 flag on last season's line and pass rush. Done when the tilt's
marginal on the played card is measured for window-spans-offseason (weeks
1-4) versus later, both cells are recorded with `nfl-ats weak-signals record`
under family `pbp08_protection_mismatch_window_split`, and the ROADMAP row
LEAD-65 states what the read implies for the served rule.

## State

- ROADMAP row LEAD-65 added 2026-09-13. Measurement delegated to a subagent
  (report expected at `tests/scratch/lanes/pbp08_early_season_split_20260913.md`).

## Tried

- Nothing yet.

## Next

- Read the subagent report, verify one cell by re-running its command, and
  write the numbers here and in the ROADMAP row.
- If the early cell resolves wrong-signed: design a season-aware window
  (reset at week 1, or require a roster-continuity gate on the offensive line
  and pass rush) as a paired challenger; otherwise the tilt stays as served.

## Open

- None.

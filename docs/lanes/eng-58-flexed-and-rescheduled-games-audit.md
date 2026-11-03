# ENG-58

## Goal
Flexed and rescheduled games: opener captured for one kickoff slot, game played in another (ROADMAP ENG-58).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: A flexed or moved game changes day or time after the opener posts. Slot labels (MKT-24), short-week flags and weather inputs may then describe the wrong slot, and the opener line may predate the move. The error is a stale join, not a number. Fitted term: None: an audit that tells which graded games carry a slot, rest or weather input from the original schedule.

## Tried
Novelty check only. Nearest: ENG-52 (opener timestamp audit), ENG-55, MKT-24. Differs: Audits the kickoff slot itself, not the capture time.

## Next
First unit for one agent: Compare the schedule row at opener capture with the final schedule row for every graded game, list games whose date or kickoff changed, and report how many feed a slot, rest-day or forecast input.

## Open
Rules: audit only, no research verdict; report counts, rows and the exact join; any graded result that changes is re-run and re-recorded through the registry validator, never weakened. Look count and family declared before any sign is seen. Data: nflverse schedules, captured schedule snapshots and opener archive timestamps (local).

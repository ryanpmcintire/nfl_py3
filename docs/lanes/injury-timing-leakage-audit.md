# Injury timing leakage audit (release-blocking, OPEN)

## Goal
Confirm or rule out look-ahead in the served model's historical injury inputs.

## State
- **Measured (LEAD-89 unit 2, docs/lead89_unit2.md):** 814 of 816 Tuesday Best
  Pick candidates (2023-2025) carry injury inputs observed after the Tuesday
  decision time.
- **Measured (root, 2026-09-29):** the served profile (football_weak_stack, 88
  features, src/nfl_ats/constants.py:565) includes 9 injury-derived columns:
  diff_injury_{offense,defense,special_teams,offensive_line,skill,front,
  secondary}_unavailability, diff_injury_skill_epa_value_lost,
  diff_injury_defense_disruption_value_lost.
- Not yet known: the timestamp each historical row's injury value used, the
  prediction timestamp the historical evaluation assumes, whether the runtime
  leakage guard covers these columns, and the live Tuesday card's snapshot.
- The Codex audit worker failed at the Codex usage limit (resets 2026-10-06
  16:33); its packet is tests/scratch/codex/leakaudit.prompt.md.

## Tried
- Nothing scored; served card unchanged.

## Next
- Run the packet: map each injury column to its report timestamp, compare with
  the evaluation's prediction timestamp, check the guard, then refit LOSO with
  timestamp-valid injury inputs and report the held-out accuracy change.

## Open
- If historical rows use Friday reports for a Tuesday-timestamped evaluation, the
  published historical accuracy is optimistic and the guard needs a fix.

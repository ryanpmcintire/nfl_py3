# SIM-26

## Goal
Leading-defense shape late: yards per play and explosive-play rate for defenses protecting a lead (ROADMAP SIM-26).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: A defense protecting a lead concedes short gains and fewer explosives, which lengthens drives and lowers scoring-drive rate; a trailing offense meets it. If the sim draws plays without this, the leading-team scoring gap persists on non-truncated drives. Measured mechanism: yards per play and explosive share by defense score sign and clock band from real play-by-play.

## Tried
Novelty check only. Nearest: SIM-19 (trailing offense risk shape), SIM-25, SIM-18. Differs: the defensive side of leading state; SIM-19 covers the offense.

## Next
First unit for one agent: Tabulate real yards per play, explosive share and completion depth by defending-team lead and quarter, compare with the sim play draws, and report the drive-level gap share this explains with intervals.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.

# Lanes

A lane file is the state a fresh session needs to continue a task after the
previous session was cleared. One file per task, named by a short slug. The
session that works a lane updates it whenever a unit of work completes, and
every response ends with a one-line clear verdict (AGENTS.md, Lanes and
clearing). The status line shows the most recently touched lane.

A lane file has five short sections and stays under one page:

- **Goal**: one or two sentences, including what done looks like.
- **State**: what is shipped, committed or pushed; exact file paths.
- **Tried**: what was attempted and the measured result, so it is not redone.
- **Next**: the exact next step, with the command or file to open.
- **Open**: questions only the owner can answer, and decisions deferred.

Lanes that are finished move to `done/` with a final State section. A fresh
session reads this index, then the lane the prompt names, or the most recently
modified lane when the prompt just says to continue.

## Active

- [conditional-tilt-effects](conditional-tilt-effects.md) — MOD-19, owner direction 2026-09-13: tilts decided with the other relevant information (line/pass-rush continuity first); stage 1 is LEAD-65
- [lead65-protection-window-split](lead65-protection-window-split.md) — owner question 2026-09-13: does the protection-mismatch tilt hold in weeks 1-4 when its window is last season's games; measurement in flight
- [sunday-gameday-2026-09-13](sunday-gameday-2026-09-13.md) — Week 1 Sunday: hand odds capture, new 09:30 lineups_sun_am refit job, daemon restarted; open: Week 1 --replace-week re-record (owner)
- [lead64-headline-parser](lead64-headline-parser.md) — LEAD-64 headline designation parser, three passes done; next is the morning comparison against the official feed
- [lead59-archive-battery](lead59-archive-battery.md) — LEAD-59 src fix and battery re-run done, 23 cells recorded; open: type-trait binning with the archive on
- [token-diet](token-diet.md) — session-startup token cost cut about 80%; remaining: trim the three 15 KB+ open ROADMAP rows (owner text) and decide whether `.claude/` hooks should be tracked

## Done

- [halves-ledger-lockday](done/halves-ledger-lockday.md) — 2026-09-12
- [dashboard-2026-09-12-evening](done/dashboard-2026-09-12-evening.md) — 2026-09-12, Books-now market-move line

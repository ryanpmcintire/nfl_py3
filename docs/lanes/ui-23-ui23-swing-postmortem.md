# UI-23 Swing postmortem

## Goal
Tell readers how each miss was decided, not only that it missed. ROADMAP row UI-23.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Per settled pick, show the cover state entering the fourth quarter and the last scoring play that decided it, so readers see which misses were late backdoor swings versus early deficits. Data: nflverse pbp after each game (nflreadpy), settled picks ledger.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: UI-20 Vs the close column and week strips, MOD-28 (team trait, different purpose). No page narrates how a miss happened.

## Next
Compute quarter-end cover states for the settled Week 9 picks. First unit: First unit: compute cover state at each quarter end for Week 9 settled picks and add a Last week column to History.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Overlap with MOD-28 data must not become a feature leak: this column is display only.

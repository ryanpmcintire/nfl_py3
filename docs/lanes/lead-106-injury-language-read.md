# LEAD-106

## Goal
Injury-report language read: an LLM maps designation text and practice notes to expected snap fraction per starter (ROADMAP LEAD-106).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Designations like questionable carry different meanings by player and note; free text may encode more than the ordinal ladder. Fitted term: expected-availability difference replacing the ordinal injury input, frozen prompt and forward capture.

## Tried
Novelty check only. Nearest: PER-03, PER-10, LEAD-99 (verbal read), LEAD-64 (capture integrity); grep 'injury text' found none. Differs: Language to availability, not language to cover.

## Next
First unit: Run a frozen prompt over the stored captures, compare Brier on realised snaps with the ordinal ladder, then pass to the four-term probability only if it adds out of week.

## Open
Data: Official injury report text captured (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

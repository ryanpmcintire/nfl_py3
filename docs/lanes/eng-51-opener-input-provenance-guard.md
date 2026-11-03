# ENG-51

## Goal
Opener-input provenance guard: every opener-graded harness declares its model input line and auto-emits the intercept-only and line-move-only controls (ROADMAP ENG-51).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: On 2026-11-03 three false leads came from the closing spread_line fed to harnesses that graded at the opener. A shared entry point that refuses a frame whose model line equals the close, and prints the controls beside every result, changes grades by making line-following visible.

## Tried
Novelty check only. Nearest: backlog-all-2026-11-02 State 13:55 correction, MOD-36 (noise floor), ENG-47 (metric plumbing), MOD-34 note that clv.py:2208 is correct. Grep for provenance of the model input line across scripts found only the single fixes. Differs: systemic audit and automatic controls.

## Next
First unit: Audit: list every script that reads the feature-table spread_line and also grades at the opener, mark which pass the opener as input, and rerun the controls on any whose result is still quoted in a lane or ROADMAP row.

## Open
Changes: Prevents false grades. Data: scripts/ that call opener evaluation (about 177) and src/nfl_ats/clv.py. Audit table, then a runtime check in the shared grading path (no tests added). Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

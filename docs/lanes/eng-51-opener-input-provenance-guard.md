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

## Unit 1 audit (2026-11-03 21:00Z, read by grep plus targeted line reads, not full reads)
Table artifacts/eng51/audit.csv, 96 scripts read a closing column and touch an opener grade. Classes: opener_input_clean 67, not_opener_graded 14, close_as_input_leaky 6, close_as_stratifier_only 6, mixed 2, production_parity 1 (counts in the report).
Leaky (all share one pattern: pre-2020 history pool built from nflverse close, scored at the opener): lead76_declared, lead76_rule_prior, lead66_unit2, mod24_u8_grade_deadline, total_conditioned_lattice, altitude_fourth_quarter_screen. The three named false leads (mod24_more_games, XLG-12, XLG-11) are already fixed and classed clean.
Rerun (tests/scratch/eng51/lead76_open_control.py, pool lines swapped to SBR open 2011-2019, 2009-2010 dropped, out artifacts/eng51/lead76_open_control/): LEAD-76 rule-vs-pool log loss +4.7e-6 (P+ 0.657) became -1.8e-9 (P+ 0.233); pool-vs-market -0.00147 became -0.00131. Zero both ways; not a refutation; stays unresolved_below_power. Confound: smaller pool.
Next: unit 2 runtime guard in the shared grading path; lead66_unit2 and mod24_u8_grade_deadline reruns with SBR-open pool.
Open: classification of the 67 clean rows is by script-text read, not by rerunning each with an intercept-only control.

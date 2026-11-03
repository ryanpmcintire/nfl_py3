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

## Unit 2b design (not implemented; 2026-11-03)
Mechanism: all six leaky scripts built the pool through `prior_pool(features, opener_lines)` (src/nfl_ats/mass_preserving_lattice.py:163), whose `pool["line"] = game_id.map(opener).fillna(spread_line)` silently falls back to the nflverse close for any game without an opener. That fallback is the single leak point.
Guard: add `line_source` column in `prior_pool` ("opener" where the map hit, "close" where it fell back) and a keyword `require_opener_through: int | None` (a season, no default magic number; callers pass the first graded season). A new function `assert_opener_pool(pool, graded_season)` raises `DataContractError` when any pool row with season >= the earliest season the caller grades has `line_source == "close"`, and when `pool.line` equals `spread_line` on more than the rows legitimately equal by coincidence (check by source column, not by value). A second check in `band_read`/`select_band` callers is unnecessary because the pool is the only shared input.
Second site: `clv.opener_pick_evaluation` (clv.py:2208 sets spread_line = tue_open_home_spread; correct) gains `assert frame.attrs["model_line_source"] == "opener"` before scoring, set where `at_open` is built (clv.py:2208). Controls (intercept-only, line-move-only) emitted next to metrics are a separate src/cli change.
Files touched and pinned status: mass_preserving_lattice.py PINNED (PINNED_NAMES, independent_validation.py:40; edit needs re-enrol of vN in the same session per memory pinned-source rule); clv.py PINNED; prediction_safety.py PINNED (not needed). No other src file. The 28 `prior_pool(` call sites (scripts and src) keep working because `require_opener_through=None` preserves current behaviour only for the served path; scripts pass the season. Owner approval for pinned edits required; verify by running `nfl-ats opener-evaluation` once and lead66_unit2 (must now raise unless the SBR open map is supplied).

## Unit 2a reruns (2026-11-03 21:05Z, measured; scratch tests/scratch/eng51/{lead66_unit2_open,mod24_u8_grade_deadline_open}.py; out artifacts/eng51/lead66_unit2_open/, log artifacts/eng51/*.log)
Pool swap: pre-2020 lines = SBR open (sbr_odds.parquet), 2020+ = opener; close fallback and 1999-2010 dropped. Confound: smaller pool (lead66 secondary 1 of 2302 games skipped, empty center band; primary 2020-25 leg not rerun because its model center is recovered from the served probability under the production pool).
LEAD-66 secondary (original quoted backlog-all-2026-11-02.md:22, results artifacts/lead66_unit2/results.json): served v market RPS -0.37095 -> -0.37239 [-0.5517,-0.2312] P+ 0.0 (wrong sign persists); exact log score served v market +0.07088 -> +0.07743 [0.0603,0.0952] P+ 1.0; model-centred v served log +0.00017 (P+ 0.519) -> +0.00050 [-0.0070,0.0081] P+ 0.548. Survives, no change in conclusion.
MOD-24 U8 deadline (quoted acc-u8-prediction-markets.md:60-66, 2025, week-block): Kalshi market side v 50: +3.18 P+ .843 -> +2.81 [-3.82,9.19] P+ .794; market minus served -1.12 [-9.56,7.20] P+ .387 (same-day original -0.75 P+ .427; quoted +2.25 reflected an older served file); Poly +0.59 P+ .567 -> +1.18 [-5.79,8.39] P+ .613; Poly combined-minus-served -11.18 [-17.5,-5.2] P+ 0 (same-day original -11.76). Positive Kalshi direction survives at P+ ~0.8, unresolved. Original scripts rerun unmodified for the same-day baseline.
Records: 12 cells family eng51_open_pool_rerun (registry/weak_signals); 2 lead66 RPS-vs-market cells refuted_mechanism with wrong_sign_resolved (whole interval negative), rest unresolved_below_power. Looks: 12 mod24+lead66 cells declared by script.
Ruff: lead66 copy clean; mod24 copy keeps 5 inherited E501 long f-strings.
Next: rerun lead76_declared/rule_prior done; remaining leaky total_conditioned_lattice and altitude_fourth_quarter_screen need the same swap; 2b needs owner approval (pinned).
- 2026-11-03 16:12 ET (root decision): unit 2b (line_source column at mass_preserving_lattice.py:163, source assert in clv.py opener_pick_evaluation) touches two v7-pinned files. Production use of past games' closing lines in the pool is legitimate pregame information; the leak only biases research grades at the opener. Deferred: bundle with the next pinned change and re-enroll v8 in the same session. Meanwhile scripts/ research harnesses carry the guard (unit 2c: a scripts-side helper that refuses close-as-input at an opener grade).

## Unit 2c scripts-side guard (2026-11-03 21:12Z, measured)
scripts/opener_input_guard.py: assert_opener_input(frame, input_col, grade_col, close_columns=CLOSE_INPUT_COLUMNS, allow_close_control=None) raises CloseInputError when input_col is a close column (spread_line, spread) differing from an opener grade column on any row, unless allow_close_control names the study (prints it). Wired once in base.to_opener (scripts/mod24_more_games.py), which every walker reaches: mod24_more_games.walk, mod24_pooled_levers.walk, pooled_screen (via lev.walk), mod24_commonfeat, mod24_pooled_era_check. Env MOD24_DECLARED_CLOSE_CONTROL names the study; MOD24_CLOSE_INPUT=1 alone now raises (3443 of 4353 rows differ); with the env it runs.
Identity: mod24_more_games report.json byte-identical to HEAD version; pooled_screen ts output identical except wall seconds.
Reruns (scratch tests/scratch/eng51/{total_conditioned_lattice_open,altitude_open}.py; pool/line = SBR open pre-2020 + Tuesday opener 2020+, close fallback dropped, 2009-10 dropped; not quoted anywhere, no records).
total_conditioned_lattice log loss challenger minus served (original 2026-09-23 -> pool-open-only -> pool+target open): primary 2020-25 -0.00424 -> -0.00464 -> +0.00192; accuracy served/challenger .5158/.5114 -> .5126/.5114 -> .5010/.5030. Extended 2011-25 -0.00147 -> -0.00090 -> +0.00168. Challenger no better than served at the opener; pool-only fixes keep the sign, a true opener target flips it, both near zero.
altitude Denver-minus-league fourth-quarter residual (line-conditioned): close 2009-25 +2.218 [0.925,3.541] P+ 0.9997 (139 games) -> opener line 2011-25 +2.258 [0.935,3.614] P+ 0.99975 (136 games). Unchanged.
Next: owner-approved pinned bundle (2b) at next pinned change.

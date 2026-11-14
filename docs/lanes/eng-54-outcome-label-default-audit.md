# ENG-54

## Goal
Outcome-label default audit: every scorer that joins players or games to outcomes and labels a miss as a negative; widened 2026-11-14 to line-sign and raw-versus-served probability labels (ROADMAP ENG-54).

## State
Unit 1 (2026-11-03): artifacts/eng54/audit.csv, 17 joins, 8 measured. Id crosswalk safe (0.0002). Name key not: mod33 marks 5.86% of starter-weeks absent that played. coach_speak +0.213 pts P+ 0.9876 survives.

Unit 2 (2026-11-14 17:42, date). Detail: artifacts/eng54/label_audit_unit2.json (rows in label_audit_scan.json, label_audit_unit2_measure.json); helpers tests/scratch/eng54/{scan_unit2,measure_unit2,measure_unit2b,finalize_unit2}.py. Measured this session unless marked. Nothing served, recorded or repaired.
- Nine joins. Survive: mod22_unit5 name key (10 of 8294 QB snap rows match neither key, 0.10% of snaps); veteran_rest (95 of 214101, 0.04%); per15 missing game (9 of 34828 interior gaps, 0.03%); overlay flags year-one, backup QB, bye, division revenge, surface, tank zone (0 unmatched, 0 NaN on 1503 games, the fill is a no-op). mod33 team-week default never fires (0 of 3074), but its "prior week" is two or more weeks old on 5.9% of team-games. Not measured: cold-visitor, precip, predicted_open, home_flagged tables (need forecast or play frames); xlg06 mislabel (stats file has no name column; 12.0% of skill snap rows have no stat row).
- Does not survive: the weekly-roster pfr crosswalk drops 35.3% of snap rows (25.0% of snap share from 2020). sky01_drift_weighted.py:147 then gives weight 0 to 24.2% of Out or Doubtful rows 2020 on, and 19.5% of all such rows (1530 of 7856) have earlier snaps by name (injury rows as a stand-in for the script's listing table). Same crosswalk, read: per15_return_ramp.py:53-54,73-83 (injury-listed flag defaults to not listed) and xlg06_rookie_priors_screen.py:160-185 (35.4% of snap rows dropped).
- sim16: registry family sim16_pregame_attrition holds three cells from the name-key join, two with the whole interval below zero (P+ 0.004). Id join, n 1283: full -0.00093 [-0.00264, +0.00090] P+ 0.15; flat gap -0.00092 [-0.00233, +0.00069] P+ 0.122; hazard only -0.00028 [-0.00197, +0.00124] P+ 0.374. Decision: re-record all three with replace; classification stays unresolved_below_power.
- mod33 timing on ids is in the artifact (non-exit OL absent 11.1% by name, 4.5% by id; exit 66.1% and 61.2%); the script still has the name key.
- Convention (read src/nfl_ats/splash_lines.py:19, features.py:49, pool_hook.py:80): a home spread is positive when the home team is favoured; home cover is result minus spread above zero. Measured: home wins 67.4% when spread_line is positive; tue_open_home_spread correlates 0.969 with spread_line; margin_vs_open equals result minus line on 1503 of 1503.
- Cover scan: 791 scripts, 104 compute a margin against a line (152 rows), 59 cited by a registry family, 100 correct. Unclear: opener_error_transfer_eval.py:123, opener_error_transfer_unit2.py:126, opener_error_transfer_unit3.py:140 (result plus line on a college team-row line, right only if that line is negative for a home favourite). Opposite sign but self-consistent: sim05_whatif.py:28 (betting-notation constants, no dependents). Definite conflict (read): opener_error_transfer_unit3.py:173-184 puts the NFL positive-favourite line into home_favorite = home_open < 0, the reverse of its college frame; FEATURES (line 24) carries home_favorite and prior_move_diff.
- Served label: 104 scripts read a raw probability, 46 put a served label beside it, 37 of those never read the served table, 23 verified by line. Raw and served choose different sides on 27.6% of 1503 games. Six were already retested on the served base (cited by mkt28_served_base_retest).

Ranked repairs, none made:
1. Forward logs: mod54_forward_log.py:183-184 and mod25_wfj_forward.py:116,198 write the model-only forecast as served; rename the field or read out_of_season_home_probability. mod24_forward_log.py:135 and the armb and draftcap served_frames are not line-checked.
2. Dashboard: none found.
3. Recorded cells: sim16 replace; opener_error_transfer_v3 (fix unit3 line 173, rerun, replace); raw-base add-on cells never retested (mod42, mod43, mod44, mod47, lead76_rule_prior, mkt27: take z from out_of_season_home_probability, replace); sky01, per15, xlg06 crosswalk (use the players-table crosswalk as availability.py does); mod33 lines 25 and 166-205; sim05 sign.

## Tried
Regex scan of single-line margin-versus-line expressions plus a short-name grep; multi-line and unnamed operands are not covered. The pool_card to per_game comparison found no shared game ids, so pool_card as raw stays reported, not measured.

## Next
Unit 3: (a) `nfl-ats weak-signals record` with replace for the three sim16 cells; (b) settle the college line sign with corr(home_margin, home_open) in opener_error_transfer_unit3.py, repair line 173, rerun, replace opener_error_transfer_v3; (c) relabel the two forward logs and line-check the other three; (d) swap the crosswalk in sky01, per15, xlg06 and rerun per15.

## Open
Whether the 23 served-label scripts not verified by line mean the served mapping rather than the pick probability. Four flag tables unmeasured.

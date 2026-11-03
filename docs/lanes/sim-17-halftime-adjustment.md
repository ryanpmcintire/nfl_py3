# SIM-17

## Goal
Halftime adjustment: second-half minus first-half EPA/play per coach-team spell, persistent or not (ROADMAP SIM-17).

## State
Unit 1 measured 2026-11-03 (4:21 PM). Script scripts/sim17_halftime_adjustment.py, artifact artifacts/sim17/halftime_adjustment.json. Data: data/pbp/team_style/raw_pbp_narrow.parquet (nflverse pbp 2009-2025 REG) + latest data/raw/*/schedules.parquet coaches. Declared before looking: pass/run plays, no kneel/spike, qtr 1-4 (no OT), wp in [0.05, 0.95] (garbage-time filter, nflverse convention), half = qtr<=2 vs 3-4; delta = H2 mean EPA minus H1 mean EPA per team-game (8166 team-games), offense, and defense allowed (defense delta of A = offense delta of B, so league means coincide). Confounds fitted leave-one-season-out, weighted: halftime margin, |margin|, opponent season mean delta (leave-game-out); residual = delta minus fit. Spell = coach+team, >=16 games (125 of 275). Looks: 2 sides x 3 reliability tests = 6.
**measured** League raw mean H2-H1: -0.025 EPA/play (se 0.0056), weighted -0.010. LOSO margin coefs stable (sd of coef across folds < 0.01).
**measured** Offense residual: odd/even r 0.075 [-0.058, 0.217], P(r>0)=0.87, Spearman-Brown 0.14; season t vs t+1 r 0.183 [0.088, 0.280] (393 pairs, P>0 0.9995); between-spell variance 0.00922 vs shuffle-null mean 0.00728 (perm p 0.042; excess share 21%); r perm p 0.25.
**measured** Defense residual: odd/even r 0.199 [0.015, 0.373], P>0 0.98, SB 0.33; season t/t+1 r 0.114 [0.012, 0.226]; between var 0.00755 vs null 0.00562 (perm p 0.012; excess 26%); r perm p 0.024.
Permutation null = random sign flip of each game's delta (equivalent to shuffling halves within game).
**inferred** Reliability is non-zero on 3 of 4 persistence tests, small (spell mean sd about 0.09 EPA/play, roughly a fifth to a quarter of it real). No closing ground met; unresolved_below_power-style. Not an admissible weak-signals effect unit (reliability), so not recorded there.

## Tried
Single pass above. Not done: effect of spell mean on actual game outcome, whether adjustment is a coach vs QB/coordinator effect (spell confound with QB), shrinkage.

## Next
(1) Control: re-run with offensive coordinator or QB spell to see if it is the head coach. (2) If it holds, enter SIM-10 as per-coach second-half shift: in scripts/sim10_drive.py at the drive outcome draw site where the half index is known, add named parameter coach_h2_shift_off and coach_h2_shift_def, each = spell residual mean estimated leave-one-season-out and shrunk by the measured reliability (SB 0.14 off, 0.33 def; measured not tuned), applied to the per-play EPA/success draw for drives with qtr>=3. Do not edit while the Tier B chain runs.

## Open
Zero crossing closes nothing; one fitted probability selects side; weights chosen LOSO. Spell-vs-QB confound unresolved. Odd/even offense interval includes zero (P>0 0.87).

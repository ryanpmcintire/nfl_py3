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

## Unit 2 (carrier) 2026-11-03 16:24 EST
Script scripts/sim17_unit2_carriers.py, artifact artifacts/sim17/unit2_carriers.json. Same unit-1 filters/LOSO residuals; coordinators from latest data/raw/coordinators/*/coordinator_history.parquet (asof gameday+1d within season, fallback first observation of the season); QB = passer with most dropbacks in the filtered plays. Spell = person+team, >=16 games. Common sample = games where every compared grouping is kept (offense 5314 games, defense 6901). Looks 43 (reliability 15 own + 15 common = 30; variance-decomposition 13).
**measured** Offense odd/even r [95% CI] (common sample): HC 0.108 [-0.06,0.28] P>0 .90; OC 0.211 [0.06,0.35]; QB 0.290 [0.13,0.44]. Season t/t+1: HC 0.158 [0.03,0.28]; OC 0.248 [0.12,0.37]; QB 0.164 [0.04,0.28]. Excess between-spell variance share: HC .30, OC .33, QB .38. r-perm p: HC .17, OC .010, QB .009. Own samples agree (HC .075, OC .179, QB .174).
**measured** Defense (common): HC odd/even 0.189 [-0.01,0.36], t/t+1 0.128 [0.01,0.25]; DC 0.126 [-0.04,0.28], 0.123 [0.01,0.24]; r-perm p HC .027, DC .063.
**measured** Nested incremental variance (sign-flip null, adj. var in EPA/play squared): offense HC given OC -0.00000 (p .29, 13 df), HC given QB -0.00056 (p .76); OC given HC 0.00194 (p .001), OC given QB 0.00109 (p .015); QB given HC 0.00025 (p .08), QB given OC -0.00005 (p .23). Defense HC given DC -0.00004 (p .22); DC given HC 0.00033 (p .13).
**inferred** Offense: the head-coach signal is absorbed by OC or QB, not the reverse; OC carries it, QB is not separable from OC (QB given OC ~0, OC given QB p .015, but OC and QB spells overlap heavily; reliability alone favours QB slightly). Defense: DC absorbs HC (HC given DC ~0) but DC given HC is p .13, so HC vs DC is not separable (spells nest; only 14 HC df beyond DC). No closing ground met for any grouping (none has zero split-half reliability).

## Design for SIM-10 entry (no implementation)
Offensive second-half shift keyed by offensive-coordinator spell (fallback: QB spell is the declared alternate if OC is missing, 475 games lack OC), defensive shift keyed by defensive-coordinator spell (HC fallback when DC missing; 359 games). Each = spell mean residual leave-one-season-out, shrunk by that grouping's measured Spearman-Brown (OC 0.35 on common sample, DC 0.22; measured not tuned), applied at the drive outcome draw where qtr>=3. Do not edit scripts/sim10_drive.py while Tier B runs. Next: LOSO check whether the OC-keyed shifted draw predicts held-out-season second-half scoring better than the HC-keyed one.

## Open (unit 2)
OC vs QB not separable; HC vs DC not separable (nested spells). Coordinator table is Wikipedia revision-based (2009-2021 preseason only): mid-season changes before 2022 are missed.

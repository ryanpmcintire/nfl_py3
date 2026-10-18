# mod25e E118 margin-variance gap, where it accrues

## Goal
Localise the sim-real margin variance excess on base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekudwtpq2ugky (s11-13) by time window and scoring channel; size what the open-field TD excess (GEO pending) explains; name what remains. Read-only, no sim run. Scripts and outputs: tests/scratch/e118/ (dec, inc, chan, an, an2, band; *.txt, *.pkl).

## State (2026-10-09; measured; game bootstrap 90%, 300 reps; ~110 looks in 4 families: windows 7, channel budget 38, RE 22, drive/open-field/dispersion 30)
- Var(final) real 2009-17 221.7 v sim 236.7: +15.0 [+3.3,+26.8], P+ .98 (gate real 2011-17 is ~214, so the gap to the gate is ~22; era window moves the baseline 7.7). Cov(increment, final) sim-real: Q1 +1.5, Q2 +5.8 [+.5,+11.4], Q3 +5.6 [+1.1,+10.1], Q4 to 5 min +2.7, last 5 min -0.9, OT +0.3. Q2+Q3 carry 76%; clock work (last 5 min) is not where it lives.
- Channels (2228 of 2304 real games where a play-delta margin equals the robust final; 76 dropped for pbp score glitches): Var(M) +13.1 [+1.4,+23.7] = Cov(D_c, M) by channel: offensive TD, drive started own <25: +7.8 [+2.6,+12.2]; own 25-50: +7.4 [+2.6,+11.2]; opp 50-25: +.2; opp <25: -.3; FG -1.1; def TD -.8; special-teams TD -1.7 [-3.0,-.3]; safety -.3 [-.6,-.1]; after-TD points +1.8 [+.1,+3.2].
- Strength v noise (season RE as mod25e_analysis.season_re): strength -2.6 [-10.3,+4.9], noise +14.6 [+1.4,+27.3], P+ .97. Noise carries it. By channel noise: TD from own 25-50 starts +6.4 [+3.4,+9.1]; own <25 +2.7; all offensive TD +8.7 [+2.6,+14.5].
- TD counts: offensive TD/game 4.57 real v 4.80 sim (+.23 [+.16,+.29]); Var(home-away TD count)/E[count] .746 v .784 (+.038 [-.003,+.084], P+ .94). Within-team-season TD/game variance +.068 [+.008,+.132]; between-team-season (strength) +.002 [-.043,+.042]. TD/drive higher in every start bucket (+.012 to +.044), FG/drive lower (-.008 to -.015): share TD +.0137, FG -.0096.
- Open field: scrimmage TD/play at snap yl>20 .0109 real v .0144 sim (+.0035 [+.0031,+.0039]); +.38 TD/game. By band the excess is +35% in every band 20-30, 30-40, 40-60, 60+; inside 20: -.15 TD/game and -1.7 plays/game. Net +.23 TD/game = +.38 - .15.
- Arithmetic (inferred): per TD/game the margin variance moves 6.94^2 x .746 = 35.9 pts^2 (real dispersion). Mean TD excess .23 x 35.9 = 8.3 pts^2 = 63% of +13.1 (55% of +15.0). If GEO restores real TD/play at yl>20 and the lost drives continue into the red zone at the real rate, TD count returns to real (-.38 + .15). Remainder about +4.8 of Var(M) (SD ~15.1 v gate band top 14.87): TD-count dispersion +8.6 (P .94) less missing special-teams TD/safety/FG -3.9. Sign of remainder is not resolved.

## Tried
Checkpoint increments, channel budget, season RE per channel, drive-level shares, band split, dispersion (binomial phi .919 v .933, +.013 [-.015,+.046]).

## Next
1. After GEO=2 3-seed (new label), rerun tests/scratch/e118/{inc,an,an2,band}.py by editing LAB in chan.py/an.py; check Var(M) gap, offensive-TD cov, TD/game by band, RZ plays.
2. Remaining candidates: (a) pool-wide long-gain tail, neighbour-row TD/yards draw sim04_engine.py:996-1012 and clamp :1121 (GEO covers consistency, not the tail rate); (b) no kickoff-return plays in the sim log (code 6 rows score 0) and safeties .07 v .13/game: special-teams TD .57 v .76 pts, mod25e_stf.py:230-246 shifts starts only; (c) game-to-game TD variance with team latents weekly AR(1) fixed (mod25_generator.py:474-508, strength matched), so noise is play-level draw dispersion; ADJ (mod25e_adj.py:311-316) only pushes leaders down.

## Open
Real kept-set mean pts 44.66 v full 45.2 (glitch games likely higher scoring), so real Var(M) 223.6 on kept v 221.7 all. Real drive count 23.3 v e5 22.67 (extra short groups in fixed_drive). Sim log PAT after OT walk-off TD (+1 in 1.4% of games) is absent from sim_games; immaterial. unresolved_below_power for dispersion; no closure.

# mod25e E120 opening/half kickoff return TD channel

## Goal
Explain the special-teams TD gap in docs/lanes/mod25e-e118-sdgap.md (Cov(D,M) -1.7 [-3.0,-.3]) and add any missing mechanism as a default-off hook, fitted LOSO 2009-17, no constants tuned.

## State (2026-10-09; measured, game bootstrap 90%, base crHpqok...ugky s11-13, qtr<=4; scripts tests/scratch/e120/{meas,dvar,hook}.py)
- Per game, real 2009-17 REG v sim: receiving-team kickoff TD .0443 [.0373,.0504] v .0339 [.0324,.0356]; split real: opening+half kickoffs .0126 [.0095,.0156], after-score .0317 [.0260,.0378] (sim .0339 sits on after-score, so the whole gap is opening/half). Kicking-team kickoff TD .0061 [.0030,.0095] v sim 0 (KICK maps every rtd to the receiver). Punt TD (defense) .0725 [.0629,.0825] v .0784; punting-team TD .0039 [.0017,.0061] v .0003; FG/missed-FG def TD .0048 v .0062. Kick fumbles lost (any team): kickoff .062, punt .105/game; carried by empirical K/transition draws, no gap measured.
- Cause (read): engine draws game start and half start from opening_pool of first live-play yardlines, sim04_engine.py:755-761, :847, :1077; no kickoff play. KICK (mod25e_kick.py:124-195) only fires on scoring rows, excluding gsr 3600/1800 (mod25e_kick.py:63). STF/STB shift spots only. chan.py (e118) files KICK rows with code 0/1 as deftd, so the sttd channel gap is partly relabelling.
- Hook scripts/mod25e_rtd.py written (RTD=1), fit artifacts/mod25e3/rtd/fit.json, import and mock-drive verified (rate .0140 events/game v fit .0135). Fit: 4598 opening/half kickoffs, recv TD 29 events, kicker TD 2. Forms const/half/k35/tb25/tb25_half x 2 events = 10 looks. recv chosen k35 (2011 kickoff to the 35): LOSO dll +7.53 v const, fold wins 7/9, coef -1.49; half -.68, tb25 +1.00 (3/9). Kicker TD: nothing resolvable (2 events), tb25 +.28, rate .0005. Pooled served rate .0062 per kickoff.
- Effect on Var(final) (inferred, MC 200 reps on s11-13 sim games, replacing the first drive of each half): +0.19 pts^2 (sd .10, 5-95 [.03,.34]), P+ ~.97; about 1.3% of the +15.0 gap. It cannot close the gap.

## Tried
Real event count by kickoff type, sim equivalents, LOSO rate fit, variance arithmetic.

## Next
1. Owner/orchestrator wires crH (see below), runs one smoke seed to see rtd_ev rows and Var(final); expect +0.2 only.
2. Reclassify sim KICK rows in tests/scratch/e118/chan.py (po>0 & pdf>=6) as sttd before reading the channel budget again.
3. Punting-team TD (.0039 v .0003, 6 pts) is unmodelled; tiny, queue only if channel work resumes.

## Open
Hook not yet run in a sim; the first-play detector relies on clock_val 3600/1800 with prev != clock (repeat plays at zero elapsed excluded). ST-form covariates not used (STF latents have no real analogue). Era enters pooled across 2009-17 to match the real pool, not a served-era setting.

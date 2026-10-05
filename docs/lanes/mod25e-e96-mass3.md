# mod25e-e96-mass3

## Goal
Name the mechanism behind the sim's short final-margin mass at |m| 3 (sim .115 vs real .141), 7 (.075 vs .092), 14 (.033 vs .048) on base crHpqokgndecsmfwtjo2as2 s11-13. Parent: docs/lanes/mod25e-generator-fidelity.md.

## State
2026-10-05 E96 done, analysis only. scripts/mod25e_e96.py, output artifacts/mod25e3/e96/e96.txt. Real REG 2011-17 (1792 games) vs sim s11-13 burn 2 (39168 games); game bootstrap 300; 373 looks, one family, all cells printed. Event sums match final margin in 95.7% real / 94.7% sim games (rest are rows the pbp score fields miss; real n_oth .036/game).
Measured:
- mass3 real .1412 [.127,.155] sim .1152 [.112,.118], s-r -.0260 [-.041,-.011] P(s-r>0) .00. Cell dTD0 dFG+1 no flags (clean one-FG margin) carries -.0232 (real .0882 sim .0650). Last score = FG by winner: real .0887 sim .0548 (-.034 [-.047,-.023]); last-TD cells sim >= real.
- Late/OT, not carried in: P(final=3 | tied entering last 5 min) real .625 [.52,.73] sim .434 [.41,.46]; at end of Q3 tied cell and bucket 1-7 gaps are not resolved (P .13-.33). Entering-margin mix explains -.0084 of the last5 gap, conditional rate -.0170 (both P(>0)<=.02).
- Overtime: P(tied after regulation) real .060 sim .056 (equal). P(final=3 | reg tie) real .682 [.58,.78] sim .432 [.41,.45], gap -.251. Mass3 from OT games -.0168 [-.025,-.008] = 65% of the gap; regulation-decided games -.0092 [-.022,+.005] P(s-r>0) .13. Sim OT games end |m|: 0 .088, 2 .005, 3 .440, 6 .467 (measured from sim_games). Real OT share of 6 inferred ~.25.
- Mass7 gap -.0164 and mass14 -.0146 are entirely regulation: mass 7 cell dTD+1 dFG0 clean -.0198; last td by loser -.0123 (late trailing TD); mass 14 last td by winner -.0129 (real .0240 sim .0111). Conditional rate in each entering bucket is lower in sim at both windows (mix terms P ~.2-.8).
- Composition: FG share of TD+FG real .395 sim .370 (-.025 [-.035,-.018]); lower in every score state (-.01 to -.04). Safeties/game .075 vs .033. No-point-after per TD real .061 sim .091; PAT made per TD .911 vs .877; 2pt per TD .028 vs .032. TD per game +.12, FG per game -.27.

## Tried
Composition cells, last-score type, entering-margin buckets (300 s, 900 s), OT split.

## Next
1. Candidate mechanism: OT. Sim OT ends on a TD (.467 at 6) far more than real (FG-first-possession rule and OT 4th-down/FG propensity); fit OT FG-vs-TD ending from real 2011-17 OT drives, LOSO. Check the sim OT rule vs real modified sudden death.
2. Regulation residual (-.009 mass3, mass 7/14): FG propensity too low in every score state (FG share -.025) and missed-PAT rate too high (.091 vs .061); late trailing-team TD (mass 7) and leader late TD (mass 14) too rare. Unresolved whether one late-game clock/score-state mechanism covers these.
3. Do not touch any constant; fit PAT-miss by era (rule change 2015) first, since pooled real mixes 2011-14 and 2015-17.

## Open
Real score-event reconstruction uses pbp score fields; 4% of games do not reconcile.

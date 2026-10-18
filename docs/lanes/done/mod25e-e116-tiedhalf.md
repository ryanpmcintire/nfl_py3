# mod25e E116 tied-at-half Q3 margin variance (real 33 v sim 51)

## Goal
Decide if the tied-at-half Var(Q3 margin change) gap (1.45 of the 5.04 Q3 excess in docs/lanes/mod25e-e111-q3var.md) is noise, composition, or an engine channel. Read-only; base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku seeds 11-13 v real 2009-17 REG. Scripts tests/scratch/e116/{a..h}.py, late.pkl, q3games.pkl. 90% game bootstrap.

## State (2026-10-09, measured)
- Not small-sample noise in 2009-17: tied n=178 Var 33.1 [27.3,38.7] v rest 52.2; sim tied 51.4 [49.2,53.6]; sim-real +18.3 [+11.9,+24.0] P>0 1.00. Permutation (random 178 of 2304) P(<=33.1)=0.0004; pooled 17 seasons tied 39.0, P=0.003.
- Not replicated 2018-25: tied n=152 Var 46.1 [37.4,54.4] v rest 46.5. Tied-minus-nontied excess (sim-real): 2009-17 +14.5 [+7.6,+21.1], 2018-25 -3.8 [-13.8,+5.5] (P>0 .25), pooled +5.9 [-0.1,+11.4] (P>0 .945). Eras disagree in sign.
- Adjacent bins (sim-real): |m| 1-3 +1.8 [-4.7,8.2]; 4-7 +6.5; 8-14 +2.6; 15+ +4.4. Only tied is off.
- Composition: tied share 7.7% real v 7.8% sim; 1H total 19.5 v 19.9; reweighting sim to real 1H-total bins leaves 51.4. Real |spread| tied 5.38 v rest 5.28 (perm P .65); sim latents not saved for this base (budget path), sim strength gap unobservable. 2H receiver: receiver-oriented d mean .46 v .52, var 32.9 v 51.1 in both receiver groups; not a channel.
- Decomposition tied (sim-real): Var d +18.0 = Var A +7.5 + Var B +5.6 + (-2 Cov AB) +4.8 (Cov real -0.15 v sim -2.53; rest -3.25 v -3.11). TD points +16.6, FG pts +0.2, safeties ~0. E pts/game 8.73 v 9.84 (+1.12 [.41,1.87]); scaling sim var to real level removes about 5.8 (inferred), rest -> 0.1. Var nTD per team .36 v .50 (A), .35 v .46 (B) v binomial(n drives, TD/drive) ~.39 real, .46 sim: real tied is under-dispersed, sim and real-rest are 10-16% over-dispersed (inferred). Drives A+B 5.42 v 5.78 (+0.36 [.20,.55]) v rest +0.20; Var drives 1.85 v 2.26. TD pts per TD 6.96 v 6.99, FG pts var equal.
- Geometry: not the channel. TD/drive +.020 [-.001,+.039] tied v +.017 rest, same population-wide excess GEO targets; TD value and FG variance equal.
- Look count: 6 half-margin bins x 3 sets = 18 headline looks (plus ~14 e111 splits), ~100 stats overall; 2009-17 tied P=.0004 x 20 = .008.

## Tried
Bootstrap and permutation of tied Var; 2018-25 replication; adjacent bins; 1H-total reweight; spread and total-line terciles; receiver; A/B team decomposition by TD/FG/safety, drives, covariance.

## Next
No engine channel found, no hook specified. Replicate the decomposition (cov A,B; nTD dispersion) on 2018-25 plays and on 2009-17 half-state tied v 1-3 to see whether under-dispersion is a tied effect or 2009-17 draw. Record with `nfl-ats weak-signals record` as unresolved_below_power (no closing ground) when the orchestrator writes it up. Do not tune.

## Open
- Is real tied-game scoring under-dispersed (game-level scoring-propensity heterogeneity shrinks when the half is tied) while the sim keeps a fixed per-game strength? Test: LOSO game-level beta-binomial TD dispersion on real Q3 by half state v sim.

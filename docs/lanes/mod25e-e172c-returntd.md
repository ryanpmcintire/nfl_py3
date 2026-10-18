# E172c return-TD excess

## Goal
Find why the sim's defensive return-TD share of run/pass turnover rows is 4.36-4.52% v real 3.40%.

## State (measured, tests/scratch/e172c, e172)
- Excess is a uniform within-state level lift after the draw, not field mix (real rate at sim mix 3.44 [3.23,3.65]), not defense gradient (per-SD slope matches, -0.86 v -0.83 %/SD), not kernel bandwidth, not Geo selection (E172d/e/h).
- Ladder by version (s11): B 3.71 (3 seeds 3.77), g2 4.36, X2 4.44, base 4.52. GEO=2 is the biggest step (+0.66); RTD=1 is not a carrier (mod25e_rtd.py:139-196 opening kickoffs only). Pool 3.52, K-set 3.9-4.05.
- E172j paired logs B v g2 s11: NOT pairable. Only 41 of 8,142 games share a state and differ in idx at the same position, and 2 same-state same-idx rows exist; RNG streams diverge from play 1. The planned play-by-play match is infeasible.
- E172j unpaired by yardline band (B v g2, s11; turnover per 1000 rows / return-TD per 1000 rows): yl 20-40 77.6/1.76 v 67.2/1.85; 60-80 32.2/3.02 v 29.3/3.37; 80-90 29.8/3.22 v 26.8/3.85; 90+ 34.5/4.34 v 40.5/6.87. g2 lowers turnovers per row 5-13% at yl 0-90 and raises them 17% at yl 90+, while return-TD per row rises 13-58% at yl 60+. Share 3.705 +-0.11 (B, 109,863 turnovers) v 4.361 +-0.13 (g2, 101,889).
- Mechanism under test: GEO=2 repicks every run/pass play (mod25e_geo.py:438 always, fix force=True) from a fp-window candidate set where every free row (flip or points_def, not TD) passes unconditionally (mod25e_geo.py:344) while non-free rows need yards+shift < q; free rows include all returned turnovers.

## Tried
Gap-conditioned share, bin decomposition, Geo candidate replay (K-set 3.2-3.6 per turnover), symmetric window, weight-stage replay, state-mix standardization, defense gradient, paired B v g2 play logs (infeasible).

## Next
1. Rerun tests/scratch/e172/p4.py in background (first run killed at cap, no output; takes over 2 min): candidate-set v window v K-set return-TD and flip rate per row by q band decides whether line 344 free-row admission creates the per-row lift.
2. If confirmed: default-off module that restricts free-row admission to the fp-window rate (no unconditional pass), fitted on real window rates, same-state paired by hand against g2 only after owner lifts sim freeze.
3. SAFEG first; crH patch deferred (another agent owns scripts/mod25e_crH.py). TILTL does not fix this.

## Open
Why returned turnovers per row fall 5-13% at yl 0-90 under g2 while return TDs rise at yl 60+.

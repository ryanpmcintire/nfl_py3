# mod25e-e100-seedboot

## Goal
Decide whether the combined base ypw2xfdcd (PATY+FGD+CDR on ypw2) differs from ypw2 beyond seed noise (strength +4.1, SD +.16 unresolved in docs/lanes/mod25e-e98-regkick.md), and re-measure the E96 mass-3 state cells on both. Parent: docs/lanes/mod25e-generator-fidelity.md.

## State
scripts/mod25e_e100.py written (analysis only, no sim reruns). Part 1: season bootstrap within seed (1000 reps, seed-mean of 3 per rep, season_re fixed per season) of margin_sd, noise, strength, xq cov, q4 slope, mass3, pts/g; late r2 only seed-level (play epa not saved). Part 2: E96 cells via mod25e_e96 with LABEL swapped. Outputs artifacts/mod25e3/e100/{e100.txt,part1.json,part2.json,run.log}.

## Tried
Ran once (measured, e100.txt; point estimates reproduce era.txt). Deltas cand-base, 90% season bootstrap, P(>0): SD +.166 [-.020,+.349] .93; noise +1.2 [-2.6,+4.6] .69; strength +4.09 [+.01,+8.39] .95; xq cov +3.0 [-1.5,+7.5] .87; Q4 slope +.006 [-.0004,+.013] .94; mass3 +.0030 [-.0007,+.0068] .90; pts/g +.11 [-.18,+.38] .74; late r2 +.0136 [+.002,+.024] .96 (3 seeds only). Per-seed strength deltas +5.2/+6.4/+0.6, SD +.13/+.33/+.04. E96: P(final3|tied entering last 5) real .625 base .474 cand .482 (cand-base +.008 [-.02,+.03]); OT share of mass3 real .289 base .268 cand .273; tied after reg real .060 base .058 cand .059.

## Next
Verdict (inferred): KEPT as a mechanism base (PATY/FGD match own real play behaviour); no aggregate moved resolvably toward or away from real except strength (+4.1, P .95, away from 58.0) and late r2/mass3/Q4 slope (toward real, P .90-.96). Next: find why strength rose (CDR vs PATY/FGD ablation by seed is the unit), then OT/last-5 tie mass3 gap (.48 v .625) is unmoved. CDR snap gsr<=5 share v real .39 from synced logs remains from the E98 lane.

## Open
Late r2 has no season-level resample (3 seeds only).

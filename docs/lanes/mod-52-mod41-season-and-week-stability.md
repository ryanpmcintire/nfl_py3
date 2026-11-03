# MOD-52

## Goal
MOD-41 stability profile: coefficient and gain by season, week block and spread band before any serving read (ROADMAP MOD-52).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: A real load effect should show a stable coefficient across seasons and weeks; a one-season artifact or a late-season-only effect would change how it is served. Fitted term: season and week-block interaction on the frozen coefficient, not a gate.

## Tried
Novelty check only. Nearest: MOD-41 units 1-3 (pooled and LOSO folds only), MOD-47. Differs: a stability profile of one frozen term, with no new feature.

## Next
First unit for one agent: Report the frozen coefficient and log-loss gain per season 2020-25, per week block (5-8, 9-13, 14-18), and per spread tercile as a stability profile with permutation nulls on the lopsided splits, counting every cell as a look.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03 17:50 local
scripts/mod52_stability.py, artifacts/mod52/unit1_stability.json. LOSO base and term predictions from unit 6 harness (usable 1142 games, weeks 5-18; weeks 1-4 are the feature warm-up). Week blocks 5-8, 9-12, 13-18: 14 post-warm-up weeks, thirds are 4.7 weeks, nearest whole-week edges give 4/4/6. Bootstrap by week-block, 2000 draws; homogeneity = permutation of cell labels on per-game log-loss differences (2000). Overall usable: gain +.0054 [-.0013,.0118] P+ .947, decisive 156-130.
- Season (gain, decisive): 2020 +.0054 27-20; 2021 +.0057 31-34; 2022 +.0104 27-25; 2023 -.0013 26-23; 2024 +.0037 20-14; 2025 +.0087 25-14. Homog p .91. 5/6 positive.
- Week block: 5-8 +.0069 47-33; 9-12 +.0057 43-37; 13-18 +.0041 66-60. Homog p .93; slope -.0014/step [-.0090,.0066] P+ .37 (perm p .72).
- Spread tercile (|opener|, edges 3.0/6.5/19.5): small -.0044 46-54; mid +.0066 56-37; large +.0139 [+.0030,+.0250] 54-39. Homog p .063.
- Direction: favours away +.0026 64-51; favours home +.0083 92-79. Homog p .37.
- |term| tercile (edges 1.98/4.67/18.7): low -.0002 16-16; mid +.0028 60-42; high +.0135 80-72. Homog p .17; slope +.0068/step [-.0022,.0164] P+ .94 (perm p .078).
Looks: 17 cells + 5 homogeneity + 2 slopes = 24. Broad across season and week block (homogeneous, 5/6 and 3/3 positive); concentrated toward large spreads and large |term| (not rejected at .05, spread homog p .063). Not recorded: no admissible unit for homogeneity statistics.
Next: record per-season gains if wanted; spread-tercile interaction as fitted term (term x |spread|, LOSO) as a separate declared unit.

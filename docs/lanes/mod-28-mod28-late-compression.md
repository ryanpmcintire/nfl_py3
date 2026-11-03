# MOD-28 Team late-game margin compression

## Goal
Test whether a per-team late-game margin compression trait is reliable and, if so, adds to the cover probability on large spreads. ROADMAP row MOD-28.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Some teams and coaches run the clock and shorten the game when leading, others keep scoring; that moves the final margin across the cover without changing who wins. Per team-season trait = mean change in margin over the final quarter given the team entered it ahead by 9 or more, built only from prior games; a fitted pregame term on favourites of large spreads. Data: nflverse play-by-play 2009-2025 already local (data/raw/nflverse_pbp_full, game_features_pbp).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: mod25e-e175 and e131 (league-level leader Q4 scoring in the simulator, not a team trait), big_spread_diagnosis.md and MOD-18 (spread-size regime, not a team trait), mod23-unit10r garbage-time label (target denoising, not a feature). No grep hit for a per-team compression or backdoor-cover trait.

## Next
Build the trait table from pbp (entered-Q4-ahead-by-9+ games, prior games only) and run split-half reliability with a team-permutation null. First unit: First unit: split-half reliability of the trait across seasons and within season (permutation null over teams), before any ATS term; only if reliability is above its null does a LOSO term get fitted and graded at the opener.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Large-spread favourites are about 25 percent of games, so the term may be below power; record it unresolved rather than closing it.

## Declared before outcomes (2026-11-03, family mod28_coasting_trait)
Script tests/scratch/mod28/m28.py. Leader = offence starting a Q4 drive up 8+ (kneels, spikes excluded). Traits per team-game: pass_oe, seconds per play (sign flipped), points per drive minus yardline-expected; composite = equal mean of the three scaled by measured between-team-season SD. As-of = prior games of current+previous season, shrunk n/(n+k), k = within/between variance from ANOVA on seasons before the target season. Reliability looks: 3 traits odd/even 2012-25, within-season team permutation null (2000), SB. ATS looks: 4 traits x {abs spread >=7, all} = 8 LOSO 2020-25 fits of one logit coefficient on offset of base home_cover_probability_excluding_push (margins/20261103T172959Z), x = favourite trait signed to home; grade log loss vs base and vs 0.5.

## Run state
m28.py launched in background (task bx46oyb7c) writing tests/scratch/mod28/out.txt and result.json; no results read yet. Next: read result.json, then `nfl-ats weak-signals record` each of 3 reliability cells and 8 ATS cells (family mod28_coasting_trait, effect_units log_loss_improvement / correlation, reliability field set). Nothing recorded yet.

## Result (measured, tests/scratch/mod28/result.json)
Trait coverage is 2018-2025 only (1291 team-games; earlier pbp lacks pass_oe or scores), not 2012-25. Split-half SB, n=235 team-seasons: pass_oe 0.33 [0.18,0.46] perm p 0.001; seconds/play 0.31 [0.12,0.45] p 0.007; points/drive 0.20 [0.04,0.34] p 0.09. LOSO 2020-25 dLL/game vs base (negative = worse), |spread|>=7 n=1374: oe -0.0081 P+ 0.00; sec -0.0007 P+ 0.42 (coef +0.08..0.25 all positive folds); ppd -0.0036 P+ 0; comp -0.0074 P+ 0. All games n=4746: oe -0.0007, sec -0.0013, ppd -0.0010, comp -0.0006 (P+ 0.07). Permutation p >= 0.71 everywhere. No cell beats base. Seconds/play is the only one with a consistent-sign coefficient, unresolved_below_power.
Next: record 11 cells via weak-signals record (reliability set; no closing ground, since reliability is nonzero and no wrong sign is resolved); extend trait to 2009-17 if pbp fields allow.

## Unit 2 state (2026-11-03)
Script moved to scripts/mod28_coasting_trait.py (ruff clean; --out arg; adds reliability_ext on 2009-2025 ordered by game_id, and p_pos_boot per reliability trait). Reproduction run writing tests/scratch/mod28/result_repro.json in background (python scripts/mod28_coasting_trait.py --out ...; output repro_out.txt). Batch draft tests/scratch/mod28/record_batch.json (11 cells; reliability cells' probability_positive still null: fill from result_repro.json reliability[t].p_pos_boot). Next: when result_repro.json exists, confirm reliability/cells equal result.json, fill P+, run `nfl-ats weak-signals record --batch tests/scratch/mod28/record_batch.json`, then record reliability_ext as a new look (family mod28_coasting_trait, seasons 2009-2025).
Update: first repro run matched result.json exactly (reliability and 8 cells). Extension 2009-2025 (n=492 team-seasons, ordered by game_id): SB pass_oe 0.387 [0.29,0.48] perm p 0.000; seconds/play 0.147 [-0.02,0.29] p 0.036; points/drive 0.149 [0.03,0.26] p 0.081. p_pos_boot was missing from that run (sed fix applied); rerun in flight to result_repro.json. Then fill reliability P+ in record_batch.json, record batch, and add 3 ext cells.
DONE 2026-11-03: rerun reproduced result.json exactly. Recorded 14 cells (11 declared + 3 ext reliability 2009-2025, P+ oe 1.0, sec 0.932, ppd 0.984) in family mod28_coasting_trait via tests/scratch/mod28/record_batch.json, all unresolved_below_power. Ext cells have no sample_games. Next: ext-coverage trait for LOSO fits needs prediction rows pre-2018 (none); MOD-28 fits stay 2020-25.

## Unit 3 (2026-11-03) proxy 2012-19 grade
scripts/mod28_proxy_2011_2019.py (ruff clean) -> tests/scratch/mod28/result_proxy.json. Base = cached SBR-opener discrete p (artifacts/mod24_postseason_rows/base_2011_2019.parquet). 2011 excluded (<3 prior seasons): n=1981 of 2231. LOSO pass_oe term, 8 folds, 2 looks. |spread|>=7 n=549: dLL -0.00227 [-0.00475,+0.00009] P+ 0.058, record 300-249 = base, 20 flips 10-10, coef_all +0.030, in-sample gap 0.0024. All n=1981: dLL -0.00046 [-0.00084,-0.00011] P+ 0.015, 1022-959 vs 1023-958, 29 flips 14-15, folds mixed sign. Not pooled with 2020-25: base differs (served margin model vs SBR proxy). Recorded 2 cells (_2011_2019), unresolved_below_power, no closing ground. Next: none required; seconds/play on proxy needs sec fit if pursued.

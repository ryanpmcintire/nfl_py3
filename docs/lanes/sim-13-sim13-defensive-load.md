# SIM-13 Defensive snap load

## Goal
Measure whether plays defended so far shifts late-game scoring beyond team strength. ROADMAP row SIM-13.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: A defence that has been on the field for many plays against tempo has fewer effective snaps late; the simulator treats each drive as independent of plays defended. Measure late-game points per drive against plays defended so far, team strength held fixed. Data: nflverse pbp 2009-2025 (local).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: sim09_personnel_catalogue.md B1 (listed only), mod25e-e182-pace (pace as a rate), mod25e-e175 (leader Q4 gap). No measured fatigue term.

## State (unit 1 done 2026-11-03)
Script scripts/sim13_defensive_fatigue.py, cells artifacts/sim13/cells.csv. Family sim13_defensive_fatigue, 28 looks declared before outcomes (outcome epa,succ x spec main,placebo x scope all-seasons Q1-Q4 + fit 2009-17 + held 2018-25 all-quarters). Offence plays 2009-2025 REG, game-by-defence fixed effect, controls CT (strength as-of, score z, down/dist/field, elapsed) plus defence own-offence snaps and minutes; game-cluster bootstrap 400. Per 10 extra snaps defended at drive start:
main epa +0.0224 [0.0089, 0.0377] p+ 1.00 (fit +0.0236, held +0.0220 p+ .98); succ +0.0055 [-0.0001, 0.0101] p+ .97. Placebo (offence own snaps) epa -0.0016 [-0.0136, 0.0094] p+ .39; succ -0.0089 [-0.0131,-0.0051].
Quarter splits are not interpretable: main and placebo both +0.12 to +0.21 EPA in Q1-Q3, -0.07/-0.09 in Q4 (within-quarter clock trend not absorbed by whole-game el, el2); difference main-placebo in quarters is about zero. Not a clean per-quarter signal.
Verdict: small same-sign fatigue effect in both eras, placebo flat; recorded unresolved_below_power (2 records).

## Engine hook spec (for the engine owner; not applied)
Where: Pool.draw, sim10_drive.py:1200-1318; the neighbour pick is the weighted draw at 1317 (cumsum w, rng.random). Add a per-drive state d = defence snaps so far (count from the sim's own plays), enter as an extra kernel dimension or an additive EPA/success shift on the picked play only through weights, w *= exp(beta*(d/10)*feature) where beta is the leave-one-season-out value +0.0224 EPA per 10 snaps (17 folds 0.0186 to 0.0243, sd 0.0017, all positive; units epa_per_play). With beta=0 or flag off, skip the reweight and keep the single rng.random() call so RNG stream is unchanged. State resets per game. Beta chosen LOSO (done, unit 2). Dose is nonlinear: slope concentrates above about 38 snaps defended, so a linear term overstates early and understates late; check e5 aggregates.

## Unit 2 (2026-11-03, 25 more looks, same family; python scripts/sim13_defensive_fatigue.py unit2)
Clock control (quarter FE x within-quarter clock): main +0.0319 [0.0183,0.0471] p+ 1.00; placebo +0.0103 [-0.0017,0.0225] p+ .95; gap ~0.02. By quarter main/placebo still move together (Q1 .17/.21, Q2 .20/.17, Q3 .12/.14, Q4 -.07/-.09): quarter splits still uninterpretable.
Drive points against per 10 snaps: main -0.104 [-0.194,-0.019]; placebo -0.252 [-0.315,-0.196]; scored: -0.019 vs -0.049. No drive-level fatigue read; main minus placebo +0.15 points, interval not computed.
Dose (knots 17, 38 snaps): main slopes -0.007, +0.005, +0.042 [0.028,0.058]; placebo -0.024, -0.020, +0.015. Nonlinear, top tercile only.
LOSO beta 0.0224 mean, sd 0.0017, 17/17 positive; held-out SSE gain positive in 17/17 folds (total 930).
Recorded 3 unresolved_below_power. Artifacts artifacts/sim13/cells_unit2.csv, loso.csv.

## Next
Engine owner applies hook behind a flag with beta above, ideally piecewise (top tercile); check e5 aggregates. Possible: drive-level interval for main minus placebo.

## Open
Opponent and defence strength fixed as-of; possession imbalance conditioned via own-offence minutes. Effect is about 1 percent of an EPA play, tiny.

## Root adjudication 2026-11-03 13:22
- Hook NOT ready. With the within-quarter clock control the placebo (offense's own snaps) rises to +0.0103 (P+ .95) while main is +0.0319, quarter splits still co-move, and next-drive points show the placebo more negative than main: part of the "fatigue" is game-state/clock confounding shared by both counts. The proposed beta 0.0224 is the unit-1 (whole-game clock) spec, not the controlled one. Estimand for any hook = main minus placebo, measured directly (paired, same plays), above the dose knot where it lives (38+ snaps).

## Unit 3 (2026-11-03, 22+3 looks, same family; scripts/sim13_unit3.py, artifacts/sim13/cells_unit3.csv, loso_unit3.csv)
Paired contrasts on the same 550,318 plays, defence-game FE, game-cluster bootstrap 400, per 10 snaps (epa_per_play).
- Placebo 2 (opponent offence's own snaps) is identical to main by construction within a defence-game (offence snaps so far = snaps defended); contrast is exactly 0. Not separable.
- Baseline controls: main +0.0112 [0.004,0.018]; own-offence placebo +0.0420; main minus placebo -0.031 [-0.052,-0.011]. Above 38: -0.004 [-0.027,0.017] p+ .39.
- Strong controls (drive-index FE, score diff x time): main -0.022, placebo -0.179, difference +0.157 [0.131,0.184]; above 38 +0.163. Sign flips with controls; main and placebo are 98 percent collinear (R2 on other regressors .977/.986 baseline, .981/.989 strong), so the difference is not identified.
- LOSO difference beta 17/17 folds 0.150-0.163 (strong), but it is the collinearity artefact; not a hook beta.
Verdict: separate defensive-load mechanism NOT supported; hook spec unchanged and still not ready. Recorded 2 unresolved_below_power. Next: need exogenous variation (e.g. tempo shocks, turnovers/long drives) to separate defence load from game progression.

## Unit 4 (in flight 2026-11-03, scripts/sim13_unit4.py, ruff clean)
Exogenous-shock design, family sim13_defensive_fatigue. Shock A = defence's own previous drive <=3 plays or turnover; B = opponent's drive two back, residual of plays on as-of strengths (LOSO fit), in residual sd; placebos pA = opponent's own previous drive short/turnover, pB = defence's previous drive residual (opponent's own rest). Note A and opponent rest are the same event by construction; pA/pB are the nearest separable proxies. Play-level EPA/succ regression on opponent next drive with CT controls, quarter dummies, score x time; game-cluster bootstrap 400; LOSO betas. Run launched in background, output /c/Users/Ryan/AppData/Local/Temp/u4.log, cells artifacts/sim13/cells_unit4.csv, loso_unit4.csv. Not yet read; nothing recorded in weak-signals.
Next: read log, record via weak-signals (epa_per_play, unresolved_below_power), write verdict here.
Unit 4 result (16 looks, 444k plays, 2009-2025): EPA/play joint with placebos. A short-rest -0.0046 [-0.0152,0.0057] P+ .20 (wrong sign for fatigue); B accumulated load +0.0007 [-0.0043,0.0060] P+ .56 (main_only B +0.0036 [-0.0004,0.0073] P+ .95). Placebos pA -0.0106 [-0.0210,-0.0001], pB -0.0026. Differences A-pA +0.0059 [-0.0081,0.0214] P+ .80; B-pB +0.0033 [-0.0040,0.0101] P+ .81. Success: B +0.0021 P+ .99, B-pB +0.0027 P+ .975. LOSO (17 folds): B-pB 17/17 positive, 0.0015-0.0049; A-pA 17/17, 0.0039-0.0080. Verdict: mechanism NOT identified; magnitudes <0.005 EPA/play, unresolved_below_power (2 recorded: A_joint, B_joint). Hook spec stays unready.

## Root decision 2026-11-03 13:30
- Parked, unresolved_below_power. The exogenous accumulated-load shock leaves a small consistent effect (B minus placebo +0.0033 EPA/play [-0.0040, +0.0101], P+ .81; success P+ .975; LOSO positive 17/17, every fold under 0.008). At that size it is below the simulator's resolution (SIM-07 measured seed noise: aggregate gates cannot see ~0.1 pt/team-game), so no engine hook. Reopen only with player-level snap load (SKY-02 tracking) or a sharper exogenous shock.

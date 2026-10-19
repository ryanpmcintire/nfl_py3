# served-probability-calibration-2026

## Goal
Owner asks why weekly picks are getting worse (W3-6 28-32 vs 32.98 expected). Test: (1) served probability overconfident? (2) first-year-coach term hurting?

## State
- W6 graded on nflverse games.csv: W DEN, CLE, CAR, HOU, NO; L LA, BUF, ATL, DAL, LAC, NE, TB, IND (5-8; MNF pending). Season 51-41, exp 50.67.
- Q1 measured (scripts tests/scratch/served_calib/hist.py, live.py; per_game artifacts/pick_probability/20261018T162018Z/per_game.parquet, LOSO refit with pick_probability_fit._fit_logit): LOSO 6 seasons n=1503 picked-side mean p .558 vs hit .574 (se .013), model-only mean .560 vs hit .534. Log loss .6828 vs .6931 (0.5) vs .6969 (model-only); Brier .2448/.25/.2517. Calibration slope .952. Bins hit vs mean: 50-52 .574/.511; 52-54 .561/.530; 54-56 .562/.549; 56-58 .549/.570; 58+ .607/.623. Live W1-6 n=92: mean .551 hit .554 (se .052), LL .6761, Brier .2416; 50-52 8-14, 52-54 7-5, 54-56 16-8, 56-58 9-9, 58+ 11-5. Not overconfident.
- Q2 measured: one shared flag_sum coefficient (no coach-specific fit). LOSO held-out LL gain of coach term +.00188/game (se .00132), by season +.0027 +.0027 +.0029 +.0026 +.0018 -.0011 (2020-25). Held-out coach-fired n=232 acc .599 with vs .543 without; side changed 87 games, served right 50 (.575). Live W1-6 coach fired n=31, 12-19 (exp 17.4); W4-6 14 fired: served 4-10 vs model-only side 9-5; coach term changed side (approx counterfactual, coef .26) in 7 W4-6 games: served 1-6.
- Orchestrator re-ran live.py (measured): coach fired W4-6 n=14, served 4-10, model-only side 9-5; coach changed side in 7, served 1-6 (NO-NYG W6 the only win). Recorded weak-signals coach-fade-2026-live-fired-vs-expected (-17.4 acc pts [-32.2,-2.6], P+ .03, unresolved_below_power).
- Owner 2026-10-19: picks are worse, fix it. Unit 3 (agent): per-member coefficient challenger (coach split from shared flag_sum) refit LOSO 2020-25 plus 2026 W1-6 as training for W7; serve for Week 7 Tuesday card only if held-out log loss beats the shared fit. pick_probability.py is PINNED (independent_validation.py PINNED_NAMES): serving it means enrolling v4.
- Alerts: every Week 6 flip logged NOTIFY-OK (ntfy http 200, priority already urgent) but none reached the phone; EMAIL-SKIP every time. Fix needs owner: NFL_ATS_SMTP_USER + NFL_ATS_SMTP_PASSWORD (Gmail app password) in HKCU env.
- Unit 3 measured (tests/scratch/served_calib/perm.py, perm_out.txt, replay13.py): per-member challenger LOSO 2020-25 n=1503 LL shared .68277, coach+rest .68274 (gain +.00004 se .00042), all 7 members .68543 (-.00266 se .00112). Coach coef is LARGER than rest in every fold (.31-.38 vs .21-.27). Live W1-6 n=92 (W1-3 inputs replayed with earliest coef artifact, inputs only): LL shared .6776, coach+rest .6828, all .6802. W7 fit (2020-25+live): shared .245; coach .228 rest .251; sides differing 0. Recommendation: do not serve per-member (no held-out gain; live worse; changes no W1-6 side).
## Tried
Per-fold coefficients; LOSO with/without coach. W1-3 side counterfactual not rebuilt (no reconstruction).
## Next
Weeks 7+: record coach-fired games prospectively; weak-signals record if the 1-6 changed-side result is written up.
## Open
Flag-member selection used the same 2020-25 seasons (validation_limitations in metadata).

# MOD-49

## Goal
MOD-41 feature-construction sensitivity: window length, play definition and norm, all declared before signs (ROADMAP MOD-49).

## State
Unit 1 DONE 2026-11-03 22:43 UTC (results below). Mechanism: The residual-load term depends on construction choices made once (window 3, pass and run plays only, season-to-date norm, tempo parts removed by regression). A real mechanism should survive reasonable alternatives; one that vanishes outside window 3 is a fit to the construction. Fitted term: the same single coefficient on each declared variant, compared leave-one-season-out on the served base.

## Unit 1 declaration 2026-11-03 22:45 UTC (before any variant run)
Script scripts/mod49_construction.py, artifact artifacts/mod49/construction.json. Family mod49_mod41_construction, 8 cells declared: V0 baseline replicate (window 3, plays run+defended pass/run no kneel/spike, season-to-date norm), W2, W4, W5, OFF (offensive plays only, not offense+defense), PRIOR (norm = team's prior-season per-game mean), SEC (possession seconds: elapsed game clock between consecutive plays of one drive attributed to posteam and defteam, same window 3 and norm), PLAC (placebo: load feature of the opponent's previous opponent as of that game, home minus away). Each: residual vs own and opponent tempo regression (own = season-to-date offensive mean, opp = window mean of defended; same unit as the variant's load), served base logit(out_of_season_home_probability) artifacts/pick_probability/20261103T195411Z, LOSO 2020-25, week-block bootstrap, within-week permutation null 500. Usable = at least window+1 prior games (unit 1 rule scaled), non-NaN feature. Looks: 8 log-loss cells + 8 accuracy cells = 16 plus 8 null runs. Sign expectation: negative coefficient. Verdict rule fixed: robust if coefficient sign is negative in nearly all folds and ll gain positive in most non-placebo variants with sizes within a factor of about two of V0; placebo should be near zero.

## Unit 1 results (served base, LOSO 2020-25, week-block bootstrap, 500-draw within-week null)
Cell: usable; decisive term-base/n; ll gain [95%] P+; acc gain P+; coef; fold signs; null p.
- V0 baseline: 1142; 157-131/288; +.00404 [-.0012,.0094] .94; +1.73 .92; -.0439; 6/6 neg; .002 (replicates unit 6 exactly).
- W2: 1231; 101-104/205; +.00120 [-.0022,.0046] .74; -0.20 .40; -.0207; 6/6 neg; .032.
- W4: 1045; 111-103/214; +.00202 [-.0017,.0055] .88; +0.53 .67; -.0397; 6/6 neg; .006.
- W5: 944; 32-43/75; -.00029 [-.0019,.0012] .36; -0.73 .13; -.0173; 6/6 neg; .373.
- OFF only: 1142; 141-114/255; +.00176 [-.0022,.0052] .83; +1.80 .92; -.0352; 6/6 neg; .020.
- PRIOR norm: 1142; 133-121/254; +.00258 [-.0010,.0061] .92; +0.80 .73; -.0279; 6/6 neg; .002.
- SEC: 1142; 55-66/121; +.00043 [-.0016,.0024] .68; -0.73 .14; -.0013/sec; 6/6 neg; .082.
- PLACEBO opp-of-opp: 1142; 17-24/41; -.00078 [-.0017,.0002] .05; -0.47 .13; -.0010; folds mixed (+ in 3); .677.
Looks: 16 (8 ll, 8 acc), 8 nulls. Reading (inferred): coefficient sign negative in 7/7 real variants x 6 folds; ll gain positive in 5 of 6 non-baseline real variants but 1/2 to 1/10 of V0 and P+ .36-.92, so size is NOT robust: it fades with window 2 and 5, with seconds, and halves with prior-season norm; window 4 and prior-norm and offense-only keep the sign and about half the gain. Placebo near zero, mixed signs (clean). Not closed: nothing is wrong-sign-resolved.
8 cells recorded (log_loss_improvement, unresolved_below_power, artifacts/mod49/weak_signals_batch.json via scripts/mod49_record_batch.py; accuracy numbers inside descriptions).
Next unit: window 3 may be a peak chosen on this sample (units 1-2 declared it a priori from the roadmap, not tuned); test a joint window-weighted decay or seasons-forward (2026 weeks 1-8) holdout; SEC scale and W5 usable shrink (944) are confounders.

## Tried
Novelty check only. Nearest: MOD-41 units 1-3 (window 3 only), MOD-29, SIM-13. Differs: varies the construction, not the mechanism or the injury mediation already run.

## Next
First unit for one agent: Declare the family first (windows 2, 3, 4, 5; plays with and without kneels and spikes; norm as season-to-date versus prior-season mean; seconds of possession instead of plays; same-week future-game placebo), count the looks, and report the gain, sign and coefficient per variant on the served base with the held-out gap.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

# MOD-45

## Goal
Tempo pair as a spread input: coordinator-keyed pace and lead-protection traits set expected plays, which scale the line-conditional margin distribution (ROADMAP MOD-45).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-18 found coordinator lead-protection and seconds-per-play traits reliable (OC split-half .69) but with no side term. The same traits fix how many possessions a game has, which sets how wide the margin distribution is, not which side wins. Enters the one calibrated probability as a fitted scale term on the discrete margin distribution conditional on the line (key-number mass at 3, 7, 10 stays discrete), never a side rule.

## Tried
Novelty check only. Nearest: SIM-18 (traits, side term only), MOD-18 (lattice, done), mod25d-variance-source lane; grep 'plays per game' and 'tempo pair' found none.

## Next
First unit for one agent: Predict each game's plays from the two offensive coordinators' and head coaches' prior-season traits (leave-season-out), report out-of-season R2 against realised plays, then add-one LOSO on cover log loss through the discrete read against intercept-only. Data: Coordinator keys from SIM-18 and PER-07, nflverse play-by-play (free).

## Open
SIM-18 says no pick term helps; the scale term is untested. Pace is priced by the total line, so the add-one must use the total as a control. Zero crossing closes nothing; one fitted probability selects the side.

## Unit 1 (2026-11-03 ~21:55 UTC, in flight)
Script scripts/mod45_tempo_spread.py written, ruff-formatted (check was run; verify clean). Steps: LOSO-k shrunk OC/HC spell play-count traits (prior games only) -> game plays R2 vs total (close 2012-25, opener 2020-25) -> |ATS resid| and within-3 regressions with total and |spread| controls, bootstrap -> discrete-lattice scale table (atom reweight exp(-g|m-line|) matching mean-abs-dev ratio, then tilt to point) -> graded 2020-25 opener log loss vs intercept and discrete base. Run launched in background (task b0p2y7oub, bash+python child); outputs artifacts/mod45/unit1.json and run.log, err.log ends with "finished" when done. Nothing read yet. Next: read unit1.json, record cells with weak-signals record (units: log_loss_improvement, slope_difference, logit_coefficient; reliability .69 from SIM-18), fill results here.

## Unit 1 RESULT (run finished; artifacts/mod45/unit1.json; looks: 8 plays, 12 dispersion, 54 lattice cells)
Plays (LOSO, k=16 every fold): total alone R2 .020 close / .001 opener; + OC expected plays .050 / .017 (SSE gain 2.5 [1.6,3.5] close, 1.3 [.4,2.1] opener, P+ 1.0). Beyond total: real but small.
Dispersion per SD OC expected plays, controls total and |spread|: |resid| +.317 pts [-.10,.78] P+ .915 opener (n 1526); +.313 [.03,.58] P+ .984 close 2012-25 (n 3403). HC arm opener +.487 [.08,.91] P+ .986. Within-3 logit opener -.03 [-.15,.09] P(neg) .71. 5/6 and 9/14 seasons positive.
Lattice (atoms reweighted, scale 0.9/1.1): at edge 3 cover prob changes about -0.5 to -1.0 pp per +10% width at lines 3 to 10; shifts under 1 pp.
Graded 2020-25 opener (n 1492): decisive flips 6, W-L 2-4. Log loss intercept .69329, discrete base .69605, scaled .69629; scaled minus base -.00024 [-.00044,-.00004] P+ .009 (worse); vs intercept -.0030 P+ .17. Base itself loses to intercept.
Recorded 4 unresolved cells, family mod45_tempo_pair_unit1 (weak-signals).
Next: unit 2 = only if scale beta is stabilised on 2012-19 (closing-line, labeled non-opener); dispersion signal is real but too small to move cover log loss.

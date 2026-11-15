# SIM-30

## Goal
Simulated conditional margin distribution as a challenger to the discrete empirical read, graded at the opener on the played card (ROADMAP SIM-30).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: The served cover and push probability is the discrete margin distribution conditional on the line; a simulation can supply the distribution with its own key-number mass. Measured mechanism: the simulator's margin mass at 3, 7, 10, 14 and 17 compared with the empirical lattice by spread band, now that the truncation double count is fixed behind a flag. Fitted term: Challenger only: simulated cover probability against the discrete read on the same games; it can be served only if it beats the discrete read on the opener grade through the played card.

## Tried
Novelty check only. Nearest: SIM-11, SIM-21, MOD-17 (lattice ledger), MOD-32. Differs: First comparison of simulator margin mass with the served discrete read; MOD-17 and MOD-32 compare lattices without the simulator.

## Next
First unit for one agent: Compare simulated and empirical margin mass at the five key numbers by spread band with game-cluster intervals, then grade the simulated cover probability against the discrete read by log loss on games where simulation exists.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: Simulator margin output after SIM-21 (flag on), served discrete read (local).

## Declaration (2026-11-11 10:37, written before any outcome was read)
Data: sim08_market_anchored/main primary_2020..2025.npz (2026-11-03 market-anchored engine, NOT a final engine; 372 draws), 1,537 games; served read = artifacts/pick_probability/20261111T011859Z/per_game.parquet (cover_excluding_push, push, loss at open); grading uses games present in both (count stated in result). Lattice = prior-season real margins (2009 to s-1) within the served band half-width (min 200 games, widened as band_read does), tilted to mean = opener line via nfl_ats.mass_preserving_lattice.tilted_atoms (imported, not edited).
Part 1 shape: bands |opener| 0-2.5, 3-3.5, 4-6.5, 7-9.5, 10+; mass at favourite-signed margin 3,7,10,14,17: sim, real, lattice; game-bootstrap CI for sim-real and lattice-real; push mass on whole-number lines sim vs real vs lattice.
Part 2 grade: 3-way and decisive 2-way log loss for sim (SIM-08 form, (c+.5)/(n+1.5)), served, market hist (sim08_evaluate.market_hist_three_way); paired d = other minus sim, week-block bootstrap (sim08_evaluate.boot_mean), probability_positive; push-line pricing on whole-number lines; decisive record where sim side != served side. Reproduce SIM-08 -0.0136 with its own served (sim04_loso.baseline_three_way).
Looks: shape 25 (descriptive, one family) + grade 3 comparisons x 2 metrics = 6; family sim30_margin_distribution_challenger. No tuning, no re-centring.

## Result (2026-11-11 10:41; measured; scripts/sim30_margin_mass.py -> artifacts/sim30/sim30_margin_mass.json; 6 cells recorded in registry/weak_signals/sim30_margin_distribution_challenger.json, unresolved_below_power)
Engine = 2026-11-03 market-anchored, not final. Served per_game has 1,503 of 1,537 games (the 34 push games absent), so grading is on 1,503; push pricing uses SIM-08's older served read on 1,537.
Shape (game bootstrap, sim minus real, favourite-signed): 3_3.5 band (n 366) mass_3 sim .077 v real .112 v lattice .085, diff -.035 [-.068,-.003]; 10plus (157) mass_3 .052 v .076 (-.025 [-.069,.014]), mass_14 .046 v .032 (+.014); 7_9.5 (246) mass_10 .040 v .020 (+.020 [-.001,.036]); 4_6.5 (388) mass_10 .039 v .052, mass_14 .039 v .028. Other cells within +-.01. Push mass whole lines (661): sim .046, real .051, lattice .049.
Grade (d = other minus sim, positive = sim better): served-sim 3way -.0108 [-.0202,-.0002] P+ .022; 2way -.0110 [-.0205,-.0014] P+ .014; market-sim 3way -.0170 P+ 0; 2way -.0139 P+ 0; served-market +.0062 against served 3way P(served better) .057. SIM-08 -.0136 reproduced exactly on 1,537 (3way). Decisive where sides differ: 729 games, sim 316 right, served 413 (binomial p .0004).
## Final run (2026-11-14, measured; artifacts/sim30/amend6/sim30_margin_mass.json; 6 cells sim30_a6run_*, unresolved_below_power; done)
Decisive where sides differ: 720 games, simulated 330 right, served 390 (binomial p .028). Served minus simulated, positive = simulated better: 3way -.0030 [-.0124, +.0071] P+ .29; 2way -.0037 [-.0129, +.0055] P+ .23; market minus simulated 3way -.0093 [-.0141, -.0046]. Mass at 3 on 3_3.5 unchanged: .077 v real .112 (-.035 [-.068, -.004]). Push mass whole lines .046 v .051. Verdict: does not beat the served discrete read; serves nothing. The run used terminal transitions, which fail the gates (10/11/11 against 13/14/13).
## Next
None here. The push and half-point mass challenger (A6_H5 +.0007 [+.0001, +.0014]) is forward-logged under the SIM-29 lane.

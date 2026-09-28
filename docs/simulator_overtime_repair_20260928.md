# Overtime terminal-rule repair

## Declaration before replay

Repair the historical 2015–2017 modified-sudden-death rules in the simulator.
The second possession ends the game if it fails to answer an opening field goal.
A winning overtime touchdown ends play without an extra-point attempt. A change
of possession ending the trailing team's second chance cannot create a later
defensive return touchdown after the game has ended.

**Measured before repair:** deterministic calls continue two nonscoring second
possessions until the play cap; opening and answering touchdowns produce margins
7 and 4 instead of 6 and 3. In the pinned 2015–2017 data, all 48 actual overtime
final margins are 0 (2 games), 3 (29), or 6 (17), with no conversion plays.
The existing fixed-seed 485 overtime simulations include 72 margins of 7,
13 of 4, eight of 8, one of 9, and four of 10.

Use the preceding frozen feature artifact and PBP snapshot; train 2009–2014,
validate 2015–2017, seed 20260925, 3,334 simulations per validation season.
Compare the existing post-terminal-repair baseline at
`artifacts/sim08_terminal_repair/20260928T161258Z/` with one repaired arm.
Preserve source/input hashes, every simulation row and per-game discrete loss.
Report the same complete diagnostic family: five key-number masses; three
log-loss values; margin mean, SD and scale ratio; total points; six gameplay
rates; ten terminal comparisons; two derived engine gates; three season effects
per arm (66 cells total), plus the full overtime absolute-margin histogram for
both arms. No cell selects weights or a served pick.

Use exact resampling of the three held-out seasons for conditional log-loss
uncertainty and probability_positive. This omits simulator Monte Carlo and
earlier model-selection uncertainty. Record effects as unresolved unless an
admissible closing ground is separately demonstrated. Zero crossing closes nothing.
The historical overtime rule repair is not validation of the different 2025+
both-possession rule and does not authorize serving this simulator.

## Results

**Measured:** the final replay is
`artifacts/sim08_overtime_repair/20260928T171712Z/`. All input hashes match;
10,002 simulations per arm have zero play caps. The initial repair replay at
`20260928T170757Z` is retained as a superseded arm: independent review found
that defensive touchdowns often lacked a possession-change flag (589/615 in
training), so suppressing their points also needed an explicit terminal flag.
Production-call probes now end those plays correctly with either flag value.
The superseded and final replay happen to have identical saved predictions;
the deterministic failure probe, rather than held-out performance, selected the fix.

| Arm | Log-loss improvement vs training histogram | Conditional 95% interval | probability_positive | Key-number bands |
| --- | ---: | --- | ---: | ---: |
| Before | +0.000939 | [-0.011431, +0.015928] | 0.592593 | 3/5 |
| Superseded | -0.000656 | [-0.013844, +0.014832] | 0.370370 | 2/5 |
| Final | -0.000656 | [-0.013844, +0.014832] | 0.370370 | 2/5 |

**Measured:** final-minus-before improvement is -0.001595
[-0.002528, -0.000795], probability_positive 0 in the 27 conditional season
resamples. The three season changes are -0.002893, -0.001211 and -0.000680.
Correcting the rules did not improve this fixed replay's predictive loss.
Monte Carlo and prior selection uncertainty are absent from these intervals;
this diagnostic does not establish a general effect or justify undoing correct rules.

**Measured:** overtime outputs no longer contain illegal margins 7–10 or
conversion-driven margin 4. Final overtime margins are 0:48, 2:4, 3:280,
5:1, 6:114 (447 games); the 48 actual overtime games are 0:2, 3:29, 6:17.
The full zero-filled histograms 0–10 for actual/before/superseded/final are saved.
The original 66-cell family expands to 99 by retaining the superseded arm;
44 histogram cells and four transparent paired-loss cells bring this repair's
reported diagnostic count to 147. All are retained, including negative results.

**Measured:** four effects were accepted by `nfl-ats weak-signals record --batch`
under `sim08-historical-overtime-rule-repair` as `unresolved_below_power`.
No signal was closed, weight fitted, or served probability changed. The
`AGENTS.md` discrete-margin and promotion rules still block simulator serving:
only 2/5 key-number bands pass, and 2025+ overtime rules are unvalidated.

The bundle preserves all 30,006 simulation rows, 768 actual-game loss rows,
source engines, hashes, reports, full histograms, registry batch and verifier.

# The simulator: what the engine is and what each mechanism did (2026-11-11; forecaster result added 2026-11-14)

All numbers below are read from the lanes named in each line, not re-run for this page. Gate counts are of 14 realism gates, shown as three single seeds. The default is 13/14/13 (read, sim-finish lane, S2); on the six-seed mean it passes 14 of 14 (read, sim11 lane).

## 1. What the engine is

A simulated game is built drive by drive. Each drive is a real possession (first snap to the next first snap: the drive, any score, the extra point, the kick and return) drawn from the 2009-17 library. It is drawn by game state: score, clock, field position, timeouts, season and week, with the clock kept feasible (read, sim10 lane). The draw is then tilted so the drawn drives hit two targets: the team's strength, and fitted odds of a touchdown, a field goal, a defensive score and elapsed time at that exact state. Extra points and two-point tries are decided at the exact score by a fitted two-point model. The last two minutes of each half tilt outcomes only. Overtime is state-only (read, sim10 lane).

Each team gets a strength latent: an offence and a defence, with a persistent part, fitted from real play on a 2009-25 era so it matches the gate era. Late-game leader clock-kill drives are kept, and fourth-quarter field-goal choices shift with the score (read, sim11 lane). Everything is fitted from real plays, with each season held out when fitting the draw widths. No constant is tuned to a gate.

Default configuration: drive resampler, joint strength transmission, real schedules for comparisons. A gate is one real-world number (for example the share of margins landing on 3, or points a game) with a 90% interval from real games. The simulator passes it when its seed-averaged value falls inside that interval. Six-seed mean 14/14; single seeds 13 to 14 (read, sim11 lane).

## 2. What each mechanism did

Rule for adoption (Tier B): gate count not lower on any seed, no passing gate lost, and a named gate moves toward real by more than 2x the default's seed spread. Read from the lanes named.

| Row | Mechanism | Real-play measurement | Engine result (gates v 13/14/13) | Verdict |
|---|---|---|---|---|
| SIM-06 | Learned drive head | Beats the plain draw held-out by +0.0577 per drive [0.0456, 0.0701] | 12 of 14 on the 3-seed mean v 13; lost 7-point mass and late-season R2 (engine-patch-queue) | Not adopted |
| SIM-07 | Personnel matchups | Team-game shift SD 0.107 points; no reliable personnel cell | Score moves with the table shift 0.764 [0.214, 1.295] hook on, -0.122 [-0.683, 0.402] off | Hook built and delivers its table; not made a default (the table is too small for any gate or forecaster cell and covers 2022-25 only) |
| SIM-08 | Forecaster | See section 4 | See section 4 | Declared bar not met; serves nothing |
| SIM-09 | No-huddle, urgency, fatigue, stakes | Real-play measure: not in the sources read | No-huddle 6/6/7 (+3.8 points a game); urgency 10/12/12; fatigue 9/12/12; stakes 10/14/11 | Not adopted (all four) |
| SIM-10 | Drive resampler | n/a | 12 of 14 at lock on the 3-seed mean | Adopted |
| SIM-11 | Strength transmission | n/a | Six-seed mean 14 of 14 v 13 | Adopted |
| SIM-12 | In-game exits | Line exits: 1,251 in 25,220 starter-games (4.96%), -0.038 points per play [-0.057, -0.017]. Quarterback exits: 958 in 8,007 team-games, +0.070 [0.0005, 0.143] v placebo +0.048 | Line exit 10/11/12; redesign 13/13/13 | Not adopted; quarterback part measured, no hook warranted |
| SIM-13 | Defensive snap load | +0.0033 per play [-0.0040, +0.0101], P+ .81; 17 of 17 seasons positive, each under 0.008 | Not run | Measured, no hook warranted |
| SIM-14 | Coach aggression on fourth down | Trait SD 0.322 logit [0.248, 0.382]; next-season slope 0.372 | 10/11/12; lost mass at 3 on all seeds; margin SD +0.087 away | Not adopted |
| SIM-15 | Wind on field goals | Games at 15 mph or more: total points -2.39 [-4.40, -0.39] | 14/12/12; recovers about a tenth of the gap (-0.14 [-0.61, +0.30] v default +0.10) | Not adopted |
| SIM-16 | Pregame attrition | Opener-graded term -0.041 (SE .054), expected sign, unresolved (P+ .074); 52.6% v 52.8% control | Not run | Measured, unresolved below power; no hook built |
| SIM-17 | Halftime adjustment, measured | Coordinator trait: offence next-season r .209, held-out slope 0.40 [0.18, 0.61] | Run as SIM-20 | See SIM-20 |
| SIM-18 | Lead-protection clock | Lead 1-8 trait SD 0.053 log [0, 0.099], P(variance above 0) .80 | 12/12/11; fourth-quarter slope toward real by .0031 against a .004 bar | Not adopted |
| SIM-19 | Trailing risk shape | Final 9 min trailing by 5+: touchdown real minus drawn +.056 [.041, .070] | Not run | Measured, no hook warranted |
| SIM-20 | Halftime hook, in the engine | As SIM-17 | 13/12/13 (nearest miss); strength variance +1.17 toward real; tie share lost on two seeds | Not adopted |
| SIM-21 | Drive length and scoring deficit | Truncation share matched; scoring inside truncated drives 4% v real 19% (trailing) | Truncation fix alone: 45.72 points a game v real 44.83; joint 13/13/13 | Not adopted |
| SIM-22 | Plays per drive logging | Trailing 26.99 v simulated 27.65 seconds a play; leading 30.60 v 29.14 | Logging only | Measured; no hook built (open finding, section 3) |
| SIM-23 | Clock-truncated drives | As SIM-21 | Judged in the SIM-21 joint arm, 13/13/13 | Not adopted |
| SIM-24 | Next-drive start field | Start field within about 0.5 yards; explains 13% of the leading gap | Not run | Measured, no hook warranted |
| SIM-25 | Pool restricted by score and clock | Late points gap .223 [.177, .267] full pool, .059 [.037, .110] restricted | 9/12/13; mass at 14 past real | Not adopted |
| SIM-26 | Leading-defence shape | Real minus drawn +.19 yards per play late (opposite to a prevent shape) | Not run | Measured, no hook warranted |
| SIM-27 | Adaptive neighbour counts | n/a | Alone 13/12/12 (points 44.47); joint 13/13/13 (points 45.44) | Not adopted |
| SIM-28 | Timeouts at end of half | Held-out log likelihood +.0128 per drive [.0102, .0153] | Timeouts 10/10/12; with score bandwidths 11/14/11 | Not adopted |
| SIM-29 | Lineup from injury wording | Missing-starter error .362 v .400 ladder, gain .038 [.012, .067], 151 team-weeks | Not built | Waiting (needs forward weeks, about Week 12) |
| SIM-30 | Key-number mass and grade | See section 4 | See section 4 | Measured; does not beat the served read; serves nothing |

## 3. What is still wrong with it (all read from the sim-finish lane)

- Late-game tempo by score is under-responsive. Leaders run 29.14 v real 30.60 seconds a play (-1.46 [-1.60, -1.32]); leaders with 9 or more in the last five minutes run 19.80 v 25.45. The two mechanisms acting there (SIM-25, SIM-18) lost gates, so no repair was adopted.
- The truncation fix raises scoring by 0.6 points a game, so the default's scoring match rests partly on truncation; the offsetting excess is not located.
- Mass at 3 is short in the forecaster margins: .0715 v real .0859 on the final run (3.5 percentage points short on lines of 3 to 3.5, [-6.9, -0.5]); terminal transitions did not move it (measured 2026-11-14, artifacts\sim08_amendment6\amend6\grade.json).
- On single seeds the default misses one gate each on two of three: overtime share on seed 11 and late-season fit on seed 13 (read, patch-queue lane). Strength variance sits low but inside its interval, 53.5 v real 58.0 [51.4, 64.9].
- Trailing drives hold too few plays: 5.52 v real 5.78 (-0.26 [-0.31, -0.21]).
- Terminal transitions (the end-of-game repair used in the final forecaster run) cut strength variance by 5.4, three times the default's seed spread, and lose two to four gates a seed; why is not identified (measured 2026-11-14, artifacts\terminal_cmp.log).

## 4. The forecaster test

Learned so far (read, sim-finish lane and ROADMAP SIM-08):
- Raw simulator grades worse than the served read: log loss -0.0136 (1,537 games). On the older margins, a two-way gap of -.0110 [-.0205, -.0014] (P+ .014); where sides differ the simulator was right 316 times and the served read 413.
- The inversion: the cover logit came out negative in all six folds (-0.452). It was not a sign bug. The first anchor left the simulated mean 1.90 points off the line, and that leftover correlated .50 with the line. Why it is anti-informative is not identified (inferred: simulated favourites about a point too strong).
- Forcing the mean onto the line (exact tilt) did not repair it: -0.498 [-1.33, +0.34], negative in all six folds, -0.99 on half-point lines.
- Why balance at the line, not the mean: on real games the favourite's mean margin sits above the line on 3.5 (+0.99) and 7 (+1.86) while the cover share stays near half. A spread is a balance point; with mass at 3 and 7, mean and balance differ most on half-point lines (inferred).
- What remains as product: push and half-point mass. Stage-1 plus simulated push mass beat stage-1 plus the empirical table by +.0005 [-.0002, +.0012] (P+ .92); three-way log loss on whole lines, served minus simulated, +.0014 [-.0086, +.0119] (P+ .61).

Final run finished 2026-11-14 19:44 ET (1,537 games, 372 draws; balance conditioning, terminal transitions, total target fitted on the other five seasons). All numbers in this paragraph are measured from artifacts\sim08_market_anchored\amend6, artifacts\sim08_amendment6\amend6 and artifacts\sim30\amend6. Where the blend and the served-only recalibration pick different sides, 124 games: blend right 66, recalibration right 58. Blend against served-only recalibration -0.0012 log loss [-0.0021, -0.0004], chance it helps 0.0015 (fitted on the same seasons it gains 0.0001; on held-out seasons it loses 0.0012; the simulator's weight runs from -0.35 to +0.02 by season); simulator alone against the served read -0.0047 [-0.0140, +0.0045], chance it helps .15. Where the simulated side differs from the served side, 720 games: simulated right 330, served right 390. Mass at 3 simulated .136 v real .151; on lines of 3 to 3.5 it is .077 v .112, the same as before terminal transitions. Total slope simulated minus real +0.080 [-0.062, +0.223] (was +0.223). With the simulated margins set to cover half the time at the opener, push pricing on whole-number lines, served minus simulated +0.0057 [-0.0036, +0.0158], chance simulated is better .89; half-point rows, stage one plus simulated push mass against the empirical table +0.0007 [+0.0001, +0.0014], chance .98, one of 25 cells graded together. The simulator serves nothing; its cover lean is closed as a term (wrong sign with the whole interval on that side), and its push and half-point mass stays a challenger to the empirical table that needs a forward log on played cards before it could replace the table.

Declared bar (docs\sim08_market_anchored_declaration.md, Amendment 4 item 5): not met. The blend fails, the total slope is inside its interval but over the size limit, and mass at 3 is closer but still short; the total anchor (+0.012 points against its target) and the simulator-alone cell pass.

Two checks the bar also names (measured 2026-11-14):
- Realism gates with terminal transitions on: 10/11/11 against the default's 13/14/13 (artifacts\terminal_cmp.log). Strength variance falls from 53.5 to 48.1 against real 58.0 on all three seeds; late-season fit and fourth-quarter slope are lost on two seeds each. Margin spread, mass at 14 and overtime share move toward real. Terminal transitions are not adopted. So the final run graded a variant that fails the gates, not the 14-of-14 default; the gate check should have come before the long run. No second long run was made: the blend already fails, nothing in either run would serve, and the push-mass reading exists on both variants (default engine +0.0005 [-0.0002, +0.0012]; terminal variant +0.0007 [+0.0001, +0.0014]).
- Planted control: the evaluator detects a planted gain of 0.008 log loss in 84% of replicates and a gain of 0.001 in 32% (artifacts\sim08_market_anchored\amend6\control.json). It can see 0.008 and cannot bound 0.001.

## 5. What it is good for now

It is a realistic generator of whole games and margin shapes, passing 14 of 14 gates on the six-seed mean, with every mechanism tried recorded and none changing the default. It is not a better cover forecaster than the served read and adds nothing to the served probability as a blended term; its one candidate use is push and half-point mass inside that read, which the final run supports by a small amount and which is not served.

Where the detail lives: docs\lanes\done\sim-finish-2026-11-11.md, docs\sim_finish_inventory.md, docs\sim08_market_anchored_declaration.md, ROADMAP.md rows SIM-06 to SIM-30.

# SIM-09 mechanism catalogue: situation x personnel

Owner idea 2026-10-02: how well a strategy change works depends on who runs
it. Examples given: no-huddle with a young QB; deep throws and the WR-CB
matchup. This catalogue expands on that idea. Each row is a mechanism to
measure on real plays, then add to the generator as a fitted policy. No
constants. Shape, margin SD and R-squared stay held-out checks.

Data: pbp 2009-25; participation 2016-25 (players on the field, routes,
man/zone, coverage type, time to throw, pressure, box count, personnel); NGS
weekly 2016-26 (separation, cushion, CPOE, time to throw); schedules
(coaches, QBs); injuries, inactives, rosters. Every rating is built pregame
from prior games only.

## Foundation: teams are a set of units, not one number
F1 Unit latent ratings per team-week: QB (depth, accuracy, decisions,
scramble), pass protection, pass rush, coverage (man and zone separately),
receivers (separation, deep, YAC), run blocking, run defence (by box count),
kicker range, and coach tendencies. Measure reliability, week-to-week drift,
injury shocks and the covariance between units. The generator then carries
unit latents instead of one offence and defence strength, and the neighbour
pool matches on matchup-relevant units.

## A. Who carries out the change (how well it works, given personnel)
A1 Tempo and no-huddle by QB experience (career dropbacks, years with the
   coordinator, rookie flag); pre-snap penalty and sack cost.
A2 Two-minute drill by QB, OL and receivers: clock management, spikes,
   sideline routes, timeouts left.
A3 Obvious passing downs (trailing late, 3rd and long): OL vs pass rush.
   Pressure rate, time to throw, sacks and INTs rise more for weak lines.
A4 Deep shots: WR deep separation and route (go/post) vs coverage shell
   (single-high vs two-high, man vs zone) and the DBs on the field; QB deep
   accuracy and arm.
A5 Clock-killing runs: run blocking vs box count; a stacked box against a
   leader; whether runs gain enough to keep the clock moving (three-and-out
   gives the ball back).
A6 Ball security when leading: RB and QB fumble rates, kneel and spike
   timing.
A7 Prevent defence when ahead: dime and nickel share, DB depth; yards given
   vs TDs given; failure when DBs are weak.
A8 Kicker range shapes 4th-down calls, end-of-half and trailing targets
   (FG range moves the go/kick line).
A9 Coach aggression latent (4th-down and 2-point go rate over expected) and
   whether aggressive coaches convert more.
A10 Mobile QB vs man coverage and the pass rush: scrambles under urgency.

## B. Fatigue, depth, attrition
B1 Defensive snap load: plays defended so far, no-huddle pace against,
   long drives in a row; yards per play allowed as fatigue builds.
B2 In-game injuries: starters leaving (participation shows who is on the
   field); backup quality drop for the rest of the game.
B3 Depth of rotation: pass rushers and DL rotation; late-game rush
   effectiveness by depth.
B4 Short week, travel and time zones with tempo and a young QB.
B5 Weather with strategy: deep passing and comebacks in wind, rain, cold.

## C. Information and adjustment
C1 Halftime adjustments: second-half vs first-half efficiency change by
   coaching staff (offence and defence), reliability across seasons.
C2 Scripted openings: opening-drive efficiency by coordinator.
C3 Familiarity: division rematch, the same coordinator facing a former
   team, second meeting in a season.
C4 In-game learning: a defence's success against a play type earlier in the
   game changes later calls and outcomes.

## D. Game state and psychology
D1 Urgency and conservatism (U1, measured).
D2 Halftime deficit carry-over (U1, measured).
D3 Momentum after turnovers, big plays and missed kicks: does the next
   drive's efficiency change in real games? The sim currently over-carries
   leads, so measure, don't assume.
D4 Crowd noise: road false starts, no-huddle and silent counts on the road
   in high leverage.
D5 Playoff stakes (U2), resting starters, eliminated teams, and stakes x
   roster age (veterans in a race vs young teams).
D6 Prime time and national TV with tempo and turnovers.

## E. Unit-vs-unit matchups below the team level
E1 Offence personnel vs defence personnel (11 vs nickel, 12 vs base, heavy
   vs light box) and efficiency per grouping.
E2 Slot WR vs nickel CB, TE vs LB and safety coverage.
E3 Edge rusher vs tackle (side-specific pressure).
E4 Play-action vs aggressive run defences (box count, linebacker depth).
E5 Screens and quick game against the blitz (number of rushers).

## Order
F1 first (everything else conditions on unit latents). Then A3, A4, A5
(largest drive-outcome levers for the E3 gap), B1, C1, D3, then the rest.
Every family is declared before signs are seen, uses partial pooling instead
of many cells, and states its look count.

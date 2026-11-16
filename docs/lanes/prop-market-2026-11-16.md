# prop-market-2026-11-16

## Goal
Test whether player props tell us something the frozen Tuesday pool line can't. Props come from a different desk and react faster to player news, such as a doubtful QB or a backup RB starting. The markets are QB passing yards and attempts, rusher yards and attempts, and anytime TD. Keep a forward log for Weeks 11-18 using a diagnostic with no fitted parameters. Declared 2026-11-16 17:52 EST: no coefficient is fitted on outcomes until at least 4 forward weeks are graded.

## State (2026-11-16 17:52 EST)
- Source (measured): the Bovada v2 single-event endpoint carries the Passing Yards, Passing Props, Rushing Yards, Rushing Props and TD Scorer Props groups. Team totals come from "Total Points - <Team>" in Alternate Lines. New policy entry `bovada_player_props` in config/source_policies.json (added; nothing else changed).
- Coverage (measured, capture 20261116T225157Z, Monday): 12 upcoming games. Team totals were available for both teams in 12 of 12. Player props were posted for only 2 of 12: Week 10 MNF LAC@BAL and Week 11 TNF IND@HOU. Sunday props were not up yet. LAC@BAL had no passing-yards line for Brissett, and IND@HOU had no rushing line for a HOU RB. The diagnostic was computed for 0 games.
- Mapping (measured, `prop_market.py mapping`, data/external/props/points_per_yard.json): team points = -4.93 + 0.0782 x yards (SE 0.0017, R2 0.40, n 3,230 regular-season team games 2020-25). Yards = completed non-sack pass yards + rush yards on scrimmage downs, from pbp run 20261111T010519Z; points from schedule run 20261116T204645Z. Per-season slopes range 0.074-0.083. As a check, game margin regressed on yard difference gives slope 0.069 (SE 0.0026).
- Declared diagnostic (frozen at the stamp above):
  - Y per team = median yards from the team's highest QB passing line + the sum of median rushing yards over every listed rusher. The median is the handicap where the no-vig over probability crosses 0.5, interpolated linearly between ladder rungs, otherwise the nearest rung.
  - D = 0.0782 x (Y_home - Y_away) - (T_home - T_away), where T is the median team total. Lean = sign(D).
  - D is computed only when each team has a QB passing line, at least one non-QB rusher line and a team total. Otherwise the `missing` column names what is absent.
  - The graded row is the last capture before the game's pick deadline.
- Declared score output: graded at the pool line (settle = result - home_spread from data/splash, pushes dropped).
  - D record.
  - Team totals against the pool-line record, plus the D record on games where the two disagree.
  - Scale control: the team-totals underdog record, and the D record when D leans to the team-totals favourite.
  - A by-week table and the correlation of D with the totals margin.
- Files: scripts/prop_market.py (mapping, capture, pregame, score, slots). data/external/props/ (.gitignore `*`) holds live/<stamp>/bovada_props.json and props.parquet, forward_log.parquet and points_per_yard.json.

## Tried
- ruff check passes and ruff format has been applied. `pregame` ran twice: 13 requests each at a 1 s delay. `score` on the real log printed "nothing graded" and exited 0. `score` was also run on a synthetic scratch log to exercise the grading path; those numbers mean nothing.
- Inferred: b x dY should come out smaller than dT, because a regression slope is attenuated and medians are not means. If so, D leans toward the team-totals underdog by construction, which is why the scale control above was declared. The log keeps every per-team median, so a scale-free variant can be computed later without recapturing.

## Next
- Root adds scheduler jobs using argv from `prop_market.py slots`: `pregame` Thu 19:30, Sat 12:00, Sun 09:15, Sun 12:50 and Sun 15:45; `score` Tue 09:10.
- After the Thursday 11-19 slot, check Sunday prop coverage in the Saturday and Sunday captures.
- After 4 graded weeks, run `weak-signals record`. The verdict is unresolved_below_power unless an admissible closing ground exists. Only then consider one fitted coefficient.

## Open
- The completeness rule drops games where the QB is out and has no passing line. That is the very case the mechanism targets. A declared variant using the backup's line or team pass attempts is not built.
- Rushing medians are summed, but medians do not add up to a median. Rusher counts are logged per side.

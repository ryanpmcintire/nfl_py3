# Late-season motivation-state ladder screen

Family: late-season motivation states, weeks 13-18, built from standings math
(elimination / clinch / cut-line / draft-order primitives), not raw win-rate
proxies. Status: **predeclaration frozen before any cell was scored** — this
document was written before `scripts/motivation_ladder_screen.py` ran against
any cover outcome; measured results are appended at the bottom, tagged per the
AGENTS.md label-how-you-know-it rule.

## Prior-work overlap check (read this session)

- `scripts/nfl_bias_battery_screen.py:269-293` (read): `bad_team_late` is a
  raw proxy — weeks 11-18 AND prior_games >= 9 AND prior_win_pct <= 0.300;
  `great_team_late` is weeks 15-18 AND prior_games >= 13 AND prior_win_pct >=
  0.800 ("proxy for a locked seed", its own words); `motivation_mismatch` is a
  competitive team (prior win pct >= 0.400) facing a `bad_team_late` opponent.
  None of them consults the schedule (games remaining), division/conference
  structure, the playoff cut line, or league-wide draft order.
- `registry/weak_signals.json` lines 609/759/884 (read): the recorded versions
  of those three cells are the same win-rate proxies.
- No other doc or registry entry builds elimination, clinching, seed-locking,
  or tank-zone states from reconstructed standings (grep for
  `eliminat|clinch|tank|playoff cut|draft order` across `docs/` and
  `registry/`: no competing family).

Conclusion: distinct family. The new cells are strictly sharper cells of the
same mechanism class (`incentive_misalignment`) because they replace the
win-rate proxies with event-defined standings states: a team at 0.250 win pct
in week 14 that still controls its division is NOT eliminated, and the old
flags cannot see that; these can.

## Mechanism

Late-season incentive misalignment: once a team's playoff position is
mathematically settled (eliminated, or clinched with nothing to play for), the
competitive edge that normally supports covering a priced line weakens, while
opponents still fighting for something keep full effort. The market may under-
adjust for event-defined state changes it prices only loosely. Predicted
NEGATIVE for eliminated visitors, locked-seed teams, and tank-zone teams;
predicted POSITIVE for a team fighting for the last playoff spot against an
opponent with nothing to play for.

## Constructions (built from local schedules, point-in-time safe)

Source: latest local `data/raw/*/schedules.parquet` snapshot, REG games,
seasons 2009-2025, franchise aliases via `TEAM_ABBREVIATION_ALIASES`, ATS
outcomes via `nfl_ats.features.add_ats_outcomes`.

**Disclosure (binding on interpretation):** elimination, clinching, seed
locking, the cut line, and tank rank are APPROXIMATIONS computed from final
standings reconstructed week-by-week from completed games only. Every state
attached to a game uses strictly earlier `gameday` values (same-day other
games are excluded), so the construction is point-in-time safe — but the
decision rules below ignore tiebreakers and inter-team future scheduling
interactions, so they fire conservatively (elimination fires late; clinching
uses a ties-included worst-case rule). Known historical examples are used as
the validation step instead of split-half reliability, because these states
are event-defined conditions, not persistent traits — there is nothing to
split-half (same reasoning recorded for `bias_battery_motivation_mismatch`,
read, `registry/weak_signals.json` line 911).

State definitions at a game's date D (conference ordering by wins desc,
losses asc, team asc; K = 6 playoff teams for 2009-2019, 7 for 2020+;
division leader = best record within division):

- `gr(team, D)` = count of the team's REG games with gameday > D.
- `eliminated`: wins + gr < wins of the conference's (K+1)-th team (first team
  out). Even running the table cannot reach the current first-out total.
- `clinched`: wins - gr >= max over conference others (wins_o + gr_o),
  excluding self and excluding current division leaders ranked above self
  (they finish above regardless and do not push self out of the top K).
  Losing every remaining game still leaves self at or above every challenger's
  ceiling. Ties-included, so it fires late relative to real-world clinching
  (real clinches often rest on tiebreakers).
- `locked_seed`: clinched AND (ranked #1 in conference OR wins + gr < wins of
  the team immediately above in the conference ordering). Cannot improve seed
  even winning out while the team above loses out.
- `tank_zone`: bottom two records league-wide (wins asc, losses desc, team
  asc for determinism) — the #1-overall-pick zone.
- `fighter`: NOT eliminated AND outside the current top K AND within one win
  of the cut line (cut_line - wins <= 1), where cut_line = wins of the
  conference's K-th team. Fighting for the last playoff spot(s).
- `nothing_to_play_for`: eliminated AND NOT tank_zone (a mid-table
  eliminated team; tank-zone teams are excluded so cell C stays distinct from
  cell D).

## Predeclared cells (4, frozen before scoring)

Population: NFL REG close-grade slate 2009-2025 restricted to weeks 13-18,
team-perspective long table (one row per team-game, value `team_covered`;
pushes dropped). Flags reference each side's state at that game's date.

| # | name | flag | sign |
|---|------|------|------|
| M1 | `elim_visitor_alive_host` | side is AWAY, self eliminated, host NOT eliminated | -1 |
| M2 | `locked_seed_wk16_18` | self locked_seed AND week 16-18 | -1 |
| M3 | `fighter_vs_nothing` | self fighter AND opponent nothing_to_play_for | +1 |
| M4 | `tank_zone_wk14_18` | self tank_zone AND week 14-18 | -1 |

Sign convention: `sign` is the PREDICTED direction; `probability_positive`
(P+) is the bootstrap probability the prediction holds. Positive
full-slate effect = prediction confirmed, in accuracy points scaled to the
full slate (`nfl_ats.experiment_runner.scale_subset_effect`).

## Validation step (replaces reliability; run BEFORE scoring)

Three historical spot-checks asserted before any cell is scored:

1. 2020 JAX weeks 14-17: eliminated AND tank_zone (finished 1-15, worst
   record, drafted #1; in week 13 the conservative elimination rule still
   showed them alive because the AFC's first-team-out had only 6 wins).
2. 2020 KC week 17: clinched AND locked_seed (rested starters week 17; #1
   seed already secured).
3. 2023 CAR weeks 15-17: eliminated (finished 2-15; out well before week 15).

A validation failure aborts the run — the heuristic would be wrong, not the
history.

## Method

Week-blocked joint multinomial block bootstrap (primary), season-blocked
secondary, algorithm-identical to `scripts/redzone_reversion_screen.py` /
`scripts/nfl_bias_battery_screen.py`. 20,000 samples, seed 20260821,
accuracy_points full-slate units. Every cell is reported and recorded
regardless of sign or interval shape; an interval whose P+ is near 0.5 is the
EXPECTED outcome for a real-but-small signal and is never a closing ground
(binding taxonomy). Terminal classifications require an admissible
`--closing-ground`; everything else is `unresolved_below_power`.

## Measured results (2026-08-21 run)

All numbers below are **measured** this session: artifact
`artifacts/motivation_ladder_screen/20260821T182643Z/results.json`, produced by
`scripts/motivation_ladder_screen.py` against schedules
`data/raw/20260817T235649Z/schedules.parquet` (4,317 REG close-graded games;
846 (season, date) state snapshots; population 2,768 team-games in weeks
13-18). The three historical spot-checks passed before scoring (measured:
JAX-2020 weeks 14+ eliminated AND tank_zone; KC-2020 week 17 clinched AND
locked_seed; CAR-2023 mid-December eliminated).

### Cell results (measured; week-blocked primary, accuracy_points full-slate units, P+ = probability_positive)

| # | cell | n_flag | effect pts | week-blocked 95% CI | P+ | season-blocked P+ |
|---|------|--------|-----------|---------------------|----|--------------------|
| M1 | elim_visitor_alive_host | 241 | +0.020 | [-0.565, +0.598] | 0.509 | 0.517 |
| M2 | locked_seed_wk16_18 | 98 | +0.150 | [-0.245, +0.537] | 0.754 | 0.763 |
| M3 | fighter_vs_nothing | 74 | -0.334 | [-0.650, -0.019] | 0.015 | 0.046 |
| M4 | tank_zone_wk14_18 | 144 | +0.305 | [-0.079, +0.697] | 0.933 | 0.986 |

### Classification

All four cells are category 3, `unresolved_below_power` (measured inputs,
classification **inferred** from them):

- M1: interval crosses zero at both blockings; P+ ~0.51 is a coin flip.
- M2: leans opposite to the predeclared direction (locked seeds covered MORE,
  not less) but neither interval excludes zero.
- M4: leans opposite to prediction (tank-zone teams over-covered), and the
  season-blocked secondary excludes zero on that opposite side — but the
  declared PRIMARY week-blocked interval crosses zero (-0.079 lower bound),
  so the wrong sign is not resolved on the primary grading. Continuous
  evidence, not a finding; recorded for direction and pooling.
- M3: the primary week-blocked interval sits WHOLLY below zero against the
  predeclared +1 direction (P+ 0.015), which under the binding taxonomy is
  the necessary condition for `wrong_sign_resolved`. It is recorded here as
  `unresolved_below_power` rather than closed because the declared secondary
  season-blocked interval crosses zero ([−0.698, +0.067]), so the wrong sign
  does not replicate across both predeclared blockings, n_flag=74 is the
  thinnest cell in the battery, and no positive control was run. If the owner
  elects to treat the primary blocking alone as decisive, reclassifying to
  `refuted_mechanism` with `--closing-ground wrong_sign_resolved` would be
  admissible under the validator's letter — that is an owner decision, not
  one this screen makes silently.

No cell had a reliability exclusion available (event-defined states have no
split-half trait), and no positive control exists for this family, so no
terminal classification is currently admissible beyond the M3 question above.
The coherent-direction read (**inferred**, my reasoning, not evidence): M2/M4
both lean toward late-season BAD teams covering more than priced and M3's
fighters covering less — i.e., every cell leans toward "the side with
something left to prove underperforms its price late" being backwards in this
sample — but each cell alone is unresolved and they share one standings
engine, so they are correlated decompositions, not independent confirmations.

# Survivor planner (POL-07)

## Decision and scope

`nfl_ats.survivor.build_survivor_plan` turns multiweek straight-up win
probabilities into one team per week, with each team usable at most once
(**read:** `src/nfl_ats/survivor.py`, `build_survivor_plan`). It is pool
decision support only: it does not place wagers and its renderer labels model
probabilities as estimates rather than guarantees (**read:**
`src/nfl_ats/survivor.py`, `survivor_plan_markdown`).

The input uses `home_win_probability`, filtered to a named outcome method. It
never substitutes `home_cover_probability`; those are different targets
(**read:** `src/nfl_ats/pool.py`, `build_straight_up_pool_card`; and
`src/nfl_ats/survivor.py`, `_validate_and_expand`). The default method is the
dedicated `straight_up` outcome model (**read:** `src/nfl_ats/survivor.py`,
`build_survivor_plan`).

## Optimization and audit trail

The planner maximizes the product of selected win probabilities over the whole
requested horizon. Equivalently, it finds the maximum-weight one-to-one matching
of weeks to teams using log probabilities (**read:**
`src/nfl_ats/survivor.py`, `_hungarian_assignment`). A focused known-answer test
shows the planner taking a 75% team over an 80% team now to preserve a 95% later
use, producing the optimal 71.25% two-week survival probability (**measured:**
`tests/test_survivor.py::test_planner_trades_current_probability_for_future_team_value`).

Every output row retains season, week, game, side, opponent, probability, model
method, whether the choice was locked, and the teams already consumed. It also
separates the following quantities (**read:** `src/nfl_ats/survivor.py`, output
row construction in `build_survivor_plan`):

- `pick_probability`: the current week's estimated survival chance;
- `current_probability_sacrifice`: probability conceded versus the best
  currently available team;
- `future_team_opportunity_cost`: the reduction in the best achievable future
  survival probability from consuming this team;
- cumulative, remaining-horizon, and full-horizon survival probabilities.

Exact ties are deterministic. `used_teams` represents choices made before the
planning horizon; `locked_picks` preserves choices already submitted inside it.
Duplicate use, unavailable locked teams, incomplete/nonconsecutive requested
weeks, duplicate team appearances within a week, invalid probabilities, and
infeasible assignments fail closed (**measured:** the 12 focused cases in
`tests/test_survivor.py`).

No CLI is added. Existing prediction commands produce a single weekly artifact,
while this API intentionally requires a multiweek probability surface; inventing
a future-schedule data source in a CLI would obscure rather than strengthen the
input provenance (**inferred:** this is an integration choice based on the
single-week writers read in `src/nfl_ats/cli.py:3844-3854`).

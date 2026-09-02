"""The one graph `team_stat` column used by the `weak_stack_graph_off_rush_epa`
candidate profile (docs/graph_team_stat_off_rush_epa_on_production.md).

`docs/graph_ratings_v2_screen.md` section 8 screened 38 graph-propagated
statistics against a BARE market baseline. `off_rush_epa_per_play` read +1.609
accuracy points at P+ 0.911 against zero, but only the 53.5th percentile of its
own within-week permutation null, which centres at +1.450 -- that doc's own
words: "essentially all of its apparent edge is the artifact." What makes the
cell worth a window anyway is a DIFFERENT, disjoint measurement: the registry's
`graph_input_screen_off_rush_epa_per_play` (opener-graded 2020-2025, n=1,503)
reads +1.996 points at P+ 0.828 with split-half reliability 0.987 -- the single
highest reliability figure recorded anywhere in this family.

The project's own recorded lesson -- "composition is not the signal" -- is that
a component positive alone can go negative once stacked on the chain that is
actually PLAYED, which is exactly what happened to both siblings of this
experiment (`off_sack_rate`: docs/graph_team_stat_on_production.md section 7,
-0.935 pts, P+ 0.122; `def_yards_per_play`:
docs/graph_team_stat_def_ypp_on_production.md section 7, -0.668 pts, P+ 0.189).
This module builds the analogous stacked feature for `off_rush_epa_per_play`:
the same graph-propagated column, at the SAME structural configuration frozen
in `docs/graph_ratings_v2_screen.md` section 5 (inherited, not refit, here),
additively joined onto a feature table by `game_id`.

Mirrors `nfl_ats.graph_team_stat_def_ypp_production_feature` (the
`def_yards_per_play` sibling module) and
`nfl_ats.forecast_weather_features.attach_forecast_weather_features`'s additive
-merge discipline: every pre-existing column comes back bit-identical, only the
one new column is added.
"""

from __future__ import annotations

import pandas as pd

from nfl_ats.constants import GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.graph_ratings_v2 import (
    GraphRatingV2Config,
    add_graph_ratings_v2_features,
    katz_feature_columns,
)

#: Frozen in docs/graph_ratings_v2_screen.md section 5, before scoring, and
#: not retuned on NFL. Identical to `scripts/graph_team_stat_screen.py`'s
#: FROZEN_STRUCTURE and to both sibling modules' own FROZEN_STRUCTURE -- this
#: module inherits that freeze, it does not repeat the decision.
FROZEN_STRUCTURE: dict[str, object] = {
    "alpha": 0.85,
    "half_life_weeks": 8.0,
    "max_row_l1": 1.0,
    "prior_weight": 1.0,
    "min_games": 16,
    "propagation": "signed_katz",
    "injury_beta": 0.0,
}

SIGNAL_FAMILY = "off_rush_epa_per_play"
GRAPH_OFF_RUSH_EPA_COLUMN = GRAPH_TEAM_STAT_OFF_RUSH_EPA_FEATURE_COLUMNS[0]


def _graph_config() -> GraphRatingV2Config:
    return GraphRatingV2Config(
        edge_signal="team_stat",
        signal_column=SIGNAL_FAMILY,
        **FROZEN_STRUCTURE,  # type: ignore[arg-type]
    )


def derive_graph_off_rush_epa_feature(features: pd.DataFrame) -> pd.DataFrame:
    """Return a ``(game_id, graph_off_rush_epa_column)`` frame.

    Computed only over games with a settled ``result``, matching
    ``scripts/graph_team_stat_screen.py``'s own ``build_arm_columns`` and both
    sibling modules: the graph's leak-safe weekly walk-forward
    (``add_graph_ratings_v2_features``) reads a week's rating through the PRIOR
    week only regardless of which games are included, but restricting the input
    to completed games keeps this a research-only, historical-evaluation
    column, matching every other candidate profile's table
    (``weak_stack_v3``/``weak_stack_v4``/``weak_stack_graph_sack``/
    ``weak_stack_graph_def_ypp``), rather than a production-live feature this
    session is not declaring as one.
    """

    required = {"game_id", "season", "week", "gameday", "home_team", "away_team", "result"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    config = _graph_config()
    completed = features.loc[features["result"].notna()].copy()
    rated = add_graph_ratings_v2_features(completed, config)
    diff_column = katz_feature_columns(config)[2]

    derived = pd.DataFrame(
        {
            "game_id": rated["game_id"].astype(str),
            GRAPH_OFF_RUSH_EPA_COLUMN: pd.to_numeric(rated[diff_column], errors="coerce"),
        }
    )
    return derived


def attach_graph_off_rush_epa_feature(features: pd.DataFrame) -> pd.DataFrame:
    """Additively join the one new column onto ``features``.

    Every pre-existing column is returned bit-identical; only
    ``graph_v2_team_stat_off_rush_epa_per_play_katz_diff`` is added. Games with
    no settled result (or that the graph could not rate, e.g. before
    ``min_games`` is reached) come back NaN, left NaN on purpose: imputation
    belongs to the model's own training-fold median (``fit_margin_model``), not
    to a feature builder that can see every season at once.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if GRAPH_OFF_RUSH_EPA_COLUMN in features.columns:
        raise DataContractError(f"features already carries {GRAPH_OFF_RUSH_EPA_COLUMN!r}")

    derived = derive_graph_off_rush_epa_feature(features)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_graph"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_graph") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "FROZEN_STRUCTURE",
    "GRAPH_OFF_RUSH_EPA_COLUMN",
    "SIGNAL_FAMILY",
    "attach_graph_off_rush_epa_feature",
    "derive_graph_off_rush_epa_feature",
]

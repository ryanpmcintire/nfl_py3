"""CFB cross-league replication of the graph ``team_stat`` arm (WP8).

Predeclared in ``docs/graph_team_stat_cfb_replication.md`` BEFORE any outcome
sign was computed. Read that document first: it declares the three cells, the
comparator, the grade, the frozen structural configuration, the two adaptations
and the recording rules.

This module is the FEATURE side only. It attaches exactly one column -- the
signed-Katz rating differential of one pregame CFB team statistic propagated
over the CFB schedule graph -- to the XLG-03 CFB game table. It fits nothing,
grades nothing, and touches no registry. The evaluator lives in
``scripts/graph_team_stat_cfb_replication.py``.

Why CFB. The NFL ``graph_input_screen`` family is the weak-signal registry's
most one-sided (82 cells, 63 favouring the candidate), yet its strongest cells
went negative once stacked on the NFL production chain. The open question is
whether the graph transform adds anything to a market-residual model at all,
and college football answers it on new football at no NFL window cost: the CFB
schedule graph is roughly five times sparser than the NFL's (measured edge
densities 0.085 vs 0.42, see the predeclaration section 1) and is heavily
conference-clustered, which is precisely where opponent-adjustment should have
the most to add over a raw statistic.

Two declared adaptations, both in the predeclaration section 5:

* **A1** -- graph nodes are ESPN team IDs (``home_id``/``away_id``), not team
  name strings. Measured: 5 of 137 ids carry more than one name string across
  seasons (program rebrands) while every name maps to exactly one id, so names
  would split one program into two nodes and reset its propagated rating.
* **A2** -- the graph is BUILT over every season present in the table and
  SCORED only on the frozen XLG-03 clean core. The walk-forward is leak-safe,
  so warm-up seasons cost nothing and are never an evaluation window. This
  module builds; the script decides what is scored.

Nothing else is adapted. ``min_games=16`` is reachable on a 12-game CFB regular
season because the gate counts games cumulatively across the corpus, not per
team, so only the first build week goes unrated.

**Leak safety.** ``add_graph_ratings_v2_features`` gives every game in week
``w`` a rating read from the graph as accumulated through week ``w-1``, folding
week ``w``'s own edges in only after the whole week has been assigned. The CFB
team-state columns are themselves strictly-lagged pregame EWMs (see
``nfl_ats.cfb_features``), so the edge weight is knowable before kickoff and the
propagation opponent-ADJUSTS a statistic rather than absorbing an outcome.
``tests/test_graph_team_stat_cfb_feature.py`` is the proof; this docstring is
not.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.graph_ratings_v2 import (
    GraphRatingV2Config,
    add_graph_ratings_v2_features,
    katz_feature_columns,
)

#: The three declared cells, one per CFB team-stat column, frozen in
#: ``docs/graph_team_stat_cfb_replication.md`` section 3 before any sign was
#: seen. No fourth cell may be added after the first three are scored.
#:
#: The mapping to the NFL screen's cells is deliberately honest rather than
#: three-for-three: the CFB table has no yards-per-play column, no rush/pass
#: split, and **no sack data at all**, so the NFL ``off_sack_rate`` cell has no
#: CFB counterpart. ``off_success_rate`` is declared as the third-best
#: available team-stat column and is NOT presented as a sack-rate analogue.
CFB_GRAPH_CELLS: Final[tuple[str, ...]] = (
    "def_epa_per_play",
    "off_epa_per_play",
    "off_success_rate",
)

#: What each cell replicates on the NFL side, for reporting only. Values are
#: the NFL ``graph_input_screen`` cell whose construct class this CFB cell
#: stands in for, plus the nearest same-named NFL cell where one exists.
CFB_GRAPH_CELL_NFL_COUNTERPART: Final[dict[str, str]] = {
    "def_epa_per_play": (
        "NFL def_yards_per_play (screen P+ 0.711; on-production -0.668, P+ 0.189); "
        "same construct class, and NFL def_epa_per_play itself scored P+ 0.583"
    ),
    "off_epa_per_play": (
        "NFL off_rush_epa_per_play (screen P+ 0.828); CFB has no rush/pass split, "
        "so this is all plays, and NFL off_epa_per_play itself scored P+ 0.659"
    ),
    "off_success_rate": (
        "NOT a counterpart to NFL off_sack_rate -- CFB has no sack column at all. "
        "Declared as the third-best available team-stat column; nearest NFL "
        "sibling is pbp_off_success_rate (P+ 0.709)"
    ),
}

#: Byte-for-byte the structural configuration
#: ``scripts/graph_team_stat_screen.py::FROZEN_STRUCTURE`` froze for NFL. Never
#: retuned on CFB: the point of a replication is the SAME transform on new
#: football, so a CFB refit would answer a different question.
CFB_GRAPH_FROZEN_STRUCTURE: Final[dict[str, Any]] = {
    "alpha": 0.85,
    "half_life_weeks": 8.0,
    "max_row_l1": 1.0,
    "prior_weight": 1.0,
    "min_games": 16,
    "propagation": "signed_katz",
    "injury_beta": 0.0,
}

#: Adaptation A1: the graph node key. Kept as module constants so the test
#: suite asserts against the same names the builder uses.
CFB_HOME_ID_COLUMN: Final[str] = "home_id"
CFB_AWAY_ID_COLUMN: Final[str] = "away_id"

_BASE_REQUIRED: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "result",
    CFB_HOME_ID_COLUMN,
    CFB_AWAY_ID_COLUMN,
)


def _validate_cell(cell: str) -> None:
    if cell not in CFB_GRAPH_CELLS:
        raise ValueError(
            f"{cell!r} is not one of the three predeclared CFB cells "
            f"{CFB_GRAPH_CELLS}; docs/graph_team_stat_cfb_replication.md section 3 "
            "froze the cell list before any sign was computed, so an undeclared "
            "column is refused rather than silently graphed"
        )


def cfb_cell_columns(cell: str) -> tuple[str, str]:
    """The ``(home, away)`` CFB column pair one cell reads."""

    _validate_cell(cell)
    return (f"home_{cell}", f"away_{cell}")


def cfb_graph_config(cell: str) -> GraphRatingV2Config:
    """The frozen ``team_stat`` configuration for one declared cell."""

    _validate_cell(cell)
    home_column, away_column = cfb_cell_columns(cell)
    return GraphRatingV2Config(
        edge_signal="team_stat",
        signal_column=cell,
        signal_column_pair=(home_column, away_column),
        **CFB_GRAPH_FROZEN_STRUCTURE,
    )


def cfb_graph_column(cell: str) -> str:
    """Name of the single katz-differential column this module attaches."""

    return katz_feature_columns(cfb_graph_config(cell))[2]


def add_cfb_graph_team_stat_feature(games: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Attach one CFB graph ``team_stat`` katz-differential column.

    Returns a copy of ``games`` in the caller's original row order and index,
    with exactly one column added. The graph itself is built on ESPN team ids
    (adaptation A1) over every row supplied; callers control the build corpus
    by what they pass in, and control the SCORED window separately.

    The join back is by ``game_id`` rather than by position, because
    ``add_graph_ratings_v2_features`` re-sorts and re-indexes its output. A
    duplicated ``game_id`` would make that join ambiguous, so it is refused
    rather than silently fanned out.
    """

    _validate_cell(cell)
    home_column, away_column = cfb_cell_columns(cell)
    missing = sorted({*_BASE_REQUIRED, home_column, away_column}.difference(games.columns))
    if missing:
        raise DataContractError(
            f"CFB graph team_stat feature requires columns: {', '.join(missing)}"
        )
    if games["game_id"].duplicated().any():
        raise DataContractError(
            "CFB graph team_stat feature requires unique game_id values; the graph "
            "column is joined back by game_id and duplicates would make that join "
            "ambiguous"
        )

    config = cfb_graph_config(cell)
    column = cfb_graph_column(cell)

    graph_input = games.copy()
    # A1: node identity is the ESPN id, cast through Int64 -> str so 12345 and
    # 12345.0 cannot become two different nodes. The caller's frame is never
    # mutated -- graph_input is already a copy.
    graph_input["home_team"] = (
        pd.to_numeric(graph_input[CFB_HOME_ID_COLUMN], errors="raise").astype("int64").astype(str)
    )
    graph_input["away_team"] = (
        pd.to_numeric(graph_input[CFB_AWAY_ID_COLUMN], errors="raise").astype("int64").astype(str)
    )

    rated = add_graph_ratings_v2_features(graph_input, config)
    values = pd.Series(
        pd.to_numeric(rated[column], errors="coerce").to_numpy(),
        index=rated["game_id"].to_numpy(),
    )

    result = games.copy()
    result[column] = games["game_id"].map(values).astype(float)
    return result

"""Graph ratings v2: residual edges, uncompressed magnitude, signed Katz centrality.

Predeclared in ``docs/graph_ratings_v2.md`` before this module ever touches an
ATS outcome. Read that document first -- it is the design record this module
implements, including which arms exist and why, and what stays undecided
until the input-screen agent's list lands.

This is a NEW exploration lane, not a patch to ``nfl_ats.graph_ratings``
(``add_schedule_strength_features``), which stays exactly as-is and keeps
powering production. Nothing here is wired into ``features.py`` or any
production pick path.

Two measured defects in the original module motivated a rebuild rather than
a patch:

1. **Magnitude compression.** ``performance[...] += 1.0 + min(margin, 28.0) /
   14.0`` squeezes every game's edge weight into ``[1.0, 3.0]`` and treats
   every margin above 28 identically. A one-point win and a forty-point win
   were nearly the same edge.
2. **Raw-margin edges measure team quality**, which this project has an
   established, measured ceiling for (features that only measure team
   quality better are bounded near zero -- the market already prices it).
   ``graph_pagerank`` (the original's PageRank column) was reported at a
   0.997 split-half reliability -- reported by the task brief that
   commissioned this rebuild, not independently re-verified in this session;
   treated here only as a motivation for keeping a raw-margin CONTROL arm,
   not as a number this module asserts.

Four arms exist, all off-except-`residual` by default so a caller must opt in:

* ``edge_signal="residual"`` (default) -- edges weighted by
  ``ats_margin = result - spread_line``: who beat their NUMBER, not who is
  good. This is the treatment arm.
* ``edge_signal="raw_margin"`` -- edges weighted by ``result`` alone, the
  exact quantity the original module used. This is the POSITIVE CONTROL: it
  should reproduce (approximately) the original's team-quality signature,
  and per the project's measured ceiling should score near zero against ATS
  outcomes once that scoring happens. A residual arm that does not beat this
  control on the same engine is not evidence of anything.
* ``edge_signal="team_stat"`` with ``signal_column="<family>"`` -- edges
  weighted by ``home_<family> - away_<family>``, one graph per SCREENED
  statistic. This is the arm the input screen
  (``docs/graph_input_screen.md``) exists to feed: each statistic is scored
  on its own first, and only the survivors are propagated. Note what this
  arm is and is not -- the family columns are PREGAME rolling team values
  (measured: ``home_def_takeaway_rate`` is populated in week 1), so the edge
  is knowable before kickoff and the graph is opponent-ADJUSTING a statistic
  rather than absorbing an outcome. Leak-safety is therefore stricter than
  the outcome arms need, not looser: a week's own stat differentials are
  still folded in only AFTER every game that week has been assigned its
  ratings, so a team_stat rating for week ``w`` reads the graph through
  week ``w-1`` exactly like every other arm.
* ``propagation="signed_katz"`` (default) -- signed edges, Katz centrality,
  row-Linf-constrained (see below). No PageRank/HITS interpretation applies
  once edges go negative (no Perron-Frobenius, no stationary distribution).
* ``propagation="nonneg_pagerank"`` -- the non-negative arm: magnitude-only
  edges (loser -> winner, uncompressed), PageRank centrality, plus HITS
  offense/defense (HITS requires non-negativity for a unique interpretable
  top eigenvector, so it is dropped entirely from the signed arm).
* ``injury_beta`` (default ``0.0``, i.e. OFF) -- an optional edge MODIFIER,
  never a per-game attribution. Per-game injury attribution is not
  identifiable (one observation, many causes, no counterfactual) and this
  module does not attempt it. What is estimable is a single shared
  coefficient: when the side that fell short of expectation was also
  carrying heavy ``injury_value_lost``, discount that game's edge so the
  graph does not conclude the opponent is strong from a game played against
  a shorthanded team. Off by default; must be explicitly enabled.

## Signed Katz, not PageRank, and why the constraint is on the row not the entry

The owner asked whether bounding learned weights to ``[-1, 1]`` helps or
hurts. It hurts: PageRank's convergence guarantee comes from the transition
matrix being row-STOCHASTIC, giving spectral radius exactly 1. Clipping
entries of an N-node signed matrix to ``[-1, 1]`` lets a single row's sum
reach N, so the spectral radius can reach N and the power iteration
diverges. The actual guarantee is on the operator norm ``||W||_inf`` (the
maximum row sum of absolute values): constraining THAT to ``<= 1`` bounds
the spectral radius by the same amount (spectral radius ``<=`` any induced
matrix norm), and it is MORE expressive than clipping every entry, because
one strong edge is allowed to dominate a row instead of every edge being
forced small. :func:`_constrain_row_linf` implements this: rows already
under the bound are untouched; only rows that exceed it are rescaled down to
exactly the bound.

With negative entries the fixed point ``x = (I - alpha*W)^-1 v`` is not
PageRank -- there is no random walk, no stationary distribution, and no
"importance" reading. It is signed **Katz centrality**, and this module
names it that throughout (see :func:`signed_katz_centrality`).
:func:`_katz_fixed_point` is the low-level Neumann-series iterate
(``x_{k+1} = v + alpha * W @ x_k``), exposed separately from the
constraint so ``tests/test_graph_ratings_v2.py`` can prove convergence
DEPENDS on the constraint (feed it a deliberately unconstrained,
spectral-radius-violating matrix and watch it fail to converge).

## Fitting where the parameters are affordable (the XLG pattern)

``data/processed/cfb_game_features.parquet`` holds 12,500 games versus NFL's
4,902 (measured, both read directly by this module's tests and the
demonstration in ``docs/graph_ratings_v2.md``). This project has an
established cross-league transfer pattern (``docs/scaling_and_transfer.md``,
the "XLG" family): fit STRUCTURE where there is enough data to support it,
transfer the structure, refit only what must differ per league.

:func:`select_structural_config_on_cfb` is that mechanism here. It is
STRUCTURAL selection only -- the hyperparameters that have many degrees of
freedom relative to NFL's smaller n (``alpha``, ``half_life_weeks``,
``offseason_retention``, ``max_row_l1``, ``prior_weight``) -- scored by a
continuous coherence diagnostic (:func:`cfb_structural_coherence`, the
Pearson correlation between the pregame rating diff and the game's own
market-relative or raw margin) walked forward on the full CFB corpus. This
is explicitly NOT an ATS accuracy number and is never recorded as a finding
by this module: no call in this file touches
``nfl_ats.weak_signals.record_signal`` or any rotation-registry command.
Team-level ratings themselves are never transferred -- NFL's graph is always
built fresh from NFL games under whichever config is frozen. ``injury_beta``
is deliberately excluded from CFB structural selection: CFB has no
player-level injury table, so that single scalar coefficient can only be
fit on NFL directly, and per this project's binding rule against inventing
constraints, that fit is deferred to a future session with its own
predeclaration -- ``injury_beta`` stays at the caller-supplied value here,
never auto-tuned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION
from nfl_ats.data import DataContractError

FloatArray = NDArray[np.float64]

EdgeSignal = Literal["residual", "raw_margin", "team_stat"]
Propagation = Literal["signed_katz", "nonneg_pagerank"]

#: Per-team injury value-lost components (``player_value`` profile), summed
#: to a single team-game total. Matches the two columns
#: ``nfl_ats.surgical_gating.VALUE_LOST_DIFF_COLUMNS`` differences, but here
#: read as home/away TOTALS rather than a diff, since the injury modifier
#: needs to know which SIDE was hurt, not just the gap between them.
HOME_INJURY_VALUE_LOST_COLUMNS: tuple[str, ...] = (
    "home_injury_skill_epa_value_lost",
    "home_injury_defense_disruption_value_lost",
)
AWAY_INJURY_VALUE_LOST_COLUMNS: tuple[str, ...] = (
    "away_injury_skill_epa_value_lost",
    "away_injury_defense_disruption_value_lost",
)


@dataclass(frozen=True)
class GraphRatingV2Config:
    """Fixed research configuration for the v2 graph engine.

    ``alpha`` is the Katz/PageRank damping factor and ``max_row_l1`` is the
    ``||W||_inf`` bound enforced on the signed matrix before propagation
    (see module docstring). ``alpha * max_row_l1 < 1`` is validated
    explicitly: that product IS the spectral-radius bound this module
    guarantees, so an invalid combination is a configuration error, not a
    runtime surprise.
    """

    edge_signal: EdgeSignal = "residual"
    signal_column: str | None = None
    signal_column_pair: tuple[str, str] | None = None
    propagation: Propagation = "signed_katz"
    half_life_weeks: float = 8.0
    offseason_retention: float = DEFAULT_OFFSEASON_RETENTION
    offseason_age_weeks: int = 8
    alpha: float = 0.85
    max_row_l1: float = 1.0
    prior_weight: float = 1.0
    min_games: int = 16
    injury_beta: float = 0.0
    iterations: int = 500
    tolerance: float = 1e-10
    column_prefix: str | None = None

    def validate(self) -> None:
        if self.edge_signal not in ("residual", "raw_margin", "team_stat"):
            raise ValueError(
                "edge_signal must be 'residual', 'raw_margin' or 'team_stat', "
                f"got {self.edge_signal!r}"
            )
        if self.edge_signal == "team_stat":
            if not self.signal_column:
                raise ValueError(
                    "edge_signal='team_stat' requires signal_column -- the screened "
                    "family name whose home_/away_ pair supplies the edge weight"
                )
            if self.signal_column.startswith(("home_", "away_")):
                raise ValueError(
                    "signal_column is the FAMILY name (e.g. 'def_takeaway_rate'), not one "
                    f"side of the pair; got {self.signal_column!r}"
                )
            if self.signal_column_pair is not None and len(self.signal_column_pair) != 2:
                raise ValueError(
                    "signal_column_pair must be exactly (home_column, away_column), got "
                    f"{self.signal_column_pair!r}"
                )
        elif self.signal_column is not None or self.signal_column_pair is not None:
            raise ValueError(
                "signal_column/signal_column_pair are only meaningful with "
                f"edge_signal='team_stat', got edge_signal={self.edge_signal!r}"
            )
        if self.propagation not in ("signed_katz", "nonneg_pagerank"):
            raise ValueError(
                f"propagation must be 'signed_katz' or 'nonneg_pagerank', got {self.propagation!r}"
            )
        if self.half_life_weeks <= 0:
            raise ValueError("half_life_weeks must be positive")
        if not 0.0 <= self.offseason_retention <= 1.0:
            raise ValueError("offseason_retention must be between 0 and 1")
        if self.offseason_age_weeks < 0:
            raise ValueError("offseason_age_weeks cannot be negative")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.max_row_l1 <= 0.0:
            raise ValueError("max_row_l1 must be positive")
        if self.alpha * self.max_row_l1 >= 1.0:
            raise ValueError(
                "alpha * max_row_l1 must be strictly < 1 -- that product bounds the "
                "spectral radius of the propagation matrix and is the actual "
                "convergence guarantee (see module docstring)"
            )
        if self.prior_weight < 0.0:
            raise ValueError("prior_weight cannot be negative")
        if self.min_games < 1:
            raise ValueError("min_games must be positive")
        if self.injury_beta < 0.0:
            raise ValueError("injury_beta cannot be negative")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")


def default_column_prefix(config: GraphRatingV2Config) -> str:
    suffix = "_injmod" if config.injury_beta > 0.0 else ""
    if config.edge_signal == "team_stat":
        return f"graph_v2_team_stat_{config.signal_column}{suffix}"
    return f"graph_v2_{config.edge_signal}{suffix}"


def team_stat_columns(config: GraphRatingV2Config) -> tuple[str, str]:
    """The ``(home, away)`` column pair a ``team_stat`` arm reads for one family.

    The feature table carries BOTH naming conventions (measured via
    ``reliability_map.discover_family_pairs``): a ``prefix`` form
    (``home_def_takeaway_rate``) and a ``suffix`` form
    (``gap_division_revenge_home``). The prefix form is the default because it
    covers most families; ``signal_column_pair`` names the columns explicitly
    for the rest, so a suffix-form family is a configuration value rather than
    a family this arm silently cannot express.
    """

    if config.signal_column_pair is not None:
        return (config.signal_column_pair[0], config.signal_column_pair[1])
    return (f"home_{config.signal_column}", f"away_{config.signal_column}")


def katz_feature_columns(config: GraphRatingV2Config) -> tuple[str, ...]:
    prefix = config.column_prefix or default_column_prefix(config)
    if config.propagation == "signed_katz":
        return (
            f"home_{prefix}_katz",
            f"away_{prefix}_katz",
            f"{prefix}_katz_diff",
        )
    return (
        f"home_{prefix}_pagerank",
        f"away_{prefix}_pagerank",
        f"{prefix}_pagerank_diff",
        f"home_{prefix}_offense",
        f"away_{prefix}_offense",
        f"home_{prefix}_defense",
        f"away_{prefix}_defense",
        f"{prefix}_matchup_diff",
    )


# --------------------------------------------------------------------------
# Small numeric primitives, independently testable.
# --------------------------------------------------------------------------


def _standardize(values: FloatArray) -> FloatArray:
    """Cross-sectional z-score across teams THIS week only -- never temporal."""

    centered = values - float(values.mean())
    scale = float(values.std())
    if not math.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return centered / scale


def _expand_matrix(
    matrix: FloatArray,
    existing_teams: list[str],
    desired_teams: list[str],
) -> FloatArray:
    if existing_teams == desired_teams:
        return matrix
    expanded = np.zeros((len(desired_teams), len(desired_teams)), dtype=np.float64)
    desired_index = {team: index for index, team in enumerate(desired_teams)}
    for old_row, row_team in enumerate(existing_teams):
        new_row = desired_index[row_team]
        for old_column, column_team in enumerate(existing_teams):
            expanded[new_row, desired_index[column_team]] = matrix[old_row, old_column]
    return expanded


def _constrain_row_linf(matrix: FloatArray, max_row_l1: float) -> FloatArray:
    """Rescale only the rows whose absolute-value sum exceeds ``max_row_l1``.

    This is the actual convergence guarantee for the signed arm: it bounds
    ``||W||_inf`` (the operator/infinity norm), which bounds the spectral
    radius, WITHOUT forcing every entry small the way clipping to
    ``[-1, 1]`` would. Rows already under the bound pass through unchanged
    -- one strong edge is allowed to dominate a row.
    """

    if matrix.size == 0:
        return matrix
    row_l1 = np.abs(matrix).sum(axis=1)
    scale = np.ones_like(row_l1)
    over = row_l1 > max_row_l1
    scale[over] = max_row_l1 / row_l1[over]
    return np.asarray(matrix * scale[:, None], dtype=np.float64)


def _katz_fixed_point(
    matrix: FloatArray,
    base: FloatArray,
    *,
    alpha: float,
    iterations: int,
    tolerance: float,
) -> tuple[FloatArray, bool]:
    """The Neumann-series iterate ``x_{k+1} = base + alpha * matrix @ x_k``.

    Deliberately does NOT apply :func:`_constrain_row_linf` -- callers that
    want the guaranteed-convergent form should constrain first (see
    :func:`signed_katz_centrality`). Exposed unconstrained so a test can
    feed it a matrix that violates the spectral-radius bound and observe
    non-convergence directly, proving the constraint is load-bearing rather
    than decorative.

    Returns ``(x, converged)``. A norm blow-up (``> 1e12``) is treated as
    divergence and returned early rather than running to ``iterations`` and
    risking overflow/NaN.
    """

    size = matrix.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.float64), True
    x = base.copy()
    for _ in range(iterations):
        updated: FloatArray = np.asarray(base + alpha * (matrix @ x), dtype=np.float64)
        if not np.all(np.isfinite(updated)) or float(np.abs(updated).max()) > 1e12:
            return updated, False
        if float(np.abs(updated - x).sum()) <= tolerance:
            return updated, True
        x = updated
    return x, False


def signed_katz_centrality(
    matrix: FloatArray,
    *,
    alpha: float,
    max_row_l1: float,
    iterations: int = 500,
    tolerance: float = 1e-10,
) -> FloatArray:
    """Signed Katz centrality: ``x = (I - alpha*W)^-1 v`` with ``v`` the
    uniform base vector (every node gets a base score of 1, the signed
    analogue of PageRank's teleport term). ``matrix`` is constrained to
    ``||.||_inf <= max_row_l1`` first, which is what guarantees the fixed
    point exists and the iteration below converges to it.

    Not PageRank: with signed entries there is no random walk and no
    stationary distribution, only a linear fixed point. Named accordingly.
    """

    size = matrix.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.float64)
    constrained = _constrain_row_linf(matrix, max_row_l1)
    base = np.ones(size, dtype=np.float64)
    x, _converged = _katz_fixed_point(
        constrained, base, alpha=alpha, iterations=iterations, tolerance=tolerance
    )
    return x


def _page_rank(
    adjacency: FloatArray, *, damping: float, iterations: int, tolerance: float
) -> FloatArray:
    size = adjacency.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.float64)
    row_sums = adjacency.sum(axis=1)
    transition: FloatArray = np.divide(
        adjacency,
        row_sums[:, None],
        out=np.full_like(adjacency, 1.0 / size),
        where=row_sums[:, None] > 0,
    )
    scores = np.full(size, 1.0 / size, dtype=np.float64)
    teleport = (1.0 - damping) / size
    for _ in range(iterations):
        updated: FloatArray = np.asarray(
            teleport + damping * (transition.T @ scores), dtype=np.float64
        )
        if float(np.abs(updated - scores).sum()) <= tolerance:
            return updated / float(updated.sum())
        scores = updated
    return scores / float(scores.sum())


def _hits(
    adjacency: FloatArray, *, iterations: int, tolerance: float
) -> tuple[FloatArray, FloatArray]:
    size = adjacency.shape[0]
    if size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    authority = np.full(size, 1.0 / math.sqrt(size), dtype=np.float64)
    hub = authority.copy()
    for _ in range(iterations):
        new_hub = adjacency @ authority
        hub_norm = float(np.linalg.norm(new_hub))
        if hub_norm > 0:
            new_hub /= hub_norm
        new_authority = adjacency.T @ new_hub
        authority_norm = float(np.linalg.norm(new_authority))
        if authority_norm > 0:
            new_authority /= authority_norm
        change = float(np.abs(new_hub - hub).sum() + np.abs(new_authority - authority).sum())
        hub, authority = new_hub, new_authority
        if change <= tolerance:
            break
    return hub, authority


def _injury_discount(home_lost: float, away_lost: float, signal: float, beta: float) -> float:
    """A single shared discount, never a per-game attribution.

    When ``signal`` (home-perspective residual or raw margin) is positive,
    home over-performed and AWAY is the side that fell short -- discount by
    away's injury value lost. When negative, discount by home's. A zero
    signal already contributes nothing (see edge construction), so beta is
    irrelevant there. ``beta == 0`` is a hard off-switch, returned before
    any lookup so the default arm never depends on injury columns existing.
    """

    if beta <= 0.0 or signal == 0.0:
        return 1.0
    underperformer_lost = away_lost if signal > 0.0 else home_lost
    if not math.isfinite(underperformer_lost) or underperformer_lost < 0.0:
        underperformer_lost = 0.0
    return 1.0 / (1.0 + beta * underperformer_lost)


def _sum_injury_value_lost(game: pd.Series, columns: tuple[str, ...]) -> float:
    total = 0.0
    for column in columns:
        value = game.get(column)
        if value is not None and pd.notna(value):
            total += float(value)
    return total


# --------------------------------------------------------------------------
# The leak-safe weekly walk-forward.
# --------------------------------------------------------------------------

_BASE_REQUIRED_COLUMNS = (
    "season",
    "week",
    "gameday",
    "game_id",
    "home_team",
    "away_team",
    "result",
)


def _required_columns(config: GraphRatingV2Config, games: pd.DataFrame) -> set[str]:
    required = set(_BASE_REQUIRED_COLUMNS)
    if config.edge_signal == "residual" and "ats_margin" not in games.columns:
        required.add("spread_line")
    if config.edge_signal == "team_stat":
        required.update(team_stat_columns(config))
    if config.injury_beta > 0.0:
        required.update(HOME_INJURY_VALUE_LOST_COLUMNS)
        required.update(AWAY_INJURY_VALUE_LOST_COLUMNS)
    if config.propagation == "nonneg_pagerank":
        required.update({"home_score", "away_score"})
    return required


def _signal_series(games: pd.DataFrame, config: GraphRatingV2Config) -> pd.Series:
    if config.edge_signal == "team_stat":
        home_column, away_column = team_stat_columns(config)
        return pd.to_numeric(games[home_column], errors="coerce") - pd.to_numeric(
            games[away_column], errors="coerce"
        )
    if config.edge_signal == "residual":
        if "ats_margin" in games.columns:
            return pd.to_numeric(games["ats_margin"], errors="coerce")
        return pd.to_numeric(games["result"], errors="coerce") - pd.to_numeric(
            games["spread_line"], errors="coerce"
        )
    return pd.to_numeric(games["result"], errors="coerce")


def add_graph_ratings_v2_features(
    games: pd.DataFrame,
    config: GraphRatingV2Config | None = None,
) -> pd.DataFrame:
    """Add pregame v2 graph-rating features for exactly one arm.

    Mirrors ``nfl_ats.graph_ratings.add_schedule_strength_features``'s
    leak-safe shape: ratings for every game in a week are read from the
    graph as accumulated through the PRIOR week, and only after every game
    in the current week has been assigned its features does that week's own
    results get folded into the graph. Call this once per arm (residual vs
    raw_margin control, signed_katz vs nonneg_pagerank, injury on/off); the
    output column names (:func:`katz_feature_columns`) are namespaced by
    ``edge_signal`` (and an ``_injmod`` suffix when the injury modifier is
    on) precisely so multiple arms can be merged onto the same frame without
    colliding.
    """

    settings = config or GraphRatingV2Config()
    settings.validate()
    missing = sorted(_required_columns(settings, games).difference(games.columns))
    if missing:
        raise DataContractError(f"graph_ratings_v2 requires columns: {', '.join(missing)}")

    columns = katz_feature_columns(settings)
    result = games.copy()
    for column in columns:
        result[column] = np.nan
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result = result.sort_values(["gameday", "game_id"]).copy()
    result["_graph_v2_signal"] = _signal_series(result, settings)

    graph_teams: list[str] = []
    signed_matrix = np.zeros((0, 0), dtype=np.float64)
    performance = np.zeros((0, 0), dtype=np.float64)
    scoring = np.zeros((0, 0), dtype=np.float64)
    weekly_decay = 0.5 ** (1.0 / settings.half_life_weeks)
    previous_season: int | None = None
    games_seen = 0

    for (season_value, _), weekly_games in result.groupby(["season", "week"], sort=True):
        season = int(str(season_value))
        if previous_season is not None and season != previous_season:
            gap = max(1, season - previous_season)
            retention = settings.offseason_retention**gap
            signed_matrix *= retention
            performance *= retention
            scoring *= retention
        previous_season = season
        signed_matrix *= weekly_decay
        performance *= weekly_decay
        scoring *= weekly_decay

        current_teams = set(weekly_games["home_team"].astype(str)) | set(
            weekly_games["away_team"].astype(str)
        )
        desired_teams = sorted(set(graph_teams) | current_teams)
        signed_matrix = _expand_matrix(signed_matrix, graph_teams, desired_teams)
        performance = _expand_matrix(performance, graph_teams, desired_teams)
        scoring = _expand_matrix(scoring, graph_teams, desired_teams)
        graph_teams = desired_teams
        team_index = {team: index for index, team in enumerate(graph_teams)}

        if games_seen >= settings.min_games and settings.propagation == "signed_katz":
            ratings = _standardize(
                signed_katz_centrality(
                    signed_matrix,
                    alpha=settings.alpha,
                    max_row_l1=settings.max_row_l1,
                    iterations=settings.iterations,
                    tolerance=settings.tolerance,
                )
            )
            for index, game in weekly_games.iterrows():
                home_index = team_index[str(game["home_team"])]
                away_index = team_index[str(game["away_team"])]
                home_rating = float(ratings[home_index])
                away_rating = float(ratings[away_index])
                result.at[index, columns[0]] = home_rating
                result.at[index, columns[1]] = away_rating
                result.at[index, columns[2]] = home_rating - away_rating
        elif games_seen >= settings.min_games:
            prior = np.full(
                (len(graph_teams), len(graph_teams)), settings.prior_weight, dtype=np.float64
            )
            np.fill_diagonal(prior, 0.0)
            page_rank = _standardize(
                _page_rank(
                    performance + prior,
                    damping=settings.alpha,
                    iterations=settings.iterations,
                    tolerance=settings.tolerance,
                )
            )
            defensive_vulnerability, offensive_strength = _hits(
                scoring + prior, iterations=settings.iterations, tolerance=settings.tolerance
            )
            offense = _standardize(offensive_strength)
            defense = -_standardize(defensive_vulnerability)

            for index, game in weekly_games.iterrows():
                home_index = team_index[str(game["home_team"])]
                away_index = team_index[str(game["away_team"])]
                home_pr = float(page_rank[home_index])
                away_pr = float(page_rank[away_index])
                home_off = float(offense[home_index])
                away_off = float(offense[away_index])
                home_def = float(defense[home_index])
                away_def = float(defense[away_index])
                result.at[index, columns[0]] = home_pr
                result.at[index, columns[1]] = away_pr
                result.at[index, columns[2]] = home_pr - away_pr
                result.at[index, columns[3]] = home_off
                result.at[index, columns[4]] = away_off
                result.at[index, columns[5]] = home_def
                result.at[index, columns[6]] = away_def
                result.at[index, columns[7]] = (home_off - away_def) - (away_off - home_def)

        for _, game in weekly_games.iterrows():
            signal = game["_graph_v2_signal"]
            if pd.isna(signal):
                continue
            signal = float(signal)
            home = str(game["home_team"])
            away = str(game["away_team"])
            home_index = team_index[home]
            away_index = team_index[away]

            discount = 1.0
            if settings.injury_beta > 0.0:
                home_lost = _sum_injury_value_lost(game, HOME_INJURY_VALUE_LOST_COLUMNS)
                away_lost = _sum_injury_value_lost(game, AWAY_INJURY_VALUE_LOST_COLUMNS)
                discount = _injury_discount(home_lost, away_lost, signal, settings.injury_beta)

            discounted = signal * discount
            if discounted != 0.0:
                # signed_katz_centrality iterates x = v + alpha * W @ x (no transpose), so
                # W[recipient, sender] is the entry that adds sender's score into recipient's
                # -- the opposite index order from the nonneg arm below, which feeds a
                # row-stochastic transition into transition.T @ scores instead. A positive
                # `discounted` means home outperformed away, so home is the recipient of a
                # positive endorsement from away, and away is the recipient of a negative one.
                signed_matrix[home_index, away_index] += discounted
                signed_matrix[away_index, home_index] += -discounted

            magnitude = abs(discounted)
            if magnitude > 0.0:
                if signal > 0.0:
                    performance[away_index, home_index] += magnitude
                elif signal < 0.0:
                    performance[home_index, away_index] += magnitude

            home_score = pd.to_numeric(pd.Series([game.get("home_score")]), errors="coerce").iloc[0]
            away_score = pd.to_numeric(pd.Series([game.get("away_score")]), errors="coerce").iloc[0]
            if pd.notna(home_score) and pd.notna(away_score):
                scoring[away_index, home_index] += max(0.0, float(home_score))
                scoring[home_index, away_index] += max(0.0, float(away_score))
            games_seen += 1

    result = result.drop(columns=["_graph_v2_signal"])
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# CFB structural fitting (the XLG "fit where affordable" pattern).
# --------------------------------------------------------------------------


def cfb_structural_coherence(cfb_games: pd.DataFrame, config: GraphRatingV2Config) -> float:
    """Leak-safe walk-forward structural coherence on the CFB corpus.

    Pearson correlation between the pregame rating diff and the game's own
    signal (``ats_margin`` for the residual arm, ``result`` for the control)
    -- a continuous diagnostic, deliberately NOT an ATS accuracy or cover
    probability. Used only to compare candidate structural hyperparameters
    against each other on CFB's larger corpus before freezing one for NFL
    transfer (see :func:`select_structural_config_on_cfb`); never recorded
    to the weak-signals registry by this module.
    """

    if config.edge_signal == "team_stat":
        raise ValueError(
            "cfb_structural_coherence does not accept edge_signal='team_stat': its "
            "correlation would be the rating diff against the very quantity the edges "
            "were built from, i.e. a self-correlation, not a structural diagnostic. A "
            "team_stat arm INHERITS the structural hyperparameters frozen from the "
            "outcome-edge arms (docs/graph_ratings_v2.md section 6); it does not refit "
            "them."
        )
    rated = add_graph_ratings_v2_features(cfb_games, config)
    columns = katz_feature_columns(config)
    diff_column = columns[2]
    signal_column = "ats_margin" if config.edge_signal == "residual" else "result"
    if signal_column not in rated.columns:
        rated = rated.copy()
        rated[signal_column] = _signal_series(rated, config)
    valid = rated[[diff_column, signal_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 30:
        return float("nan")
    correlation = valid[diff_column].corr(valid[signal_column])
    return float(correlation) if pd.notna(correlation) else float("nan")


def select_structural_config_on_cfb(
    cfb_games: pd.DataFrame,
    candidates: Sequence[GraphRatingV2Config],
) -> tuple[GraphRatingV2Config, pd.DataFrame]:
    """Score every candidate structural config on CFB, return the winner.

    "Winner" is the config with the largest |coherence|. Only the
    STRUCTURAL hyperparameters are meant to transfer to NFL (``alpha``,
    ``half_life_weeks``, ``offseason_retention``, ``max_row_l1``,
    ``prior_weight``) -- team-level ratings are never transferred; NFL's own
    graph is always rebuilt fresh from NFL games under the frozen config.
    ``injury_beta`` is out of scope here (CFB has no injury table); callers
    must not vary it across ``candidates``.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    rows = []
    best_config = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        coherence = cfb_structural_coherence(cfb_games, candidate)
        rows.append(
            {
                "edge_signal": candidate.edge_signal,
                "propagation": candidate.propagation,
                "alpha": candidate.alpha,
                "half_life_weeks": candidate.half_life_weeks,
                "offseason_retention": candidate.offseason_retention,
                "max_row_l1": candidate.max_row_l1,
                "prior_weight": candidate.prior_weight,
                "coherence": coherence,
            }
        )
        magnitude = abs(coherence) if math.isfinite(coherence) else -1.0
        if magnitude > best_score:
            best_score = magnitude
            best_config = candidate
    return best_config, pd.DataFrame(rows)

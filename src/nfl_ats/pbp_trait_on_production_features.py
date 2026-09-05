"""PBP coaching-trait leads stacked on PRODUCTION (Wave 4: LEAD-26/27/30).

Lane J (``src/nfl_ats/pbp_coaching_traits.py``, ``docs/pbp_trait_reliability.md``)
measured split-half reliability for four PBP-derived coaching-preparation
traits and left the ATS look to a later lane. All four cleared the (very
low) bar this project uses -- non-zero within-season reliability with
``probability_positive`` 1.0 -- so this module builds three of them (LEAD-26
opening-drive EPA, LEAD-27 third-quarter point differential, and LEAD-30's
predeclared fourth-down-aggression x opener-spread interaction) as single
game-level feature columns, additively joined onto a feature table by
``game_id``, so ``docs/schedule_flag_battery.md`` "Wave 4" can measure each on
top of PRODUCTION ``weak_stack`` instead of a bare baseline.

**Reuses lane J's leak-safe builders verbatim, does not rebuild them.**
``build_opening_drive_rolling`` and ``build_third_quarter_rolling``
(``nfl_ats.pbp_coaching_traits``) already compute each team's trailing state
from STRICTLY EARLIER completed games only (see that module's own leakage
regression tests); LEAD-26/LEAD-27 pivot each team-game rolling row onto the
home/away shape a game-level production feature needs via an exact
``game_id`` join (every game has an opening drive and reaches Q3/Q4, so a
team-game row exists for essentially every game either side played).
LEAD-30 instead reuses lane J's ``build_fourth_down_team_games`` (the
opportunity-conditional team-game table, NOT the ``_rolling`` convenience
wrapper) directly, because its opportunity population is rare by
construction -- most games carry no row for most teams -- so an exact
``game_id`` join would misread "no opportunity in THIS game" as "no rolling
history at all" for the large majority of games. See
:func:`_fourth_down_asof_go_rate` for the as-of merge that carries each
team's cumulative go/eligible totals forward across every game instead.

**Home-minus-away differential (LEAD-26, LEAD-27).** Both candidates are
``home_rolling_value - away_rolling_value``; NaN whenever either side lacks
enough rolling history (lane J's ``PBP_TRAIT_MIN_PER_HALF``-free rolling
builders return NaN for a team's own first game of a season -- there is no
"first game" imputation here, only the model's own training-fold median,
matching ``nfl_ats.redzone_reversion_production_feature``'s documented
philosophy: "no prior information" and "prior information showing a
specific value" are different states, and only the model's own imputer may
decide what to do with the first.

**LEAD-30's interaction, worked through explicitly (frozen before scoring,
see ``docs/schedule_flag_battery.md`` "Wave 4" for the mechanism).** Let
``diff = home_trailing_go_rate - away_trailing_go_rate`` (each side's
cumulative go/eligible totals as of strictly before this game; positive =
home has been the more aggressive fourth-down team) and
``home_spread = tue_open_home_spread`` (this repo's uniform convention,
positive = home favoured, read from
``nfl_ats.schedule_flag_features.default_opener_lines``, i.e. the Tuesday
OPENER, never the nflverse schedule's own closing ``spread_line``).  The
interaction is::

    fourth_down_interaction = diff * (-sign(home_spread))

``-sign(home_spread)`` is ``+1`` when the HOME team is the underdog,
``-1`` when the HOME team is favoured, ``0`` at an exact opener pick'em, and
``NaN`` when the opener store lacks a resolved spread for this game (numpy
propagates the ``NaN`` through the product automatically -- no special
casing is needed, and none is added). Working the four cases:

- Home aggressive (``diff > 0``) AND home is the dog -> ``+1`` ->
  interaction **positive** (aggression on the underdog side).
- Home aggressive AND home is favoured -> ``-1`` -> interaction
  **negative** (aggression on the FAVOURITE side -- the FADE case).
- Away aggressive (``diff < 0``) AND away is the dog (home favoured) ->
  ``-1`` -> interaction **positive** (``diff`` negative times ``-1``).
- Away aggressive AND away is favoured (home is the dog) -> ``+1`` ->
  interaction **negative**.

So positive values of this single column always mean "the more aggressive
team, whichever side it is, is also the underdog" -- exactly the task's
frozen reading of the sign -- and the predeclared direction (BACK
aggressive underdogs, FADE aggressive favourites) is a single monotone
claim about this one column, never two separately-signed sub-claims pooled
together. ``diff`` is built from the same frozen 4th-and-<=3 /
yardline_100 in [30,70] population LEAD-30's reliability gate measured.

Mirrors ``nfl_ats.redzone_reversion_production_feature.attach_redzone_third_down_features``'s
additive-merge discipline: every pre-existing column comes back
bit-identical, only the one new column is added.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot
from nfl_ats.pbp_coaching_traits import (
    build_fourth_down_team_games,
    build_opening_drive_rolling,
    build_third_quarter_rolling,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_ats.schedule_flag_features import (  # noqa: E402
    DEFAULT_MARKET_ROOT,
    default_opener_lines,
)

#: The one new column each candidate profile adds. Frozen names -- matching
#: the fleet task's own candidate names verbatim.
OPENING_DRIVE_EPA_COLUMN = "opening_drive_epa"
Q3_POINT_DIFF_COLUMN = "q3_point_diff"
FOURTH_DOWN_INTERACTION_COLUMN = "fourth_down_interaction"

OPENING_DRIVE_EPA_ON_PRODUCTION_FEATURE_COLUMNS = (OPENING_DRIVE_EPA_COLUMN,)
Q3_POINT_DIFF_ON_PRODUCTION_FEATURE_COLUMNS = (Q3_POINT_DIFF_COLUMN,)
FOURTH_DOWN_INTERACTION_ON_PRODUCTION_FEATURE_COLUMNS = (FOURTH_DOWN_INTERACTION_COLUMN,)

_REQUIRED_FEATURE_COLUMNS = {"game_id", "season", "week", "home_team", "away_team"}


def _alias_team(values: pd.Series) -> pd.Series:
    """Fold OAK/LV, SD/LAC, STL-SL/LA (etc.) onto one continuous franchise code,
    matching ``nfl_ats.pbp_coaching_traits._normalize_teams``'s own convention
    for the ``team`` column of every rolling frame this module pivots."""

    return values.astype("string").replace(TEAM_ABBREVIATION_ALIASES)


def load_pbp_panel(pbp_root: Path | None = None) -> pd.DataFrame:
    """Load the newest local play-by-play snapshot (REG season only).

    Production callers use this; tests pass a synthetic play-by-play frame to
    the ``derive_*`` functions directly instead, so they run in a fresh clone
    with no local data.
    """

    root = pbp_root if pbp_root is not None else REPO_ROOT / "data/pbp/raw"
    snapshot = latest_pbp_snapshot(root)
    return load_pbp_snapshot(snapshot)


def _require_feature_columns(features: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_FEATURE_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")


def _pivot_rolling_home_away(
    features: pd.DataFrame, rolling: pd.DataFrame, value_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """(home_value, away_value) arrays, aligned to ``features``'s own row order.

    ``rolling`` is any of lane J's ``build_*_rolling`` outputs: one row per
    (game_id, team), the ``team`` column already aliased to a continuous
    franchise code. ``features``'s own ``home_team``/``away_team`` are
    aliased the identical way before the join so a relocation-era mismatch
    (e.g. a 2016 Oakland home game recorded as ``OAK`` in the schedule but
    ``LV`` in the rolling frame) cannot silently produce a missing value.
    """

    frame = features.loc[:, ["game_id", "home_team", "away_team"]].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["home_team"] = _alias_team(frame["home_team"])
    frame["away_team"] = _alias_team(frame["away_team"])

    trait = rolling.loc[:, ["game_id", "team", value_col]].copy()
    trait["game_id"] = trait["game_id"].astype(str)
    if trait.duplicated(subset=["game_id", "team"]).any():
        raise DataContractError("rolling trait table carries duplicate (game_id, team) rows")

    home = frame.merge(
        trait.rename(columns={"team": "home_team", value_col: "home_value"}),
        on=["game_id", "home_team"],
        how="left",
        validate="one_to_one",
    )
    away = frame.merge(
        trait.rename(columns={"team": "away_team", value_col: "away_value"}),
        on=["game_id", "away_team"],
        how="left",
        validate="one_to_one",
    )
    return (
        home["home_value"].to_numpy(dtype=float),
        away["away_value"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# LEAD-26: opening-drive EPA differential
# ---------------------------------------------------------------------------


def derive_opening_drive_epa_features(
    features: pd.DataFrame, *, pbp: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return ``(game_id, opening_drive_epa)``: home minus away trailing
    opening-drive EPA per play (:func:`nfl_ats.pbp_coaching_traits.build_opening_drive_rolling`).

    NaN when either side has no rolling history yet (a team's first game of
    the loaded PBP panel) -- never imputed here.
    """

    _require_feature_columns(features)
    if pbp is None:
        pbp = load_pbp_panel()
    rolling = build_opening_drive_rolling(pbp)
    home, away = _pivot_rolling_home_away(features, rolling, "rolling_opening_drive_epa_per_play")
    both_known = np.isfinite(home) & np.isfinite(away)
    value = np.where(both_known, home - away, np.nan)
    return pd.DataFrame(
        {"game_id": features["game_id"].astype(str), OPENING_DRIVE_EPA_COLUMN: value}
    )


def attach_opening_drive_epa_features(
    features: pd.DataFrame, *, pbp: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Additively join ``opening_drive_epa`` onto ``features`` by ``game_id``."""

    return _attach(
        features,
        lambda: derive_opening_drive_epa_features(features, pbp=pbp),
        OPENING_DRIVE_EPA_ON_PRODUCTION_FEATURE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# LEAD-27: third-quarter point-differential differential
# ---------------------------------------------------------------------------


def derive_q3_point_diff_features(
    features: pd.DataFrame, *, pbp: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return ``(game_id, q3_point_diff)``: home minus away trailing
    third-quarter point differential per game
    (:func:`nfl_ats.pbp_coaching_traits.build_third_quarter_rolling`).

    NaN when either side has no rolling history yet -- never imputed here.
    """

    _require_feature_columns(features)
    if pbp is None:
        pbp = load_pbp_panel()
    rolling = build_third_quarter_rolling(pbp)
    home, away = _pivot_rolling_home_away(features, rolling, "rolling_q3_point_diff")
    both_known = np.isfinite(home) & np.isfinite(away)
    value = np.where(both_known, home - away, np.nan)
    return pd.DataFrame({"game_id": features["game_id"].astype(str), Q3_POINT_DIFF_COLUMN: value})


def attach_q3_point_diff_features(
    features: pd.DataFrame, *, pbp: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Additively join ``q3_point_diff`` onto ``features`` by ``game_id``."""

    return _attach(
        features,
        lambda: derive_q3_point_diff_features(features, pbp=pbp),
        Q3_POINT_DIFF_ON_PRODUCTION_FEATURE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# LEAD-30: fourth-down aggressiveness x opener-spread interaction
# ---------------------------------------------------------------------------

#: A team-game row exists in ``build_fourth_down_team_games`` ONLY for a game
#: where that team faced at least one eligible 4th-down opportunity -- lane
#: J's frozen population (4th-and-<=3, yardline_100 in [30, 70]) is rare by
#: construction, so MOST games are absent for MOST teams. Joining a target
#: game onto that table by exact ``game_id`` (the way :func:`_pivot_rolling_home_away`
#: works for LEAD-26/LEAD-27, where a team-game row exists for essentially
#: every game) would therefore read "team had zero opportunities THIS game"
#: as "team has no rolling history at all," discarding real, known trailing
#: state for the vast majority of games. Instead this candidate carries each
#: team's cumulative (go, eligible) totals FORWARD across every game via an
#: as-of merge on a chronological ``order_key = season * 100 + week`` (REG
#: weeks never exceed 99; a genuine POST-season row could in principle share
#: an ``order_key`` with an earlier REG week -- e.g. week 18 REG vs. a later
#: POST week also labelled >=18 in some seasons -- a known, rare, disclosed
#: limitation, not a silent one), taking the LATEST cumulative snapshot
#: STRICTLY BEFORE the target game's own ``order_key`` (``allow_exact_matches
#: =False`` -- if the target game is itself an opportunity game, its own
#: contribution is excluded, preserving leak safety).
_FOURTH_DOWN_REQUIRED_TEAM_GAME_COLUMNS = (
    "game_id",
    "season",
    "week",
    "team",
    "go_count",
    "eligible_count",
)


def _fourth_down_asof_cumulative(team_games: pd.DataFrame) -> pd.DataFrame:
    """(team, order_key, cum_go, cum_eligible), one row per opportunity-game,
    sorted for :func:`pandas.merge_asof`."""

    frame = team_games.loc[:, list(_FOURTH_DOWN_REQUIRED_TEAM_GAME_COLUMNS)].copy()
    frame["team"] = _alias_team(frame["team"])
    frame["order_key"] = pd.to_numeric(frame["season"], errors="raise").astype(
        int
    ) * 100 + pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame = frame.sort_values(["team", "order_key", "game_id"]).reset_index(drop=True)
    grp = frame.groupby("team", sort=False)
    frame["cum_go"] = grp["go_count"].cumsum()
    frame["cum_eligible"] = grp["eligible_count"].cumsum()
    return frame[["team", "order_key", "cum_go", "cum_eligible"]]


def _fourth_down_asof_go_rate(
    features: pd.DataFrame, cumulative: pd.DataFrame, *, team_column: str
) -> np.ndarray:
    """This side's trailing (as of strictly before this game) go rate, aligned
    to ``features``'s own row order."""

    left = features.loc[:, ["game_id", "season", "week", team_column]].copy()
    left["game_id"] = left["game_id"].astype(str)
    left["team"] = _alias_team(left[team_column])
    left["order_key"] = pd.to_numeric(left["season"], errors="raise").astype(
        int
    ) * 100 + pd.to_numeric(left["week"], errors="raise").astype(int)
    left = left.reset_index(drop=False).rename(columns={"index": "_original_position"})
    left_sorted = left.sort_values(["order_key"]).reset_index(drop=True)
    cumulative_sorted = cumulative.sort_values(["order_key"]).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        cumulative_sorted,
        on="order_key",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_original_position").reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        go_rate = merged["cum_go"].to_numpy(dtype=float) / merged["cum_eligible"].to_numpy(
            dtype=float
        )
    return np.asarray(go_rate, dtype=float)


def derive_fourth_down_interaction_features(
    features: pd.DataFrame,
    *,
    pbp: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Return ``(game_id, fourth_down_interaction)``.

    ``diff = home minus away trailing fourth-down go rate`` (each side's
    cumulative go/eligible totals as of strictly before this game -- see
    :func:`_fourth_down_asof_go_rate` -- built from
    :func:`nfl_ats.pbp_coaching_traits.build_fourth_down_team_games`), times
    ``-sign(tue_open_home_spread)`` (the Tuesday OPENER consensus, from
    :func:`nfl_ats.schedule_flag_features.default_opener_lines` -- never the
    nflverse schedule's own closing ``spread_line``). See the module
    docstring for the worked sign-convention cases. NaN when either side has
    faced zero eligible opportunities in ANY strictly-prior game, OR the
    opener store lacks a resolved spread for this game (numpy's own
    NaN-propagating arithmetic handles both automatically); ``0.0`` at an
    exact opener pick'em with both trailing go rates known (no favourite
    exists to call an "underdog," a genuine state, not missing information).
    """

    _require_feature_columns(features)
    if pbp is None:
        pbp = load_pbp_panel()
    team_games = build_fourth_down_team_games(pbp)
    cumulative = _fourth_down_asof_cumulative(team_games)
    home_go = _fourth_down_asof_go_rate(features, cumulative, team_column="home_team")
    away_go = _fourth_down_asof_go_rate(features, cumulative, team_column="away_team")
    both_known = np.isfinite(home_go) & np.isfinite(away_go)
    diff = np.where(both_known, home_go - away_go, np.nan)

    if opener_lines is None:
        opener_lines = default_opener_lines(features, market_root=market_root)
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")
    lines = (
        features[["game_id"]]
        .astype({"game_id": str})
        .merge(
            opener_lines[["game_id", "tue_open_home_spread"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    )
    home_spread = pd.to_numeric(lines["tue_open_home_spread"], errors="coerce").to_numpy(
        dtype=float
    )
    dog_sign = -np.sign(home_spread)
    value = diff * dog_sign
    return pd.DataFrame(
        {"game_id": features["game_id"].astype(str), FOURTH_DOWN_INTERACTION_COLUMN: value}
    )


def attach_fourth_down_interaction_features(
    features: pd.DataFrame,
    *,
    pbp: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``fourth_down_interaction`` onto ``features``.

    ``opener_lines`` may be supplied directly (fixtures, tests) to avoid
    touching the real market store; otherwise it is loaded via
    :func:`nfl_ats.schedule_flag_features.default_opener_lines` from
    ``features`` itself (which already carries ``game_id``/``season``/``week``).
    """

    return _attach(
        features,
        lambda: derive_fourth_down_interaction_features(
            features, pbp=pbp, opener_lines=opener_lines, market_root=market_root
        ),
        FOURTH_DOWN_INTERACTION_ON_PRODUCTION_FEATURE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Shared additive-merge helper
# ---------------------------------------------------------------------------


def _attach(
    features: pd.DataFrame,
    derive: Callable[[], pd.DataFrame],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(columns).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive()
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_pbp_trait"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_pbp_trait") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "DEFAULT_MARKET_ROOT",
    "FOURTH_DOWN_INTERACTION_COLUMN",
    "FOURTH_DOWN_INTERACTION_ON_PRODUCTION_FEATURE_COLUMNS",
    "OPENING_DRIVE_EPA_COLUMN",
    "OPENING_DRIVE_EPA_ON_PRODUCTION_FEATURE_COLUMNS",
    "Q3_POINT_DIFF_COLUMN",
    "Q3_POINT_DIFF_ON_PRODUCTION_FEATURE_COLUMNS",
    "attach_fourth_down_interaction_features",
    "attach_opening_drive_epa_features",
    "attach_q3_point_diff_features",
    "derive_fourth_down_interaction_features",
    "derive_opening_drive_epa_features",
    "derive_q3_point_diff_features",
    "load_pbp_panel",
]

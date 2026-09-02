"""Injury-report illness-designation indicators, stacked on PRODUCTION.

``docs/illness_battery.md`` predeclared and froze five cells against a BARE
market baseline. Two of them lead: ``illness_home_ge2`` (+0.297 accuracy
points, week-blocked 95% [-0.176, +0.784], ``probability_positive`` 0.890) and
``illness_away_active_ge1`` (+0.307, [-0.659, +1.280], P+ 0.733), both at
split-half reliability **0.702** -- the highest of any construct in the
2026-09-01 on-production sweep that is not an attention-volume series. The
project's own recorded lesson ("composition is not the signal") is that a
component positive alone can go negative once stacked on the chain that is
actually PLAYED. This module builds those same two indicators, at the SAME
frozen as-of construction, additively joined onto a feature table by
``game_id``, so ``docs/illness_on_production.md`` can measure them on top of
PRODUCTION ``weak_stack`` instead of a bare baseline.

**Distinct from the FluView columns already tested this way**
(``nfl_ats.fluview_production_feature``): FluView measures CDC *regional*
influenza-like-illness activity in a team's market. These columns measure the
*club's own* injury-report illness designations, reconciled at 97.13% against
the independent NFL.com scrape (``scripts/nflverse_injuries_reconcile.py``,
per the registry note on ``illness_home_ge2``).

**Reuses the frozen point-in-time-safe construction verbatim, does not rebuild
it**: ``attach_cutoffs``, ``load_injuries``, ``build_team_week_cutoffs``,
``resolve_asof_team_week`` and ``attach_team_week_features`` are imported
directly from ``scripts/illness_battery_screen.py``, which in turn imports
``nfl_ats.pick_refresh.pick_deadline`` / ``sunday_pick_lock`` -- the project's
own binding per-game pick deadline, ``min(that game's own kickoff, that week's
Sunday 16:00 ET)``. Per ``(season, week, team, gsis_id)`` entity only report
revisions with ``date_modified <= cutoff`` are visible, and the as-of state is
the latest surviving revision; a team-week with zero visible rows resolves to
MISSING, never a zero count. ``scripts`` is not part of the installed package,
so this module puts the repository root on ``sys.path`` the same guarded way
``nfl_ats.fluview_production_feature`` already does for the same reason.

Mirrors ``nfl_ats.forecast_weather_features.attach_forecast_weather_features``'s
additive-merge discipline: every pre-existing column comes back bit-identical,
only the two new columns are added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.illness_battery_screen import (  # noqa: E402
    attach_cutoffs,
    build_team_week_cutoffs,
    default_injuries,
    load_injuries,
    resolve_asof_team_week,
)

#: The two new columns this module adds. Frozen names, matching the already
#: -recorded weak-signal registry cell names 1:1 so the lineage between the
#: bare-baseline screen and this on-production stacking is legible from the
#: column name alone.
ILLNESS_AWAY_ACTIVE_GE1_COLUMN = "illness_away_active_ge1"
ILLNESS_HOME_GE2_COLUMN = "illness_home_ge2"
ILLNESS_ON_PRODUCTION_FEATURE_COLUMNS = (
    ILLNESS_AWAY_ACTIVE_GE1_COLUMN,
    ILLNESS_HOME_GE2_COLUMN,
)

#: Inherited unchanged from ``scripts/illness_battery_screen.py``.
ILLNESS_COUNT_THRESHOLD = 2

_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
}


def derive_illness_features(
    features: pd.DataFrame,
    *,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a ``(game_id, illness_away_active_ge1, illness_home_ge2)`` frame,
    point-in-time-safe.

    ``injuries`` defaults to the already-ingested nflverse snapshot
    (``scripts.illness_battery_screen.default_injuries``); it is accepted as a
    parameter purely for testability -- production callers leave it unset.

    A team-week the as-of resolution cannot see (no report revision on or
    before that game's own pick deadline) comes back **NaN**, not ``0``. The
    frozen battery folded that case into ``False`` because it scored a subset
    cover-rate gap over a population it had already restricted; a feature
    column may not, because "no visible report" and "a visible report showing
    nobody ill" are different states and only the model's own training-fold
    median may decide what to do with the first.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["home_team"] = frame["home_team"].astype("string")
    frame["away_team"] = frame["away_team"].astype("string")

    frame = attach_cutoffs(frame)
    team_week_cutoffs = build_team_week_cutoffs(frame)

    if injuries is None:
        injuries = load_injuries(default_injuries())
    as_of = resolve_asof_team_week(injuries, team_week_cutoffs)

    home = as_of.rename(
        columns={
            "team": "home_team",
            "illness_count": "home_illness_count",
            "active_illness_count": "home_active_illness_count",
        }
    )
    away = as_of.rename(
        columns={
            "team": "away_team",
            "illness_count": "away_illness_count",
            "active_illness_count": "away_active_illness_count",
        }
    )
    home["home_team"] = home["home_team"].astype("string")
    away["away_team"] = away["away_team"].astype("string")
    frame = frame.merge(home, on=["season", "week", "home_team"], how="left")
    frame = frame.merge(away, on=["season", "week", "away_team"], how="left")

    home_visible = frame["home_illness_count"].notna().to_numpy()
    away_visible = frame["away_active_illness_count"].notna().to_numpy()
    home_ge2 = (
        pd.to_numeric(frame["home_illness_count"], errors="coerce").to_numpy()
        >= ILLNESS_COUNT_THRESHOLD
    )
    away_active_ge1 = (
        pd.to_numeric(frame["away_active_illness_count"], errors="coerce").to_numpy() >= 1
    )

    return pd.DataFrame(
        {
            "game_id": frame["game_id"],
            ILLNESS_AWAY_ACTIVE_GE1_COLUMN: np.where(
                away_visible, away_active_ge1.astype(float), np.nan
            ),
            ILLNESS_HOME_GE2_COLUMN: np.where(home_visible, home_ge2.astype(float), np.nan),
        }
    )


def attach_illness_features(
    features: pd.DataFrame,
    *,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Additively join the two new columns onto ``features``.

    Every pre-existing column is returned bit-identical; only
    ``illness_away_active_ge1`` / ``illness_home_ge2`` are added. Games whose
    team-week the as-of resolution cannot see come back NaN, left NaN on
    purpose: imputation belongs to the model's own training-fold median
    (``fit_margin_model``), not to a feature builder that can see every season
    at once.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(ILLNESS_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive_illness_features(features, injuries=injuries)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_illness"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_illness") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "ILLNESS_AWAY_ACTIVE_GE1_COLUMN",
    "ILLNESS_COUNT_THRESHOLD",
    "ILLNESS_HOME_GE2_COLUMN",
    "ILLNESS_ON_PRODUCTION_FEATURE_COLUMNS",
    "attach_illness_features",
    "derive_illness_features",
]

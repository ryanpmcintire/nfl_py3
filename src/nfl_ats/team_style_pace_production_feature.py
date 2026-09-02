"""Team-style pace-mismatch indicator, stacked on PRODUCTION.

``docs/team_style.md`` (PBP-08) predeclared and froze five team-style cells and
scored them against a BARE market baseline. The pace cell leads the battery on
the field AGENTS.md makes decisive: ``team_style_pace_mismatch_dog_cover``
carries split-half/YoY reliability **0.489**, the highest of the five (+0.229
accuracy points, week-blocked 95% [-0.559, +1.040], ``probability_positive``
0.711, n 4313, n_flag 1018). The project's own recorded lesson ("composition is
not the signal") is that a component positive alone can go negative once
stacked on the chain that is actually PLAYED. This module builds that same
indicator as a game-level feature column, additively joined onto a feature
table by ``game_id``, so ``docs/team_style_pace_on_production.md`` can measure
it on top of PRODUCTION ``weak_stack`` instead of a bare baseline.

**Why a pace column and not another quality feature.** The measured
team-quality ceiling (ROADMAP.md PBP-05, ``.claude`` memory
``team-quality-is-already-priced``) bounds features that merely measure team
quality better near zero. ``scripts/team_style_features.py``'s style dimensions
are deliberately quality-ORTHOGONAL, and this one is an **absolute gap**,
``|home - away|`` -- a shape a linear ridge cannot form from its inputs even if
both sides were present, matching the symmetric variance mechanism
``docs/team_style.md`` predeclares (a pace mismatch compresses total offensive
possessions; fewer possessions favour the dog).

**Source, reused not rebuilt.** ``data/pbp/team_style/team_season_style.parquet``
(544 rows, seasons 2009-2025), the exact artifact
``scripts/team_style_features.py`` writes and ``scripts/team_style_screen.py``
reads. It is read as a parquet rather than by importing the builder: no new
structural choice is made, and the ``scripts`` package does not join
``mypy src``'s import graph. Each game's home and away team is joined to that
team's PRIOR season row (the same one-season forward shift
``scripts/team_style_screen.py::_prior`` performs), with
``nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`` applied to both sides.

**The one deliberate, declared deviation** (``docs/team_style_pace_on_production.md``
section 2.1): the screen computes its top-quartile cut over the WHOLE 2009-2025
panel, a mild look-ahead a pregame feature column may not carry. Here the cut
is recomputed EXPANDING over strictly prior seasons only -- for a game in
season S, the 75th percentile of the absolute pace gap across every game in the
panel whose season is strictly less than S -- NaN where no prior season carries
a defined gap. That makes this column a slightly different quantity from the
registered cell; a noisier cut misclassifies near-threshold games in both
directions and so can only attenuate the fitted coefficient toward the null,
never manufacture an effect. ``tests/test_team_style_pace_production_feature.py``
pins it in both directions.

Mirrors ``nfl_ats.illness_production_feature`` and
``nfl_ats.forecast_weather_features.attach_forecast_weather_features``'s
additive-merge discipline: every pre-existing column comes back bit-identical,
only the one new column is added.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    TEAM_ABBREVIATION_ALIASES,
    TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError

#: The one new column this module adds. Frozen name.
TEAM_STYLE_PACE_MISMATCH_COLUMN = TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS[0]

#: Inherited unchanged from ``scripts/team_style_screen.py`` (``QUARTILE``).
#: Deliberately NOT re-tuned here: re-cutting a threshold after seeing an
#: outcome number is exactly what the predeclaration discipline prevents.
PACE_QUARTILE = 0.75

#: The team-season style panel ``scripts/team_style_features.py`` writes.
#: Gitignored (``data/pbp/**``), so callers in a fresh clone must build it
#: first; tests inject their own in-memory panel instead.
REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_SEASON_STYLE_PATH = REPO_ROOT / "data/pbp/team_style/team_season_style.parquet"

PACE_CENTERED_COLUMN = "seconds_per_play_pace_centered"

_REQUIRED_GAME_COLUMNS = {"game_id", "season", "home_team", "away_team"}
_REQUIRED_STYLE_COLUMNS = {"season", "team", PACE_CENTERED_COLUMN}


def default_team_season_style() -> pd.DataFrame:
    """Load the team-season style panel from its standard cached location."""

    if not TEAM_SEASON_STYLE_PATH.is_file():
        raise DataContractError(
            f"missing {TEAM_SEASON_STYLE_PATH}; build it with "
            "`python scripts/team_style_features.py` first"
        )
    return pd.read_parquet(TEAM_SEASON_STYLE_PATH)


def _prior_season_pace(team_season: pd.DataFrame, side: str) -> pd.DataFrame:
    """Shift the (season, team) panel forward one season and rename ``team``
    to ``<side>_team``, so joining on (``<side>_team``, ``season``) pulls the
    PRIOR season's centred pace onto that season's games.

    Same construction as ``scripts/team_style_screen.py::_prior``.
    """

    shifted = team_season.loc[:, ["season", "team", PACE_CENTERED_COLUMN]].copy()
    shifted["season"] = pd.to_numeric(shifted["season"], errors="raise").astype(int) + 1
    shifted["team"] = shifted["team"].astype("string").replace(TEAM_ABBREVIATION_ALIASES)
    shifted[PACE_CENTERED_COLUMN] = pd.to_numeric(shifted[PACE_CENTERED_COLUMN], errors="coerce")
    return shifted.rename(
        columns={"team": f"{side}_team", PACE_CENTERED_COLUMN: f"{side}_prior_pace_centered"}
    )


def _expanding_prior_season_thresholds(
    seasons: pd.Series, gaps: pd.Series, *, quartile: float
) -> pd.Series:
    """Per-game top-quartile threshold estimated from STRICTLY PRIOR seasons.

    For a game in season S the cut is the ``quartile`` quantile of ``gaps``
    over every row whose season is strictly less than S. Seasons with no prior
    row carrying a defined gap get NaN, which propagates to a NaN flag.

    This is the declared deviation from ``scripts/team_style_screen.py``, whose
    single cut is estimated over the whole 2009-2025 panel and is therefore a
    mild look-ahead no pregame feature column may carry.
    """

    season_values = pd.to_numeric(seasons, errors="raise").astype(int)
    gap_values = pd.to_numeric(gaps, errors="coerce")
    thresholds: dict[int, float] = {}
    for season in sorted(season_values.unique()):
        prior = gap_values.loc[season_values.lt(season).to_numpy()].dropna()
        thresholds[int(season)] = float(prior.quantile(quartile)) if len(prior) else float("nan")
    return season_values.map(thresholds).astype(float)


def derive_team_style_pace_features(
    features: pd.DataFrame,
    *,
    team_season: pd.DataFrame | None = None,
    quartile: float = PACE_QUARTILE,
) -> pd.DataFrame:
    """Return a ``(game_id, team_style_pace_mismatch_flag)`` frame, pregame-safe.

    ``team_season`` defaults to the cached team-season style panel
    (``default_team_season_style``); it is accepted as a parameter purely for
    testability -- production callers leave it unset.

    A game whose home or away team has no prior-season centred pace, or whose
    season has no strictly-prior season carrying a defined gap, comes back
    **NaN**, not ``0``. The frozen battery folded both cases into ``False``
    because it scored a subset cover-rate gap over a population it had already
    restricted; a feature column may not, because "no prior-season pace" and
    "a measured small gap" are different states and only the model's own
    training-fold median may decide what to do with the first.
    """

    missing = sorted(_REQUIRED_GAME_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    if team_season is None:
        team_season = default_team_season_style()
    style_missing = sorted(_REQUIRED_STYLE_COLUMNS.difference(team_season.columns))
    if style_missing:
        raise DataContractError(
            f"team-season style panel is missing columns: {', '.join(style_missing)}"
        )

    frame = features.loc[:, sorted(_REQUIRED_GAME_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    for side in ("home", "away"):
        column = f"{side}_team"
        frame[column] = frame[column].astype("string").replace(TEAM_ABBREVIATION_ALIASES)

    for side in ("home", "away"):
        frame = frame.merge(
            _prior_season_pace(team_season, side),
            on=[f"{side}_team", "season"],
            how="left",
            validate="many_to_one",
        )

    gap = (frame["home_prior_pace_centered"] - frame["away_prior_pace_centered"]).abs()
    threshold = _expanding_prior_season_thresholds(frame["season"], gap, quartile=quartile)

    usable = gap.notna().to_numpy() & threshold.notna().to_numpy()
    flag = np.where(usable, (gap.to_numpy() >= threshold.to_numpy()).astype(float), np.nan)

    return pd.DataFrame({"game_id": frame["game_id"], TEAM_STYLE_PACE_MISMATCH_COLUMN: flag})


def attach_team_style_pace_features(
    features: pd.DataFrame,
    *,
    team_season: pd.DataFrame | None = None,
    quartile: float = PACE_QUARTILE,
) -> pd.DataFrame:
    """Additively join ``team_style_pace_mismatch_flag`` onto ``features``.

    Every pre-existing column is returned bit-identical; only the one new
    column is added. Games with no prior-season pace on either side, or in a
    season with no prior-season threshold, come back NaN and are left NaN on
    purpose: imputation belongs to the model's own training-fold median
    (``fit_margin_model``), not to a feature builder.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(
        set(TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns)
    )
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive_team_style_pace_features(features, team_season=team_season, quartile=quartile)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_team_style_pace"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_team_style_pace") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "PACE_CENTERED_COLUMN",
    "PACE_QUARTILE",
    "TEAM_SEASON_STYLE_PATH",
    "TEAM_STYLE_PACE_MISMATCH_COLUMN",
    "TEAM_STYLE_PACE_MISMATCH_ON_PRODUCTION_FEATURE_COLUMNS",
    "attach_team_style_pace_features",
    "default_team_season_style",
    "derive_team_style_pace_features",
]

"""Subreddit fan-attention indicators, stacked on PRODUCTION.

``docs/arctic_shift_ats_battery.md`` predeclared and froze five cells against a
BARE market baseline. Two of them lead: ``reddit_home_comment_ratio_elevated``
(+0.329 accuracy points, week-blocked 95% [-0.207, +0.883],
``probability_positive`` 0.885, season-blocked secondary [+0.078, +0.603] at
P+ 0.996) and ``reddit_away_spike_value`` (+0.214, [-0.224, +0.644], P+ 0.832).
Three of the five lean positive. The project's own recorded lesson
("composition is not the signal") is that a component positive alone can go
negative once stacked on the chain that is actually PLAYED. This module builds
those same two indicators, at the SAME frozen construction, additively joined
onto a feature table by ``game_id``, so
``docs/reddit_attention_on_production.md`` can measure them on top of
PRODUCTION ``weak_stack`` instead of a bare baseline.

**The attention channel has never been stacked on production in any form** --
neither Reddit, nor GDELT, nor Wikipedia pageviews (``docs/
on_production_sweep_20260901.md`` section 1.1: the four earlier on-production
tests are graph-propagated team statistics and CDC regional influenza-like
illness). This is the channel's first contact with the played model.

**Reuses the frozen point-in-time-safe construction verbatim, does not rebuild
it**: ``load_subreddit_daily_counts``, ``build_team_game_long``,
``attach_game_level`` and ``SPIKE_THRESHOLD`` are imported directly from
``scripts/arctic_shift_battery_screen.py``, which in turn imports
``TRAILING_MIN_GAMES`` / ``TRAILING_WINDOW_GAMES`` from
``scripts/attention_battery_screen.py`` and ``SUBREDDITS_ALL`` from
``scripts/arctic_shift_battery_fetch.py``. The window ends on the **Tuesday of
the game's own week** and the trailing baseline is ``shift(1)``-ed before the
rolling window and reset per ``(team, season)``, so neither the window nor its
normalizer can see a day at or after kickoff. ``scripts`` is not part of the
installed package, so this module puts the repository root on ``sys.path`` the
same guarded way ``nfl_ats.illness_production_feature`` and
``nfl_ats.fluview_production_feature`` already do for the same reason.

Two deliberate deviations from the frozen battery, both declared in
``docs/reddit_attention_on_production.md`` section 2 before any outcome number
existed:

1. A row without a computable trailing baseline comes back **NaN**, never
   ``False``. "No subreddit coverage" and "coverage showing no spike" are
   different states, and only ``fit_margin_model``'s training-fold median may
   decide what to do with the first.
2. The baseline sequence is **not** filtered on the outcome. The battery's own
   ``load_games`` additionally dropped pushes and unplayed games (``home_cover``
   non-null); a pregame feature column may not replicate an outcome-dependent
   restriction, so the sequence here is every REG row with a ``spread_line``.

Mirrors ``nfl_ats.forecast_weather_features.attach_forecast_weather_features``'s
additive-merge discipline: every pre-existing column comes back bit-identical,
only the two new columns are added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.arctic_shift_battery_screen import (  # noqa: E402
    SPIKE_THRESHOLD,
    attach_game_level,
    build_team_game_long,
    load_subreddit_daily_counts,
)

#: The two new columns this module adds. Frozen names, matching the already
#: -recorded weak-signal registry cell names 1:1 so the lineage between the
#: bare-baseline screen and this on-production stacking is legible from the
#: column name alone.
REDDIT_HOME_RATIO_ELEVATED_COLUMN = "reddit_home_comment_ratio_elevated"
REDDIT_AWAY_SPIKE_COLUMN = "reddit_away_spike_value"
REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS = (
    REDDIT_HOME_RATIO_ELEVATED_COLUMN,
    REDDIT_AWAY_SPIKE_COLUMN,
)

#: Default location of the Arctic Shift raw daily-count fetch
#: (``scripts/arctic_shift_battery_fetch.py``; gitignored, per repo convention).
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "arctic_shift"

TeamDaily = dict[str, dict[str, pd.Series]]

_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "spread_line",
}


def _canonical(team: pd.Series) -> pd.Series:
    """Team-code canonicalization, identical to the frozen screen's own."""

    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def derive_reddit_attention_features(
    features: pd.DataFrame,
    *,
    team_daily: TeamDaily | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a ``(game_id, reddit_home_comment_ratio_elevated,
    reddit_away_spike_value)`` frame, point-in-time-safe.

    ``team_daily`` defaults to the already-fetched Arctic Shift daily-count
    snapshot (``scripts.arctic_shift_battery_screen.load_subreddit_daily_counts``
    over ``raw_dir``); it is accepted as a parameter purely for testability --
    production callers leave it unset.

    A row whose team-week has no computable trailing baseline -- the first two
    games of a team's season, a franchise whose subreddit did not yet exist, a
    playoff row -- comes back **NaN**, not ``0``. See the module docstring and
    ``docs/reddit_attention_on_production.md`` section 2 for why a feature
    column may not fold that into ``False``.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["home_team"] = _canonical(frame["home_team"].astype(str))
    frame["away_team"] = _canonical(frame["away_team"].astype(str))
    frame["spread_line"] = pd.to_numeric(frame["spread_line"], errors="coerce")

    # The baseline sequence: REG rows with a market line, in each team's own
    # chronological order. Deliberately NOT filtered on the outcome (deviation
    # 2 in the predeclaration) -- a pregame column may not inherit the frozen
    # battery's push/unplayed-game drop.
    eligible = frame.loc[
        frame["game_type"].astype(str).eq("REG") & frame["spread_line"].notna()
    ].copy()
    eligible = eligible.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    empty = pd.DataFrame(
        {
            "game_id": frame["game_id"],
            REDDIT_HOME_RATIO_ELEVATED_COLUMN: np.nan,
            REDDIT_AWAY_SPIKE_COLUMN: np.nan,
        }
    ).reset_index(drop=True)
    if eligible.empty:
        return empty

    if team_daily is None:
        team_daily = load_subreddit_daily_counts(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    if not team_daily:
        return empty

    long_df = build_team_game_long(eligible, team_daily)
    game_level = attach_game_level(eligible, long_df)

    home_ratio_z = pd.to_numeric(game_level["home_ratio_z"], errors="coerce").to_numpy()
    away_volume_z = pd.to_numeric(game_level["away_volume_z"], errors="coerce").to_numpy()
    home_has_baseline = game_level["home_has_baseline_ratio"].fillna(False).to_numpy(dtype=bool)
    away_has_baseline = game_level["away_has_baseline_volume"].fillna(False).to_numpy(dtype=bool)

    with np.errstate(invalid="ignore"):
        home_elevated = home_ratio_z >= SPIKE_THRESHOLD
        away_spike = away_volume_z >= SPIKE_THRESHOLD

    derived = pd.DataFrame(
        {
            "game_id": game_level["game_id"].astype(str),
            REDDIT_HOME_RATIO_ELEVATED_COLUMN: np.where(
                home_has_baseline, home_elevated.astype(float), np.nan
            ),
            REDDIT_AWAY_SPIKE_COLUMN: np.where(away_has_baseline, away_spike.astype(float), np.nan),
        }
    )
    return (
        empty.drop(columns=list(REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS))
        .merge(derived, on="game_id", how="left", validate="one_to_one")
        .reset_index(drop=True)
    )


def attach_reddit_attention_features(
    features: pd.DataFrame,
    *,
    team_daily: TeamDaily | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Additively join the two new columns onto ``features``.

    Every pre-existing column is returned bit-identical; only
    ``reddit_home_comment_ratio_elevated`` / ``reddit_away_spike_value`` are
    added. Rows with no computable trailing baseline come back NaN, left NaN on
    purpose: imputation belongs to the model's own training-fold median
    (``fit_margin_model``), not to a feature builder that can see every season
    at once.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(
        set(REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS).intersection(features.columns)
    )
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived = derive_reddit_attention_features(features, team_daily=team_daily, raw_dir=raw_dir)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_reddit"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_reddit") if c in merged.columns])
    merged.index = features.index
    return merged


__all__ = [
    "REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS",
    "REDDIT_AWAY_SPIKE_COLUMN",
    "REDDIT_HOME_RATIO_ELEVATED_COLUMN",
    "SPIKE_THRESHOLD",
    "attach_reddit_attention_features",
    "derive_reddit_attention_features",
]

"""Two Phase 12 weather/venue pregame-adjacent flags, each stacked on
PRODUCTION (``docs/weather_venue_leads.md`` predeclares both before either
was scored): ROADMAP LEAD-36 (open-corner stadium wind), LEAD-37
(rain-on-grass fumble chaos). LEAD-38 (snow-game home preparation) is NOT
built here -- see ``docs/weather_venue_leads.md``'s source-gap note.

LEAD-36: open-corner stadium wind
------------------------------------
**Disclosure, stated up front and never elided.** This flag's ``wind``/
``roof`` inputs are the newest ``data/raw/*/schedules.parquet`` snapshot's
own OBSERVED, game-time-actual columns -- the same columns
``nfl_ats.forecast_weather_features.derive_observed_weather_features`` calls
"POSITIVE CONTROL ONLY... deliberately leaky and must never reach
production" (its ``actual_temp_f``/``actual_wind_mph`` are this exact
``wind``/``temp`` pair, one join upstream). This is NOT a pregame forecast.
It is, however, the SAME data convention already used by every "venue-blind"
wind cell already recorded in ``registry/weak_signals.json``
(``weather_battery_high_wind_outdoor``, ``weather_battery_high_wind_road_favorite``,
both from ``scripts/nfl_weather_battery_screen.py``, each disclosed there
verbatim as "an actual-weather mechanism screen (game-time actuals, not
pregame-available)... upper bound for a forecast-time feature"). This family
runs the identical mechanism-screen methodology those cells already use,
adding only the open-corner VENUE interaction they never tested (they are
venue-blind by construction -- ROADMAP's own framing of what's new here). It
answers a mechanism question about MAGNITUDE, never a claim of a deployable
pregame feature: this column is never wired into ``MODEL_FEATURE_COLUMNS``
and never promoted to a live weekly card.

Frozen open-corner venue list (JUDGEMENT CALL, disclosed):
:data:`OPEN_CORNER_STADIUMS` is exactly the task's own given anchor list --
BUF, CHI, NE, CLE, GB, PIT, KC, DEN, NYJ/NYG (MetLife only), PHI. No further
venue was added: a repo-wide grep for "open corner"/"open end" stadium
geometry found only one general essay
(``docs/new_lead_classes_20260826.md`` section 6, sun-glare geometry -- an
architectural-orientation mechanism, not a wind venue catalogue) and no other
locally documented open-corner list to extend from.

Gated on this game's own ``stadium_id`` (that team's own PRIMARY, most
frequent code), never on ``home_team`` alone: measured against
``data/raw/20260824T115346Z/schedules.parquet``, five of the eleven teams
also host a handful of international/blizzard-relocation one-off "home"
games at a DIFFERENT stadium (e.g. BUF at ``DET00``/``LON02``, NE at
``FRA00``/``IND00``/``MIN01``/``SFO01``, KC at
``LON00``/``MIA00``/``FRA00``/``VEG00``, GB at ``DAL00``/``LON02``, CHI at
``LON02``, CLE at ``LON01``, DEN at ``NYC01``/``SFO01``, PHI at
``PHO00``/``SAO00``/``NOR00``) -- none of those is the open-corner design
this lead is about. NYJ/NYG are restricted to ``stadium_id`` ``NYC01``
(MetLife, 2010-present) specifically, per the task's own "MetLife"
qualifier: their earlier shared Giants Stadium (``NYC00``, pre-2010, 8 games
each) is excluded, mirroring
``nfl_ats.schedule_flag_features.NEW_STADIUM_HONEYMOON_SEASONS``'s own use of
``NYC01`` for the identical venue.

Signed ``open_corner_wind_dog_flag``: ``+1`` when the HOME team is the
underdog at the Tuesday opener AND this game qualifies (a frozen open-corner
venue, this game's own recorded roof is outdoors/open, AND this game's own
recorded wind is >= 15 mph); ``-1`` when the AWAY team is the opener
underdog AND the game qualifies; ``0`` otherwise (including a non-qualifying
game, an exact opener pick'em, or a missing opener spread).

LEAD-37: rain-on-grass fumble chaos
-------------------------------------
No observed historical precipitation/snow column exists locally -- measured:
none of ``home_team``/``away_team``/``game_id``/``season``/``gameday``/
``weekday``/``gametime``/``away_rest``/``home_rest``/``div_game``/
``roof``/``surface``/``temp``/``wind``/``spread_line``/``total_line`` (the
full column list of the newest ``schedules.parquet`` snapshot) names
precipitation, rain, or snow in any form -- see ``docs/weather_venue_leads.md``
for the full disclosure. ``forecast_precip_prob_pct``
(``nfl_ats.forecast_weather_features``, the validated ``pool_decision``
forecast archive) is used instead as a DISCLOSED, genuinely pregame-safe
PROXY: built at the pool's real decision timestamp
(``min(kickoff, Sunday 16:00 America/New_York)``), it measures a FORECAST
probability of precipitation, not whether rain or snow actually fell. Its
manifest (``data/raw/forecast_archive/pool_decision_2009_2025/manifest.json``)
declares ``start_season: 2009``, ``end_season: 2025``, 4,431 rows -- the
archive's full manifested window, well beyond the "3 scored seasons"
threshold the task set for using it as a screened proxy rather than a
source-gap note.

Signed ``rain_on_grass_dog_flag``: ``+1`` when the HOME team is the
underdog at the Tuesday opener AND this game qualifies (this game's own
recorded surface normalizes to grass, per
``nfl_ats.surface_switch_tilt_overlay.GRASS_SURFACES``, AND this game's own
``forecast_precip_prob_pct`` >= 60); ``-1`` when the AWAY team is the opener
underdog AND the game qualifies; ``0`` otherwise.

Both mirror ``nfl_ats.schedule_flag_features``'s additive-merge discipline:
every pre-existing column returns bit-identical, only the one new column is
added. Both duplicate (rather than import) their small schedule/opener-line
loader helpers from ``nfl_ats.schedule_flag_features``, matching
``nfl_ats.qb_identity_features``'s own stated rationale: several fleet lanes
concurrently edit that module this session, and this module needs none of
its other candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, build_pairing_table
from nfl_ats.constants import (
    OPEN_CORNER_WIND_DOG_ON_PRODUCTION_FEATURE_COLUMNS,
    RAIN_ON_GRASS_DOG_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.forecast_weather_features import DEFAULT_FORECAST_ARCHIVE, load_forecast_archive
from nfl_ats.surface_switch_tilt_overlay import GRASS_SURFACES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one new column each candidate profile adds. Frozen names.
OPEN_CORNER_WIND_DOG_COLUMN = OPEN_CORNER_WIND_DOG_ON_PRODUCTION_FEATURE_COLUMNS[0]
RAIN_ON_GRASS_DOG_COLUMN = RAIN_ON_GRASS_DOG_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"

OPEN_CORNER_WIND_MPH_MIN = 15.0
OPEN_CORNER_OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
RAIN_ON_GRASS_PRECIP_PROB_MIN = 60.0

#: (stadium_id -> team code(s)), frozen 2026-09-05 against
#: ``data/raw/20260824T115346Z/schedules.parquet`` -- see the module
#: docstring for the disclosed judgement call and the exclusion rationale.
OPEN_CORNER_STADIUMS: dict[str, str] = {
    "BUF00": "BUF",
    "CHI98": "CHI",
    "BOS00": "NE",
    "CLE00": "CLE",
    "GNB00": "GB",
    "PIT00": "PIT",
    "KAN00": "KC",
    "DEN00": "DEN",
    "NYC01": "NYJ/NYG",
    "PHI00": "PHI",
}

_OPEN_CORNER_WIND_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "home_team",
    "away_team",
    "stadium_id",
    "roof",
    "wind",
}
_RAIN_ON_GRASS_REQUIRED_SCHEDULE_COLUMNS = {"game_id", "home_team", "away_team", "surface"}


# ---------------------------------------------------------------------------
# Shared loaders (duplicated from nfl_ats.schedule_flag_features -- see the
# module docstring for why).
# ---------------------------------------------------------------------------


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:
    """Load the newest ``data/raw/*/schedules.parquet`` snapshot."""

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def default_opener_lines(
    schedule: pd.DataFrame, *, market_root: Path | None = None
) -> pd.DataFrame:
    """Tuesday-opener consensus home spread, keyed by ``game_id``.

    Same sign convention as every sibling on-production candidate
    (``nfl_ats.schedule_flag_features.default_opener_lines``): positive
    means the HOME team is favored.
    """

    if "game_id" not in schedule.columns:
        raise DataContractError("schedule is missing the game_id column")
    root = market_root or DEFAULT_MARKET_ROOT
    pairing = build_pairing_table(
        root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        schedule=schedule[["game_id", "season", "week"]],
    )
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")]
    lines = tue_open[["game_id", "home_spread", "total_line"]].rename(
        columns={"home_spread": "tue_open_home_spread", "total_line": "tue_open_total_line"}
    )
    return lines.drop_duplicates("game_id").reset_index(drop=True)


def default_forecast_archive(
    archive_path: Path | None = None, *, repo_root: Path | None = None
) -> pd.DataFrame:
    """Load and verify the immutable pool-decision forecast archive."""

    root = repo_root or REPO_ROOT
    path = archive_path or (root / DEFAULT_FORECAST_ARCHIVE)
    return load_forecast_archive(path)


def _dog_flag_from_opener_spread(eligible: pd.Series, spread: pd.Series) -> np.ndarray:
    """``+1`` home dog / ``-1`` away dog / ``0`` otherwise, restricted to
    ``eligible`` rows with a resolved opener spread. Same shape as
    ``nfl_ats.schedule_flag_features._dog_flag_from_opener_spread``."""

    home_dog = eligible & spread.notna() & spread.lt(0.0)
    away_dog = eligible & spread.notna() & spread.gt(0.0)
    return np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))


def _attach(
    features: pd.DataFrame,
    column: str,
    derived: pd.DataFrame,
) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if column in features.columns:
        raise DataContractError(f"features already carries {column}")
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_weather_venue_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_weather_venue_flag") if c in merged.columns]
    )
    merged.index = features.index
    return merged


# ---------------------------------------------------------------------------
# LEAD-36: open-corner stadium wind
# ---------------------------------------------------------------------------


def derive_open_corner_wind_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, open_corner_wind_dog_flag)`` for every game in
    ``schedule``. See the module docstring for the full construct."""

    missing = sorted(_OPEN_CORNER_WIND_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    merged = schedule[["game_id", "stadium_id", "roof", "wind"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    open_corner_venue = merged["stadium_id"].isin(OPEN_CORNER_STADIUMS)
    outdoor = merged["roof"].isin(OPEN_CORNER_OUTDOOR_ROOFS)
    wind = pd.to_numeric(merged["wind"], errors="coerce")
    high_wind = wind.notna() & wind.ge(OPEN_CORNER_WIND_MPH_MIN)
    eligible = open_corner_venue & outdoor & high_wind
    flag = _dog_flag_from_opener_spread(eligible, merged["tue_open_home_spread"])
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), OPEN_CORNER_WIND_DOG_COLUMN: flag}
    )


def attach_open_corner_wind_dog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``open_corner_wind_dog_flag`` onto ``features``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_lines = (
        opener_lines
        if opener_lines is not None
        else default_opener_lines(resolved_schedule, market_root=market_root)
    )
    derived = derive_open_corner_wind_dog_features(resolved_schedule, resolved_lines)
    return _attach(features, OPEN_CORNER_WIND_DOG_COLUMN, derived)


def open_corner_wind_population_diagnostic(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame
) -> dict:
    """Measured population counts, reported alongside the flag, never used
    to build it: how many games sit at a frozen open-corner venue, how many
    of those are outdoor, how many clear the wind threshold, and the
    resulting flag's split by season and by venue."""

    merged = schedule[["game_id", "season", "stadium_id", "roof", "wind"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    open_corner_venue = merged["stadium_id"].isin(OPEN_CORNER_STADIUMS)
    outdoor = merged["roof"].isin(OPEN_CORNER_OUTDOOR_ROOFS)
    wind = pd.to_numeric(merged["wind"], errors="coerce")
    high_wind = wind.notna() & wind.ge(OPEN_CORNER_WIND_MPH_MIN)
    eligible = open_corner_venue & outdoor & high_wind
    return {
        "n_open_corner_venue_games": int(open_corner_venue.sum()),
        "n_open_corner_outdoor_games": int((open_corner_venue & outdoor).sum()),
        "n_eligible_high_wind_games": int(eligible.sum()),
        "eligible_by_season": {
            str(season): int(count)
            for season, count in merged.loc[eligible, "season"].value_counts().sort_index().items()
        },
        "eligible_by_venue": {
            str(code): int(count)
            for code, count in merged.loc[eligible, "stadium_id"]
            .map(OPEN_CORNER_STADIUMS)
            .value_counts()
            .items()
        },
        "eligible_missing_opener_spread": int(
            (eligible & merged["tue_open_home_spread"].isna()).sum()
        ),
    }


# ---------------------------------------------------------------------------
# LEAD-37: rain-on-grass fumble chaos
# ---------------------------------------------------------------------------


def derive_rain_on_grass_dog_features(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame, forecast: pd.DataFrame
) -> pd.DataFrame:
    """Return ``(game_id, rain_on_grass_dog_flag)`` for every game in
    ``schedule``. See the module docstring for the full construct."""

    missing = sorted(_RAIN_ON_GRASS_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")
    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")
    required_forecast = {"game_id", "forecast_precip_prob_pct"}
    missing_forecast = sorted(required_forecast.difference(forecast.columns))
    if missing_forecast:
        raise DataContractError(f"forecast is missing columns: {', '.join(missing_forecast)}")

    merged = schedule[["game_id", "surface"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        forecast[["game_id", "forecast_precip_prob_pct"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    grass = merged["surface"].astype(str).str.strip().str.lower().isin(GRASS_SURFACES)
    precip_prob = pd.to_numeric(merged["forecast_precip_prob_pct"], errors="coerce")
    high_precip = precip_prob.notna() & precip_prob.ge(RAIN_ON_GRASS_PRECIP_PROB_MIN)
    eligible = grass & high_precip
    flag = _dog_flag_from_opener_spread(eligible, merged["tue_open_home_spread"])
    return pd.DataFrame({"game_id": merged["game_id"].astype(str), RAIN_ON_GRASS_DOG_COLUMN: flag})


def attach_rain_on_grass_dog_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    market_root: Path | None = None,
    archive_path: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``rain_on_grass_dog_flag`` onto ``features``."""

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_lines = (
        opener_lines
        if opener_lines is not None
        else default_opener_lines(resolved_schedule, market_root=market_root)
    )
    resolved_forecast = forecast if forecast is not None else default_forecast_archive(archive_path)
    derived = derive_rain_on_grass_dog_features(
        resolved_schedule, resolved_lines, resolved_forecast
    )
    return _attach(features, RAIN_ON_GRASS_DOG_COLUMN, derived)


def rain_on_grass_population_diagnostic(
    schedule: pd.DataFrame, opener_lines: pd.DataFrame, forecast: pd.DataFrame
) -> dict:
    """Measured population counts, reported alongside the flag, never used
    to build it: how many games are on grass, how many of those have a
    resolved forecast precip probability, how many clear the threshold, and
    the resulting flag's split by season."""

    merged = schedule[["game_id", "season", "surface"]].merge(
        opener_lines[["game_id", "tue_open_home_spread"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        forecast[["game_id", "forecast_precip_prob_pct"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    grass = merged["surface"].astype(str).str.strip().str.lower().isin(GRASS_SURFACES)
    precip_prob = pd.to_numeric(merged["forecast_precip_prob_pct"], errors="coerce")
    high_precip = precip_prob.notna() & precip_prob.ge(RAIN_ON_GRASS_PRECIP_PROB_MIN)
    eligible = grass & high_precip
    return {
        "n_grass_games": int(grass.sum()),
        "n_grass_games_with_forecast": int((grass & precip_prob.notna()).sum()),
        "n_eligible_high_precip_games": int(eligible.sum()),
        "eligible_by_season": {
            str(season): int(count)
            for season, count in merged.loc[eligible, "season"].value_counts().sort_index().items()
        },
        "eligible_missing_opener_spread": int(
            (eligible & merged["tue_open_home_spread"].isna()).sum()
        ),
    }


__all__ = [
    "DEFAULT_MARKET_ROOT",
    "OPEN_CORNER_OUTDOOR_ROOFS",
    "OPEN_CORNER_STADIUMS",
    "OPEN_CORNER_WIND_DOG_COLUMN",
    "OPEN_CORNER_WIND_MPH_MIN",
    "RAIN_ON_GRASS_DOG_COLUMN",
    "RAIN_ON_GRASS_PRECIP_PROB_MIN",
    "attach_open_corner_wind_dog_features",
    "attach_rain_on_grass_dog_features",
    "default_forecast_archive",
    "default_opener_lines",
    "default_schedule",
    "derive_open_corner_wind_dog_features",
    "derive_rain_on_grass_dog_features",
    "open_corner_wind_population_diagnostic",
    "rain_on_grass_population_diagnostic",
]

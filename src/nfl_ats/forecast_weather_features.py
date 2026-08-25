"""weak_stack_v4 forecast-weather features (docs/weak_stack_v4.md).

Six continuous/structural columns joined by ``game_id`` from the completed
kickoff-nearest forecast archive
(``data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet``,
one row per REG game, 4,431/4,431).

Deliberately continuous. ``weak_stack_v3`` already tested fifteen hand-coded
situational FLAGS and was refused at the opener on EV; the registered
forecast-weather cells are the same shape (the strongest,
``forecast_weather_kn_warm_team_cold_late_full``, fires on 1.51% of the slate).
The open question is whether ridge finds more in the raw variables than the
cells did, so this family hands it the variables.

Leak safety: the ``kickoff_nearest`` cutoff selects the forecast issuance
nearest each kickoff, and every archive row's ``issuance_runtime_utc``
precedes its ``kickoff_utc``. It is pregame by construction, and playable
under this pool's rules -- picks stay editable until each game's own kickoff
and only the LINES freeze on Tuesday.

These columns stay OUT of ``MODEL_FEATURE_COLUMNS``, on the same
``BIAS_METRICS``/``SURFACE_SWITCH_FEATURE_COLUMNS`` precedent, so only the
explicitly opted-in ``weak_stack_v4`` profile ever reads them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.data import DataContractError

#: The completed kickoff-nearest archive, relative to the repository root.
DEFAULT_FORECAST_ARCHIVE = Path(
    "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
)

#: Frozen in docs/weak_stack_v4.md before any scoring.
FORECAST_WEATHER_COLUMNS: tuple[str, ...] = (
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
    "forecast_is_outdoors",
    "forecast_temp_f_outdoor",
    "forecast_wind_mph_outdoor",
)

_ARCHIVE_SOURCE_COLUMNS = (
    "game_id",
    "roof",
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
)


def load_forecast_archive(path: Path) -> pd.DataFrame:
    """Read the archive and keep only the columns this family consumes."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Forecast archive not found: {path}. Build it with "
            "`scripts/ingest_forecast_archive.py --cutoff-mode kickoff_nearest`."
        )
    archive = pd.read_parquet(path)
    missing = sorted(set(_ARCHIVE_SOURCE_COLUMNS).difference(archive.columns))
    if missing:
        raise DataContractError(f"Forecast archive is missing columns: {', '.join(missing)}")
    frame = archive.loc[:, list(_ARCHIVE_SOURCE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise DataContractError("Forecast archive contains duplicate game_id rows")
    return frame


def derive_forecast_features(archive: pd.DataFrame) -> pd.DataFrame:
    """The six declared columns, from the archive's raw fields.

    The two ``_outdoor`` interactions exist because a dome game's forecast
    temperature is not a football input at all. They are masked to the OUTDOOR
    MEDIAN rather than to zero, so a dome never reads as an extreme cold game
    -- zero-filling would invent the very signal the family is testing for.
    """

    frame = archive.copy()
    outdoors = frame["roof"].astype(str).str.lower().eq("outdoors")
    frame["forecast_is_outdoors"] = outdoors.astype(float)

    for source, target in (
        ("forecast_temp_f", "forecast_temp_f_outdoor"),
        ("forecast_wind_mph", "forecast_wind_mph_outdoor"),
    ):
        values = pd.to_numeric(frame[source], errors="coerce")
        outdoor_median = float(values.loc[outdoors].median())
        frame[target] = values.where(outdoors, outdoor_median)

    for column in ("forecast_temp_f", "forecast_wind_mph", "forecast_precip_prob_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.loc[:, ["game_id", *FORECAST_WEATHER_COLUMNS]]


def attach_forecast_weather_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    archive_path: Path | None = None,
) -> pd.DataFrame:
    """Additively join the six forecast columns onto a feature table.

    Every pre-existing column is returned bit-identical; only the six new
    columns are added. Games with no archive row (52 of 4,431, all rows whose
    ``icao_station`` never mapped) come back NaN and are left NaN here on
    purpose -- imputation belongs to the model's own training-fold median, not
    to a feature builder that can see every season at once.
    """

    root = repo_root or Path.cwd()
    path = archive_path or (root / DEFAULT_FORECAST_ARCHIVE)
    derived = derive_forecast_features(load_forecast_archive(path))

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(FORECAST_WEATHER_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(
            f"features already carries forecast columns: {', '.join(collisions)}"
        )

    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_archive"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_archive") if c in merged.columns])
    merged.index = features.index
    return merged


def coverage_summary(features: pd.DataFrame) -> dict[str, float | int]:
    """Coverage of the six columns, for a build log or a write-up."""

    present = features["forecast_temp_f"].notna()
    return {
        "rows": len(features),
        "forecast_rows": int(present.sum()),
        "forecast_coverage": float(present.mean()) if len(features) else 0.0,
        "outdoor_rows": int(features["forecast_is_outdoors"].fillna(0).sum()),
        "median_outdoor_temp_f": float(
            features.loc[features["forecast_is_outdoors"].eq(1.0), "forecast_temp_f"].median()
        )
        if features["forecast_is_outdoors"].eq(1.0).any()
        else float("nan"),
    }


__all__ = [
    "DEFAULT_FORECAST_ARCHIVE",
    "FORECAST_WEATHER_COLUMNS",
    "attach_forecast_weather_features",
    "coverage_summary",
    "derive_forecast_features",
    "load_forecast_archive",
]

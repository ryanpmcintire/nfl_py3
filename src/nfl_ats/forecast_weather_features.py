"""weak_stack_v4 forecast-weather features (docs/weak_stack_v4.md).

Six continuous/structural columns joined by ``game_id`` from a forecast
archive built at the pool's real decision timestamp: the earlier of kickoff
and Sunday 16:00 America/New_York in that game's NFL week.  The former
``kickoff_nearest`` archive is intentionally rejected because its Sunday-night
and Monday-night rows can use bulletins published after the card locked.

Deliberately continuous. ``weak_stack_v3`` already tested fifteen hand-coded
situational FLAGS and was refused at the opener on EV; the registered
forecast-weather cells are the same shape (the strongest,
``forecast_weather_kn_warm_team_cold_late_full``, fires on 1.51% of the slate).
The open question is whether ridge finds more in the raw variables than the
cells did, so this family hands it the variables.

Leak safety: every consumed row proves ``issuance_runtime_utc <=
decision_cutoff_utc == min(kickoff, Sunday 16:00 ET)``.  The loader verifies
that chronology, the declared cutoff mode, complete game-id coverage, allowed
fetch statuses, and the parquet hash recorded in the sibling manifest before
returning any feature values.

These columns stay OUT of ``MODEL_FEATURE_COLUMNS``, on the same
``BIAS_METRICS``/``SURFACE_SWITCH_FEATURE_COLUMNS`` precedent, so only the
explicitly opted-in ``weak_stack_v4`` profile ever reads them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.nfl_week import pool_decision_cutoff
from nfl_ats.provenance import sha256_file

#: The completed kickoff-nearest archive, relative to the repository root.
DEFAULT_FORECAST_ARCHIVE = Path(
    "data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet"
)
POOL_DECISION_CUTOFF_MODE = "pool_decision"

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
    "kickoff_utc",
    "decision_cutoff_utc",
    "issuance_runtime_utc",
    "cutoff_mode",
    "fetch_status",
    "roof",
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
)

_CONSUMABLE_FETCH_STATUSES = frozenset({"ok", "unmappable_international_stadium"})

#: POSITIVE CONTROL ONLY -- the weather that ACTUALLY happened, which is not
#: knowable before kickoff. Never promotable, never a production feature; see
#: :func:`derive_observed_weather_features`.
OBSERVED_WEATHER_COLUMNS: tuple[str, ...] = (
    "observed_temp_f",
    "observed_wind_mph",
    "observed_is_outdoors",
    "observed_temp_f_outdoor",
    "observed_wind_mph_outdoor",
)

_OBSERVED_SOURCE_COLUMNS = ("game_id", "roof", "actual_temp_f", "actual_wind_mph")


def derive_observed_weather_features(archive: pd.DataFrame) -> pd.DataFrame:
    """Observed weather as an ORACLE, for bounding the whole weather channel.

    **This is deliberately leaky and must never reach production.** It answers
    a question a better forecast cannot: if the model is handed the weather
    that actually occurred -- a forecast of infinite skill -- does forced-pick
    accuracy move at all?

    That makes it a positive control in the AGENTS.md sense. If even perfect
    weather knowledge does not beat the baseline, the channel is
    ``bounded_by_control``: no improvement in forecasting can recover an effect
    the oracle itself cannot produce. If it DOES help, the gap between the
    oracle and the real forecast arm is exactly the headroom better forecasting
    could buy, which is the number worth having before spending effort on a
    better wind source.

    Mirrors :func:`derive_forecast_features` exactly, substituting
    ``actual_*`` for ``forecast_*``, so the two arms differ in the ORACLE and
    nothing else.
    """

    missing = sorted(set(_OBSERVED_SOURCE_COLUMNS).difference(archive.columns))
    if missing:
        raise DataContractError(f"Forecast archive is missing columns: {', '.join(missing)}")

    frame = archive.loc[:, list(_OBSERVED_SOURCE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    outdoors = frame["roof"].astype(str).str.lower().eq("outdoors")
    frame["observed_is_outdoors"] = outdoors.astype(float)
    frame["observed_temp_f"] = pd.to_numeric(frame["actual_temp_f"], errors="coerce")
    frame["observed_wind_mph"] = pd.to_numeric(frame["actual_wind_mph"], errors="coerce")

    for source, target in (
        ("observed_temp_f", "observed_temp_f_outdoor"),
        ("observed_wind_mph", "observed_wind_mph_outdoor"),
    ):
        values = frame[source]
        outdoor_median = float(values.loc[outdoors].median())
        frame[target] = values.where(outdoors, outdoor_median)

    return frame.loc[:, ["game_id", *OBSERVED_WEATHER_COLUMNS]]


def attach_observed_weather_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    archive_path: Path | None = None,
) -> pd.DataFrame:
    """Additively join the oracle columns. POSITIVE CONTROL ONLY."""

    root = repo_root or Path.cwd()
    path = archive_path or (root / DEFAULT_FORECAST_ARCHIVE)
    derived = derive_observed_weather_features(load_observed_archive(path))

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(OBSERVED_WEATHER_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(
            f"features already carries observed columns: {', '.join(collisions)}"
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


def load_observed_archive(path: Path) -> pd.DataFrame:
    """The archive with its observed-weather columns kept."""

    if not path.is_file():
        raise FileNotFoundError(f"Forecast archive not found: {path}")
    archive = pd.read_parquet(path)
    missing = sorted(set(_OBSERVED_SOURCE_COLUMNS).difference(archive.columns))
    if missing:
        raise DataContractError(f"Forecast archive is missing columns: {', '.join(missing)}")
    frame = archive.loc[:, list(_OBSERVED_SOURCE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise DataContractError("Forecast archive contains duplicate game_id rows")
    return frame


def _validate_archive_manifest(path: Path) -> int:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise DataContractError(f"Forecast archive manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Forecast archive manifest is unreadable: {error}") from error
    if manifest.get("cutoff_mode") != POOL_DECISION_CUTOFF_MODE:
        raise DataContractError("Forecast archive manifest must declare cutoff_mode=pool_decision")
    if manifest.get("mos_model") != "GFS":
        raise DataContractError("Pool-decision forecast archive must declare mos_model=GFS")
    files = manifest.get("files")
    file_metadata = files.get(path.name) if isinstance(files, dict) else None
    if not isinstance(file_metadata, dict):
        raise DataContractError("Forecast archive manifest is missing parquet metadata")
    expected_hash = file_metadata.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise DataContractError("Forecast archive manifest is missing the parquet SHA-256")
    if sha256_file(path) != expected_hash:
        raise DataContractError("Forecast archive parquet does not match its manifest SHA-256")
    expected_rows = file_metadata.get("rows")
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise DataContractError("Forecast archive manifest has an invalid parquet row count")
    return expected_rows


def _validate_pool_decision_archive(frame: pd.DataFrame) -> None:
    if frame.empty or frame["game_id"].isna().any():
        raise DataContractError("Forecast archive requires non-null game rows")
    modes = frame["cutoff_mode"].astype(str)
    if not modes.eq(POOL_DECISION_CUTOFF_MODE).all():
        raise DataContractError("Forecast archive contains rows outside pool_decision mode")

    statuses = frame["fetch_status"].astype(str)
    invalid_statuses = sorted(set(statuses).difference(_CONSUMABLE_FETCH_STATUSES))
    if invalid_statuses:
        raise DataContractError(
            "Forecast archive has unresolved domestic coverage failures: "
            + ", ".join(invalid_statuses)
        )

    kickoff = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(frame["decision_cutoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any() or cutoff.isna().any():
        raise DataContractError("Forecast archive has invalid kickoff or decision timestamps")
    expected_cutoff = pd.Series(
        [pd.Timestamp(pool_decision_cutoff(value.to_pydatetime())) for value in kickoff],
        index=frame.index,
    )
    if not cutoff.eq(expected_cutoff).all():
        raise DataContractError(
            "Forecast archive decision cutoff does not match min(kickoff, Sunday 16:00 ET)"
        )

    ok = statuses.eq("ok")
    issuance = pd.to_datetime(frame["issuance_runtime_utc"], utc=True, errors="coerce")
    if issuance.loc[ok].isna().any() or issuance.loc[ok].gt(cutoff.loc[ok]).any():
        raise DataContractError(
            "Forecast archive has an OK bulletin issued after its pool decision cutoff"
        )
    required_weather = frame.loc[ok, ["forecast_temp_f", "forecast_wind_mph"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if required_weather.isna().any(axis=None):
        raise DataContractError("Forecast archive has an OK domestic row without temp/wind")
    unmappable = statuses.eq("unmappable_international_stadium")
    weather = frame.loc[
        unmappable,
        ["forecast_temp_f", "forecast_wind_mph", "forecast_precip_prob_pct"],
    ]
    if weather.notna().any(axis=None):
        raise DataContractError("Unmappable forecast rows must not carry weather values")


def load_forecast_archive(path: Path) -> pd.DataFrame:
    """Read and verify the immutable pool-decision archive."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Forecast archive not found: {path}. Build it with "
            "`scripts/ingest_forecast_archive.py --cutoff-mode pool_decision`."
        )
    expected_rows = _validate_archive_manifest(path)
    archive = pd.read_parquet(path)
    if len(archive) != expected_rows:
        raise DataContractError("Forecast archive row count does not match its manifest")
    missing = sorted(set(_ARCHIVE_SOURCE_COLUMNS).difference(archive.columns))
    if missing:
        raise DataContractError(f"Forecast archive is missing columns: {', '.join(missing)}")
    frame = archive.loc[:, list(_ARCHIVE_SOURCE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise DataContractError("Forecast archive contains duplicate game_id rows")
    _validate_pool_decision_archive(frame)
    return frame.loc[:, ["game_id", "roof", *FORECAST_WEATHER_COLUMNS[:3]]]


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
    columns are added.  Every feature-table game must have exactly one archive
    record. Deliberately unmappable international rows remain NaN; any absent
    row or unresolved domestic fetch fails closed. Imputation belongs to the
    model's own training fold, not to this join.
    """

    root = repo_root or Path.cwd()
    path = archive_path or (root / DEFAULT_FORECAST_ARCHIVE)
    derived = derive_forecast_features(load_forecast_archive(path))

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    feature_game_ids = features["game_id"].astype(str)
    missing_game_ids = sorted(set(feature_game_ids).difference(derived["game_id"]))
    if missing_game_ids:
        preview = ", ".join(missing_game_ids[:5])
        raise DataContractError(
            f"Forecast archive is missing {len(missing_game_ids)} feature-table games: {preview}"
        )
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

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_SOURCE = "internet_archive_pfr_boxscores"
NFLVERSE_SOURCE = "nflverse_officials"

MANIFEST_SCHEMA = "officials_pfr_wayback_manifest/1"

CORE_CREW_POSITIONS: tuple[str, ...] = (
    "Referee",
    "Umpire",
    "Head Linesman",
    "Line Judge",
    "Field Judge",
    "Side Judge",
    "Back Judge",
)

POSITION_ALIASES: dict[str, str] = {"Down Judge": "Head Linesman"}

CREW_COLUMN_BY_POSITION: dict[str, str] = {
    "Referee": "referee",
    "Umpire": "umpire",
    "Head Linesman": "head_linesman",
    "Line Judge": "line_judge",
    "Field Judge": "field_judge",
    "Side Judge": "side_judge",
    "Back Judge": "back_judge",
}

CREW_COLUMNS: tuple[str, ...] = tuple(CREW_COLUMN_BY_POSITION[p] for p in CORE_CREW_POSITIONS)

CANONICAL_CREW_COLUMNS: tuple[str, ...] = (
    "game_id",
    "old_game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "pfr_id",
    *CREW_COLUMNS,
    "complete_crew",
    "n_crew_rows",
    "n_positions",
    "discarded_duplicate_position_rows",
    "source",
    "source_run_id",
    "wayback_capture_timestamp",
    "wayback_captured_at_utc",
    "wayback_url",
    "fetched_at_utc",
)

NFLVERSE_OFFICIALS_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_key",
    "official_name",
    "position",
    "jersey_number",
    "official_id",
    "season",
    "season_type",
    "week",
)

ARCHIVE_SEASON_TYPE = "REG"

ARCHIVE_TIMING_CLASS = "crew_identity_public_pregame_not_provably_captured_pregame"

INCLUDE_ARCHIVE_DEFAULT = False

_MANIFEST_READ_ATTEMPTS = 5
_MANIFEST_READ_RETRY_SECONDS = 0.05


class OfficialsArchiveError(DataContractError):
    pass


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    path: Path
    capture_policy: str | None
    season_start: int | None
    season_end: int | None
    n_manifest_rows: int


_SWEEP_MODULE_NAME = "nfl_ats_officials_wayback_sweep"
_SWEEP_MODULES: dict[str, Any] = {}


def _load_sweep_module(repo_root: Path) -> Any:

    path = repo_root / "scripts" / "officials_wayback_sweep.py"
    key = str(path)
    cached = _SWEEP_MODULES.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_SWEEP_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise OfficialsArchiveError(f"cannot load the Wayback sweep parser from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SWEEP_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SWEEP_MODULE_NAME, None)
        raise
    _SWEEP_MODULES[key] = module
    return module


def normalize_position(position: str) -> str:

    return POSITION_ALIASES.get(position, position)


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    last_error: Exception | None = None
    for attempt in range(_MANIFEST_READ_ATTEMPTS):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < _MANIFEST_READ_ATTEMPTS:
                time.sleep(_MANIFEST_READ_RETRY_SECONDS)
            continue
        if not isinstance(payload, dict):
            return None
        return payload
    raise OfficialsArchiveError(
        f"could not read {manifest_path} after {_MANIFEST_READ_ATTEMPTS} attempts: {last_error}"
    )


def discover_sweep_runs(raw_root: Path) -> list[SweepRun]:

    if not raw_root.is_dir():
        return []
    runs: list[SweepRun] = []
    for run_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        payload = _read_manifest(run_dir)
        if payload is None or payload.get("schema") != MANIFEST_SCHEMA:
            continue
        games = payload.get("games")
        if not isinstance(games, list):
            continue
        runs.append(
            SweepRun(
                run_id=str(payload.get("run_id") or run_dir.name),
                path=run_dir,
                capture_policy=(
                    str(payload["capture_policy"])
                    if payload.get("capture_policy") is not None
                    else None
                ),
                season_start=(
                    int(payload["season_start"])
                    if payload.get("season_start") is not None
                    else None
                ),
                season_end=(
                    int(payload["season_end"]) if payload.get("season_end") is not None else None
                ),
                n_manifest_rows=len(games),
            )
        )
    return runs


def default_archive_root(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / "data" / "raw" / "officials_pfr_wayback"


_CREW_ROW_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], pd.DataFrame] = {}


def _manifest_fingerprint(runs: list[SweepRun]) -> tuple[tuple[str, int, int], ...]:
    fingerprint: list[tuple[str, int, int]] = []
    for run in runs:
        stat = (run.path / "manifest.json").stat()
        fingerprint.append((run.run_id, stat.st_size, stat.st_mtime_ns))
    return tuple(fingerprint)


def clear_cache() -> None:

    _CREW_ROW_CACHE.clear()


ARCHIVE_CREW_ROW_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "pfr_id",
    "position",
    "raw_position",
    "official_name",
    "row_order",
    "source",
    "source_run_id",
    "wayback_capture_timestamp",
    "wayback_url",
    "fetched_at_utc",
    "html_file",
)


def load_archive_crew_rows(
    *, repo_root: Path | None = None, raw_root: Path | None = None, use_cache: bool = True
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    archive_root = raw_root if raw_root is not None else default_archive_root(root)
    runs = discover_sweep_runs(archive_root)

    cache_key = (str(archive_root), _manifest_fingerprint(runs))
    if use_cache and cache_key in _CREW_ROW_CACHE:
        return _CREW_ROW_CACHE[cache_key].copy()

    sweep = _load_sweep_module(root)
    records: list[dict[str, Any]] = []
    for run in runs:
        payload = _read_manifest(run.path)
        if payload is None:
            continue
        for game_row in payload.get("games", []):
            html_file = game_row.get("html_file")
            if not html_file:
                continue
            html_path = run.path / str(html_file)
            if not html_path.is_file():
                continue
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
            parsed, _warnings = sweep.parse_officials_block(html_text)
            for order, (position, official_name) in enumerate(parsed):
                records.append(
                    {
                        "game_id": str(game_row.get("game_id")),
                        "season": game_row.get("season"),
                        "week": game_row.get("week"),
                        "gameday": str(game_row.get("gameday")),
                        "home_team": game_row.get("home_team"),
                        "away_team": game_row.get("away_team"),
                        "pfr_id": str(game_row.get("pfr_id")),
                        "position": normalize_position(str(position)),
                        "raw_position": str(position),
                        "official_name": str(official_name),
                        "row_order": order,
                        "source": ARCHIVE_SOURCE,
                        "source_run_id": run.run_id,
                        "wayback_capture_timestamp": game_row.get("wayback_capture_timestamp"),
                        "wayback_url": game_row.get("wayback_url"),
                        "fetched_at_utc": game_row.get("fetch_instant_utc"),
                        "html_file": str(html_file),
                    }
                )

    frame = pd.DataFrame(records, columns=list(ARCHIVE_CREW_ROW_COLUMNS))
    if use_cache:
        _CREW_ROW_CACHE[cache_key] = frame.copy()
    return frame


def _as_text(value: Any) -> str | None:

    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _text_series(series: pd.Series) -> pd.Series:
    return pd.Series([_as_text(v) for v in series], index=series.index, dtype="str")


def _capture_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(_text_series(series), format="%Y%m%d%H%M%S", errors="coerce")


def assert_captures_are_post_game(frame: pd.DataFrame) -> None:

    if frame.empty:
        return
    captured = _capture_datetime(frame["wayback_capture_timestamp"])
    gameday = pd.to_datetime(_text_series(frame["gameday"]), errors="coerce")
    bad = captured.isna() | gameday.isna() | (captured < gameday + pd.Timedelta(days=1))
    if bool(bad.any()):
        offenders = sorted(set(frame.loc[bad, "game_id"].astype("str")))[:5]
        raise OfficialsArchiveError(
            "officials archive capture timestamps must land after the game's own day "
            f"(timing class {ARCHIVE_TIMING_CLASS}); offending game_ids: {', '.join(offenders)}"
        )


def refuse_archive_rows(frame: pd.DataFrame, *, channel: str) -> None:

    if "source" not in frame.columns:
        return
    n_archive = int((frame["source"].astype("str") == ARCHIVE_SOURCE).sum())
    if n_archive:
        raise OfficialsArchiveError(
            f"{n_archive} Wayback-archive officials rows were offered to the prospective "
            f"channel {channel!r}; the archive's timing class is {ARCHIVE_TIMING_CLASS}, so "
            "its captures are always after kickoff and can never satisfy a "
            "captured-before-the-pick-deadline test"
        )


def latest_schedules_path(repo_root: Path | None = None) -> Path:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise OfficialsArchiveError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return candidates[-1]


def _default_schedules(repo_root: Path) -> pd.DataFrame:
    return pd.read_parquet(latest_schedules_path(repo_root))


_SCHEDULE_REQUIRED = ("game_id", "old_game_id", "season", "week", "gameday")


def _resolve_cross_run_duplicates(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

    if rows.empty:
        return rows, []
    per_game_run = (
        rows.groupby(["game_id", "source_run_id"], dropna=False)["wayback_capture_timestamp"]
        .max()
        .reset_index()
    )
    duplicated = per_game_run.loc[per_game_run["game_id"].duplicated(keep=False), "game_id"]
    duplicate_games = sorted(set(duplicated.astype("str")))
    if not duplicate_games:
        return rows, []
    per_game_run = per_game_run.sort_values(
        ["game_id", "wayback_capture_timestamp", "source_run_id"], ascending=[True, False, True]
    )
    winners = per_game_run.drop_duplicates("game_id")[["game_id", "source_run_id"]]
    kept = rows.merge(winners, on=["game_id", "source_run_id"], how="inner")
    return kept.reset_index(drop=True), duplicate_games


def _validate_schedule_join(rows: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_SCHEDULE_REQUIRED).difference(schedules.columns))
    if missing:
        raise OfficialsArchiveError(f"schedules snapshot is missing columns: {', '.join(missing)}")
    lookup = schedules.loc[:, list(_SCHEDULE_REQUIRED)].copy()
    lookup["game_id"] = lookup["game_id"].astype("str")
    counts = lookup.groupby("game_id").size()

    archive_games = sorted(set(rows["game_id"].astype("str")))
    unmatched = [g for g in archive_games if g not in counts.index]
    if unmatched:
        raise OfficialsArchiveError(
            "officials archive rows do not join to any schedule game: "
            f"{', '.join(unmatched[:5])} ({len(unmatched)} total)"
        )
    ambiguous = [g for g in archive_games if int(counts.loc[g]) > 1]
    if ambiguous:
        raise OfficialsArchiveError(
            "officials archive rows join to more than one schedule game: "
            f"{', '.join(ambiguous[:5])} ({len(ambiguous)} total)"
        )
    return lookup.drop_duplicates("game_id")


def canonical_crew_table(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    crew = rows if rows is not None else load_archive_crew_rows(repo_root=root, raw_root=raw_root)
    if crew.empty:
        return pd.DataFrame(columns=list(CANONICAL_CREW_COLUMNS))

    assert_captures_are_post_game(crew)
    crew, _duplicate_games = _resolve_cross_run_duplicates(crew)

    sched = schedules if schedules is not None else _default_schedules(root)
    lookup = _validate_schedule_join(crew, sched)

    crew = crew.sort_values(["game_id", "row_order"]).reset_index(drop=True)
    deduped = crew.drop_duplicates(["game_id", "position"], keep="first")
    core = deduped.loc[deduped["position"].isin(CORE_CREW_POSITIONS)]
    if core.empty:
        return pd.DataFrame(columns=list(CANONICAL_CREW_COLUMNS))
    discarded = (crew.groupby("game_id").size() - deduped.groupby("game_id").size()).rename(
        "discarded_duplicate_position_rows"
    )

    wide = core.pivot(index="game_id", columns="position", values="official_name").rename(
        columns=CREW_COLUMN_BY_POSITION
    )
    for column in CREW_COLUMNS:
        if column not in wide.columns:
            wide[column] = pd.Series(pd.NA, index=wide.index, dtype="str")
    wide = wide.loc[:, list(CREW_COLUMNS)]

    per_game = crew.drop_duplicates("game_id").set_index("game_id")
    table = wide.join(
        per_game.loc[
            :,
            [
                "home_team",
                "away_team",
                "pfr_id",
                "source",
                "source_run_id",
                "wayback_capture_timestamp",
                "wayback_url",
                "fetched_at_utc",
            ],
        ]
    )
    table["n_crew_rows"] = crew.groupby("game_id").size()
    table["n_positions"] = deduped.groupby("game_id")["position"].nunique()
    table["discarded_duplicate_position_rows"] = discarded.astype("int64")
    table["complete_crew"] = table.loc[:, list(CREW_COLUMNS)].notna().all(axis=1)
    table["wayback_captured_at_utc"] = _capture_datetime(table["wayback_capture_timestamp"])
    table = table.reset_index()

    table = table.merge(
        lookup.loc[:, ["game_id", "old_game_id", "season", "week", "gameday"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    table["season"] = table["season"].astype("int32")
    table["week"] = table["week"].astype("int32")
    table["gameday"] = table["gameday"].astype("str")
    table = table.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return table.loc[:, list(CANONICAL_CREW_COLUMNS)]


def archive_officials_long(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:

    wide = (
        table
        if table is not None
        else canonical_crew_table(
            repo_root=repo_root, raw_root=raw_root, rows=rows, schedules=schedules
        )
    )
    if wide.empty:
        return _empty_merged_frame()

    long = wide.melt(
        id_vars=["old_game_id", "season", "week"],
        value_vars=list(CREW_COLUMNS),
        var_name="crew_column",
        value_name="official_name",
    )
    long = long.loc[long["official_name"].notna()].copy()
    column_to_position = {v: k for k, v in CREW_COLUMN_BY_POSITION.items()}
    long["position"] = long["crew_column"].map(column_to_position)
    long["game_id"] = long["old_game_id"].astype("str")
    long["game_key"] = pd.Series(pd.NA, index=long.index, dtype="str")
    long["jersey_number"] = pd.Series(pd.NA, index=long.index, dtype="Int32")
    long["official_id"] = pd.Series(pd.NA, index=long.index, dtype="str")
    long["season_type"] = ARCHIVE_SEASON_TYPE
    long["source"] = ARCHIVE_SOURCE
    long["official_name"] = long["official_name"].astype("str")
    long["position"] = long["position"].astype("str")
    long["season_type"] = long["season_type"].astype("str")
    long["season"] = long["season"].astype("int32")
    long["week"] = long["week"].astype("int32")

    ordered = long.loc[:, [*NFLVERSE_OFFICIALS_COLUMNS, "source"]]
    position_rank = {name: index for index, name in enumerate(CORE_CREW_POSITIONS)}
    ordered = ordered.assign(_rank=ordered["position"].map(position_rank))
    ordered = ordered.sort_values(["season", "week", "game_id", "_rank"]).drop(columns="_rank")
    return ordered.reset_index(drop=True)


def _empty_merged_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series([], dtype="str"),
            "game_key": pd.Series([], dtype="str"),
            "official_name": pd.Series([], dtype="str"),
            "position": pd.Series([], dtype="str"),
            "jersey_number": pd.Series([], dtype="Int32"),
            "official_id": pd.Series([], dtype="str"),
            "season": pd.Series([], dtype="int32"),
            "season_type": pd.Series([], dtype="str"),
            "week": pd.Series([], dtype="int32"),
            "source": pd.Series([], dtype="str"),
        }
    )


def latest_nflverse_officials_path(repo_root: Path | None = None) -> Path:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "officials").glob("*/officials.parquet"))
    if not candidates:
        raise OfficialsArchiveError(
            f"no data/raw/officials/*/officials.parquet snapshot found under {root}"
        )
    return candidates[-1]


def load_officials(
    repo_root: Path | None = None,
    *,
    officials_path: Path | None = None,
    feed: pd.DataFrame | None = None,
    include_archive: bool = INCLUDE_ARCHIVE_DEFAULT,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    if feed is None:
        feed = pd.read_parquet(officials_path or latest_nflverse_officials_path(root))
    if not include_archive:
        return feed

    missing = sorted(set(NFLVERSE_OFFICIALS_COLUMNS).difference(feed.columns))
    if missing:
        raise OfficialsArchiveError(
            f"nflverse officials feed is missing columns: {', '.join(missing)}"
        )
    archive = archive_officials_long(
        repo_root=root, raw_root=raw_root, rows=rows, schedules=schedules
    )
    feed_side = feed.copy()
    feed_side["jersey_number"] = feed_side["jersey_number"].astype("Int32")
    feed_side["source"] = pd.Series(NFLVERSE_SOURCE, index=feed_side.index, dtype="str")
    feed_side = feed_side.loc[:, [*NFLVERSE_OFFICIALS_COLUMNS, "source"]]
    if archive.empty:
        return feed_side.reset_index(drop=True)

    overlap = archive["game_id"].astype("str").isin(set(feed_side["game_id"].astype("str")))
    archive = archive.loc[~overlap]
    if archive.empty:
        return feed_side.reset_index(drop=True)
    return pd.concat([feed_side, archive], ignore_index=True)


def load_officials_for_prospective_channel(
    repo_root: Path | None = None,
    *,
    officials_path: Path | None = None,
    feed: pd.DataFrame | None = None,
    channel: str = "crew_tilt_refresh",
) -> pd.DataFrame:

    frame = load_officials(
        repo_root, officials_path=officials_path, feed=feed, include_archive=False
    )
    refuse_archive_rows(frame, channel=channel)
    return frame


def describe_archive_coverage(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    table: pd.DataFrame | None = None,
) -> dict[str, Any]:

    wide = (
        table if table is not None else canonical_crew_table(repo_root=repo_root, raw_root=raw_root)
    )
    if wide.empty:
        return {"n_games": 0, "seasons": {}, "timing_class": ARCHIVE_TIMING_CLASS}
    by_season: dict[str, Any] = {}
    for season, group in wide.groupby("season"):
        by_season[str(cast(int, season))] = {
            "n_games": len(group),
            "n_complete_crews": int(group["complete_crew"].sum()),
            "n_distinct_referees": int(group["referee"].dropna().nunique()),
            "n_discarded_duplicate_position_rows": int(
                group["discarded_duplicate_position_rows"].sum()
            ),
        }
    return {
        "n_games": len(wide),
        "n_complete_crews": int(wide["complete_crew"].sum()),
        "seasons": by_season,
        "timing_class": ARCHIVE_TIMING_CLASS,
        "include_archive_default": INCLUDE_ARCHIVE_DEFAULT,
    }


__all__ = [
    "ARCHIVE_CREW_ROW_COLUMNS",
    "ARCHIVE_SEASON_TYPE",
    "ARCHIVE_SOURCE",
    "ARCHIVE_TIMING_CLASS",
    "CANONICAL_CREW_COLUMNS",
    "CORE_CREW_POSITIONS",
    "CREW_COLUMNS",
    "CREW_COLUMN_BY_POSITION",
    "INCLUDE_ARCHIVE_DEFAULT",
    "MANIFEST_SCHEMA",
    "NFLVERSE_OFFICIALS_COLUMNS",
    "NFLVERSE_SOURCE",
    "POSITION_ALIASES",
    "OfficialsArchiveError",
    "SweepRun",
    "archive_officials_long",
    "assert_captures_are_post_game",
    "canonical_crew_table",
    "clear_cache",
    "default_archive_root",
    "describe_archive_coverage",
    "discover_sweep_runs",
    "latest_nflverse_officials_path",
    "latest_schedules_path",
    "load_archive_crew_rows",
    "load_officials",
    "load_officials_for_prospective_channel",
    "normalize_position",
    "refuse_archive_rows",
]

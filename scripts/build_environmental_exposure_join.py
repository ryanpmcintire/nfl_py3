"""Join EPA AQS air-quality + US Drought Monitor drought data to the NFL
schedule, at the HOME stadium's county, as-of the Tuesday of each game week.

INGESTION + COVERAGE REPORT ONLY -- this script performs no experiments and
writes nothing to the weak-signals/rotation registries
(registry/weak_signals.json, registry/experiments/). It reads two raw
snapshots already produced by scripts/ingest_air_quality.py and
scripts/ingest_drought_monitor.py, joins them to the NFL schedule, and
prints/saves coverage tables consumed by docs/environmental_exposures.md.

"As-of Tuesday" definition: for each game, `tuesday_date` is the most recent
Tuesday on or before `gameday` (Tuesday itself if the game is played on a
Tuesday, which has happened for weather/schedule-disruption reasons). This
matches the pool's Tuesday-noon line-freeze convention referenced elsewhere
in this project (picks stay editable til kickoff; only LINES freeze
Tuesday) -- these environmental readings are being aligned to the SAME
weekly checkpoint as a natural "what was knowable by the frozen-line moment"
cut, not because AQI/drought themselves are claimed to be pool inputs.

  - AQI join: most recent EPA AQS daily county row with `date <= tuesday_date`
    (no extra publication-lag buffer -- AQS daily provisional values are
    same/next-day in the live AirNow system this archive descends from; the
    archived value itself is dated to the true measurement day).
  - Drought join: most recent USDM weekly county row whose official release
    timestamp (Thursday 08:30 America/New_York, two days after Tuesday's
    `validStart`) is no later than Tuesday noon ET.  The timestamp is the
    source's actual publication schedule, not an inferred day-count buffer.

Dome / retractable-roof games are INCLUDED in the join (not dropped) but
flagged via the `roof` column already present in nflverse schedules
(dome / closed / open / outdoors) -- AQI's playing-conditions mechanism is
expected to be much weaker indoors (filtered air, no direct exposure) but
outdoor-adjacent retractable-roof games and the roof-open subset of games at
those venues may still be relevant, so this script reports coverage for both
"outdoors only" and "all roof types" rather than silently excluding domes.

International games (roughly 87 rows, per the `location == 'Neutral'`
nflverse field, all pre-flagged `in_scope=False` in
registry/reference/stadium_county_fips.csv) are reported as a separate
out-of-scope bucket, not silently dropped from the denominator.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/build_environmental_exposure_join.py \\
        --air-quality-snapshot data/raw/air_quality/<UTC> \\
        --drought-snapshot data/raw/drought/<UTC>

Writes data/processed/environmental_exposures/game_join.parquet (gitignored,
data/processed/**) plus prints per-season coverage tables to stdout, which
this session hand-transcribes into docs/environmental_exposures.md (no
automated doc-writing here, per this task's scope).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.provenance import sha256_file

REPO = Path(__file__).resolve().parents[1]
STADIUM_COUNTY_FIPS_PATH = REPO / "registry/reference/stadium_county_fips.csv"
EASTERN = ZoneInfo("America/New_York")
USDM_RELEASE_OFFSET_DAYS = 2
USDM_RELEASE_TIME = time(8, 30)
TUESDAY_CHECKPOINT_TIME = time(12, 0)
MAX_LIVE_AQI_CAPTURE_AGE = pd.Timedelta(hours=3)


def _eastern_local_to_utc(value: pd.Timestamp, *, clock: time) -> pd.Timestamp:
    """Attach Eastern wall-clock time to a date, then convert it to UTC.

    USDM states its release schedule in Eastern time, so using a fixed UTC
    offset would be wrong across daylight-saving transitions.
    """

    local = datetime.combine(value.date(), clock, tzinfo=EASTERN)
    return pd.Timestamp(local.astimezone(UTC))


def drought_release_timestamps(valid_start: pd.Series) -> pd.Series:
    """Official USDM availability: Thursday 08:30 ET after Tuesday validity."""

    release_dates = pd.to_datetime(valid_start) + timedelta(days=USDM_RELEASE_OFFSET_DAYS)
    return release_dates.map(lambda value: _eastern_local_to_utc(value, clock=USDM_RELEASE_TIME))


def tuesday_checkpoint_timestamps(tuesday_date: pd.Series) -> pd.Series:
    """The pool's Tuesday-noon ET weekly decision checkpoint, represented in UTC."""

    return pd.to_datetime(tuesday_date).map(
        lambda value: _eastern_local_to_utc(value, clock=TUESDAY_CHECKPOINT_TIME)
    )


def _latest(glob_pattern: str) -> Path:
    candidates = sorted(REPO.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"No path matches {glob_pattern}")
    return candidates[-1]


def _latest_schedules() -> Path:
    candidates = sorted(
        p for p in (REPO / "data/raw").glob("*/schedules.parquet") if p.parent.name[0].isdigit()
    )
    if not candidates:
        raise FileNotFoundError("No data/raw/<snapshot>/schedules.parquet found.")
    return candidates[-1]


def load_schedule() -> pd.DataFrame:
    path = _latest_schedules()
    df = pd.read_parquet(
        path,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "stadium",
            "roof",
            "surface",
            "location",
        ],
    )
    df["gameday"] = pd.to_datetime(df["gameday"])
    # Tuesday on/before gameday: Monday=0..Sunday=6, Tuesday=1.
    days_since_tuesday = (df["gameday"].dt.weekday - 1) % 7
    df["tuesday_date"] = df["gameday"] - pd.to_timedelta(days_since_tuesday, unit="D")
    df["decision_at_utc"] = tuesday_checkpoint_timestamps(df["tuesday_date"])
    return df


def load_stadium_counties() -> pd.DataFrame:
    df = pd.read_csv(STADIUM_COUNTY_FIPS_PATH, dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].where(~df["in_scope"], df["county_fips"].str.zfill(5))
    return df[
        ["stadium", "in_scope", "county_fips", "county_name", "state_code", "roof_values_seen"]
    ]


def asof_merge_aqi(games: pd.DataFrame, aqi: pd.DataFrame) -> pd.DataFrame:
    aqi = aqi.copy()
    aqi["aqi_date"] = pd.to_datetime(aqi.pop("date"))
    # A daily AQI summarizes the complete local calendar day. At a Tuesday
    # decision checkpoint, Tuesday's eventual daily value is future
    # information; the earliest safe archive proxy is therefore Monday.
    aqi["aqi_available_date"] = aqi["aqi_date"] + pd.Timedelta(days=1)
    aqi = aqi.sort_values("aqi_available_date")
    games_sorted = games.sort_values("tuesday_date")
    out_parts = []
    for fips, grp in games_sorted.groupby("county_fips", dropna=True):
        county_aqi = aqi[aqi["county_fips"] == fips]
        if county_aqi.empty:
            merged = grp.copy()
            merged["aqi"] = pd.NA
            merged["aqi_date"] = pd.NaT
            merged["aqi_category"] = pd.NA
        else:
            merged = pd.merge_asof(
                grp,
                county_aqi[["aqi_available_date", "aqi_date", "aqi", "category"]].rename(
                    columns={"category": "aqi_category"}
                ),
                left_on="tuesday_date",
                right_on="aqi_available_date",
                direction="backward",
            )
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True) if out_parts else games.copy()


def load_airnow_captures(root: Path) -> pd.DataFrame:
    """Load complete immutable AirNow snapshots after verifying manifest hashes."""

    captures = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        files = {entry["path"]: entry for entry in manifest.get("files", [])}
        required = {"source.dat", "stadium_aqi.parquet"}
        if set(files) != required:
            raise ValueError(f"Complete AirNow manifest lacks required files: {manifest_path}")
        for name, entry in files.items():
            artifact = manifest_path.parent / name
            if not artifact.is_file() or sha256_file(artifact) != entry.get("sha256"):
                raise ValueError(f"AirNow snapshot failed its SHA-256 check: {artifact}")
        data_path = manifest_path.parent / "stadium_aqi.parquet"
        captures.append(pd.read_parquet(data_path))
    if not captures:
        raise FileNotFoundError(f"No complete AirNow captures under {root}")
    return pd.concat(captures, ignore_index=True)


def asof_merge_live_aqi(games: pd.DataFrame, captures: pd.DataFrame) -> pd.DataFrame:
    """Expose an hourly AQI only after capture, and only for three hours."""

    live = captures.copy()
    live["available_at_utc"] = pd.to_datetime(live["available_at_utc"], utc=True)
    live["observed_at_utc"] = pd.to_datetime(live["observed_at_utc"], utc=True)
    live = live.sort_values(["available_at_utc", "stadium"]).drop_duplicates(
        ["county_fips", "available_at_utc"]
    )
    games_sorted = games.copy()
    games_sorted["decision_at_utc"] = pd.to_datetime(games_sorted["decision_at_utc"], utc=True)
    games_sorted = games_sorted.sort_values("decision_at_utc")
    out_parts = []
    live_columns = [
        "available_at_utc",
        "observed_at_utc",
        "aqi",
        "parameter",
        "aqs_site_id",
        "site_name",
    ]
    for fips, group in games_sorted.groupby("county_fips", dropna=True):
        county = live.loc[live["county_fips"] == fips, live_columns]
        merged = pd.merge_asof(
            group,
            county,
            left_on="decision_at_utc",
            right_on="available_at_utc",
            direction="backward",
        )
        out_parts.append(merged)
    if not out_parts:
        return games.copy()
    joined = pd.concat(out_parts, ignore_index=True).rename(
        columns={
            "available_at_utc": "live_aqi_available_at_utc",
            "observed_at_utc": "live_aqi_observed_at_utc",
            "aqi": "live_aqi",
            "parameter": "live_aqi_parameter",
            "aqs_site_id": "live_aqi_site_id",
            "site_name": "live_aqi_site_name",
        }
    )
    age = joined["decision_at_utc"] - joined["live_aqi_available_at_utc"]
    stale = age > MAX_LIVE_AQI_CAPTURE_AGE
    value_columns = [
        "live_aqi",
        "live_aqi_parameter",
        "live_aqi_site_id",
        "live_aqi_site_name",
        "live_aqi_observed_at_utc",
        "live_aqi_available_at_utc",
    ]
    joined.loc[stale, value_columns] = pd.NA
    return joined


def asof_merge_drought(games: pd.DataFrame, drought: pd.DataFrame) -> pd.DataFrame:
    drought = drought.copy()
    drought["available_at_utc"] = drought_release_timestamps(drought["valid_start"])
    drought = drought.sort_values("available_at_utc")
    games_sorted = games.copy()
    if "decision_at_utc" not in games_sorted:
        games_sorted["decision_at_utc"] = tuesday_checkpoint_timestamps(
            games_sorted["tuesday_date"]
        )
    games_sorted = games_sorted.sort_values("decision_at_utc")
    out_parts = []
    for fips, grp in games_sorted.groupby("county_fips", dropna=True):
        county_drought = drought[drought["county_fips"] == fips]
        if county_drought.empty:
            merged = grp.copy()
            for col in ["none", "d0", "d1", "d2", "d3", "d4", "valid_start", "valid_end"]:
                merged[f"drought_{col}"] = pd.NA
        else:
            cols = [
                "available_at_utc",
                "valid_start",
                "valid_end",
                "none",
                "d0",
                "d1",
                "d2",
                "d3",
                "d4",
            ]
            merged = pd.merge_asof(
                grp,
                county_drought[cols]
                .add_prefix("drought_")
                .rename(columns={"drought_available_at_utc": "drought_available_at_utc"}),
                left_on="decision_at_utc",
                right_on="drought_available_at_utc",
                direction="backward",
            )
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True) if out_parts else games.copy()


def primary_drought_category(row: pd.Series) -> str:
    if pd.isna(row.get("drought_d0")):
        return "no_data"
    # d0..d4 are CUMULATIVE percentages ("at least this severe"), per USDM convention.
    for level in ["d4", "d3", "d2", "d1", "d0"]:
        val = row.get(f"drought_{level}")
        if pd.notna(val) and val >= 50.0:
            return level.upper()
    return "none/mixed<50pct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air-quality-snapshot", type=Path, default=None)
    parser.add_argument("--drought-snapshot", type=Path, default=None)
    parser.add_argument(
        "--airnow-live-root",
        type=Path,
        default=None,
        help="Optional immutable hourly captures from capture_airnow_hourly.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data/processed/environmental_exposures/game_join.parquet",
    )
    args = parser.parse_args()

    aq_snapshot = args.air_quality_snapshot or _latest("data/raw/air_quality/*")
    drought_snapshot = args.drought_snapshot or _latest("data/raw/drought/*")
    print(f"Air quality snapshot: {aq_snapshot}")
    print(f"Drought snapshot: {drought_snapshot}")

    aqi = pd.read_parquet(aq_snapshot / "index.parquet")
    drought = pd.read_parquet(drought_snapshot / "index.parquet")

    games = load_schedule()
    stadium_counties = load_stadium_counties()
    games = games.merge(stadium_counties, on="stadium", how="left")

    in_scope_games = games[games["in_scope"] == True].copy()  # noqa: E712
    out_of_scope_games = games[games["in_scope"] != True].copy()  # noqa: E712

    joined = asof_merge_aqi(in_scope_games, aqi)
    if args.airnow_live_root is not None:
        joined = asof_merge_live_aqi(joined, load_airnow_captures(args.airnow_live_root))
    joined = asof_merge_drought(joined, drought)
    joined["drought_primary_category"] = joined.apply(primary_drought_category, axis=1)
    joined["is_dome_or_closed"] = joined["roof"].isin(["dome", "closed"])
    joined["is_outdoor_exposed"] = joined["roof"].isin(["outdoors", "open"])

    # `merge_asof(direction="backward")` silently carries the LAST available
    # archive value forward for any tuesday_date past the archive's own max
    # date (both archives stop at end-of-2025) -- e.g. a 2026 game's Tuesday
    # would match the same stale late-2025 AQI/drought reading rather than a
    # real 2026 measurement. Track staleness explicitly rather than letting a
    # match-found/match-missing boolean claim "100% coverage" when a chunk of
    # it is actually carried-forward stale data.
    joined["aqi_staleness_days"] = (joined["tuesday_date"] - joined["aqi_date"]).dt.days
    joined["drought_staleness_days"] = (
        joined["decision_at_utc"] - joined["drought_available_at_utc"]
    ).dt.total_seconds() / 86_400.0
    # AQI is daily, drought weekly; beyond ~10 days a match is archive
    # exhaustion carry-forward, not a real as-of-Tuesday reading.
    STALE_THRESHOLD_DAYS = 10
    joined["aqi_is_stale_carryforward"] = joined["aqi_staleness_days"] > STALE_THRESHOLD_DAYS
    joined["drought_is_stale_carryforward"] = joined["drought_staleness_days"] > (
        STALE_THRESHOLD_DAYS + 7
    )  # drought rows are weekly, so allow one extra week before flagging

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.out, index=False)

    out_of_scope_path = args.out.parent / "out_of_scope_games.parquet"
    out_of_scope_games.to_parquet(out_of_scope_path, index=False)

    print(f"\nWrote {args.out} ({len(joined)} in-scope games)")
    print(f"Wrote {out_of_scope_path} ({len(out_of_scope_games)} out-of-scope international games)")

    # --- Per-season coverage report ---
    joined["has_aqi"] = joined["aqi"].notna()
    joined["has_drought"] = joined["drought_d0"].notna()
    joined["has_fresh_aqi"] = joined["has_aqi"] & ~joined["aqi_is_stale_carryforward"]
    joined["has_fresh_drought"] = joined["has_drought"] & ~joined["drought_is_stale_carryforward"]

    by_season = (
        joined.groupby("season")
        .agg(
            n_games=("game_id", "count"),
            n_aqi_covered=("has_aqi", "sum"),
            n_aqi_fresh=("has_fresh_aqi", "sum"),
            n_drought_covered=("has_drought", "sum"),
            n_drought_fresh=("has_fresh_drought", "sum"),
            n_outdoor=("is_outdoor_exposed", "sum"),
            n_dome_closed=("is_dome_or_closed", "sum"),
        )
        .reset_index()
    )
    by_season["aqi_coverage_pct"] = (100 * by_season["n_aqi_covered"] / by_season["n_games"]).round(
        1
    )
    by_season["aqi_fresh_pct"] = (100 * by_season["n_aqi_fresh"] / by_season["n_games"]).round(1)
    by_season["drought_coverage_pct"] = (
        100 * by_season["n_drought_covered"] / by_season["n_games"]
    ).round(1)
    by_season["drought_fresh_pct"] = (
        100 * by_season["n_drought_fresh"] / by_season["n_games"]
    ).round(1)

    print("\n=== Per-season join coverage (in-scope domestic games only) ===")
    print(
        "'covered' = any archive match found; 'fresh' = match within staleness "
        "threshold (not archive-exhaustion carry-forward)"
    )
    print(
        by_season[
            [
                "season",
                "n_games",
                "aqi_coverage_pct",
                "aqi_fresh_pct",
                "drought_coverage_pct",
                "drought_fresh_pct",
                "n_outdoor",
                "n_dome_closed",
            ]
        ].to_string(index=False)
    )

    n_out_of_scope_by_season = out_of_scope_games.groupby("season")["game_id"].count()
    print("\n=== Out-of-scope international games by season ===")
    print(n_out_of_scope_by_season.to_string())

    overall = {
        "total_games": len(games),
        "in_scope_games": len(in_scope_games),
        "out_of_scope_international_games": len(out_of_scope_games),
        "aqi_coverage_pct_overall": round(100 * joined["has_aqi"].mean(), 1),
        "drought_coverage_pct_overall": round(100 * joined["has_drought"].mean(), 1),
        "outdoor_exposed_games": int(joined["is_outdoor_exposed"].sum()),
        "dome_or_closed_games": int(joined["is_dome_or_closed"].sum()),
    }
    print("\n=== Overall summary ===")
    print(json.dumps(overall, indent=2))

    print("\n=== Drought primary-category distribution (in-scope games) ===")
    print(joined["drought_primary_category"].value_counts().to_string())

    print("\n=== AQI category distribution (in-scope games) ===")
    print(joined["aqi_category"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

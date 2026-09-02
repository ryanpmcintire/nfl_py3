"""ENV-01: point-in-time PREGAME weather-FORECAST archive.

Fetches the archived GFS MOS extended ("MEX") text bulletin that existed at
Tuesday 12:00 ET of each NFL REG game's week, from the Iowa Environmental
Mesonet (IEM) MOS archive, and extracts the forecast temperature + wind speed
valid nearest to that game's actual kickoff hour.

Why MOS instead of NDFD GRIB2 (the route ``docs/weather_forecast_sourcing.md``
scouted first): **measured** this session, the IEM MOS JSON API
(``https://mesonet.agron.iastate.edu/api/1/mos.json``) serves plain station
bulletins with an explicit ``runtime`` (issuance) and ``ftime`` (forecast
valid time) on every row -- point-in-time by construction, no GRIB2 decoder
needed, no gridded lat/lon lookup needed (station-based). The GFS extended
model (``model=MEX``) issues at 00Z and 12Z, reaches +192h, and is archived
at IEM from **2020-07-12 onward** (measured: ``runtime=2020-09-08T00:00Z``
returns data; ``runtime=2018-12-04T00:00Z`` returns "no results" for the
same station) -- this is why ingestion is scoped to 2020-2025, not because
2020 was an arbitrary choice: it is exactly where this free archive starts,
matching the AWS NDFD 2020+ boundary independently found in the GRIB2 route.

Point-in-time discipline: every stored row carries the ACTUAL issuance
timestamp used (``issuance_runtime_utc``), which is walked strictly
BACKWARD from the Tuesday-noon-ET cutoff in 12-hour steps until a bulletin
with data is found -- never forward, never substituted with a later,
fresher issuance. A game whose cutoff has no bulletin within
``--max-lookback-steps`` steps is recorded with ``fetch_status`` describing
why, not silently dropped.

Station mapping: ``registry/reference/stadium_station_map.csv`` (built this
session), keyed on the schedules parquet's own ``stadium`` display-name
string (not ``stadium_id``, which is measurably unreliable for neutral-site
international games -- e.g. "Bernabeu" in Madrid is stamped with Atlanta's
own domestic stadium_id in the source data). International stadiums
(Wembley, Tottenham, Munich, Frankfurt, Mexico City, Sao Paulo, etc.) are
marked ``mappable=false``: **measured**, the GFS MOS network has zero
non-US-domestic stations (EGLL/EDDF/EDDM/MMMX/SBGR/CYYZ all return "no
results" for any runtime).

Usage (from repo root, locked env):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_forecast_archive.py \\
        --start-season 2020 --end-season 2025

    # Resume an interrupted run (skips game_ids already fetched):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_forecast_archive.py \\
        --start-season 2020 --end-season 2025 \\
        --resume-from data/raw/forecast_archive/<timestamp>

    # Pool-decision cutoff: min(kickoff, Sunday 16:00 America/New_York).
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_forecast_archive.py \\
        --start-season 2009 --end-season 2025 --cutoff-mode pool_decision

Writes ``data/raw/forecast_archive/<UTC timestamp>/forecasts.parquet`` (one
row per REG game in range) plus a gitignored ``manifest.json`` (coverage
counts, source, delay policy, elapsed time -- ``data/raw/**`` is gitignored
per .gitignore, verified).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.nfl_week import pool_decision_cutoff  # noqa: E402
from nfl_ats.provenance import sha256_file  # noqa: E402

MOS_API = "https://mesonet.agron.iastate.edu/api/1/mos.json"
# Model choice depends on how far the decision cutoff sits from kickoff:
#   MEX (GFS MOS Extended, issues 00Z/12Z, +192h range, IEM archive from
#     2020-07-12 -- measured) is required for tuesday_noon, whose cutoff is
#     typically ~120-144h before a Sunday kickoff.
#   GFS (GFS MOS "short-range", issues 00Z/12Z/06Z/18Z, ~+69h range) is used
#     for kickoff_nearest instead of MEX: **measured** this session, MEX's
#     finest granularity near its OWN issuance time starts far from the run
#     (mean forecast-valid gap from kickoff 18.5h, min 12.5h, across the
#     2024 kickoff_nearest pilot -- MEX is built for days-out extended
#     guidance, not near-term), whereas GFS returns rows starting at
#     issuance+6h in 3h steps -- e.g. a 2024-09-08T12:00Z GFS run's first row
#     is valid 2024-09-08T18:00Z, exactly +6h. GFS's IEM archive also
#     measurably reaches further back than MEX: present for
#     runtime=2005-01-01T00:00Z and 2009-09-01T00:00Z, absent for
#     2003-01-01T00:00Z (KDFW probes) -- comfortably covering this project's
#     full 2009-2025 window for a near-kickoff forecast, free and instant,
#     with no NCEI HAS/AIRS order needed for that use case.
MOS_MODEL_BY_CUTOFF_MODE = {
    "tuesday_noon": "MEX",
    "kickoff_nearest": "GFS",
    "pool_decision": "GFS",
}
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
DELAY_SECONDS_DEFAULT = 0.3
MAX_LOOKBACK_STEPS_DEFAULT = 10  # 10 * 12h = 5 days back from the Tuesday-noon-ET cutoff
KNOTS_TO_MPH = 1.15078
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# 1. Population: NFL REG games in range, joined to kickoff + stadium/station
# ---------------------------------------------------------------------------


def _latest(glob_pattern: str) -> Path:
    candidates = sorted((REPO / "data/raw").glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/{glob_pattern} snapshot found")
    return candidates[-1]


DEFAULT_SCHEDULES = _latest("*/schedules.parquet")
DEFAULT_GAME_FEATURES = REPO / "data/processed/game_features.parquet"
DEFAULT_STATION_MAP = REPO / "registry/reference/stadium_station_map.csv"


def load_population(
    schedules_path: Path,
    game_features_path: Path,
    station_map_path: Path,
    *,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    sched = pd.read_parquet(
        schedules_path,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "stadium",
            "roof",
            "temp",
            "wind",
        ],
    )
    df = sched.loc[sched["game_type"] == "REG"].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df = df.loc[df["season"].between(start_season, end_season)].reset_index(drop=True)

    kickoff = pd.read_parquet(game_features_path, columns=["game_id", "kickoff"])
    df = df.merge(kickoff, on="game_id", how="left", validate="1:1")
    missing_kickoff = int(df["kickoff"].isna().sum())
    if missing_kickoff:
        print(
            f"WARNING: {missing_kickoff} games in range have no kickoff timestamp in "
            f"{game_features_path} (likely not yet played/scheduled-only) -- dropped",
            file=sys.stderr,
        )
        df = df.loc[df["kickoff"].notna()].reset_index(drop=True)

    station_map = pd.read_csv(station_map_path)
    df = df.merge(station_map[["stadium", "icao_station", "mappable"]], on="stadium", how="left")
    # A stadium genuinely ABSENT from the reference table leaves BOTH
    # icao_station and mappable null after the merge; a deliberately
    # unmappable international stadium is present in the table with
    # icao_station null but mappable=False (not null) -- only the former is
    # an error to fail closed on.
    unmapped_stadiums = sorted(df.loc[df["mappable"].isna(), "stadium"].unique().tolist())
    if unmapped_stadiums:
        raise ValueError(
            f"{len(unmapped_stadiums)} stadium name(s) in the population are not in "
            f"{station_map_path} at all: {unmapped_stadiums}. Add them before running."
        )
    df["mappable"] = df["mappable"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# 2. Tuesday-noon-ET cutoff and MOS fetch
# ---------------------------------------------------------------------------


def tuesday_noon_et_cutoff_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """Most recent Tuesday <= the kickoff's ET calendar date, at 12:00 ET,
    returned as a UTC timestamp. Monday=0 ... Tuesday=1 ... Sunday=6.
    """

    kickoff_et = kickoff_utc.tz_convert(ET)
    et_date: date = kickoff_et.date()
    days_since_tuesday = (et_date.weekday() - 1) % 7
    tuesday_date = et_date - timedelta(days=days_since_tuesday)
    tuesday_noon_et = datetime(
        tuesday_date.year, tuesday_date.month, tuesday_date.day, 12, 0, tzinfo=ET
    )
    return pd.Timestamp(tuesday_noon_et).tz_convert(UTC)


def kickoff_nearest_cutoff_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """The decision timestamp for the "closest-before-kickoff" cutoff mode:
    kickoff itself. ``candidate_runtimes`` floors this to the most recent
    00Z/12Z MOS issuance cycle at or before kickoff and walks strictly
    backward from there, so this never selects a bulletin issued after
    kickoff -- point-in-time discipline is enforced by the walk, not by this
    function.
    """

    return kickoff_utc


def pool_decision_cutoff_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """Actual pool deadline: kickoff or Sunday 16:00 ET, whichever is first."""

    return pd.Timestamp(pool_decision_cutoff(kickoff_utc.to_pydatetime()))


def decision_cutoff_utc(kickoff_utc: pd.Timestamp, cutoff_mode: str) -> pd.Timestamp:
    """Resolve a declared archive mode to its decision timestamp."""

    if cutoff_mode == "tuesday_noon":
        return tuesday_noon_et_cutoff_utc(kickoff_utc)
    if cutoff_mode == "kickoff_nearest":
        return kickoff_nearest_cutoff_utc(kickoff_utc)
    if cutoff_mode == "pool_decision":
        return pool_decision_cutoff_utc(kickoff_utc)
    raise ValueError(f"Unknown cutoff mode: {cutoff_mode}")


def floor_to_12h_utc(dt: pd.Timestamp) -> datetime:
    hour = 12 if dt.hour >= 12 else 0
    return datetime(dt.year, dt.month, dt.day, hour, 0, tzinfo=UTC)


def candidate_runtimes(cutoff_utc: pd.Timestamp, max_steps: int) -> list[datetime]:
    start = floor_to_12h_utc(cutoff_utc)
    return [start - timedelta(hours=12 * i) for i in range(max_steps)]


class MosFetchError(RuntimeError):
    pass


def fetch_mos_bulletin(
    station: str,
    runtime_utc: datetime,
    *,
    model: str,
    timeout: float = 20.0,
    retries: int = 2,
) -> list[dict[str, Any]]:
    """Return the ``data`` rows for one station/runtime, or [] if IEM has no
    bulletin for that exact runtime (a normal, expected outcome to walk past,
    not an error). Raises MosFetchError on a genuine transport failure after
    retries (so the caller can distinguish "no bulletin" from "IEM is down").
    """

    runtime_str = runtime_utc.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{MOS_API}?station={station}&model={model}&runtime={runtime_str}"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.load(resp)
            if "data" in payload:
                return list(payload["data"])
            return []  # "no results" detail response -> no bulletin, not an error
        except urllib.error.HTTPError as exc:
            # IEM uses HTTP 404 for an expected "no bulletin at this exact
            # station/runtime" result in parts of the older archive.  It is a
            # miss in the declared backward search, not a transport failure.
            if exc.code == 404:
                return []
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    raise MosFetchError(f"{station} {runtime_str}: {last_exc}")


def nearest_row(rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp) -> dict[str, Any] | None:
    if not rows:
        return None
    target = kickoff_utc.to_pydatetime()
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)

    def gap(row: dict[str, Any]) -> float:
        ftime = datetime.fromisoformat(row["ftime_utc"]).replace(tzinfo=UTC)
        return abs((ftime - target).total_seconds())

    return min(rows, key=gap)


def nearest_row_with_field(
    rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp, field: str
) -> dict[str, Any] | None:
    """Like ``nearest_row``, restricted to rows where ``field`` is non-null.

    Added for the 2009-2019 backward extension (2026-08-20 session): GFS MOS
    precipitation-probability fields (``p06``/``p12``) are only populated on a
    subset of rows within a bulletin (measured: every OTHER 3h row, i.e. the
    6h-boundary rows), so the plain ``nearest_row`` pick (nearest by valid time
    to kickoff, ANY field) frequently lands on a row where ``p06``/``p12`` are
    both null even though a nearby row in the SAME already-fetched bulletin has
    them. This does a second, field-restricted nearest-by-valid-time pick over
    the SAME rows already returned by ``fetch_mos_bulletin`` -- no extra HTTP
    call, no relaxation of the point-in-time issuance walk (that discipline is
    still enforced by the caller's bulletin selection, unchanged).
    """

    candidates = [row for row in rows if row.get(field) is not None]
    return nearest_row(candidates, kickoff_utc)


def fetch_one_game(
    station: str,
    kickoff_utc: pd.Timestamp,
    cutoff_utc: pd.Timestamp,
    *,
    model: str,
    max_lookback_steps: int,
    delay_seconds: float,
) -> dict[str, Any]:
    for runtime_utc in candidate_runtimes(cutoff_utc, max_lookback_steps):
        try:
            rows = fetch_mos_bulletin(station, runtime_utc, model=model)
        except MosFetchError as exc:
            time.sleep(delay_seconds)
            return {
                "fetch_status": "transport_error",
                "fetch_error": str(exc),
                "issuance_runtime_utc": None,
            }
        time.sleep(delay_seconds)
        if rows:
            row = nearest_row(rows, kickoff_utc)
            assert row is not None
            issuance = pd.to_datetime(row.get("runtime_utc"), utc=True, errors="coerce")
            if pd.isna(issuance) or issuance > cutoff_utc:
                return {
                    "fetch_status": "invalid_issuance_timestamp",
                    "fetch_error": (
                        f"bulletin runtime {row.get('runtime_utc')!r} is invalid or later than "
                        f"decision cutoff {cutoff_utc}"
                    ),
                    "issuance_runtime_utc": row.get("runtime_utc"),
                }
            ftime_actual = row["ftime_utc"]
            tmp = row.get("tmp")
            wsp = row.get("wsp")
            # Precip probability (p06/p12, percent) is sparse within a bulletin
            # (populated only on 6h-boundary rows) -- picked via a SEPARATE
            # field-restricted nearest-by-valid-time search over the same
            # already-fetched rows, not the plain nearest_row pick above, which
            # would silently return null on most games. p06 (6h prob) preferred
            # over p12 (12h prob, coarser window) when both are available.
            precip_row = nearest_row_with_field(rows, kickoff_utc, "p06") or nearest_row_with_field(
                rows, kickoff_utc, "p12"
            )
            precip_prob_pct: float | None = None
            precip_field_used: str | None = None
            precip_valid_utc: str | None = None
            if precip_row is not None:
                if precip_row.get("p06") is not None:
                    precip_prob_pct = float(precip_row["p06"])
                    precip_field_used = "p06"
                elif precip_row.get("p12") is not None:
                    precip_prob_pct = float(precip_row["p12"])
                    precip_field_used = "p12"
                precip_valid_utc = precip_row["ftime_utc"]
            return {
                "fetch_status": "ok",
                "fetch_error": None,
                "issuance_runtime_utc": row["runtime_utc"],
                "forecast_valid_utc": ftime_actual,
                "forecast_temp_f": float(tmp) if tmp is not None else None,
                "forecast_wind_knots": float(wsp) if wsp is not None else None,
                "forecast_wind_mph": (float(wsp) * KNOTS_TO_MPH if wsp is not None else None),
                "forecast_precip_prob_pct": precip_prob_pct,
                "forecast_precip_prob_field": precip_field_used,
                "forecast_precip_prob_valid_utc": precip_valid_utc,
                "lookback_steps_used": candidate_runtimes(cutoff_utc, max_lookback_steps).index(
                    runtime_utc
                ),
            }
    return {
        "fetch_status": "no_bulletin_within_lookback",
        "fetch_error": None,
        "issuance_runtime_utc": None,
    }


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------


def load_resume_cache(
    resume_from: Path | None,
    *,
    cutoff_mode: str,
    mos_model: str,
) -> dict[str, dict[str, Any]]:
    if resume_from is None:
        return {}
    jsonl_path = resume_from / "results.jsonl"
    config_path = resume_from / "run_config.json"
    if not config_path.is_file():
        config_path = resume_from / "manifest.json"
    if not jsonl_path.is_file() or not config_path.is_file():
        raise ValueError(
            "Resume archive requires results.jsonl plus run_config.json or manifest.json"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("cutoff_mode") != cutoff_mode or config.get("mos_model") != mos_model:
        raise ValueError(
            "Resume archive cutoff_mode/mos_model does not match this run: "
            f"expected {cutoff_mode}/{mos_model}"
        )
    cache: dict[str, dict[str, Any]] = {}
    retryable_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("cutoff_mode") != cutoff_mode:
                raise ValueError(
                    f"Resume row {record.get('game_id')} has cutoff_mode="
                    f"{record.get('cutoff_mode')!r}, expected {cutoff_mode!r}"
                )
            game_id = record["game_id"]
            if record.get("fetch_status") in {"ok", "unmappable_international_stadium"}:
                cache[game_id] = record
            else:
                # The JSONL is an append-only attempt log. A later failed
                # attempt must invalidate any earlier cached row for this game
                # so resume retries it rather than silently preserving failure.
                cache.pop(game_id, None)
                retryable_rows += 1
    print(
        f"Resumed {len(cache)} completed games from {jsonl_path}; "
        f"{retryable_rows} failed attempt row(s) remain retryable"
    )
    return cache


def rewrite_resume_cache(jsonl_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """Replace an attempt log with its validated one-row-per-game terminal cache."""

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in cache.values():
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--schedules", type=Path, default=DEFAULT_SCHEDULES)
    parser.add_argument("--game-features", type=Path, default=DEFAULT_GAME_FEATURES)
    parser.add_argument("--station-map", type=Path, default=DEFAULT_STATION_MAP)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--cutoff-mode",
        choices=list(MOS_MODEL_BY_CUTOFF_MODE),
        default="tuesday_noon",
        help=(
            "tuesday_noon (default): Tuesday 12:00 ET grading-line context; "
            "kickoff_nearest: kickoff itself (legacy research mode); pool_decision: "
            "the real pick deadline, min(kickoff, Sunday 16:00 America/New_York). "
            "Every mode walks strictly backward from its cutoff."
        ),
    )
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS_DEFAULT)
    parser.add_argument("--max-lookback-steps", type=int, default=MAX_LOOKBACK_STEPS_DEFAULT)
    parser.add_argument(
        "--model",
        default=None,
        choices=["AVN", "GFS", "ETA", "NAM", "NBS", "NBE", "ECM", "LAV", "MEX"],
        help=(
            "Override the MOS model IEM is queried with (default: chosen from "
            "--cutoff-mode via MOS_MODEL_BY_CUTOFF_MODE -- MEX for tuesday_noon, GFS "
            "for kickoff_nearest and pool_decision)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N games (testing)"
    )
    args = parser.parse_args()

    mos_model = args.model or MOS_MODEL_BY_CUTOFF_MODE[args.cutoff_mode]

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "data" / "raw" / "forecast_archive" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"

    print(
        f"=== loading population {args.start_season}-{args.end_season} "
        f"(cutoff_mode={args.cutoff_mode}, model={mos_model}) ==="
    )
    population = load_population(
        args.schedules,
        args.game_features,
        args.station_map,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    print(f"REG games in range: {len(population)}")
    print(f"  mappable stadium (domestic): {int(population['mappable'].sum())}")
    print(f"  unmappable (international):  {int((~population['mappable']).sum())}")

    cache = load_resume_cache(
        args.resume_from,
        cutoff_mode=args.cutoff_mode,
        mos_model=mos_model,
    )
    atomic_json(
        {
            "cutoff_mode": args.cutoff_mode,
            "mos_model": mos_model,
            "start_season": args.start_season,
            "end_season": args.end_season,
        },
        output_dir / "run_config.json",
    )
    if args.resume_from is not None:
        # Rewrite from the validated terminal cache even when resuming in
        # place. This removes failed/superseded attempts and guarantees the
        # final parquet has exactly one row per game.
        rewrite_resume_cache(jsonl_path, cache)

    rows = population if args.limit is None else population.head(args.limit)
    n_total = len(rows)
    n_fetched_this_run = 0
    n_skipped_international = 0
    n_reused_from_cache = 0

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for i, game in enumerate(rows.itertuples(index=False), start=1):
            game_id = game.game_id
            if game_id in cache:
                n_reused_from_cache += 1
                continue

            kickoff_utc = pd.Timestamp(game.kickoff)
            if kickoff_utc.tzinfo is None:
                kickoff_utc = kickoff_utc.tz_localize(UTC)
            cutoff_utc = decision_cutoff_utc(kickoff_utc, args.cutoff_mode)

            if not game.mappable:
                record = {
                    "game_id": game_id,
                    "season": int(game.season),
                    "week": int(game.week),
                    "home_team": game.home_team,
                    "stadium": game.stadium,
                    "icao_station": None,
                    "kickoff_utc": str(kickoff_utc),
                    "cutoff_mode": args.cutoff_mode,
                    "decision_cutoff_utc": str(cutoff_utc),
                    "fetch_status": "unmappable_international_stadium",
                    "fetch_error": None,
                    "issuance_runtime_utc": None,
                    "forecast_valid_utc": None,
                    "forecast_temp_f": None,
                    "forecast_wind_knots": None,
                    "forecast_wind_mph": None,
                    "forecast_precip_prob_pct": None,
                    "forecast_precip_prob_field": None,
                    "forecast_precip_prob_valid_utc": None,
                    "actual_temp_f": float(game.temp) if pd.notna(game.temp) else None,
                    "actual_wind_mph": float(game.wind) if pd.notna(game.wind) else None,
                    "roof": game.roof,
                }
                handle.write(json.dumps(record) + "\n")
                n_skipped_international += 1
                continue

            fetch_result = fetch_one_game(
                game.icao_station,
                kickoff_utc,
                cutoff_utc,
                model=mos_model,
                max_lookback_steps=args.max_lookback_steps,
                delay_seconds=args.delay,
            )
            record = {
                "game_id": game_id,
                "season": int(game.season),
                "week": int(game.week),
                "home_team": game.home_team,
                "stadium": game.stadium,
                "icao_station": game.icao_station,
                "kickoff_utc": str(kickoff_utc),
                "cutoff_mode": args.cutoff_mode,
                "decision_cutoff_utc": str(cutoff_utc),
                "actual_temp_f": float(game.temp) if pd.notna(game.temp) else None,
                "actual_wind_mph": float(game.wind) if pd.notna(game.wind) else None,
                "roof": game.roof,
                **fetch_result,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            n_fetched_this_run += 1

            if i % 50 == 0 or i == n_total:
                elapsed = time.time() - started
                rate = n_fetched_this_run / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{i}/{n_total}] fetched_this_run={n_fetched_this_run} "
                    f"reused_cache={n_reused_from_cache} skipped_intl={n_skipped_international} "
                    f"elapsed={elapsed:.0f}s rate={rate:.2f} games/sec"
                )

    # Assemble the final parquet from the full jsonl (cache + this run).
    all_records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    result_df = pd.DataFrame.from_records(all_records)
    parquet_path = output_dir / "forecasts.parquet"
    atomic_parquet(result_df, parquet_path)

    status_counts = {
        str(status): int(count)
        for status, count in result_df["fetch_status"].value_counts().items()
    }
    n_ok = status_counts.get("ok", 0)
    n_unmappable = int((result_df["fetch_status"] == "unmappable_international_stadium").sum())

    elapsed_total = time.time() - started
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source": (
            f"Iowa Environmental Mesonet MOS archive "
            f"(mesonet.agron.iastate.edu/api/1/mos.json), model={mos_model}"
        ),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "cutoff_mode": args.cutoff_mode,
        "mos_model": mos_model,
        "n_games_in_population": len(population),
        "n_rows_written": len(result_df),
        "n_fetched_this_run": n_fetched_this_run,
        "n_reused_from_cache": n_reused_from_cache,
        "files": {
            "forecasts.parquet": {
                "rows": len(result_df),
                "sha256": sha256_file(parquet_path),
            },
        },
        "inputs": {
            "station_map": {
                "path": str(args.station_map),
                "sha256": sha256_file(args.station_map),
            },
            "schedules": {
                "path": str(args.schedules),
                "sha256": sha256_file(args.schedules),
            },
            "game_features": {
                "path": str(args.game_features),
                "sha256": sha256_file(args.game_features),
            },
        },
        "fetch_status_counts": status_counts,
        "coverage_fraction_of_domestic": (
            n_ok / (len(result_df) - n_unmappable) if (len(result_df) - n_unmappable) > 0 else None
        ),
        "delay_seconds_between_requests": args.delay,
        "max_lookback_steps": args.max_lookback_steps,
        "elapsed_seconds_this_run": elapsed_total,
        "games_per_second_this_run": (
            n_fetched_this_run / elapsed_total if elapsed_total > 0 else None
        ),
        "selection_policy": (
            "min(kickoff, Sunday 16:00 America/New_York in the game's "
            "Tuesday-through-Monday NFL week)"
            if args.cutoff_mode == "pool_decision"
            else args.cutoff_mode
        ),
        "point_in_time_note": (
            "Each row's issuance_runtime_utc is the ACTUAL MOS bulletin cycle used, walked "
            "strictly backward from decision_cutoff_utc (cutoff_mode="
            f"{args.cutoff_mode}: "
            + (
                "Tuesday 12:00 ET of the game's week, the pool-relevant decision time"
                if args.cutoff_mode == "tuesday_noon"
                else (
                    "the real pool deadline: min(kickoff, Sunday 16:00 America/New_York)"
                    if args.cutoff_mode == "pool_decision"
                    else (
                        "kickoff itself, floored to the last 00Z/12Z MOS cycle at-or-before kickoff"
                    )
                )
            )
            + ") in 12h steps until a non-empty bulletin was found; never substituted with a "
            "later/fresher issuance."
        ),
    }
    atomic_json(manifest, output_dir / "manifest.json")
    print(f"\nWrote {parquet_path} ({len(result_df)} rows)")
    print(f"Wrote {output_dir / 'manifest.json'}")
    print(json.dumps(manifest["fetch_status_counts"], indent=2))
    if n_fetched_this_run:
        print(
            f"\nRate this run: {n_fetched_this_run / elapsed_total:.2f} games/sec "
            f"({elapsed_total:.0f}s for {n_fetched_this_run} games)"
        )


if __name__ == "__main__":
    main()

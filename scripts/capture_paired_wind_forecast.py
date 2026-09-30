from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.io import atomic_bytes, atomic_json, atomic_parquet
from nfl_ats.nfl_week import pool_decision_cutoff

ET = ZoneInfo("America/New_York")
ROOT = Path("data/raw/paired_wind_forecast")
MOS_API = "https://mesonet.agron.iastate.edu/api/1/mos.json"
PRODUCTS = ("GFS", "MEX")
WINDOW = timedelta(minutes=90)
MAX_VALID_GAP = timedelta(hours=6)


def utc(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing forecast timestamp")
    return (
        stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
    ).to_pydatetime()


def cycle_start(now: datetime) -> datetime:
    local = now.astimezone(ET)
    day = local.date() - timedelta(days=(local.weekday() - 1) % 7)
    return datetime(day.year, day.month, day.day, 12, 5, tzinfo=ET)


def load_games(now: datetime, schedules: Path, station_map: Path) -> pd.DataFrame:
    games = pd.read_parquet(
        schedules,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "gametime",
            "stadium",
            "roof",
        ],
    )
    cycle = cycle_start(now)
    days = pd.to_datetime(games["gameday"])
    games = games.loc[
        games["game_type"].eq("REG")
        & days.ge(pd.Timestamp(cycle.date()))
        & days.lt(pd.Timestamp(cycle.date() + timedelta(days=7)))
    ].copy()
    games["kickoff"] = (
        pd.to_datetime(
            games["gameday"].astype(str) + " " + games["gametime"].astype(str), errors="raise"
        )
        .dt.tz_localize(ET)
        .dt.tz_convert(UTC)
    )
    games["deadline"] = games["kickoff"].map(lambda value: pool_decision_cutoff(value))
    stations = pd.read_csv(station_map, usecols=["stadium", "icao_station", "mappable"])
    return games.merge(stations, on="stadium", how="left", validate="many_to_one")


def tuesday_rows(root: Path, cycle: str) -> dict[tuple[str, ...], dict[str, Any]]:
    found: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted((root / cycle / "tuesday").glob("*/forecasts.parquet")):
        for row in pd.read_parquet(path).to_dict("records"):
            if row.get("pair_eligible") is not True:
                continue
            key = tuple(str(row[name]) for name in ("game_id", "product_id", "station", "kickoff"))
            if key not in found or row["captured_at_utc"] < found[key]["captured_at_utc"]:
                found[key] = row
    return found


def fetch_forecast(
    station: str,
    product: str,
    kickoff: datetime,
    target: datetime | None,
    output: Path | None,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(UTC)
    cycle_hours = 6 if product == "GFS" else 12
    latest = now.replace(
        hour=(now.hour // cycle_hours) * cycle_hours, minute=0, second=0, microsecond=0
    )
    status = "no_matching_forecast"
    for step in range(6):
        runtime = latest - timedelta(hours=cycle_hours * step)
        query = urllib.parse.urlencode(
            {"station": station, "model": product, "runtime": runtime.strftime("%Y-%m-%dT%H:%MZ")}
        )
        url = f"{MOS_API}?{query}"
        started = datetime.now(UTC)
        request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-paired-wind/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read()
                received = datetime.now(UTC)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                time.sleep(0.3)
                continue
            raise
        receipt = {
            "source_url": url,
            "product_id": product,
            "station": station,
            "requested_runtime_utc": runtime.isoformat(),
            "request_started_at_utc": started.isoformat(),
            "captured_at_utc": received.isoformat(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "raw_file": f"receipt_{len(receipts):03d}.json",
        }
        if output is not None:
            atomic_bytes(output / receipt["raw_file"], body)
        receipts.append(receipt)
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("MOS response lacks a data array")
        candidates = []
        for row in payload["data"]:
            wind = row.get("wsp")
            if wind is None or not math.isfinite(float(wind)) or float(wind) < 0:
                continue
            valid = utc(row["ftime_utc"])
            issued = utc(row["runtime_utc"])
            if issued != runtime or issued > started or issued > valid:
                continue
            if target is not None and valid != target:
                continue
            if abs(valid - kickoff) > MAX_VALID_GAP:
                continue
            candidates.append((abs((valid - kickoff).total_seconds()), valid, row))
        time.sleep(0.3)
        if candidates:
            _, valid, row = min(candidates, key=lambda item: (item[0], item[1]))
            return {
                "fetch_status": "ok",
                "issuance_runtime_utc": utc(row["runtime_utc"]).isoformat(),
                "forecast_valid_utc": valid.isoformat(),
                "forecast_wind_knots": float(row["wsp"]),
                "forecast_wind_mph": float(row["wsp"]) * 1.15078,
                **receipt,
            }
        if payload["data"]:
            status = "target_outside_product_horizon_or_missing_wind"
    return {"fetch_status": status}


def capture(args: argparse.Namespace) -> int:
    started = datetime.now(UTC)
    cycle = cycle_start(started)
    cycle_key = cycle.date().isoformat()
    paths = sorted(Path("data/raw").glob("*/schedules.parquet"))
    schedules = args.schedules or (paths[-1] if paths else None)
    if schedules is None:
        raise FileNotFoundError("No local schedules.parquet snapshot")
    games = load_games(started, schedules, args.station_map)
    games = games.loc[games["deadline"] > started]
    if args.game_id:
        games = games.loc[games["game_id"].eq(args.game_id)]
        if games.empty:
            raise ValueError("Requested game is not upcoming in this Tuesday cycle")
    if args.phase == "tuesday":
        window_start = cycle.astimezone(UTC)
        window_end = window_start + timedelta(minutes=15)
        if not args.dry and not window_start <= started < window_end:
            print("SKIP: outside Tuesday 12:05-12:20 ET; no late checkpoint is fabricated")
            return 0
    elif not args.dry:
        games = games.loc[games["deadline"] - WINDOW <= started]
    if games.empty:
        print("SKIP: no games in the capture window")
        return 0
    baseline = tuesday_rows(args.output_root, cycle_key) if args.phase == "deadline" else {}
    output = None
    if not args.dry:
        output = args.output_root / cycle_key / args.phase / started.strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True, exist_ok=False)
    rows = []
    receipts: list[dict[str, Any]] = []
    source_error = None
    for game in games.itertuples(index=False):
        for product in PRODUCTS:
            row = {
                "game_id": game.game_id,
                "season": int(game.season),
                "week": int(game.week),
                "stadium": game.stadium,
                "roof": None if pd.isna(game.roof) else game.roof,
                "station": None if pd.isna(game.icao_station) else game.icao_station,
                "kickoff": game.kickoff.isoformat(),
                "pool_deadline_utc": game.deadline.isoformat(),
                "phase": args.phase,
                "product_id": product,
                "source": "IEM MOS",
                "pair_eligible": False,
            }
            if pd.isna(game.icao_station) or pd.isna(game.mappable) or not game.mappable:
                row["fetch_status"] = "unmapped_station"
                rows.append(row)
                continue
            prior = baseline.get((game.game_id, product, game.icao_station, row["kickoff"]))
            if source_error:
                row["fetch_status"] = "not_attempted_after_source_error"
                rows.append(row)
                continue
            try:
                row.update(
                    fetch_forecast(
                        game.icao_station,
                        product,
                        game.kickoff.to_pydatetime(),
                        utc(prior["forecast_valid_utc"]) if prior else None,
                        output,
                        receipts,
                    )
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                source_error = f"{type(exc).__name__}: {exc}"
                row.update(fetch_status="source_error", error=source_error)
            if row["fetch_status"] == "ok":
                received = utc(row["captured_at_utc"])
                if args.phase == "tuesday":
                    timely = window_start <= received < min(window_end, game.deadline)
                    row["pair_eligible"] = timely
                    row["pair_status"] = "tuesday_ready" if timely else "outside_tuesday_window"
                else:
                    timely = game.deadline - WINDOW <= received < game.deadline
                    ordered = bool(
                        prior
                        and utc(prior["captured_at_utc"]) < received
                        and utc(prior["issuance_runtime_utc"]) <= utc(row["issuance_runtime_utc"])
                    )
                    row["pair_eligible"] = timely and ordered
                    row["pair_status"] = (
                        "paired" if timely and ordered else "late_or_missing_comparable_tuesday"
                    )
                    if prior:
                        row["tuesday_captured_at_utc"] = prior["captured_at_utc"]
                        row["tuesday_issuance_runtime_utc"] = prior["issuance_runtime_utc"]
                        row["tuesday_forecast_wind_mph"] = prior["forecast_wind_mph"]
                        row["tuesday_raw_sha256"] = prior["sha256"]
                row["pair_key"] = "|".join(
                    str(row[key])
                    for key in ("game_id", "product_id", "station", "kickoff", "forecast_valid_utc")
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "phase": args.phase,
        "dry": args.dry,
        "games": len(games),
        "rows": len(rows),
        "receipts": len(receipts),
        "eligible_product_rows": int(frame["pair_eligible"].sum()),
        "statuses": frame["fetch_status"].value_counts().to_dict(),
        "output": str(output) if output else None,
    }
    if output is not None:
        atomic_parquet(output / "forecasts.parquet", frame)
        atomic_json(
            output / "manifest.json",
            {
                **summary,
                "schema_version": 1,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "schedules": str(schedules),
                "schedules_sha256": hashlib.sha256(schedules.read_bytes()).hexdigest(),
                "station_map": str(args.station_map),
                "station_map_sha256": hashlib.sha256(args.station_map.read_bytes()).hexdigest(),
                "receipts": receipts,
            },
        )
    print(json.dumps(summary, sort_keys=True))
    if source_error:
        print(f"SOURCE ERROR: {source_error}")
        return 1
    return int(
        not frame["fetch_status"].eq("ok").any()
        and not frame["fetch_status"].eq("unmapped_station").all()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture comparable prospective MOS wind forecasts"
    )
    parser.add_argument("--phase", choices=("tuesday", "deadline"), required=True)
    parser.add_argument("--schedules", type=Path)
    parser.add_argument(
        "--station-map", type=Path, default=Path("registry/reference/stadium_station_map.csv")
    )
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--game-id", help="Limit a source check to one upcoming game")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Fetch upcoming forecasts without saving; retain actual timing eligibility",
    )
    return capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nfl_ats.forecast_cold_visitor_tilt_overlay import MosFetchError, fetch_mos_bulletin
from nfl_ats.nfl_week import pool_decision_cutoff, week_cycle_sunday

ROOT = Path(__file__).resolve().parents[1]
SEASON = 2026
BULLETIN_ROOT = ROOT / "data" / "raw" / "forecast_archive" / "mos_bulletins"
STATION_MAP = ROOT / "registry" / "reference" / "stadium_station_map.csv"
OPEN_AIR_ROOFS = ("outdoors", "open")
MODEL_CYCLE_HOURS = {"GFS": 6, "MEX": 12}
LOOKBACK_CYCLES = 4
GFS_MOS_PROJECTION_STEP = pd.Timedelta(hours=3)
KNOTS_TO_MPH = 1.15078


def load_schedule() -> pd.DataFrame:
    path = sorted(glob.glob(str(ROOT / "data" / "raw" / "2026*" / "schedules.parquet")))[-1]
    schedule = pd.read_parquet(path)
    schedule = schedule[(schedule["season"] == SEASON) & (schedule["game_type"] == "REG")].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    return schedule


def resolve_week(schedule: pd.DataFrame, today: date) -> int:
    days = pd.to_datetime(schedule["gameday"]).dt.date
    cycle = days.map(week_cycle_sunday)
    last_cycle = cycle.groupby(schedule["week"]).max()
    open_weeks = last_cycle[last_cycle >= today]
    if open_weeks.empty:
        raise SystemExit("no remaining week in the schedule")
    return int(open_weeks.index.min())


def week_games(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    rows = schedule[schedule["week"] == week].copy()
    stations = pd.read_csv(STATION_MAP)[["stadium", "icao_station"]].drop_duplicates("stadium")
    rows = rows.merge(stations, on="stadium", how="left")
    stamp = pd.to_datetime(rows["gameday"].astype(str) + " " + rows["gametime"].astype(str))
    rows["kickoff"] = stamp.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    rows["open_air"] = rows["roof"].astype(str).str.lower().isin(OPEN_AIR_ROOFS)
    return rows.sort_values("kickoff")


def fetch_latest(station: str, model: str, now: datetime) -> dict:
    step = MODEL_CYCLE_HOURS[model]
    head = now.replace(minute=0, second=0, microsecond=0)
    head = head - timedelta(hours=head.hour % step)
    errors = []
    for k in range(LOOKBACK_CYCLES):
        runtime = head - timedelta(hours=step * k)
        try:
            rows = fetch_mos_bulletin(station, runtime, model=model)
        except MosFetchError as exc:
            errors.append(f"{runtime:%m-%dT%HZ} {exc}")
            continue
        if rows:
            return {"runtime": f"{runtime:%Y-%m-%dT%HZ}", "rows": len(rows), "failed": False}
    return {"runtime": None, "rows": 0, "failed": len(errors) == LOOKBACK_CYCLES}


def best_bulletin(station: str, kickoff: pd.Timestamp, cutoff: pd.Timestamp) -> dict | None:
    best = None
    for model in MODEL_CYCLE_HOURS:
        for path in sorted((BULLETIN_ROOT / model / station).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = pd.DataFrame(payload.get("data", []))
            cached_at = pd.Timestamp(payload["cached_at_utc"])
            if rows.empty or cached_at > cutoff:
                continue
            gap = (pd.to_datetime(rows["ftime"], utc=True) - kickoff).abs()
            i = int(gap.to_numpy().argmin())
            rank = (gap.iloc[i], -pd.to_datetime(rows["runtime"], utc=True).iloc[i].value)
            if best is None or rank < best["rank"]:
                best = {
                    "rank": rank,
                    "model": model,
                    "cached_at": cached_at,
                    "gap": gap.iloc[i],
                    "temp": float(rows["tmp"].iloc[i]),
                    "wind": float(rows["wsp"].iloc[i]) * KNOTS_TO_MPH,
                }
    return best


def coverage_line(row: pd.Series) -> tuple[str, bool]:
    label = f"{row['game_id']:<14} {row['kickoff']:%a %m-%d %H:%MZ}"
    if not row["open_air"]:
        return f"{label}  not open-air (roof={row['roof']})", False
    if not isinstance(row["icao_station"], str):
        return f"{label}  no station for {row['stadium']}", False
    cutoff = pd.Timestamp(pool_decision_cutoff(row["kickoff"].to_pydatetime()))
    found = best_bulletin(row["icao_station"], row["kickoff"], cutoff)
    if found is None:
        return f"{label}  {row['icao_station']}  no bulletin cached by the pick deadline", False
    covered = found["gap"] <= GFS_MOS_PROJECTION_STEP
    state = "covered" if covered else f"missing: nearest forecast {found['gap']} from kickoff"
    text = (
        f"{label}  {row['icao_station']}  {state}  {found['model']} "
        f"cached_at={found['cached_at']:%m-%d %H:%MZ} "
        f"temp={found['temp']:.0f}F wind={found['wind']:.1f}mph"
    )
    return text, covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch GFS and MEX MOS bulletins for the week")
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args(argv)
    now = datetime.now(UTC)
    schedule = load_schedule()
    week = args.week if args.week is not None else resolve_week(schedule, now.date())
    games = week_games(schedule, week)
    stations = sorted(games.loc[games["open_air"], "icao_station"].dropna().unique())
    failures = []
    for station in stations:
        for model in MODEL_CYCLE_HOURS:
            result = fetch_latest(station, model, now)
            if result["failed"]:
                failures.append(f"{model}/{station}")
    print(f"week {week} season {SEASON} fetched {now:%Y-%m-%d %H:%MZ} stations {len(stations)}")
    covered = 0
    for _, row in games.iterrows():
        text, ok = coverage_line(row)
        covered += int(ok)
        print(text)
    print(f"covered {covered} of {int(games['open_air'].sum())} open-air games")
    if failures:
        print("fetch failed for " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

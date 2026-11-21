from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import capture_scheduler as scheduler
import card_term_recorder as recorder
import check_served_terms
import mos_fetch
import pandas as pd

from nfl_ats import pick_refresh
from nfl_ats.active_model import active_forecast_season_week
from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
from nfl_ats.pick_probability import (
    CALIBRATED_HOME_PROBABILITY_COLUMN,
    CALIBRATED_PICK_SIDE_COLUMN,
    MARKET_MOVE_FEATURE_SUNDAY,
    MOVE_AVAILABLE_COLUMN,
)
from nfl_ats.qb_identity_features import default_depth_chart_observations
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features

REPO = Path(__file__).resolve().parents[1]
CARD_PATH = REPO / "CURRENT_PREDICTIONS.md"
CARD_TERMS_ROOT = REPO / "data" / "card_terms"
FOLLOW_REFRESH = timedelta(minutes=5)
CYCLE_START = scheduler.DAYS["tue"]
DISPLAYED_COLUMN = "displayed_pick_probability"
WEATHER_COLUMNS = ("temp", "wind")
CARD_HEADER = re.compile(r"^# NFL ATS predictions: (\d{4}) Week (\d+)", re.M)
CARD_ROW = re.compile(
    r"^\|[^|\n]+\|\s*([A-Z]{2,3}) at ([A-Z]{2,3})\s*\|([^|\n]+)\|\s*(\d+(?:\.(\d+))?)%\s*\|", re.M
)
PICK_TEAM = re.compile(r"\b([A-Z]{2,3})\b\s+(?:[+-]?\d|PK)")
ONE_OFF_JOB = re.compile(r"_\d{8}$")
JOB_SCRIPT = re.compile(r"[\\/]([\w.]+\.(?:py|ps1))")
AGE_GUARD_FLAG = "--max-age-minutes"


def week_tuesday(kickoffs: pd.Series) -> date:
    first = pd.to_datetime(kickoffs, utc=True).min().tz_convert(scheduler.ET)
    return (first - pd.Timedelta(days=(first.weekday() - CYCLE_START) % 7)).date()


def slot_time(tuesday: date, job: scheduler.Job, shift: timedelta = timedelta()) -> pd.Timestamp:
    hour, minute = (int(part) for part in job.at.split(":"))
    day = tuesday + timedelta(days=(scheduler.DAYS[job.day] - CYCLE_START) % 7)
    local = datetime.combine(day, time(hour, minute), tzinfo=scheduler.ET) + shift
    return pd.Timestamp(local).tz_convert("UTC")


def enabled(prefix: str = "", command: str = "") -> list[scheduler.Job]:
    return [
        job
        for job in scheduler.SCHEDULE
        if job.enabled
        and job.name.startswith(prefix)
        and (not command or command in job.command)
        and not ONE_OFF_JOB.search(job.name)
    ]


def latest(slots: list[tuple[Any, pd.Timestamp]], limit: pd.Timestamp):
    eligible = [item for item in slots if item[1] <= limit]
    return max(eligible, key=lambda item: item[1]) if eligible else None


def job_script(job: scheduler.Job) -> str:
    found = JOB_SCRIPT.findall(" ".join(job.command))
    return found[-1] if found else job.name


def move_capture_slots(
    tuesday: date, leader_quotes: pd.DataFrame
) -> list[tuple[scheduler.Job, pd.Timestamp]]:
    jobs = [job for job in enabled("odds_") if "halves" not in job.name]
    observed = pd.to_datetime(leader_quotes["observed_at_utc"], utc=True).reset_index(drop=True)
    covered: dict[str, pd.Series] = {}
    for job in jobs:
        start = slot_time(tuesday, job)
        window = observed.ge(start) & observed.le(start + timedelta(minutes=job.grace_minutes))
        script = job_script(job)
        covered[script] = covered[script] | window if script in covered else window
    owners = pd.concat(covered.values(), axis=1).sum(axis=1) if covered else observed.lt(observed)
    feeding = {script for script, window in covered.items() if bool((window & owners.eq(1)).any())}
    return [(job, slot_time(tuesday, job)) for job in jobs if job_script(job) in feeding]


@cache
def script_age_guard(script: Path) -> int:
    if not script.is_file():
        return 0
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == AGE_GUARD_FLAG
        ):
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    return int(keyword.value.value)
    return 0


def capture_skip_minutes(job: scheduler.Job) -> int:
    argv = [str(part) for part in job.command]
    if AGE_GUARD_FLAG in argv[:-1]:
        guard = int(argv[argv.index(AGE_GUARD_FLAG) + 1])
    else:
        scripts = [Path(part) for part in argv if part.endswith(".py")]
        guard = script_age_guard(scripts[-1]) if scripts else 0
    return max(job.dedupe_minutes, guard)


def parse_card(path: Path, season: int, week: int) -> dict[str, dict[str, Any]] | str:
    text = path.read_text(encoding="utf-8")
    header = CARD_HEADER.search(text)
    if header is None or (int(header.group(1)), int(header.group(2))) != (season, week):
        shown = "none" if header is None else f"{header.group(1)} week {header.group(2)}"
        return f"card file shows {shown}"
    table = text.split("<!-- LATE_WEEK_REFRESH:START -->")[0]
    rows: dict[str, dict[str, Any]] = {}
    for away, home, pick, percent, decimals in CARD_ROW.findall(table):
        team = PICK_TEAM.search(pick)
        side = "" if team is None else "HOME" if team.group(1) == home else "AWAY"
        rows[f"{season}_{week:02d}_{away}_{home}"] = {
            "side": side,
            "percent": percent,
            "decimals": len(decimals),
        }
    return rows


def recorder_rows(season: int, week: int) -> tuple[str, pd.Timestamp | None, pd.DataFrame]:
    files = sorted((CARD_TERMS_ROOT / f"{season}_week_{week:02d}").glob("*.parquet"))
    if not files:
        return "", None, pd.DataFrame()
    frame = pd.read_parquet(files[-1])
    recorded = pd.to_datetime(frame["run_at_utc"], utc=True).max()
    return (
        files[-1].name,
        None if pd.isna(recorded) else recorded,
        frame.assign(game_id=frame["game_id"].astype(str)).set_index("game_id"),
    )


def written_before(
    refresh: tuple[scheduler.Job, pd.Timestamp] | None,
    card_written: pd.Timestamp,
    limit: pd.Timestamp,
) -> tuple[str, pd.Timestamp] | None:
    writes = [("card write", card_written)]
    if refresh is not None:
        writes.append((refresh[0].name, refresh[1]))
    return latest(writes, limit)


def depth_starters(as_of: pd.Timestamp) -> dict[str, str]:
    depth = default_depth_chart_observations()
    if depth.empty:
        return {}
    observed = pd.to_datetime(depth["depth_observed_at"], utc=True, errors="coerce", format="mixed")
    depth = depth.assign(
        observed=observed, rank=pd.to_numeric(depth["depth_rank"], errors="coerce")
    ).loc[lambda rows: rows["position"].eq("QB") & rows["observed"].le(as_of)]
    newest = depth.groupby("team")["observed"].transform("max")
    current = depth.loc[depth["observed"].eq(newest)].sort_values("rank", kind="stable")
    return current.groupby("team")["gsis_id"].first().astype(str).to_dict()


def check(season: int, week: int, now: pd.Timestamp) -> dict[str, Any]:
    artifacts, data = REPO / "artifacts", REPO / "data"
    if pick_refresh.original_card(artifacts, season=season, week=week).empty:
        return {"season": season, "week": week, "card": "none", "problems": [], "locked": []}
    plan, inputs, card = recorder.capture_plan(artifacts, data, season, week)
    served = card.predictions.assign(game_id=card.predictions["game_id"].astype(str))
    served = served.set_index("game_id")
    inputs = inputs.assign(game_id=inputs["game_id"].astype(str)).set_index("game_id")
    games = list(plan.games)
    tuesday = week_tuesday(pd.Series([game.kickoff for game in games]))
    card_file = parse_card(CARD_PATH, season, week)
    ledger = recorder.ledger_card_sides(artifacts, [game.game_id for game in games])
    term_file, recorded_at, terms = recorder_rows(season, week)
    card_written = pd.Timestamp(CARD_PATH.stat().st_mtime, unit="s", tz="UTC")
    include_sunday = card.pick_probability.market_move_feature_version == (
        MARKET_MOVE_FEATURE_SUNDAY
    )
    quotes = load_decision_quotes(data / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    quotes = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["bookmaker_key"].isin(LEADER_BOOKS)
        & quotes["nflverse_game_id"].astype(str).isin([game.game_id for game in games])
    ].assign(observed=lambda rows: pd.to_datetime(rows["observed_at_utc"], utc=True))
    week_start = pd.Timestamp(datetime.combine(tuesday, time(), tzinfo=scheduler.ET)).tz_convert(
        "UTC"
    ) - timedelta(days=1)
    quotes = quotes.loc[quotes["observed"].ge(week_start)]
    refreshes = [(job, slot_time(tuesday, job)) for job in enabled(command="refresh-picks")]
    captures = move_capture_slots(tuesday, quotes)
    fetches = [(job, slot_time(tuesday, job)) for job in enabled("mos_fetch_")]
    schedule = mos_fetch.week_games(mos_fetch.load_schedule(), week)
    open_air = dict(zip(schedule["game_id"].astype(str), schedule["open_air"], strict=True))
    stations = dict(zip(schedule["game_id"].astype(str), schedule["icao_station"], strict=True))
    kickoffs = {game.game_id: pd.Timestamp(game.kickoff).tz_convert("UTC") for game in games}
    deadlines = {game.game_id: pd.Timestamp(game.deadline).tz_convert("UTC") for game in games}
    as_of = {gid: min(now, deadline) for gid, deadline in deadlines.items()}
    refresh_at = {gid: latest(refreshes, instant) for gid, instant in as_of.items()}
    cutoff = {
        gid: min(
            kickoffs[gid] if include_sunday else deadlines[gid],
            refresh_at[gid][1] if refresh_at[gid] else as_of[gid],
        )
        for gid in kickoffs
    }
    exposure = sharp_book_movement_features(
        quotes.drop(columns="observed"),
        pd.DataFrame(
            {
                "game_id": list(kickoffs),
                "commence_time_utc": list(kickoffs.values()),
                "week_first_commence_utc": min(kickoffs.values()),
                "cutoff_utc": [cutoff[gid] for gid in kickoffs],
            }
        ),
        include_sunday=include_sunday,
    ).set_index("game_id")
    starters_by_instant: dict[pd.Timestamp, dict[str, str]] = {}
    problems: list[str] = []
    locked: list[str] = []
    pending: list[str] = []
    rows: list[dict[str, Any]] = []
    for game in games:
        gid = game.game_id
        label = f"{game.away_team} at {game.home_team}"
        found: list[str] = []
        row = served.loc[gid]
        side = str(row[CALIBRATED_PICK_SIDE_COLUMN])
        home_probability = float(row[CALIBRATED_HOME_PROBABILITY_COLUMN])
        picked = float(
            row[DISPLAYED_COLUMN]
            if DISPLAYED_COLUMN in row.index and pd.notna(row[DISPLAYED_COLUMN])
            else home_probability
            if side == "HOME"
            else 1.0 - home_probability
        )
        if isinstance(card_file, str):
            found.append(card_file)
        elif gid not in card_file:
            found.append("not on the card file")
        else:
            shown = card_file[gid]
            if shown["side"] != side:
                found.append(f"card side {shown['side']} != recomputed {side}")
            recomputed = f"{100.0 * picked:.{shown['decimals']}f}"
            if shown["percent"] != recomputed:
                found.append(f"card shows {shown['percent']}% != recomputed {recomputed}%")
        ledger_side = ledger.get(gid, game.published_pick_side or game.previous_pick_side)
        if ledger_side != side:
            found.append(f"ledger side {ledger_side} != recomputed {side}")
        refresh = refresh_at[gid]
        written = written_before(refresh, card_written, as_of[gid])
        recorder_pending = written is not None and (recorded_at is None or recorded_at < written[1])
        if recorder_pending:
            after = written[1].tz_convert(scheduler.ET).strftime("%a %H:%M ET")
            pending.append(f"{label}: recorder pending after {written[0]} ({after})")
        elif gid not in terms.index:
            found.append("no card-term recorder row")
        else:
            term = terms.loc[gid]
            if str(term["pick_side"]) != side:
                found.append(f"recorder side {term['pick_side']} != recomputed {side}")
            gap = abs(float(term["home_cover_probability"]) - home_probability)
            if gap > recorder.PROBABILITY_TOLERANCE:
                found.append(f"recorder probability differs by {gap:.4f}")
        own = quotes.loc[quotes["nflverse_game_id"].astype(str).eq(gid)]
        own = own.loc[own["observed"].lt(cutoff[gid])]
        newest = own["observed"].max() if not own.empty else None
        books = int(exposure.loc[gid, "leader_books"]) if gid in exposure.index else 0
        move_available = float(row.get(MOVE_AVAILABLE_COLUMN, 0.0) or 0.0) > 0
        capture = None
        if refresh is not None:
            capture = latest(captures, refresh[1])
            if capture is not None:
                required = capture[1] - timedelta(minutes=capture_skip_minutes(capture[0]))
                if newest is None or newest < required:
                    found.append(
                        f"newest move quote {newest} older than capture {capture[0].name} "
                        f"before refresh {refresh[0].name}"
                    )
            if move_available and 0 < books < check_served_terms.ANCHOR_MINIMUM_BOOKS:
                found.append(f"move rests on {books} book")
            fetched = latest(fetches, refresh[1])
            missing = [
                column
                for column in WEATHER_COLUMNS
                if column not in inputs.columns or pd.isna(inputs.loc[gid, column])
            ]
            station = stations.get(gid)
            forecast = (
                mos_fetch.best_bulletin(station, kickoffs[gid], refresh[1])
                if open_air.get(gid, False) and fetched is not None and isinstance(station, str)
                else None
            )
            available = (
                forecast is not None and forecast["gap"] <= mos_fetch.GFS_MOS_PROJECTION_STEP
            )
            if available and missing:
                found.append(f"open-air game without served {'/'.join(missing)} after MOS fetch")
        instant = refresh[1] if refresh is not None else as_of[gid]
        if instant not in starters_by_instant:
            starters_by_instant[instant] = depth_starters(instant)
        starters = starters_by_instant[instant]
        for team_side, team in (("home", game.home_team), ("away", game.away_team)):
            projected = inputs.loc[gid].get(f"{team_side}_projected_qb_id")
            starter = starters.get(team)
            if starter is None:
                found.append(f"{team} has no depth-chart QB")
            elif str(projected) != starter:
                found.append(f"{team} QB input {projected} != depth starter {starter}")
        (problems if game.eligible else locked).extend(f"{label}: {item}" for item in found)
        rows.append(
            {
                "game": label,
                "open": bool(game.eligible),
                "side": side,
                "pick_pct": round(100.0 * picked, 1),
                "refresh": refresh[0].name if refresh else None,
                "move_capture": capture[0].name if capture else None,
                "newest_move_quote_age_min": None
                if newest is None or refresh is None
                else round((refresh[1] - newest).total_seconds() / 60.0),
                "move_books": books,
                "open_air": bool(open_air.get(gid, False)),
                "recorder": "pending" if recorder_pending else "compared",
                "problems": len(found),
            }
        )
    return {
        "season": season,
        "week": week,
        "checked_at_utc": now.isoformat(),
        "recorder_file": term_file,
        "recorder_written_utc": recorded_at.isoformat() if recorded_at is not None else None,
        "card_written_utc": card_written.isoformat(),
        "games": rows,
        "problems": problems,
        "locked": locked,
        "pending": pending,
    }


def print_schedule() -> None:
    argv = [str(scheduler.UV), "run", "--no-sync", "python", str(Path(__file__).resolve())]
    print(json.dumps({"argv": argv}))
    for job in sorted(
        enabled(command="refresh-picks"),
        key=lambda job: ((scheduler.DAYS[job.day] - CYCLE_START) % 7, job.at),
    ):
        hour, minute = (int(part) for part in job.at.split(":"))
        moved = datetime.combine(date.min, time(hour, minute)) + FOLLOW_REFRESH
        at = moved.strftime("%H:%M")
        print(f"card_fidelity_{job.day}_{at.replace(':', '')} {job.day} {at} after {job.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute every open card game's served probability and compare it with "
        "the card file, the pick-revision ledger and the card-term recorder; check the move, "
        "weather and quarterback inputs for freshness."
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Report without notifying.")
    parser.add_argument("--print-schedule", action="store_true")
    args = parser.parse_args()
    if args.print_schedule:
        print_schedule()
        return 0
    if (args.season is None) != (args.week is None):
        parser.error("pass both --season and --week, or neither")
    if args.season is None:
        resolved = active_forecast_season_week(REPO / "artifacts")
        if resolved is None:
            parser.error("no active forecast season/week")
        args.season, args.week = int(resolved[0]), int(resolved[1])
    result = check(args.season, args.week, pd.Timestamp(datetime.now(UTC)))
    print(json.dumps(result, default=str))
    for line in result.get("pending", []):
        print(line)
    for problem in result["problems"]:
        print(problem, file=sys.stderr)
    if not result["problems"]:
        return 0
    if not args.dry_run:
        detail = "; ".join(result["problems"])
        scheduler.send_notification(
            "NFL card check failed",
            f"Week {args.week}: {detail}"[: scheduler.ALARM_DETAIL_CHARS],
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

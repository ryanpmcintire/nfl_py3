from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfl_ats import pool_hook  # noqa: E402
from nfl_ats.active_model import load_active_ats_model  # noqa: E402
from nfl_ats.forecast_cold_visitor_tilt_overlay import tuesday_noon_et_cutoff_utc  # noqa: E402
from nfl_ats.market_data import POOL_TIMEZONE, load_quote_history  # noqa: E402
from nfl_ats.pick_probability import (  # noqa: E402
    MOVE_AVAILABLE_COLUMN,
    load_pick_probability_model,
    market_move_toward_home,
)
from nfl_ats.splash_lines import load_splash_capture  # noqa: E402

TUESDAY_LOCK_CHECK_MINUTES = 12 * 60 + 30
ANCHOR_MINIMUM_BOOKS = 2
ANCHOR_MAXIMUM_BOOK_RANGE = 2.0


class AnchorBooks(NamedTuple):
    basis: str
    books: int
    book_range: float
    anchor: float


class AnchorContract(NamedTuple):
    game_id: str
    game: str
    pool_line: float
    anchor: float | None
    anchor_basis: str
    anchor_books: int
    anchor_book_range: float | None
    crossed_mass: float
    room: float | None
    crossed_mass_exceeds_room: bool | None


def _newest_capture_week(data_root: Path, now: pd.Timestamp) -> tuple[int, int] | None:
    newest: tuple[pd.Timestamp, int, int] | None = None
    live: tuple[pd.Timestamp, int, int] | None = None
    for path in sorted((data_root / "splash").glob("*_week*_*.json")):
        stem = path.stem.split("_")
        season = int(stem[0])
        week = int(stem[1].removeprefix("week"))
        capture = load_splash_capture(data_root, season, week)
        if capture is None:
            continue
        stamp = pd.Timestamp(capture.captured_at_et)
        if newest is None or stamp > newest[0]:
            newest = (stamp, season, week)
        pending = any(pd.Timestamp(game.kickoff_et) > now for game in capture.games)
        if pending and (live is None or stamp > live[0]):
            live = (stamp, season, week)
    chosen = live or newest
    return None if chosen is None else (chosen[1], chosen[2])


def _forecast_rows(artifacts_root: Path) -> pd.DataFrame:
    manifest = load_active_ats_model(artifacts_root)
    if manifest is None:
        raise SystemExit("no active ATS model manifest")
    directory = artifacts_root / str(manifest["weekly_forecast"]["artifact"])
    rows = pd.read_csv(directory / "recommendations.csv")
    if "method" in rows.columns:
        rows = rows.loc[rows["method"].eq(manifest["method"])]
    return rows.drop_duplicates("game_id").reset_index(drop=True)


def _anchor_books(
    data_root: Path, season: int, week: int, game_ids: list[str], now: pd.Timestamp
) -> dict[str, AnchorBooks]:
    capture = load_splash_capture(data_root, season, week)
    if capture is None:
        return {}
    opened = tuesday_noon_et_cutoff_utc(pd.Timestamp(capture.captured_at_et).tz_convert("UTC"))
    quotes = load_quote_history(data_root / "market" / "raw", since=opened)
    if quotes.empty:
        return {}
    quotes = quotes.loc[quotes["market"].eq("spreads") & quotes["nflverse_game_id"].isin(game_ids)]
    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    commence = pd.to_datetime(quotes["commence_time_utc"], utc=True)
    quotes = quotes.assign(observed=observed).loc[
        observed.ge(opened) & observed.lt(commence) & observed.le(now)
    ]
    wednesday = (opened.tz_convert(POOL_TIMEZONE).normalize() + pd.Timedelta(days=1)).tz_convert(
        "UTC"
    )
    detail: dict[str, AnchorBooks] = {}
    for game, own in quotes.sort_values("observed").groupby("nflverse_game_id"):
        tuesday = own.loc[own["observed"].lt(wednesday)]
        if tuesday.empty:
            first = own.loc[own["observed"].eq(own["observed"].min())]
            latest = first.groupby("bookmaker_key")["home_spread_line"].last()
            anchor = float(first["home_spread_line"].median())
            basis = "first_snapshot"
        else:
            latest = tuesday.groupby("bookmaker_key")["home_spread_line"].last()
            anchor = float(latest.median())
            basis = "tuesday_books"
        detail[str(game)] = AnchorBooks(
            basis, len(latest), float(latest.max() - latest.min()), anchor
        )
    return detail


def _anchor_contract(
    pending: pd.DataFrame, detail: dict[str, AnchorBooks], push_table: dict[float, float]
) -> list[AnchorContract]:
    games = pending["game_id"].astype(str).tolist()
    labels = (pending["away_team"].astype(str) + " at " + pending["home_team"].astype(str)).tolist()
    pools = pd.to_numeric(pending["spread_line"], errors="coerce").tolist()
    stated = (
        pd.to_numeric(pending["home_cover_probability"], errors="coerce").tolist()
        if "home_cover_probability" in pending.columns
        else [float("nan")] * len(pending)
    )
    rows: list[AnchorContract] = []
    for game, label, pool, probability in zip(games, labels, pools, stated, strict=True):
        books = detail.get(game)
        if books is None:
            rows.append(
                AnchorContract(game, label, float(pool), None, "none", 0, None, 0.0, None, None)
            )
            continue
        mass = pool_hook.crossed_push_mass(books.anchor, float(pool), push_table)
        room = None
        if mass > 0.0 and pd.notna(probability):
            room = float(1.0 - probability) if pool > books.anchor else float(probability)
        rows.append(
            AnchorContract(
                game,
                label,
                float(pool),
                books.anchor,
                books.basis,
                books.books,
                books.book_range,
                mass,
                room,
                None if room is None else bool(mass > room),
            )
        )
    return rows


def _contract_problems(contract: list[AnchorContract]) -> list[str]:
    problems: list[str] = []
    thin = [row.game for row in contract if 0 < row.anchor_books < ANCHOR_MINIMUM_BOOKS]
    wide = [
        row.game
        for row in contract
        if row.anchor_book_range is not None and row.anchor_book_range > ANCHOR_MAXIMUM_BOOK_RANGE
    ]
    over = [row.game for row in contract if row.crossed_mass_exceeds_room]
    if thin:
        problems.append(
            f"{', '.join(thin)}: the market anchor for the pool-line offset comes from fewer "
            f"than {ANCHOR_MINIMUM_BOOKS} books, so it is one book's number and not a market "
            "consensus."
        )
    if wide:
        problems.append(
            f"{', '.join(wide)}: the books behind the market anchor are more than "
            f"{ANCHOR_MAXIMUM_BOOK_RANGE:g} points apart, wider than on any week the offset "
            "term was replayed on."
        )
    if over:
        problems.append(
            f"{', '.join(over)}: the pool-line offset moves more push probability than the "
            "model gives that side at all."
        )
    return problems


def check(
    season: int | None, week: int | None, now: datetime, *, replay: bool = False
) -> dict[str, Any]:
    data_root = REPO / "data"
    artifacts_root = REPO / "artifacts"
    now_ts = pd.Timestamp(now).tz_convert("UTC")
    if season is None or week is None:
        found = _newest_capture_week(data_root, now_ts)
        if found is None:
            raise SystemExit("no pool board capture under data/splash")
        season, week = found
    model = load_pick_probability_model(artifacts_root)
    rows = _forecast_rows(artifacts_root)
    rows = rows.loc[rows["season"].eq(season) & rows["week"].eq(week)].reset_index(drop=True)
    kickoff = pd.to_datetime(rows["kickoff"], utc=True, errors="coerce")
    first_kickoff = kickoff.min()
    pending = rows.loc[kickoff.gt(now_ts)].reset_index(drop=True)
    problems: list[str] = []
    anchors: dict[str, float] = {}
    contract: list[AnchorContract] = []
    differs = 0
    nonzero = 0
    move_games = 0
    if not pending.empty:
        push_table = dict(model.hook_push_mass)
        game_ids = pending["game_id"].astype(str).tolist()
        anchors = pool_hook.tuesday_market_lines(data_root, season, week, game_ids)
        pool = pd.to_numeric(pending["spread_line"], errors="coerce")
        for game, line in zip(game_ids, pool, strict=True):
            anchor = anchors.get(game)
            if anchor is not None and float(anchor) != float(line):
                differs += 1
        hook = pool_hook.served_hook_frame(pending, data_root, push_table)
        nonzero = int(
            (pd.to_numeric(hook[pool_hook.HOOK_PUSH_MASS_COLUMN], errors="coerce") > 0).sum()
        )
        move = market_move_toward_home(
            pending,
            data_root,
            now=now_ts.to_pydatetime(),
            feature_version=model.market_move_feature_version,
        )
        move_games = int((pd.to_numeric(move[MOVE_AVAILABLE_COLUMN], errors="coerce") > 0).sum())
        contract = _anchor_contract(
            pending, _anchor_books(data_root, season, week, game_ids, now_ts), push_table
        )
    capture = load_splash_capture(data_root, season, week)
    if model.has_pool_hook and differs > 0 and nonzero == 0:
        problems.append(
            f"The pool-line offset term is zero on every game even though {differs} games have "
            "a pool line different from the market anchor."
        )
    if model.has_pool_hook:
        problems.extend(_contract_problems(contract))
        drift = [row.game for row in contract if anchors.get(row.game_id) != row.anchor]
        if drift and not replay:
            problems.append(
                f"{', '.join(drift)}: the book-by-book read of the market anchor no longer "
                "matches the anchor the offset term uses, so this check has to be brought back "
                "in line with the served term."
            )
    if pd.notna(first_kickoff) and not pending.empty:
        first_et = first_kickoff.tz_convert(POOL_TIMEZONE)
        tuesday = (first_et - pd.Timedelta(days=(first_et.weekday() - 1) % 7)).normalize()
        now_et = now_ts.tz_convert(POOL_TIMEZONE)
        tuesday_check = tuesday + pd.Timedelta(minutes=TUESDAY_LOCK_CHECK_MINUTES)
        if capture is not None and now_et >= tuesday_check and not anchors:
            problems.append(
                "The pool board is captured and it is past Tuesday 12:30 Eastern, "
                "but no game has a market anchor for the offset term."
            )
        if now_et >= tuesday + timedelta(days=2) and move_games == 0:
            problems.append(
                "It is Thursday or later of the game week and no game has a line-move input."
            )
    return {
        "season": season,
        "week": week,
        "games": len(pending),
        "anchors_available": len(anchors),
        "pool_differs_from_anchor": differs,
        "offset_term_nonzero": nonzero,
        "move_available_games": move_games,
        "checked_at_utc": now_ts.isoformat(),
        "anchor_minimum_books": ANCHOR_MINIMUM_BOOKS,
        "anchor_maximum_book_range": ANCHOR_MAXIMUM_BOOK_RANGE,
        "anchor_contract": [
            {
                **row._asdict(),
                "crossed_mass": round(row.crossed_mass, 4),
                "room": None if row.room is None else round(row.room, 4),
            }
            for row in contract
        ],
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a served probability term is silent for the live week, or when "
        "the market anchor behind the pool-line offset breaks its book-agreement contract."
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--as-of",
        help="Replay the check at an earlier UTC instant. The book-by-book anchor read uses "
        "only quotes observed by then; the served-term counts still read every captured quote.",
    )
    args = parser.parse_args()
    now = datetime.now(UTC)
    if args.as_of:
        stamp = pd.Timestamp(args.as_of)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        now = stamp.to_pydatetime()
    result = check(args.season, args.week, now, replay=bool(args.as_of))
    print(json.dumps(result))
    for problem in result["problems"]:
        print(problem, file=sys.stderr)
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.tiebreaker import (
    TOTAL_LOW_SIDE_SHADE_POINTS,
    _neighborhood,
    lined_finals,
    newest_schedules_path,
    tiebreaker_report,
)

REPO = Path(__file__).resolve().parents[1]
CARD_ROW = re.compile(
    r"^\|[^|]*\|\s*(?P<away>[A-Z]{2,3}) at (?P<home>[A-Z]{2,3})\s*\|\s*(?P<pick>[^|]+?)\s*\|"
)
PICK_TEAM = re.compile(r"(?P<team>[A-Z]{2,3})\s+(?:[+-]\d+(?:\.\d+)?|pick'em)")
GRID = 0.5


def card_sides(card_path: Path) -> dict[tuple[str, str], str]:
    sides: dict[tuple[str, str], str] = {}
    for line in card_path.read_text(encoding="utf-8").splitlines():
        row = CARD_ROW.match(line)
        if row is None:
            continue
        team = PICK_TEAM.search(row["pick"])
        if team is None:
            continue
        away, home = row["away"], row["home"]
        if team["team"] == home:
            sides[(away, home)] = "HOME"
        elif team["team"] == away:
            sides[(away, home)] = "AWAY"
    return sides


def residual_pmf(
    finals: pd.DataFrame, home_margin: float, total_line: float, side: str | None
) -> tuple[dict[int, float], float, float]:
    hood = _neighborhood(finals, home_margin, total_line)
    rows = hood.frame
    weights = hood.weights.astype(float)
    margin = (rows["home_score"] - rows["away_score"]).to_numpy(dtype=float)
    line = rows["spread_line"].to_numpy(dtype=float)
    if side == "HOME":
        weights = weights * (margin > line)
    elif side == "AWAY":
        weights = weights * (margin < line)
    totals = rows["home_score"] + rows["away_score"]
    residual = (totals - rows["total_line"]).to_numpy(dtype=float)
    pmf: dict[int, float] = {}
    for value, weight in zip(np.round(residual / GRID).astype(int), weights, strict=True):
        if weight > 0.0:
            pmf[int(value)] = pmf.get(int(value), 0.0) + float(weight)
    total = sum(pmf.values())
    ess = total * total / float((weights**2).sum()) if total > 0 else 0.0
    mean = float(np.sum(weights * residual) / total) if total > 0 else float("nan")
    return {key: value / total for key, value in pmf.items()}, ess, mean


def convolve(pmfs: list[dict[int, float]]) -> dict[int, float]:
    out = {0: 1.0}
    for pmf in pmfs:
        nxt: dict[int, float] = {}
        for a, pa in out.items():
            for b, pb in pmf.items():
                nxt[a + b] = nxt.get(a + b, 0.0) + pa * pb
        out = nxt
    return out


def quantile(pmf: dict[int, float], q: float) -> float:
    running = 0.0
    for key in sorted(pmf):
        running += pmf[key]
        if running >= q:
            return key * GRID
    return max(pmf) * GRID


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combined final-score guess for a set of games, conditioned on the card's "
            "sides all covering."
        )
    )
    parser.add_argument("--games", required=True, help="Comma-separated nflverse game ids.")
    parser.add_argument("--card", default=str(REPO / "CURRENT_PREDICTIONS.md"))
    parser.add_argument("--data-root", default=str(REPO / "data"))
    parser.add_argument("--out", default=None, help="JSON file to write the result to.")
    parser.add_argument(
        "--only-on",
        default=None,
        help="Exit 0 without work unless today (local) is one of these ISO dates, comma-separated.",
    )
    args = parser.parse_args()

    if args.only_on:
        today = datetime.now(UTC).astimezone().date()
        allowed = {date.fromisoformat(value) for value in args.only_on.split(",")}
        if today not in allowed:
            reason = f"{today} not in {sorted(map(str, allowed))}"
            print(json.dumps({"status": "skipped", "reason": reason}))
            return 0

    data_root = Path(args.data_root)
    schedules = pd.read_parquet(newest_schedules_path(data_root))
    finals = lined_finals(schedules)
    sides = card_sides(Path(args.card))
    games = []
    conditional, unconditional = [], []
    served_sum = 0.0
    for game_id in args.games.split(","):
        report = tiebreaker_report(data_root, game_id=game_id)
        served = report.guess_total_line - TOTAL_LOW_SIDE_SHADE_POINTS
        side = sides.get((report.away, report.home))
        margin = report.consensus.home_expected_margin
        market_total = report.consensus.total_line
        pmf_side, ess_side, mean_side = residual_pmf(finals, margin, market_total, side)
        pmf_all, _ess_all, mean_all = residual_pmf(finals, margin, market_total, None)
        conditional.append(pmf_side)
        unconditional.append(pmf_all)
        served_sum += served
        pick = report.home if side == "HOME" else report.away if side == "AWAY" else None
        games.append(
            {
                "game_id": game_id,
                "matchup": f"{report.away} at {report.home}",
                "market_total": market_total,
                "market_source": report.consensus.source,
                "served_total": round(served, 2),
                "card_pick": pick,
                "conditional_mean_shift": round(mean_side, 2),
                "conditional_games": round(ess_side),
                "unconditional_mean_shift": round(mean_all, 2),
            }
        )
    joint = convolve(conditional)
    plain = convolve(unconditional)
    median = served_sum + quantile(joint, 0.5)
    result = {
        "status": "ok",
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "games": games,
        "sides_known": all(game["card_pick"] for game in games),
        "sum_of_market_totals": sum(game["market_total"] for game in games),
        "sum_of_served_totals": round(served_sum, 2),
        "combined_guess": round(median),
        "combined_median": round(median, 2),
        "middle_half": [round(served_sum + quantile(joint, q)) for q in (0.25, 0.75)],
        "middle_80": [round(served_sum + quantile(joint, q)) for q in (0.10, 0.90)],
        "unconditional_median": round(served_sum + quantile(plain, 0.5), 2),
    }
    picks = ", ".join(
        f"{g['card_pick'] or '?'} ({g['matchup']}, total {g['market_total']:g})" for g in games
    )
    result["message"] = (
        f"Thanksgiving combined score guess: {result['combined_guess']}. Picks: {picks}. "
        f"Half of outcomes land between {result['middle_half'][0]} and {result['middle_half'][1]}."
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

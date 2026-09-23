from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.io import atomic_json
from nfl_ats.published_picks import load_published_picks


@dataclass(frozen=True)
class TiebreakerResult:
    week: int
    game_id: str
    guessed_total: float
    market_total: float
    actual_total: float
    source: Path


def record_tiebreaker(artifacts_root: Path, payload: dict[str, Any]) -> Path:
    generated = pd.to_datetime(payload["generated_at_utc"], utc=True, errors="raise")
    if pd.isna(generated):
        raise ValueError("A saved tiebreaker needs its publication time")
    path = (
        artifacts_root
        / "published"
        / "tiebreakers"
        / f"{int(payload['season'])}-week-{int(payload['week']):02d}"
        / f"{generated.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("A saved tiebreaker cannot be replaced")
    else:
        atomic_json(payload, path)
    return path


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def settled_tiebreakers(
    artifacts_root: Path, *, season: int, outcomes: pd.DataFrame
) -> list[TiebreakerResult]:
    needed = {"game_id", "result", "home_score", "away_score"}
    if not needed.issubset(outcomes.columns):
        return []
    publications = load_published_picks(artifacts_root)
    publications = publications.loc[
        pd.to_numeric(publications["season"], errors="coerce").eq(season)
    ]
    deadlines = publications.groupby("game_id")["pick_deadline_utc"].min().to_dict()
    finals = outcomes.drop_duplicates("game_id", keep="last").set_index("game_id")
    paths = sorted(
        (artifacts_root / "margin_predictions").glob(f"{season}-week-*/tiebreaker.json")
    ) + sorted((artifacts_root / "published" / "tiebreakers").glob(f"{season}-week-*/*.json"))
    selected: dict[str, tuple[pd.Timestamp, TiebreakerResult]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("season") != season:
            continue
        game_id = str(payload.get("game_id", ""))
        deadline = deadlines.get(game_id)
        if deadline is None or pd.isna(deadline) or game_id not in finals.index:
            continue
        generated = pd.to_datetime(
            str(payload.get("generated_at_utc", "")), utc=True, errors="coerce"
        )
        if generated is None or pd.isna(generated) or generated > deadline:
            continue
        final = finals.loc[game_id]
        if _finite_number(final["result"]) is None:
            continue
        home_score = _finite_number(final["home_score"])
        away_score = _finite_number(final["away_score"])
        guess_home = _finite_number(payload.get("guess_home"))
        guess_away = _finite_number(payload.get("guess_away"))
        market_total = _finite_number(payload.get("market_total"))
        week = _finite_number(payload.get("week"))
        if (
            home_score is None
            or away_score is None
            or guess_home is None
            or guess_away is None
            or market_total is None
            or week is None
        ):
            continue
        previous = selected.get(game_id)
        if previous is None or generated >= previous[0]:
            selected[game_id] = (
                generated,
                TiebreakerResult(
                    week=int(week),
                    game_id=game_id,
                    guessed_total=guess_home + guess_away,
                    market_total=market_total,
                    actual_total=home_score + away_score,
                    source=path,
                ),
            )
    return sorted((row for _, row in selected.values()), key=lambda row: (row.week, row.game_id))

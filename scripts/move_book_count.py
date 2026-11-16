import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests" / "scratch" / "serve_time_move"))

import flip_attribution_audit as faa  # noqa: E402
import serve_time_move as stm  # noqa: E402

from nfl_ats.pick_refresh import original_card  # noqa: E402
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS  # noqa: E402

OUT = REPO / "artifacts" / "move_book_count"
FIT_MOVE = (
    REPO / "artifacts" / "sunday_market_probability" / "20260920_fixed" / "market_move.parquet"
)
SEASON = 2026
WEEKS = tuple(range(1, 11))
FULL_DEPTH = len(LEADER_BOOKS)
NULL_DRAWS = 2000
SEED = 20261116
FAMILY = "move_book_count_2026_11_16"


def load_captures() -> pd.DataFrame:
    rows = []
    for path in sorted(
        (REPO / "artifacts" / "prospective").glob("independent_validation*/captures/*.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        capture = payload.get("capture", payload)
        for row in capture.get("rows", []):
            if int(row.get("season", 0)) == SEASON and int(row.get("week", 0)) in WEEKS:
                rows.append({**row, "capture_file": path.name})
    frame = pd.DataFrame(rows)
    frame["recorded"] = pd.to_datetime(frame["recorded_at_utc"], utc=True, format="ISO8601")
    frame["kick"] = pd.to_datetime(frame["kickoff"], utc=True, format="ISO8601")
    return frame


def record(frame: pd.DataFrame, column: str) -> dict:
    settled = frame.loc[frame[column].notna()]
    wins = int(settled[column].astype(bool).sum())
    return faa.exact(wins, int(len(settled) - wins))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fit = pd.read_parquet(FIT_MOVE)
    history = {
        "fit_games": len(fit),
        "leader_books_counts": {
            str(k): int(v) for k, v in fit["leader_books"].value_counts().items()
        },
        "one_book_games": int(fit["leader_books"].eq(1).sum()),
        "two_book_games": int(fit["leader_books"].eq(2).sum()),
    }
    print("HISTORY", history)
    captures = load_captures()
    last = (
        captures.loc[captures["recorded"].lt(captures["kick"])]
        .sort_values("recorded")
        .groupby("game_id", as_index=False)
        .tail(1)
    )
    print("CAPTURE GAMES by week", last.groupby("week").size().to_dict())
    store, _ = stm.load_store()
    store = store.loc[~store["snapshot_dir"].str.endswith(stm.REPARSE_SUFFIX)]
    old_function = stm.old_function() if hasattr(stm, "old_function") else None
    from nfl_ats.sharp_book_movement_features import sharp_book_movement_features as new_function

    sets = faa.load_coefficient_sets()
    margins, _ = faa.load_results()
    rows = []
    for week in WEEKS:
        card = original_card(faa.ARTIFACTS, season=SEASON, week=week)
        if card.empty:
            print("NO CARD week", week)
            continue
        line = dict(
            zip(
                card["game_id"].astype(str),
                pd.to_numeric(card["decision_home_spread"]),
                strict=True,
            )
        )
        scoped = last.loc[last["week"].astype(int).eq(week)]
        cache: dict = {}
        for row in scoped.itertuples(index=False):
            when = row.recorded
            name, payload = faa.coefficients_at(sets, when)
            coefficient = payload["coefficients"]
            include_sunday = (
                payload.get("market_move_feature_version") == faa.MARKET_MOVE_FEATURE_SUNDAY
            )
            served_move = float(row.market_move) if pd.notna(row.market_move) else 0.0
            options = []
            functions = [("new", new_function)] + ([("old", old_function)] if old_function else [])
            for code, function in functions:
                key = (code, when)
                if key not in cache:
                    cache[key] = stm.move_table(function, store, card, when, include_sunday)
                move, books = stm.value(cache[key], str(row.game_id))
                options.append((code, move, books))
            match = [o for o in options if np.isclose(o[1], served_move)]
            code, replay_move, books = match[0] if match else options[0]
            available = 1.0 if bool(row.market_available) else 0.0
            served_logit = faa.logit(float(row.combined_home_probability))
            move_terms = (
                float(coefficient["move_toward_home"]) * served_move
                + float(coefficient["move_available"]) * available
            )
            without = served_logit - move_terms
            margin = margins.get(str(row.game_id))
            spread = line.get(str(row.game_id), float(row.spread_line))
            covered = None if margin is None or margin == spread else bool(margin > spread)
            rows.append(
                {
                    "game_id": row.game_id,
                    "week": week,
                    "capture_recorded_utc": when.isoformat(),
                    "coefficient_set": name,
                    "served_move": served_move,
                    "market_available": available,
                    "replay_move": replay_move,
                    "replay_code": code,
                    "replay_matches_served": bool(match),
                    "leader_books": int(books),
                    "served_home_probability": float(row.combined_home_probability),
                    "served_side": "HOME" if served_logit > 0 else "AWAY",
                    "side_without_move": "HOME" if without > 0 else "AWAY",
                    "move_side": "HOME" if served_move > 0 else ("AWAY" if served_move < 0 else ""),
                    "pool_home_spread": spread,
                    "home_margin": margin,
                    "push": margin is not None and margin == spread,
                    "home_covered": covered,
                }
            )
    games = pd.DataFrame(rows)
    games["move_side_right"] = [
        None if c is None or s == "" else (s == "HOME") == c
        for c, s in zip(games["home_covered"], games["move_side"], strict=True)
    ]
    games["served_right"] = [
        None if c is None else (s == "HOME") == c
        for c, s in zip(games["home_covered"], games["served_side"], strict=True)
    ]
    games["move_decides"] = games["served_side"].ne(games["side_without_move"])
    games.to_csv(OUT / "games.csv", index=False)
    print("REPLAY MATCH", int(games["replay_matches_served"].sum()), "of", len(games))
    print(
        "BOOKS x week",
        games.groupby(["week", "leader_books"]).size().unstack(fill_value=0).to_dict(),
    )
    moved = games.loc[games["served_move"].ne(0.0)]
    one = moved.loc[moved["leader_books"].eq(1)]
    many = moved.loc[moved["leader_books"].ge(2)]
    cells = {
        "one_book_move_side": record(one, "move_side_right"),
        "multi_book_move_side": record(many, "move_side_right"),
        "one_book_move_decides_served": record(one.loc[one["move_decides"]], "served_right"),
        "multi_book_move_decides_served": record(many.loc[many["move_decides"]], "served_right"),
        "one_book_pushes": int(one["push"].sum()),
        "multi_book_pushes": int(many["push"].sum()),
    }
    settled = moved.loc[moved["move_side_right"].notna() & moved["leader_books"].ge(1)].copy()
    settled["one"] = settled["leader_books"].eq(1)
    right = settled["move_side_right"].astype(float).to_numpy()

    def gap(flag: np.ndarray) -> float:
        if flag.sum() == 0 or (~flag).sum() == 0:
            return float("nan")
        return 100.0 * (right[flag].mean() - right[~flag].mean())

    flag = settled["one"].to_numpy()
    observed = gap(flag)
    rng = np.random.default_rng(SEED)
    weeks = settled["week"].to_numpy()
    draws = []
    for _ in range(NULL_DRAWS):
        permuted = flag.copy()
        for w in np.unique(weeks):
            index = np.flatnonzero(weeks == w)
            permuted[index] = rng.permutation(permuted[index])
        draws.append(gap(permuted))
    draws = np.array(draws)
    draws = draws[np.isfinite(draws)]
    boot = []
    for _ in range(NULL_DRAWS):
        picked = rng.choice(np.unique(weeks), size=len(np.unique(weeks)), replace=True)
        index = np.concatenate([np.flatnonzero(weeks == w) for w in picked])
        f = flag[index]
        r = right[index]
        if f.sum() and (~f).sum():
            boot.append(100.0 * (r[f].mean() - r[~f].mean()))
    boot = np.array(boot)
    cells["difference_accuracy_points"] = {
        "one_minus_multi": observed,
        "week_block_interval": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "probability_positive": float((boot > 0).mean()),
        "null_percentile": float((draws < observed).mean() + 0.5 * (draws == observed).mean()),
        "null_draws": len(draws),
        "bootstrap_draws": len(boot),
    }
    summary = {"history": history, "games": len(games), "moved_games": len(moved), "cells": cells}
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=float))
    print(
        one[
            [
                "game_id",
                "week",
                "served_move",
                "replay_code",
                "replay_matches_served",
                "move_side",
                "served_side",
                "side_without_move",
                "pool_home_spread",
                "home_margin",
                "move_side_right",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()

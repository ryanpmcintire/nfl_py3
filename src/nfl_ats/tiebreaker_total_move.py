from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.io import atomic_json
from nfl_ats.provenance import sha256_file
from nfl_ats.score_lattice import pick_consistent_top_score, score_lattice
from nfl_ats.tiebreaker import lined_finals, newest_schedules_path, weighted_median

if TYPE_CHECKING:
    from nfl_ats.pick_refresh import RefreshResult

FIT_PATH = Path("tiebreaker_total_move/fit.json")
CONSTRUCTION = "lead88_unit3"


def fit_total_move(
    baseline_paths: list[Path], schedules_path: Path, destination: Path
) -> dict[str, Any]:
    schedules = pd.read_parquet(schedules_path)
    completed = schedules.loc[
        schedules["season"].ge(2020)
        & schedules["game_type"].eq("REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].copy()
    frames = pd.concat([pd.read_parquet(path) for path in baseline_paths], ignore_index=True)
    if frames["game_id"].duplicated().any() or completed["game_id"].duplicated().any():
        raise ValueError("Duplicate total-move training games")
    frame = frames.loc[frames["game_id"].isin(completed["game_id"])].copy()
    for observed, cutoff in (
        ("tuesday_last_observed", "freeze"),
        ("deadline_last_observed", "deadline"),
        ("deadline", "kickoff"),
    ):
        left = pd.to_datetime(frame[observed], utc=True, errors="raise")
        right = pd.to_datetime(frame[cutoff], utc=True, errors="raise")
        valid = left.lt(right) if observed == "deadline" else left.le(right)
        if not valid.all():
            raise ValueError(f"Total-move training clock violation: {observed}")
    for column, clock in (
        ("margin_training_cutoff", "deadline"),
        ("joint_training_cutoff", "freeze"),
    ):
        training_cutoff = pd.to_datetime(frame[column], utc=True, errors="raise")
        freeze = pd.to_datetime(frame[clock], utc=True, errors="raise")
        if not training_cutoff.add(pd.Timedelta(days=1)).le(freeze).all():
            raise ValueError(f"Total-move baseline training reaches its prediction: {column}")
    indexed = completed.set_index("game_id")
    actual = indexed["home_score"] + indexed["away_score"]
    if not np.allclose(frame["actual_total"], frame["game_id"].map(actual)):
        raise ValueError("Total-move outcomes differ from completed schedules")
    expected_side = np.where(frame["home_probability"].ge(0.5), "HOME", "AWAY")
    if not np.array_equal(frame["pick_side"].to_numpy(), expected_side):
        raise ValueError("Total-move baseline side differs from its fitted probability")
    if not np.allclose(frame["total_move"], frame["deadline_total"] - frame["tuesday_total"]):
        raise ValueError("Total-move baseline does not match the frozen total quotes")
    usable = frame.loc[frame["served"].notna()].copy()
    columns = ["actual_total", "served", "total_move", "served_centre_total", "centre_margin"]
    if not np.isfinite(usable[columns].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite total-move training values")
    if not np.allclose(usable["served"], usable["guess_home"] + usable["guess_away"]):
        raise ValueError("Total-move training target is not the served integer total")
    moving = usable.loc[usable["total_move"].ne(0)]
    slope = weighted_median(
        ((moving["actual_total"] - moving["served"]) / moving["total_move"]).to_numpy(dtype=float),
        moving["total_move"].abs().to_numpy(dtype=float),
    )
    if not math.isfinite(slope):
        raise ValueError("No completed games with a defined total-move response")
    missing = completed.loc[~completed["game_id"].isin(frame["game_id"])]
    payload = {
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "slope": slope,
        "fitted_at_utc": datetime.now(UTC).isoformat(),
        "training_games": len(usable),
        "moving_games": len(moving),
        "season_start": int(usable["season"].min()),
        "season_end": int(usable["season"].max()),
        "training_completed_through": str(usable["gameday"].max()),
        "coverage": {
            "completed_regular_season_games": len(completed),
            "paired_baselines": len(frame),
            "declined_baselines": int(frame["served"].isna().sum()),
            "missing_construction_by_season": {
                str(year): int(count) for year, count in missing.groupby("season").size().items()
            },
            "missing_construction_game_ids": sorted(missing["game_id"].astype(str)),
        },
        "lineage": {
            "sources": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in [*baseline_paths, schedules_path]
            ],
            "protocol": "docs/lead88_unit3.md",
            "target": "actual total minus served integer total",
            "fit": "no-intercept least absolute deviation on deadline minus Tuesday total",
            "historical_pool_line": "opener",
            "retrospective_limit": (
                "Unit-3 margin histories precede deadline; some follow Tuesday freeze"
            ),
            "evaluation": "Existing six-season held-out unit-3 replay; this fit is for serving",
        },
    }
    atomic_json(payload, destination)
    return payload


def initialize_total_move(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "total_move_baseline" not in result:
        result["total_move_baseline"] = {
            key: payload.get(key)
            for key in (
                "guess_home",
                "guess_away",
                "market_total",
                "blended_total",
                "served_total",
                "lattice_centre_margin",
                "pick_side",
                "pick_spread_line",
                "generated_at_utc",
                "pick_cover_probability",
                "pick_push_probability",
                "consistency_note",
            )
        }
        result.update(
            tuesday_total=payload["market_total"],
            current_median_total=payload["market_total"],
            total_move_slope=None,
            total_move_adjustment=0.0,
            total_move_applied_adjustment=0.0,
            total_move_status="tuesday_baseline",
            total_move_fit=None,
        )
    return result


def adjusted_total_move(
    payload: dict[str, Any], finals: pd.DataFrame, *, current_total: float, slope: float
) -> dict[str, Any]:
    result = initialize_total_move(payload)
    base = result["total_move_baseline"]
    if not math.isfinite(current_total) or not math.isfinite(slope):
        raise ValueError("Total-move adjustment must be finite")
    total = float(base["served_total"])
    margin = float(base["lattice_centre_margin"])
    side = str(base["pick_side"])
    spread = float(base["pick_spread_line"])
    delta = slope * (current_total - float(base["market_total"]))
    result.update(base)
    result.update(
        current_median_total=current_total,
        total_move_slope=slope,
        total_move_adjustment=delta,
        total_move_applied_adjustment=0.0,
        total_move_status="zero_adjustment",
    )
    original_cell = (math.floor((total + margin) / 2), math.floor((total - margin) / 2))
    adjusted_cell = (
        math.floor((total + delta + margin) / 2),
        math.floor((total + delta - margin) / 2),
    )
    if delta != 0 and original_cell == adjusted_cell:
        result["total_move_status"] = "same_cell"
    elif delta != 0:
        lattice = score_lattice(finals, margin, total + delta)
        chosen = pick_consistent_top_score(
            lattice,
            pick_side=side,
            spread_line=spread,
            served_total=total + delta,
            centre_margin=margin,
        )
        if chosen is None:
            result["total_move_status"] = "kept_original_for_pick_consistency"
        else:
            result.update(
                guess_home=chosen[0],
                guess_away=chosen[1],
                served_total=total + delta,
                blended_total=total + delta,
                total_move_applied_adjustment=delta,
                total_move_status="adjusted",
            )
    result["projected_total"] = result["guess_home"] + result["guess_away"]
    result["implied_margin"] = result["guess_home"] - result["guess_away"]
    return result


def current_total_quote(
    data_root: Path, game_id: str, *, as_of: pd.Timestamp
) -> tuple[float, dict[str, Any]] | None:
    latest: tuple[pd.Timestamp, float, dict[str, Any]] | None = None
    for path in sorted((data_root / "market" / "raw").glob("*/quotes.parquet"), reverse=True):
        frame = pd.read_parquet(
            path,
            columns=[
                "nflverse_game_id",
                "market",
                "outcome_side",
                "line",
                "bookmaker_key",
                "observed_at_utc",
            ],
            filters=[
                ("nflverse_game_id", "=", game_id),
                ("market", "=", "totals"),
                ("outcome_side", "=", "OVER"),
            ],
        )
        if frame.empty:
            continue
        observed = pd.to_datetime(frame["observed_at_utc"], utc=True, errors="coerce")
        frame = frame.loc[observed.le(as_of)].copy()
        if frame.empty:
            continue
        timestamp = observed.loc[frame.index].max()
        if latest is not None and timestamp <= latest[0]:
            continue
        frame = frame.sort_values("observed_at_utc").drop_duplicates("bookmaker_key", keep="last")
        values = pd.to_numeric(frame["line"], errors="coerce")
        values = values.loc[np.isfinite(values) & values.gt(0)]
        if values.empty:
            continue
        latest = (
            timestamp,
            float(values.median()),
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "observed_at_utc": timestamp.isoformat(),
                "books": len(values),
            },
        )
    return None if latest is None else (latest[1], latest[2])


def refresh_total_move(
    artifacts_root: Path,
    data_root: Path,
    plan: RefreshResult,
    *,
    write: bool = False,
    destination: Path | None = None,
) -> dict[str, Any]:
    active = load_active_ats_model(artifacts_root)
    if active is None:
        return {"written": False, "reason": "No active tiebreaker model"}
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        return {"written": False, "reason": "No active tiebreaker forecast"}
    path = forecast / "tiebreaker.json"
    if not path.is_file():
        return {"written": False, "reason": "No original tiebreaker guess"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("season"), payload.get("week"), payload.get("model_id")) != (
        plan.season,
        plan.week,
        plan.model_id,
    ):
        return {"written": False, "reason": "Tiebreaker does not match this card"}
    game = next((game for game in plan.games if game.game_id == payload["game_id"]), None)
    if game is None or not game.eligible or plan.computed_at_utc >= game.deadline:
        return {"written": False, "reason": "Tiebreaker pick deadline passed or unavailable"}
    side = (game.published_pick_side or game.new_pick_side) if write else game.previous_pick_side
    if payload["pick_side"] != side:
        return {"written": False, "reason": "Tiebreaker needs a baseline consistent with the pick"}
    fit_path = artifacts_root / FIT_PATH
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("construction") != CONSTRUCTION:
        raise ValueError("Total-move fit uses a different score construction")
    if pd.Timestamp(fit["training_completed_through"], tz="UTC") >= plan.computed_at_utc:
        raise ValueError("Total-move fit reaches the refresh prediction time")
    quote = current_total_quote(data_root, game.game_id, as_of=plan.computed_at_utc)
    if quote is None:
        return {"written": False, "reason": "No pre-deadline market total"}
    schedules = pd.read_parquet(newest_schedules_path(data_root))
    baseline = initialize_total_move(payload)
    cutoff = pd.Timestamp(baseline["total_move_baseline"]["generated_at_utc"])
    finals = lined_finals(schedules)
    finals = finals.loc[
        pd.to_datetime(finals["gameday"], utc=True).add(pd.Timedelta(days=1)).le(cutoff)
    ]
    result = adjusted_total_move(
        baseline, finals, current_total=quote[0], slope=float(fit["slope"])
    )
    result.update(
        total_move_fit={"path": str(fit_path), "sha256": sha256_file(fit_path)},
        total_move_quote=quote[1],
        total_move_refreshed_at_utc=plan.computed_at_utc.isoformat(),
        total_move_deadline_utc=game.deadline.isoformat(),
    )
    if write:
        atomic_json(result, path)
        if destination is not None and destination.parent != path.parent:
            atomic_json(result, destination.parent / "tiebreaker.json")
    return {"written": write, "artifact": str(path), "tiebreaker": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the tiebreaker response to market total news")
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts") / FIT_PATH)
    args = parser.parse_args()
    result = fit_total_move(args.baseline, newest_schedules_path(args.data_root), args.destination)
    result["coverage"].pop("missing_construction_game_ids")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

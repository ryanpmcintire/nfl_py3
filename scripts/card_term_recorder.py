from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import nfl_ats.card_view as card_view
import nfl_ats.pick_refresh as pick_refresh
from nfl_ats.active_model import active_forecast_season_week
from nfl_ats.pick_probability import (
    CALIBRATED_HOME_PROBABILITY_COLUMN,
    CALIBRATED_PICK_SIDE_COLUMN,
    FLAG_SUM_COLUMN,
    MODEL_HOME_PROBABILITY_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PROBABILITY_EPSILON,
    PickProbabilityModel,
)
from nfl_ats.pool_hook import HOOK_COLUMN

REPO = Path(__file__).resolve().parents[1]
PROBABILITY_TOLERANCE = 1e-9


def input_row_sha256(row: pd.Series) -> str:
    body = row.to_json(date_format="iso", double_precision=15, default_handler=str)
    return hashlib.sha256(str(body).encode()).hexdigest()


def finite_or_zero(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) and math.isfinite(float(number)) else 0.0


def logit(probability: float) -> float:
    stated = min(max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    return math.log(stated / (1.0 - stated))


def served_terms(served: pd.Series, pick_model: PickProbabilityModel) -> dict[str, Any]:
    base = finite_or_zero(served[MODEL_HOME_PROBABILITY_COLUMN])
    model_logit = logit(base)
    flag_sum = finite_or_zero(served.get(FLAG_SUM_COLUMN))
    move = finite_or_zero(served.get(MOVE_COLUMN))
    move_available = finite_or_zero(served.get(MOVE_AVAILABLE_COLUMN))
    hook = finite_or_zero(served.get(HOOK_COLUMN)) if pick_model.has_pool_hook else 0.0
    contributions = {
        "intercept_contribution": pick_model.intercept,
        "model_contribution": pick_model.model_logit * model_logit,
        "flag_contribution": pick_model.flag_sum * flag_sum,
        "move_contribution": pick_model.move_toward_home * move,
        "move_available_contribution": pick_model.move_available * move_available,
        "hook_contribution": pick_model.pool_hook * hook,
    }
    return {
        "model_home_probability": base,
        "model_logit": model_logit,
        **contributions,
        "flag_sum": flag_sum,
        "move_toward_home": move,
        "move_available": move_available,
        "hook_mass": hook,
        "home_logit": float(sum(contributions.values())),
        "home_cover_probability": float(served[CALIBRATED_HOME_PROBABILITY_COLUMN]),
        "pick_side": str(served[CALIBRATED_PICK_SIDE_COLUMN]),
        "fit_artifact": pick_model.artifact,
    }


def ledger_card_sides(artifacts_root: Path, game_ids: list[str]) -> dict[str, str]:
    ledger = pick_refresh.load_pick_revisions(artifacts_root)
    if ledger.empty:
        return {}
    ledger = ledger.assign(game_id=ledger["game_id"].astype(str))
    ledger = ledger[ledger["game_id"].isin(game_ids)]
    ledger = ledger.sort_values("revision_recorded_at_utc", kind="stable")
    return ledger.groupby("game_id")["new_pick_side"].last().astype(str).to_dict()


def capture_plan(artifacts_root: Path, data_root: Path, season: int, week: int):
    captured: dict[str, Any] = {}
    original_lines = pick_refresh.apply_external_lines
    original_resolve = card_view.resolve_card_probabilities

    def lines_spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        captured["input"] = original_lines(*args, **kwargs)
        return captured["input"]

    def resolve_spy(*args: Any, **kwargs: Any):
        captured["card"] = original_resolve(*args, **kwargs)
        return captured["card"]

    pick_refresh.apply_external_lines = lines_spy
    card_view.resolve_card_probabilities = resolve_spy
    try:
        plan = pick_refresh.plan_refresh(artifacts_root, data_root, season=season, week=week)
    finally:
        pick_refresh.apply_external_lines = original_lines
        card_view.resolve_card_probabilities = original_resolve
    if "card" not in captured or captured["card"].pick_probability is None:
        raise SystemExit("card path did not resolve a served pick_probability fit")
    return plan, captured["input"], captured["card"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append every current-card game's served logit decomposition, one row per run."
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "card_terms")
    args = parser.parse_args()
    if (args.season is None) != (args.week is None):
        parser.error("pass both --season and --week, or neither")
    if args.season is None:
        resolved = active_forecast_season_week(args.artifacts)
        if resolved is None:
            parser.error("no active forecast season/week")
        season, week = int(resolved[0]), int(resolved[1])
    else:
        season, week = args.season, args.week

    run_at = pd.Timestamp(datetime.now(UTC))
    plan, inputs, card = capture_plan(args.artifacts, args.data, season, week)
    pick_model = card.pick_probability
    served = card.predictions.assign(game_id=card.predictions["game_id"].astype(str))
    served = served.set_index("game_id")
    inputs = inputs.assign(game_id=inputs["game_id"].astype(str)).set_index("game_id")
    games = list(plan.games)
    card_sides = ledger_card_sides(args.artifacts, [game.game_id for game in games])

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for game in games:
        terms = served_terms(served.loc[game.game_id], pick_model)
        sum_gap = abs(terms["home_logit"] - logit(terms["home_cover_probability"]))
        card_side = card_sides.get(
            game.game_id, game.published_pick_side or game.previous_pick_side
        )
        if terms["pick_side"] != game.new_pick_side:
            failures.append(
                f"{game.game_id} side {terms['pick_side']} != served {game.new_pick_side}"
            )
        if abs(terms["home_cover_probability"] - game.new_home_cover_probability) > (
            PROBABILITY_TOLERANCE
        ):
            failures.append(f"{game.game_id} probability differs from served")
        if sum_gap > PROBABILITY_TOLERANCE:
            failures.append(f"{game.game_id} contributions miss served logit by {sum_gap:.3g}")
        if game.eligible and card_side != terms["pick_side"]:
            failures.append(
                f"{game.game_id} open card side {card_side} != served {terms['pick_side']}"
            )
        rows.append(
            {
                "run_at_utc": run_at,
                "season": season,
                "week": week,
                "game_id": game.game_id,
                **terms,
                "card_pick_side": card_side,
                "card_agrees": card_side == terms["pick_side"],
                "eligible": bool(game.eligible),
                "input_row_sha256": (
                    input_row_sha256(inputs.loc[game.game_id])
                    if game.game_id in inputs.index
                    else ""
                ),
            }
        )
    frame = pd.DataFrame(rows)
    target = args.out / f"{season}_week_{week:02d}"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{run_at.strftime('%Y%m%dT%H%M%S%fZ')}.parquet"
    with path.open("xb") as handle:
        frame.to_parquet(handle, index=False)
    print(f"wrote {len(frame)} rows to {path}")
    print(f"week rows total {len(pd.read_parquet(target))}")
    if failures:
        print("MISMATCH", *failures, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.card_explanation import (
    PickExplanation,
    check_language,
    explanations_to_dict,
    family_contributions_from_waterfall_entry,
    from_json,
    game_explanation_from_contributions,
    render_pick_text,
)
from nfl_ats.io import atomic_json
from nfl_ats.key_line_pick_read import key_line_touched_games
from nfl_ats.public_board import load_waterfall_feed


class RegenerateExplanationsError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegenerateExplanationsError(f"Required JSON input is missing: {path}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def regenerate(artifacts_root: Path) -> dict[str, Any]:

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise RegenerateExplanationsError(
            f"No synchronized active ATS model under {artifacts_root}"
        )
    forecast_dir = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast_dir is None:
        raise RegenerateExplanationsError("Active ATS model has no linked weekly forecast")

    explanations_path = forecast_dir / "explanations.json"
    if not explanations_path.is_file():
        raise RegenerateExplanationsError(f"No explanations.json under {forecast_dir}")
    stored = from_json(explanations_path.read_text(encoding="utf-8"))

    metadata = _load_json(forecast_dir / "metadata.json")
    key_line_games = key_line_touched_games(metadata)

    predictions = pd.read_csv(forecast_dir / "predictions.csv")
    predictions_by_game = {str(row["game_id"]): row for _, row in predictions.iterrows()}

    waterfall_by_game = load_waterfall_feed(artifacts_root)

    rewritten: list[PickExplanation] = []
    changed = 0
    for explanation in stored:
        row = predictions_by_game.get(explanation.game_id)
        push_probability = (
            float(row["push_probability"])
            if row is not None and pd.notna(row.get("push_probability"))
            else None
        )
        home_team = str(row["home_team"]) if row is not None else ""
        away_team = str(row["away_team"]) if row is not None else ""
        game_explanation = game_explanation_from_contributions(
            explanation.game_id,
            home_team,
            away_team,
            family_contributions_from_waterfall_entry(waterfall_by_game.get(explanation.game_id)),
        )
        new_text = render_pick_text(
            explanation.matchup,
            explanation.market_line,
            explanation.model_probability,
            explanation.overlays,
            explanation.freshness,
            explanation.refresh,
            game_explanation,
            push_probability,
            key_line_read=explanation.game_id in key_line_games,
        )
        check_language(new_text)
        if new_text != explanation.text:
            changed += 1
        rewritten.append(replace(explanation, text=new_text))

    atomic_json(explanations_to_dict(rewritten), explanations_path)
    return {
        "forecast_directory": str(forecast_dir),
        "explanations": len(rewritten),
        "changed": changed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-render pick-explanation text from already-stored components, "
        "for cases where only the wording function changed"
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    result = regenerate(args.artifacts_root.resolve())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

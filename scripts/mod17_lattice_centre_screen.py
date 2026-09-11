from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.lattice_centre_challenger import (
    CENTRE_POLICY,
    RULE_MINIMUM_CONSISTENT_STEP,
    challenger_centre,
    minimum_consistent_centre,
    pick_consistent_cell,
    strictly_on_pick_side,
)
from nfl_ats.provenance import artifact_provenance
from nfl_ats.tiebreaker import lined_finals, newest_schedules_path

BOOTSTRAP_SAMPLES = 2000

BOOTSTRAP_SEED = 20260816

ARMS = ("served", "challenger", "oracle")


def recover_mapping(group: pd.DataFrame) -> tuple[float, float, float]:
    probabilities = group["home_cover_probability_at_open_raw"].to_numpy(dtype=float)
    residuals = group["residual_at_open"].to_numpy(dtype=float)
    z = stats.norm.isf(probabilities)
    design = np.column_stack([np.ones_like(z), z])
    coefficients, *_ = np.linalg.lstsq(design, residuals, rcond=None)
    location = -float(coefficients[0])
    scale = -float(coefficients[1])
    rebuilt = stats.norm.sf(-residuals, loc=location, scale=scale)
    return location, scale, float(np.max(np.abs(rebuilt - probabilities)))


def newest_evaluation(artifacts_root: Path, active_model_id: str | None) -> Path:
    root = artifacts_root / "opener_evaluation"
    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "per_game.parquet").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no opener evaluation archives under {root}")
    if active_model_id is not None:
        for path in reversed(candidates):
            metadata_path = path / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("active_model_id")) == active_model_id:
                return path
    return candidates[-1]


def oracle_centre(actual_margin: float, spread_line: float, pick_side: str) -> float:
    if strictly_on_pick_side(actual_margin, spread_line, pick_side):
        return float(actual_margin)
    return minimum_consistent_centre(spread_line, pick_side)


def score_games(
    per_game: pd.DataFrame, schedules: pd.DataFrame, *, verbose: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schedule_rows = schedules.drop_duplicates("game_id").set_index(
        schedules.drop_duplicates("game_id")["game_id"].astype(str)
    )
    gamedays = pd.to_datetime(schedules["gameday"], utc=True)
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "weeks": 0,
        "games_in_archive": len(per_game),
        "max_mapping_recovery_error": 0.0,
        "missing_schedule_row": 0,
        "missing_total_or_score": 0,
        "served_refused": 0,
        "challenger_refused": 0,
        "oracle_refused": 0,
        "flip_rule_games": 0,
        "point_side_disagrees_with_pick": 0,
        "margin_side_disagrees_with_pick": 0,
    }
    for (season, week), group in per_game.groupby(["season", "week"], sort=True):
        season, week = int(season), int(week)
        location, _scale, recovery_error = recover_mapping(group)
        diagnostics["weeks"] += 1
        diagnostics["max_mapping_recovery_error"] = max(
            diagnostics["max_mapping_recovery_error"], recovery_error
        )
        week_ids = set(group["game_id"].astype(str))
        week_days = gamedays.loc[schedules["game_id"].astype(str).isin(week_ids)]
        if week_days.empty:
            diagnostics["missing_schedule_row"] += len(group)
            continue
        cutoff = week_days.min()
        previous = schedules["season"].lt(season) | (
            schedules["season"].eq(season) & schedules["week"].lt(week)
        )
        history = lined_finals(schedules.loc[previous & gamedays.lt(cutoff)])
        if history.empty:
            diagnostics["missing_schedule_row"] += len(group)
            continue
        for _, entry in group.iterrows():
            game_id = str(entry["game_id"])
            if game_id not in schedule_rows.index:
                diagnostics["missing_schedule_row"] += 1
                continue
            game = schedule_rows.loc[game_id]
            market_total = pd.to_numeric(game.get("total_line"), errors="coerce")
            actual_home = pd.to_numeric(game.get("home_score"), errors="coerce")
            actual_away = pd.to_numeric(game.get("away_score"), errors="coerce")
            if not np.isfinite([market_total, actual_home, actual_away]).all():
                diagnostics["missing_total_or_score"] += 1
                continue
            spread_line = float(entry["tue_open_home_spread"])
            predicted_margin = spread_line + float(entry["residual_at_open"])
            point = predicted_margin + location
            pick_side = "HOME" if float(entry["home_cover_probability_at_open"]) >= 0.5 else "AWAY"
            if not strictly_on_pick_side(predicted_margin, spread_line, pick_side):
                diagnostics["margin_side_disagrees_with_pick"] += 1
            choice = challenger_centre(point, spread_line, pick_side)
            if choice.rule == RULE_MINIMUM_CONSISTENT_STEP:
                diagnostics["flip_rule_games"] += 1
                diagnostics["point_side_disagrees_with_pick"] += 1
            actual_margin = float(actual_home) - float(actual_away)
            centres = {
                "served": predicted_margin,
                "challenger": choice.centre,
                "oracle": oracle_centre(actual_margin, spread_line, pick_side),
            }
            cells: dict[str, tuple[int, int, float] | None] = {}
            for arm, centre in centres.items():
                cells[arm] = pick_consistent_cell(
                    history,
                    centre_margin=centre,
                    served_total=float(market_total),
                    pick_side=pick_side,
                    spread_line=spread_line,
                )
                if cells[arm] is None:
                    diagnostics[f"{arm}_refused"] += 1
            if cells["served"] is None or cells["challenger"] is None:
                continue
            row: dict[str, Any] = {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": str(game["home_team"]),
                "away_team": str(game["away_team"]),
                "spread_line": spread_line,
                "market_total": float(market_total),
                "pick_side": pick_side,
                "predicted_margin": predicted_margin,
                "residual_location": location,
                "pick_deciding_point": point,
                "challenger_centre_margin": choice.centre,
                "challenger_centre_rule": choice.rule,
                "actual_home": float(actual_home),
                "actual_away": float(actual_away),
                "actual_total": float(actual_home) + float(actual_away),
                "actual_margin": actual_margin,
                "served_centre_on_pick_side": bool(
                    strictly_on_pick_side(predicted_margin, spread_line, pick_side)
                ),
                "challenger_centre_on_pick_side": bool(
                    strictly_on_pick_side(choice.centre, spread_line, pick_side)
                ),
            }
            for arm in ARMS:
                cell = cells[arm]
                if cell is None:
                    row[f"{arm}_home"] = float("nan")
                    row[f"{arm}_away"] = float("nan")
                    row[f"{arm}_total_absolute_error"] = float("nan")
                    row[f"{arm}_score_absolute_error"] = float("nan")
                    continue
                home_score, away_score, _tolerance = cell
                row[f"{arm}_home"] = float(home_score)
                row[f"{arm}_away"] = float(away_score)
                row[f"{arm}_total_absolute_error"] = abs(
                    (home_score + away_score) - row["actual_total"]
                )
                row[f"{arm}_score_absolute_error"] = abs(home_score - float(actual_home)) + abs(
                    away_score - float(actual_away)
                )
            rows.append(row)
        if verbose:
            print(
                f"  {season} week {week:>2}: {len(group)} archived, {len(rows)} paired so far, "
                f"location {location:+.4f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    diagnostics["paired_games"] = len(frame)
    diagnostics["last_game_of_week_games"] = 0
    if not frame.empty:
        frame["is_last_game_of_week"] = _last_game_flags(frame, schedules)
        diagnostics["last_game_of_week_games"] = int(frame["is_last_game_of_week"].sum())
    return frame, diagnostics


def _last_game_flags(frame: pd.DataFrame, schedules: pd.DataFrame) -> pd.Series:
    regular = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    keys = regular["gameday"].astype(str)
    if "gametime" in regular.columns:
        keys = keys + " " + regular["gametime"].astype(str).fillna("")
    regular = regular.assign(_key=keys)
    last_ids = (
        regular.sort_values("_key")
        .groupby(["season", "week"])["game_id"]
        .last()
        .astype(str)
        .tolist()
    )
    return frame["game_id"].astype(str).isin(set(last_ids))


def _paired_metric(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "closest_total_mae_improvement": float(
            (frame["served_total_absolute_error"] - frame["challenger_total_absolute_error"]).mean()
        ),
        "closest_score_mae_improvement": float(
            (frame["served_score_absolute_error"] - frame["challenger_score_absolute_error"]).mean()
        ),
        "pick_consistency_accuracy_points": float(
            (
                frame["challenger_centre_on_pick_side"].astype(float)
                - frame["served_centre_on_pick_side"].astype(float)
            ).mean()
            * 100.0
        ),
    }


def _oracle_metric(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "oracle_closest_total_mae_improvement": float(
            (frame["served_total_absolute_error"] - frame["oracle_total_absolute_error"]).mean()
        ),
        "oracle_closest_score_mae_improvement": float(
            (frame["served_score_absolute_error"] - frame["oracle_score_absolute_error"]).mean()
        ),
    }


def _bootstrap(frame: pd.DataFrame, metric: Any, samples: int, seed: int) -> list[dict[str, Any]]:
    table = week_blocked_bootstrap(frame, metric, block="week", samples=samples, seed=seed)
    return [
        {
            "metric": str(row["metric"]),
            "estimate": float(row["estimate"]),
            "interval_low": float(row["lower"]),
            "interval_high": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
            "blocks": int(frame.groupby(["season", "week"]).ngroups),
            "games": len(frame),
        }
        for _, row in table.iterrows()
    ]


def run(artifacts_root: Path, data_root: Path, *, samples: int, seed: int, verbose: bool) -> int:
    active_path = artifacts_root / "active_ats_model.json"
    active = json.loads(active_path.read_text(encoding="utf-8")) if active_path.is_file() else {}
    evaluation = newest_evaluation(artifacts_root, str(active.get("model_id") or "") or None)
    per_game = pd.read_parquet(evaluation / "per_game.parquet")
    schedules = pd.read_parquet(newest_schedules_path(data_root))

    print("MOD-17 lattice-centre challenger screen -- predeclared in docs/tiebreaker.md")
    print(f"opener archive {evaluation.relative_to(artifacts_root)}")
    print(f"active model {active.get('model_id')}  centre policy {CENTRE_POLICY}")
    seasons = f"{per_game['season'].min()}-{per_game['season'].max()}"
    print(f"archive games {len(per_game)}  seasons {seasons}")
    print()

    frame, diagnostics = score_games(per_game, schedules, verbose=verbose)
    if frame.empty:
        print("no paired games")
        return 1

    primary = _bootstrap(frame, _paired_metric, samples, seed)
    control = _bootstrap(frame, _oracle_metric, samples, seed)
    last_game = frame.loc[frame["is_last_game_of_week"]]
    secondary = _bootstrap(last_game, _paired_metric, samples, seed) if not last_game.empty else []

    differing = frame.loc[
        (frame["served_home"] != frame["challenger_home"])
        | (frame["served_away"] != frame["challenger_away"])
    ]
    raw = {
        "served_total_mae": float(frame["served_total_absolute_error"].mean()),
        "challenger_total_mae": float(frame["challenger_total_absolute_error"].mean()),
        "oracle_total_mae": float(frame["oracle_total_absolute_error"].mean()),
        "served_score_mae": float(frame["served_score_absolute_error"].mean()),
        "challenger_score_mae": float(frame["challenger_score_absolute_error"].mean()),
        "oracle_score_mae": float(frame["oracle_score_absolute_error"].mean()),
        "served_centre_on_pick_side_rate": float(frame["served_centre_on_pick_side"].mean()),
        "challenger_centre_on_pick_side_rate": float(
            frame["challenger_centre_on_pick_side"].mean()
        ),
        "games_where_the_two_guesses_differ": len(differing),
        "share_where_the_two_guesses_differ": float(len(differing) / len(frame)),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = artifacts_root / "mod17_lattice_centre" / stamp
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "predictions.csv", index=False)
    configuration = {
        "command": "mod17-lattice-centre-screen",
        "centre_policy": CENTRE_POLICY,
        "opener_archive": str(evaluation.relative_to(artifacts_root)).replace("\\", "/"),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "predeclaration": "docs/tiebreaker.md",
    }
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "active_model_id": active.get("model_id"),
        "diagnostics": diagnostics,
        "raw": raw,
        "primary_population": primary,
        "last_game_population": secondary,
        "positive_control": control,
        "by_rule": {
            rule: {
                "games": len(part),
                "closest_total_mae_improvement": float(
                    (
                        part["served_total_absolute_error"]
                        - part["challenger_total_absolute_error"]
                    ).mean()
                ),
                "closest_score_mae_improvement": float(
                    (
                        part["served_score_absolute_error"]
                        - part["challenger_score_absolute_error"]
                    ).mean()
                ),
            }
            for rule, part in frame.groupby("challenger_centre_rule")
        },
        "provenance": artifact_provenance(configuration, evaluation / "per_game.parquet"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("diagnostics", "raw")}, indent=2, default=str))
    print()
    for label, table in (
        ("primary population (every graded game)", primary),
        ("secondary population (the week's last game)", secondary),
        ("positive control (oracle margin centre)", control),
    ):
        print(label)
        for entry in table:
            print(
                f"  {entry['metric']}: {entry['estimate']:+.6f} "
                f"95% [{entry['interval_low']:+.6f}, {entry['interval_high']:+.6f}] "
                f"P+ {entry['probability_positive']:.4f} "
                f"({entry['games']} games, {entry['blocks']} blocks)"
            )
        print()
    print(f"artifact {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the MOD-17 pick-deciding-margin lattice centre against the served "
            "predicted_margin centre on the walk-forward opener archive."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts")),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    return run(
        args.artifacts_root,
        args.data_root,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())

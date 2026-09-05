"""Run frozen SIM-03 weekly lineup-mixture screen without changing production."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from expected_lineup_loss_on_production import PROFILE, candidate_profile  # noqa: E402

from nfl_ats.calibration import fit_residual_smoother  # noqa: E402
from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    build_pairing_table,
    close_reference_table,
)
from nfl_ats.expected_lineup_loss_features import (  # noqa: E402
    EXPECTED_LINEUP_LOSS_COLUMNS,
    _lineup_group,
    _visible_panel,
    attach_asof_injury_features,
    attach_play_probabilities,
    team_week_decision_instants,
    team_week_expected_loss,
    visible_injury_lookup,
)
from nfl_ats.lineup_availability import depth_chart_position_group  # noqa: E402
from nfl_ats.lineup_mixture import (  # noqa: E402
    GROUPS,
    SCENARIOS,
    SEED,
    lineup_draws,
    loss_coefficients,
    mixture_probability,
    paired_summary,
)
from nfl_ats.margin import fit_margin_model  # noqa: E402
from nfl_ats.play_probability import serving_player_history  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    stamp_sidecar,
    write_experiment_artifact,
)

OUT = ROOT / "artifacts/experiments/lineup_mixture"
BASE = ROOT / "data/processed/game_features_weak_stack.parquet"
PANEL = ROOT / "data/processed/play_probability_panel.parquet"


def prepare(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    panel = pd.read_parquet(PANEL)
    snapshot = json.loads(BASE.with_suffix(".manifest.json").read_text())["source_player_snapshot"]
    source = ROOT / "data/players/raw" / snapshot
    decisions = team_week_decision_instants(base)
    joined = panel.merge(
        decisions[["season", "week", "team", "decision_at"]],
        on=["season", "week", "team"],
        suffixes=("_panel", ""),
    )
    if (
        not pd.to_datetime(joined.decision_at_panel, utc=True)
        .eq(pd.to_datetime(joined.decision_at, utc=True))
        .all()
    ):
        raise ValueError("Rebuild panel at pool cutoff")
    rows = _visible_panel(joined.drop(columns="decision_at_panel"))
    rows["lineup_group"] = _lineup_group(rows.position, rows.position_group)
    lookup = visible_injury_lookup(pd.read_parquet(source / "injuries.parquet"), decisions)
    rows = attach_asof_injury_features(rows, lookup)
    rows = attach_play_probabilities(rows, panel)
    rows = rows.loc[rows.play_probability.notna()].copy()
    totals = team_week_expected_loss(rows.loc[rows.depth_rank.eq(1)])
    features = base.copy()
    for side in ("home", "away"):
        features = features.merge(
            totals.rename(
                columns={
                    "team": side + "_team",
                    **{"expected_lineup_loss_" + g: side + "_loss_" + g for g in GROUPS},
                }
            ),
            on=["season", "week", side + "_team"],
            how="left",
            validate="m:1",
        )
    for group, column in zip(GROUPS, EXPECTED_LINEUP_LOSS_COLUMNS, strict=True):
        features[column] = features["home_loss_" + group] - features["away_loss_" + group]
    return features, rows, source


def evaluate_rows(scoring, model, candidate, scenarios):
    coefficients = loss_coefficients(candidate, scoring)
    smoother = fit_residual_smoother(model.residuals, "gaussian")
    centers = model._predicted_margin(scoring, model._spread(scoring))[0]
    output = []
    for (_, row), center in zip(scoring.iterrows(), centers, strict=True):
        home = scenarios[(int(row.season), int(row.week), row.home_team)]
        away = scenarios[(int(row.season), int(row.week), row.away_team)]
        expected = home["expected"] - away["expected"]
        result = {
            "game_id": row.game_id,
            "season": int(row.season),
            "week": int(row.week),
            "result": row.result,
            "line": row.spread_line,
            "center": center,
            "sigma": smoother.std,
            "residual_mean": smoother.mean,
            "baseline": float(ndtr((center + smoother.mean - row.spread_line) / smoother.std)),
            **{f"coefficient_{g}": float(c) for g, c in zip(GROUPS, coefficients, strict=True)},
        }
        for arm in ("mixture", "permutation", "oracle"):
            if arm not in home or arm not in away:
                continue
            delta = (home[arm] - away[arm] - expected) @ coefficients
            stats = mixture_probability(center, row.spread_line, smoother.mean, smoother.std, delta)
            result.update({f"{arm}_{k}": v for k, v in stats.items()})
        output.append(result)
    return output


def live_scenarios(base, source):
    serving_path = ROOT / "artifacts/lineups/current/lineups.json"
    serving = json.loads(serving_path.read_text())
    history = serving_player_history(
        pd.read_parquet(source / "weekly_rosters.parquet"),
        pd.read_parquet(source / "snap_counts.parquet"),
        as_of_season=2026,
        as_of_week=1,
    )
    rows = []
    decisions = team_week_decision_instants(base).set_index(["season", "week", "team"])
    for game_id, game in serving["games"].items():
        game_row = base.loc[base.game_id.eq(game_id)].iloc[0]
        for side in ("home", "away"):
            team = game_row[side + "_team"]
            decision = decisions.loc[(2026, 1, team), "decision_at"]
            observed = pd.Timestamp(game[side]["as_of"])
            if observed >= decision:
                raise ValueError("Serving lineup is not before decision")
            for player in game[side]["players"]:
                # Unidentified slots have no linked snap history and contribute zero
                # under LEAD-62; do not fabricate their availability probabilities.
                if player["gsis_id"] is None:
                    continue
                rows.append(
                    {
                        "season": 2026,
                        "week": 1,
                        "team": team,
                        "gsis_id": player["gsis_id"],
                        "position": player["position"],
                        "position_group": depth_chart_position_group(player["position"]),
                        "depth_rank": player["depth"],
                        "depth_rank_bucket": str(player["depth"]),
                        "source_schema": "daily",
                        "depth_observed_at": observed,
                        "decision_at": decision,
                        "season_week": 202601,
                        "weeks_since_last_snap": history.get(player["gsis_id"], {}).get(
                            "weeks_since_last_snap", np.nan
                        ),
                        "trailing4_snap_share": history.get(player["gsis_id"], {}).get(
                            "trailing4_snap_share", np.nan
                        ),
                        "play_probability": player["play_probability"],
                    }
                )
    return lineup_draws(pd.DataFrame(rows))


def summarize(frame):
    result = {}
    actual = frame.result.gt(frame.line)
    baseline_correct = frame.baseline.ge(0.5).eq(actual)
    for arm in ("mixture", "permutation", "oracle"):
        probability = frame[arm + "_probability"]
        candidate_correct = probability.ge(0.5).eq(actual)
        frame[arm + "_accuracy_delta"] = 100 * (
            candidate_correct.astype(float) - baseline_correct.astype(float)
        )
        frame[arm + "_brier_delta"] = (frame.baseline - actual) ** 2 - (probability - actual) ** 2
        flips = probability.ge(0.5).ne(frame.baseline.ge(0.5))
        result[arm] = {
            "accuracy": paired_summary(frame, arm + "_accuracy_delta"),
            "brier": paired_summary(frame, arm + "_brier_delta"),
            "baseline_brier": float(((frame.baseline - actual) ** 2).mean()),
            "candidate_brier": float(((probability - actual) ** 2).mean()),
            "baseline_accuracy": float(baseline_correct.mean()),
            "candidate_accuracy": float(candidate_correct.mean()),
            "flips": int(flips.sum()),
            "flip_accuracy_delta": float(frame.loc[flips, arm + "_accuracy_delta"].mean())
            if flips.any()
            else None,
            "mean_scenario_sd": float(frame[arm + "_scenario_sd"].mean()),
            "mean_total_sd_increase": float(frame[arm + "_total_sd_increase"].mean()),
            "per_season": frame.groupby("season")[[arm + "_accuracy_delta", arm + "_brier_delta"]]
            .mean()
            .to_dict("index"),
        }
    reliability = []
    for name in (
        "baseline",
        "mixture_probability",
        "permutation_probability",
        "oracle_probability",
    ):
        for bin_id in range(10):
            selected = np.minimum((frame[name] * 10).astype(int), 9).eq(bin_id)
            reliability.append(
                {
                    "arm": name,
                    "bin_low": bin_id / 10,
                    "games": int(selected.sum()),
                    "probability": float(frame.loc[selected, name].mean())
                    if selected.any()
                    else None,
                    "observed": float(actual.loc[selected].mean()) if selected.any() else None,
                }
            )
    return result, reliability


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise ValueError("Frozen outcome look already exists")
    active = json.loads((ROOT / "artifacts/active_ats_model.json").read_text())
    if active["feature_table_sha256"] != sha256_file(BASE):
        raise ValueError("Production feature digest changed")
    base = pd.read_parquet(BASE)
    base = base.loc[base.game_type.eq("REG")].copy()
    features, players, source = prepare(base)
    players.attrs = {}
    players.to_parquet(OUT / "players.parquet", index=False)
    stamp_sidecar(OUT / "players.parquet")
    scenarios = lineup_draws(players)
    print(f"Prepared {len(players)} player rows and {len(scenarios)} team-games", flush=True)
    pairing = build_pairing_table(
        ROOT / "data/market/raw",
        capture_kind="historical_backfill",
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    close = close_reference_table(pairing, features)
    targets = pairing.loc[pairing.decision_label.eq("tue_open"), ["game_id", "home_spread"]].merge(
        close[["game_id"]], on="game_id"
    )
    features["gameday"] = pd.to_datetime(features.gameday)
    eligible = features.loc[features.season.between(2020, 2025) & features.result.notna()].merge(
        targets, on="game_id"
    )
    coverage = eligible.groupby("season").size().to_dict()
    eligible = eligible.dropna(subset=list(EXPECTED_LINEUP_LOSS_COLUMNS))
    rows = []
    with candidate_profile():
        for (season, week), scoring in eligible.groupby(["season", "week"], sort=True):
            cutoff = features.loc[
                features.season.eq(season) & features.week.eq(week), "gameday"
            ].min()
            training = features.loc[features.gameday.lt(cutoff) & features.result.notna()]
            model = fit_margin_model(
                training, target="market_residual", feature_profile="weak_stack", ridge_alpha=10.0
            )
            candidate = fit_margin_model(
                training, target="market_residual", feature_profile=PROFILE, ridge_alpha=10.0
            )
            scoring = scoring.copy()
            scoring["spread_line"] = scoring.home_spread
            rows.extend(evaluate_rows(scoring, model, candidate, scenarios))
            print(f"Scored {season} week {week}: {len(scoring)} games", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "historical_before_live.csv", index=False)
        stamp_sidecar(OUT / "historical_before_live.csv")
        current = features.loc[features.season.eq(2026) & features.week.eq(1)].copy()
        training = features.loc[
            features.gameday.lt(current.gameday.min()) & features.result.notna()
        ]
        model = fit_margin_model(
            training, target="market_residual", feature_profile="weak_stack", ridge_alpha=10.0
        )
        candidate = fit_margin_model(
            training, target="market_residual", feature_profile=PROFILE, ridge_alpha=10.0
        )
        live = live_scenarios(base, source)
        forecast_path = (
            ROOT / "artifacts" / active["weekly_forecast"]["artifact"] / "predictions.csv"
        )
        forecast = pd.read_csv(forecast_path)
        live_rows = pd.DataFrame(evaluate_rows(current, model, candidate, live))
        # Preserve published forecast's baseline probabilities exactly; apply only lineup delta.
        forecast = forecast.loc[forecast.method.eq(active["method"])].set_index("game_id")
        if not forecast.index.is_unique:
            raise ValueError("Active forecast must contain one row per game")
        coefficients = loss_coefficients(candidate, current)
        for index, row in live_rows.iterrows():
            original = forecast.loc[row.game_id]
            p = float(original.home_cover_probability)
            game = current.loc[current.game_id.eq(row.game_id)].iloc[0]
            home, away = live[(2026, 1, game.home_team)], live[(2026, 1, game.away_team)]
            delta = (
                home["mixture"] - away["mixture"] - home["expected"] + away["expected"]
            ) @ coefficients
            live_rows.loc[index, "baseline"] = p
            live_rows.loc[index, "mixture_probability"] = float(
                ndtr(ndtri(p) + delta / row.sigma).mean()
            )
        live_rows["base_pick_flips"] = live_rows.baseline.ge(0.5).ne(
            live_rows.mixture_probability.ge(0.5)
        )
    frame = pd.DataFrame(rows)
    pushes = int(frame.result.eq(frame.line).sum())
    frame = frame.loc[frame.result.ne(frame.line)].copy()
    results, reliability = summarize(frame)
    for name, table in (
        ("paired_predictions", frame),
        ("week1", live_rows),
        ("reliability", pd.DataFrame(reliability)),
    ):
        path = OUT / (name + ".csv")
        table.to_csv(path, index=False)
        stamp_sidecar(path)
    configuration = {
        "seed": SEED,
        "scenarios": SCENARIOS,
        "bootstrap_draws": 20000,
        "model_id": active["model_id"],
        "base_sha256": sha256_file(BASE),
        "panel_sha256": sha256_file(PANEL),
        "source_player_snapshot": source.name,
        "requested_seasons": list(range(2020, 2026)),
        "opener_coverage_before_lineups": coverage,
        "scored_coverage": frame.groupby("season").size().to_dict(),
        "missing_lineup_seasons": [2025],
        "excluded_pushes": pushes,
        "week1_base_pick_flips": int(live_rows.base_pick_flips.sum()),
        "week1_flipped_games": live_rows.loc[live_rows.base_pick_flips, "game_id"].tolist(),
        "forecast_sha256": sha256_file(forecast_path),
        "serving_sha256": sha256_file(ROOT / "artifacts/lineups/current/lineups.json"),
    }
    write_experiment_artifact(
        OUT,
        "results.json",
        {
            "configuration": configuration,
            "results": results,
            "provenance": artifact_provenance(configuration, BASE),
        },
        command="lineup-mixture-screen",
        metrics={"games": len(frame)},
        registry_root=OUT / "experiment_registry",
    )
    print(json.dumps({"configuration": configuration, "results": results}, indent=2))


if __name__ == "__main__":
    main()

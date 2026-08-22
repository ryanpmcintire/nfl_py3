"""Deliberate-leak positive control: the ABSOLUTE classification ceiling.

Question: if the model is fit ON THE SAME OUTCOMES it predicts (total leak),
what accuracy can ANY feature set achieve on this dataset? This bounds every
legitimate model from above and empirically tests the 55-58% oracle-wall claim
against OUR grading structure.

LEAK DECLARATION (loud, deliberate): every arm's training set is the ENTIRE
completed frame -- all seasons 2009-2025, INCLUDING the target seasons and the
target games themselves. The walk-forward skeleton is retained only to define
the standard scored population; under this policy every fold's training set is
identical (the full frame), so one fit per arm is algebraically equivalent to
refitting per fold and no fold loop is executed.

Arms:
  A market_line_leak      ridge(alpha=10) on ats_margin, features = spread_line
                          and its square (line-level curvature only; a linear
                          term alone cannot express favorite-longshot bias).
  B weak_stack_leak       ridge(alpha=10) on ats_margin, features =
                          margin_feature_columns("market_residual",
                          "weak_stack") on the production weak-stack table.
  C pbp_same_game_leak    ridge(alpha=10) on ats_margin, features = same-game
                          PBP offensive summaries (EPA/success/yards/TD/TO/
                          play counts, home-minus-away). These summaries are
                          themselves outcome-contaminated BY CONSTRUCTION --
                          that is the point of a ceiling instrument.
  B2 weak_stack_leak_a1   arm B repeated at alpha=1 to show the ceiling is not
                          an artifact of the production shrinkage level.

All arms share one leak-fit path: in-sample residuals become the predictive
distribution, cover probabilities via MarginModel.predict with the production
"gaussian" probability method (the same read that graded the active model's
52.10% close-grade figure). Scoring contract matches backtest.summarize_
predictions: forced picks at p >= 0.5 over non-push games.

Expected shape: mid-60s-to-70s%, NOT ~100% -- outcomes are noisy given any
observable structure, and THAT number is the empirical answer to "how much do
outcomes respect any observable structure". If leak accuracy lands BELOW ~58%,
the ceiling is revised DOWNWARD, not upward.

Read-only positive control: nothing recorded to the rotation registry, no pick
played, no promotion decision. No wagering implication claimed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.margin import (  # noqa: E402
    MarginModel,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

START_SEASON = 2018
MIN_TRAIN_GAMES = 500
RIDGE_ALPHA = 10.0
PROBABILITY_METHOD = "gaussian"
FEATURE_TABLE = REPO / "data/processed/game_features_weak_stack.parquet"
REFERENCE_SUMMARY = REPO / "artifacts/margins/20260820T004951Z/summary.csv"
PBP_ROOT = REPO / "data/pbp/raw"
SEED = 20260822

MARKET_LINE_FEATURES: tuple[str, ...] = ("spread_line", "spread_line_squared")
PBP_FEATURES: tuple[str, ...] = (
    "pbp_epa_diff",
    "pbp_success_diff",
    "pbp_yards_diff",
    "pbp_turnovers_lost_diff",
    "pbp_off_td_diff",
    "pbp_plays_diff",
)


def latest_snapshot(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no pbp snapshot under {root}")
    return candidates[-1]


def load_population(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = regular_season_rows(pd.read_parquet(path))
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = (
        frame.loc[frame["result"].notna() & frame["spread_line"].notna()]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    completed["spread_line_squared"] = np.square(
        pd.to_numeric(completed["spread_line"], errors="coerce")
    )
    test = completed.loc[completed["season"].ge(START_SEASON)].reset_index(drop=True)
    if len(completed) < MIN_TRAIN_GAMES:
        raise ValueError("completed frame smaller than the min-training floor")
    return completed, test


def build_pbp_features(games: pd.DataFrame, snapshot: Path) -> pd.DataFrame:
    seasons = sorted(int(s) for s in games["season"].unique())
    frames = []
    for season in seasons:
        path = snapshot / f"season={season}" / "plays.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing pbp season file {path}")
        frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "game_id",
                    "posteam",
                    "epa",
                    "yards_gained",
                    "success",
                    "interception",
                    "fumble_lost",
                    "touchdown",
                ],
            )
        )
    plays = pd.concat(frames, ignore_index=True)
    plays = plays.loc[plays["game_id"].isin(set(games["game_id"])) & plays["posteam"].notna()]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "epa": float(pd.to_numeric(g["epa"], errors="coerce").fillna(0.0).sum()),
                "success": float(pd.to_numeric(g["success"], errors="coerce").fillna(0.0).mean()),
                "yards": float(pd.to_numeric(g["yards_gained"], errors="coerce").fillna(0.0).sum()),
                "turnovers": float(
                    (
                        g["interception"].fillna(0).astype(bool)
                        | g["fumble_lost"].fillna(0).astype(bool)
                    ).sum()
                ),
                "off_td": float(pd.to_numeric(g["touchdown"], errors="coerce").fillna(0.0).sum()),
                "plays": float(len(g)),
            }
        )

    per_side = plays.groupby(["game_id", "posteam"]).apply(_agg, include_groups=False).reset_index()
    sides = per_side.pivot(index="game_id", columns="posteam")
    home_map = games.drop_duplicates("game_id").set_index("game_id")["home_team"]

    def _pair(column: str) -> tuple[pd.Series, pd.Series]:
        long_values = (
            sides[column].rename_axis("game_id").rename_axis("team", axis=1).stack().rename("value")
        ).reset_index()
        tagged = long_values.merge(
            home_map.rename("home_team"), left_on="game_id", right_index=True, how="inner"
        )
        tagged = tagged.loc[tagged["value"].notna()]
        home = (
            tagged.loc[tagged["team"].eq(tagged["home_team"])]
            .set_index("game_id")["value"]
            .reindex(home_map.index)
        )
        away = (
            tagged.loc[tagged["team"].ne(tagged["home_team"])]
            .set_index("game_id")["value"]
            .reindex(home_map.index)
        )
        return home, away

    features = pd.DataFrame(index=home_map.index)
    for column, name in (
        ("epa", "pbp_epa_diff"),
        ("success", "pbp_success_diff"),
        ("yards", "pbp_yards_diff"),
        ("turnovers", "pbp_turnovers_lost_diff"),
        ("off_td", "pbp_off_td_diff"),
        ("plays", "pbp_plays_diff"),
    ):
        home, away = _pair(column)
        features[name] = home - away
    return features.reset_index().rename(columns={"index": "game_id"})


def leak_fit(
    frame: pd.DataFrame, feature_columns: tuple[str, ...], ridge_alpha: float = RIDGE_ALPHA
) -> MarginModel:
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"leak-fit design missing columns: {', '.join(missing)}")
    target = pd.to_numeric(frame["ats_margin"], errors="coerce")
    estimator = make_margin_estimator("ridge", ridge_alpha=ridge_alpha)
    estimator.fit(frame.loc[:, list(feature_columns)], target)
    fitted = np.asarray(estimator.predict(frame.loc[:, list(feature_columns)]), dtype=float)
    residuals = target.to_numpy(dtype=float) - fitted
    residuals = residuals[np.isfinite(residuals)]
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name="ridge",
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=tuple(feature_columns),
        training_rows=len(frame),
        distribution_rows=len(residuals),
        training_max_gameday=frame["gameday"].max().date().isoformat(),
    )


def score_arm(model: MarginModel, test: pd.DataFrame) -> dict[str, Any]:
    forecasts = model.predict(test, probability_method=PROBABILITY_METHOD)
    batch = test.copy()
    for column in forecasts.columns:
        batch[column] = forecasts[column].to_numpy()
    evaluated = batch.loc[batch["home_cover"].notna()].copy()
    actual = evaluated["home_cover"].astype(int).to_numpy()
    probability = evaluated["home_cover_probability"].to_numpy(dtype=float)
    correct = ((probability >= 0.5).astype(int) == actual).astype(float)
    brier = np.square(probability - actual)
    eps = np.finfo(float).eps
    clipped = np.clip(probability, eps, 1.0 - eps)
    logloss = -(actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped))
    season_rows: list[dict[str, Any]] = []
    for season, group in evaluated.assign(_correct=correct, _brier=brier, _logloss=logloss).groupby(
        "season", sort=True
    ):
        season_rows.append(
            {
                "season": int(str(season)),
                "games": len(group),
                "correct": int(group["_correct"].sum()),
                "accuracy": float(group["_correct"].mean()),
                "brier_score": float(group["_brier"].mean()),
                "log_loss": float(group["_logloss"].mean()),
            }
        )
    return {
        "feature_columns": list(model.feature_columns),
        "training_rows": int(model.training_rows),
        "distribution_rows": int(model.distribution_rows),
        "games_scored": len(batch),
        "games_evaluated": len(evaluated),
        "pushes_excluded": int(len(batch) - len(evaluated)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "brier_score": float(brier.mean()),
        "log_loss": float(logloss.mean()),
        "per_season": season_rows,
    }


def reference_rows() -> dict[str, Any]:
    summary = pd.read_csv(REFERENCE_SUMMARY)
    references: dict[str, Any] = {"artifact": str(REFERENCE_SUMMARY.relative_to(REPO))}
    for method in ("market", "market_residual"):
        row = summary.loc[summary["method"].eq(method)].iloc[0]
        references[method] = {
            "cover_accuracy": float(row["cover_accuracy"]),
            "cover_games": int(row["cover_games"]),
        }
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output or (REPO / "artifacts" / "leak_ceiling" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed, test = load_population(FEATURE_TABLE)
    print(
        f"population: {len(test)} scored games {int(test['season'].min())}-"
        f"{int(test['season'].max())}, leak training rows {len(completed)}"
    )

    arms: dict[str, dict[str, Any]] = {}

    print("arm A: market line alone + leak fit")
    arms["market_line_leak"] = score_arm(leak_fit(completed, MARKET_LINE_FEATURES), test)

    print("arm B: full weak_stack design + leak fit")
    weak_columns = margin_feature_columns("market_residual", "weak_stack")
    arms["weak_stack_leak"] = score_arm(leak_fit(completed, weak_columns), test)

    try:
        snapshot = latest_snapshot(PBP_ROOT)
        print(f"arm C: PBP same-game summaries + leak fit ({snapshot.name})")
        pbp = build_pbp_features(completed, snapshot)
        merged = completed.merge(pbp, on="game_id", how="inner")
        dropped = len(completed) - len(merged)
        kept_test = test.merge(pbp, on="game_id", how="inner")
        arms["pbp_same_game_leak"] = score_arm(leak_fit(merged, PBP_FEATURES), kept_test)
        arms["pbp_same_game_leak"]["training_rows_dropped_missing_pbp"] = int(dropped)
    except Exception as error:
        arms["pbp_same_game_leak"] = {"skipped": True, "reason": repr(error)}

    print("arm B2: full weak_stack design + leak fit, alpha=1 shrinkage sensitivity")
    arms["weak_stack_leak_alpha1"] = score_arm(
        leak_fit(completed, weak_columns, ridge_alpha=1.0), test
    )

    payload = {
        "schema": 1,
        "generated_at_utc": timestamp,
        "leak_declaration": (
            "TOTAL LEAK, DELIBERATE: every arm trains on the entire completed "
            "frame including the target seasons and the target games "
            "themselves. This is a positive-control ceiling instrument, not a "
            "model; its accuracy bounds every legitimate walk-forward model "
            "from above and must never be compared to a market edge."
        ),
        "population": {
            "start_season": START_SEASON,
            "end_season": int(test["season"].max()),
            "feature_table": str(FEATURE_TABLE.relative_to(REPO)),
            "games_scored": len(test),
            "min_train_games": MIN_TRAIN_GAMES,
            "probability_method": PROBABILITY_METHOD,
            "grading_contract": "forced pick at p>=0.5 over non-push games",
        },
        "arms": arms,
        "references_no_leak": reference_rows(),
        "seed": SEED,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    payload["provenance"] = artifact_provenance(
        {
            "arms": sorted(arms),
            "probability_method": PROBABILITY_METHOD,
            "ridge_alpha": RIDGE_ALPHA,
            "start_season": START_SEASON,
        },
        FEATURE_TABLE,
        project_root=REPO,
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="leak-ceiling-control",
        metrics={
            f"{name}.accuracy": arm.get("accuracy", float("nan")) for name, arm in arms.items()
        },
        notes=(
            "Deliberate-leak positive control: total-leak fits on market line, "
            "weak_stack design, and same-game PBP summaries; measures the "
            "absolute ATS classification ceiling of this grading structure. "
            "Read-only research instrument; no registry entry, no pick played."
        ),
        source="scripts/leak_ceiling_control.py",
    )

    for name, arm in arms.items():
        if arm.get("skipped"):
            print(f"{name}: SKIPPED ({arm['reason']})")
        else:
            print(
                f"{name}: {arm['correct']}/{arm['games_evaluated']} = "
                f"{100 * arm['accuracy']:.2f}% (brier {arm['brier_score']:.4f})"
            )
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_RIDGE,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
    build_fit_population,
)

BOOTSTRAP_DRAWS = 2000


def loso_probabilities(population: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=population.index, dtype=float)
    for held in sorted(int(value) for value in population["season"].unique()):
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
    return out


def season_block_bootstrap(
    frame: pd.DataFrame, draws: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    seasons = sorted(frame["season"].unique())
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(len(seasons), size=len(seasons), replace=True)
        pooled = pd.concat([frame.loc[frame["season"].eq(seasons[index])] for index in picked])
        effects[draw] = pooled["paired_diff"].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high), float((effects > 0).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    artifacts_root = REPO / "artifacts"
    data_root = REPO / "data"
    population, provenance = build_fit_population(artifacts_root, data_root)
    population["out_of_season_home_probability"] = loso_probabilities(population)
    scored = population.loc[population["out_of_season_home_probability"].notna()].copy()
    calibrated_home = scored["out_of_season_home_probability"].ge(0.5)
    scored["calibrated_correct"] = (
        calibrated_home.astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["paired_diff"] = scored["calibrated_correct"] - scored["model_correct"]
    effect_points = float(scored["paired_diff"].mean() * 100.0)
    low, high, probability_positive = season_block_bootstrap(
        scored[["season", "paired_diff"]], args.draws, args.seed
    )
    low_points, high_points = low * 100.0, high * 100.0
    per_season = (
        scored.groupby("season")["paired_diff"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "diff_points_fraction", "size": "games"})
    )
    per_season["diff_points"] = per_season["diff_points_fraction"] * 100.0
    decisive = scored.loc[scored["paired_diff"].ne(0.0)]
    cal_wins = int((decisive["paired_diff"] > 0).sum())
    model_wins = int((decisive["paired_diff"] < 0).sum())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = artifacts_root / "pooled_signal" / stamp
    directory.mkdir(parents=True, exist_ok=True)
    results = {
        "command": "python scripts/pooled_signal_paired_eval.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "effect_accuracy_points": effect_points,
        "interval_low": low_points,
        "interval_high": high_points,
        "probability_positive": probability_positive,
        "sample_games": len(scored),
        "sample_blocks": int(scored["season"].nunique()),
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "calibrated_record": (
            f"{int(scored['calibrated_correct'].sum())}-"
            f"{len(scored) - int(scored['calibrated_correct'].sum())}"
        ),
        "model_only_record": (
            f"{int(scored['model_correct'].sum())}-"
            f"{len(scored) - int(scored['model_correct'].sum())}"
        ),
        "decisive_games": len(decisive),
        "decisive_calibrated_wins": cal_wins,
        "decisive_model_wins": model_wins,
        "per_season": {
            str(season): {
                "diff_points": float(row["diff_points"]),
                "games": int(row["games"]),
            }
            for season, row in per_season.iterrows()
        },
        "provenance": provenance,
    }
    (directory / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# MOD-20 unit 1 close-out: pooled first model vs model-only, paired LOSO",
        "",
        f"Effect {effect_points:+.3f} accuracy points "
        f"[{low_points:+.3f}, {high_points:+.3f}], "
        f"P+ {probability_positive:.4f}, season-block bootstrap {args.draws} draws.",
        f"Calibrated {results['calibrated_record']} vs "
        f"model-only {results['model_only_record']} on {len(scored)} games; "
        f"decisive {len(decisive)} ({cal_wins}-{model_wins}).",
        "",
        "Per-season diff (points): "
        + "; ".join(
            f"{season} {detail['diff_points']:+.2f} ({detail['games']})"
            for season, detail in results["per_season"].items()
        ),
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(directory.relative_to(REPO)), **results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

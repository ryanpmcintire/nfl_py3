"""MOD-18 lane AH: score the walk-forward reliability transform on the displayed score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.card_view import resolve_card_view
from nfl_ats.displayed_confidence import (
    DISPLAY_BUCKETS,
    DISPLAYED_CONFIDENCE_POLICY,
    DISPLAYED_PICK_PROBABILITY_COLUMN,
    PROBABILITY_BANDS,
    PSEUDO_OBSERVATIONS,
    ProductionDisplayedConfidence,
    archive_display_stream,
    attach_displayed_confidence,
    display_spread_bucket,
    fit_reliability_cells,
    probability_band,
    walk_forward_displayed_confidence,
)
from nfl_ats.evidence_conventions import probability_positive_from_draws

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260821


def _brier(displayed: np.ndarray, correct: np.ndarray) -> np.ndarray:
    return (displayed - correct) ** 2


def _log_loss(displayed: np.ndarray, correct: np.ndarray) -> np.ndarray:
    clipped = np.clip(displayed, 1e-12, 1 - 1e-12)
    return -(correct * np.log(clipped) + (1.0 - correct) * np.log(1.0 - clipped))


def _paired_block_bootstrap(
    per_game_difference: np.ndarray, blocks: np.ndarray
) -> tuple[float, float, float]:
    codes, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=per_game_difference, minlength=codes.size)
    counts = np.bincount(inverse, minlength=codes.size).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, codes.size, size=(BOOTSTRAP_SAMPLES, codes.size))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper), float(probability_positive_from_draws(draws))


def _metric_row(
    name: str,
    scope: str,
    stated_metric: np.ndarray,
    calibrated_metric: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, object]:
    difference = stated_metric - calibrated_metric
    lower, upper, positive = _paired_block_bootstrap(difference, blocks)
    return {
        "metric": name,
        "scope": scope,
        "games": int(difference.size),
        "stated": float(stated_metric.mean()),
        "calibrated": float(calibrated_metric.mean()),
        "difference": float(difference.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "probability_positive": positive,
    }


def _reliability_table(displayed: pd.Series, correct: pd.Series, label: str) -> pd.DataFrame:
    band = probability_band(displayed)
    rows = []
    for level in PROBABILITY_BANDS:
        mask = band.eq(level)
        n = int(mask.sum())
        rows.append(
            {
                "stage": label,
                "band": level,
                "games": n,
                "mean_displayed": float(displayed.loc[mask].mean()) if n else float("nan"),
                "realised_accuracy": float(correct.loc[mask].mean()) if n else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    table["gap_points"] = 100.0 * (table["realised_accuracy"] - table["mean_displayed"])
    return table


def _bucket_diagnosis(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in DISPLAY_BUCKETS:
        group = scored.loc[scored["bucket"].eq(bucket)]
        rows.append(
            {
                "bucket": bucket,
                "games": len(group),
                "mean_stated": float(group["stated"].mean()),
                "realised_accuracy": float(group["correct"].mean()),
                "mean_calibrated": float(group["calibrated"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["stated_gap_points"] = 100.0 * (table["realised_accuracy"] - table["mean_stated"])
    table["calibrated_gap_points"] = 100.0 * (table["realised_accuracy"] - table["mean_calibrated"])
    return table


def _week1_display(
    forecast: Path, data_root: Path, calibration: ProductionDisplayedConfidence
) -> pd.DataFrame:
    """The card the reader actually sees, through the served overlay path."""

    predictions = pd.read_csv(forecast / "recommendations.csv")
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    sweep_path = forecast / "line_sweep.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else pd.DataFrame()
    view = resolve_card_view(
        predictions,
        sweep,
        metadata,
        data_root=data_root,
        require_fresh_arrest_overlay=False,
    )
    served = view.predictions.reset_index(drop=True)
    pick_home_before = pd.to_numeric(served["home_cover_probability"], errors="raise").ge(0.5)
    pick_team_before = served["home_team"].where(pick_home_before, served["away_team"])
    card = attach_displayed_confidence(served, calibration)
    home_probability = pd.to_numeric(card["home_cover_probability"], errors="raise")
    pick_home = home_probability.ge(0.5)
    before = home_probability.where(pick_home, 1.0 - home_probability)
    after = pd.to_numeric(card[DISPLAYED_PICK_PROBABILITY_COLUMN], errors="raise")
    line = pd.to_numeric(card["spread_line"], errors="raise")
    pick_team = card["home_team"].where(pick_home, card["away_team"])
    raw = pd.read_csv(forecast / "recommendations.csv").set_index("game_id")
    raw_home = pd.to_numeric(raw["home_cover_probability"], errors="raise")
    return pd.DataFrame(
        {
            "game_id": card["game_id"].astype(str),
            "matchup": card["away_team"] + " at " + card["home_team"],
            "pick_team": pick_team,
            "home_spread_line": line,
            "line_size": line.abs(),
            "bucket": display_spread_bucket(line),
            "band": probability_band(before),
            "displayed_before": before,
            "displayed_after": after,
            "change_points": 100.0 * (after - before),
            "side_changed": pick_team.ne(pick_team_before),
            "model_home_cover_probability": card["game_id"].astype(str).map(raw_home),
            "card_home_cover_probability": home_probability,
        }
    ).sort_values("displayed_before", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    per_game = pd.read_parquet(arguments.archive / "per_game.parquet")
    stream = archive_display_stream(per_game)
    scored = stream.loc[stream["correct"].notna()].copy()
    scored["calibrated"] = walk_forward_displayed_confidence(scored)

    stated = scored["stated"].to_numpy(dtype=float)
    calibrated = scored["calibrated"].to_numpy(dtype=float)
    correct = scored["correct"].to_numpy(dtype=float)
    blocks = (
        scored["season"].astype(str) + "-" + scored["week"].astype(str).str.zfill(2)
    ).to_numpy()

    rows: list[dict[str, object]] = []
    for name, metric in (("brier", _brier), ("log_loss", _log_loss)):
        rows.append(
            _metric_row(
                name, "overall", metric(stated, correct), metric(calibrated, correct), blocks
            )
        )
        for bucket in DISPLAY_BUCKETS:
            mask = scored["bucket"].eq(bucket).to_numpy()
            rows.append(
                _metric_row(
                    name,
                    bucket,
                    metric(stated[mask], correct[mask]),
                    metric(calibrated[mask], correct[mask]),
                    blocks[mask],
                )
            )
    metrics = pd.DataFrame(rows)

    reliability = pd.concat(
        [
            _reliability_table(scored["stated"], scored["correct"], "before"),
            _reliability_table(scored["calibrated"], scored["correct"], "after"),
        ],
        ignore_index=True,
    )

    diagnosis = _bucket_diagnosis(scored)
    calibration = ProductionDisplayedConfidence(
        policy=DISPLAYED_CONFIDENCE_POLICY,
        cells=fit_reliability_cells(stream),
        source_path=str(arguments.archive).replace("\\", "/"),
        source_model_id=None,
        active_model_id=None,
        prior_rows=int(stream["correct"].notna().sum()),
        warnings=(),
    )
    cells = calibration.cells.to_frame()
    week1 = _week1_display(arguments.forecast, arguments.data_root, calibration)

    arguments.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(arguments.out / "metrics.csv", index=False)
    reliability.to_csv(arguments.out / "reliability.csv", index=False)
    diagnosis.to_csv(arguments.out / "bucket_diagnosis.csv", index=False)
    cells.to_csv(arguments.out / "calibration_cells.csv", index=False)
    week1.to_csv(arguments.out / "week1_display.csv", index=False)
    scored.to_csv(arguments.out / "per_game_display.csv", index=False)
    (arguments.out / "summary.json").write_text(
        json.dumps(
            {
                "archive": str(arguments.archive).replace("\\", "/"),
                "forecast": str(arguments.forecast).replace("\\", "/"),
                "games": len(scored),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "pseudo_observations": PSEUDO_OBSERVATIONS,
                "metrics": rows,
                "sides_changed": int(week1["side_changed"].sum()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print()
    print(diagnosis.to_string(index=False))
    print()
    print(reliability.to_string(index=False))
    print()
    print(cells.to_string(index=False))
    print()
    print(week1.to_string(index=False))


if __name__ == "__main__":
    main()

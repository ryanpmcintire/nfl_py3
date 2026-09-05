"""MOD-11 screen: does a regime-conditional calibrator beat the pooled one?

Predeclaration (frozen before this script was written or run):
``C:\\Users\\Ryan\\AppData\\Local\\Temp\\claude\\F--Repos-nfl-py3\\
c8c5fbdd-027f-438d-b992-979e83a91c2e\\scratchpad\\mod11_scope\\predeclaration.md``.
Design this predeclaration implements:
``...\\scratchpad\\mod11_scope\\design.md``.

CFB only, free under rotation rule 8 (``docs/rotation_registry.md``): this
script never calls ``nfl_ats.rotation.assign_window``/``record_look``, and
never touches ``registry/*.json``. Two predeclared regime candidates run
this pass -- **line magnitude (primary) and season-era (secondary)** -- each
swept across the project's four calibration methods
(``none``/``platt``/``isotonic``/``beta``). That is the full multiplicity
budget: 2 regimes x 4 methods = 8 paired comparisons, one predeclared
primary and one predeclared secondary, never independently multiplied.

``nfl_ats.calibration.calibrate_cover_prediction_stream`` (unmodified,
imported directly) is the ``pooled`` control arm: today's production
calibration machinery, which conditions on chronology only. The two
candidate arms (``line_bucket``, ``season_era``) are produced by a
script-local ``stratified_calibrate`` that mirrors that function's exact
chronology filter and reuses its private per-method fit
(``nfl_ats.calibration._calibrated_probabilities``) -- adding only one
restriction: calibration history is further filtered to rows sharing the
target week's regime bucket. Nothing in ``src/`` is edited or extended;
implementation shape (A), fully stratified calibrators, is the orchestrator's
adjudicated shape (design decision 2).

Calibration changes are pick-moving here, never free (MOD-06, AGENTS.md
binding): the primary metric is paired forced-pick accuracy improvement,
never a calibration metric alone, and every arm's pick-flip count/location is
reported regardless of whether that arm clears any convention threshold. An
interval containing zero is never grounds to close this line of work; the
0.75 probability_positive figure used below is the project's own established
screen-stage citation convention (``ecdf_smoothing.md``/SPEC-5), not a
decision bar (adjudication 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.calibration import (  # noqa: E402
    COVER_CALIBRATION_METHODS,
    _calibrated_probabilities,  # reused, not reimplemented -- adjudication 2
    calibrate_cover_prediction_stream,
)
from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    CFB_CLEAN_CORE_SEASONS,
    cfb_evaluation_window,
    cfb_walk_forward_benchmark,
)
from nfl_ats.constants import DEFAULT_MIN_CALIBRATION_GAMES  # noqa: E402
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.key_numbers import cover_reliability_by_line_bucket, line_bucket  # noqa: E402

DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

REGIME_COLUMNS: tuple[str, ...] = ("line_bucket", "season_era")
CONTROL_ARM = "pooled"
DERIVATION_METHOD = "platt"
DERIVATION_FEASIBILITY_MIN_ROWS = 10
DERIVATION_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 100),
    (100, 200),
    (200, 400),
    (400, 800),
    (800, None),
)
MULTIPLICITY_NOTE = (
    "predeclared: 2 regime candidates this pass (line_bucket PRIMARY, "
    "season_era SECONDARY) x 4 calibration methods (none/platt/isotonic/"
    "beta) = 8 paired comparisons; multiplicity budget is the 2-regime axis, "
    "not further split per method or per block type."
)


def _regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["line_bucket"] = line_bucket(frame["spread_line"])
    frame["season_era"] = frame["season"].map(lambda season: cfb_evaluation_window(int(season)))
    if frame["line_bucket"].eq("unclassified").any():
        raise ValueError("line_bucket produced an unclassified row on CFB completed games")
    return frame


def build_raw_stream(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_games: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Walk-forward market_residual predictions, unmodified frozen recipe."""

    result = cfb_walk_forward_benchmark(
        features,
        start_season=start_season,
        end_season=end_season,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
    )
    predictions = result.predictions.loc[result.predictions["method"].eq("market_residual")].copy()
    predictions = _regime_columns(predictions)
    return predictions.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 2 of the predeclaration: derive the per-bucket calibration floor.
# ---------------------------------------------------------------------------


def derive_bucket_calibration_floor(
    predictions: pd.DataFrame,
    *,
    regime_column: str = "line_bucket",
    method: str = DERIVATION_METHOD,
    feasibility_min_rows: int = DERIVATION_FEASIBILITY_MIN_ROWS,
    bins: tuple[tuple[int, int | None], ...] = DERIVATION_BINS,
) -> tuple[int, pd.DataFrame, bool]:
    """Bucket-local analogue of the pooled DEFAULT_MIN_CALIBRATION_GAMES derivation.

    Mirrors ``docs/rotation_registry.md`` rule 9's own methodology for the
    pooled 200 (bucket calibrated-vs-raw Brier by history size), restricted
    to bucket-LOCAL history rather than pooled history. Returns
    ``(derived_floor, bin_summary, demonstrated)``; when no bin shows a
    reproducible improvement, ``demonstrated`` is False and the floor falls
    back to the largest tested bin's lower edge, flagged as unproven.
    """

    frame = predictions.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    frame = frame.loc[frame["home_cover"].notna() & frame["home_cover_probability"].notna()].copy()
    frame["_regime"] = frame[regime_column]

    raw_errors: list[float] = []
    cal_errors: list[float] = []
    history_sizes: list[int] = []

    for (_season, _week), weekly_rows in frame.groupby(["season", "week"], sort=True):
        cutoff = weekly_rows["gameday"].min()
        history_all = frame.loc[frame["gameday"].lt(cutoff)]
        for bucket, bucket_rows in weekly_rows.groupby("_regime"):
            bucket_history = history_all.loc[history_all["_regime"].eq(bucket)]
            n_history = len(bucket_history)
            if n_history < feasibility_min_rows:
                continue
            outcomes = bucket_history["home_cover"].to_numpy(dtype=int)
            if len(np.unique(outcomes)) < 2:
                continue
            training_probability = bucket_history["home_cover_probability"].to_numpy(dtype=float)
            target_probability = bucket_rows["home_cover_probability"].to_numpy(dtype=float)
            calibrated = _calibrated_probabilities(
                method, training_probability, outcomes, target_probability
            )
            actual = bucket_rows["home_cover"].to_numpy(dtype=float)
            raw_errors.extend(((target_probability - actual) ** 2).tolist())
            cal_errors.extend(((calibrated - actual) ** 2).tolist())
            history_sizes.extend([n_history] * len(bucket_rows))

    detail = pd.DataFrame(
        {
            "bucket_history_rows": history_sizes,
            "raw_sq_error": raw_errors,
            "calibrated_sq_error": cal_errors,
        }
    )
    if detail.empty:
        raise ValueError("Derivation pass produced no eligible (bucket, week) cells")

    def _bin_label(n: int) -> str:
        for lo, hi in bins:
            if hi is None:
                if n >= lo:
                    return f"{lo}+"
            elif lo <= n < hi:
                return f"{lo}-{hi - 1}"
        raise ValueError(f"history size {n} did not fall into any declared bin")

    detail["history_bin"] = detail["bucket_history_rows"].map(_bin_label)
    order = [f"{lo}-{hi - 1}" if hi is not None else f"{lo}+" for lo, hi in bins]
    summary = (
        detail.groupby("history_bin")
        .agg(
            games=("bucket_history_rows", "count"),
            raw_brier=("raw_sq_error", "mean"),
            calibrated_brier=("calibrated_sq_error", "mean"),
        )
        .reset_index()
    )
    summary["brier_improvement"] = summary["raw_brier"] - summary["calibrated_brier"]
    summary["_order"] = summary["history_bin"].map({label: i for i, label in enumerate(order)})
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # Decision rule: smallest bin where improvement holds AND every larger
    # bin also improves.
    derived_floor: int | None = None
    demonstrated = False
    present = list(summary["history_bin"])
    improves = dict(zip(summary["history_bin"], summary["brier_improvement"] > 0.0, strict=True))
    for lo, hi in bins:
        label = f"{lo}-{hi - 1}" if hi is not None else f"{lo}+"
        if label not in present:
            continue
        remaining_labels = [
            f"{blo}-{bhi - 1}" if bhi is not None else f"{blo}+"
            for blo, bhi in bins
            if blo >= lo and (f"{blo}-{bhi - 1}" if bhi is not None else f"{blo}+") in present
        ]
        if remaining_labels and all(improves[label2] for label2 in remaining_labels):
            derived_floor = lo
            demonstrated = True
            break

    if derived_floor is None:
        derived_floor = bins[-1][0]

    return derived_floor, summary, demonstrated


# ---------------------------------------------------------------------------
# Section 3 of the predeclaration: the stratified calibrator.
# ---------------------------------------------------------------------------


def stratified_calibrate(
    predictions: pd.DataFrame,
    *,
    method: str,
    regime_column: str,
    min_calibration_games: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bucket-local analogue of calibrate_cover_prediction_stream.

    Mirrors that function's chronology-only history filter exactly, adding
    one restriction: history is further filtered to rows sharing the target
    row's regime bucket. A bucket-week whose bucket-local history is below
    ``min_calibration_games``, or has only one outcome class, falls back to
    the raw (uncalibrated) probability for that bucket-week -- reported, not
    silently dropped -- rather than raising, since a stratified screen is
    expected to run into thin cells by construction.
    """

    if method not in COVER_CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    frame = predictions.sort_values(["gameday", "game_id"]).reset_index(drop=True).copy()
    frame["_regime"] = frame[regime_column]

    calibrated_probability = frame["home_cover_probability"].to_numpy(dtype=float).copy()
    bucket_history_rows = np.zeros(len(frame), dtype=int)
    fell_back = np.zeros(len(frame), dtype=bool)

    if method != "none":
        for (_season, _week), weekly_rows in frame.groupby(["season", "week"], sort=True):
            cutoff = weekly_rows["gameday"].min()
            history_all = frame.loc[
                frame["gameday"].lt(cutoff)
                & frame["home_cover"].notna()
                & frame["home_cover_probability"].notna()
            ]
            for bucket, bucket_rows in weekly_rows.groupby("_regime"):
                positions = frame.index.get_indexer(bucket_rows.index)
                bucket_history = history_all.loc[history_all["_regime"].eq(bucket)]
                n_history = len(bucket_history)
                bucket_history_rows[positions] = n_history
                outcomes = bucket_history["home_cover"].to_numpy(dtype=int)
                if n_history < min_calibration_games or len(np.unique(outcomes)) < 2:
                    fell_back[positions] = True
                    continue
                training_probability = bucket_history["home_cover_probability"].to_numpy(
                    dtype=float
                )
                target_probability = bucket_rows["home_cover_probability"].to_numpy(dtype=float)
                calibrated = _calibrated_probabilities(
                    method, training_probability, outcomes, target_probability
                )
                calibrated_probability[positions] = calibrated

    result = frame.copy()
    result["home_cover_probability"] = calibrated_probability
    result["bucket_history_rows"] = bucket_history_rows
    result["fell_back_to_raw"] = fell_back
    result = result.drop(columns="_regime")
    diagnostics = {
        "method": method,
        "regime_column": regime_column,
        "min_calibration_games": min_calibration_games,
        "scored_rows": len(result),
        "fallback_rows": int(fell_back.sum()) if method != "none" else 0,
        "fallback_rate": float(fell_back.mean()) if method != "none" else 0.0,
    }
    return result, diagnostics


def first_season_with_sufficient_pooled_history(
    predictions: pd.DataFrame, *, min_calibration_games: int, safety_margin: int = 10
) -> int:
    """Earliest season whose first scored week already has enough PRIOR history.

    ``calibrate_cover_prediction_stream`` raises if any evaluated week's prior
    history is short of ``min_calibration_games`` -- by design (production
    behaviour, unmodified). Since history only grows within a season, once a
    season's first week clears the floor every later week in that season
    does too, so checking just the first week per season is sufficient.
    """

    frame = predictions.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    for season, season_rows in frame.groupby("season", sort=True):
        cutoff = season_rows["gameday"].min()
        prior = frame.loc[
            frame["gameday"].lt(cutoff)
            & frame["home_cover"].notna()
            & frame["home_cover_probability"].notna()
        ]
        if len(prior) >= min_calibration_games + safety_margin:
            return int(season)
    raise ValueError("No season in this stream ever accumulates sufficient prior history")


def pooled_calibrate(
    predictions: pd.DataFrame, *, method: str, evaluation_start_season: int
) -> pd.DataFrame:
    """Today's production calibrator: unmodified, pooled history only."""

    calibrated = calibrate_cover_prediction_stream(
        predictions,
        method=method,
        evaluation_start_season=evaluation_start_season,
        min_calibration_games=DEFAULT_MIN_CALIBRATION_GAMES,
    )
    return calibrated


# ---------------------------------------------------------------------------
# Section 4 of the predeclaration: split-half calibration-gap reliability.
# ---------------------------------------------------------------------------


def _cover_reliability_by_regime(predictions: pd.DataFrame, *, regime_column: str) -> pd.DataFrame:
    """Same statistic as key_numbers.cover_reliability_by_line_bucket, any regime column."""

    frame = predictions.copy()
    frame["_probability"] = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    frame["_actual"] = pd.to_numeric(frame["home_cover"], errors="coerce")
    frame = frame.dropna(subset=["_probability", "_actual"])
    rows = []
    for bucket, group in frame.groupby(regime_column, sort=False):
        mean_predicted = float(group["_probability"].mean())
        realized = float(group["_actual"].mean())
        rows.append(
            {
                "bucket": bucket,
                "games": len(group),
                "mean_predicted_probability": mean_predicted,
                "realized_cover_rate": realized,
                "calibration_gap": mean_predicted - realized,
            }
        )
    return pd.DataFrame(rows)


def split_half_reliability(
    raw_clean_core: pd.DataFrame, *, regime_column: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Odd/even season split; sign agreement + Spearman rank correlation of gaps."""

    seasons = sorted(raw_clean_core["season"].unique())
    odd = raw_clean_core.loc[raw_clean_core["season"] % 2 == 1]
    even = raw_clean_core.loc[raw_clean_core["season"] % 2 == 0]

    if regime_column == "line_bucket":
        odd_gap = cover_reliability_by_line_bucket(odd).set_index("line_bucket")["calibration_gap"]
        even_gap = cover_reliability_by_line_bucket(even).set_index("line_bucket")[
            "calibration_gap"
        ]
    else:
        odd_gap = _cover_reliability_by_regime(odd, regime_column=regime_column).set_index(
            "bucket"
        )["calibration_gap"]
        even_gap = _cover_reliability_by_regime(even, regime_column=regime_column).set_index(
            "bucket"
        )["calibration_gap"]

    all_buckets = sorted(set(odd_gap.index) | set(even_gap.index))
    table_rows = []
    paired_odd = []
    paired_even = []
    for bucket in all_buckets:
        odd_value = float(odd_gap.get(bucket, np.nan))
        even_value = float(even_gap.get(bucket, np.nan))
        measurable = np.isfinite(odd_value) and np.isfinite(even_value)
        table_rows.append(
            {
                "bucket": bucket,
                "odd_season_gap": odd_value,
                "even_season_gap": even_value,
                "measurable": measurable,
                "sign_agrees": (np.sign(odd_value) == np.sign(even_value)) if measurable else None,
            }
        )
        if measurable:
            paired_odd.append(odd_value)
            paired_even.append(even_value)

    table = pd.DataFrame(table_rows)
    n_measurable = len(paired_odd)
    sign_agreement_rate = (
        float(
            np.mean(
                [np.sign(a) == np.sign(b) for a, b in zip(paired_odd, paired_even, strict=True)]
            )
        )
        if n_measurable
        else float("nan")
    )
    if n_measurable >= 3:
        rho, pvalue = spearmanr(paired_odd, paired_even)
        rho = float(rho)
        pvalue = float(pvalue)
    else:
        rho, pvalue = float("nan"), float("nan")

    summary = {
        "regime_column": regime_column,
        "odd_seasons": [int(s) for s in seasons if s % 2 == 1],
        "even_seasons": [int(s) for s in seasons if s % 2 == 0],
        "buckets_total": len(all_buckets),
        "buckets_measurable": n_measurable,
        "sign_agreement_rate": sign_agreement_rate,
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
    }
    return table, summary


# ---------------------------------------------------------------------------
# Section 6 of the predeclaration: pick-flip reporting.
# ---------------------------------------------------------------------------


def flip_report(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    residual_quartile_edges: np.ndarray,
    label: str,
) -> dict[str, Any]:
    control_indexed = control.set_index("game_id")
    candidate_indexed = candidate.set_index("game_id")
    common = control_indexed.index.intersection(candidate_indexed.index)
    control_indexed = control_indexed.loc[common]
    candidate_indexed = candidate_indexed.loc[common]

    control_side = control_indexed["home_cover_probability"].ge(0.5)
    candidate_side = candidate_indexed["home_cover_probability"].ge(0.5)
    flipped = control_side.ne(candidate_side)

    all_games = control_indexed.copy()
    all_games["line_bucket"] = line_bucket(all_games["spread_line"])
    flipped_games = all_games.loc[flipped]

    flip_rate_by_bucket = (
        flipped_games.groupby("line_bucket").size() / all_games.groupby("line_bucket").size()
    ).fillna(0.0)

    abs_residual = all_games["predicted_market_residual"].abs()
    quartile_labels = pd.cut(
        abs_residual, bins=residual_quartile_edges, include_lowest=True, duplicates="drop"
    )
    all_games["_residual_quartile"] = quartile_labels.astype(str)
    flip_rate_by_quartile = (
        all_games.loc[flipped].groupby("_residual_quartile").size()
        / all_games.groupby("_residual_quartile").size()
    ).fillna(0.0)

    flipped_abs = flipped_games["predicted_market_residual"].abs()
    return {
        "label": label,
        "games": len(common),
        "flipped": int(flipped.sum()),
        "flip_rate": float(flipped.mean()) if len(common) else float("nan"),
        "flip_rate_by_line_bucket": {str(k): float(v) for k, v in flip_rate_by_bucket.items()},
        "flip_rate_by_residual_quartile": {
            str(k): float(v) for k, v in flip_rate_by_quartile.items()
        },
        "flipped_mean_abs_predicted_residual": float(flipped_abs.mean())
        if len(flipped_abs)
        else float("nan"),
        "all_mean_abs_predicted_residual": float(abs_residual.mean()),
        "flipped_median_abs_predicted_residual": float(flipped_abs.median())
        if len(flipped_abs)
        else float("nan"),
        "all_median_abs_predicted_residual": float(abs_residual.median()),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_paired_comparison(
    *,
    pooled_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    candidate_name: str,
    method: str,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """One regime vs its own pooled control, on their shared headline population.

    Two calls per method (line_bucket-vs-pooled-on-clean-core,
    season_era-vs-pooled-on-full-window) rather than one shared-baseline call
    with three feature sets, because the two regimes' headline populations
    differ -- see predeclaration.md section 5's implementation-time
    correction.
    """

    columns = ["game_id", "season", "week", "home_cover", "home_cover_probability"]
    pooled = pooled_predictions.loc[:, columns].copy()
    pooled["feature_set"] = CONTROL_ARM
    candidate = candidate_predictions.loc[:, columns].copy()
    candidate["feature_set"] = candidate_name
    combined = pd.concat([pooled, candidate], ignore_index=True)

    rows = []
    for block in ("week", "season"):
        paired = paired_feature_comparisons(
            combined,
            baseline_feature_set=CONTROL_ARM,
            samples=samples,
            block=block,
            seed=seed,
            on_degenerate="raise",
        )
        paired.insert(0, "calibration_method", method)
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    parser.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    parser.add_argument("--min-train-games", type=int, default=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    parser.add_argument("--ridge-alpha", type=float, default=CFB_BENCHMARK_RIDGE_ALPHA)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.time()

    print(f"[{time.time() - t0:7.1f}s] Loading CFB features and fitting the walk-forward stream")
    features = pd.read_parquet(args.cfb_features)
    raw_stream = build_raw_stream(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
    )
    timings["raw_stream_seconds"] = time.time() - t0
    print(
        f"[{time.time() - t0:7.1f}s] raw stream: {len(raw_stream)} scored games, "
        f"{raw_stream.groupby(['season', 'week']).ngroups} weeks"
    )
    raw_stream.to_parquet(output / "raw_stream.parquet", index=False)

    clean_core_raw = raw_stream.loc[raw_stream["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()

    # --- Section 4: reliability BEFORE any accuracy claim -----------------
    print(f"[{time.time() - t0:7.1f}s] Split-half calibration-gap reliability (before accuracy)")
    line_reliability_table, line_reliability_summary = split_half_reliability(
        clean_core_raw, regime_column="line_bucket"
    )
    era_reliability_table, era_reliability_summary = split_half_reliability(
        raw_stream, regime_column="season_era"
    )
    line_reliability_table.to_csv(output / "reliability_line_bucket.csv", index=False)
    era_reliability_table.to_csv(output / "reliability_season_era.csv", index=False)

    # --- Section 2: derive the per-bucket calibration floor ---------------
    print(f"[{time.time() - t0:7.1f}s] Deriving the per-bucket calibration-games floor")
    derived_floor, derivation_summary, demonstrated = derive_bucket_calibration_floor(
        clean_core_raw, regime_column="line_bucket"
    )
    derivation_summary.to_csv(output / "derivation_bin_summary.csv", index=False)
    print(
        f"[{time.time() - t0:7.1f}s] derived per-bucket min_calibration_games = "
        f"{derived_floor} (demonstrated={demonstrated})"
    )
    print(derivation_summary.to_string(index=False))

    # --- Section 5/3: score every arm x method -----------------------------
    residual_quartile_edges = np.quantile(
        raw_stream["predicted_market_residual"].abs(), [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    residual_quartile_edges[0] = -np.inf
    residual_quartile_edges[-1] = np.inf

    paired_frames: list[pd.DataFrame] = []
    flip_reports: list[dict[str, Any]] = []
    stratified_diagnostics: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []

    clean_eval_start = first_season_with_sufficient_pooled_history(
        clean_core_raw, min_calibration_games=DEFAULT_MIN_CALIBRATION_GAMES
    )
    full_eval_start = first_season_with_sufficient_pooled_history(
        raw_stream, min_calibration_games=DEFAULT_MIN_CALIBRATION_GAMES
    )
    print(
        f"[{time.time() - t0:7.1f}s] pooled-calibration evaluation start: "
        f"clean_core={clean_eval_start}, full_window={full_eval_start} "
        f"(earliest season whose first week already clears {DEFAULT_MIN_CALIBRATION_GAMES} "
        "prior pooled rows within that population)"
    )

    for method in COVER_CALIBRATION_METHODS:
        print(f"[{time.time() - t0:7.1f}s] method={method}: pooled + line_bucket + season_era")

        pooled_clean = pooled_calibrate(
            clean_core_raw, method=method, evaluation_start_season=clean_eval_start
        )
        pooled_full = pooled_calibrate(
            raw_stream, method=method, evaluation_start_season=full_eval_start
        )

        line_bucket_clean, line_diag = stratified_calibrate(
            clean_core_raw,
            method=method,
            regime_column="line_bucket",
            min_calibration_games=derived_floor,
        )
        season_era_full, era_diag = stratified_calibrate(
            raw_stream,
            method=method,
            regime_column="season_era",
            min_calibration_games=derived_floor,
        )
        stratified_diagnostics.append(line_diag)
        stratified_diagnostics.append(era_diag)

        for frame, tag in (
            (
                pooled_clean.assign(feature_set=CONTROL_ARM, calibration_method=method),
                "pooled_clean",
            ),
            (
                line_bucket_clean.assign(feature_set="line_bucket", calibration_method=method),
                "line_bucket_clean",
            ),
            (pooled_full.assign(feature_set=CONTROL_ARM, calibration_method=method), "pooled_full"),
            (
                season_era_full.assign(feature_set="season_era", calibration_method=method),
                "season_era_full",
            ),
        ):
            keep = [
                "game_id",
                "season",
                "week",
                "gameday",
                "spread_line",
                "line_bucket",
                "season_era",
                "home_cover",
                "home_cover_probability",
                "predicted_market_residual",
                "feature_set",
                "calibration_method",
            ]
            scored_frames.append(
                frame.loc[:, [c for c in keep if c in frame.columns]].assign(population=tag)
            )

        # line_bucket headline: clean-core only, vs pooled (clean-core-fit)
        paired_line = run_paired_comparison(
            pooled_predictions=pooled_clean,
            candidate_predictions=line_bucket_clean,
            candidate_name="line_bucket",
            method=method,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        paired_line["headline_population"] = "clean_core_for_line_bucket"
        paired_frames.append(paired_line)

        # season_era headline: full window, vs pooled (full-window-fit)
        paired_era = run_paired_comparison(
            pooled_predictions=pooled_full,
            candidate_predictions=season_era_full,
            candidate_name="season_era",
            method=method,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        paired_era["headline_population"] = "full_window_for_season_era"
        paired_frames.append(paired_era)

        if method != "none":
            flip_reports.append(
                flip_report(
                    pooled_clean,
                    line_bucket_clean,
                    residual_quartile_edges=residual_quartile_edges,
                    label=f"line_bucket__{method}",
                )
            )
            flip_reports.append(
                flip_report(
                    pooled_full,
                    season_era_full,
                    residual_quartile_edges=residual_quartile_edges,
                    label=f"season_era__{method}",
                )
            )
        else:
            flip_reports.append(
                {
                    "label": f"line_bucket__{method}",
                    "note": "method=none is an identity check (no transform); flips expected zero",
                    "games": len(pooled_clean),
                    "flipped": 0,
                    "flip_rate": 0.0,
                }
            )
            flip_reports.append(
                {
                    "label": f"season_era__{method}",
                    "note": "method=none is an identity check (no transform); flips expected zero",
                    "games": len(pooled_full),
                    "flipped": 0,
                    "flip_rate": 0.0,
                }
            )

    all_paired = pd.concat(paired_frames, ignore_index=True)
    all_paired.to_csv(output / "paired_comparisons.csv", index=False)
    pd.concat(scored_frames, ignore_index=True).to_parquet(
        output / "scored_predictions.parquet", index=False
    )

    print(f"\n[{time.time() - t0:7.1f}s] === Week-blocked accuracy, both regimes, all methods ===")
    print(MULTIPLICITY_NOTE)
    headline = all_paired.loc[
        all_paired["block"].eq("week") & all_paired["metric"].eq("accuracy_improvement")
    ]
    print(
        headline.loc[
            :,
            [
                "calibration_method",
                "candidate_feature_set",
                "headline_population",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
                "paired_games",
                "blocks",
            ],
        ].to_string(index=False)
    )

    print(f"\n[{time.time() - t0:7.1f}s] === Pick flips vs pooled control ===")
    for report in flip_reports:
        note = report.get("note", "")
        print(
            f"{report['label']:>24}: {report['flipped']}/{report['games']} flipped "
            f"({report['flip_rate']:.4%}) {note}"
        )

    timings["total_seconds"] = time.time() - t0
    diagnostics: dict[str, Any] = {
        "predeclaration": str(
            Path(
                "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
                "c8c5fbdd-027f-438d-b992-979e83a91c2e/scratchpad/mod11_scope/predeclaration.md"
            )
        ),
        "rotation_registry_touched": False,
        "weak_signals_registry_touched": False,
        "multiplicity_note": MULTIPLICITY_NOTE,
        "derived_min_calibration_games": int(derived_floor),
        "derived_floor_demonstrated": bool(demonstrated),
        "derivation_method": DERIVATION_METHOD,
        "derivation_bins": [
            f"{lo}-{hi - 1}" if hi is not None else f"{lo}+" for lo, hi in DERIVATION_BINS
        ],
        "pooled_min_calibration_games": DEFAULT_MIN_CALIBRATION_GAMES,
        "reliability_line_bucket": line_reliability_summary,
        "reliability_season_era": era_reliability_summary,
        "stratified_calibration_diagnostics": stratified_diagnostics,
        "flip_reports": flip_reports,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "on_degenerate": "raise",
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "ridge_alpha": args.ridge_alpha,
        "clean_core_games": len(clean_core_raw),
        "full_window_games": len(raw_stream),
        "timings_seconds": timings,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\n[{time.time() - t0:7.1f}s] artifacts written to {output}")


if __name__ == "__main__":
    main()

"""CFB QB-dependence interaction screen (SPEC-6 Step 0 + Step 2).

Research item: ``docs/qb_dependence.md`` (predeclaration, mirroring
``docs/cfb_role_features.md``'s shape). This is a SCREEN, not a
confirmation: it never calls ``nfl_ats.rotation.assign_window`` /
``record_look`` / ``declare_family`` (CFB and non-reserved seasons are free,
``docs/rotation_registry.md`` rule 8), declares no NFL rotation family, and
touches no NFL data or NFL feature table. Structured like
``scripts/residual_location_screen.py``: one walk-forward pass, one set of
artifacts, printed diagnostics before any accuracy number is read.

Two things happen, in the order AGENTS.md requires:

1. **Step 0 -- split-half reliability audit** (``cfb_qb_dependence.
   cfb_qb_dependence_reliability``), run on the interaction column and its
   two constituents, BEFORE any accuracy number is computed or printed.
   Mirrors ``docs/injury_value_lost.md`` sec 3.1's method exactly.
2. **Step 2 -- the two-arm accuracy screen.** Baseline
   (``CFB_MODEL_FEATURE_COLUMNS``, the frozen XLG-03 contract) vs candidate
   (baseline **plus** ``diff_qb_dependence_interaction`` and its two
   constituent diff columns, added alongside the baseline, never instead of
   it -- see ``docs/qb_dependence.md`` trap 3). Walk-forward over the full
   CFB benchmark history (``CFB_BENCHMARK_START_SEASON`` to
   ``CFB_BENCHMARK_END_SEASON``), paired on ``game_id``, scored with
   ``experiments.paired_feature_comparisons`` on the clean-core window
   (week- and season-blocked, 20,000 samples, ``on_degenerate="raise"`` per
   the task's binding instruction). Before any accuracy number is READ, the
   script prints the MDE80 power check
   (``estimation_variance.mde80``) for this screen's own disagreement
   fraction ``f`` and sample size ``n``, mirroring
   ``docs/cfb_role_features.md``'s own reclassification use of the same
   check.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/qb_dependence_cfb_screen.py \\
        --output artifacts/qb_dependence_cfb/<UTC ts>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import nfl_ats.cli as cli  # noqa: E402
from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, load_cfb_seasons  # noqa: E402
from nfl_ats.cfb_qb_dependence import (  # noqa: E402
    CFB_QB_DEPENDENCE_COLUMNS,
    build_and_attach_cfb_qb_dependence,
    cfb_qb_dependence_reliability,
)
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.estimation_variance import mde80, picks_differ_fraction  # noqa: E402
from nfl_ats.experiments import paired_feature_comparisons  # noqa: E402
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id  # noqa: E402

CFB_FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"
OUT_ROOT = REPO / "artifacts" / "qb_dependence_cfb"

BASELINE_ARM = "baseline"
CANDIDATE_ARM = "candidate_qb_dependence"
CANDIDATE_FEATURE_COLUMNS: tuple[str, ...] = (
    *CFB_MODEL_FEATURE_COLUMNS,
    "diff_qb_dependence_interaction",
    "diff_qb_starter_epa_per_dropback",
    "diff_off_pass_rate",
)

# BINDING (task instruction): 20,000 samples, on_degenerate="raise" -- an
# interval containing zero never rejects on its own, but a look this script
# records must not silently under-report its own block count.
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

# Predeclared gate (docs/qb_dependence.md, mirroring SPEC-5's screen
# convention). Applied here mechanically as printed diagnostics only -- this
# script does not record a verdict; a reviewing orchestrator/human does.
ACCURACY_CLEAR_THRESHOLD = 0.75
BRIER_CLEAR_THRESHOLD = 0.90

_PASSTHROUGH: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "result",
    "ats_margin",
    "home_cover",
)


def load_inputs(start_season: int, end_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pbp with passer identity, canonical CFB games) for the requested window."""

    seasons = list(range(start_season, end_season + 1))
    pbp = load_cfb_seasons(
        cli._data_root() / "cfb",
        "pbp",
        seasons,
        columns=[
            "game_id",
            "season",
            "week",
            "seasonType",
            "pos_team_id",
            "homeTeamId",
            "awayTeamId",
            "is_home",
            "EPA",
            "EPA_success",
            "rush",
            "pass",
            "kneel_down",
            "statYardage",
            "home_wp_before",
            "away_wp_before",
            "passer_player_id",
        ],
    )
    canonical_games = pd.read_parquet(CFB_FEATURES_PATH)
    canonical_games = canonical_games.loc[
        canonical_games["season"].between(start_season, end_season)
    ]
    return pbp, canonical_games.reset_index(drop=True)


def _score_week(weekly_games: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    base = weekly_games.loc[:, [c for c in _PASSTHROUGH if c in weekly_games.columns]].copy()
    batches: list[pd.DataFrame] = []
    for name, model in models.items():
        forecasts = model.predict(weekly_games)
        batch = base.copy()
        batch["home_cover_probability"] = forecasts["home_cover_probability"].to_numpy(dtype=float)
        batch["predicted_margin"] = forecasts["predicted_margin"].to_numpy(dtype=float)
        batch["feature_set"] = name
        batch["train_rows"] = model.training_rows
        batches.append(batch)
    return pd.concat(batches, ignore_index=True)


def run_screen(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Walk-forward the two matched arms (baseline vs baseline+interaction)."""

    required = {*_PASSTHROUGH, *CANDIDATE_FEATURE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB qb-dependence screen is missing columns: {', '.join(missing)}"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = {
            BASELINE_ARM: fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=CFB_MODEL_FEATURE_COLUMNS
            ),
            CANDIDATE_ARM: fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=CANDIDATE_FEATURE_COLUMNS
            ),
        }
        batches.append(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    return predictions.sort_values(["gameday", "game_id", "feature_set"]).reset_index(drop=True)


def compute_mde80(predictions: pd.DataFrame) -> dict[str, float]:
    """The power check: what accuracy effect this screen's own (f, n) could detect at 80% power."""

    clean = predictions.loc[predictions["evaluation_window"].eq("clean_core")]
    baseline = clean.loc[
        clean["feature_set"].eq(BASELINE_ARM), ["game_id", "home_cover_probability"]
    ]
    candidate = clean.loc[
        clean["feature_set"].eq(CANDIDATE_ARM), ["game_id", "home_cover_probability"]
    ]
    paired = baseline.merge(
        candidate,
        on="game_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    f = picks_differ_fraction(
        paired["home_cover_probability_baseline"].to_numpy(dtype=float),
        paired["home_cover_probability_candidate"].to_numpy(dtype=float),
    )
    n = len(paired)
    return {"f_picks_differ": f, "n_games": n, "mde80_accuracy_points": mde80(f, n)}


def paired_evidence(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    clean = predictions.loc[predictions["evaluation_window"].eq("clean_core")].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    frames = []
    for block in ("week", "season"):
        paired = paired_feature_comparisons(
            clean,
            baseline_feature_set=BASELINE_ARM,
            samples=samples,
            block=block,
            seed=seed,
            on_degenerate="raise",
        )
        paired["evaluation_window"] = "clean_core"
        frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-season",
        type=int,
        default=CFB_BENCHMARK_START_SEASON,
        help="pbp load + walk-forward start",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=CFB_BENCHMARK_END_SEASON,
        help="pbp load + walk-forward end",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    output: Path = args.output or (OUT_ROOT / run_id())
    output.mkdir(parents=True, exist_ok=True)
    print(f"Artifact directory: {output}", flush=True)

    started = time.perf_counter()
    timings: dict[str, float] = {}

    print(
        f"=== Step 1: load pbp+canonical games ({args.start_season}-{args.end_season}) and "
        "build the two new CFB state metrics ===",
        flush=True,
    )
    t0 = time.perf_counter()
    pbp, canonical_games = load_inputs(args.start_season, args.end_season)
    features = build_and_attach_cfb_qb_dependence(canonical_games, pbp)
    timings["step1_load_and_build_seconds"] = time.perf_counter() - t0
    coverage = {
        column: {
            "non_null": int(features[column].notna().sum()),
            "total": len(features),
            "fraction_non_null": float(features[column].notna().mean()),
        }
        for column in CFB_QB_DEPENDENCE_COLUMNS
    }
    print(f"pbp rows={len(pbp)} canonical games={len(features)}", flush=True)
    print(json.dumps(coverage, indent=2), flush=True)

    print(
        "\n=== Step 0: split-half reliability audit (BEFORE any accuracy number) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    reliability = cfb_qb_dependence_reliability(features)
    timings["step0_reliability_audit_seconds"] = time.perf_counter() - t0
    reliability_payload = asdict(reliability)
    print(json.dumps(reliability_payload, indent=2, default=float), flush=True)

    print(
        "\n=== Step 2: two-arm accuracy screen (baseline vs baseline+interaction) ===", flush=True
    )
    t0 = time.perf_counter()
    predictions = run_screen(features, start_season=args.start_season, end_season=args.end_season)
    timings["step2_walk_forward_seconds"] = time.perf_counter() - t0

    print("\n--- MDE80 power check (read BEFORE the accuracy number below) ---", flush=True)
    power = compute_mde80(predictions)
    print(json.dumps(power, indent=2), flush=True)

    t0 = time.perf_counter()
    paired = paired_evidence(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    timings["step2_bootstrap_seconds"] = time.perf_counter() - t0

    print(
        "\n--- Clean-core paired comparisons (positive = candidate beats baseline) ---", flush=True
    )
    headline = paired.loc[
        paired["metric"].isin(("accuracy_improvement", "brier_improvement", "log_loss_improvement"))
    ]
    print(
        headline.loc[
            :,
            [
                "block",
                "metric",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
                "paired_games",
                "blocks",
                "degenerate_blocks",
            ],
        ].to_string(index=False),
        flush=True,
    )

    accuracy_week = headline.loc[
        headline["metric"].eq("accuracy_improvement") & headline["block"].eq("week")
    ]
    brier_week = headline.loc[
        headline["metric"].eq("brier_improvement") & headline["block"].eq("week")
    ]
    accuracy_p = float(accuracy_week["probability_positive"].iloc[0])
    brier_p = float(brier_week["probability_positive"].iloc[0])
    brier_resolved_gain = float(brier_week["estimate"].iloc[0]) > 0.0
    screen_clears = accuracy_p >= ACCURACY_CLEAR_THRESHOLD or (
        brier_resolved_gain and brier_p >= BRIER_CLEAR_THRESHOLD
    )
    print(
        f"\nPredeclared gate (mechanical read, not a recorded verdict): "
        f"accuracy P+={accuracy_p:.4f} (clear >= {ACCURACY_CLEAR_THRESHOLD}), "
        f"brier P+={brier_p:.4f} resolved_gain={brier_resolved_gain} "
        f"(clear >= {BRIER_CLEAR_THRESHOLD}) -> screen_clears={screen_clears}",
        flush=True,
    )

    timings["total_seconds"] = time.perf_counter() - started

    print("\n=== Writing artifacts ===", flush=True)
    atomic_parquet(
        features.loc[:, ["game_id", *CFB_QB_DEPENDENCE_COLUMNS]],
        output / "qb_dependence_features.parquet",
    )
    atomic_parquet(predictions, output / "predictions.parquet")
    atomic_csv(paired, output / "paired_comparisons.csv")
    atomic_json(reliability_payload, output / "reliability_audit.json")
    atomic_json(power, output / "mde80.json")
    atomic_json(coverage, output / "column_coverage.json")

    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/qb_dependence_cfb_screen.py",
        "predeclaration_source": "docs/qb_dependence.md",
        "rotation_registry_touched": False,
        "league": "cfb",
        "baseline_arm": BASELINE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "candidate_feature_columns": list(CANDIDATE_FEATURE_COLUMNS),
        "start_season": args.start_season,
        "end_season": args.end_season,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "column_coverage": coverage,
        "reliability_audit": reliability_payload,
        "mde80": power,
        "predeclared_gate": {
            "accuracy_clear_threshold": ACCURACY_CLEAR_THRESHOLD,
            "brier_clear_threshold": BRIER_CLEAR_THRESHOLD,
            "accuracy_probability_positive_week": accuracy_p,
            "brier_probability_positive_week": brier_p,
            "brier_resolved_gain": brier_resolved_gain,
            "screen_clears": screen_clears,
        },
        "timing": timings,
    }
    atomic_json(metadata, output / "metadata.json")
    print(f"\nWrote artifacts to {output}", flush=True)
    print(f"Total runtime: {timings['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

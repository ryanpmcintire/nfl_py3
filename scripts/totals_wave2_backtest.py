"""CLI entry point for the wave-2 totals screen (WP18).

Executes the frozen predeclaration in ``docs/totals_model_wave2.md``: an
instrument-sanity positive control (``--mode positive-control``) and the
real drive-pace screen against wave 1 (``--mode screen``), both built on top
of ``nfl_ats.totals_wave2``, which itself reuses ``nfl_ats.totals`` (the
frozen, tested wave-1 module) unmodified wherever possible.

Not wired into ``nfl-ats`` (``src/nfl_ats/cli.py`` is off-limits to this work
package this session -- another agent owns it) -- run directly:

    .\\.tools\\uv.exe run --no-sync python scripts/totals_wave2_backtest.py --mode positive-control
    .\\.tools\\uv.exe run --no-sync python scripts/totals_wave2_backtest.py --mode screen

Writes JSON + prediction-level parquet to
``artifacts/totals_backtest_wave2/<stamp>/<mode>/`` and prints a summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nfl_ats.totals import DEFAULT_MIN_TRAIN_GAMES, TOTALS_RIDGE_ALPHA
from nfl_ats.totals_wave2 import (
    POSITIVE_CONTROL_COLUMN,
    format_positive_control_results,
    format_screen_results,
    run_positive_control,
    run_screen,
)


def _data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _artifacts_root() -> Path:
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("positive-control", "screen"), required=True)
    parser.add_argument(
        "--wave1-features",
        type=Path,
        default=_data_root() / "processed" / "game_features.parquet",
        help="wave-1 feature table (comparator source)",
    )
    parser.add_argument(
        "--wave2-features",
        type=Path,
        default=_data_root() / "processed" / "game_features_pbp.parquet",
        help="wave-2 feature table (carries the drive-pace columns)",
    )
    parser.add_argument("--ridge-alpha", type=float, default=TOTALS_RIDGE_ALPHA)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--control-column", type=str, default=POSITIVE_CONTROL_COLUMN)
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="reuse the same stamp across --mode positive-control and --mode screen "
        "to land both under one artifacts/totals_backtest_wave2/<stamp>/ directory",
    )
    args = parser.parse_args()

    if args.mode == "positive-control":
        results = run_positive_control(
            _data_root(),
            args.wave2_features,
            _artifacts_root(),
            ridge_alpha=args.ridge_alpha,
            min_train_games=args.min_train_games,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            control_column=args.control_column,
            stamp=args.stamp,
        )
        print(format_positive_control_results(results))
    else:
        results = run_screen(
            _data_root(),
            args.wave1_features,
            args.wave2_features,
            _artifacts_root(),
            ridge_alpha=args.ridge_alpha,
            min_train_games=args.min_train_games,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            stamp=args.stamp,
        )
        print(format_screen_results(results))


if __name__ == "__main__":
    main()

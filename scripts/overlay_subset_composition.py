"""Compatibility wrapper for the importable overlay composition computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_ats.overlay_composition import ARREST_MEMBER_NAME as ARREST_MEMBER_NAME
from nfl_ats.overlay_composition import CONFIDENCE as CONFIDENCE
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    run_overlay_composition,
)
from nfl_ats.overlay_composition import blocked_bootstrap_matrix as blocked_bootstrap_matrix
from nfl_ats.overlay_composition import build_delta_matrix as build_delta_matrix
from nfl_ats.overlay_composition import reconstruct_arrest_flip_set as reconstruct_arrest_flip_set
from nfl_ats.public_board import find_matching_opener_evaluation

READ_ONLY_SCRIPT = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.per_game_artifact is None:
        match = find_matching_opener_evaluation(args.output_root.parent)
        if match is None:
            raise ValueError("No opener-evaluation matches the active model")
        args.per_game_artifact = match[1] / "per_game.parquet"
    print(json.dumps(run_overlay_composition(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

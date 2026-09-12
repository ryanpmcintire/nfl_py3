from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_ats.overlay_composition import DEFAULT_FEATURES, DEFAULT_INCIDENTS
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.unserved_tilt_marginals import (
    CARD_CHOICES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    run_unserved_tilt_marginals,
)

READ_ONLY_SCRIPT = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--card", choices=CARD_CHOICES, default="served")
    args = parser.parse_args(argv)
    if args.per_game_artifact is None:
        match = find_matching_opener_evaluation(args.output_root.parent)
        if match is None:
            raise ValueError("No opener-evaluation matches the active model")
        args.per_game_artifact = match[1] / "per_game.parquet"
    result = run_unserved_tilt_marginals(**vars(args))
    print(json.dumps({k: v for k, v in result.items() if k != "members"}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

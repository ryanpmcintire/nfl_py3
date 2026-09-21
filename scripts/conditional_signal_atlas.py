from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_ats.signal_atlas import build_signal_atlas, load_signal_atlas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the declared paired conditional signal atlas."
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--registry-root", type=Path, default=Path("registry"))
    args = parser.parse_args()
    directory = build_signal_atlas(args.artifacts_root, args.registry_root)
    report = load_signal_atlas(args.artifacts_root, args.registry_root)
    if report is None:
        raise ValueError("Signal atlas activation failed")
    print(f"Signal atlas: {directory}")
    for view in ("out_of_season", "chronological"):
        for cell in report["evaluations"][view]:
            print(
                json.dumps(
                    {
                        "view": view,
                        **{
                            key: cell[key]
                            for key in (
                                "cell",
                                "games",
                                "decisive_games",
                                "full_decisive_wins",
                                "reduced_decisive_wins",
                                "accuracy_delta_points",
                                "accuracy_interval",
                                "probability_positive",
                                "brier_improvement",
                            )
                        },
                    }
                )
            )


if __name__ == "__main__":
    main()

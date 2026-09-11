from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nfl_ats.served_refresh_card import measure, reuse_or_measure

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts")),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")),
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force:
        result = measure(
            artifacts_root=args.artifacts_root,
            data_root=args.data_root,
            repo_root=args.repo_root,
            record=args.record,
            registry_dir=args.registry_dir,
        )
    else:
        result = reuse_or_measure(
            artifacts_root=args.artifacts_root,
            data_root=args.data_root,
            repo_root=args.repo_root,
            record=args.record,
            registry_dir=args.registry_dir,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

"""ENG-06: print (and archive) a settled-week prospective evidence scorecard.

Read-only against the append-only prospective ledgers -- it never writes to
``artifacts/prospective/`` or ``registry/``, and it never calls
``nfl-ats weak-signals record`` / ``nfl-ats rotation record-look``. See
``docs/prospective_scorecard.md`` and ``src/nfl_ats/prospective_scorecard.py``
for the full contract.

Usage::

    .tools/uv.exe run --no-sync python scripts/prospective_scorecard.py --season 2025
    .tools/uv.exe run --no-sync python scripts/prospective_scorecard.py --season 2026 \
        --through-week 3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfl_ats.io import atomic_csv, run_id  # noqa: E402
from nfl_ats.prospective_scorecard import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    SCORECARD_SCHEMA_VERSION,
    build_season_scorecards,
    render_markdown,
    scorecards_to_frame,
)
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--through-week",
        type=int,
        default=None,
        help="Cap the scope at this week (inclusive); default is the whole season.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "game_features.parquet",
        help="Canonical feature table supplying game_id/result and the close-line schedule.",
    )
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help=(
            "ENG-33: read-only root for weak_signals.json/rotation_registry.json (the "
            "closing_ground_candidate/next_admissible_action advisories); defaults to the "
            "same tracked registry/ every other registry reader in this repo uses."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/prospective_scorecards/<season>_<stamp>/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the full per-entrant JSON payload to stdout.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.features.is_file():
        print(
            f"Feature table not found: {args.features}. Run `nfl-ats build-features` first.",
            file=sys.stderr,
        )
        return 1

    features = pd.read_parquet(args.features)
    rows = build_season_scorecards(
        args.artifacts_root,
        args.data_root,
        features,
        season=args.season,
        through_week=args.through_week,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        registry_root=args.registry_root,
    )
    markdown = render_markdown(rows, season=args.season, through_week=args.through_week)
    print(markdown)

    out_dir = args.out
    if out_dir is None:
        out_dir = args.artifacts_root / "prospective_scorecards" / f"{args.season}_{run_id()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # ENG-29: write_stamped_artifact(), not write_experiment_artifact() -- this
    # scorecard is explicitly not an experiment (module docstring; every row's
    # classification is the fixed unresolved_below_power), and the latter
    # always creates a registry/experiments/ row, which this script's own
    # contract forbids.
    write_stamped_artifact(
        {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "season": args.season,
            "through_week": args.through_week,
            "entrants": rows,
        },
        out_dir / "scorecard.json",
    )
    atomic_csv(scorecards_to_frame(rows), out_dir / "scorecard.csv")
    (out_dir / "scorecard.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir}")

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

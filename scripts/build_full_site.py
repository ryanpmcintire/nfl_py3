"""Render the FULL ATS Terminal site (This Week, The Model, What We've
Learned) to a scratch directory -- never a real publish destination.

Extended from the single picks page (the retired ``build_design_pilot.py``
pilot) to the whole site via :func:`nfl_ats.board_site.build_site`. Nothing
here writes to ``docs/`` or any other real site destination, and this script
never touches ``nfl-ats publish-board``'s own code path
(:func:`nfl_ats.cli._write_public_site`, which calls the SAME ``build_site``
function for the real site).

2026-08-31 owner redirect: the Cover Desk skin (and its parallel
``desk/``/``terminal/`` directory split, top-level redirect page, and
header skin-toggle) is dropped entirely -- this script now writes exactly
``index.html``, ``model.html``, ``findings.html`` flat into ``--out-dir``.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts\\build_full_site.py \\
        --out-dir <scratch dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.board_site import build_site  # noqa: E402
from nfl_ats.io import atomic_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="artifacts root to read the synchronized weekly forecast from (default: ./artifacts)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="data root for schedule/market snapshots (default: NFL_ATS_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="registry root for weak_signals.json/rotation_registry.json (default: each "
        "module's own NFL_ATS_REGISTRY_DIR-aware default)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="scratch directory to write the full site into -- never a real publish destination",
    )
    parser.add_argument(
        "--require-fresh-arrest-overlay",
        action="store_true",
        help="fail rather than degrade when the player-arrests snapshot is stale (default: "
        "off, a rehearsal read matching the test suite -- this script never writes to a real "
        "publish destination so there is nothing a stale snapshot could corrupt)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pages = build_site(
        args.artifacts_root,
        data_root=args.data_root,
        registry_root=args.registry_root,
        require_fresh_arrest_overlay=args.require_fresh_arrest_overlay,
    )

    for relative_path, html in pages.items():
        atomic_text(html, args.out_dir / relative_path)

    print(f"wrote {len(pages)} pages to {args.out_dir}")
    for relative_path in sorted(pages):
        print(f"  {relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

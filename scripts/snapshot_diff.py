"""ENG-18: CLI for ``nfl_ats.snapshot_diff`` -- decision-time snapshot diff.

Generates a compact diff between one week's Tuesday lock and every later
refresh pass this project can currently reconstruct from already-recorded
evidence: the paper-decision ledger, the pick-revision ledger
(``nfl_ats.pick_refresh``, POL-11/MKT-08), later ``margin_predictions``
forecast artifacts, each artifact's ``lineage.json`` (ENG-16), the ENG-08
refresh-trigger evidence log, and the ENG-14 source-freshness policy (for one
present-tense context section only). See ``src/nfl_ats/snapshot_diff.py``'s
module docstring for the full design and every field's "how do we know this"
basis.

**Read-only.** Never runs ``weekly-run``/``publish-predictions``/
``refresh-picks``, never fits a model, never writes to ``registry/``. The
only writes this script performs are the diff's own Markdown/JSON under
``artifacts/snapshot_diffs/`` (gitignored), and only when ``--no-write`` is
not passed.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1
    .\\.tools\\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1 --json
    .\\.tools\\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1 \\
        --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.io import atomic_text, run_id  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402
from nfl_ats.snapshot_diff import (  # noqa: E402
    build_snapshot_diff,
    render_markdown,
    to_dict,
    to_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write markdown+json under <out>/<season>_wk<week>_<stamp>/ "
        "(default: artifacts/snapshot_diffs/<season>_wk<week>_<stamp>/)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print only; write nothing under artifacts/"
    )
    args = parser.parse_args(argv)

    diff = build_snapshot_diff(
        args.season,
        args.week,
        artifacts_root=args.artifacts,
        data_root=args.data_root,
    )

    markdown = render_markdown(diff)

    if not args.no_write:
        out_root = args.out or (args.artifacts / "snapshot_diffs")
        stamp = run_id()
        directory = out_root / f"{args.season}_wk{args.week:02d}_{stamp}"
        atomic_text(markdown, directory / "snapshot_diff.md")
        # ENG-29: write_stamped_artifact(), not write_experiment_artifact() --
        # this diff is explicitly not an experiment (see module docstring),
        # and the latter always creates a registry/experiments/ row.
        write_stamped_artifact(to_dict(diff), directory / "snapshot_diff.json")
        print(f"wrote {directory}", file=sys.stderr)

    if args.json:
        print(to_json(diff))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

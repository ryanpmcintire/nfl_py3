"""LEAD-58: build and stamp the snap-weighted career-age x position-group curves.

Writes real artifacts under ``artifacts/age_curves/<UTC timestamp>/`` --
``age_curves.parquet`` (the cross-sectional curve, empirical-Bayes shrinkage,
local-linear smooth, and the delta-method within-player curve joined on
career age), ``reliability.parquet`` (both split-half reliability schemes),
and ``manifest.json`` (resolved snapshot ids, the frozen position-group and
metric mapping, and every diagnostic count from the build). This is QUALITY
infrastructure -- descriptive curves and their measured reliability, no ATS
direction, no hypothesis, no closing ground -- so it uses
:func:`nfl_ats.provenance.write_stamped_artifact`/``stamp_sidecar`` rather
than :func:`nfl_ats.provenance.write_experiment_artifact` (which would create
a ``registry/experiments/`` row implying an adjudicated screen; see
``docs/script_contracts.md``). See ``src/nfl_ats/age_curves.py`` and
``docs/age_curves.md`` for the full method and the measured results.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts\\build_age_curves.py
    .\\.tools\\uv.exe run --no-sync python scripts\\build_age_curves.py --as-of-season 2022
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.age_curves import build_age_curves  # noqa: E402
from nfl_ats.io import atomic_parquet, run_id  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of-season",
        type=int,
        default=None,
        help="Point-in-time cutoff: only seasons strictly before this one are used.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Split-half reliability bootstrap draws per (group, scheme). Default 2000.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    result = build_age_curves(
        REPO_ROOT / "data" / "players" / "raw",
        REPO_ROOT / "data" / "players" / "values" / "raw",
        REPO_ROOT / "data" / "pbp" / "raw",
        as_of_season=args.as_of_season,
        bootstrap_samples=args.bootstrap_samples,
    )

    delta_for_join = result.delta.rename(
        columns={
            "career_age_from": "career_age",
            "n_pairs": "delta_n_pairs",
            "mean_delta": "delta_mean",
            "cumulative_delta": "delta_cumulative",
        }
    )
    curves = result.curve.merge(delta_for_join, on=["pos_group", "career_age"], how="left")

    output_id = run_id()
    output_dir = REPO_ROOT / "artifacts" / "age_curves" / output_id
    atomic_parquet(curves, output_dir / "age_curves.parquet")
    stamp_sidecar(output_dir / "age_curves.parquet", {"rows": len(curves)})
    atomic_parquet(result.reliability, output_dir / "reliability.parquet")
    stamp_sidecar(output_dir / "reliability.parquet", {"rows": len(result.reliability)})

    manifest = dict(result.manifest)
    manifest["built_at_utc"] = output_id
    manifest["rows"] = {"age_curves": len(curves), "reliability": len(result.reliability)}
    write_stamped_artifact(manifest, output_dir / "manifest.json")

    print(f"Wrote {len(curves)} curve rows and {len(result.reliability)} reliability rows to")
    print(f"  {output_dir}")
    print(
        "Diagnostics:",
        {k: v for k, v in manifest["diagnostics"].items() if k != "snap_rows_in_panel"},
    )
    with pd.option_context("display.width", 120, "display.max_rows", 40):
        print("\nCross-sectional raw_rate by (pos_group, career_age), ages 0-10:")
        preview = curves.loc[
            curves["career_age"].between(0, 10),
            ["pos_group", "career_age", "n_players", "snaps", "raw_rate", "shrunk_rate"],
        ].sort_values(["pos_group", "career_age"])
        print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

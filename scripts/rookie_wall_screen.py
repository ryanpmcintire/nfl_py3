"""LEAD-24 Stage 1: measure the rookie workload wall and its dependence metric.

Writes real artifacts under ``artifacts/rookie_wall/<UTC timestamp>/`` --
``wall_measurement.parquet`` (per position-group/era rookie-vs-veteran
within-player half-season delta, season-blocked bootstrap), ``dependence_
shares.parquet`` (per team-week raw + trailing + late-season-flag dependence
metric), ``dependence_reliability.parquet`` (both split-half reliability
schemes), and ``manifest.json`` (resolved snapshot ids, diagnostics, and the
top-50-pick join-rate disclosure). This is QUALITY-stage measurement --
descriptive numbers and their measured reliability, no ATS direction, no
hypothesis closure -- so it uses
:func:`nfl_ats.provenance.write_stamped_artifact`/``stamp_sidecar`` rather
than :func:`nfl_ats.provenance.write_experiment_artifact` (which would create
a ``registry/experiments/`` row implying an adjudicated screen), exactly like
``scripts/build_age_curves.py``. See ``src/nfl_ats/rookie_wall.py`` and
``docs/rookie_wall.md`` for the full method and the measured results.

Per AGENTS.md's commensurability rule, the wall delta (a per-snap
performance difference) is NOT an admissible ``weak-signals record
--effect-units`` entry and this script never calls that command for it --
the artifact + ``docs/rookie_wall.md`` are the record. Only the dependence
metric's reliability (a correlation) belongs in the weak-signal registry,
and that recording is a separate, explicit ``nfl-ats weak-signals record``
invocation run from the session, not automated here.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts\\rookie_wall_screen.py
    .\\.tools\\uv.exe run --no-sync python scripts\\rookie_wall_screen.py --bootstrap-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.io import atomic_parquet, run_id  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402
from nfl_ats.rookie_wall import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_COMBINE_RAW_ROOT,
    DEFAULT_PBP_RAW_ROOT,
    DEFAULT_PLAYERS_RAW_ROOT,
    DEFAULT_PLAYERS_VALUES_RAW_ROOT,
    ROOKIE_WALL_VERSION,
    build_rookie_wall_panel,
    dependence_split_half_reliability,
    late_season_high_dependence_flag,
    load_rookie_wall_inputs,
    rookie_wall_measurement,
    team_week_dependence_shares,
    trailing_dependence_feature,
    wall_candidates,
)


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
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Season-blocked bootstrap draws. Default {DEFAULT_BOOTSTRAP_SAMPLES}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine, snapshot_ids = (
        load_rookie_wall_inputs(
            DEFAULT_PLAYERS_RAW_ROOT,
            DEFAULT_PLAYERS_VALUES_RAW_ROOT,
            DEFAULT_PBP_RAW_ROOT,
            DEFAULT_COMBINE_RAW_ROOT,
            as_of_season=args.as_of_season,
        )
    )
    panel, diagnostics = build_rookie_wall_panel(
        raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine, as_of_season=args.as_of_season
    )

    candidates = wall_candidates(panel)
    wall_measurement = rookie_wall_measurement(candidates, samples=args.bootstrap_samples)

    shares = team_week_dependence_shares(panel)
    trailing = trailing_dependence_feature(shares)
    flagged = late_season_high_dependence_flag(trailing)
    reliability = dependence_split_half_reliability(shares, samples=args.bootstrap_samples)

    output_id = run_id()
    output_dir = REPO_ROOT / "artifacts" / "rookie_wall" / output_id

    atomic_parquet(wall_measurement, output_dir / "wall_measurement.parquet")
    stamp_sidecar(output_dir / "wall_measurement.parquet", {"rows": len(wall_measurement)})
    atomic_parquet(flagged, output_dir / "dependence_shares.parquet")
    stamp_sidecar(output_dir / "dependence_shares.parquet", {"rows": len(flagged)})
    atomic_parquet(reliability, output_dir / "dependence_reliability.parquet")
    stamp_sidecar(output_dir / "dependence_reliability.parquet", {"rows": len(reliability)})

    manifest = {
        "builder_version": ROOKIE_WALL_VERSION,
        "as_of_season": args.as_of_season,
        "bootstrap_samples": args.bootstrap_samples,
        "resolved_snapshots": snapshot_ids,
        "diagnostics": diagnostics,
        "rows": {
            "wall_candidates": len(candidates),
            "wall_measurement": len(wall_measurement),
            "dependence_shares": len(flagged),
            "dependence_reliability": len(reliability),
        },
        "n_rookie_top50_high_workload_player_seasons": int(
            (candidates["population"] == "rookie_top50_high_workload").sum()
        ),
        "n_veteran_high_workload_control_player_seasons": int(
            (candidates["population"] == "veteran_high_workload_control").sum()
        ),
    }
    manifest["built_at_utc"] = output_id
    write_stamped_artifact(manifest, output_dir / "manifest.json")

    print(f"Wrote rookie-wall artifacts to {output_dir}")
    print(
        "Panel diagnostics:",
        {k: v for k, v in diagnostics.items() if k != "top50_pick_lookup"},
    )
    print("Top-50-pick lookup diagnostics:", diagnostics["top50_pick_lookup"])
    print(
        f"Wall candidates: {manifest['n_rookie_top50_high_workload_player_seasons']} rookie "
        f"top50-high-workload player-seasons, "
        f"{manifest['n_veteran_high_workload_control_player_seasons']} veteran-control "
        "player-seasons."
    )
    with pd.option_context("display.width", 200, "display.max_rows", 60, "display.max_columns", 20):
        print("\nWall measurement (rookie-minus-veteran, per position group / era):")
        preview_columns = [
            "pos_group",
            "era",
            "n_rookie_player_seasons",
            "n_veteran_player_seasons",
            "rookie_delta_mean",
            "veteran_delta_mean",
            "rookie_minus_veteran",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "probability_wall_direction",
        ]
        print(
            wall_measurement.loc[:, preview_columns]
            .sort_values(["era", "pos_group"])
            .to_string(index=False)
        )
        print("\nDependence-metric split-half reliability:")
        print(reliability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Split-half reliability for the 41 ``graph_team_stat_*`` registry cells (ORCH-D).

**What these cells are.** Each one adds ONE team-stat column as the edge
signal of the graph-ratings-v2 model and scores the resulting arm against a
control (read: ``scripts/graph_team_stat_screen.py:123-176`` --
``build_arm_columns`` builds ``gts_control_<family>`` /
``gts_treatment_<family>`` from the ``home_<family>``/``away_<family>`` pair
that ``reliability_map.discover_family_pairs`` finds). So the construct behind
``graph_team_stat_<family>`` is exactly the continuous team-week column
``<family>``, and the registry cell's reliability is that column's split-half
reliability -- the same convention the six ``attention_battery_*`` cells
already follow, where all six share the attention-gap trait's number.

**Method.** ``METHOD_TRAIT`` from ``scripts/reliability_lib.py``: unit =
team-season, halves = odd/even weeks, Spearman-Brown corrected, block
bootstrap over team-seasons, seed 20260901, 4000 draws, restricted to each
cell's OWN seasons. The estimator is
``nfl_ats.cfb_qb_dependence.split_half_reliability``, imported through the
shared harness, never reimplemented.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``; report
``probability_positive``, never "contains zero". This script CLOSES NOTHING:
it measures, and a low number is a candidate for the reliability ground, never
the closure itself. Within-week correlation is ZERO.

A construct with too few usable team-seasons is reported as UNMEASURED, never
as reliability 0 -- writing a NaN through as a number would manufacture the
appearance of a closing ground out of nothing.

Writes ``artifacts/reliability_sweep/graph_team_stat/<stamp>/results.json``
and prints the ``set-reliability`` commands it would run (``--record`` runs
nothing itself; recording goes through the locked CLI).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

import reliability_lib as rlib  # noqa: E402
import reliability_map as relmap  # noqa: E402

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

#: Registry cells whose name does not simply prefix the family name. Each was
#: read off the entry's own ``source`` artifact directory and description
#: rather than guessed from the name.
NAME_OVERRIDES: dict[str, str] = {
    "graph_team_stat_off_sack_rate_on_production": "off_sack_rate",
    "graph_def_ypp_on_production": "def_yards_per_play",
    "graph_off_rush_epa_on_production": "off_rush_epa_per_play",
}

PREFIX = "graph_team_stat_"


def family_for(entry_name: str) -> str | None:
    """Map a registry cell name to the team-week column it is built on."""

    if entry_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[entry_name]
    if entry_name.startswith(PREFIX):
        return entry_name[len(PREFIX) :]
    return None


def target_entries() -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name, signal in registry.signals.items():
        if signal.league != "nfl" or signal.effect_units != "accuracy_points":
            continue
        if not (name.startswith(PREFIX) or name in NAME_OVERRIDES):
            continue
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
            "source": signal.source,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    parser.add_argument(
        "--only-null",
        action="store_true",
        help="restrict to cells that currently carry reliability: null",
    )
    args = parser.parse_args()

    started = time.time()
    entries = target_entries()
    if args.only_null:
        entries = {n: e for n, e in entries.items() if e["reliability"] is None}
    print(f"=== {len(entries)} graph_team_stat registry cells in scope ===")

    features = relmap.load_feature_table()
    dtypes = {column: features[column].dtype for column in features.columns}
    families, _excluded = relmap.discover_family_pairs(list(features.columns), dtypes)
    long = relmap.build_long_frame(features, families)
    print(f"team-week long frame: {long.shape}, {len(families)} discovered families")

    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for name in sorted(entries):
        entry = entries[name]
        family = family_for(name)
        if family is None or family not in families:
            unmapped.append(f"{name} (family={family!r} not in the discovered pair set)")
            rows.append(
                {
                    "entry": name,
                    "family": family,
                    "seasons": list(entry["seasons"]),
                    "status": "no_matching_feature_column",
                    "reliability": None,
                }
            )
            continue
        measured = rlib.measure_reliability(
            long,
            family,
            method=rlib.METHOD_TRAIT,
            seasons=entry["seasons"],
            n_boot=args.n_boot,
        )
        home_col, away_col, pattern = families[family]
        rows.append(
            {
                "entry": name,
                "family": family,
                "home_column": home_col,
                "away_column": away_col,
                "pattern": pattern,
                "seasons": list(entry["seasons"]),
                "registry_effect": entry["effect"],
                "registry_classification": entry["classification"],
                "n_units": measured["n_units"],
                "pearson_r": measured["pearson_r"],
                "pearson_r_ci95": measured["pearson_r_ci95"],
                "spearman_rho": measured["spearman_rho"],
                "spearman_brown_full_length_reliability": measured[
                    "spearman_brown_full_length_reliability"
                ],
                "probability_positive": measured["probability_positive"],
                "reliability": measured["reliability"],
                "reliability_low": measured["reliability_low"],
                "reliability_high": measured["reliability_high"],
                "status": measured["status"],
                "method": measured["method"],
            }
        )
        rel = measured["reliability"]
        shown = f"{rel:+.4f}" if rel is not None else "  n/a "
        print(
            f"  {name:<58} {family:<38} n={measured['n_units']:>4} rel={shown} {measured['status']}"
        )

    windows = sorted({tuple(r["seasons"]) for r in rows if r.get("n_units")})
    controls: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        restricted = long.loc[long["season"].between(window[0], window[1])]
        controls[f"{window[0]}-{window[1]}"] = rlib.positive_control(restricted, n_boot=1000)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "graph_team_stat" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "reliability-graph-team-stat",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "method": rlib.METHOD_TRAIT,
        "entries": sorted(entries),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "method": rlib.METHOD_TRAIT,
        "mapping_provenance": (
            "scripts/graph_team_stat_screen.py:123-176 (build_arm_columns) builds each cell's "
            "arm from the home_<family>/away_<family> pair that "
            "reliability_map.discover_family_pairs returns, so cell "
            "'graph_team_stat_<family>' IS the team-week column <family>. Read 2026-09-01."
        ),
        "name_overrides": NAME_OVERRIDES,
        "unmapped": unmapped,
        "positive_control": controls,
        "results": rows,
        "provenance": artifact_provenance(configuration, relmap.V4_PATH, project_root=REPO),
    }
    measured_count = sum(1 for r in rows if r["status"] == rlib.STATUS_MEASURED)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-graph-team-stat",
        metrics={
            "n_entries": len(rows),
            "n_measured": measured_count,
            "n_unmeasured": len(rows) - measured_count,
        },
        notes=(
            "Measure-only split-half reliability for the graph_team_stat registry cells; "
            "every cell measured regardless of sign or interval shape, and nothing is "
            "closed or reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    measured_rows = [r for r in rows if r["status"] == rlib.STATUS_MEASURED]
    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    for label, predicate in (
        ("<= 0.10", lambda v: v <= 0.10),
        (">= 0.80", lambda v: v >= 0.80),
    ):
        hits = [r for r in measured_rows if predicate(r["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            print(
                f"    {row['entry']:<58} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

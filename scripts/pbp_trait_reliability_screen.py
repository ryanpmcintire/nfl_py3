"""Phase 12 lane J: LEAD-26/27/30 PBP coaching-trait RELIABILITY screen.

Reliability-first, per the ROADMAP rows: this script measures split-half
reliability for four trait metrics --

  - ``opening_drive_td_rate``       (LEAD-26)
  - ``opening_drive_epa_per_play``  (LEAD-26)
  - ``q3_point_diff``               (LEAD-27)
  - ``fourth_down_go_rate``         (LEAD-30)

-- two ways (within-season odd/even week, season-to-season same franchise),
each with a season-blocked bootstrap and a team-label-shuffle null. It NEVER
runs an ATS comparison; that is a later lane's job, gated on a trait
clearing the reliability bar measured here. All builders and the reliability
engine live in ``nfl_ats.pbp_coaching_traits`` (pure functions, unit-tested
in ``tests/test_pbp_coaching_traits.py``); this script is orchestration only.

**Binding closing-grounds taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains
zero". This script does not call ``weak-signals record`` itself -- it
prints every number the record commands need, and the four calls are run
separately so the actual CLI validator (not this script's own judgment)
is what accepts each entry.

Data: the newest ``data/pbp/raw/*/`` snapshot (regular season only, via
``nfl_ats.pbp.load_pbp_snapshot``), override with ``--snapshot``.

Writes ``artifacts/pbp_trait_reliability/<UTC timestamp>/results.json`` via
``write_experiment_artifact`` and prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.pbp import (  # noqa: E402
    latest_pbp_snapshot,
    load_pbp_snapshot,
    snapshot_from_root,
)
from nfl_ats.pbp_coaching_traits import (  # noqa: E402
    PBP_TRAIT_MIN_PER_HALF,
    PBP_TRAIT_N_BOOT,
    PBP_TRAIT_N_NULL,
    PBP_TRAIT_RELIABILITY_SEED,
    run_all_trait_reliabilities,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_SNAPSHOT_ROOT = REPO / "data" / "pbp" / "raw"

WEAK_SIGNAL_NAMES = {
    "opening_drive_td_rate": "pbp_trait_opening_drive_td_rate_reliability",
    "opening_drive_epa_per_play": "pbp_trait_opening_drive_epa_reliability",
    "q3_point_diff": "pbp_trait_q3_point_diff_reliability",
    "fourth_down_go_rate": "pbp_trait_fourth_down_go_rate_reliability",
}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fvalue != fvalue:  # NaN
        return "nan"
    return f"{fvalue:+.{digits}f}"


def _print_method(label: str, payload: dict[str, Any]) -> None:
    if payload["status"] != "measured":
        print(f"    {label}: {payload['status']} (n_units={payload['n_units']})")
        return
    print(
        f"    {label}: n_units={payload['n_units']} n_seasons={payload['n_seasons']} "
        f"pearson_r={_fmt(payload['pearson_r'])} "
        f"95%[{_fmt(payload['pearson_r_ci95'][0])}, {_fmt(payload['pearson_r_ci95'][1])}] "
        f"P+={_fmt(payload['pearson_probability_positive'])} "
        f"spearman_rho={_fmt(payload['spearman_rho'])} "
        f"spearman_brown={_fmt(payload['spearman_brown_full_length_reliability'])} "
        f"null_mean_r={_fmt(payload['null_mean_r'])} null_sd_r={_fmt(payload['null_sd_r'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=None, help="a data/pbp/raw/<stamp> directory"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=PBP_TRAIT_RELIABILITY_SEED)
    parser.add_argument("--n-boot", type=int, default=PBP_TRAIT_N_BOOT)
    parser.add_argument("--n-null", type=int, default=PBP_TRAIT_N_NULL)
    parser.add_argument("--min-per-half", type=int, default=PBP_TRAIT_MIN_PER_HALF)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "pbp_trait_reliability" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = (
        snapshot_from_root(args.snapshot)
        if args.snapshot
        else latest_pbp_snapshot(DEFAULT_SNAPSHOT_ROOT)
    )
    season_range = f"{snapshot.seasons[0]}-{snapshot.seasons[-1]}"
    print(f"=== loading PBP snapshot {snapshot.root} (seasons {season_range}) ===")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    print(f"REG plays loaded: {len(pbp)}")

    results = run_all_trait_reliabilities(
        pbp,
        seed=args.seed,
        n_boot=args.n_boot,
        n_null=args.n_null,
        min_per_half=args.min_per_half,
    )

    print("\n=== reliability, by metric ===")
    for metric, payload in results.items():
        weak_signal_name = WEAK_SIGNAL_NAMES[metric]
        print(f"\n{metric}  (weak-signals name: {weak_signal_name})")
        print(f"  n_team_seasons={payload['n_team_seasons']}")
        _print_method("within_season_odd_even_week", payload["within_season_odd_even_week"])
        _print_method("season_to_season_same_franchise", payload["season_to_season_same_franchise"])
        within = payload["within_season_odd_even_week"]
        if within["status"] == "measured":
            earns_look = within["pearson_probability_positive"] > 0.5
            verdict = "YES" if earns_look else "no"
            p_plus = _fmt(within["pearson_probability_positive"])
            print(
                f"    -> earns an ATS look (non-zero reliability, P+>0.5): {verdict} (P+={p_plus})"
            )

    representative_season_path = snapshot.season_path(snapshot.seasons[-1])
    configuration = {
        "command": "pbp-trait-reliability-screen",
        "snapshot_root": str(snapshot.root),
        "snapshot_id": snapshot.snapshot_id,
        "seasons": list(snapshot.seasons),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "min_per_half": args.min_per_half,
    }
    payload_out = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "snapshot_id": snapshot.snapshot_id,
        "seasons": list(snapshot.seasons),
        "n_plays_regular_season": len(pbp),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "min_per_half": args.min_per_half,
        "predeclaration": "docs/pbp_trait_reliability.md (frozen before scoring)",
        "weak_signal_names": WEAK_SIGNAL_NAMES,
        "results": results,
        "provenance": artifact_provenance(
            configuration, representative_season_path, project_root=REPO
        ),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload_out,
        command="pbp-trait-reliability-screen",
        metrics=payload_out,
        notes=(
            "Reliability-only screen for LEAD-26/27/30 (lane J): no ATS window is run here. "
            "Records nothing itself; four separate `nfl-ats weak-signals record` calls "
            "(classification unresolved_below_power) turn this artifact's numbers into "
            "registry entries pbp_trait_opening_drive_td_rate_reliability, "
            "pbp_trait_opening_drive_epa_reliability, pbp_trait_q3_point_diff_reliability, "
            "pbp_trait_fourth_down_go_rate_reliability."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

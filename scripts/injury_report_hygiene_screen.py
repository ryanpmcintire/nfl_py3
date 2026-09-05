"""Injury-report text hygiene screen (Phase 12 LEAD-18, LEAD-19; frozen).

Predeclared in ``docs/injury_report_hygiene.md`` before any number below was
computed. Both leads are QUALITY / feature-hygiene measurements: no ATS
direction, no rotation window, no registry verdict (see that document's
section 7 for why neither result is commensurable with any
``nfl_ats.weak_signals.EFFECT_UNITS`` entry).

LEAD-18: does a Wednesday-equivalent concussion DNP predict Sunday
inactivity differently by position (skill vs literal OL/DL "line"), and how
does that compare to the same split for non-concussion DNPs?

LEAD-19: what is the Sunday-action base rate for "personal matter" DNPs
versus genuine-injury DNPs (same practice status) and versus "rest day"
DNPs -- the measurement ``nfl_ats.availability``'s feature construction
needs to decide whether personal-matter designations belong in the
injury-unavailability feature at all.

Writes ``artifacts/injury_report_hygiene/<stamp>/results.json`` via
``write_experiment_artifact`` (this is a measurement, so it DOES get an
experiment-registry row under ``registry/experiments/`` -- distinct from,
and not a substitute for, the weak-signals/rotation registries neither lead
qualifies for).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _path in (str(REPO), str(REPO / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.availability import position_group as standard_position_group  # noqa: E402
from nfl_ats.injury_report_hygiene import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DID_NOT_PARTICIPATE_STATUS,
    build_player_week_frame,
    rate_gap_to_dict,
    season_block_bootstrap_gap,
)
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    configuration_hash,
    git_state,
    write_experiment_artifact,
)

OUT_ROOT = REPO / "artifacts" / "injury_report_hygiene"
DEFAULT_INJURIES_PATH = (
    REPO / "data" / "raw" / "nflverse_injuries" / "20260826T122850Z" / "injuries.parquet"
)
DEFAULT_PLAYER_SNAPSHOT = REPO / "data" / "players" / "raw" / "20260817T184901Z"
SEASON_START = 2013
SEASON_END = 2025
ERA_BOUNDARY = 2018  # 2013-2017 vs 2018-2025, per docs/injury_report_hygiene.md


def _era_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "2013_2017": frame.loc[frame["season"].between(2013, ERA_BOUNDARY - 1)],
        "2018_2025": frame.loc[frame["season"].between(ERA_BOUNDARY, SEASON_END)],
    }


def _safe_gap(
    frame: pd.DataFrame, *, group_column: str, outcome_column: str, group_a: str, group_b: str
) -> dict[str, Any] | dict[str, str]:
    """Compute a season-blocked gap, or report why it could not be computed.

    Both leads slice into eras and designations where one side of a
    comparison can legitimately be empty (e.g. LEAD-19's "personal_matter"
    designation has zero 2013-2017 rows, see docs/injury_report_hygiene.md
    section 8) -- that absence is itself a measured, reportable fact, not an
    error to hide.
    """

    present_a = int(frame[group_column].eq(group_a).sum())
    present_b = int(frame[group_column].eq(group_b).sum())
    if present_a == 0 or present_b == 0:
        return {
            "skipped": True,
            "reason": f"{group_a}={present_a} rows, {group_b}={present_b} rows in this slice",
        }
    result = season_block_bootstrap_gap(
        frame,
        group_column=group_column,
        outcome_column=outcome_column,
        group_a=group_a,
        group_b=group_b,
    )
    return rate_gap_to_dict(result)


def lead_18(frame: pd.DataFrame) -> dict[str, Any]:
    dnp = frame.loc[frame["practice_status"].eq(DID_NOT_PARTICIPATE_STATUS)]
    concussion_dnp = dnp.loc[dnp["is_concussion_report"]]
    non_concussion_dnp = dnp.loc[~dnp["is_concussion_report"]]
    output: dict[str, Any] = {
        "concussion_dnp_population": len(concussion_dnp),
        "concussion_dnp_position_counts": concussion_dnp["position"].value_counts().to_dict(),
        "concussion_dnp_group_counts": concussion_dnp["concussion_group"].value_counts().to_dict(),
        "concussion_dnp_skill_vs_line_sit_rate": _safe_gap(
            concussion_dnp,
            group_column="concussion_group",
            outcome_column="sat_out",
            group_a="skill",
            group_b="line",
        ),
        "non_concussion_dnp_population": len(non_concussion_dnp),
        "non_concussion_dnp_skill_vs_line_sit_rate": _safe_gap(
            non_concussion_dnp,
            group_column="concussion_group",
            outcome_column="sat_out",
            group_a="skill",
            group_b="line",
        ),
        "by_era": {},
    }
    for era_name, era_concussion in _era_split(concussion_dnp).items():
        era_non_concussion = _era_split(non_concussion_dnp)[era_name]
        output["by_era"][era_name] = {
            "concussion_dnp_n": len(era_concussion),
            "concussion_dnp_skill_vs_line_sit_rate": _safe_gap(
                era_concussion,
                group_column="concussion_group",
                outcome_column="sat_out",
                group_a="skill",
                group_b="line",
            ),
            "non_concussion_dnp_n": len(era_non_concussion),
            "non_concussion_dnp_skill_vs_line_sit_rate": _safe_gap(
                era_non_concussion,
                group_column="concussion_group",
                outcome_column="sat_out",
                group_a="skill",
                group_b="line",
            ),
        }
    return output


def lead_19(frame: pd.DataFrame) -> dict[str, Any]:
    dnp = frame.loc[frame["practice_status"].eq(DID_NOT_PARTICIPATE_STATUS)].copy()
    dnp["standard_group"] = dnp["position"].map(standard_position_group)
    base_rates = (
        dnp.groupby("designation")["played_f"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "played_rate"})
        .to_dict(orient="index")
    )
    by_position_group = (
        dnp.groupby(["designation", "standard_group"])["played_f"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "played_rate"})
    )
    by_position_group_output = {
        f"{designation}/{group}": {
            "played_rate": float(row["played_rate"]),
            "count": int(row["count"]),
        }
        for (designation, group), row in by_position_group.iterrows()
    }
    no_report_status_dnp = dnp.loc[dnp["report_status"].isna()]
    mixed_cell = (
        no_report_status_dnp.groupby("designation")["played_f"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "played_rate"})
        .to_dict(orient="index")
    )
    output: dict[str, Any] = {
        "dnp_population": len(dnp),
        "designation_counts_dnp": dnp["designation"].value_counts().to_dict(),
        "base_rates_dnp": {
            key: {"played_rate": float(value["played_rate"]), "count": int(value["count"])}
            for key, value in base_rates.items()
        },
        "by_position_group": by_position_group_output,
        "no_report_status_mixed_cell_diagnostic": {
            "description": (
                "Rows with practice_status=DNP and a blank report_status -- the "
                "combo bucket nfl_ats.availability's report_category='none' x "
                "practice_category='dnp' cell pools today, mixing designations "
                "with very different Sunday-action rates."
            ),
            "by_designation": {
                key: {"played_rate": float(value["played_rate"]), "count": int(value["count"])}
                for key, value in mixed_cell.items()
            },
        },
        "personal_matter_vs_injury_played_rate": _safe_gap(
            dnp,
            group_column="designation",
            outcome_column="played_f",
            group_a="personal_matter",
            group_b="injury",
        ),
        "personal_matter_vs_rest_day_played_rate": _safe_gap(
            dnp,
            group_column="designation",
            outcome_column="played_f",
            group_a="personal_matter",
            group_b="rest_day",
        ),
        "by_era": {},
    }
    for era_name, era_dnp in _era_split(dnp).items():
        output["by_era"][era_name] = {
            "designation_counts": era_dnp["designation"].value_counts().to_dict(),
            "personal_matter_vs_injury_played_rate": _safe_gap(
                era_dnp,
                group_column="designation",
                outcome_column="played_f",
                group_a="personal_matter",
                group_b="injury",
            ),
            "personal_matter_vs_rest_day_played_rate": _safe_gap(
                era_dnp,
                group_column="designation",
                outcome_column="played_f",
                group_a="personal_matter",
                group_b="rest_day",
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injuries-path", type=Path, default=DEFAULT_INJURIES_PATH)
    parser.add_argument("--player-snapshot", type=Path, default=DEFAULT_PLAYER_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()

    injuries = pd.read_parquet(args.injuries_path)
    snaps = pd.read_parquet(args.player_snapshot / "snap_counts.parquet")
    rosters = pd.read_parquet(args.player_snapshot / "weekly_rosters.parquet")

    frame, multi_revision_count = build_player_week_frame(
        injuries, snaps, rosters, season_start=SEASON_START, season_end=SEASON_END
    )

    lead_18_result = lead_18(frame)
    lead_19_result = lead_19(frame)

    configuration = {
        "command": "injury-report-hygiene-screen",
        "injuries_path": str(args.injuries_path),
        "player_snapshot": args.player_snapshot.name,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "era_boundary": ERA_BOUNDARY,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "predeclaration": "docs/injury_report_hygiene.md (frozen before scoring)",
    }
    payload = {
        "total_player_weeks": len(frame),
        "multi_revision_player_weeks": multi_revision_count,
        "lead_18": lead_18_result,
        "lead_19": lead_19_result,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "code": git_state(REPO),
        },
    }

    output_dir = args.output or (OUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=False)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="injury-report-hygiene-screen",
        metrics=payload,
        notes=(
            "LEAD-18/LEAD-19 QUALITY feature-hygiene measurements; no ATS "
            "outcome, no rotation window, no weak-signals registry entry "
            "(units not commensurable, see docs/injury_report_hygiene.md "
            "section 7)."
        ),
    )
    print(f"total player-weeks: {payload['total_player_weeks']}")
    print(f"multi-revision player-weeks: {multi_revision_count}")
    print("LEAD-18 concussion+DNP skill-vs-line sit rate gap:")
    print(lead_18_result["concussion_dnp_skill_vs_line_sit_rate"])
    print("LEAD-19 personal_matter vs injury played-rate gap:")
    print(lead_19_result["personal_matter_vs_injury_played_rate"])
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

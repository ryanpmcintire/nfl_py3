"""Reproduce CX8 assignment audit and opener reruns from local evidence only.

Registry mutations remain separate, via weak-signals record. Untimestamped
QB history is coverage missingness, never an observed agreement or disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402
import qb_identity_on_production as qb_screen  # noqa: E402
import schedule_flag_on_production as schedule_screen  # noqa: E402

from nfl_ats.clv import opener_pick_evaluation  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402
from nfl_ats.qb_identity_features import (  # noqa: E402
    attach_qb_revenge_features,
    attach_rookie_qb_debut_fade_features,
    decision_time_qb_schedule,
    default_combine,
    default_schedule,
    default_weekly_rosters,
    draft_team_by_gsis_id,
    oracle_derive_qb_revenge_features,
    oracle_derive_rookie_qb_debut_fade_features,
)
from nfl_ats.schedule_flag_features import (  # noqa: E402
    attach_dome_shootout_favorite_features,
    attach_sept_heat_home_features,
    decision_time_roof_schedule,
    default_opener_lines,
    oracle_derive_dome_shootout_favorite_features,
    oracle_derive_sept_heat_home_features,
)
from nfl_ats.transaction_flag_features import (  # noqa: E402
    attach_deadline_integration_drag_features,
)

OUTPUT = ROOT / "artifacts/cx8_posthoc_assignments/20260905"
ATTACH = {
    "rookie_debut": attach_rookie_qb_debut_fade_features,
    "qb_revenge": attach_qb_revenge_features,
    "dome_shootout": attach_dome_shootout_favorite_features,
    "sept_heat": attach_sept_heat_home_features,
}


def audit() -> None:
    schedule = default_schedule()
    schedule = schedule.loc[schedule.game_type.eq("REG") & schedule.season.between(2013, 2025)]
    projected = decision_time_qb_schedule(schedule)
    roof = decision_time_roof_schedule(schedule)
    rows = []
    for season, group in projected.groupby("season"):
        complete = group[["home_qb_id", "away_qb_id"]].notna().all(axis=1)
        different = pd.Series(False, index=group.index)
        side_observed = 0
        side_different = 0
        for side in ("home", "away"):
            comparable = group[f"{side}_qb_id"].notna() & group[f"oracle_{side}_qb_id"].notna()
            differs = comparable & group[f"{side}_qb_id"].ne(group[f"oracle_{side}_qb_id"])
            side_observed += int(comparable.sum())
            side_different += int(differs.sum())
            different |= differs.fillna(False)
        venue = roof.loc[group.index]
        roof_known = venue.roof.notna() & venue.oracle_roof.notna()
        state_aliases = {"dome": "closed", "outdoors": "open"}
        state_differs = venue.roof.replace(state_aliases).ne(
            venue.oracle_roof.replace(state_aliases)
        )
        rows.append(
            {
                "season": int(season),
                "games": len(group),
                "qb_both_sides_observed_games": int(complete.sum()),
                "qb_observed_sides": side_observed,
                "qb_different_sides": side_different,
                "qb_games_with_observed_difference": int(different.sum()),
                "qb_difference_on_complete_games": int((different & complete).sum()),
                "roof_comparable_games": int(roof_known.sum()),
                "roof_different_games": int((roof_known & state_differs).sum()),
                "roof_raw_label_different_games": int(
                    (roof_known & venue.roof.ne(venue.oracle_roof)).sum()
                ),
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(OUTPUT / "qb_assignments.parquet", index=False)
    stamp_sidecar(OUTPUT / "qb_assignments.parquet", project_root=ROOT)
    write_stamped_artifact(
        {
            "seasons": rows,
            "roof_state_comparison": "dome=closed; outdoors=open; raw labels retained",
            "sources": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    sorted((ROOT / "data/raw").glob("*/schedules.parquet"))[-1],
                    *sorted(
                        (ROOT / "data/players/raw/depth_charts").glob("*/depth_charts.parquet")
                    ),
                    *sorted((ROOT / "data/quarterbacks/depth/raw").glob("*/quarterbacks.parquet")),
                    ROOT / "registry/stadium_coordinates.json",
                ]
            },
            "cutoff": "pool_decision_cutoff; strictly before",
            "roof_policy": (
                "fixed venue metadata; retractable defaults closed; "
                "no announcement archive supplied"
            ),
        },
        OUTPUT / "audit.json",
        project_root=ROOT,
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def reliability(features: pd.DataFrame, column: str) -> dict:
    """Team-season odd/even exposure stability, not measurement-error reliability."""
    pieces = []
    for side in ("home", "away"):
        rows = features[["season", "week", f"{side}_team", column]].copy()
        rows = rows.rename(columns={f"{side}_team": "team", column: "flag"})
        if side == "away":
            rows["flag"] *= -1
        pieces.append(rows)
    long = pd.concat(pieces)
    long["half"] = long.week.astype(int) % 2
    paired = long.groupby(["season", "team", "half"]).flag.mean().unstack("half").dropna()
    value = paired[0].corr(paired[1]) if len(paired) > 1 else float("nan")
    return {
        "odd_even_team_season_exposure_correlation": float(value) if np.isfinite(value) else None,
        "paired_team_seasons": len(paired),
        "interpretation": (
            "exposure stability only; not an admissible no_split_half_reliability closing ground"
        ),
    }


def rerun(name: str, *, oracle: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "input_features.parquet"
    if not frozen.exists():
        source = ROOT / "data/processed/game_features_weak_stack.parquet"
        frozen.write_bytes(source.read_bytes())
        stamp_sidecar(frozen, project_root=ROOT)
    base = pd.read_parquet(frozen)
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    baseline_path = OUTPUT / "baseline.parquet"
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
    else:
        print("scoring baseline", flush=True)
        baseline = opener_pick_evaluation(
            confirmation.DEFAULT_MARKET_ROOT,
            base,
            active_model_config=confirmation.model_config("weak_stack"),
            min_train_games=500,
        )
        baseline.to_parquet(baseline_path, index=False)
        stamp_sidecar(baseline_path, {"feature_sha256": digest}, project_root=ROOT)
    print(f"scoring {name}", flush=True)
    if name == "both":
        features = attach_deadline_integration_drag_features(attach_qb_revenge_features(base))
        profile, column = "weak_stack_qb_revenge_deadline_drag", "qb_revenge_flag"
    else:
        features = ATTACH[name](base)
        candidate = (
            qb_screen.CANDIDATES if name in qb_screen.CANDIDATES else schedule_screen.CANDIDATES
        )[name]
        profile, column = candidate.profile, candidate.column
    if oracle:
        schedule = default_schedule()
        if name in ("qb_revenge", "both"):
            lookup = draft_team_by_gsis_id(default_combine(), default_weekly_rosters())
            derived = oracle_derive_qb_revenge_features(schedule, lookup)
        elif name == "rookie_debut":
            derived = oracle_derive_rookie_qb_debut_fade_features(
                schedule, default_weekly_rosters()
            )
        elif name == "dome_shootout":
            derived = oracle_derive_dome_shootout_favorite_features(
                schedule, default_opener_lines(schedule=schedule)
            )
        else:
            derived = oracle_derive_sept_heat_home_features(schedule)
        lookup_flag = derived.set_index("game_id")["oracle_" + column]
        features[column] = features.game_id.map(lookup_flag)
    output_name = ("oracle_" if oracle else "") + name
    treatment = opener_pick_evaluation(
        confirmation.DEFAULT_MARKET_ROOT,
        features,
        active_model_config=confirmation.model_config(profile),
        min_train_games=500,
    )
    paired = confirmation.paired_frame(baseline, treatment)
    paired.to_csv(OUTPUT / f"{output_name}_paired.csv", index=False)
    stamp_sidecar(OUTPUT / f"{output_name}_paired.csv", project_root=ROOT)
    results = {}
    windows = {"rotation_2020_2021": (2020, 2021)}
    if name in ("qb_revenge", "both"):
        windows["promotion_2020_2025"] = (2020, 2025)
    if name == "both":
        del windows["rotation_2020_2021"]
    for label, (start, end) in windows.items():
        window = paired.loc[paired.season.between(start, end)]
        summary = confirmation.summarize(
            window, "baseline_correct_open_pr", "candidate_correct_open_pr", 20000, 20260902
        )
        results[label] = {
            "summary": summary,
            "reliability": reliability(features.loc[features.season.between(start, end)], column),
            "feature_missing_games": int(
                features.loc[features.season.between(start, end), column].isna().sum()
            ),
        }
    payload = {
        "candidate": name,
        "oracle": oracle,
        "profile": profile,
        "input_sha256": digest,
        "grade": "opener",
        "results": results,
        "baseline_warning": (
            "rerun uses frozen current table; compare old feature digest "
            "before attributing changes solely to assignment fix"
        ),
    }
    write_stamped_artifact(payload, OUTPUT / f"{output_name}.json", project_root=ROOT)
    print(json.dumps(payload, indent=2), flush=True)


RECORDS = {
    "rookie_qb_debut_fade_on_production": ("rookie_debut", "rotation_2020_2021"),
    "qb_revenge_on_production": ("qb_revenge", "rotation_2020_2021"),
    "dome_shootout_favorite_on_production": ("dome_shootout", "rotation_2020_2021"),
    "sept_heat_home_on_production": ("sept_heat", "rotation_2020_2021"),
    "qb_revenge_promotion_eval_20260905": ("qb_revenge", "promotion_2020_2025"),
    "qb_revenge_deadline_drag_stack_promotion_eval_20260905": ("both", "promotion_2020_2025"),
}


def record() -> None:
    registry = json.loads((ROOT / "registry/weak_signals.json").read_text())
    old_path = OUTPUT / "superseded_registry_entries.json"
    if not old_path.exists():
        write_stamped_artifact(
            {"signals": {name: registry["signals"][name] for name in RECORDS}},
            old_path,
            project_root=ROOT,
        )
    old_entries = json.loads(old_path.read_text())["signals"]
    commands = []
    for name, (candidate, window) in RECORDS.items():
        source = OUTPUT / f"{candidate}.json"
        result = json.loads(source.read_text())["results"][window]
        oracle = json.loads((OUTPUT / f"oracle_{candidate}.json").read_text())["results"][window]
        summary = result["summary"]
        old = old_entries[name]
        note = (
            f"CX8 decision-time rerun supersedes recorded-assignment read {old['source']}. "
            f"Original effect {old['effect']} accuracy points, "
            f"probability_positive {old['probability_positive']}. "
            f"Matched-current-table oracle: oracle_{candidate}.json, {window}. "
            "Same chronological reused windows, no new rotation window. "
            "Untimestamped QB observations stay missing; "
            "2020-2021 QB arms reproduce baseline picks exactly "
            "and probability_positive=0 there counts strict gains from an exact tie, "
            "not wrong-sign evidence. "
            f"Odd/even exposure diagnostic: {result['reliability']}; "
            f"oracle: {oracle['reliability']}. "
            "Exposure stability is not repeated-observation reliability "
            "and is not a closing ground."
        )
        args = [
            str(ROOT / ".tools/uv.exe"),
            "run",
            "--no-sync",
            "nfl-ats",
            "weak-signals",
            "record",
            "--replace",
            "--name",
            name,
            "--description",
            "CX8 decision-time rerun: " + old["description"],
            "--source",
            str(source.relative_to(ROOT)),
            "--effect",
            str(100 * summary["delta_accuracy"]),
            "--effect-units",
            "accuracy_points",
            "--classification",
            "unresolved_below_power",
            "--league",
            "nfl",
            "--season-start",
            "2020",
            "--season-end",
            "2021" if "rotation" in window else "2025",
            "--interval-low",
            str(100 * summary["week_blocked_ci95"][0]),
            "--interval-high",
            str(100 * summary["week_blocked_ci95"][1]),
            "--probability-positive",
            str(summary["week_blocked_probability_positive"]),
            "--sample-games",
            str(summary["n_games"]),
            "--sample-blocks",
            str(summary["n_weeks"]),
            "--family",
            old["family"],
            "--category",
            old.get("category", "modeling"),
            "--classification-evidence",
            (
                "No resolved wrong sign or demonstrated candidate-sized positive-control bound; "
                "assignment coverage is not a mechanism refutation."
            ),
            "--plain-summary",
            (
                "This historical rule was checked again using information available before "
                "the pick deadline. The result remains unresolved."
            ),
            "--notes",
            note,
        ]
        completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        commands.append(
            {
                "args": args,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        write_stamped_artifact(
            {"commands": commands}, OUTPUT / "record_commands.json", project_root=ROOT
        )
        print(name, completed.returncode, completed.stdout, completed.stderr, flush=True)
        if completed.returncode:
            raise RuntimeError(
                "record CLI failed; inspect record_commands.json before adjudicating"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=(*ATTACH, "both"))
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record:
        record()
    elif args.candidate:
        rerun(args.candidate, oracle=args.oracle)
    else:
        audit()

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nfl_ats.best_pick_refresh_prospective import (
    CONTENDER_FIELDS,
    PROBABILITY_SEMANTICS,
    _contender_snapshot_sha256,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL = (
    REPO / "artifacts" / "confidence_best_pick_sunday_matched" / "20260920_fixed" / "weekly.parquet"
)
DEFAULT_PAIRED = REPO / "artifacts" / "prospective" / "best_pick_refresh_decisions.parquet"
DEFAULT_PAPER = REPO / "artifacts" / "clv_ledger" / "decisions.parquet"
DEFAULT_CONTENDERS = REPO / "artifacts" / "prospective" / "best_pick_refresh_contenders.parquet"
EMBARGO_START_WEEK = 4
EMBARGO_END_WEEK = 18
GROWTH_FRACTION = 0.10


def display_path(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return str(resolved)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path, label):
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {display_path(resolved)}")
    return resolved


def iso(value):
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def load_baseline(path):
    frame = pd.read_parquet(path, columns=["protocol", "season", "week"])
    chronological = frame.loc[frame["protocol"].astype(str).eq("chronological")]
    weeks = chronological[["season", "week"]].drop_duplicates()
    if weeks.empty:
        raise ValueError("historical artifact has no chronological weeks")
    return {
        "path": display_path(path),
        "sha256": file_sha256(path),
        "chronological_weeks": len(weeks),
        "season_start": int(pd.to_numeric(weeks["season"]).min()),
        "season_end": int(pd.to_numeric(weeks["season"]).max()),
    }


def paired_nominees(path, season):
    ledger_columns = set(pq.read_schema(path).names)
    columns = [
        "season",
        "week",
        "pool_json",
        "paired_at_utc",
        "nominees_differ",
        "tuesday_game_id",
        "tuesday_kickoff",
        "tuesday_recorded_at_utc",
        "sunday_game_id",
        "sunday_kickoff",
        "sunday_recorded_at_utc",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)].copy()
    if frame.duplicated(["season", "week"]).any():
        raise ValueError("paired nominee ledger has duplicate season-week rows")
    rows = []
    all_pool_fields = set()
    for row in frame.sort_values(["season", "week"]).itertuples(index=False):
        pool = json.loads(row.pool_json)
        if not isinstance(pool, list):
            raise ValueError(f"paired nominee pool is not a list for week {int(row.week)}")
        game_ids = [str(item.get("game_id", "")) for item in pool]
        if any(not game_id for game_id in game_ids) or len(game_ids) != len(set(game_ids)):
            raise ValueError(f"paired nominee pool has invalid game IDs for week {int(row.week)}")
        pool_fields = sorted({str(key) for item in pool for key in item})
        all_pool_fields.update(pool_fields)
        rows.append(
            {
                "season": int(row.season),
                "week": int(row.week),
                "tuesday_game_id": str(row.tuesday_game_id),
                "tuesday_recorded_at_utc": iso(row.tuesday_recorded_at_utc),
                "tuesday_kickoff": iso(row.tuesday_kickoff),
                "sunday_game_id": None if pd.isna(row.sunday_game_id) else str(row.sunday_game_id),
                "sunday_recorded_at_utc": iso(row.sunday_recorded_at_utc),
                "sunday_kickoff": iso(row.sunday_kickoff),
                "paired_at_utc": iso(row.paired_at_utc),
                "nominees_differ": (
                    None if pd.isna(row.nominees_differ) else bool(row.nominees_differ)
                ),
                "tuesday_pool_members": len(pool),
                "tuesday_pool_fields": pool_fields,
            }
        )
    probability_fields = {
        "probability",
        "home_cover_probability",
        "pick_probability",
        "statistic",
    }
    return {
        "path": display_path(path),
        "sha256": file_sha256(path),
        "rows": rows,
        "weeks_with_tuesday_nominee": sum(row["tuesday_game_id"] is not None for row in rows),
        "weeks_with_sunday_nominee": sum(row["sunday_game_id"] is not None for row in rows),
        "tuesday_pool_fields": sorted(all_pool_fields),
        "tuesday_pool_contains_contender_probabilities": bool(
            probability_fields.intersection(all_pool_fields)
        ),
        "sunday_pool_retained": "sunday_pool_json" in ledger_columns,
    }


def paper_nominees(path, season):
    columns = [
        "season",
        "week",
        "game_id",
        "kickoff",
        "recorded_at_utc",
        "forecast_artifact",
        "is_best_pick",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)].copy()
    if frame.duplicated(["season", "week", "game_id"]).any():
        raise ValueError("paper ledger has duplicate season-week-game rows")
    grouped = frame.groupby(["season", "week"], sort=True)
    rows = []
    for (row_season, week), group in grouped:
        nominees = group.loc[group["is_best_pick"].eq(True)]
        if len(nominees) != 1:
            raise ValueError(f"paper ledger week {int(week)} has {len(nominees)} Best Pick rows")
        selected = nominees.iloc[0]
        timestamps = pd.to_datetime(group["recorded_at_utc"], utc=True)
        rows.append(
            {
                "season": int(row_season),
                "week": int(week),
                "game_id": str(selected["game_id"]),
                "recorded_at_utc": iso(selected["recorded_at_utc"]),
                "kickoff": iso(selected["kickoff"]),
                "forecast_artifact": str(selected["forecast_artifact"]).replace("\\", "/"),
                "card_games": len(group),
                "card_recording_instants": int(timestamps.nunique()),
                "card_recorded_at_min_utc": iso(timestamps.min()),
                "card_recorded_at_max_utc": iso(timestamps.max()),
            }
        )
    return {
        "path": display_path(path),
        "sha256": file_sha256(path),
        "rows": rows,
        "weeks_with_nominee": len(rows),
    }


def contender_snapshots(path, season):
    if not path.is_file():
        return {
            "path": display_path(path),
            "exists": False,
            "sha256": None,
            "snapshots": [],
            "weeks_with_both_phases": [],
            "pre_embargo_snapshot_weeks": [],
        }
    metadata_columns = [
        "refresh_run_id",
        "model_id",
        "feature_table_sha256",
        "source_forecast_artifact",
        "probability_semantics",
    ]
    columns = [
        "season",
        "week",
        "phase",
        "captured_at_utc",
        *metadata_columns,
        "snapshot_sha256",
        *CONTENDER_FIELDS,
    ]
    columns = list(dict.fromkeys(columns))
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)].copy()
    snapshots = []
    phases_by_week = {}
    for (row_season, week, phase), group in frame.groupby(["season", "week", "phase"], sort=True):
        hashes = set(group["snapshot_sha256"].astype(str))
        captures = pd.to_datetime(group["captured_at_utc"], utc=True)
        selected = group.loc[group["selected"].fillna(False).astype(bool)]
        eligible = group["eligible_at_capture"].fillna(False).astype(bool)
        if str(phase) not in {"tuesday", "sunday"}:
            raise ValueError(f"contender snapshot phase is invalid for week {int(week)}")
        if len(hashes) != 1 or captures.nunique() != 1:
            raise ValueError(f"contender snapshot phase is ambiguous for week {int(week)} {phase}")
        inconsistent = [
            column for column in metadata_columns if group[column].astype(str).nunique() != 1
        ]
        if inconsistent:
            raise ValueError(
                f"contender snapshot metadata is inconsistent for week {int(week)} {phase}: "
                + ", ".join(inconsistent)
            )
        if group["game_id"].astype(str).duplicated().any():
            raise ValueError(f"contender snapshot has duplicate games for week {int(week)} {phase}")
        if len(selected) != 1 or not bool(eligible.loc[selected.index].all()):
            raise ValueError(f"contender snapshot selection is invalid for week {int(week)}")
        metadata = {
            "season": int(row_season),
            "week": int(week),
            "phase": str(phase),
            "captured_at_utc": captures.iloc[0].isoformat(),
            **{column: str(group[column].iloc[0]) for column in metadata_columns},
        }
        snapshot_sha256 = next(iter(hashes))
        if metadata["probability_semantics"] != PROBABILITY_SEMANTICS:
            raise ValueError(
                f"contender snapshot probability semantics are invalid for week {int(week)} {phase}"
            )
        if _contender_snapshot_sha256(group, metadata) != snapshot_sha256:
            raise ValueError(f"contender snapshot hash is invalid for week {int(week)} {phase}")
        phases_by_week.setdefault(int(week), set()).add(str(phase))
        snapshots.append(
            {
                "season": int(row_season),
                "week": int(week),
                "phase": str(phase),
                "captured_at_utc": iso(captures.iloc[0]),
                "candidate_count": len(group),
                "eligible_count": int(eligible.sum()),
                "selected_game_id": str(selected.iloc[0]["game_id"]),
                "snapshot_sha256": snapshot_sha256,
                "probability_semantics": metadata["probability_semantics"],
                "canonical_hash_verified": True,
            }
        )
    pre_embargo = sorted(
        int(week) for week in frame["week"].dropna().unique() if 1 <= int(week) < EMBARGO_START_WEEK
    )
    return {
        "path": display_path(path),
        "exists": True,
        "sha256": file_sha256(path),
        "snapshots": snapshots,
        "weeks_with_both_phases": sorted(
            week for week, phases in phases_by_week.items() if phases == {"tuesday", "sunday"}
        ),
        "pre_embargo_snapshot_weeks": pre_embargo,
    }


def reconciliation(paired, paper):
    paper_by_week = {row["week"]: row for row in paper["rows"]}
    rows = []
    for row in paired["rows"]:
        paper_row = paper_by_week.get(row["week"])
        rows.append(
            {
                "week": row["week"],
                "paired_tuesday_game_id": row["tuesday_game_id"],
                "paper_game_id": None if paper_row is None else paper_row["game_id"],
                "same_nominee": (
                    None if paper_row is None else row["tuesday_game_id"] == paper_row["game_id"]
                ),
            }
        )
    paired_weeks = {row["week"] for row in paired["rows"]}
    missing_paired = sorted(row["week"] for row in paper["rows"] if row["week"] not in paired_weeks)
    return {"overlap": rows, "paper_weeks_missing_from_paired_ledger": missing_paired}


def embargo_status(paper, now):
    rows = [row for row in paper["rows"] if EMBARGO_START_WEEK <= row["week"] <= EMBARGO_END_WEEK]
    captured = {row["week"] for row in rows}
    expected = set(range(EMBARGO_START_WEEK, EMBARGO_END_WEEK + 1))
    last_kickoff = max(
        (pd.Timestamp(row["kickoff"]) for row in rows if row["week"] == EMBARGO_END_WEEK),
        default=None,
    )
    complete = captured == expected and last_kickoff is not None and now > last_kickoff
    return {
        "season": rows[0]["season"] if rows else None,
        "start_week": EMBARGO_START_WEEK,
        "end_week": EMBARGO_END_WEEK,
        "captured_weeks": sorted(captured),
        "missing_weeks": sorted(expected.difference(captured)),
        "latest_week_18_nominee_kickoff": iso(last_kickoff),
        "outcome_fields_read": False,
        "interim_performance_reported": False,
        "complete": complete,
    }


def build_report(historical_path, paired_path, paper_path, contender_path, season, now):
    historical = load_baseline(historical_path)
    paired = paired_nominees(paired_path, season)
    paper = paper_nominees(paper_path, season)
    contenders = contender_snapshots(contender_path, season)
    embargo = embargo_status(paper, now)
    required_growth = math.ceil(historical["chronological_weeks"] * GROWTH_FRACTION)
    observed_growth = paper["weeks_with_nominee"]
    required_embargo_weeks = set(range(EMBARGO_START_WEEK, EMBARGO_END_WEEK + 1))
    full_distribution = required_embargo_weeks.issubset(contenders["weeks_with_both_phases"])
    growth_trigger_met = observed_growth >= required_growth
    return {
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "purpose": "confidence and Best Pick nominee readiness without outcome scoring",
        "historical_baseline": historical,
        "prospective_paired_nominees": paired,
        "prospective_paper_nominees": paper,
        "prospective_contender_snapshots": contenders,
        "nominee_reconciliation": reconciliation(paired, paper),
        "growth_trigger": {
            "fraction": GROWTH_FRACTION,
            "baseline_weeks": historical["chronological_weeks"],
            "required_new_weeks": required_growth,
            "observed_timestamped_paper_nominee_weeks": observed_growth,
            "met": growth_trigger_met,
            "actionable_before_embargo_completion": False,
        },
        "lock_time_contender_distribution": {
            "tuesday_eligibility_pool_retained": bool(paired["rows"]),
            "tuesday_contender_probabilities_retained_in_paired_ledger": paired[
                "tuesday_pool_contains_contender_probabilities"
            ],
            "sunday_contender_distribution_retained_in_paired_ledger": paired[
                "sunday_pool_retained"
            ],
            "future_same_instant_recorder_wired": True,
            "historical_backfill_performed_this_session": False,
            "legacy_pre_embargo_snapshot_weeks_present": contenders["pre_embargo_snapshot_weeks"],
            "weeks_with_both_future_snapshots": contenders["weeks_with_both_phases"],
            "full_distribution_retained": full_distribution,
            "source_evidence": [
                "src/nfl_ats/best_pick_refresh_prospective.py",
                "src/nfl_ats/pick_refresh.py:1627",
            ],
        },
        "week_4_18_embargo": embargo,
        "readiness": {
            "coverage_reaudit_trigger_met": growth_trigger_met,
            "whole_2026_fold_available": embargo["complete"],
            "top_pick_recalibration_ready": growth_trigger_met and embargo["complete"],
            "within_week_ranking_reassessment_ready": (
                growth_trigger_met and embargo["complete"] and full_distribution
            ),
            "missing_requirements": [
                label
                for label, present in (
                    ("growth_trigger", growth_trigger_met),
                    ("week_4_18_embargo_complete", embargo["complete"]),
                    ("full_lock_time_contender_distribution", full_distribution),
                )
                if not present
            ],
        },
        "interpretation_contract": {
            "fits_run": False,
            "candidates_scored": False,
            "outcomes_read": False,
            "week_4_18_interim_performance_exposed": False,
            "zero_crossing_closes_signal": False,
            "one_fitted_probability_selects_side": True,
        },
    }


def write_report(path, report):
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary report already exists: {display_path(temporary)}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--paired-ledger", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--paper-ledger", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--contender-ledger", type=Path, default=DEFAULT_CONTENDERS)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--now")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    now = pd.Timestamp(args.now or datetime.now(UTC))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    report = build_report(
        require_file(args.historical, "historical weekly artifact"),
        require_file(args.paired_ledger, "paired nominee ledger"),
        require_file(args.paper_ledger, "paper decision ledger"),
        args.contender_ledger.resolve(),
        args.season,
        now,
    )
    if args.output is not None:
        write_report(args.output, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    ready = report["readiness"]["within_week_ranking_reassessment_ready"]
    return 1 if args.require_ready and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())

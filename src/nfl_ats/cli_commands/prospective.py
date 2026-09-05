"""Prospective challenger recording and scoring commands."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.cli_common import (
    _add_bootstrap_args,
    _add_features_arg,
    _add_season_week_args,
    _artifacts_root,
    _data_root,
    _load_features,
    _print_json,
    _registry_root,
)
from nfl_ats.clv import live_close_reference, load_paper_decisions, week_blocked_bootstrap
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.prospective_scoring import (
    active_challenger_ids,
    find_challenger,
    find_challenger_artifact,
    load_challenger_decisions,
    prospective_accuracy,
    prospective_accuracy_metrics,
    prospective_week_summary,
    record_challenger_decisions,
    settle_prospective_picks,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact


def _cmd_prospective_record(args: argparse.Namespace) -> None:
    artifacts = _artifacts_root()
    entry = find_challenger(artifacts, args.challenger)
    artifact = args.artifact
    if artifact is None:
        artifact = find_challenger_artifact(artifacts, entry, season=args.season, week=args.week)
        if artifact is None:
            raise ValueError(
                f"No margin-predict artifact for {args.season} week {args.week} matches "
                f"challenger {args.challenger!r}. Generate it first with the challenger's "
                "registered weekly_generation_command, then re-run."
            )
    _print_json(
        record_challenger_decisions(artifacts, args.challenger, artifact, now=datetime.now(UTC))
    )


def _prospective_entrant_report(
    name: str,
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    close_reference: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Settle one entrant's ledger slice and summarize it (with intervals when settled)."""

    settled = settle_prospective_picks(decisions, outcomes, close_reference=close_reference)
    settled.insert(0, "entrant", name)
    report: dict[str, Any] = {
        "entrant": name,
        **prospective_accuracy(settled),
        "weeks": prospective_week_summary(settled).to_dict(orient="records"),
    }
    resolved = settled.dropna(subset=["correct_at_decision_line"])
    if not resolved.empty:
        report["uncertainty"] = week_blocked_bootstrap(
            resolved,
            prospective_accuracy_metrics,
            block="week",
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ).to_dict(orient="records")
    return settled, report


def _prospective_primary_entrants(active: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Expose the played policy and its frozen raw-model control."""

    entrants = [("active_model", active)]
    if active.empty or "model_pick_side" not in active.columns:
        return entrants
    raw_incumbent = active.copy()
    raw_incumbent["pick_side"] = raw_incumbent["model_pick_side"].astype(str)
    raw_incumbent["bet_side"] = "PASS"
    raw_incumbent["edge"] = float("nan")
    entrants.append(("base_model_no_pick_overlays", raw_incumbent))
    return entrants


def _cmd_prospective_score(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    artifacts = _artifacts_root()
    features = _load_features(args.features)
    outcomes = features.loc[:, ["game_id", "result"]].copy()
    close_reference = live_close_reference(_data_root() / "market" / "raw", features, as_of=now)

    active = load_paper_decisions(artifacts)
    if not active.empty:
        active = active.loc[active["season"].astype(int).ge(args.start_season)]
    entrants = _prospective_primary_entrants(active)
    if not args.skip_challengers:
        challengers = load_challenger_decisions(artifacts)
        if not challengers.empty:
            challengers = challengers.loc[challengers["season"].astype(int).ge(args.start_season)]
        for challenger_id in sorted(set(challengers["challenger_id"].astype(str))):
            entrants.append(
                (challenger_id, challengers.loc[challengers["challenger_id"].eq(challenger_id)])
            )

    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for name, decisions in entrants:
        settled, report = _prospective_entrant_report(
            name,
            decisions.reset_index(drop=True),
            outcomes,
            close_reference,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        frames.append(settled)
        reports.append(report)

    output = _artifacts_root() / "prospective_scoring" / run_id(now)
    combined = (
        pd.concat(frames, ignore_index=True)
        if any(not frame.empty for frame in frames)
        else pd.DataFrame()
    )
    if not combined.empty:
        atomic_parquet(combined, output / "settled_decisions.parquet")
        atomic_csv(
            pd.concat(
                [
                    prospective_week_summary(frame).assign(entrant=frame["entrant"].iloc[0])
                    for frame in frames
                    if not frame.empty
                ],
                ignore_index=True,
            ),
            output / "week_summary.csv",
        )
    configuration = {
        "command": "prospective-score",
        "start_season": args.start_season,
        "skip_challengers": args.skip_challengers,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    try:
        registered = [] if args.skip_challengers else active_challenger_ids(artifacts)
    except FileNotFoundError:
        registered = []
    metadata = {
        "created_at_utc": now.isoformat(),
        **configuration,
        "registered_challengers": registered,
        "entrants": reports,
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="prospective-score",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the prospective ledger commands."""

    prospective_record = subparsers.add_parser(
        "prospective-record",
        help="append a registered challenger's pre-kickoff weekly picks to the prospective "
        "ledger (POL-10); the active model's own picks are recorded by publish-predictions",
    )
    prospective_record.add_argument(
        "--challenger",
        required=True,
        help="challenger_id from artifacts/prospective/challengers.json",
    )
    _add_season_week_args(prospective_record)
    prospective_record.add_argument(
        "--artifact",
        type=Path,
        help="margin-predict artifact directory to record from; by default the newest card "
        "for the season/week whose configuration fingerprint matches the registration",
    )
    prospective_record.set_defaults(handler=_cmd_prospective_record)

    prospective_score = subparsers.add_parser(
        "prospective-score",
        help="settle every recorded prospective pick against results and report forced-pick "
        "ATS accuracy at the recorded line (primary) and the close (secondary)",
    )
    _add_features_arg(prospective_score)
    prospective_score.add_argument(
        "--start-season",
        type=int,
        default=2026,
        help="first season to score; defaults to the prospective era (2026+), because "
        "earlier seasons are historical backtests, not pre-kickoff decisions",
    )
    prospective_score.add_argument(
        "--skip-challengers",
        action="store_true",
        help="score only the active model's ledger",
    )
    _add_bootstrap_args(prospective_score, seed=20260817)
    prospective_score.set_defaults(handler=_cmd_prospective_score)

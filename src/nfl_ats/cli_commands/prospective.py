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
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.paired_prospective import paired_prospective_report
from nfl_ats.prospective_nominees import (
    SUNDAY_ENTRANT_ID,
    TUESDAY_ENTRANT_ID,
    InvalidProspectiveNomineeArmError,
    adapt_best_pick_nominee_arm,
)
from nfl_ats.prospective_scoring import (
    InvalidProspectiveArmError,
    active_challenger_ids,
    adapt_settlement_arm_for_prospective_scoring,
    dedicated_challenger_settlement_arm,
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
from nfl_ats.settlement import (
    load_results,
    render_arm_table,
    results_artifact_path,
    seasons_in_scope,
    settle_ledgers,
)


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
        record_challenger_decisions(
            artifacts,
            args.challenger,
            artifact,
            now=datetime.now(UTC),
            replace_week=bool(getattr(args, "replace_week", False)),
        )
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

    entrants = [("active_model", active)]
    if active.empty or "model_pick_side" not in active.columns:
        return entrants
    raw_incumbent = active.copy()
    raw_incumbent["pick_side"] = raw_incumbent["model_pick_side"].astype(str)
    raw_incumbent["bet_side"] = "PASS"
    raw_incumbent["edge"] = float("nan")
    entrants.append(("base_model_no_pick_overlays", raw_incumbent))
    return entrants


def _best_pick_nominee_entrants(
    artifacts: Path,
    features: pd.DataFrame,
    generic: pd.DataFrame,
    start_season: int,
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, Any]]]:
    entrant_ids = (TUESDAY_ENTRANT_ID, SUNDAY_ENTRANT_ID)
    if not generic.empty:
        generic_ids = set(generic["challenger_id"].astype(str))
        collisions = sorted(
            generic_ids.intersection({"best_pick_sunday_renomination", *entrant_ids})
        )
        if collisions:
            raise DataContractError(
                f"Best-pick nominee arms have both generic and dedicated decisions: {collisions}"
            )
    ledger_path = artifacts / "prospective" / "best_pick_refresh_decisions.parquet"
    if not ledger_path.is_file():
        return [], [
            {
                "challenger_id": entrant_id,
                "status": "missing_ledger",
                "source": f"best_pick_refresh:{entrant_id.rsplit(':', 1)[-1]}",
                "diagnostics": {},
            }
            for entrant_id in entrant_ids
        ]
    ledger = pd.read_parquet(ledger_path)
    entrants: list[tuple[str, pd.DataFrame]] = []
    statuses: list[dict[str, Any]] = []
    for entrant_id in entrant_ids:
        diagnostics: dict[str, Any] = {}
        invalid_arm = False
        try:
            decisions = adapt_best_pick_nominee_arm(
                ledger,
                features,
                entrant_id=entrant_id,
                diagnostics=diagnostics,
            )
        except InvalidProspectiveNomineeArmError as exc:
            diagnostics.update(exc.diagnostics)
            decisions = pd.DataFrame()
            invalid_arm = True
        if not decisions.empty:
            seasons = pd.to_numeric(decisions["season"], errors="coerce")
            decisions = decisions.loc[seasons.ge(start_season)].copy()
        if invalid_arm:
            status = "invalid_arm"
        elif not decisions.empty:
            entrants.append((entrant_id, decisions))
            status = "scored"
        elif diagnostics.get("status") == "no_recorded_rows":
            status = "no_recorded_rows"
        else:
            status = "no_eligible_rows"
        statuses.append(
            {
                "challenger_id": entrant_id,
                "status": status,
                "source": f"best_pick_refresh:{entrant_id.rsplit(':', 1)[-1]}",
                "diagnostics": diagnostics,
            }
        )
    return entrants, statuses


def _prospective_challenger_entrants(
    artifacts: Path,
    registered: list[str],
    start_season: int,
    features: pd.DataFrame,
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, Any]]]:
    generic = load_challenger_decisions(artifacts)
    entrants: list[tuple[str, pd.DataFrame]] = []
    statuses: list[dict[str, Any]] = []
    registered_ids = set(registered)
    nominee_ids = {TUESDAY_ENTRANT_ID, SUNDAY_ENTRANT_ID}
    if "best_pick_sunday_renomination" in registered_ids and registered_ids.intersection(
        nominee_ids
    ):
        raise DataContractError(
            "Best-pick nominee base and expanded entrant IDs are both registered"
        )
    for challenger_id in sorted(registered_ids):
        if challenger_id == "best_pick_sunday_renomination":
            nominee_entrants, nominee_statuses = _best_pick_nominee_entrants(
                artifacts,
                features,
                generic,
                start_season,
            )
            entrants.extend(nominee_entrants)
            statuses.extend(nominee_statuses)
            continue
        generic_rows = (
            generic.loc[generic["challenger_id"].astype(str).eq(challenger_id)].copy()
            if not generic.empty
            else pd.DataFrame()
        )
        had_generic = not generic_rows.empty
        binding = dedicated_challenger_settlement_arm(challenger_id)
        dedicated_rows = pd.DataFrame()
        diagnostics: dict[str, Any] = {}
        source = "challenger_decisions" if had_generic else None
        ledger_exists = False
        invalid_arm = False
        if binding is not None:
            spec, arm = binding
            ledger_path = spec.path(artifacts)
            ledger_exists = ledger_path.is_file()
            dedicated_source = f"{spec.key}:{arm.label}"
            if ledger_exists:
                try:
                    dedicated_rows = adapt_settlement_arm_for_prospective_scoring(
                        pd.read_parquet(ledger_path),
                        spec=spec,
                        arm=arm,
                        diagnostics=diagnostics,
                    )
                except InvalidProspectiveArmError as exc:
                    if had_generic:
                        raise DataContractError(
                            f"Challenger {challenger_id!r} has both generic and dedicated decisions"
                        ) from None
                    diagnostics.update(exc.diagnostics)
                    invalid_arm = True
                    source = dedicated_source
            if not dedicated_rows.empty:
                if had_generic:
                    raise DataContractError(
                        f"Challenger {challenger_id!r} has both generic and dedicated decisions"
                    )
                source = dedicated_source
        decisions = generic_rows if had_generic else dedicated_rows
        if not decisions.empty:
            seasons = pd.to_numeric(decisions["season"], errors="coerce")
            decisions = decisions.loc[seasons.ge(start_season)].copy()
        if invalid_arm:
            status = "invalid_arm"
        elif not decisions.empty:
            entrants.append((challenger_id, decisions))
            status = "scored"
        elif had_generic or (binding is not None and ledger_exists):
            status = "no_eligible_rows"
        elif binding is not None:
            status = "missing_ledger"
            source = dedicated_source
        else:
            status = "unsupported"
        statuses.append(
            {
                "challenger_id": challenger_id,
                "status": status,
                "source": source,
                "diagnostics": diagnostics,
            }
        )
    return entrants, statuses


def _prospective_outcomes(features: pd.DataFrame, artifacts: Path) -> pd.DataFrame:

    outcomes = features.loc[:, ["game_id", "result"]].copy()
    try:
        settled = pd.read_parquet(results_artifact_path(artifacts), columns=["game_id", "result"])
    except (OSError, ValueError):
        return outcomes
    settled = settled.loc[pd.to_numeric(settled["result"], errors="coerce").notna()]
    if settled.empty:
        return outcomes
    fresher = settled.drop_duplicates("game_id")
    kept = outcomes.loc[~outcomes["game_id"].astype(str).isin(set(fresher["game_id"].astype(str)))]
    return pd.concat([kept, fresher], ignore_index=True)


def _cmd_prospective_score(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    artifacts = _artifacts_root()
    features = _load_features(args.features)
    outcomes = _prospective_outcomes(features, artifacts)
    close_reference = live_close_reference(_data_root() / "market" / "raw", features, as_of=now)

    active = load_paper_decisions(artifacts)
    if not active.empty:
        active = active.loc[active["season"].astype(int).ge(args.start_season)]
    entrants = _prospective_primary_entrants(active)
    challenger_statuses: list[dict[str, Any]] = []
    try:
        registered = [] if args.skip_challengers else active_challenger_ids(artifacts)
    except FileNotFoundError:
        registered = []
    if not args.skip_challengers:
        challenger_entrants, challenger_statuses = _prospective_challenger_entrants(
            artifacts, registered, args.start_season, features
        )
        entrants.extend(challenger_entrants)

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

    for settled, report in zip(frames[1:], reports[1:], strict=True):
        report["paired_comparison"] = paired_prospective_report(
            settled, frames[0], samples=args.bootstrap_samples, seed=args.bootstrap_seed
        )

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
    metadata = {
        "created_at_utc": now.isoformat(),
        **configuration,
        "registered_challengers": registered,
        "challenger_statuses": challenger_statuses,
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


def _cmd_settle(args: argparse.Namespace) -> None:
    artifacts = _artifacts_root()
    data_root = _data_root()
    if args.season is not None:
        seasons = [int(args.season)]
    else:
        local, _ = load_results(data_root, refresh=False)
        seasons = seasons_in_scope(local, start_season=args.start_season)
    results, provenance = load_results(
        data_root,
        seasons=seasons,
        refresh=not args.no_refresh_results,
        results_path=args.results,
    )
    finals = int(pd.to_numeric(results["result"], errors="coerce").notna().sum())
    report = settle_ledgers(
        artifacts,
        results,
        season=args.season,
        week=args.week,
        start_season=args.start_season,
        write=args.write_graded,
    )
    graded = report.pop("graded")
    if args.write_graded and not results.empty:
        atomic_parquet(results, results_artifact_path(artifacts))
    payload = {
        **report,
        "results_source": provenance,
        "results_rows": len(results),
        "results_with_a_final_score": finals,
        "wrote_graded_parquet": bool(args.write_graded),
    }
    _print_json(payload)
    print(render_arm_table(report["arms"]))
    pending = sum(int(row["pending"]) for row in report["arms"])
    settled = sum(int(row["won"]) + int(row["lost"]) + int(row["pushed"]) for row in report["arms"])
    print(
        f"settle: {len(graded)} graded rows across {len(report['arms'])} arms; "
        f"{settled} settled, {pending} pending"
    )


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:

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
    prospective_record.add_argument(
        "--replace-week",
        action="store_true",
        help=(
            "operator override (owner, 2026-09-09): drop this challenger's existing rows "
            "for the season/week first, keeping the prior ledger as a timestamped .bak "
            "beside it, so a week recorded from a superseded card can be re-recorded. "
            "Only rows for games still before kickoff are replaced; a row for a game "
            "already under way is left exactly as it is"
        ),
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

    settle = subparsers.add_parser(
        "settle",
        help="grade every recorded 2026 ledger against the latest final scores at its own "
        "decision line, and print won/lost/pushed/pending per arm",
    )
    settle.add_argument(
        "--season",
        type=int,
        default=None,
        help="settle one season; omit to settle every season from --start-season onward",
    )
    settle.add_argument(
        "--week",
        type=int,
        default=None,
        help="settle one week of --season; omit to settle every recorded week",
    )
    settle.add_argument(
        "--start-season",
        type=int,
        default=2026,
        help="first season to grade when --season is omitted; earlier seasons are historical "
        "backtests, not pre-kickoff decisions",
    )
    settle.add_argument(
        "--results",
        type=Path,
        default=None,
        help="a schedules parquet to grade against instead of fetching or reading the newest "
        "local snapshot",
    )
    settle.add_argument(
        "--no-refresh-results",
        action="store_true",
        help="never reach the network; grade against the newest local schedules snapshot",
    )
    settle.add_argument(
        "--write-graded",
        action="store_true",
        help="write each ledger's graded rows beside it (the recorded rows are never touched); "
        "without this the command only reports",
    )
    settle.set_defaults(handler=_cmd_settle)

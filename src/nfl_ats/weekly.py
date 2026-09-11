from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.reporting import read_json
from nfl_ats.snapshots import latest_snapshot

PLAYER_FEATURE_TABLE = "game_features_player.parquet"
PLAYER_FEATURE_PROFILE = "player"

PBP_FEATURE_MANIFEST = "game_features_pbp.manifest.json"
PLAYER_FEATURE_MANIFEST = "game_features_player.manifest.json"

WEAK_STACK_CHALLENGER_ID = "mod07_weak_signal_stack"
WEAK_STACK_FEATURE_PROFILE = "weak_stack"
WEAK_STACK_FEATURE_TABLE = "game_features_weak_stack.parquet"
WEAK_STACK_FEATURE_MANIFEST = "game_features_weak_stack.manifest.json"
WEAK_STACK_RATES_TABLE = "weak_stack_availability_rates.parquet"
WEAK_STACK_EVALUATION_TABLE = "weak_stack_availability_evaluation.csv"
WEAK_STACK_SOURCE_TABLE = "game_features_pbp.parquet"

CARD_PATH_TABLES = {
    PLAYER_FEATURE_PROFILE: PLAYER_FEATURE_TABLE,
    WEAK_STACK_FEATURE_PROFILE: WEAK_STACK_FEATURE_TABLE,
}

StepRunner = Callable[[Sequence[str]], dict[str, Any]]


def _weak_stack_build_command(processed: Path, *, refresh_player_data: bool) -> tuple[str, ...]:

    command = [
        "build-learned-availability-features",
        "--features",
        str(processed / WEAK_STACK_SOURCE_TABLE),
        "--destination",
        str(processed / WEAK_STACK_FEATURE_TABLE),
        "--rates-destination",
        str(processed / WEAK_STACK_RATES_TABLE),
        "--evaluation-destination",
        str(processed / WEAK_STACK_EVALUATION_TABLE),
    ]
    if not refresh_player_data:
        manifest = processed / WEAK_STACK_FEATURE_MANIFEST
        command += [
            "--player-snapshot",
            _manifest_snapshot(manifest, "source_player_snapshot"),
            "--player-value-snapshot",
            _manifest_snapshot(manifest, "source_player_value_snapshot"),
            "--pbp-snapshot",
            _manifest_snapshot(manifest, "source_pbp_snapshot"),
        ]
    return tuple(command)


def active_card_profile(artifacts_root: Path) -> str:

    path = artifacts_root / "active_ats_model.json"
    if not path.is_file():
        return PLAYER_FEATURE_PROFILE
    try:
        manifest = read_json(path)
    except (ValueError, OSError):
        return PLAYER_FEATURE_PROFILE
    profile = manifest.get("feature_profile")
    if not isinstance(profile, str) or not profile:
        return PLAYER_FEATURE_PROFILE
    if profile not in CARD_PATH_TABLES:
        raise WeeklyRunError(
            f"Active model {manifest.get('model_id')!r} uses "
            f"feature_profile={profile!r}, which weekly-run's card path cannot "
            f"build. Known profiles: {', '.join(sorted(CARD_PATH_TABLES))}. Add "
            "its feature table to nfl_ats.weekly.CARD_PATH_TABLES (and a build "
            "step feeding it) before running the weekly card."
        )
    return profile


class WeeklyRunError(ValueError):
    def __init__(self, message: str, *, summary: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.summary = summary


@dataclass(frozen=True)
class WeeklyStep:
    number: int
    name: str
    description: str
    command: tuple[str, ...] = ()
    skipped: bool = False
    optional: bool = False
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self.number,
            "name": self.name,
            "description": self.description,
            "command": ["python", "-m", "nfl_ats", *self.command] if self.command else [],
            "skipped": self.skipped,
        }
        if self.optional:
            payload["optional"] = True
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


def _manifest_snapshot(manifest_path: Path, key: str) -> str:

    if not manifest_path.is_file():
        raise WeeklyRunError(
            f"Production manifest not found: {manifest_path}. Re-run with "
            "--refresh-player-data to build from the latest snapshots instead."
        )
    manifest = read_json(manifest_path)
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise WeeklyRunError(f"Manifest {manifest_path} has no {key!r} snapshot id")
    return value


def _ingest_step(data_root: Path, season: int, *, skip: bool) -> WeeklyStep:

    snapshot = latest_snapshot(data_root / "raw")
    manifest = read_json(snapshot.manifest_path)
    seasons = [int(value) for value in manifest.get("seasons", [])]
    if not seasons:
        raise WeeklyRunError(f"Snapshot {snapshot.snapshot_id} manifest records no seasons")
    if season not in seasons:
        raise WeeklyRunError(
            f"Snapshot {snapshot.snapshot_id} covers seasons {min(seasons)}-{max(seasons)}, "
            f"which excludes the requested season {season}"
        )
    team_stat_seasons = [int(value) for value in manifest.get("team_stat_seasons", seasons)]
    command = (
        "ingest",
        "--start-season",
        str(min(seasons)),
        "--end-season",
        str(max(seasons)),
        "--stats-end-season",
        str(max(team_stat_seasons)),
    )
    return WeeklyStep(
        number=1,
        name="ingest",
        description="download a fresh immutable nflverse snapshot",
        command=command,
        skipped=skip,
        notes=(f"season span copied from snapshot {snapshot.snapshot_id}",),
    )


def _prospective_steps(
    *,
    season: int,
    week: int,
    data_root: Path,
    refresh_player_data: bool,
    record_decisions: bool,
) -> list[WeeklyStep]:

    processed = data_root / "processed"
    try:
        build_command = _weak_stack_build_command(
            processed, refresh_player_data=refresh_player_data
        )
    except WeeklyRunError as error:
        return [
            WeeklyStep(
                number=9,
                name="build-weak-stack-features",
                description="rebuild the MOD-07 challenger table with learned availability",
                skipped=True,
                optional=True,
                notes=(f"challenger evidence unavailable: {error}",),
            )
        ]
    snapshot_note = (
        "latest snapshots (--refresh-player-data)"
        if refresh_player_data
        else "snapshot ids pinned to the weak-stack production manifest"
    )
    return [
        WeeklyStep(
            number=9,
            name="build-weak-stack-features",
            description="rebuild the MOD-07 challenger table with learned availability",
            command=tuple(build_command),
            optional=True,
            notes=(snapshot_note,),
        ),
        WeeklyStep(
            number=10,
            name="margin-predict-challenger",
            description=f"score {season} week {week} with the MOD-07 weak-signal stack",
            command=(
                "margin-predict",
                "--season",
                str(season),
                "--week",
                str(week),
                "--features",
                str(processed / WEAK_STACK_FEATURE_TABLE),
                "--feature-profile",
                WEAK_STACK_FEATURE_PROFILE,
            ),
            optional=True,
            notes=("stays UNLINKED from the active model by construction",),
        ),
        WeeklyStep(
            number=11,
            name="prospective-record",
            description="append the challenger's pre-kickoff picks to the prospective ledger",
            command=(
                "prospective-record",
                "--challenger",
                WEAK_STACK_CHALLENGER_ID,
                "--season",
                str(season),
                "--week",
                str(week),
            ),
            skipped=not record_decisions,
            optional=True,
            notes=(
                ("the card is matched by configuration fingerprint, not by name",)
                if record_decisions
                else (
                    "skipped: pass --record-decisions to append this challenger's picks "
                    "to the prospective ledger (safe default is not to record)",
                )
            ),
        ),
        WeeklyStep(
            number=12,
            name="prospective-score",
            description="settle every recorded 2026 pick at the recorded line and the close",
            command=("prospective-score",),
            optional=True,
        ),
    ]


def plan_weekly_run(
    *,
    season: int,
    week: int,
    data_root: Path,
    artifacts_root: Path | None = None,
    refresh_player_data: bool = False,
    skip_ingest: bool = False,
    skip_prospective: bool = False,
    skip_drift: bool = False,
    record_decisions: bool = False,
    replace_week: bool = False,
) -> list[WeeklyStep]:

    processed = data_root / "processed"
    card_profile = (
        PLAYER_FEATURE_PROFILE if artifacts_root is None else active_card_profile(artifacts_root)
    )
    player_table = processed / CARD_PATH_TABLES[card_profile]
    steps = [_ingest_step(data_root, season, skip=skip_ingest)]

    steps.append(
        WeeklyStep(
            number=2,
            name="build-features",
            description="rebuild the canonical pregame feature table (postseason included)",
            command=("build-features",),
        )
    )

    pbp_command = ["build-pbp-features"]
    player_command = ["build-player-features"]
    if refresh_player_data:
        snapshot_note = "latest snapshots (--refresh-player-data)"
    else:
        pbp_snapshot = _manifest_snapshot(processed / PBP_FEATURE_MANIFEST, "source_pbp_snapshot")
        player_manifest = processed / PLAYER_FEATURE_MANIFEST
        pbp_command += ["--snapshot", pbp_snapshot]
        player_command += [
            "--player-snapshot",
            _manifest_snapshot(player_manifest, "source_player_snapshot"),
            "--player-value-snapshot",
            _manifest_snapshot(player_manifest, "source_player_value_snapshot"),
            "--pbp-snapshot",
            _manifest_snapshot(player_manifest, "source_pbp_snapshot"),
        ]
        snapshot_note = "snapshot ids pinned to the current production manifests"
    steps.append(
        WeeklyStep(
            number=3,
            name="build-pbp-features",
            description="add leak-safe play-by-play states",
            command=tuple(pbp_command),
            notes=(snapshot_note,),
        )
    )
    steps.append(
        WeeklyStep(
            number=3,
            name="build-player-features",
            description="add expected-lineup, injury, QB and continuity states",
            command=tuple(player_command),
            notes=(snapshot_note,),
        )
    )
    if card_profile == WEAK_STACK_FEATURE_PROFILE:
        steps.append(
            WeeklyStep(
                number=3,
                name="build-weak-stack-features",
                description="add learned-availability states the active model scores on",
                command=tuple(
                    _weak_stack_build_command(processed, refresh_player_data=refresh_player_data)
                ),
                notes=(snapshot_note,),
            )
        )

    steps.append(
        WeeklyStep(
            number=4,
            name="margin-backtest",
            description="re-run the walk-forward evaluation behind the weekly card",
            command=(
                "margin-backtest",
                "--features",
                str(player_table),
                "--feature-profile",
                card_profile,
                "--probability-method",
                "gaussian_median",
            ),
        )
    )
    steps.append(
        WeeklyStep(
            number=5,
            name="margin-predict",
            description=f"score {season} week {week} and activate the matching model",
            command=(
                "margin-predict",
                "--season",
                str(season),
                "--week",
                str(week),
                "--features",
                str(player_table),
                "--feature-profile",
                card_profile,
                "--probability-method",
                "gaussian_median",
            ),
        )
    )
    steps.append(
        WeeklyStep(
            number=6,
            name="assert-synchronized",
            description=(
                "abort unless the active manifest is SYNCHRONIZED on this season and week"
            ),
            notes=("no publish runs unless this check passes",),
        )
    )
    steps.append(
        WeeklyStep(
            number=7,
            name="ingest-player-arrests",
            description=(
                "build a fresh, complete player-arrests snapshot required by the production overlay"
            ),
            command=("ingest-player-arrests",),
            notes=("fatal: publication is refused when this source refresh fails",),
        )
    )
    steps.extend(
        [
            WeeklyStep(
                number=7,
                name="opener-evaluation",
                description="refresh opener grades for the newly active model",
                command=("opener-evaluation", "--features", str(player_table)),
                notes=("skip only if model id is unchanged and both matching measurements exist",),
            ),
            WeeklyStep(
                number=7,
                name="overlay-composition",
                description="refresh overlay composition against the matching opener evaluation",
                command=("overlay-composition",),
                notes=("skip only if model id is unchanged and both matching measurements exist",),
            ),
        ]
    )
    publish_command = ["publish-predictions", "--with-board"]
    if record_decisions:
        publish_command.append("--record-decisions")
        if replace_week:
            publish_command.append("--replace-week")
    steps.append(
        WeeklyStep(
            number=8,
            name="publish-predictions",
            description=(
                "write the tracked card (frozen four-overlay OR-union policy), "
                "the public site, and (with "
                "--record-decisions) the CLV ledger, this week's Best Pick, and the "
                "overlay's own prospective challenger ledger row"
            ),
            command=tuple(publish_command),
            notes=()
            if record_decisions
            else (
                "not recording to the paper-decision ledger: pass --record-decisions "
                "to weekly-run for the real weekly lock",
            ),
        )
    )
    if not skip_prospective:
        steps.extend(
            _prospective_steps(
                season=season,
                week=week,
                data_root=data_root,
                refresh_player_data=refresh_player_data,
                record_decisions=record_decisions,
            )
        )
    if not skip_drift:
        steps.append(
            WeeklyStep(
                number=13,
                name="drift-report",
                description=(
                    "monitor feature, missingness, probability and calibration drift "
                    "against recent history (read-only telemetry)"
                ),
                command=(
                    "drift-report",
                    "--season",
                    str(season),
                    "--week",
                    str(week),
                    "--features",
                    str(player_table),
                    "--feature-profile",
                    card_profile,
                ),
                optional=True,
                notes=(
                    "never blocks the card: a drift finding is operational "
                    "telemetry, never evidence about a signal",
                ),
            )
        )
    steps.append(
        WeeklyStep(
            number=14,
            name="waterfall-feed",
            description=(
                "rebuild the per-game attribution feed for the active model; "
                "publish-board fails closed on a feed built for another model"
            ),
            command=("waterfall-feed",),
            optional=False,
        )
    )
    steps.append(
        WeeklyStep(
            number=15,
            name="publish-board",
            description="regenerate the public site from the synchronized card and measurements",
            command=("publish-board",),
            optional=False,
        )
    )
    return steps


def assert_synchronized(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    predict_output: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if predict_output is not None:
        status = predict_output.get("synchronization_status")
        if status is not None and status != "SYNCHRONIZED":
            raise WeeklyRunError(
                f"margin-predict reported synchronization_status={status!r}; "
                "no evaluation matches this card's configuration and feature table"
            )
    manifest = load_active_ats_model(artifacts_root)
    if manifest is None:
        raise WeeklyRunError(f"No active ATS model manifest under {artifacts_root}")
    forecast = manifest.get("weekly_forecast")
    forecast = forecast if isinstance(forecast, dict) else {}
    if forecast.get("season") != season or forecast.get("week") != week:
        raise WeeklyRunError(
            "Active ATS model points at "
            f"{forecast.get('season')} week {forecast.get('week')}, "
            f"not the requested {season} week {week}"
        )
    return manifest


def _final_json_document(text: str) -> dict[str, Any]:

    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        return payload

    lines = text.splitlines()
    for start in reversed([index for index, line in enumerate(lines) if line.startswith("{")]):
        try:
            candidate = json.loads("\n".join(lines[start:]))
        except ValueError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise WeeklyRunError(
        f"subcommand produced no parsable JSON document on stdout "
        f"({len(lines)} lines captured; last line: {lines[-1][:160]!r})"
    )


def _cli_runner(command: Sequence[str]) -> dict[str, Any]:

    from nfl_ats import cli

    args = cli.build_parser().parse_args(list(command))
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        args.handler(args)
    text = buffer.getvalue().strip()
    return _final_json_document(text) if text else {}


def run_weekly(
    *,
    season: int,
    week: int,
    data_root: Path,
    artifacts_root: Path,
    refresh_player_data: bool = False,
    skip_ingest: bool = False,
    skip_prospective: bool = False,
    skip_drift: bool = False,
    record_decisions: bool = False,
    dry_run: bool = False,
    replace_week: bool = False,
    runner: StepRunner | None = None,
    progress: bool = True,
) -> dict[str, Any]:

    started = perf_counter()
    steps = plan_weekly_run(
        season=season,
        week=week,
        data_root=data_root,
        artifacts_root=artifacts_root,
        refresh_player_data=refresh_player_data,
        skip_ingest=skip_ingest,
        skip_prospective=skip_prospective,
        skip_drift=skip_drift,
        record_decisions=record_decisions,
        replace_week=replace_week,
    )
    summary: dict[str, Any] = {
        "command": "weekly-run",
        "season": season,
        "week": week,
        "dry_run": dry_run,
        "skip_ingest": skip_ingest,
        "skip_prospective": skip_prospective,
        "skip_drift": skip_drift,
        "refresh_player_data": refresh_player_data,
        "record_decisions": record_decisions,
        "data_root": str(data_root),
        "artifacts_root": str(artifacts_root),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "steps": [step.to_dict() for step in steps],
    }
    if dry_run:
        summary["published"] = False
        return summary

    execute = runner if runner is not None else _cli_runner
    executed: list[dict[str, Any]] = []
    summary["steps"] = executed
    predict_output: dict[str, Any] | None = None
    previous_model_id: str | None = None
    reuse_measurements = False
    published = False
    for step in steps:
        record = step.to_dict()
        executed.append(record)
        if step.skipped:
            record["status"] = "skipped"
            continue
        if progress:
            print(f"weekly-run step {step.number} {step.name} ...", file=sys.stderr)
        step_started = perf_counter()
        try:
            if step.name == "margin-predict":
                previous_model_id = (load_active_ats_model(artifacts_root) or {}).get("model_id")
            if step.name == "assert-synchronized":
                manifest = assert_synchronized(
                    artifacts_root,
                    season=season,
                    week=week,
                    predict_output=predict_output,
                )
                record["output"] = {
                    "model_id": manifest.get("model_id"),
                    "historical_evaluation": manifest.get("historical_evaluation"),
                }
                summary["active_model_id"] = manifest.get("model_id")
                summary["historical_evaluation"] = manifest.get("historical_evaluation")
                summary["previous_model_id"] = previous_model_id
                summary["model_changed"] = previous_model_id != manifest.get("model_id")
                if previous_model_id is not None and not summary["model_changed"]:
                    from nfl_ats.public_board import (
                        find_matching_opener_evaluation,
                        find_matching_overlay_composition,
                    )

                    evaluation = find_matching_opener_evaluation(artifacts_root, manifest)
                    reuse_measurements = (
                        evaluation is not None
                        and (evaluation[1] / "per_game.parquet").is_file()
                        and find_matching_overlay_composition(artifacts_root, manifest) is not None
                    )
            elif step.name in {"opener-evaluation", "overlay-composition"} and reuse_measurements:
                record["status"] = "skipped"
                record["skipped"] = True
                record["notes"] = [
                    "skipped: active model id unchanged; matching opener evaluation and "
                    "overlay composition already exist"
                ]
                record["seconds"] = perf_counter() - step_started
                continue
            else:
                record["output"] = execute(step.command)
                if step.name == "margin-predict":
                    predict_output = record["output"]
                if step.name == "publish-predictions":
                    published = True
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["seconds"] = perf_counter() - step_started
            if step.optional:
                summary.setdefault("optional_failures", []).append(step.name)
                print(
                    f"weekly-run OPTIONAL step {step.name} failed: {error}",
                    file=sys.stderr,
                )
                continue
            summary["failed_step"] = step.name
            summary["published"] = published
            summary["total_seconds"] = perf_counter() - started
            raise WeeklyRunError(
                f"weekly-run aborted at step {step.name!r}: {error}", summary=summary
            ) from error
        record["status"] = "ok"
        record["seconds"] = perf_counter() - step_started

    summary["published"] = published
    for record in executed:
        if record.get("name") != "publish-predictions":
            continue
        output = record.get("output")
        if isinstance(output, dict) and isinstance(output.get("source_policy"), dict):
            summary["source_policy"] = output["source_policy"]
    summary["total_seconds"] = perf_counter() - started
    return summary

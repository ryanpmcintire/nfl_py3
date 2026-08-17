"""The Tuesday weekly-ops sequence as one fail-closed command (SPEC-3).

Every regular-season Tuesday the same seven steps have to happen, in order,
before the pool locks: refresh the nflverse snapshot, rebuild the canonical and
enriched feature tables, re-run the walk-forward evaluation, score the week,
prove the weekly card and the evaluation behind it are the *same* model, and
only then publish. Doing that by hand is seven commands with easy-to-miss
flags, and the expensive mistake is publishing a card whose historical number
came from a different model.

So the sequence lives here instead of in a crib sheet. Steps run in order, any
failure aborts the whole run naming the step, and ``publish-predictions`` sits
strictly behind the synchronization assertion -- there is no path where a
desynchronized card reaches the public site.

Steps are described declaratively (``plan_weekly_run``) so ``--dry-run`` can
print exactly the commands a human would run as the manual fallback, and so the
ordering is testable without touching production data.

Steps 8-11 (POL-10) collect prospective 2026 evidence: they rebuild the MOD-07
weak-stack table, score the challenger's own card, record its pre-kickoff picks,
and settle everything recorded so far. They run AFTER the publish and are
``optional``: a failure is reported loudly and does not abort the run, because
the pool card is the deliverable and a missed week of research evidence must
never take the card down with it. They must still run *weekly*, before the
Tuesday lock -- a challenger pick invented after kickoff is worthless, and the
recording path refuses it (``nfl_ats.prospective_scoring``).
"""

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

#: The feature table the frozen active model is trained and scored on.
PLAYER_FEATURE_TABLE = "game_features_player.parquet"
#: Margin profile of the frozen active model.
PLAYER_FEATURE_PROFILE = "player"

#: Production manifests holding the snapshot ids the frozen tables were built
#: from. Re-passing these keeps a weekly rebuild bit-identical upstream of the
#: schedule refresh; ``--refresh-player-data`` opts into the latest snapshots.
PBP_FEATURE_MANIFEST = "game_features_pbp.manifest.json"
PLAYER_FEATURE_MANIFEST = "game_features_player.manifest.json"

#: MOD-07 weak-signal stack (SPEC-4), registered for prospective 2026 scoring in
#: ``artifacts/prospective/challengers.json``. It finished ``unresolved`` at
#: ``probability_positive`` 0.8745 against a pre-fixed 0.90 bar; prospective
#: outcomes are the only way left to resolve it that costs no registry window.
WEAK_STACK_CHALLENGER_ID = "mod07_weak_signal_stack"
WEAK_STACK_FEATURE_PROFILE = "weak_stack"
WEAK_STACK_FEATURE_TABLE = "game_features_weak_stack.parquet"
WEAK_STACK_FEATURE_MANIFEST = "game_features_weak_stack.manifest.json"
WEAK_STACK_RATES_TABLE = "weak_stack_availability_rates.parquet"
WEAK_STACK_EVALUATION_TABLE = "weak_stack_availability_evaluation.csv"
#: The weak-stack table is enriched from the PBP table, not the player table:
#: its injury columns carry LEARNED availability semantics under the same names.
WEAK_STACK_SOURCE_TABLE = "game_features_pbp.parquet"

StepRunner = Callable[[Sequence[str]], dict[str, Any]]


class WeeklyRunError(ValueError):
    """Raised when the weekly sequence cannot proceed.

    A ``ValueError`` subclass so the CLI reports it as a user-facing error
    rather than a traceback.
    """


@dataclass(frozen=True)
class WeeklyStep:
    """One entry in the Tuesday sequence.

    ``command`` is the full ``nfl-ats`` argv for a step that shells out to an
    existing subcommand, or empty for an in-process check (step 6). ``number``
    is the step's number in SPEC-3, which is why two steps share number 3.

    ``optional`` marks a step whose failure is recorded and reported but does
    not abort the run. Only the POL-10 evidence-collection steps use it; every
    step on the path to a published card stays fail-closed.
    """

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
    """Read one snapshot id out of a production feature manifest.

    Fail closed: a weekly run that silently fell back to "latest" would rebuild
    the frozen table from different player data without anybody asking for it.
    """

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
    """Step 1: refresh the nflverse snapshot over the seasons already covered.

    The season span is copied from the latest snapshot's manifest rather than
    from argparse defaults, and the requested prediction season must already be
    in it -- a snapshot that stops at last season cannot serve this week.
    """

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
    *, season: int, week: int, data_root: Path, refresh_player_data: bool
) -> list[WeeklyStep]:
    """Steps 8-11: produce and preserve this week's prospective challenger evidence.

    The weak-stack table is rebuilt from the freshly refreshed PBP table so the
    challenger sees the same week the active model does, pinned to the same
    snapshots unless ``--refresh-player-data`` was asked for. ``margin-predict``
    on this profile can never disturb the published card: no evaluation matches
    its configuration, so ``activate_matching_ats_model`` returns ``None`` and
    leaves the active manifest exactly where the publish left it.
    """

    processed = data_root / "processed"
    build_command = [
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
    if refresh_player_data:
        snapshot_note = "latest snapshots (--refresh-player-data)"
    else:
        manifest = processed / WEAK_STACK_FEATURE_MANIFEST
        try:
            build_command += [
                "--player-snapshot",
                _manifest_snapshot(manifest, "source_player_snapshot"),
                "--player-value-snapshot",
                _manifest_snapshot(manifest, "source_player_value_snapshot"),
                "--pbp-snapshot",
                _manifest_snapshot(manifest, "source_pbp_snapshot"),
            ]
        except WeeklyRunError as error:
            # Planning an optional step must never take the card path down.
            return [
                WeeklyStep(
                    number=8,
                    name="build-weak-stack-features",
                    description="rebuild the MOD-07 challenger table with learned availability",
                    skipped=True,
                    optional=True,
                    notes=(f"challenger evidence unavailable: {error}",),
                )
            ]
        snapshot_note = "snapshot ids pinned to the weak-stack production manifest"
    return [
        WeeklyStep(
            number=8,
            name="build-weak-stack-features",
            description="rebuild the MOD-07 challenger table with learned availability",
            command=tuple(build_command),
            optional=True,
            notes=(snapshot_note,),
        ),
        WeeklyStep(
            number=9,
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
            number=10,
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
            optional=True,
            notes=("the card is matched by configuration fingerprint, not by name",),
        ),
        WeeklyStep(
            number=11,
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
    refresh_player_data: bool = False,
    skip_ingest: bool = False,
    skip_prospective: bool = False,
) -> list[WeeklyStep]:
    """The Tuesday sequence, in order, resolved against local manifests."""

    processed = data_root / "processed"
    player_table = processed / PLAYER_FEATURE_TABLE
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
                PLAYER_FEATURE_PROFILE,
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
                PLAYER_FEATURE_PROFILE,
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
            name="publish-predictions",
            description=(
                "write the tracked card, the public site, the CLV ledger and this week's Best Pick"
            ),
            command=("publish-predictions", "--with-board"),
        )
    )
    if not skip_prospective:
        steps.extend(
            _prospective_steps(
                season=season,
                week=week,
                data_root=data_root,
                refresh_player_data=refresh_player_data,
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
    """Step 6: prove the card about to be published is the activated model.

    ``margin-predict`` writes the active manifest only when an evaluation with
    byte-identical configuration and feature-table hash exists; when it does
    not, it reports ``UNLINKED`` and leaves LAST week's manifest in place, still
    reading ``SYNCHRONIZED``. So checking the status alone is not enough -- the
    manifest's weekly forecast must also be this season and week.
    """

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


def _cli_runner(command: Sequence[str]) -> dict[str, Any]:
    """Run one subcommand in-process and return its JSON payload.

    Imported lazily: ``cli`` imports this module to wire the subcommand up.
    Stdout is captured so the weekly run emits exactly one JSON document.
    """

    from nfl_ats import cli

    args = cli.build_parser().parse_args(list(command))
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        args.handler(args)
    text = buffer.getvalue().strip()
    return json.loads(text) if text else {}


def run_weekly(
    *,
    season: int,
    week: int,
    data_root: Path,
    artifacts_root: Path,
    refresh_player_data: bool = False,
    skip_ingest: bool = False,
    skip_prospective: bool = False,
    dry_run: bool = False,
    runner: StepRunner | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the Tuesday sequence and return the single JSON summary payload."""

    started = perf_counter()
    steps = plan_weekly_run(
        season=season,
        week=week,
        data_root=data_root,
        refresh_player_data=refresh_player_data,
        skip_ingest=skip_ingest,
        skip_prospective=skip_prospective,
    )
    summary: dict[str, Any] = {
        "command": "weekly-run",
        "season": season,
        "week": week,
        "dry_run": dry_run,
        "skip_ingest": skip_ingest,
        "skip_prospective": skip_prospective,
        "refresh_player_data": refresh_player_data,
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
                # POL-10 evidence collection: loud, recorded, and not fatal. The
                # card is already published by the time these run.
                summary.setdefault("optional_failures", []).append(step.name)
                print(
                    f"weekly-run OPTIONAL step {step.name} failed: {error}",
                    file=sys.stderr,
                )
                continue
            summary["failed_step"] = step.name
            summary["published"] = published
            raise WeeklyRunError(f"weekly-run aborted at step {step.name!r}: {error}") from error
        record["status"] = "ok"
        record["seconds"] = perf_counter() - step_started

    summary["published"] = published
    summary["total_seconds"] = perf_counter() - started
    return summary

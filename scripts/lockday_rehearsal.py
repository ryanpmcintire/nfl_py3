"""End-to-end dress rehearsal of the lock-day RECORDING chain.

Why this exists
---------------
Week 1 2026 locks Tuesday 2026-09-08. On that one day, every registered
``ACTIVE_PROSPECTIVE`` challenger gets its only chance to write a prospective
2026 row. The write happens through three different commands and roughly
twenty independent recorder functions, and -- deliberately, so a broken
challenger can never un-publish the card -- almost every one of them is
wrapped in ``try/except -> {"recorded": 0, "error": ...}``. A challenger that
silently records nothing therefore looks *exactly* like a successful run
unless somebody reads twenty nested JSON keys.

That failure mode is not hypothetical here. It has already happened three
times (the 2026-08-18 ledger refill, the structurally-unsatisfiable NFL.com
Friday gate, and the ``refresh-picks`` cadence nobody was going to remember),
each time discovered only after the fact.

The recording guards make this chain untestable at ordinary wall-clock time:
``nfl_ats.clv.refuse_if_outside_recording_lock_window`` refuses any write
whose week's earliest kickoff is more than ``RECORDING_LOCK_WINDOW`` (7 days)
away, so before 2026-09-03 the whole chain is a documented no-op. This script
shifts the CLOCK rather than the data -- every recorder here accepts a ``now``
override -- so the real code paths, the real guards and the real registry run
against a real card at a simulated lock instant.

Nothing here can touch production evidence: the artifacts root is an isolated
copy, and the real ``artifacts/`` tree is only ever read.

Usage
-----
    uv run --no-sync python scripts/lockday_rehearsal.py
    uv run --no-sync python scripts/lockday_rehearsal.py --season 2026 --week 1

Exit code is 0 only when every ACTIVE_PROSPECTIVE challenger recorded at
least one row.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import lockday_verify  # noqa: E402  -- sibling script, same directory

from nfl_ats.active_model import active_artifact_path, load_active_ats_model  # noqa: E402
from nfl_ats.best_pick_big_spread_challenger import (  # noqa: E402
    record_big_spread_nomination_challenger_decisions,
)
from nfl_ats.best_pick_nomination import (  # noqa: E402
    record_nomination_challenger_decisions,
    record_nomination_v3_challenger_decisions,
)
from nfl_ats.clv import record_paper_decisions  # noqa: E402
from nfl_ats.coach_fade_overlay import record_overlay_challenger_decisions  # noqa: E402
from nfl_ats.division_revenge_tilt_overlay import (  # noqa: E402
    record_division_revenge_tilt_challenger_decisions,
)
from nfl_ats.ecdf_mapping_incumbent_overlay import (  # noqa: E402
    record_ecdf_mapping_incumbent_challenger_decisions,
)
from nfl_ats.era_weighted_half_life_8_overlay import (  # noqa: E402
    record_era_weighted_half_life_8_challenger_decisions,
)
from nfl_ats.forecast_cold_visitor_tilt_overlay import (  # noqa: E402
    record_forecast_cold_visitor_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (  # noqa: E402
    record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (  # noqa: E402
    record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions,
)
from nfl_ats.four_overlay_incumbent import (  # noqa: E402
    record_former_production_incumbent_decisions,
)
from nfl_ats.injury_value_tilt_overlay import (  # noqa: E402
    record_injury_value_tilt_challenger_decisions,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (  # noqa: E402
    record_interim_hc_first_game_tilt_challenger_decisions,
)
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (  # noqa: E402
    record_pbp08_protection_mismatch_tilt_challenger_decisions,
)
from nfl_ats.prospective import (  # noqa: E402
    record_movement_rule_composed_challenger_decisions,
    record_nflcom_refresh_out2_starters_challenger_decisions,
)
from nfl_ats.prospective_scoring import (  # noqa: E402
    find_challenger,
    find_challenger_artifact,
    record_challenger_decisions,
)
from nfl_ats.spread_gap_zone_fade_overlay import (  # noqa: E402
    record_spread_gap_zone_fade_challenger_decisions,
)
from nfl_ats.surface_switch_tilt_overlay import (  # noqa: E402
    record_surface_switch_tilt_challenger_decisions,
)
from nfl_ats.weekly import WEAK_STACK_CHALLENGER_ID  # noqa: E402

#: Tuesday 2026-09-08, noon ET -- the instant the pool locks Week 1.
DEFAULT_LOCK_INSTANT = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)

#: A Thursday-afternoon late-week refresh pass, after Wednesday's NE@SEA
#: opener has already kicked off (so the per-game deadline guard is exercised
#: on a real started game, not only on future ones).
DEFAULT_REFRESH_INSTANT = datetime(2026, 9, 10, 19, 0, tzinfo=UTC)

#: Artifacts a recorder resolves by path rather than by scanning. Copied into
#: the isolated root; anything else a recorder needs shows up as a named
#: error in the report rather than as a silent zero.
ALWAYS_COPY = ("active_ats_model.json",)
ALWAYS_COPY_DIRS = ("prospective", "clv_ledger", "player_arrests_policy_eval")


def build_isolated_root(real_artifacts: Path, destination: Path) -> dict[str, Any]:
    """Copy the minimum artifact set a lock-day recording needs.

    The linked weekly forecast and the evaluation it is synchronized against
    are resolved from the manifest rather than hardcoded, so this keeps
    working after the next promotion.
    """

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    copied: list[str] = []
    for name in ALWAYS_COPY:
        source = real_artifacts / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    for name in ALWAYS_COPY_DIRS:
        source = real_artifacts / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
            copied.append(f"{name}/")

    manifest = load_active_ats_model(destination)
    if manifest is None:
        raise SystemExit(f"No active ATS model manifest under {real_artifacts}")

    for key in ("weekly_forecast", "historical_evaluation"):
        linked = active_artifact_path(real_artifacts, manifest, key)
        if linked is None or not linked.is_dir():
            continue
        relative = linked.relative_to(real_artifacts)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(linked, target, dirs_exist_ok=True)
        copied.append(str(relative))

    # The paper-decision ledger must start empty: this rehearsal is about
    # proving a FIRST write lands, and a copied-in row would mask a recorder
    # that never wrote anything.
    for ledger in (
        destination / "clv_ledger" / "decisions.parquet",
        destination / "prospective" / "challenger_decisions.parquet",
        destination / "prospective" / "pick_revisions.parquet",
        destination / "prospective" / "injury_signal_refresh_decisions.parquet",
        destination / "prospective" / "nflcom_friday_refresh_decisions.parquet",
    ):
        if ledger.is_file():
            ledger.unlink()

    return {
        "root": str(destination),
        "copied": copied,
        "model_id": manifest.get("model_id"),
        "feature_profile": manifest.get("feature_profile"),
    }


#: Trees the NFL lock-day recorders never read. Skipped when shadowing the
#: data root purely to keep the shadow cheap.
SHADOW_SKIP_TOP_LEVEL = ("cfb",)

#: Real-copied rather than hard-linked, because the rehearsal restamps a
#: snapshot inside it. Editing a hard link would edit the production file.
ARRESTS_RELATIVE = Path("raw") / "player_arrests"


def build_shadow_data_root(
    real_data: Path, destination: Path, *, lock_instant: datetime
) -> dict[str, Any]:
    """Mirror the data root, with one player-arrests snapshot restamped fresh.

    Two guards make the lock-day chain unrehearsable at wall-clock time and
    they pull in opposite directions:
    ``clv.refuse_if_outside_recording_lock_window`` needs a simulated ``now``
    inside the real lock week, while
    ``player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE`` needs a snapshot no
    more than 36 hours before that same instant. Nothing fetched today can
    satisfy both.

    On the real lock day weekly-run step 7 (``ingest-player-arrests``, fatal)
    resolves this by fetching minutes before step 8 publishes. This reproduces
    that condition without writing a fabricated future-dated snapshot into the
    production data root: the mirror is built from hard links (no extra disk,
    and removing the mirror never touches the originals), and only the small
    arrests tree is copied for real so the restamped manifest cannot alias a
    production file.

    Skipping this and passing ``--assume-fresh-arrests`` instead yields a
    ledger whose ``decision_policy`` is missing its arrests member, which the
    four-overlay incumbent and the whole refresh path then reject -- the
    rehearsal blocks on its own scaffolding rather than on a real defect.
    """

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    linked = copied = 0
    for source_dir, _subdirs, filenames in os.walk(real_data):
        source_path = Path(source_dir)
        relative = source_path.relative_to(real_data)
        parts = relative.parts
        if parts and parts[0] in SHADOW_SKIP_TOP_LEVEL:
            continue
        if parts[: len(ARRESTS_RELATIVE.parts)] == ARRESTS_RELATIVE.parts:
            continue  # real-copied wholesale below
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_file = source_path / filename
            target_file = target_dir / filename
            try:
                os.link(source_file, target_file)
                linked += 1
            except OSError:
                shutil.copy2(source_file, target_file)
                copied += 1

    real_arrests = real_data / ARRESTS_RELATIVE
    shadow_arrests = destination / ARRESTS_RELATIVE
    shutil.copytree(real_arrests, shadow_arrests)

    snapshots = sorted((path for path in shadow_arrests.iterdir() if path.is_dir()), reverse=True)
    if not snapshots:
        raise SystemExit(f"No player-arrests snapshots under {real_arrests}")
    newest = snapshots[0]

    fetched_at = lock_instant - timedelta(hours=1)
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    restamped = shadow_arrests / stamp
    shutil.copytree(newest, restamped)
    manifest_path = restamped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = stamp
    manifest["fetched_at_utc"] = fetched_at.isoformat()
    manifest["rehearsal_restamped_from"] = newest.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "root": str(destination),
        "hard_linked_files": linked,
        "copied_files": copied,
        "skipped_top_level": list(SHADOW_SKIP_TOP_LEVEL),
        "arrests_snapshot": stamp,
        "arrests_restamped_from": newest.name,
        "arrests_fetched_at_utc": fetched_at.isoformat(),
    }


def _call(label: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one recorder, capturing the failure the same way production does.

    Production swallows these into ``{"recorded": 0, "error": ...}`` so the
    card still publishes. The rehearsal keeps the same shape but also keeps
    the traceback, because here the failure IS the finding.
    """

    try:
        payload = fn()
    except Exception as error:
        return {
            "recorded": 0,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=6),
        }
    if not isinstance(payload, dict):
        return {"recorded": 0, "error": f"recorder returned {type(payload).__name__}, not a dict"}
    return payload


def run_publish_recorders(
    artifacts: Path,
    data: Path,
    registry: Path,
    *,
    lock_instant: datetime,
    assume_fresh_arrests: bool = False,
) -> dict[str, dict[str, Any]]:
    """Every recorder ``publish-predictions --record-decisions`` fires, in order.

    The order matches ``nfl_ats.cli._cmd_publish_predictions`` because several
    challengers read the primary ledger this sequence writes first (the
    four-overlay incumbent reads ``former_policy_pick_side``; the movement
    rule composes onto the recorded chain pick).

    Every recorder is passed ``now=lock_instant`` explicitly. Production
    passes it to four of them and lets the other thirteen read the wall
    clock, which agrees to within seconds on the real lock day -- see
    ``report`` for the note this rehearsal emits about that.
    """

    results: dict[str, dict[str, Any]] = {}

    # The player-arrests overlay refuses any snapshot older than 36 hours
    # (nfl_ats.player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE). On the real
    # lock day that guard always passes, because weekly-run step 7
    # (``ingest-player-arrests``, fatal) fetches a fresh snapshot minutes
    # before step 8 publishes. In a rehearsal the two guards are mutually
    # unsatisfiable: the recording-lock window needs a simulated ``now``
    # inside the real lock week, and no snapshot fetched today is 36 hours
    # from that instant. ``--assume-fresh-arrests`` resolves that the
    # supported way rather than by restamping a snapshot, which would put a
    # fabricated future-dated fetch into the production data root.
    results["clv_ledger"] = _call(
        "clv_ledger",
        lambda: record_paper_decisions(
            artifacts,
            data_root=data,
            now=lock_instant,
            require_fresh_arrest_overlay=not assume_fresh_arrests,
        ),
    )
    simple: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
        ("hc_year_one_fade_overlay", record_overlay_challenger_decisions),
        ("best_pick_nomination_v2", record_nomination_challenger_decisions),
        ("best_pick_nomination_v3", record_nomination_v3_challenger_decisions),
        ("best_pick_big_spread_eligibility", record_big_spread_nomination_challenger_decisions),
        ("injury_value_lost_tilt_overlay", record_injury_value_tilt_challenger_decisions),
        ("division_revenge_tilt_overlay", record_division_revenge_tilt_challenger_decisions),
        ("surface_switch_tilt_overlay", record_surface_switch_tilt_challenger_decisions),
        ("spread_gap_zone_fade_overlay", record_spread_gap_zone_fade_challenger_decisions),
        (
            "pbp08_protection_mismatch_tilt_overlay",
            record_pbp08_protection_mismatch_tilt_challenger_decisions,
        ),
        (
            "overlay_production_chain_coach_arrest_incumbent",
            record_former_production_incumbent_decisions,
        ),
        ("ecdf_mapping_incumbent", record_ecdf_mapping_incumbent_challenger_decisions),
        ("era_weighted_half_life_8", record_era_weighted_half_life_8_challenger_decisions),
    )
    for label, recorder in simple:
        results[label] = _call(
            label, lambda recorder=recorder: recorder(artifacts, data, now=lock_instant)
        )

    results["forecast_cold_visitor_tilt"] = _call(
        "forecast_cold_visitor_tilt",
        lambda: record_forecast_cold_visitor_tilt_challenger_decisions(
            artifacts, data, registry, now=lock_instant
        ),
    )
    results["interim_hc_first_game_tilt_overlay"] = _call(
        "interim_hc_first_game_tilt_overlay",
        lambda: record_interim_hc_first_game_tilt_challenger_decisions(
            artifacts, data, now=lock_instant
        ),
    )
    results["forecast_weather_kn_warm_team_cold_late_tilt"] = _call(
        "forecast_weather_kn_warm_team_cold_late_tilt",
        lambda: record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
            artifacts, data, registry, now=lock_instant
        ),
    )
    results["forecast_weather_kn_precip_high_total_tilt"] = _call(
        "forecast_weather_kn_precip_high_total_tilt",
        lambda: record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
            artifacts, data, registry, now=lock_instant
        ),
    )
    results["movement_rule_composed_v1"] = _call(
        "movement_rule_composed_v1",
        lambda: record_movement_rule_composed_challenger_decisions(
            artifacts, data, now=lock_instant
        ),
    )
    results["nflcom_friday_refresh_out2_starters_v1"] = _call(
        "nflcom_friday_refresh_out2_starters_v1",
        lambda: record_nflcom_refresh_out2_starters_challenger_decisions(
            artifacts, data, now=lock_instant
        ),
    )
    return results


def run_weekly_run_step_11(
    artifacts: Path, real_artifacts: Path, *, season: int, week: int, lock_instant: datetime
) -> dict[str, Any]:
    """``weekly-run`` step 11 (``prospective-record``), the MOD-07 arm.

    A bare ``publish-predictions --record-decisions`` never reaches this, which
    is why the lock-day command has to be ``weekly-run``. The challenger's card
    is matched by configuration fingerprint, so this also surfaces the case
    where the challenger has drifted into being the active model's own card.
    """

    entry = find_challenger(real_artifacts, WEAK_STACK_CHALLENGER_ID)
    source = find_challenger_artifact(real_artifacts, entry, season=season, week=week)
    if source is None:
        return {
            "recorded": 0,
            "error": (
                f"no {season} week {week} card matches {WEAK_STACK_CHALLENGER_ID}'s "
                "configuration fingerprint"
            ),
        }

    relative = source.relative_to(real_artifacts)
    target = artifacts / relative
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)

    manifest = load_active_ats_model(artifacts) or {}
    active_forecast = active_artifact_path(artifacts, manifest, "weekly_forecast")
    payload = _call(
        WEAK_STACK_CHALLENGER_ID,
        lambda: record_challenger_decisions(
            artifacts, WEAK_STACK_CHALLENGER_ID, target, now=lock_instant
        ),
    )
    payload["artifact"] = str(relative)
    payload["is_the_active_models_own_card"] = active_forecast is not None and (
        active_forecast.name == source.name
    )
    return payload


def run_refresh_recorders(
    artifacts: Path, data: Path, *, season: int, week: int, refresh_instant: datetime
) -> dict[str, dict[str, Any]]:
    """The late-week pass: ``refresh-picks --record-decisions``.

    Imported lazily so a rehearsal can still report the publish half when the
    refresh module itself fails to import.
    """

    from nfl_ats.injury_signal_refresh_tilt import record_injury_signal_refresh_tilt
    from nfl_ats.nflcom_refresh_overlay import record_nflcom_refresh_overlay
    from nfl_ats.pick_refresh import plan_refresh, record_plan, refresh_summary

    results: dict[str, dict[str, Any]] = {}
    try:
        plan = plan_refresh(artifacts, data, season=season, week=week, now=refresh_instant)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "plan_refresh": {"recorded": 0, "error": message},
            "model_only_refresh_incumbent": {"recorded": 0, "error": f"no plan: {message}"},
            "injury_signal_refresh_tilt": {"recorded": 0, "error": f"no plan: {message}"},
        }

    results["plan_refresh"] = refresh_summary(plan, record_decisions=True)
    results["model_only_refresh_incumbent"] = _call(
        "model_only_refresh_incumbent",
        lambda: record_plan(artifacts, plan, note="lockday_rehearsal", record_decisions=True),
    )
    results["injury_signal_refresh_tilt"] = _call(
        "injury_signal_refresh_tilt",
        lambda: record_injury_signal_refresh_tilt(artifacts, data, plan, record_decisions=True),
    )
    results["nflcom_refresh_out2_starters_overlay"] = _call(
        "nflcom_refresh_out2_starters_overlay",
        lambda: record_nflcom_refresh_overlay(artifacts, data, plan, record_decisions=True),
    )
    return results


def ledger_coverage(
    artifacts: Path, *, season: int, week: int, report: dict[str, Any]
) -> dict[str, Any]:
    """Score coverage with the SAME verifier that will run on the real lock day.

    Deliberately not a second implementation: an audit that disagrees with the
    tool the lock day actually uses is worse than no audit. The rehearsal's own
    recorder outputs are handed over as the run summary, so a recorder that
    reported a gate ("no fresh captured line yet") is scored ``skipped`` rather
    than ``MISSING`` -- the same way it will be in production.
    """

    return lockday_verify.verify(artifacts, season=season, week=week, run_summary=report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument(
        "--lock-instant",
        default=DEFAULT_LOCK_INSTANT.isoformat(),
        help="simulated Tuesday lock instant (ISO-8601, UTC)",
    )
    parser.add_argument(
        "--refresh-instant",
        default=DEFAULT_REFRESH_INSTANT.isoformat(),
        help="simulated late-week refresh instant (ISO-8601, UTC)",
    )
    # The simulated tree mirrors the repo's own layout -- ``<sim>/artifacts``
    # beside ``<sim>/data`` -- because several overlays resolve sibling paths
    # from ``artifacts_root.parent`` (the interim-coach join reads
    # ``<repo>/data/raw/interim_coaches``). A rehearsal root parked anywhere
    # else silently fails those joins open to zero flags, so the recorder
    # writes rows that never exercise its own signal.
    parser.add_argument(
        "--rehearsal-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "rehearsal_lockday" / "sim" / "artifacts",
        help="isolated artifacts root to build and write into",
    )
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "registry")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument(
        "--assume-fresh-arrests",
        action="store_true",
        help=(
            "skip the 36-hour player-arrests freshness guard, which weekly-run "
            "step 7 satisfies for real on lock day but which no rehearsal clock "
            "can satisfy (see run_publish_recorders)"
        ),
    )
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    lock_instant = datetime.fromisoformat(args.lock_instant)
    refresh_instant = datetime.fromisoformat(args.refresh_instant)

    report: dict[str, Any] = {
        "season": args.season,
        "week": args.week,
        "lock_instant": lock_instant.isoformat(),
        "refresh_instant": refresh_instant.isoformat(),
        "rehearsed_at_utc": datetime.now(UTC).isoformat(),
    }
    report["isolated_root"] = build_isolated_root(args.artifacts, args.rehearsal_root)
    print(f"isolated root built: {args.rehearsal_root}", file=sys.stderr)

    report["assume_fresh_arrests"] = bool(args.assume_fresh_arrests)
    if args.assume_fresh_arrests:
        data_root = args.data
    else:
        shadow = args.rehearsal_root.parent / "data"
        print(f"mirroring data root (hard links): {shadow}", file=sys.stderr)
        report["shadow_data_root"] = build_shadow_data_root(
            args.data, shadow, lock_instant=lock_instant
        )
        data_root = shadow
    report["data_root"] = str(data_root)

    print("stage 1: publish-predictions --record-decisions recorders", file=sys.stderr)
    report["publish"] = run_publish_recorders(
        args.rehearsal_root,
        data_root,
        args.registry,
        lock_instant=lock_instant,
        assume_fresh_arrests=args.assume_fresh_arrests,
    )

    print("stage 2: weekly-run step 11 (prospective-record)", file=sys.stderr)
    report["weekly_run_step_11"] = {
        WEAK_STACK_CHALLENGER_ID: run_weekly_run_step_11(
            args.rehearsal_root,
            args.artifacts,
            season=args.season,
            week=args.week,
            lock_instant=lock_instant,
        )
    }

    if not args.skip_refresh:
        print("stage 3: refresh-picks --record-decisions recorders", file=sys.stderr)
        report["refresh"] = run_refresh_recorders(
            args.rehearsal_root,
            data_root,
            season=args.season,
            week=args.week,
            refresh_instant=refresh_instant,
        )

    coverage = ledger_coverage(
        args.rehearsal_root, season=args.season, week=args.week, report=report
    )
    report["coverage"] = coverage

    destination = args.report or (args.rehearsal_root.parent / "rehearsal_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(lockday_verify.render(coverage))
    print(f"full report: {destination}", file=sys.stderr)
    return 0 if not coverage["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
    uv run --no-sync python scripts/lockday_rehearsal.py --full-replay

The default is a millisecond-scale static wiring audit: it does not import the
model stack, mirror ``data/``, execute a recorder, or touch a ledger.
``--full-replay`` opts into the older isolated end-to-end recorder run; each
recorder reports start/end and fails fast after the bounded per-recorder budget.

Exit code is 0 only when every ACTIVE_PROSPECTIVE challenger recorded at
least one row.
"""

from __future__ import annotations

# Full replay dependencies are deliberately bound dynamically only after the
# fast default exits; static analysis therefore cannot see those names.
# ruff: noqa: F821
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_full_replay_dependencies() -> None:
    """Load the model/data stack only for the explicit slow replay mode."""

    bindings = {
        "record_expected_lineup_loss_challenger_decisions": (
            "nfl_ats.expected_lineup_loss_challenger",
            "record_expected_lineup_loss_challenger_decisions",
        ),
        "record_deadline_drag_challenger_decisions": (
            "nfl_ats.deadline_drag_challenger",
            "record_deadline_drag_challenger_decisions",
        ),
        "pd": ("pandas", None),
        "lockday_verify": ("lockday_verify", None),
        "active_artifact_path": ("nfl_ats.active_model", "active_artifact_path"),
        "load_active_ats_model": ("nfl_ats.active_model", "load_active_ats_model"),
        "record_big_spread_nomination_challenger_decisions": (
            "nfl_ats.best_pick_big_spread_challenger",
            "record_big_spread_nomination_challenger_decisions",
        ),
        "record_nomination_challenger_decisions": (
            "nfl_ats.best_pick_nomination",
            "record_nomination_challenger_decisions",
        ),
        "record_nomination_v3_challenger_decisions": (
            "nfl_ats.best_pick_nomination",
            "record_nomination_v3_challenger_decisions",
        ),
        "record_bye_edge_fade_challenger_decisions": (
            "nfl_ats.bye_edge_fade_overlay",
            "record_bye_edge_fade_challenger_decisions",
        ),
        "RECORDING_LOCK_WINDOW": ("nfl_ats.clv", "RECORDING_LOCK_WINDOW"),
        "record_paper_decisions": ("nfl_ats.clv", "record_paper_decisions"),
        "refuse_if_outside_recording_lock_window": (
            "nfl_ats.clv",
            "refuse_if_outside_recording_lock_window",
        ),
        "record_overlay_challenger_decisions": (
            "nfl_ats.coach_fade_overlay",
            "record_overlay_challenger_decisions",
        ),
        "record_crew_tilt_refresh_overlay": (
            "nfl_ats.crew_tilt_refresh_overlay",
            "record_crew_tilt_refresh_overlay",
        ),
        "record_division_revenge_tilt_challenger_decisions": (
            "nfl_ats.division_revenge_tilt_overlay",
            "record_division_revenge_tilt_challenger_decisions",
        ),
        "record_ecdf_mapping_incumbent_challenger_decisions": (
            "nfl_ats.ecdf_mapping_incumbent_overlay",
            "record_ecdf_mapping_incumbent_challenger_decisions",
        ),
        "record_era_weighted_half_life_8_challenger_decisions": (
            "nfl_ats.era_weighted_half_life_8_overlay",
            "record_era_weighted_half_life_8_challenger_decisions",
        ),
        "record_forecast_cold_visitor_tilt_challenger_decisions": (
            "nfl_ats.forecast_cold_visitor_tilt_overlay",
            "record_forecast_cold_visitor_tilt_challenger_decisions",
        ),
        "record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions": (
            "nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay",
            "record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions",
        ),
        "record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions": (
            "nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay",
            "record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions",
        ),
        "record_former_production_incumbent_decisions": (
            "nfl_ats.four_overlay_incumbent",
            "record_former_production_incumbent_decisions",
        ),
        "record_inactives_refresh_overlay": (
            "nfl_ats.inactives_refresh_overlay",
            "record_inactives_refresh_overlay",
        ),
        "record_injury_value_tilt_challenger_decisions": (
            "nfl_ats.injury_value_tilt_overlay",
            "record_injury_value_tilt_challenger_decisions",
        ),
        "record_interim_hc_first_game_tilt_challenger_decisions": (
            "nfl_ats.interim_hc_first_game_tilt_overlay",
            "record_interim_hc_first_game_tilt_challenger_decisions",
        ),
        "record_pace_mismatch_dog_tilt_challenger_decisions": (
            "nfl_ats.pace_mismatch_dog_tilt_overlay",
            "record_pace_mismatch_dog_tilt_challenger_decisions",
        ),
        "record_pbp08_protection_mismatch_tilt_challenger_decisions": (
            "nfl_ats.pbp08_protection_mismatch_tilt_overlay",
            "record_pbp08_protection_mismatch_tilt_challenger_decisions",
        ),
        "record_movement_rule_composed_challenger_decisions": (
            "nfl_ats.prospective",
            "record_movement_rule_composed_challenger_decisions",
        ),
        "record_nflcom_refresh_out2_starters_challenger_decisions": (
            "nfl_ats.prospective",
            "record_nflcom_refresh_out2_starters_challenger_decisions",
        ),
        "artifact_model_config": ("nfl_ats.prospective_scoring", "artifact_model_config"),
        "challenger_ledger_path": ("nfl_ats.prospective_scoring", "challenger_ledger_path"),
        "config_fingerprint": ("nfl_ats.prospective_scoring", "config_fingerprint"),
        "find_challenger": ("nfl_ats.prospective_scoring", "find_challenger"),
        "find_challenger_artifact": (
            "nfl_ats.prospective_scoring",
            "find_challenger_artifact",
        ),
        "load_challenger_decisions": (
            "nfl_ats.prospective_scoring",
            "load_challenger_decisions",
        ),
        "load_challenger_registry": (
            "nfl_ats.prospective_scoring",
            "load_challenger_registry",
        ),
        "record_challenger_decisions": (
            "nfl_ats.prospective_scoring",
            "record_challenger_decisions",
        ),
        "record_special_teams_return_tilt_challenger_decisions": (
            "nfl_ats.special_teams_return_tilt_overlay",
            "record_special_teams_return_tilt_challenger_decisions",
        ),
        "record_spread_gap_zone_fade_challenger_decisions": (
            "nfl_ats.spread_gap_zone_fade_overlay",
            "record_spread_gap_zone_fade_challenger_decisions",
        ),
        "record_surface_switch_tilt_challenger_decisions": (
            "nfl_ats.surface_switch_tilt_overlay",
            "record_surface_switch_tilt_challenger_decisions",
        ),
        "record_tank_zone_fade_tilt_challenger_decisions": (
            "nfl_ats.tank_zone_fade_tilt_overlay",
            "record_tank_zone_fade_tilt_challenger_decisions",
        ),
        "record_third_down_reversion_fade_challenger_decisions": (
            "nfl_ats.third_down_reversion_fade_overlay",
            "record_third_down_reversion_fade_challenger_decisions",
        ),
        "record_turnover_luck_rebound_tilt_challenger_decisions": (
            "nfl_ats.turnover_luck_rebound_tilt_overlay",
            "record_turnover_luck_rebound_tilt_challenger_decisions",
        ),
        "WEAK_STACK_CHALLENGER_ID": ("nfl_ats.weekly", "WEAK_STACK_CHALLENGER_ID"),
    }
    for name, (module_name, attribute) in bindings.items():
        module = importlib.import_module(module_name)
        globals()[name] = module if attribute is None else getattr(module, attribute)


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
        destination / "prospective" / "inactives_refresh_decisions.parquet",
        destination / "prospective" / "crew_tilt_refresh_decisions.parquet",
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


def probe_command_surface(repo_root: Path) -> dict[str, Any]:
    """Prove the lock-day commands DISPATCH through the real console script.

    Added 2026-08-25 after this rehearsal returned a clean "0 MISSING" while
    ``weekly-run`` step 7 was in fact broken. The rehearsal drove the recorder
    FUNCTIONS directly, so it never touched the entry point the documented
    Tuesday command actually uses -- and ``nfl-ats ingest-player-arrests``
    was raising ``ModuleNotFoundError: No module named 'scripts'`` because the
    console script does not put the repository root on ``sys.path`` the way
    ``python -m nfl_ats`` does.

    A rehearsal that cannot fail the way production fails is worse than no
    rehearsal, because it produces confidence. This stage runs the console
    script in a subprocess -- the same binary, the same ``sys.path`` -- so the
    entry point itself is under test.

    Read-only probes only: ``doctor`` touches nothing and ``weekly-run
    --dry-run`` resolves the full step plan without executing a step.
    """

    console = repo_root / ".venv" / "Scripts" / "nfl-ats.exe"
    if not console.is_file():
        console = repo_root / ".venv" / "bin" / "nfl-ats"
    if not console.is_file():
        return {"ok": False, "error": f"console script not found under {repo_root / '.venv'}"}

    probes: dict[str, Any] = {}
    for label, argv in (
        ("doctor", ["doctor"]),
        (
            "weekly_run_dry_run",
            ["weekly-run", "--season", "2026", "--week", "1", "--record-decisions", "--dry-run"],
        ),
        ("ingest_player_arrests_help", ["ingest-player-arrests", "--help"]),
    ):
        completed = subprocess.run(
            [str(console), *argv],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=300,
        )
        probes[label] = {
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "stderr_tail": completed.stderr.strip()[-400:],
        }

    # The import that actually broke is lazy, so --help cannot reach it. Run it
    # in a subprocess whose sys.path is the console script's, and confirm the
    # module resolves.
    lazy = subprocess.run(
        [
            str(repo_root / ".venv" / "Scripts" / "python.exe")
            if (repo_root / ".venv" / "Scripts" / "python.exe").is_file()
            else sys.executable,
            "-c",
            "import nfl_ats.cli as c; c._repo_root_on_path();"
            " import scripts.ingest_player_arrests as m; print(m.__name__)",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root.parent),
        timeout=120,
    )
    probes["lazy_scripts_import_off_cwd"] = {
        "returncode": lazy.returncode,
        "ok": lazy.returncode == 0,
        "stderr_tail": lazy.stderr.strip()[-400:],
        "note": (
            "run from OUTSIDE the repo so a working-directory sys.path entry "
            "cannot mask the failure"
        ),
    }

    return {"ok": all(p["ok"] for p in probes.values()), "probes": probes}


FULL_REPLAY_RECORDER_TIMEOUT_SECONDS = 30.0


def _call(
    label: str,
    fn: Callable[[], Any],
    *,
    timeout_seconds: float = FULL_REPLAY_RECORDER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one recorder, capturing the failure the same way production does.

    Production swallows these into ``{"recorded": 0, "error": ...}`` so the
    card still publishes. The rehearsal keeps the same shape but also keeps
    the traceback, because here the failure IS the finding.
    """

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    print(f"  recorder start: {label}", file=sys.stderr)
    try:
        payload = fn()
    except Exception as error:
        payload = {
            "recorded": 0,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=6),
        }
    elapsed = time.perf_counter() - started
    finished_at = datetime.now(UTC)
    print(f"  recorder end:   {label} ({elapsed:.3f}s)", file=sys.stderr)
    if not isinstance(payload, dict):
        payload = {
            "recorded": 0,
            "error": f"recorder returned {type(payload).__name__}, not a dict",
        }
    payload = dict(payload)
    payload.update(
        {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": round(elapsed, 3),
        }
    )
    if elapsed > timeout_seconds:
        payload.update(
            {
                "recorded": 0,
                "timeout_seconds": timeout_seconds,
                "error": (
                    f"TimeoutError: recorder exceeded {timeout_seconds:.1f}s "
                    f"budget ({elapsed:.3f}s)"
                ),
            }
        )
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
        ("weak_stack_expected_lineup_loss", record_expected_lineup_loss_challenger_decisions),
        ("weak_stack_deadline_drag", record_deadline_drag_challenger_decisions),
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
        ("bye_edge_fade_overlay", record_bye_edge_fade_challenger_decisions),
        ("tank_zone_fade_tilt_overlay", record_tank_zone_fade_tilt_challenger_decisions),
        (
            "third_down_reversion_fade_overlay",
            record_third_down_reversion_fade_challenger_decisions,
        ),
        (
            "turnover_luck_rebound_tilt_overlay",
            record_turnover_luck_rebound_tilt_challenger_decisions,
        ),
        (
            "special_teams_return_tilt_overlay",
            record_special_teams_return_tilt_challenger_decisions,
        ),
        ("pace_mismatch_dog_tilt_overlay", record_pace_mismatch_dog_tilt_challenger_decisions),
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
    results["inactives_refresh_v1"] = _call(
        "inactives_refresh_v1",
        lambda: record_inactives_refresh_overlay(artifacts, data, plan, record_decisions=True),
    )
    results["crew_tilt_refresh_v1"] = _call(
        "crew_tilt_refresh_v1",
        lambda: record_crew_tilt_refresh_overlay(
            artifacts,
            data,
            plan,
            repo_root=artifacts.parent,
            record_decisions=True,
        ),
    )
    from nfl_ats.late_week_move_follow_refresh_overlay import (
        record_late_week_move_follow_refresh_overlay,
    )

    results["late_week_move_follow_refresh_overlay"] = _call(
        "late_week_move_follow_refresh_overlay",
        lambda: record_late_week_move_follow_refresh_overlay(
            artifacts, data, plan, record_decisions=True
        ),
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


REFRESH_RESULT_KEYS = {
    "best_pick_sunday_renomination": "best_pick_refresh_ledger",
    "model_only_refresh_incumbent": "ledger",
    "injury_signal_refresh_tilt": "injury_signal_refresh_tilt",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_refresh_out2_starters_overlay",
    "inactives_refresh_v1": "inactives_refresh_overlay",
    "crew_tilt_refresh_v1": "crew_tilt_refresh_overlay",
    "specialist_absence_fade_refresh_v1": "specialist_absence_fade_refresh_overlay",
    "late_week_move_follow_refresh_v1": "late_week_move_follow_refresh_overlay",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_live_ledgers(artifacts: Path) -> dict[str, str]:
    """Hash known live ledgers without opening any model or historical input."""

    relative_paths = (
        Path("clv_ledger/decisions.parquet"),
        Path("prospective/challenger_decisions.parquet"),
        Path("prospective/pick_revisions.parquet"),
        Path("prospective/injury_signal_refresh_decisions.parquet"),
        Path("prospective/nflcom_friday_refresh_decisions.parquet"),
        Path("prospective/inactives_refresh_decisions.parquet"),
        Path("prospective/crew_tilt_refresh_decisions.parquet"),
        Path("prospective/specialist_absence_fade_refresh_decisions.parquet"),
        Path("prospective/late_week_move_follow_refresh_decisions.parquet"),
    )
    return {
        str(relative): _file_sha256(artifacts / relative)
        for relative in relative_paths
        if (artifacts / relative).is_file()
    }


def _recording_path(command: str) -> tuple[str, str | None]:
    """Return the documented command family and its CLI result key."""

    if "publish-predictions --record-decisions" in command:
        return "publish", None
    if "refresh-picks --record-decisions" in command:
        return "refresh", None
    if "prospective-record" in command:
        return "weekly-run", "prospective_record"
    if "scripts/record_" in command:
        return "standalone", None
    return "unknown", None


def probe_recorder_wiring(artifacts: Path) -> dict[str, Any]:
    """Audit every active registry path against the real CLI result channels.

    This is intentionally structural. Importing the command module and reading
    the registry proves dispatch wiring in milliseconds; calling each recorder
    would refit/re-read production inputs and belongs only to ``--full-replay``.
    """

    from nfl_ats import cli

    registry = load_challenger_registry(artifacts)
    entries = [
        entry
        for entry in registry["challengers"]
        if isinstance(entry, dict) and entry.get("status") == "ACTIVE_PROSPECTIVE"
    ]
    dispatch: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in entries:
        challenger_id = str(entry.get("challenger_id"))
        command = str(entry.get("weekly_recording_command", ""))
        path, _ = _recording_path(command)
        result_key: str | None = None
        entry_error = False
        if path == "publish":
            result_key = cli.PUBLISH_CHALLENGER_RESULT_KEYS.get(challenger_id)
        elif path == "refresh":
            result_key = REFRESH_RESULT_KEYS.get(challenger_id)
        elif path == "weekly-run":
            result_key = "prospective_record" if challenger_id == WEAK_STACK_CHALLENGER_ID else None
        if path == "unknown":
            errors.append(f"{challenger_id}: unrecognised recording command")
            entry_error = True
        elif path == "standalone":
            errors.append(f"{challenger_id}: standalone recorder is not CLI-wired")
            entry_error = True
        elif result_key is None:
            errors.append(f"{challenger_id}: no result key for {path} dispatch")
            entry_error = True
        dispatch.append(
            {
                "challenger_id": challenger_id,
                "path": path,
                "result_key": result_key,
                "command": command,
                "wired": not entry_error,
            }
        )
    publish_ids = {
        str(entry.get("challenger_id"))
        for entry in entries
        if "publish-predictions --record-decisions"
        in str(entry.get("weekly_recording_command", ""))
    }
    stale_publish_ids = sorted(set(cli.PUBLISH_CHALLENGER_RESULT_KEYS) - publish_ids)
    errors.extend(
        f"{challenger_id}: stale publish result-map entry" for challenger_id in stale_publish_ids
    )
    return {
        "ok": not errors,
        "active_registered": len(entries),
        "dispatch": dispatch,
        "errors": errors,
        "publish_result_keys": dict(cli.PUBLISH_CHALLENGER_RESULT_KEYS),
        "refresh_result_keys": dict(REFRESH_RESULT_KEYS),
    }


def _build_contract_fixture(root: Path) -> tuple[Path, str]:
    """Build one tiny card/registry fixture for the real append implementation."""

    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "prospective").mkdir()
    challenger_id = "lockday_contract_fixture"
    model = {
        "method": "market_residual",
        "target": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "feature_table": "data/processed/game_features_weak_stack.parquet",
    }
    registry = {
        "challengers": [
            {
                "challenger_id": challenger_id,
                "status": "ACTIVE_PROSPECTIVE",
                "weekly_recording_command": "nfl-ats prospective-record --challenger "
                f"{challenger_id}",
                "model": model,
            }
        ]
    }
    (artifacts / "prospective" / "challengers.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    card_dir = artifacts / "margin_predictions" / "2026-week-01-contract"
    card_dir.mkdir(parents=True)
    metadata = {
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "provenance": {"feature_table": {"path": "game_features_weak_stack.parquet"}},
    }
    (card_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "game_id": "contract-game-1",
                "season": 2026,
                "week": 1,
                "kickoff": "2026-09-10T23:00:00+00:00",
                "away_team": "AWY",
                "home_team": "HOM",
                "spread_line": -2.5,
                "home_cover_probability": 0.53,
                "bet_side": "HOME",
                "edge": 0.03,
            }
        ]
    ).to_csv(card_dir / "recommendations.csv", index=False)
    # Assert the fixture really matches what the recorder will fingerprint.
    if config_fingerprint(model) != config_fingerprint(artifact_model_config(metadata)):
        raise AssertionError("contract fixture configuration fingerprint does not match")
    return card_dir, challenger_id


def run_fast_contract(artifacts: Path, *, season: int, week: int) -> dict[str, Any]:
    """Run the seconds-scale, non-production lock-day readiness rehearsal."""

    started = time.perf_counter()
    report: dict[str, Any] = {"mode": "contract", "season": season, "week": week}
    report["live_ledgers_before"] = snapshot_live_ledgers(artifacts)
    report["recorder_wiring"] = probe_recorder_wiring(artifacts)

    lock_time = pd.Timestamp("2026-09-08T16:00:00Z")
    kickoffs = pd.Series(pd.to_datetime(["2026-09-10T23:00:00Z"], utc=True))
    guard: dict[str, Any] = {"window_days": RECORDING_LOCK_WINDOW.days}
    try:
        refuse_if_outside_recording_lock_window(kickoffs, lock_time, ledger="contract")
        guard["inside_window_allowed"] = True
    except ValueError as error:
        guard["inside_window_allowed"] = False
        guard["inside_window_error"] = str(error)
    try:
        refuse_if_outside_recording_lock_window(
            kickoffs, pd.Timestamp("2026-08-20T16:00:00Z"), ledger="contract"
        )
    except ValueError as error:
        guard["outside_window_refused"] = True
        guard["outside_window_error"] = str(error)
    else:
        guard["outside_window_refused"] = False
    report["recording_guard"] = guard

    with tempfile.TemporaryDirectory(prefix="lockday-contract-") as temporary:
        fixture_root = Path(temporary)
        card_dir, challenger_id = _build_contract_fixture(fixture_root)
        first = record_challenger_decisions(
            fixture_root / "artifacts", challenger_id, card_dir, now=lock_time.to_pydatetime()
        )
        ledger = challenger_ledger_path(fixture_root / "artifacts")
        after_first = _file_sha256(ledger)
        second = record_challenger_decisions(
            fixture_root / "artifacts", challenger_id, card_dir, now=lock_time.to_pydatetime()
        )
        after_second = _file_sha256(ledger)
        fixture_rows = load_challenger_decisions(fixture_root / "artifacts")
        report["ledger_contract"] = {
            "first_append": first,
            "second_append": second,
            "rows": len(fixture_rows),
            "append_idempotent": (
                first["recorded"] == 1
                and second["recorded"] == 0
                and second["already_recorded"] == 1
                and after_first == after_second
                and len(fixture_rows) == 1
            ),
        }
        report["fixture_coverage"] = lockday_verify.verify(
            fixture_root / "artifacts", season=season, week=week
        )

    report["live_lockday_verify"] = lockday_verify.verify(artifacts, season=season, week=week)
    report["live_ledgers_after"] = snapshot_live_ledgers(artifacts)
    report["live_ledgers_unchanged"] = report["live_ledgers_before"] == report["live_ledgers_after"]
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["ok"] = all(
        (
            report["recorder_wiring"]["ok"],
            guard["inside_window_allowed"],
            guard["outside_window_refused"],
            report["ledger_contract"]["append_idempotent"],
            report["live_ledgers_unchanged"],
            not report["live_lockday_verify"].get("pending_wiring"),
        )
    )
    return report


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
        "--full-replay",
        action="store_true",
        help="run the legacy isolated full recorder replay (slow; default is contract mode)",
    )
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

    if not args.full_replay:
        from lockday_contract import REFRESH_RESULT_KEYS as contract_refresh_keys
        from lockday_contract import main as contract_main

        # Use the rehearsal's complete refresh dispatch contract in the
        # static audit too; the CLI result assignment is still checked.
        contract_refresh_keys.update(REFRESH_RESULT_KEYS)
        return contract_main([])

    _load_full_replay_dependencies()

    lock_instant = datetime.fromisoformat(args.lock_instant)
    refresh_instant = datetime.fromisoformat(args.refresh_instant)

    report: dict[str, Any] = {
        "season": args.season,
        "week": args.week,
        "lock_instant": lock_instant.isoformat(),
        "refresh_instant": refresh_instant.isoformat(),
        "rehearsed_at_utc": datetime.now(UTC).isoformat(),
    }
    # Stage 0 FIRST: if the console script cannot even dispatch, every
    # downstream "recorded" below is measuring a path production will never
    # reach on lock day.
    print("stage 0: console-script command surface", file=sys.stderr)
    report["command_surface"] = probe_command_surface(REPO_ROOT)
    if not report["command_surface"]["ok"]:
        print("  !! command surface FAILED -- see report", file=sys.stderr)

    report["isolated_root"] = build_isolated_root(args.artifacts, args.rehearsal_root)
    print(f"isolated root built: {args.rehearsal_root}", file=sys.stderr)

    # ENG-01 (docs/lockday_package.md): the isolated root's ledger state BEFORE
    # any recorder runs, so the rehearsal can also rehearse the decision
    # package. Read-only, and only against the isolated copy -- the real
    # artifacts tree is never touched here.
    from nfl_ats.lockday_package import (
        capture_ledger_state,
        package_directory,
        write_decision_package,
    )
    from nfl_ats.provenance import write_stamped_artifact

    rehearsal_ledgers_before = capture_ledger_state(args.rehearsal_root)

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

    # ENG-01: rehearse the decision package too, tagged ``rehearsal: true``.
    # write_decision_package never raises, so this cannot break the rehearsal.
    report["decision_package"] = write_decision_package(
        season=args.season,
        week=args.week,
        artifacts_root=args.rehearsal_root,
        data_root=data_root,
        repo_root=REPO_ROOT,
        run_summary=report,
        ledger_state_before=rehearsal_ledgers_before,
        rehearsal=True,
        command="scripts/lockday_rehearsal.py --full-replay",
        # A DIFFERENT directory name from the real artifacts/lockday_packages/,
        # so a rehearsal package can never be mistaken for a real lock's even if
        # somebody points --rehearsal-root at an unusual place. Beside the
        # isolated root rather than inside it, because the next rehearsal
        # rmtree's that tree and a read-only manifest would make removal fail.
        destination=package_directory(
            args.rehearsal_root.parent / "lockday_packages_rehearsal",
            args.season,
            args.week,
        ),
    )

    destination = args.report or (args.rehearsal_root.parent / "rehearsal_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(report, destination)  # ENG-38

    print(lockday_verify.render(coverage))
    print(f"full report: {destination}", file=sys.stderr)
    surface_ok = bool(report["command_surface"]["ok"])
    if not surface_ok:
        print(
            "\n  !! the lock-day COMMAND SURFACE is broken -- the recorders above "
            "were driven directly and do not prove the real command works"
        )
    return 0 if (not coverage["missing"] and surface_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

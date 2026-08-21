"""RWB-09 experiment-provenance registry: enforcement and backfill tests.

Two concerns live here, distinct from ``tests/test_provenance.py``'s
round-trip/schema tests on the ``write_experiment_artifact()`` helper itself:

1. **Enforcement** -- a static, grep-style check that every ``scripts/*.py``
   file writing JSON into ``artifacts/`` goes through the helper, so the
   "automatic, not a discipline" property survives past this session. Ships
   with an explicit burn-down allowlist of every script that predates the
   helper (RWB-09's retrofit scope covered ``cli.py``'s 24 call sites this
   pass, not the standalone research scripts under ``scripts/`` also found
   to need it) so the suite is green today, while any newly-added unstamped
   script fails immediately -- this caught one mid-session
   (``ridge_alpha_promotion_eval.py``, added by a concurrent agent).
2. **Backfill** -- the one-time mechanical lift from existing
   ``artifacts/**/{metadata,run}.json`` files into ``registry/experiments/``
   (``scripts/backfill_experiment_registry.py``) produces rows that actually
   parse, and correctly refuses to invent a row when nothing in an artifact
   carries recoverable provenance.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from nfl_ats.provenance import ExperimentRecordError, load_experiment_record

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

# BURN-DOWN LIST, generated 2026-08-18 by:
#   grep -lE 'json\.dumps?\(|atomic_json\(' scripts/*.py  (writes JSON somewhere)
#     intersected with
#   grep -l 'artifacts' scripts/*.py                       (mentions artifacts/)
# Every name below is a script that writes JSON into an `artifacts/` path
# without importing `write_experiment_artifact` -- retrofitting these
# standalone research scripts was explicitly out of scope for the pass that
# added the helper (RWB-09 landed it in `cli.py`'s 24 call sites only).
# Remove an entry as its script is migrated. NEVER add one: a script added
# after this list was drawn that writes artifacts/ JSON without the helper
# should fail `test_every_script_writing_artifacts_json_uses_the_provenance_helper`,
# not be quietly grandfathered in.
_ALLOWLISTED_UNSTAMPED_SCRIPTS = frozenset(
    {
        "anytime_validate.py",
        "anytime_weekly_monitor.py",
        "audit_terminal_verdicts.py",
        "availability_ablation.py",
        "availability_mechanism_screen.py",
        "availability_overlap_audit.py",
        "best_pick_nomination_v3_audit.py",  # added 2026-08-19; measure-only
        # audit/head-to-head, same pattern as best_pick_opener_ranker_eval.py
        # and best_pick_ranker_tiebreak_audit.py below (no rotation window,
        # no confirmation look -- see the script's own docstring).
        "best_pick_opener_ranker_eval.py",  # added 2026-08-18 measurement wave
        "best_pick_ranker_tiebreak_audit.py",
        "best_pick_tiebreak_cfb_screen.py",
        "build_metagame_series.py",  # added 2026-08-19; descriptive per-season
        # series builder for docs/era_events.md, not an experiment -- no
        # hypothesis, no verdict, no closing ground, nothing recorded to
        # registry/weak_signals.json. Its manifest.json is a plain build
        # manifest (source snapshot id, column coverage), the same kind
        # nfl_ats.pbp/nfl_ats.features write via atomic_json for every other
        # feature-table build; wiring in write_experiment_artifact here would
        # misrepresent it as an adjudicated screen.
        "calibration_by_regime_cfb_screen.py",
        "cover_odds.py",  # added 2026-08-20; read-only query tool -- reads the
        # synchronized forecast artifacts and prints cover odds for an
        # arbitrary spread to stdout (`--json` prints, never writes files).
        # It runs no experiment and records nothing; the heuristic matches it
        # only because it references artifacts/ paths and json.dumps.
        "calibration_distortion_screen.py",
        "cfb_bias_battery_screen.py",  # added 2026-08-18 measurement wave
        "cfb_james_stein_unit_screen.py",
        "cfb_opponent_adjustment_screen.py",
        "cfb_role_continuity_remeasurement.py",
        "cfb_value_weighted_continuity_screen.py",
        "decision_apply.py",
        "ecdf_smoothing.py",
        "estvar_f_lever_confirmation.py",
        "estvar_planted_effects.py",
        "estvar_real_cfb_audit.py",
        "estvar_refit_intervals.py",
        "groupwise_ridge_screen.py",
        "hc_year_one_fade.py",
        "era_magnitude_profile.py",  # added 2026-08-19; measure-only era
        # diagnostic, recorded via its own CLI-recorder companion script.
        "novig_diagnostics_screen.py",
        "overlay_stack_backtest.py",  # added 2026-08-19; measure-only combined-
        # overlay diagnostic, recorded via its own CLI-recorder companion script.
        "overlay_subset_composition.py",  # added 2026-08-21; measure-only subset
        # composition diagnostic, recorded via its own CLI-recorder companion
        # commands (four predeclared weak-signal identities).
        "overlay_selection_holdout.py",  # added 2026-08-21; measure-only split-
        # half de-biasing of the composition selection, recorded via its own
        # CLI-recorder companion commands (two predeclared holdout identities).
        "proxy_opener_replication.py",  # added 2026-08-19; measure-only replication
        # at the SBR proxy-opener grade, recorded via its CLI-recorder companion.
        "surface_profile_opener_eval.py",  # added 2026-08-19; measure-only opener
        # head-to-head, recorded via its CLI-recorder companion script.
        "nfl_bias_battery_screen.py",  # added 2026-08-18 measurement wave
        "odds_microstructure_battery.py",  # added 2026-08-18 measurement wave
        "offseason_retention_cfb_permetric_screen.py",
        "offseason_retention_cfb_screen.py",
        "offseason_retention_nfl_free_seasons.py",
        "penalty_discipline_interval.py",
        "pool_levers.py",
        "purged_control_replication.py",
        "purged_validate.py",
        "qb_dependence_cfb_screen.py",
        "recency_weighted_training_screen.py",
        "residual_location_reaudit.py",
        "ridge_alpha_promotion_eval.py",
        "residual_location_screen.py",
        "ridge_alpha_screen.py",
        "scaling_cross_league_transfer.py",
        "scaling_learning_curve.py",
        "sensitivity_audit.py",
        "surface_familiarity_screen.py",  # added 2026-08-19; writes only to
        # <scratchpad>/agent_surface/results.json (never artifacts/ or
        # registry/ -- see its own docstring), the static check's documented
        # false-positive case: it mentions an artifacts/... path only as a
        # citation of nfl_weather_battery_screen.py's prior output.
        "surgical_cfb_recipe_validation.py",
        "variance_planted_effects.py",
        "weak_stack_v2_eval.py",  # added 2026-08-18 measurement wave
        "weak_stack_v3_opener_eval.py",  # added 2026-08-20; measure-only opener
        # head-to-head (docs/weak_stack_v3.md), same pattern as
        # surface_profile_opener_eval.py above -- recorded via a separate
        # `nfl-ats weak-signals record` call, not this script.
        "public_betting_battery_screen.py",  # added 2026-08-20; measure-only
        # mined battery (docs/public_betting_battery_predeclaration.md),
        # same pattern as odds_microstructure_battery.py above -- proposes
        # `nfl-ats weak-signals record` commands (printed + written to its
        # own metadata.json), never executes them itself.
        "xlg06_rookie_prior_cfb_screen.py",
    }
)

_JSON_WRITE_MARKERS = ("json.dump(", "json.dumps(", "atomic_json(")


def _writes_artifacts_json_without_helper(path: Path) -> bool:
    """A script counts as an "offender" if it writes JSON somewhere, mentions
    an `artifacts/` path at all, and never calls the provenance helper.

    Deliberately a light, static, textual check (matching this project's own
    "static grep-based test, not a runtime hook" design choice) -- it can
    over-flag a script that mentions "artifacts" in prose and writes JSON
    somewhere unrelated (harmless: it just needs an allowlist entry), but it
    is meant to catch the real case: a NEW script copying the established
    `OUTPUT_DIR = Path("artifacts/...")` + `json.dump(...)` convention
    without the helper this registry depends on.
    """

    text = path.read_text(encoding="utf-8")
    if "write_experiment_artifact" in text:
        return False
    if "artifacts" not in text:
        return False
    return any(marker in text for marker in _JSON_WRITE_MARKERS)


def test_every_script_writing_artifacts_json_uses_the_provenance_helper() -> None:
    offenders = {
        path.name
        for path in sorted(SCRIPTS_ROOT.glob("*.py"))
        if _writes_artifacts_json_without_helper(path)
    }
    unexpected = offenders - _ALLOWLISTED_UNSTAMPED_SCRIPTS
    assert not unexpected, (
        "New/changed script(s) write JSON into artifacts/ without calling "
        f"write_experiment_artifact(): {sorted(unexpected)}. Either wire in the "
        "helper (see src/nfl_ats/provenance.py, write_experiment_artifact) or, if "
        "it genuinely cannot use it yet, add the filename to "
        "_ALLOWLISTED_UNSTAMPED_SCRIPTS above with a reason in the commit message."
    )


def test_unstamped_scripts_allowlist_has_no_stale_entries() -> None:
    """A deleted/renamed script should be pruned from the burn-down list, not
    linger forever -- keeps the list an honest measure of remaining work."""

    existing = {path.name for path in SCRIPTS_ROOT.glob("*.py")}
    stale = _ALLOWLISTED_UNSTAMPED_SCRIPTS - existing
    assert not stale, f"Allowlist references scripts that no longer exist: {sorted(stale)}"


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def _load_backfill_module() -> ModuleType:
    path = SCRIPTS_ROOT / "backfill_experiment_registry.py"
    spec = importlib.util.spec_from_file_location("backfill_experiment_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(directory: Path, filename: str, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_backfill_produces_registry_rows_that_parse(tmp_path: Path) -> None:
    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"

    # A cli.py-shaped metadata.json: a "provenance" block to lift.
    _make_run(
        artifacts_root / "demo_command" / "20260101T000000Z",
        "metadata.json",
        {
            "command": "demo-command",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "games": 42,
            "provenance": {
                "configuration": {"command": "demo-command"},
                "configuration_sha256": "abc123",
                "feature_table": {"sha256": "feat456"},
                "code": {"revision": "deadbeef", "dirty": False},
                "uv_lock_sha256": "lock789",
            },
        },
    )
    # A bare run.json: the provenance dict IS the whole file.
    _make_run(
        artifacts_root / "other_command" / "20260102T000000Z",
        "run.json",
        {
            "configuration": {"command": "other-command"},
            "configuration_sha256": "xyz999",
            "feature_table": {"sha256": "featabc"},
            "code": {"revision": "cafef00d", "dirty": True},
            "uv_lock_sha256": "lockdef",
        },
    )
    # A dual-provenance file (the availability-ablation shape): both should
    # be noted, one used, and the row still has to parse.
    _make_run(
        artifacts_root / "dual_command" / "20260103T000000Z",
        "metadata.json",
        {
            "command": "dual-command",
            "created_at_utc": "2026-01-03T00:00:00+00:00",
            "baseline_provenance": {
                "configuration": {},
                "configuration_sha256": "base1",
                "feature_table": {"sha256": "f1"},
                "code": {"revision": "rev1", "dirty": False},
                "uv_lock_sha256": None,
            },
            "learned_provenance": {
                "configuration": {},
                "configuration_sha256": "learned1",
                "feature_table": {"sha256": "f2"},
                "code": {"revision": "rev1", "dirty": False},
                "uv_lock_sha256": None,
            },
        },
    )

    summary = backfill.run_backfill(artifacts_root, registry_root)
    assert summary["run_directories"] == 3
    assert summary["backfilled"] == 3
    assert summary["unbackfillable"] == 0

    registry_files = sorted((registry_root / "experiments").glob("*/*.json"))
    assert len(registry_files) == 3
    for path in registry_files:
        record = load_experiment_record(path)
        assert record.provenance_backfilled is True
        assert record.backfill_note

    dual_row = load_experiment_record(
        registry_root / "experiments" / "dual-command" / "20260103T000000Z.json"
    )
    assert dual_row.config_hash == "base1"  # alphabetically-first key, deterministic
    assert "learned_provenance" in (dual_row.backfill_note or "")


def test_backfill_invents_nothing_for_a_run_without_recoverable_provenance(
    tmp_path: Path,
) -> None:
    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"

    # metadata.json with no provenance-shaped block anywhere.
    _make_run(
        artifacts_root / "bare_metadata" / "20260101T000000Z",
        "metadata.json",
        {"games": 10, "note": "no code revision recorded here"},
    )
    # A run directory with no metadata.json/run.json at all.
    (artifacts_root / "totally_bare" / "20260102T000000Z").mkdir(parents=True)

    summary = backfill.run_backfill(artifacts_root, registry_root)
    assert summary["backfilled"] == 0
    assert summary["unbackfillable"] == 2

    doc = (registry_root / "experiments" / "UNBACKFILLABLE.md").read_text(encoding="utf-8")
    assert "bare_metadata" in doc
    assert "totally_bare" in doc
    assert "no metadata.json or run.json" in doc
    assert "no artifact_provenance()-shaped block" in doc

    # No registry rows were fabricated for either directory.
    rows = list((registry_root / "experiments").glob("*/*.json"))
    assert rows == []


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """Re-running the backfill (e.g. after a new week's artifacts appear)
    must not raise or corrupt already-written rows."""

    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"
    _make_run(
        artifacts_root / "demo_command" / "20260101T000000Z",
        "metadata.json",
        {
            "command": "demo-command",
            "provenance": {
                "configuration": {},
                "configuration_sha256": "abc123",
                "feature_table": {"sha256": "feat456"},
                "code": {"revision": "deadbeef", "dirty": False},
                "uv_lock_sha256": None,
            },
        },
    )
    first = backfill.run_backfill(artifacts_root, registry_root)
    second = backfill.run_backfill(artifacts_root, registry_root)
    assert first == second


# ---------------------------------------------------------------------------
# The real deliverable: every row already committed under
# registry/experiments/ must itself parse and be honestly labeled.
# ---------------------------------------------------------------------------


def test_committed_registry_experiment_rows_all_parse() -> None:
    experiments_root = REPO_ROOT / "registry" / "experiments"
    if not experiments_root.is_dir():
        return  # nothing backfilled yet in this checkout
    rows = sorted(experiments_root.glob("*/*.json"))
    assert rows, "expected at least one backfilled experiment row"
    for path in rows:
        try:
            record = load_experiment_record(path)
        except ExperimentRecordError as error:  # pragma: no cover - failure path
            raise AssertionError(f"{path} does not parse as an ExperimentRecord: {error}") from None
        assert record.experiment_id
        assert record.command

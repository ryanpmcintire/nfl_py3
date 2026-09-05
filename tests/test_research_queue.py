"""Tests for ENG-20's research-queue evidence ledger.

BINDING (AGENTS.md, pasted through verbatim per the closing-ground taxonomy
this project requires in every subagent/test that adjudicates an experiment):
an interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size can close a line; everything else is
`unresolved_below_power`. Verdicts flow through `rotation.record_look` /
`weak_signals.record_signal` (the "record" commands), never through prose.

These tests build small synthetic roadmap/rotation/weak-signal fixtures --
never the real ``ROADMAP.md`` or tracked registries -- so every join, the
circular-run guard, every ``next_admissible_action`` outcome, and the
``--check`` staleness contract are pinned independently of the live data.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.research_queue as research_queue_cli
from nfl_ats import research_queue, rotation, weak_signals
from scripts.capture_scheduler import Job

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROADMAP_TEXT = """\
## Phase 1 — other work

| ID | Status | Item | Definition of done |
|---|---|---|---|
| OTH-01 | 🚧 | In-progress row naming a family | Uses `alpha_family` in prose. |
| OTH-02 | 🔬 | Research row naming no family | Matches no registry entry. |
| OTH-03 | ✅ | Done row naming a family | Excluded: `alpha_family`, status done. |

## Phase 12 — open lead queue

| ID | Status | Item | Definition of done |
|---|---|---|---|
| LEAD-01 | 🔬 | Alpha follow | References `alpha_family`, signal `widget_screen_v1`. |
| LEAD-02 | 🔬 | Beta follow | References `beta_family` (close, both blocks spent). |
| LEAD-03 | 🔬 | Beta on production | References `beta_family_on_production` (no control). |
| LEAD-04 | 🔬 | Gamma on production | References `gamma_family_on_production` (control sized). |
| LEAD-05 | 🔬 | Delta pending | References `delta_family` (assigned, unrecorded). |
| LEAD-06 | ✅ | Epsilon closed | References `epsilon_family` (closed_negative already). |
| LEAD-07 | 🔬 | No prior art | Friday injury designation, practice status; no family yet. |
| LEAD-08 | 🔬 | Terminal weak signal only | References signal `gizmo_bench`, no family. |
| LEAD-09 | 🔬 | Fresh opener idea | Names no family/signal; a fresh opener proposal. |
"""


def _rotation_window(
    seasons: tuple[int, int],
    *,
    state: str = "spent",
    verdict: str | None = "unresolved",
    closing_ground: str | None = None,
    notes: str = "",
    probability_positive: float | None = 0.6,
) -> dict:
    window: dict = {
        "seasons": list(seasons),
        "state": state,
        "window_kind": "contiguous",
        "assigned_at": "2026-07-01",
        "notes": notes,
    }
    if state == "spent":
        window.update(
            {
                "spent_at": "2026-07-15",
                "artifact": "docs/fixture_screen.md",
                "verdict": verdict,
                "closing_ground": closing_ground,
                "probability_positive": probability_positive,
                "effect": 1.1,
                "effect_units": "accuracy_points",
                "interval": [-0.5, 2.7],
            }
        )
    return window


ROTATION_PAYLOAD = {
    "version": 1,
    "notes": [],
    "families": {
        # One spent window, one block still eligible in the close pool
        # ([2011, 2013], [2014, 2016] are the only two non-mined close-grade
        # blocks given MIN_ELIGIBLE_START_SEASON=2011 and MINED_SEASONS
        # (2018, 2025)) -> run_unspent_window.
        "alpha_family": {
            "declared_at": "2026-07-01",
            "description": "Alpha family screen only, not yet tested on production.",
            "grade": "close",
            "status": "open",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [_rotation_window((2011, 2013))],
        },
        # Both non-mined close blocks spent, description has no "production"
        # word -> test_on_top_of_production.
        "beta_family": {
            "declared_at": "2026-07-01",
            "description": "Beta family, screened but never measured on top of anything.",
            "grade": "close",
            "status": "open",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [
                _rotation_window((2011, 2013)),
                _rotation_window((2014, 2016)),
            ],
        },
        # Both blocks spent, name says on_production, latest verdict
        # unresolved, no positive-control note -> run_positive_control.
        "beta_family_on_production": {
            "declared_at": "2026-07-01",
            "description": "Beta measured on top of the production weak_stack chain.",
            "grade": "close",
            "status": "open",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [
                _rotation_window((2011, 2013)),
                _rotation_window((2014, 2016)),
            ],
        },
        # Same shape, but a positive control was already sized ->
        # run_reused_window_with_discount.
        "gamma_family_on_production": {
            "declared_at": "2026-07-01",
            "description": "Gamma measured on top of the production weak_stack chain.",
            "grade": "close",
            "status": "open",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [
                _rotation_window((2011, 2013)),
                _rotation_window(
                    (2014, 2016),
                    notes="a positive control sized to the candidate effect scored +49.0 pts",
                ),
            ],
        },
        # An assigned, not-yet-recorded window -> record_pending_look.
        "delta_family": {
            "declared_at": "2026-07-01",
            "description": "Delta family, look drawn but not yet recorded.",
            "grade": "close",
            "status": "open",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [_rotation_window((2011, 2013), state="assigned", verdict=None)],
        },
        # Admissible closed_negative -> closed.
        "epsilon_family": {
            "declared_at": "2026-07-01",
            "description": "Epsilon family, resolved wrong sign.",
            "grade": "close",
            "status": "closed_negative",
            "inherits": [],
            "acknowledges_mined_2018_2025": False,
            "windows": [
                _rotation_window(
                    (2011, 2013),
                    verdict="closed_negative",
                    closing_ground="wrong_sign_resolved",
                    probability_positive=0.01,
                )
            ],
        },
    },
}

WEAK_SIGNAL_PAYLOAD = {
    "version": 1,
    "notes": [],
    "signals": {
        "widget_screen_v1": {
            "recorded_at": "2026-07-20",
            "description": "Alpha family's weak-signal companion read.",
            "source": "docs/fixture_screen.md",
            "effect": 0.8,
            "effect_units": "accuracy_points",
            "classification": "unresolved_below_power",
            "league": "nfl",
            "seasons": [2011, 2013],
            "probability_positive": 0.58,
            "family": "alpha_family",
            "notes": "",
        },
        "gizmo_bench": {
            "recorded_at": "2026-08-20",
            "description": "Terminal negative control, resolved wrong sign.",
            "source": "docs/fixture_screen.md",
            "effect": -4.0,
            "effect_units": "accuracy_points",
            "classification": "refuted_mechanism",
            "classification_evidence": "whole interval below zero, measured this session",
            "closing_ground": "wrong_sign_resolved",
            "interval": [-6.0, -2.0],
            "league": "nfl",
            "seasons": [2011, 2013],
            "probability_positive": 0.01,
        },
    },
}

FIXTURE_JOBS = (
    Job("odds_tue_open", "tue", "09:00", 180, [], True, "opener capture"),
    Job("injuries_wed", "wed", "17:30", 240, [], False, "paused injury capture"),
    Job("weekly_lock", "tue", "09:15", 120, [], True, "internal pipeline, not a source"),
    Job("refresh_thu", "thu", "15:00", 240, [], True, "internal pipeline, not a source"),
)


@pytest.fixture
def rotation_registry() -> rotation.Registry:
    return rotation.registry_from_payload(ROTATION_PAYLOAD)


@pytest.fixture
def weak_signal_registry() -> weak_signals.Registry:
    return weak_signals.registry_from_payload(WEAK_SIGNAL_PAYLOAD)


@pytest.fixture
def rows(rotation_registry: rotation.Registry, weak_signal_registry: weak_signals.Registry):
    return research_queue.build_queue(
        ROADMAP_TEXT,
        rotation_registry,
        weak_signal_registry,
        experiment_spec_names=frozenset({"alpha_family"}),
        jobs=FIXTURE_JOBS,
    )


def _row(rows_list, item_id: str):
    matches = [row for row in rows_list if row.item_id == item_id]
    assert len(matches) == 1, f"expected exactly one row for {item_id!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


def test_selects_phase_12_rows_regardless_of_status(rows) -> None:
    ids = {row.item_id for row in rows}
    assert "LEAD-06" in ids  # ✅ status, still a Phase 12 lead


def test_selects_non_phase_12_rows_only_when_they_name_a_declared_family(rows) -> None:
    ids = {row.item_id for row in rows}
    assert "OTH-01" in ids  # 🚧, names alpha_family
    assert "OTH-02" not in ids  # 🔬, names no family
    assert "OTH-03" not in ids  # ✅ status outside Phase 12: excluded even though it names one


def test_row_count_matches_selection_rule(rows) -> None:
    # OTH-01 + nine Phase 12 leads.
    assert len(rows) == 10


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_rotation_family_and_weak_signal_joins(rows) -> None:
    alpha = _row(rows, "LEAD-01")
    assert alpha.rotation_family == "alpha_family"
    assert alpha.weak_signal_ids == ("widget_screen_v1",)
    assert alpha.predeclaration_spec == "registry/experiment_specs/alpha_family.json"


def test_rows_without_a_matched_family_report_none(rows) -> None:
    lead09 = _row(rows, "LEAD-09")
    assert lead09.rotation_family == "none"
    assert lead09.weak_signal_ids == ()
    assert lead09.predeclaration_spec == "none"


def test_windows_used_and_unspent_reflect_the_rotation_registry(rows) -> None:
    alpha = _row(rows, "LEAD-01")
    assert alpha.windows_used == ("[2011, 2013]:unresolved",)
    # rotation.eligible_blocks scans every start season, not just a fixed
    # tiling, so both [2014, 2016] and [2015, 2017] are independently
    # eligible (neither touches alpha_family's own spent [2011, 2013] nor
    # the mined 2018-2025 seasons).
    assert alpha.windows_unspent == 2

    beta = _row(rows, "LEAD-02")
    assert beta.windows_unspent == 0  # both non-mined close blocks spent


def test_last_attempt_prefers_the_later_of_rotation_and_weak_signal_dates(rows) -> None:
    alpha = _row(rows, "LEAD-01")
    # rotation window spent_at=2026-07-15, weak signal recorded_at=2026-07-20: the
    # signal is later and must win.
    assert alpha.last_attempt == "2026-07-20"
    assert alpha.last_attempt_source == "weak_signal"
    assert alpha.last_attempt_classification == "unresolved_below_power"


def test_never_attempted_row_reports_never_not_a_guess(rows) -> None:
    lead09 = _row(rows, "LEAD-09")
    assert lead09.last_attempt == "never"
    assert lead09.last_attempt_source == "none"
    assert lead09.last_attempt_classification == "not_yet_run"


def test_required_source_keyword_match_and_captured_today(rows) -> None:
    lead07 = _row(rows, "LEAD-07")
    assert lead07.required_source == "injuries"
    assert lead07.source_captured_today == "no"  # fixture injuries job is disabled

    alpha = _row(rows, "LEAD-01")
    assert alpha.required_source == "unknown"
    assert alpha.source_captured_today == "unknown"


# ---------------------------------------------------------------------------
# next_admissible_action: every value in the fixed vocabulary, never "wait".
# ---------------------------------------------------------------------------


def test_next_admissible_action_run_unspent_window(rows) -> None:
    row = _row(rows, "LEAD-01")
    assert row.next_admissible_action == research_queue.ACTION_RUN_UNSPENT_WINDOW
    assert "2014" in row.next_admissible_action_detail


def test_next_admissible_action_test_on_top_of_production(rows) -> None:
    row = _row(rows, "LEAD-02")
    assert row.next_admissible_action == research_queue.ACTION_TEST_ON_TOP_OF_PRODUCTION


def test_next_admissible_action_run_positive_control(rows) -> None:
    row = _row(rows, "LEAD-03")
    assert row.next_admissible_action == research_queue.ACTION_RUN_POSITIVE_CONTROL


def test_next_admissible_action_run_reused_window_with_discount(rows) -> None:
    row = _row(rows, "LEAD-04")
    assert row.next_admissible_action == research_queue.ACTION_RUN_REUSED_WINDOW_WITH_DISCOUNT


def test_next_admissible_action_record_pending_look(rows) -> None:
    row = _row(rows, "LEAD-05")
    assert row.next_admissible_action == research_queue.ACTION_RECORD_PENDING_LOOK


def test_next_admissible_action_closed_requires_admissible_ground(rows) -> None:
    row = _row(rows, "LEAD-06")
    assert row.next_admissible_action == research_queue.ACTION_CLOSED
    assert "wrong_sign_resolved" in row.next_admissible_action_detail


def test_next_admissible_action_closed_from_a_terminal_weak_signal(rows) -> None:
    row = _row(rows, "LEAD-08")
    assert row.next_admissible_action == research_queue.ACTION_CLOSED
    assert "gizmo_bench" in row.next_admissible_action_detail


def test_next_admissible_action_run_unspent_window_for_an_undeclared_family(rows) -> None:
    row = _row(rows, "LEAD-09")
    assert row.next_admissible_action == research_queue.ACTION_RUN_UNSPENT_WINDOW


def test_every_row_action_is_in_the_fixed_vocabulary(rows) -> None:
    for row in rows:
        assert row.next_admissible_action in research_queue.NEXT_ACTIONS


def test_no_row_ever_needs_a_games_needed_field(rows) -> None:
    for row in rows:
        assert not hasattr(row, "games_needed")
        assert "games needed" not in row.next_admissible_action_detail.lower()


# ---------------------------------------------------------------------------
# Circular-run guard
# ---------------------------------------------------------------------------


def test_is_circular_true_when_family_reuses_its_own_seasons_without_disclosure() -> None:
    family = rotation.Family(
        name="solo_family",
        declared_at="2026-07-01",
        description="fixture",
        grade="close",
        status="open",
        windows=(rotation.Window(seasons=(2011, 2012), state="spent", assigned_at="2026-07-01"),),
    )
    candidate = rotation.Window(seasons=(2012, 2013), state="assigned", assigned_at="2026-08-01")
    assert research_queue.is_circular(family, candidate) is True


def test_is_circular_false_when_disclosed() -> None:
    family = rotation.Family(
        name="solo_family",
        declared_at="2026-07-01",
        description="fixture",
        grade="close",
        status="open",
        windows=(rotation.Window(seasons=(2011, 2012), state="spent", assigned_at="2026-07-01"),),
    )
    candidate = rotation.Window(
        seasons=(2012, 2013),
        state="assigned",
        assigned_at="2026-08-01",
        notes="disclosed re-read of 2012",
    )
    assert research_queue.is_circular(family, candidate) is False


def test_is_circular_false_with_no_overlap() -> None:
    family = rotation.Family(
        name="solo_family",
        declared_at="2026-07-01",
        description="fixture",
        grade="close",
        status="open",
        windows=(rotation.Window(seasons=(2011, 2012), state="spent", assigned_at="2026-07-01"),),
    )
    candidate = rotation.Window(seasons=(2014, 2015), state="assigned", assigned_at="2026-08-01")
    assert research_queue.is_circular(family, candidate) is False


def test_cross_family_reuse_true_without_disclosure(rotation_registry: rotation.Registry) -> None:
    # beta_family and beta_family_on_production both spent [2011, 2013];
    # neither window's notes disclose the overlap in this fixture.
    beta = rotation_registry.families["beta_family"]
    window = beta.windows[0]
    assert research_queue.cross_family_reuse(rotation_registry, "beta_family", window) is True


def test_cross_family_reuse_false_when_disclosed(rotation_registry: rotation.Registry) -> None:
    beta = rotation_registry.families["beta_family"]
    disclosed_window = dataclasses.replace(
        beta.windows[0], notes="reuse of beta_family_on_production's block, disclosed"
    )
    assert (
        research_queue.cross_family_reuse(rotation_registry, "beta_family", disclosed_window)
        is False
    )


def test_reuse_flag_is_exposed_on_the_row(rows) -> None:
    beta = _row(rows, "LEAD-02")
    assert beta.reuse_flag is True  # shares its seasons with two other undisclosed families


# ---------------------------------------------------------------------------
# Capture-source mapping helpers
# ---------------------------------------------------------------------------


def test_capture_job_families_collapses_day_and_slot_suffixes() -> None:
    families = research_queue.capture_job_families(FIXTURE_JOBS)
    assert families["odds"] is True
    assert families["injuries"] is False
    assert "weekly_lock" not in families
    assert "refresh_thu" not in families
    assert "refresh" not in families


def test_guess_required_source_returns_none_when_unmappable() -> None:
    assert research_queue.guess_required_source("a generic sentence with no keywords") is None


def test_guess_required_source_matches_keywords() -> None:
    assert research_queue.guess_required_source("Friday DNP designation") == "injuries"
    assert (
        research_queue.guess_required_source("a public betting handle snapshot") == "public_betting"
    )


# ---------------------------------------------------------------------------
# Generated output never contains banned phrases; JSON/Markdown round-trip.
# ---------------------------------------------------------------------------


def test_generated_output_never_contains_banned_phrases(rows) -> None:
    payload = research_queue.queue_payload(rows)
    markdown = research_queue.queue_markdown(rows)
    haystacks = [json.dumps(payload).lower(), markdown.lower()]
    for haystack in haystacks:
        for phrase in research_queue.BANNED_PHRASES:
            assert phrase not in haystack, f"banned phrase {phrase!r} found in generated output"


def test_queue_payload_is_json_stable_across_two_builds(rows) -> None:
    first = research_queue.queue_payload(rows)
    second = research_queue.queue_payload(rows)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_markdown_header_says_generated(rows) -> None:
    markdown = research_queue.queue_markdown(rows)
    assert "Generated by `scripts/research_queue.py`" in markdown
    assert "Do not hand-edit" in markdown


# ---------------------------------------------------------------------------
# CLI --check contract
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> dict[str, Path]:
    roadmap_path = tmp_path / "ROADMAP.md"
    roadmap_path.write_text(ROADMAP_TEXT, encoding="utf-8")

    rotation_path = tmp_path / "rotation_registry.json"
    rotation_path.write_text(json.dumps(ROTATION_PAYLOAD), encoding="utf-8")

    weak_signals_path = tmp_path / "weak_signals.json"
    weak_signals_path.write_text(json.dumps(WEAK_SIGNAL_PAYLOAD), encoding="utf-8")

    specs_dir = tmp_path / "experiment_specs"
    specs_dir.mkdir()
    (specs_dir / "alpha_family.json").write_text("{}", encoding="utf-8")

    return {
        "roadmap": roadmap_path,
        "rotation_registry": rotation_path,
        "weak_signals": weak_signals_path,
        "experiment_specs": specs_dir,
        "output_json": tmp_path / "research_queue.json",
        "output_md": tmp_path / "research_queue.md",
    }


def _cli_argv(paths: dict[str, Path], *, check: bool) -> list[str]:
    argv = [
        "--roadmap",
        str(paths["roadmap"]),
        "--rotation-registry",
        str(paths["rotation_registry"]),
        "--weak-signals",
        str(paths["weak_signals"]),
        "--experiment-specs",
        str(paths["experiment_specs"]),
        "--output-json",
        str(paths["output_json"]),
        "--output-md",
        str(paths["output_md"]),
    ]
    if check:
        argv.append("--check")
    return argv


def test_cli_check_exits_nonzero_when_files_are_missing(tmp_path: Path) -> None:
    paths = _write_cli_fixture(tmp_path)
    exit_code = research_queue_cli.main(_cli_argv(paths, check=True))
    assert exit_code == 1
    assert not paths["output_json"].exists()
    assert not paths["output_md"].exists()


def test_cli_writes_files_then_check_reports_clean(tmp_path: Path) -> None:
    paths = _write_cli_fixture(tmp_path)

    write_exit = research_queue_cli.main(_cli_argv(paths, check=False))
    assert write_exit == 0
    assert paths["output_json"].is_file()
    assert paths["output_md"].is_file()

    check_exit = research_queue_cli.main(_cli_argv(paths, check=True))
    assert check_exit == 0


def test_cli_check_detects_a_hand_edit(tmp_path: Path) -> None:
    paths = _write_cli_fixture(tmp_path)
    research_queue_cli.main(_cli_argv(paths, check=False))

    paths["output_md"].write_text("hand-edited, no longer matches a fresh build", encoding="utf-8")
    check_exit = research_queue_cli.main(_cli_argv(paths, check=True))
    assert check_exit == 1


def test_cli_check_is_insensitive_to_rerun_wall_clock(tmp_path: Path) -> None:
    """Two consecutive writes from identical inputs must agree byte-for-byte.

    Guards the specific design choice in ``queue_payload`` (no wall-clock
    timestamp in the persisted JSON): otherwise ``--check`` would report
    stale on every single run even when nothing about the underlying roadmap
    or registries changed.
    """

    paths = _write_cli_fixture(tmp_path)
    research_queue_cli.main(_cli_argv(paths, check=False))
    first_json = paths["output_json"].read_text(encoding="utf-8")
    first_md = paths["output_md"].read_text(encoding="utf-8")

    research_queue_cli.main(_cli_argv(paths, check=False))
    second_json = paths["output_json"].read_text(encoding="utf-8")
    second_md = paths["output_md"].read_text(encoding="utf-8")

    assert first_json == second_json
    assert first_md == second_md

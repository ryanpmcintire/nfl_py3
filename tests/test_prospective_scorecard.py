"""ENG-06: prospective evidence scorecards.

Fixtures write directly to a ``tmp_path`` ``artifacts_root`` in the exact
on-disk shape ``nfl_ats.clv``/``nfl_ats.prospective_scoring``/
``nfl_ats.pick_refresh`` already write (full column contracts, minus the
columns those loaders backfill with legacy defaults), then read them back
through the SAME loaders :mod:`nfl_ats.prospective_scorecard` uses, so these
tests exercise the real read path rather than a parallel one.

All game outcomes below are hand-chosen so every accuracy delta and flip
count is exactly computable; only the bootstrap interval bounds and
``probability_positive`` are left to the (seeded, deterministic) resampler,
and those are only checked for basic sanity (finite, ``probability_positive``
in ``[0, 1]``), never for an exact value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats import rotation as rotation_module
from nfl_ats import weak_signals as weak_signals_module
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.prospective_scorecard import (
    ACTIVE_MODEL_ENTRANT_ID,
    NEXT_ADMISSIBLE_ACTIONS,
    NO_SPLIT_HALF_RELIABILITY,
    WRONG_SIGN_RESOLVED,
    _classification,
    _closing_ground_candidate,
    _next_admissible_action,
    _predeclared_sign,
    _referenced_signal_names,
    _split_half_reliability,
    build_season_scorecards,
    render_markdown,
    scorecards_to_frame,
)
from nfl_ats.weak_signals import CLASSIFICATIONS, POOLABLE_CLASSIFICATION

SEASON = 2099
CHALLENGER_ID = "test_overlay_challenger"

#: Forbidden phrasing (AGENTS.md, "An interval crossing zero is NOT grounds
#: for rejection"): a scorecard is a report, never a verdict, and must never
#: say a signal "failed" or was "rejected", must never write the binary
#: "contains zero", and must never compute or state a "needs more" games
#: figure.
_FORBIDDEN_SUBSTRINGS = ("failed", "rejected", "contains zero", "needs more")

WEEK_KICKOFFS = {
    1: pd.Timestamp("2099-09-08T17:00:00Z"),
    2: pd.Timestamp("2099-09-15T17:00:00Z"),
    3: pd.Timestamp("2099-09-22T17:00:00Z"),
}

#: (game_id, week, home_team, away_team, decision_home_spread, result,
#:  active_pick_side, challenger_pick_side)
#: result is home points minus away points; margin = result - line.
GAMES: tuple[tuple[str, int, str, str, float, float, str, str], ...] = (
    ("g1", 1, "HOME1", "AWAY1", -3.0, 13.0, "HOME", "HOME"),  # home covers; both correct
    (
        "g2",
        1,
        "HOME2",
        "AWAY2",
        2.5,
        -1.0,
        "HOME",
        "AWAY",
    ),  # away covers; active wrong, challenger right
    ("g3", 2, "HOME3", "AWAY3", -1.0, 5.0, "HOME", "HOME"),  # home covers; both correct
    ("g4", 2, "HOME4", "AWAY4", 6.0, -10.0, "AWAY", "AWAY"),  # away covers; both correct
    ("g5", 3, "HOME5", "AWAY5", 4.0, 4.0, "HOME", "HOME"),  # push on both sides
    (
        "g6",
        3,
        "HOME6",
        "AWAY6",
        -2.5,
        8.0,
        "AWAY",
        "HOME",
    ),  # home covers; active wrong, challenger right
)

#: home_cover_probability recorded on each week's own forecast card, used by
#: the calibration test. Chosen to roughly track the actual outcome.
PROBABILITIES = {"g1": 0.70, "g2": 0.40, "g3": 0.65, "g4": 0.30, "g5": 0.50, "g6": 0.55}

ACTIVE_FORECAST_ARTIFACT = {
    1: "margin_predictions/active_w1",
    2: "margin_predictions/active_w2",
    3: "margin_predictions/active_w3",
}
CHALLENGER_SOURCE_ARTIFACT = "challenger_card_all"


def _paper_decisions_frame() -> pd.DataFrame:
    rows = []
    for game_id, week, home, away, line, _result, active_side, _challenger_side in GAMES:
        kickoff = WEEK_KICKOFFS[week]
        rows.append(
            {
                "recorded_at_utc": kickoff - pd.Timedelta(days=3),
                "forecast_artifact": ACTIVE_FORECAST_ARTIFACT[week],
                "forecast_created_at_utc": kickoff - pd.Timedelta(days=4),
                "model_id": "test_active_model",
                "method": "market_residual",
                "game_id": game_id,
                "season": SEASON,
                "week": week,
                "kickoff": kickoff,
                "away_team": away,
                "home_team": home,
                "pick_side": active_side,
                "bet_side": "PASS",
                "decision_home_spread": line,
                "edge": 0.05,
            }
        )
    return pd.DataFrame(rows)


def _challenger_decisions_frame(game_ids: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for game_id, week, home, away, line, _result, _active_side, challenger_side in GAMES:
        if game_ids is not None and game_id not in game_ids:
            continue
        kickoff = WEEK_KICKOFFS[week]
        rows.append(
            {
                "recorded_at_utc": kickoff - pd.Timedelta(days=3),
                "challenger_id": CHALLENGER_ID,
                "config_fingerprint": "test_fingerprint",
                "source_artifact": CHALLENGER_SOURCE_ARTIFACT,
                "source_sha256": "0" * 64,
                "forecast_created_at_utc": kickoff - pd.Timedelta(days=4),
                "feature_profile": "test_profile",
                "feature_table_sha256": "0" * 64,
                "game_id": game_id,
                "season": SEASON,
                "week": week,
                "kickoff": kickoff,
                "away_team": away,
                "home_team": home,
                "pick_side": challenger_side,
                "bet_side": "PASS",
                "decision_home_spread": line,
                "edge": 0.05,
            }
        )
    return pd.DataFrame(rows)


def _pick_revisions_frame() -> pd.DataFrame:
    """Two revised games: g2 flips (an earlier no-op revision precedes it,
    so the ledger also exercises "latest revision wins"), g4 is reconsidered
    and kept.
    """

    def row(
        game_id: str, week: int, previous: str, new: str, recorded_offset_hours: int
    ) -> dict[str, Any]:
        home, away, line = next((g[2], g[3], g[4]) for g in GAMES if g[0] == game_id)
        kickoff = WEEK_KICKOFFS[week]
        return {
            "revision_recorded_at_utc": kickoff - pd.Timedelta(hours=recorded_offset_hours),
            "refresh_run_id": f"refresh_{game_id}_{recorded_offset_hours}",
            "season": SEASON,
            "week": week,
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "kickoff": kickoff,
            "decision_home_spread": line,
            "original_recorded_at_utc": kickoff - pd.Timedelta(days=3),
            "previous_pick_side": previous,
            "previous_home_cover_probability": 0.5,
            "new_pick_side": new,
            "new_home_cover_probability": 0.5,
            "coach_fade_flip": False,
            "movement_policy": "test_policy",
            "movement_delta": 0.0,
            "movement_pick_side": new,
            "model_only_pick_side": new,
            "model_id": "test_active_model",
            "feature_table_sha256": "0" * 64,
            "reason": "test",
        }

    return pd.DataFrame(
        [
            # g2: an earlier revision that KEEPS the Tuesday pick, then the
            # real, later revision that FLIPS it. Only the later one should count.
            row("g2", 1, previous="HOME", new="HOME", recorded_offset_hours=48),
            row("g2", 1, previous="HOME", new="AWAY", recorded_offset_hours=24),
            row("g4", 2, previous="AWAY", new="AWAY", recorded_offset_hours=24),
        ]
    )


def _features_frame() -> pd.DataFrame:
    rows = []
    for game_id, week, _home, _away, line, result, _active_side, _challenger_side in GAMES:
        rows.append(
            {
                "game_id": game_id,
                "season": SEASON,
                "week": week,
                "spread_line": line,
                "result": result,
            }
        )
    return pd.DataFrame(rows)


def _write_registry(artifacts_root: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    payload = {
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": CHALLENGER_ID,
                "status": status,
                "evidence": {
                    "probability_positive": 0.87,
                    "registry_verdict": "unresolved",
                },
            }
        ],
    }
    atomic_json(payload, artifacts_root / "prospective" / "challengers.json")


def _write_recommendation_cards(artifacts_root: Path) -> None:
    for week, artifact in ACTIVE_FORECAST_ARTIFACT.items():
        game_ids = [g[0] for g in GAMES if g[1] == week]
        card = pd.DataFrame(
            {"game_id": game_ids, "home_cover_probability": [PROBABILITIES[g] for g in game_ids]}
        )
        path = artifacts_root / artifact / "recommendations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        card.to_csv(path, index=False)

    challenger_game_ids = [g[0] for g in GAMES]
    challenger_card = pd.DataFrame(
        {
            "game_id": challenger_game_ids,
            "home_cover_probability": [PROBABILITIES[g] for g in challenger_game_ids],
        }
    )
    challenger_path = (
        artifacts_root / "margin_predictions" / CHALLENGER_SOURCE_ARTIFACT / "recommendations.csv"
    )
    challenger_path.parent.mkdir(parents=True, exist_ok=True)
    challenger_card.to_csv(challenger_path, index=False)


def _build_full_fixture(
    artifacts_root: Path, *, challenger_game_ids: set[str] | None = None
) -> None:
    atomic_parquet(_paper_decisions_frame(), artifacts_root / "clv_ledger" / "decisions.parquet")
    atomic_parquet(
        _challenger_decisions_frame(challenger_game_ids),
        artifacts_root / "prospective" / "challenger_decisions.parquet",
    )
    atomic_parquet(
        _pick_revisions_frame(), artifacts_root / "prospective" / "pick_revisions.parquet"
    )
    _write_registry(artifacts_root)
    _write_recommendation_cards(artifacts_root)


def _find_row(rows: list[dict[str, Any]], entrant_id: str) -> dict[str, Any]:
    for row in rows:
        if row["entrant_id"] == entrant_id:
            return row
    raise AssertionError(f"No scorecard row for {entrant_id!r}")


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------


def test_coverage_counts_games_on_card_vs_games_recorded(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    # The challenger only ever recorded 4 of the 6 games on the active
    # model's card (e.g. it skipped g5/g6 some week).
    _build_full_fixture(artifacts_root, challenger_game_ids={"g1", "g2", "g3", "g4"})

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)

    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    assert active["games_on_card"] == 6
    assert active["games_recorded"] == 6
    assert active["coverage_ratio"] == pytest.approx(1.0)

    challenger = _find_row(rows, CHALLENGER_ID)
    assert challenger["games_on_card"] == 6
    assert challenger["games_recorded"] == 4
    assert challenger["coverage_ratio"] == pytest.approx(4 / 6)


def test_empty_ledgers_report_cleanly_with_no_games_needed_language(tmp_path: Path) -> None:
    """The real 2025 case: no ledger files exist at all yet."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    (artifacts_root / "prospective").mkdir(parents=True)
    _write_registry(artifacts_root)

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)

    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    assert active["games_on_card"] == 0
    assert active["games_recorded"] == 0
    assert active["coverage_ratio"] is None
    assert active["settled_games"] == 0
    assert active["classification"] == POOLABLE_CLASSIFICATION

    challenger = _find_row(rows, CHALLENGER_ID)
    assert challenger["games_recorded"] == 0
    assert challenger["settled_games"] == 0
    assert challenger["classification"] == POOLABLE_CLASSIFICATION

    rendered = render_markdown(rows, season=SEASON, through_week=None) + json.dumps(
        rows, default=str
    )
    lowered = rendered.lower()
    for phrase in _FORBIDDEN_SUBSTRINGS:
        assert phrase not in lowered


# ---------------------------------------------------------------------------
# 2. Paired delta vs. the active model, with interval + probability_positive
# ---------------------------------------------------------------------------


def test_paired_delta_and_probability_positive_vs_active_model(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    challenger = _find_row(rows, CHALLENGER_ID)

    # Active model settles correctly on g1, g3, g4 and incorrectly on g2, g6
    # (g5 is a push on both sides): 3/5 = 60%. The challenger flips g2 and g6
    # to the winning side and agrees everywhere else: 5/5 = 100%.
    active_accuracy = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)["accuracy_decision_line"]
    assert active_accuracy == pytest.approx(0.6)
    assert challenger["accuracy_decision_line"] == pytest.approx(1.0)

    paired = challenger["paired_vs_active"]
    assert paired["shared_settled_games"] == 5
    metric = paired["metrics"]["paired_delta_accuracy_points"]
    # (0 + 1 + 0 + 0 + 1) / 5 * 100 = 40.0 accuracy points, hand-computed.
    assert metric["estimate"] == pytest.approx(40.0)
    assert 0.0 <= metric["probability_positive"] <= 1.0
    assert metric["interval_lower"] <= metric["interval_upper"]


# ---------------------------------------------------------------------------
# 3. Overlay marginal effect: only the games this entrant's pick differs on
# ---------------------------------------------------------------------------


def test_overlay_marginal_effect_on_disagreement_games_only(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    challenger = _find_row(rows, CHALLENGER_ID)
    marginal = challenger["overlay_marginal"]

    # The challenger disagrees with the active model's chain pick on exactly
    # g2 and g6, and wins both.
    assert marginal["shared_settled_games"] == 5
    assert marginal["disagreement_games"] == 2
    metric = marginal["metrics"]["marginal_paired_delta_accuracy_points"]
    assert metric["estimate"] == pytest.approx(100.0)
    assert 0.0 <= metric["probability_positive"] <= 1.0


def test_active_model_row_has_no_overlay_marginal_or_paired_delta(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)
    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    assert active["paired_vs_active"] is None
    assert active["overlay_marginal"] is None


# ---------------------------------------------------------------------------
# 4. Refresh effect: Tuesday pick vs. final refresh pick
# ---------------------------------------------------------------------------


def test_refresh_effect_counts_flips_and_uses_the_latest_revision_only(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    refresh = active["refresh_effect"]

    assert refresh["available"] is True
    assert refresh["revised_games"] == 2  # g2 and g4
    # g2's LATEST revision flips HOME -> AWAY (the earlier no-op revision to
    # HOME must not count); g4's revision keeps AWAY -> AWAY.
    assert refresh["flips"] == 1
    assert refresh["kept"] == 1
    assert refresh["settled_games"] == 2

    metric = refresh["metrics"]["refresh_paired_delta_accuracy_points"]
    # g2: previous(HOME) wrong -> 0, new(AWAY) right -> 1, delta=+100.
    # g4: previous(AWAY) right -> 1, new(AWAY) right -> 1, delta=0.
    # mean = 50.0 accuracy points, hand-computed.
    assert metric["estimate"] == pytest.approx(50.0)

    challenger = _find_row(rows, CHALLENGER_ID)
    assert challenger["refresh_effect"]["available"] is False


# ---------------------------------------------------------------------------
# 5. Calibration: Brier score + reliability bins at the existing bin width
# ---------------------------------------------------------------------------


def test_calibration_brier_score_and_reliability_bins(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)
    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    calibration = active["calibration"]

    assert calibration["available"] is True
    assert calibration["settled_games"] == 5
    assert calibration["games_with_recorded_probability"] == 5
    # actual=[1,0,1,0,1] for g1,g2,g3,g4,g6; probability=[.70,.40,.65,.30,.55].
    expected_brier = (
        (0.7 - 1) ** 2 + (0.4 - 0) ** 2 + (0.65 - 1) ** 2 + (0.3 - 0) ** 2 + (0.55 - 1) ** 2
    ) / 5
    assert calibration["brier_score"] == pytest.approx(expected_brier)
    assert isinstance(calibration["bins"], list)
    assert sum(bin_row["games"] for bin_row in calibration["bins"]) == 5


def test_calibration_reports_missing_cards_without_crashing(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    atomic_parquet(_paper_decisions_frame(), artifacts_root / "clv_ledger" / "decisions.parquet")
    _write_registry(artifacts_root)
    # Deliberately do NOT write any recommendations.csv cards.

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)
    active = _find_row(rows, ACTIVE_MODEL_ENTRANT_ID)
    assert active["calibration"]["available"] is False
    assert active["calibration"]["games_with_recorded_probability"] == 0


# ---------------------------------------------------------------------------
# 6. Classification: an interval containing zero is unresolved_below_power,
#    and so is every other row -- this module never emits a terminal verdict.
# ---------------------------------------------------------------------------


def test_classification_is_always_a_registry_admissible_state(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)
    for row in rows:
        assert row["classification"] in CLASSIFICATIONS
        assert row["classification"] == POOLABLE_CLASSIFICATION


def test_interval_containing_zero_classifies_unresolved_below_power_directly() -> None:
    """Unit test on the classifier itself, both sides of the invariant.

    An interval crossing zero -> unresolved_below_power (the required case).
    An interval NOT crossing zero (all-positive or all-negative) is ALSO
    unresolved_below_power: this module never issues refuted_mechanism or
    bounded_by_control on its own (see the module docstring) -- those are
    registry-level verdicts this report does not compute.
    """

    from nfl_ats.prospective_scorecard import _classification

    crossing = {
        "metrics": {"paired_delta_accuracy_points": {"interval_lower": -1.0, "interval_upper": 1.0}}
    }
    classification, crosses_zero = _classification(crossing)
    assert classification == POOLABLE_CLASSIFICATION
    assert crosses_zero is True

    all_positive = {
        "metrics": {"paired_delta_accuracy_points": {"interval_lower": 0.5, "interval_upper": 2.0}}
    }
    classification, crosses_zero = _classification(all_positive)
    assert classification == POOLABLE_CLASSIFICATION
    assert crosses_zero is False

    all_negative = {
        "metrics": {
            "paired_delta_accuracy_points": {"interval_lower": -3.0, "interval_upper": -0.5}
        }
    }
    classification, crosses_zero = _classification(all_negative)
    assert classification == POOLABLE_CLASSIFICATION
    assert crosses_zero is False

    no_data = {"shared_settled_games": 0, "note": "no shared settled games yet"}
    classification, crosses_zero = _classification(no_data)
    assert classification == POOLABLE_CLASSIFICATION
    assert crosses_zero is None


# ---------------------------------------------------------------------------
# 7. The registered (challengers.json) status is surfaced, not overridden
# ---------------------------------------------------------------------------


def test_registered_evidence_is_surfaced_without_changing_the_fresh_classification(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(artifacts_root, data_root, _features_frame(), season=SEASON)
    challenger = _find_row(rows, CHALLENGER_ID)

    assert challenger["challenger_status"] == "ACTIVE_PROSPECTIVE"
    assert challenger["registered_evidence"]["probability_positive"] == pytest.approx(0.87)
    assert challenger["registered_evidence"]["registry_verdict"] == "unresolved"
    # The registered evidence is informational; this report's own fresh
    # classification is still the invariant-safe default.
    assert challenger["classification"] == POOLABLE_CLASSIFICATION


# ---------------------------------------------------------------------------
# 8. settled_games is reported; no "games needed" figure ever appears
# ---------------------------------------------------------------------------


def test_settled_games_present_and_forbidden_phrasing_absent_everywhere(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _build_full_fixture(artifacts_root)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    for row in rows:
        assert isinstance(row["settled_games"], int)

    markdown = render_markdown(rows, season=SEASON, through_week=None)
    frame = scorecards_to_frame(rows)
    payload = json.dumps(rows, default=str)

    combined = (markdown + payload + frame.to_string()).lower()
    for phrase in _FORBIDDEN_SUBSTRINGS:
        assert phrase not in combined, f"forbidden phrase {phrase!r} leaked into scorecard output"

    # No key anywhere in the payload is literally a "games needed" figure.
    assert "games_needed" not in payload


# ---------------------------------------------------------------------------
# 9. ENG-33: closing-ground CANDIDATE detection and next_admissible_action.
#
# BINDING (pasted verbatim, AGENTS.md): an interval or CI that contains zero
# is NEVER grounds to reject, fail, or close an experiment. At this
# evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome
# for a real small signal. Only two grounds ever close a line of work: (1)
# refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
# side of zero) or zero split-half reliability; (2) bounded by a positive
# control proven able to detect an effect that size. Everything else is
# unresolved_below_power: record it with `nfl-ats weak-signals record`,
# report probability_positive, never the binary "contains zero". Verdicts
# flow through `nfl-ats weak-signals record` / `nfl-ats rotation
# record-look` only -- this module's `closing_ground_candidate` and
# `next_admissible_action` are advisory reports, never verdicts, and never
# change `classification`.
# ---------------------------------------------------------------------------


def _empty_rotation_registry() -> rotation_module.Registry:
    return rotation_module.Registry(version=1, notes=(), families={})


def _empty_weak_signal_registry() -> weak_signals_module.Registry:
    return weak_signals_module.Registry(version=1, notes=(), signals={})


def _write_minimal_registries(registry_root: Path, *, with_closed_signal_for: str | None) -> None:
    """A minimal, schema-valid pair of registries for read-only ENG-33 tests.

    ``with_closed_signal_for``, when given, adds one ``refuted_mechanism``
    signal (an admissibly RESOLVED wrong sign: interval entirely below zero,
    per ``weak_signals.validate_closure``) whose name a test's challenger
    fixture cites via ``evidence.registry_source``.
    """

    registry_root.mkdir(parents=True, exist_ok=True)
    signals: dict[str, Any] = {}
    if with_closed_signal_for is not None:
        signals[with_closed_signal_for] = {
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "description": "ENG-33 fixture: an already-closed refuted mechanism",
            "source": "test",
            "effect": -2.0,
            "effect_units": "accuracy_points",
            "classification": "refuted_mechanism",
            "classification_evidence": "week-blocked interval below zero, wrong sign resolved",
            "closing_ground": "wrong_sign_resolved",
            "league": "nfl",
            "seasons": [2020, 2021],
            "standard_error": None,
            "interval": [-5.0, -1.0],
            "probability_positive": 0.02,
            "sample_games": 100,
            "sample_blocks": 10,
            "reliability": None,
            "family": None,
            "notes": "",
            "plain_summary": None,
            "category": None,
        }
    (registry_root / "weak_signals.json").write_text(
        json.dumps({"version": 1, "notes": [], "signals": signals}), encoding="utf-8"
    )
    (registry_root / "rotation_registry.json").write_text(
        json.dumps({"version": 1, "notes": [], "families": {}}), encoding="utf-8"
    )


def test_predeclared_sign_reads_challenger_declaration_or_reports_reason() -> None:
    assert _predeclared_sign(None) == (None, "no_predeclared_sign")
    assert _predeclared_sign({}) == (None, "no_predeclared_sign")
    assert _predeclared_sign({"evidence": "not a dict"}) == (None, "no_predeclared_sign")
    # Reporting probability_positive IS the predeclared direction under this
    # repo's "positive favours candidate" convention -- present regardless of
    # its value.
    assert _predeclared_sign({"evidence": {"probability_positive": 0.87}}) == (1, None)
    # A signed accuracy-point effect the challenger declared for itself.
    assert _predeclared_sign({"evidence": {"effect_accuracy_points": 3.92}}) == (1, None)
    assert _predeclared_sign({"evidence": {"source_effect_accuracy_points": -7.905401}}) == (
        -1,
        None,
    )
    # A zero effect and nothing else declares no direction.
    assert _predeclared_sign({"evidence": {"effect_accuracy_points": 0}}) == (
        None,
        "no_predeclared_sign",
    )


def test_referenced_signal_names_extracts_from_nested_evidence() -> None:
    entry = {
        "evidence": {
            "registry_source": "registry/weak_signals.json:hc_year_one_fade",
            "parent_cell": {
                "nested_source": [
                    "registry/weak_signals.json:penalty_crew_holding_tilt_run_heavy",
                    "registry/weak_signals.json:hc_year_one_fade",  # duplicate, must dedupe
                ]
            },
            "artifact": "artifacts/experiment_runner/20260820T113432Z/metadata.json",
        }
    }
    assert _referenced_signal_names(entry) == (
        "hc_year_one_fade",
        "penalty_crew_holding_tilt_run_heavy",
    )
    assert _referenced_signal_names({}) == ()


def test_split_half_reliability_perfect_negative_correlation_is_at_or_below_zero() -> None:
    # Consecutive-week pairs (1,4), (2,3), (3,2), (4,1): first=[1,2,3,4],
    # second=[4,3,2,1], a perfect negative correlation (r = -1.0).
    per_week = pd.Series([1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0], index=range(1, 9))
    result = _split_half_reliability(per_week, samples=200, seed=1)
    assert result["available"] is True
    assert result["week_pairs"] == 4
    assert result["reliability"] == pytest.approx(-1.0)
    assert result["interval_upper"] <= 0.0
    assert "independent unit" in result["note"]


def test_split_half_reliability_perfect_positive_correlation_has_positive_upper_bound() -> None:
    per_week = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], index=range(1, 9))
    result = _split_half_reliability(per_week, samples=200, seed=1)
    assert result["available"] is True
    assert result["reliability"] == pytest.approx(1.0)
    assert result["interval_upper"] > 0.0


def test_split_half_reliability_fewer_than_three_pairs_reports_unavailable() -> None:
    per_week = pd.Series([1.0, 2.0, 3.0, 4.0], index=range(1, 5))
    result = _split_half_reliability(per_week, samples=200, seed=1)
    assert result["available"] is False
    assert result["week_pairs"] == 2
    assert "independent unit" in result["note"]


def test_closing_ground_candidate_wrong_sign_resolved_when_whole_interval_opposite_sign() -> None:
    paired_metric = {"interval_lower": -5.0, "interval_upper": -1.0, "estimate": -3.0}
    candidate, evidence = _closing_ground_candidate(
        predeclared_sign=1,
        predeclared_sign_reason=None,
        paired_metric=paired_metric,
        split_half={"available": False},
    )
    assert candidate == WRONG_SIGN_RESOLVED == "wrong_sign_resolved"
    assert evidence["paired_delta_interval"] == [-5.0, -1.0]
    assert evidence["predeclared_sign"] == "positive"
    # The candidate is advisory only: the module's own classification never
    # moves off unresolved_below_power, on this or any interval.
    classification, crosses_zero = _classification(
        {"metrics": {"paired_delta_accuracy_points": paired_metric}}
    )
    assert classification == "unresolved_below_power"
    assert crosses_zero is False


def test_closing_ground_candidate_null_when_interval_crosses_zero() -> None:
    candidate, _evidence = _closing_ground_candidate(
        predeclared_sign=1,
        predeclared_sign_reason=None,
        paired_metric={"interval_lower": -1.0, "interval_upper": 2.0, "estimate": 0.5},
        split_half={"available": False},
    )
    assert candidate is None


def test_closing_ground_candidate_no_split_half_reliability_when_upper_bound_at_or_below_zero() -> (
    None
):
    for upper in (-0.01, 0.0):
        candidate, evidence = _closing_ground_candidate(
            predeclared_sign=None,
            predeclared_sign_reason="no_predeclared_sign",
            paired_metric={"interval_lower": -1.0, "interval_upper": 2.0, "estimate": 0.5},
            split_half={"available": True, "reliability": -0.4, "interval_upper": upper},
        )
        assert candidate == NO_SPLIT_HALF_RELIABILITY == "no_split_half_reliability"
        assert evidence["split_half_reliability"]["interval_upper"] == upper


def test_closing_ground_candidate_null_with_reason_when_no_predeclared_sign() -> None:
    candidate, evidence = _closing_ground_candidate(
        predeclared_sign=None,
        predeclared_sign_reason="no_predeclared_sign",
        paired_metric={"interval_lower": -5.0, "interval_upper": -1.0, "estimate": -3.0},
        split_half={"available": False},
    )
    # No predeclared sign means the wrong-sign ground can never fire, no
    # matter how one-sided the interval is.
    assert candidate is None
    assert evidence["predeclared_sign"] is None
    assert evidence["predeclared_sign_reason"] == "no_predeclared_sign"


def test_next_admissible_action_active_model_is_record_pending_look() -> None:
    action, detail = _next_admissible_action(
        None,
        has_settled_shared_data=False,
        weak_signal_registry=_empty_weak_signal_registry(),
        rotation_registry=_empty_rotation_registry(),
    )
    assert action == "record_pending_look"
    assert action in NEXT_ADMISSIBLE_ACTIONS
    assert "active model" in detail


def test_next_admissible_action_test_on_production_when_no_family_and_no_data() -> None:
    action, _detail = _next_admissible_action(
        {"evidence": {"note": "no registry_source cited"}},
        has_settled_shared_data=False,
        weak_signal_registry=_empty_weak_signal_registry(),
        rotation_registry=_empty_rotation_registry(),
    )
    assert action == "test_on_production"


def test_next_admissible_action_record_pending_look_when_data_exists_and_no_family() -> None:
    action, _detail = _next_admissible_action(
        {"evidence": {"note": "no registry_source cited"}},
        has_settled_shared_data=True,
        weak_signal_registry=_empty_weak_signal_registry(),
        rotation_registry=_empty_rotation_registry(),
    )
    assert action == "record_pending_look"


def test_next_admissible_action_closed_only_when_registry_holds_admissible_closure(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    _write_minimal_registries(registry_root, with_closed_signal_for="foo_signal")
    weak_signal_registry = weak_signals_module.load_registry(registry_root / "weak_signals.json")
    entry = {"evidence": {"registry_source": "registry/weak_signals.json:foo_signal"}}

    action, detail = _next_admissible_action(
        entry,
        has_settled_shared_data=True,
        weak_signal_registry=weak_signal_registry,
        rotation_registry=_empty_rotation_registry(),
    )
    assert action == "closed"
    assert "foo_signal" in detail
    assert "wrong_sign_resolved" in detail
    # Independently re-verify against the registry the action claims to read
    # (not just trusting the function's own internals): the cited signal
    # really is a TERMINAL classification with an admissible closing_ground.
    signal = weak_signal_registry.signals["foo_signal"]
    assert signal.classification in weak_signals_module.TERMINAL_CLASSIFICATIONS
    assert signal.closing_ground in weak_signals_module.CLOSING_GROUNDS[signal.classification]

    # A challenger citing NO closed signal must never get "closed".
    unclosed_action, _detail = _next_admissible_action(
        {"evidence": {"registry_source": "registry/weak_signals.json:not_a_recorded_signal"}},
        has_settled_shared_data=True,
        weak_signal_registry=weak_signal_registry,
        rotation_registry=_empty_rotation_registry(),
    )
    assert unclosed_action != "closed"


def test_next_admissible_action_is_never_wait(tmp_path: Path) -> None:
    """Full-pipeline check: every row's next_admissible_action is one of the
    fixed six strings and is never a disguised instruction to keep waiting.
    """

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    registry_root = tmp_path / "registry"
    _build_full_fixture(artifacts_root)
    _write_minimal_registries(registry_root, with_closed_signal_for=None)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        registry_root=registry_root,
    )
    for row in rows:
        assert row["next_admissible_action"] in NEXT_ADMISSIBLE_ACTIONS
        assert row["next_admissible_action"] != "wait"
        assert "wait" not in row["next_admissible_action"].lower()
        assert row["classification"] == POOLABLE_CLASSIFICATION


def test_closed_next_action_requires_an_admissible_registry_closure(tmp_path: Path) -> None:
    """Language guard (ENG-33): extends the forbidden-phrase tests above.

    ``next_admissible_action`` may equal the literal string "closed" ONLY
    when the registry this report read actually holds an admissible
    terminal closure for a signal the row cites -- never for a row whose
    registry state is unresolved. This is the same
    BINDING taxonomy pasted at the top of this section: a terminal verdict
    needs a RESOLVED wrong sign, zero split-half reliability, or a proven
    positive control, and only `nfl-ats weak-signals record` /
    `nfl-ats rotation record-look` may act on it -- this report only
    reflects what those commands would already find.
    """

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    registry_root = tmp_path / "registry"
    _build_full_fixture(artifacts_root)
    # Give the registered challenger's own evidence a registry_source
    # pointing at a signal the registry has already closed admissibly.
    payload = json.loads((artifacts_root / "prospective" / "challengers.json").read_text())
    payload["challengers"][0]["evidence"]["registry_source"] = (
        "registry/weak_signals.json:hc_year_one_fade_test"
    )
    (artifacts_root / "prospective" / "challengers.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    _write_minimal_registries(registry_root, with_closed_signal_for="hc_year_one_fade_test")

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        registry_root=registry_root,
    )
    weak_signal_registry = weak_signals_module.load_registry(registry_root / "weak_signals.json")

    saw_closed = False
    for row in rows:
        if row["next_admissible_action"] == "closed":
            saw_closed = True
            cited = row["next_admissible_action_detail"]
            assert "hc_year_one_fade_test" in cited
            signal = weak_signal_registry.signals["hc_year_one_fade_test"]
            assert signal.classification in weak_signals_module.TERMINAL_CLASSIFICATIONS
            assert (
                signal.closing_ground in weak_signals_module.CLOSING_GROUNDS[signal.classification]
            )
        else:
            assert row["next_admissible_action"] != "closed"
        # classification is untouched either way.
        assert row["classification"] == POOLABLE_CLASSIFICATION
    assert saw_closed, "fixture was built with an admissible closure but no row reported it"


def test_registry_files_are_never_modified_by_a_scorecard_run(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    registry_root = tmp_path / "registry"
    _build_full_fixture(artifacts_root)
    _write_minimal_registries(registry_root, with_closed_signal_for="hc_year_one_fade_test")

    weak_signals_before = (registry_root / "weak_signals.json").read_bytes()
    rotation_before = (registry_root / "rotation_registry.json").read_bytes()

    build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        registry_root=registry_root,
    )

    assert (registry_root / "weak_signals.json").read_bytes() == weak_signals_before
    assert (registry_root / "rotation_registry.json").read_bytes() == rotation_before


def test_closing_ground_fields_present_on_every_row_and_advisory_only(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    registry_root = tmp_path / "registry"
    _build_full_fixture(artifacts_root)
    _write_minimal_registries(registry_root, with_closed_signal_for=None)

    rows = build_season_scorecards(
        artifacts_root,
        data_root,
        _features_frame(),
        season=SEASON,
        registry_root=registry_root,
    )
    for row in rows:
        assert "closing_ground_candidate" in row
        assert row["closing_ground_candidate"] in (
            None,
            WRONG_SIGN_RESOLVED,
            NO_SPLIT_HALF_RELIABILITY,
            "positive_control_bound",
        )
        assert "closing_ground_evidence" in row
        assert row["classification"] == POOLABLE_CLASSIFICATION

    markdown = render_markdown(rows, season=SEASON, through_week=None)
    assert "candidate" in markdown.lower()
    assert "Next admissible action" in markdown
    frame = scorecards_to_frame(rows)
    assert "closing_ground_candidate" in frame.columns
    assert "next_admissible_action" in frame.columns

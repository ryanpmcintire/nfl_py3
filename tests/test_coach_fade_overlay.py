"""Year-1 head-coach fade overlay (PER-07 offshoot, docs/coach_fade_overlay.md).

Three things are load-bearing here, matching AGENTS.md's "add a leakage
regression test for every new feature family" and the owner's dual-arm
recording requirement:

1. :func:`year_one_by_game`'s flag is derived from data, not hand-typed, and
   is pregame-safe -- a later week's or a later season's coach data must never
   retroactively change an earlier game's flag.
2. :func:`apply_coach_fade_overlay` flips ONLY the clean case (model sides
   with a year-1 team against a KEPT-coach opponent), inside weeks 1-8, and
   touches no other column.
3. :func:`record_overlay_challenger_decisions` writes the overlay's own
   picks -- which can diverge from the active model's raw picks on a flipped
   game -- to the prospective challenger ledger, so both arms are on record.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.coach_fade_overlay import (
    CHALLENGER_ID,
    apply_coach_fade_overlay,
    overlay_disclosure_note,
    record_overlay_challenger_decisions,
    year_one_by_game,
)
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# Teams: YR1 and YR1C both change coaches heading into 2026 (year-1); YR1B
# also changes -- it is paired against YR1C in a both-year-1 game; KEEP does
# not change (the clean comparison side). OPP1-4 exist only to seed a REG
# game for each team's 2025 season so `team_season_primary_coach` has a prior
# row to read.


def _tenure_schedules() -> pd.DataFrame:
    rows = [
        # 2025 REG season: seeds each team's PRIOR coach of record.
        ("2025_01_YR1_OPP1", 2025, "REG", 1, "YR1", "OPP1", "Old1", "OppC1"),
        ("2025_01_YR1B_OPP2", 2025, "REG", 1, "YR1B", "OPP2", "Old2", "OppC2"),
        ("2025_01_YR1C_OPP4", 2025, "REG", 1, "YR1C", "OPP4", "Old3", "OppC4"),
        ("2025_01_KEEP_OPP3", 2025, "REG", 1, "KEEP", "OPP3", "Steady", "OppC3"),
        # 2026 REG season.
        # Week 1: the clean flip candidate -- KEEP (not year-1) hosts YR1
        # (year-1, new coach "New1").
        ("2026_01_KEEP_YR1", 2026, "REG", 1, "KEEP", "YR1", "Steady", "New1"),
        # Week 1: a both-year-1 game -- neither side is a clean fade target.
        ("2026_01_YR1B_YR1C", 2026, "REG", 1, "YR1B", "YR1C", "New2", "New3"),
        # Week 9: the SAME clean matchup as week 1, to prove weeks 9+ are
        # never touched regardless of the coach flags themselves.
        ("2026_09_KEEP_YR1", 2026, "REG", 9, "KEEP", "YR1", "Steady", "New1"),
        # Week 5: YR1's OWN credited coach changes again (an in-season
        # interim hire). Used only by the leakage regression below.
        ("2026_05_YR1_OPP1", 2026, "REG", 5, "YR1", "OPP1", "Interim1", "OppC1"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
        ],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_KEEP_YR1", "2026_01_YR1B_YR1C", "2026_09_KEEP_YR1"],
            "season": [2026, 2026, 2026],
            "week": [1, 1, 9],
            "game_type": ["REG", "REG", "REG"],
            "home_team": ["KEEP", "YR1B", "KEEP"],
            "away_team": ["YR1", "YR1C", "YR1"],
            "kickoff": [
                "2026-09-10T17:00:00+00:00",
                "2026-09-10T17:00:00+00:00",
                "2026-10-29T17:00:00+00:00",
            ],
            "spread_line": [-3.5, 6.0, -3.5],
            # Home is NOT picked in any row (the away side is favored by the
            # model each time). Row 3 is deliberately the SAME shape as row 1
            # (same matchup, same probability) so the week cutoff is the ONLY
            # thing distinguishing "flips" from "does not flip".
            "home_cover_probability": [0.35, 0.30, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. year_one_by_game: derived, pregame-safe
# ---------------------------------------------------------------------------


def test_year_one_by_game_requires_an_observed_contiguous_prior_season() -> None:
    schedules = pd.concat(
        [
            _tenure_schedules(),
            # NEWTEAM's only appearance anywhere in the data is 2026 -- no
            # observed prior season, so it must not be flagged even though it
            # obviously has "a coach" this year.
            pd.DataFrame(
                [("2026_01_NEWTEAM_OPP1", 2026, "REG", 1, "NEWTEAM", "OPP1", "Brand New", "OppC1")],
                columns=[
                    "game_id",
                    "season",
                    "game_type",
                    "week",
                    "home_team",
                    "away_team",
                    "home_coach",
                    "away_coach",
                ],
            ),
        ],
        ignore_index=True,
    )
    flags = year_one_by_game(schedules).set_index("game_id")

    assert bool(flags.loc["2026_01_KEEP_YR1", "year_one_away"]) is True  # YR1: New1 != Old1
    assert bool(flags.loc["2026_01_KEEP_YR1", "year_one_home"]) is False  # KEEP: Steady == Steady
    assert bool(flags.loc["2026_01_NEWTEAM_OPP1", "year_one_home"]) is False  # no prior season


def test_year_one_flags_reproduce_the_seven_2026_teams_from_local_schedule_data() -> None:
    """End-to-end sanity check against the SAME source the Week 1 2026 brief
    used, proving the flag is derived rather than hand-typed."""

    path = Path("data/raw/20260817T235649Z/schedules.parquet")
    if not path.is_file():
        pytest.skip("local nflverse snapshot not present in this environment")
    schedules = pd.read_parquet(path)
    flags = year_one_by_game(schedules)
    sched_2026 = schedules.loc[schedules["season"].eq(2026), ["game_id", "home_team", "away_team"]]
    merged = flags.merge(sched_2026, on="game_id").loc[lambda frame: frame["season"].eq(2026)]
    teams = set(merged.loc[merged["year_one_home"], "home_team"]) | set(
        merged.loc[merged["year_one_away"], "away_team"]
    )
    assert teams == {"BAL", "CLE", "LV", "MIA", "NYG", "PIT", "TEN"}


def test_year_one_flag_is_leak_safe_against_a_later_week_mutation() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    Week 1's flag for YR1 must not move when week 5's OWN credited coach
    value changes -- that would mean a later, in-season fact reached back and
    relabeled an earlier, already-pregame-known game.
    """

    baseline = year_one_by_game(_tenure_schedules()).set_index("game_id")

    mutated = _tenure_schedules()
    mutated.loc[mutated["game_id"].eq("2026_05_YR1_OPP1"), "home_coach"] = "SomeOtherInterim"
    changed = year_one_by_game(mutated).set_index("game_id")

    assert bool(changed.loc["2026_01_KEEP_YR1", "year_one_away"]) == bool(
        baseline.loc["2026_01_KEEP_YR1", "year_one_away"]
    )
    # The week-5 row itself is free to move -- only week 1 is insulated.
    assert bool(baseline.loc["2026_05_YR1_OPP1", "year_one_home"]) is True


def test_year_one_flag_is_leak_safe_across_the_season_boundary() -> None:
    """A future season's schedule (even one with the same teams) must never
    change an earlier season's already-computed flag."""

    schedules = _tenure_schedules()
    baseline = year_one_by_game(schedules)

    future = pd.DataFrame(
        [
            ("2027_01_YR1_OPP1", 2027, "REG", 1, "YR1", "OPP1", "Yet Another Coach", "OppC1"),
            ("2027_01_KEEP_OPP3", 2027, "REG", 1, "KEEP", "OPP3", "Someone Else", "OppC3"),
        ],
        columns=schedules.columns,
    )
    changed = year_one_by_game(pd.concat([schedules, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 2. apply_coach_fade_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_the_clean_case_week_one_game() -> None:
    result = apply_coach_fade_overlay(_predictions(), _tenure_schedules())

    assert result.flip_count == 1
    flip = result.flips[0]
    assert flip.game_id == "2026_01_KEEP_YR1"
    assert flip.year_one_team == "YR1"
    assert flip.opponent_team == "KEEP"

    overlaid = result.overlaid_predictions.set_index("game_id")
    # 0.35 -> 0.65: the pick flips from AWAY (YR1) to HOME (KEEP).
    assert overlaid.loc["2026_01_KEEP_YR1", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_touch_a_both_year_one_game() -> None:
    result = apply_coach_fade_overlay(_predictions(), _tenure_schedules())

    assert "2026_01_YR1B_YR1C" in result.both_year_one_games
    assert all(flip.game_id != "2026_01_YR1B_YR1C" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_YR1B_YR1C", "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_leaves_week_nine_and_later_untouched() -> None:
    result = apply_coach_fade_overlay(_predictions(), _tenure_schedules())

    assert all(flip.game_id != "2026_09_KEEP_YR1" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_09_KEEP_YR1", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_coach_fade_overlay(predictions, _tenure_schedules(), enabled=False)

    assert result.flip_count == 0
    assert result.both_year_one_games == ()
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_the_flipped_row() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point."""

    predictions = _predictions()
    result = apply_coach_fade_overlay(predictions, _tenure_schedules())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [column for column in predictions.columns if column != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].ne("2026_01_KEEP_YR1")
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    week_nine_only = _predictions().loc[lambda frame: frame["week"].eq(9)]
    result = apply_coach_fade_overlay(week_nine_only, _tenure_schedules())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_coach_fade_overlay(_predictions(), _tenure_schedules(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_the_flipped_game() -> None:
    result = apply_coach_fade_overlay(_predictions(), _tenure_schedules())
    note = overlay_disclosure_note(result)

    assert "Overlay applied: 1 pick flipped" in note
    assert "YR1 -> KEEP" in note
    assert "docs/coach_fade_overlay.md" in note


# ---------------------------------------------------------------------------
# 4. record_overlay_challenger_decisions: both arms on the ledger
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
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


def _write_overlay_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=2026,
        week=1,
        created_at_utc="2026-09-08T15:00:00+00:00",
        ridge_alpha=ridge_alpha,
        recommendations=_predictions()
        .loc[lambda frame: frame["week"].eq(1)]
        .reset_index(drop=True),
    )


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _tenure_schedules(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )
    return data_root


def test_record_overlay_challenger_decisions_records_both_arms(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_overlay_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)

    result = record_overlay_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_01_KEEP_YR1"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The overlay's OWN arm: the flipped game now records HOME (KEEP), which
    # diverges from what the raw card's home_cover_probability (0.35, an AWAY
    # pick) would have recorded on the active model's own paper ledger --
    # exactly the "both arms" divergence this challenger exists to capture.
    assert ledger.loc["2026_01_KEEP_YR1", "pick_side"] == "HOME"
    # The un-flipped, both-year-1 game keeps the model's own AWAY pick.
    assert ledger.loc["2026_01_YR1B_YR1C", "pick_side"] == "AWAY"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_overlay_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_overlay_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_overlay_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_overlay_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_overlay_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_overlay_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_overlay_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_overlay_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_overlay_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_overlay_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )


def test_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches CONFIG_FINGERPRINT_KEYS."""

    metadata = {
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {"path": "data/processed/game_features_weak_stack.parquet"}
        },
    }
    assert config_fingerprint(artifact_model_config(metadata)) == config_fingerprint(_MODEL_CONFIG)

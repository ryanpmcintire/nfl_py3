"""Special-teams return top-quartile tilt overlay (docs/special_teams_battery.md,
docs/special_teams_return_tilt_overlay.md).

Four things are load-bearing here, mirroring
``tests/test_interim_hc_first_game_tilt_overlay.py`` and
``tests/test_coach_fade_overlay.py``'s structure and AGENTS.md's "add a
leakage regression test for every new feature family" spirit:

1. :func:`return_composite_z_with_threshold` reproduces
   ``scripts/special_teams_screen.py``'s own composite/quartile-cut
   construction (pooled-sd z-score of the two return legs, mean of the two,
   ``QUARTILE_TOP=0.75`` quantile over the WHOLE panel).
2. :func:`special_teams_return_flag_by_game` is derived from data (a
   strictly PRIOR-season lookup, never the current season's or current
   game's own data, never an outcome column), and
   :func:`special_teams_return_flag_by_game_fail_open` FAILS OPEN (returns
   zero flags, never raises) when the team-season source snapshot is
   unavailable.
3. :func:`apply_special_teams_return_tilt_overlay` flips ONTO the flagged
   team only when exactly one side is flagged and the model's own pick is
   not already on that side, leaves a both-flagged game untouched, has no
   effect outside the flagged population, and is parameter-free.
4. :func:`record_special_teams_return_tilt_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, dual-tracked
   and at no rotation-registry window cost.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot
from nfl_ats.special_teams_return_tilt_overlay import (
    CHALLENGER_ID,
    QUARTILE_TOP,
    apply_special_teams_return_tilt_overlay,
    latest_special_teams_team_season,
    overlay_disclosure_note,
    record_special_teams_return_tilt_challenger_decisions,
    return_composite_z_with_threshold,
    special_teams_return_flag_by_game,
    special_teams_return_flag_by_game_fail_open,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# A 10-row 2025 team-season panel (n picked to avoid the (n-1)*0.75 == integer
# coincidence that would land a boundary team exactly ON the quantile cut).
# Top quartile (>= the 0.75 quantile of return_composite_z): FILLER, TEAMD,
# TEAMA -- verified empirically before writing this file, not hand-derived.
#
#   TEAMA=10, TEAMD=8, FILLER=5   -> top quartile
#   TEAMC=0.5, OPPW=0, OPPZ=-1, TEAMB=-2, OPPX=-3, OPPY=-4 -> NOT top quartile
#   TEAME=-10 -> NOT top quartile (used for the "current season not used as
#     its own prior" leak test: TEAME gets an EXTREME season-2026 row too)
#
# 2026 REG games (each team's PRIOR season is 2025, so these consult the
# panel above via the shift-by-one join):
#   2026_01_TEAMA_OPPX  -- home flagged only  -> flip-to-home candidate
#   2026_02_OPPY_TEAMD  -- away flagged only  -> "already on flagged" case
#   2026_03_TEAMA_TEAMD -- both flagged       -> never flips
#   2026_04_TEAMB_TEAMC -- neither flagged    -> no effect
#   2026_05_NEWTEAM_OPPW -- NEWTEAM has no 2025 row at all -> missing data
#   2026_06_TEAME_OPPW  -- TEAME's 2025 value is deeply negative -> not
#     flagged, even if a season-2026 row for TEAME is added to the panel


def _team_season_rows() -> list[tuple[int, str, float, float]]:
    return [
        (2025, "TEAMA", 10.0, 10.0),
        (2025, "TEAMD", 8.0, 8.0),
        (2025, "FILLER", 5.0, 5.0),
        (2025, "TEAMC", 0.5, 0.5),
        (2025, "OPPW", 0.0, 0.0),
        (2025, "OPPZ", -1.0, -1.0),
        (2025, "TEAMB", -2.0, -2.0),
        (2025, "OPPX", -3.0, -3.0),
        (2025, "OPPY", -4.0, -4.0),
        (2025, "TEAME", -10.0, -10.0),
    ]


def _team_season(rows: list[tuple[int, str, float, float]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        rows if rows is not None else _team_season_rows(),
        columns=["season", "team", "punt_return_yards_centered", "kickoff_return_yards_centered"],
    )


def _schedule() -> pd.DataFrame:
    rows = [
        ("2026_01_TEAMA_OPPX", 2026, "REG", 1, "TEAMA", "OPPX"),
        ("2026_02_OPPY_TEAMD", 2026, "REG", 2, "OPPY", "TEAMD"),
        ("2026_03_TEAMA_TEAMD", 2026, "REG", 3, "TEAMA", "TEAMD"),
        ("2026_04_TEAMB_TEAMC", 2026, "REG", 4, "TEAMB", "TEAMC"),
        ("2026_05_NEWTEAM_OPPW", 2026, "REG", 5, "NEWTEAM", "OPPW"),
        ("2026_06_TEAME_OPPW", 2026, "REG", 6, "TEAME", "OPPW"),
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "season", "game_type", "week", "home_team", "away_team"]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_01_TEAMA_OPPX",
                "2026_02_OPPY_TEAMD",
                "2026_03_TEAMA_TEAMD",
                "2026_04_TEAMB_TEAMC",
                "2026_05_NEWTEAM_OPPW",
                "2026_06_TEAME_OPPW",
            ],
            "season": [2026] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "game_type": ["REG"] * 6,
            "home_team": ["TEAMA", "OPPY", "TEAMA", "TEAMB", "NEWTEAM", "TEAME"],
            "away_team": ["OPPX", "TEAMD", "TEAMD", "TEAMC", "OPPW", "OPPW"],
            "kickoff": ["2026-09-24T17:00:00+00:00"] * 6,
            "spread_line": [-3.0, 2.0, -1.0, 1.0, 0.5, 2.5],
            # G01: model picks AWAY (OPPX) -- home (TEAMA) is the ONLY flagged
            #   side -> should flip to HOME.
            # G02: model picks AWAY (TEAMD) -- TEAMD IS the flagged side, so
            #   the model is already on it -> no flip.
            # G03: both TEAMA and TEAMD flagged -> never flips regardless of
            #   the model's pick.
            # G04: neither TEAMB nor TEAMC flagged -> no effect regardless of
            #   the model's pick.
            # G05: NEWTEAM has no prior-season row -> flag False, no flip,
            #   never an error.
            # G06: TEAME's actual (2025) prior is deeply negative -> not
            #   flagged, even though a leak-test variant adds an extreme
            #   season-2026 row for TEAME to the panel (see the leak test).
            "home_cover_probability": [0.30, 0.20, 0.55, 0.45, 0.50, 0.30],
        }
    )


def _write_data_root(tmp_path: Path, *, team_season_rows: list | None = None) -> Path:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    ts_dir = data_root / "raw" / "special_teams" / "20260819T232400Z"
    ts_dir.mkdir(parents=True)
    _team_season(team_season_rows).to_parquet(ts_dir / "team_season.parquet", index=False)
    return data_root


# ---------------------------------------------------------------------------
# 1. return_composite_z_with_threshold: reproduces the screen's own cut
# ---------------------------------------------------------------------------


def test_quartile_top_matches_the_screens_own_constant() -> None:
    assert QUARTILE_TOP == 0.75


def test_return_composite_threshold_reproduces_an_independent_quantile_call() -> None:
    """The composite is a pooled-sd z-score average of the two return legs;
    since both legs are set EQUAL here, the composite is a pure positive
    linear rescaling of the raw centered value, so an INDEPENDENT quantile
    computed directly on the raw centered values (divided by the same pooled
    sd, computed independently here too) must equal the function's own
    threshold -- this is not just re-calling the function under test."""

    team_season = _team_season()
    composite, threshold = return_composite_z_with_threshold(team_season)

    raw = team_season["punt_return_yards_centered"].to_numpy(dtype=float)
    independent_sd = float(np.std(raw, ddof=1))
    independent_threshold = float(np.quantile(raw, QUARTILE_TOP)) / independent_sd
    assert threshold == pytest.approx(independent_threshold)

    flagged = set(composite.loc[composite["return_composite_z"] >= threshold, "team"])
    assert flagged == {"TEAMA", "TEAMD", "FILLER"}


def test_return_composite_reproduces_the_live_registry_threshold() -> None:
    """Reproduction gate against the actual measured artifact
    (artifacts/special_teams_battery/20260819T232856Z/results.json), when
    the real snapshot is present locally (skipped in a fresh clone, where
    data/raw/** is gitignored)."""

    snapshot = Path("data/raw/special_teams/20260819T232400Z/team_season.parquet")
    if not snapshot.is_file():
        pytest.skip("data/raw/special_teams snapshot not present locally")
    team_season = pd.read_parquet(snapshot)
    _composite, threshold = return_composite_z_with_threshold(team_season)
    assert threshold == pytest.approx(0.4769479933229231)


def test_return_composite_requires_its_columns() -> None:
    with pytest.raises(DataContractError, match="return composite"):
        return_composite_z_with_threshold(pd.DataFrame({"season": [2025], "team": ["TEAMA"]}))


# ---------------------------------------------------------------------------
# 2. special_teams_return_flag_by_game: derived, prior-season-only, fail-open
# ---------------------------------------------------------------------------


def test_flag_fires_on_the_top_quartile_prior_season_team_only() -> None:
    flags = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(flags.loc["2026_01_TEAMA_OPPX", "home_return_top_quartile"]) is True
    assert bool(flags.loc["2026_01_TEAMA_OPPX", "away_return_top_quartile"]) is False


def test_flag_fires_for_the_away_side_too() -> None:
    flags = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(flags.loc["2026_02_OPPY_TEAMD", "home_return_top_quartile"]) is False
    assert bool(flags.loc["2026_02_OPPY_TEAMD", "away_return_top_quartile"]) is True


def test_flag_fires_for_both_sides_of_a_mutual_top_quartile_game() -> None:
    flags = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(flags.loc["2026_03_TEAMA_TEAMD", "home_return_top_quartile"]) is True
    assert bool(flags.loc["2026_03_TEAMA_TEAMD", "away_return_top_quartile"]) is True


def test_flag_is_false_for_neither_team_top_quartile() -> None:
    flags = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(flags.loc["2026_04_TEAMB_TEAMC", "home_return_top_quartile"]) is False
    assert bool(flags.loc["2026_04_TEAMB_TEAMC", "away_return_top_quartile"]) is False


def test_flag_is_false_with_no_prior_season_row_and_never_errors() -> None:
    """NEWTEAM has no 2025 row at all -- missing prior data folds into
    'not flagged', mirroring the screen's own n_missing_required_data
    handling, never an exception."""

    flags = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(flags.loc["2026_05_NEWTEAM_OPPW", "home_return_top_quartile"]) is False
    assert pd.isna(flags.loc["2026_05_NEWTEAM_OPPW", "home_prior_return_composite_z"])


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="special-teams return tracking"):
        special_teams_return_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _team_season())


def test_flag_excludes_non_reg_games() -> None:
    schedule = _schedule()
    postseason = schedule.copy()
    postseason["game_type"] = "POST"
    flags = special_teams_return_flag_by_game(postseason, _team_season())
    assert flags.empty


# --- leakage regression tests -----------------------------------------------


def test_flag_never_uses_the_current_seasons_own_row_as_its_own_prior() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    TEAME's actual PRIOR (2025) value is deeply negative -- not flagged.
    Adding an EXTREME season-2026 row for TEAME to the very same panel must
    NOT flip the 2026 game's flag: a team-season row is only ever consulted
    as the PRIOR for the season immediately AFTER the one it describes, never
    for its own season.
    """

    baseline = special_teams_return_flag_by_game(_schedule(), _team_season()).set_index("game_id")
    assert bool(baseline.loc["2026_06_TEAME_OPPW", "home_return_top_quartile"]) is False

    augmented_rows = [*_team_season_rows(), (2026, "TEAME", 50.0, 50.0)]
    augmented = special_teams_return_flag_by_game(
        _schedule(), _team_season(augmented_rows)
    ).set_index("game_id")
    assert bool(augmented.loc["2026_06_TEAME_OPPW", "home_return_top_quartile"]) is False


def test_flag_is_leak_safe_across_the_season_boundary() -> None:
    """A future season's team-season row (even an extreme one for a team
    that already appears in-panel) must never change an earlier season's
    already-computed FLAG classification."""

    baseline = special_teams_return_flag_by_game(_schedule(), _team_season())

    future_rows = [*_team_season_rows(), (2027, "TEAMA", -200.0, -200.0)]
    changed = special_teams_return_flag_by_game(_schedule(), _team_season(future_rows))

    flag_columns = ["game_id", "home_return_top_quartile", "away_return_top_quartile"]
    pd.testing.assert_frame_equal(
        changed[flag_columns].reset_index(drop=True),
        baseline[flag_columns].reset_index(drop=True),
        check_exact=True,
    )


def test_flag_never_reads_outcome_columns() -> None:
    """``special_teams_return_flag_by_game`` does not even require/read
    ``result``/``spread_line`` -- adding them (with arbitrary values) and
    mutating them must never change the already-computed flags."""

    schedule = _schedule()
    schedule["result"] = 0.0
    schedule["spread_line"] = -3.0
    baseline = special_teams_return_flag_by_game(schedule, _team_season()).set_index("game_id")

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"].eq("2026_01_TEAMA_OPPX"), "result"] = 999.0
    mutated.loc[mutated["game_id"].eq("2026_01_TEAMA_OPPX"), "spread_line"] = 55.0
    changed = special_teams_return_flag_by_game(mutated, _team_season()).set_index("game_id")

    flag_columns = ["home_return_top_quartile", "away_return_top_quartile"]
    pd.testing.assert_frame_equal(changed[flag_columns], baseline[flag_columns], check_exact=True)


# --- fail-open ---------------------------------------------------------------


def test_fail_open_with_no_special_teams_snapshot_at_all(tmp_path: Path) -> None:
    empty_data_root = tmp_path / "empty_data"
    with pytest.warns(RuntimeWarning, match=CHALLENGER_ID):
        flags = special_teams_return_flag_by_game_fail_open(empty_data_root, _schedule())
    assert flags.empty
    assert list(flags.columns) == [
        "game_id",
        "season",
        "home_return_top_quartile",
        "away_return_top_quartile",
    ]


def test_fail_open_finds_the_latest_snapshot(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    path = latest_special_teams_team_season(data_root)
    assert path is not None
    assert path.name == "team_season.parquet"

    flags = special_teams_return_flag_by_game_fail_open(data_root, _schedule())
    assert not flags.empty


# ---------------------------------------------------------------------------
# 3. apply_special_teams_return_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_onto_the_sole_flagged_side(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_01_TEAMA_OPPX" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_01_TEAMA_OPPX")
    assert flip.flagged_team == "TEAMA"
    assert flip.opponent_team == "OPPX"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMA_OPPX", "home_cover_probability"] == pytest.approx(0.70)


def test_overlay_leaves_a_pick_already_on_the_flagged_team_untouched(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)
    assert all(flip.game_id != "2026_02_OPPY_TEAMD" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_02_OPPY_TEAMD", "home_cover_probability"] == pytest.approx(0.20)


def test_overlay_never_flips_a_mutual_top_quartile_game(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)
    assert all(flip.game_id != "2026_03_TEAMA_TEAMD" for flip in result.flips)
    assert "2026_03_TEAMA_TEAMD" in result.both_flagged_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TEAMA_TEAMD", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_has_no_effect_outside_the_flagged_population(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)
    assert all(flip.game_id != "2026_04_TEAMB_TEAMC" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_04_TEAMB_TEAMC", "home_cover_probability"] == pytest.approx(0.45)


def test_overlay_treats_missing_prior_data_as_no_flip_never_an_error(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)
    assert all(flip.game_id != "2026_05_NEWTEAM_OPPW" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_NEWTEAM_OPPW", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    predictions = pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMA_OPPX"],
            "season": [2026],
            "week": [1],
            "game_type": ["POST"],
            "home_team": ["TEAMA"],
            "away_team": ["OPPX"],
            "kickoff": ["2026-09-24T17:00:00+00:00"],
            "spread_line": [-3.0],
            "home_cover_probability": [0.30],
        }
    )
    result = apply_special_teams_return_tilt_overlay(predictions, _schedule(), data_root)
    assert result.flip_count == 0
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMA_OPPX", "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_disabled_is_a_no_op(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    predictions = _predictions()
    result = apply_special_teams_return_tilt_overlay(
        predictions, _schedule(), data_root, enabled=False
    )
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_fails_open_with_no_special_teams_snapshot(tmp_path: Path) -> None:
    empty_data_root = tmp_path / "empty_data"
    predictions = _predictions()
    with pytest.warns(RuntimeWarning):
        result = apply_special_teams_return_tilt_overlay(predictions, _schedule(), empty_data_root)
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows(tmp_path: Path) -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical."""

    data_root = _write_data_root(tmp_path)
    predictions = _predictions()
    result = apply_special_teams_return_tilt_overlay(predictions, _schedule(), data_root)
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].ne("2026_01_TEAMA_OPPX")
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_special_teams_return_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), tmp_path / "data"
        )


# ---------------------------------------------------------------------------
# 4. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped(tmp_path: Path) -> None:
    data_root = _write_data_root(tmp_path)
    matched_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_04_TEAMB_TEAMC")]
    result = apply_special_teams_return_tilt_overlay(matched_only, _schedule(), data_root)
    assert overlay_disclosure_note(result) == ""

    disabled = apply_special_teams_return_tilt_overlay(
        _predictions(), _schedule(), data_root, enabled=False
    )
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production(
    tmp_path: Path,
) -> None:
    data_root = _write_data_root(tmp_path)
    result = apply_special_teams_return_tilt_overlay(_predictions(), _schedule(), data_root)
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "OPPX -> TEAMA" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 5. record_special_teams_return_tilt_challenger_decisions: dual-tracked
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


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMA_OPPX", "2026_04_TEAMB_TEAMC"],
            "season": [2026, 2026],
            "week": [1, 4],
            "game_type": ["REG", "REG"],
            "home_team": ["TEAMA", "TEAMB"],
            "away_team": ["OPPX", "TEAMC"],
            "kickoff": ["2026-09-24T17:00:00+00:00", "2026-09-24T17:00:00+00:00"],
            "spread_line": [-3.0, 1.0],
            "home_cover_probability": [0.30, 0.45],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 1, "2026-09-17T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def test_record_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    result = record_special_teams_return_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_01_TEAMA_OPPX"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The tilt's own arm diverges from the active model's raw pick
    # (0.30 -> AWAY): the tilt flips it to HOME (TEAMA), the flagged side.
    assert ledger.loc["2026_01_TEAMA_OPPX", "pick_side"] == "HOME"
    # The no-signal game keeps the model's own pick (0.45 -> AWAY).
    assert ledger.loc["2026_04_TEAMB_TEAMC", "pick_side"] == "AWAY"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_special_teams_return_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_challenger_decisions_fails_open_with_no_special_teams_snapshot(
    tmp_path: Path,
) -> None:
    """Recording must still succeed (both games recorded, un-flipped) when
    the special-teams source snapshot is unavailable -- the fail-open
    contract must hold at the recording layer too, not just apply_*."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)

    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = record_special_teams_return_tilt_challenger_decisions(
            artifacts, data_root, now=now
        )

    assert result["recorded"] == 2
    assert result["flip_count"] == 0
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert ledger.loc["2026_01_TEAMA_OPPX", "pick_side"] == "AWAY"


def test_record_challenger_decisions_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_special_teams_return_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_decisions_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion, or a foreign
    # config) since this challenger was pinned -- recording must refuse, not
    # silently switch base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_special_teams_return_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_decisions_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_special_teams_return_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
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
    # This is the SAME shared fingerprint the other three live pick-level
    # tilt overlays (surface_switch, interim_hc_first_game,
    # pbp08_protection_mismatch) are registered against, since they are all
    # pinned to the identical active-model configuration snapshot.
    assert config_fingerprint(_MODEL_CONFIG) == "bc77638d47e2748c"

"""Tests for the turnover-luck rebound tilt overlay (docs/close_game_luck_screen.md,
cell ``turnover_under_rebound`` / registry ``close_game_luck_turnover_under_rebound``).

Four things are load-bearing here, mirroring
``tests/test_interim_hc_first_game_tilt_overlay.py``'s structure and
AGENTS.md's "add a leakage regression test for every new feature family":

1. :func:`turnover_under_flag_by_game` fires ONLY when a team's PRIOR-season
   centered turnover differential is at/below the frozen bottom-quartile
   threshold, is derived from data (ported from
   ``scripts/close_game_luck_screen.py``'s giveaways/team-game/panel
   construction), and never reads the current season's or the target game's
   own outcome.
2. :func:`apply_turnover_luck_rebound_tilt_overlay` flips ONTO the flagged
   team whenever the model's own pick is not already on that side, respects
   the REG-only gate, leaves a both-flagged game untouched, and is
   parameter-free (the threshold is frozen, not tuned).
3. :func:`overlay_disclosure_note` states the flip count and matchups.
4. :func:`record_turnover_luck_rebound_tilt_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, dual-tracked
   and at no rotation-registry window cost, and refuses on a fingerprint
   mismatch or an inactive registration.

Fixture design (see ``_pbp_2025`` / ``_schedule`` below): season 2025 (the
PRIOR season for every 2026 test game) is built so exactly TWO teams,
TEAMA and TEAME, land comfortably below the frozen threshold
(``TURNOVER_UNDER_Q25_THRESHOLD`` = -0.4026832217261905) while every other
team in the fixture lands comfortably above it -- TEAMA/TEAME each give away
the ball repeatedly against a shared two-team opponent pool (OPPX/OPPY) with
zero takeaways of their own, and a separate neutral pair (TEAMB/TEAMC) never
turns the ball over at all. The opponent pool's inflated takeaway rate is
matched in size to TEAMA+TEAME's combined giveaways, which keeps the
season's league mean at exactly 0 -- so the fixture does not accidentally
drag a neutral team across the frozen cutoff the way an unbalanced giveaway/
takeaway fixture would.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    CHALLENGER_ID,
    TURNOVER_UNDER_Q25_THRESHOLD,
    apply_turnover_luck_rebound_tilt_overlay,
    overlay_disclosure_note,
    record_turnover_luck_rebound_tilt_challenger_decisions,
    turnover_under_flag_by_game,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _pbp_row(
    game_id: str, posteam: str, *, interception: float = 0.0, fumble_lost: float = 0.0
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season_type": "REG",
        "posteam": posteam,
        "interception": interception,
        "fumble_lost": fumble_lost,
    }


def _pbp_2025(*, extra_2026_rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """Season 2025 play-by-play: TEAMA and TEAME are heavy prior-season
    giveaway teams (5 and 6 giveaways per game, twice each, zero takeaways);
    OPPX and OPPY absorb all of those takeaways with zero giveaways of their
    own; TEAMB and TEAMC never turn the ball over. See module docstring for
    why this keeps the league mean at 0.
    """

    rows = [
        # 2025_01_TEAMA_OPPX: TEAMA home, gives the ball away 5 times (3 INT + 2 FL).
        *[_pbp_row("2025_01_TEAMA_OPPX", "TEAMA", interception=1.0) for _ in range(3)],
        *[_pbp_row("2025_01_TEAMA_OPPX", "TEAMA", fumble_lost=1.0) for _ in range(2)],
        # 2025_02_OPPY_TEAMA: TEAMA away, gives the ball away 5 more times.
        *[_pbp_row("2025_02_OPPY_TEAMA", "TEAMA", interception=1.0) for _ in range(3)],
        *[_pbp_row("2025_02_OPPY_TEAMA", "TEAMA", fumble_lost=1.0) for _ in range(2)],
        # 2025_03_TEAME_OPPX: TEAME home, gives the ball away 6 times.
        *[_pbp_row("2025_03_TEAME_OPPX", "TEAME", interception=1.0) for _ in range(4)],
        *[_pbp_row("2025_03_TEAME_OPPX", "TEAME", fumble_lost=1.0) for _ in range(2)],
        # 2025_04_OPPY_TEAME: TEAME away, gives the ball away 6 more times.
        *[_pbp_row("2025_04_OPPY_TEAME", "TEAME", interception=1.0) for _ in range(4)],
        *[_pbp_row("2025_04_OPPY_TEAME", "TEAME", fumble_lost=1.0) for _ in range(2)],
        # 2025_05_TEAMB_TEAMC: no turnovers at all (a clean neutral game).
        _pbp_row("2025_05_TEAMB_TEAMC", "TEAMB"),
        _pbp_row("2025_05_TEAMB_TEAMC", "TEAMC"),
    ]
    if extra_2026_rows:
        rows.extend(extra_2026_rows)
    return pd.DataFrame(rows)


def _schedule() -> pd.DataFrame:
    rows_2025 = [
        ("2025_01_TEAMA_OPPX", 2025, 1, "REG", "TEAMA", "OPPX"),
        ("2025_02_OPPY_TEAMA", 2025, 2, "REG", "OPPY", "TEAMA"),
        ("2025_03_TEAME_OPPX", 2025, 3, "REG", "TEAME", "OPPX"),
        ("2025_04_OPPY_TEAME", 2025, 4, "REG", "OPPY", "TEAME"),
        ("2025_05_TEAMB_TEAMC", 2025, 5, "REG", "TEAMB", "TEAMC"),
    ]
    rows_2026 = [
        # G-flip: TEAMD (home, no 2025 history) hosts TEAMA (away, flagged).
        ("2026_01_TEAMD_TEAMA", 2026, 1, "REG", "TEAMD", "TEAMA"),
        # G-neutral: neither TEAMB nor TEAMC is flagged.
        ("2026_01_TEAMB_TEAMC", 2026, 1, "REG", "TEAMB", "TEAMC"),
        # G-both: TEAMA (home) vs TEAME (away) -- both flagged.
        ("2026_01_TEAMA_TEAME", 2026, 1, "REG", "TEAMA", "TEAME"),
        # G-already: OPPX (home) vs TEAMA (away) -- flagged side already picked.
        ("2026_01_OPPX_TEAMA", 2026, 1, "REG", "OPPX", "TEAMA"),
    ]
    columns = ["game_id", "season", "week", "game_type", "home_team", "away_team"]
    return pd.DataFrame(rows_2025 + rows_2026, columns=columns)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_01_TEAMD_TEAMA",
                "2026_01_TEAMB_TEAMC",
                "2026_01_TEAMA_TEAME",
                "2026_01_OPPX_TEAMA",
            ],
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "home_team": ["TEAMD", "TEAMB", "TEAMA", "OPPX"],
            "away_team": ["TEAMA", "TEAMC", "TEAME", "TEAMA"],
            "kickoff": ["2026-09-13T17:00:00+00:00"] * 4,
            "spread_line": [-2.5, 1.0, -3.0, -6.0],
            # G-flip: model picks HOME (TEAMD) -- TEAMA (away, flagged) is not
            #   the pick -- should flip to AWAY.
            # G-neutral: model picks HOME (TEAMB) -- neither side flagged --
            #   should not move regardless of the model's own probability.
            # G-both: model picks HOME (TEAMA) -- both TEAMA and TEAME are
            #   flagged -- mutual case, must not flip.
            # G-already: model already has AWAY (TEAMA, the flagged team) --
            #   already on the flagged side, must not flip.
            "home_cover_probability": [0.58, 0.55, 0.65, 0.30],
        }
    )


# ---------------------------------------------------------------------------
# 1. turnover_under_flag_by_game: derived, pregame-safe
# ---------------------------------------------------------------------------


def test_the_frozen_threshold_matches_the_screens_measured_value() -> None:
    assert pytest.approx(-0.4026832217261905) == TURNOVER_UNDER_Q25_THRESHOLD


def test_flag_fires_on_the_bottom_quartile_prior_season_teams() -> None:
    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    row = flags.set_index("game_id").loc["2026_01_TEAMD_TEAMA"]
    assert bool(row["home_turnover_under_flag"]) is False
    assert bool(row["away_turnover_under_flag"]) is True


def test_flag_reproduces_the_screens_quartile_cut_on_this_fixture() -> None:
    """TEAMA's prior-season centered turnover differential is -5 (well below
    the frozen -0.4027 cutoff); OPPX's is +5.5, comfortably above it -- so
    the SAME arithmetic the screen uses (giveaways/takeaways per game, minus
    the season's own mean) reproduces the expected flag split on this
    fixture, not just a hand-picked boolean."""

    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    both = flags.set_index("game_id").loc["2026_01_TEAMA_TEAME"]
    assert bool(both["home_turnover_under_flag"]) is True  # TEAMA
    assert bool(both["away_turnover_under_flag"]) is True  # TEAME

    neutral = flags.set_index("game_id").loc["2026_01_TEAMB_TEAMC"]
    assert bool(neutral["home_turnover_under_flag"]) is False  # TEAMB
    assert bool(neutral["away_turnover_under_flag"]) is False  # TEAMC


def test_flag_is_false_when_the_team_has_no_prior_season_data() -> None:
    """TEAMD never appears in the 2025 fixture at all -- missing prior data
    must resolve to False, never raise."""

    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    row = flags.set_index("game_id").loc["2026_01_TEAMD_TEAMA"]
    assert bool(row["home_turnover_under_flag"]) is False


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="schedules is missing columns"):
        turnover_under_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _pbp_2025())


def test_flag_requires_its_play_by_play_columns() -> None:
    with pytest.raises(DataContractError, match="play-by-play is missing columns"):
        turnover_under_flag_by_game(_schedule(), pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 1b. Leakage regression: prior-season only, never current-season/current-game
# ---------------------------------------------------------------------------


def test_flag_is_unchanged_by_the_current_seasons_own_turnover_events() -> None:
    """A leakage regression test proving the trait uses only the PRIOR
    season: adding a pile of 2026 turnover events for TEAMA in ITS OWN
    target game (interceptions thrown in 2026_01_TEAMD_TEAMA) must not move
    the flag at all -- the function never even loads current-season
    play-by-play into the panel the flag is looked up from."""

    baseline = turnover_under_flag_by_game(_schedule(), _pbp_2025())

    leaky_extra = [
        _pbp_row("2026_01_TEAMD_TEAMA", "TEAMA", interception=1.0) for _ in range(20)
    ] + [_pbp_row("2026_01_TEAMD_TEAMA", "TEAMD", fumble_lost=1.0) for _ in range(20)]
    with_current_season = turnover_under_flag_by_game(
        _schedule(), _pbp_2025(extra_2026_rows=leaky_extra)
    )

    pd.testing.assert_frame_equal(
        baseline.sort_values("game_id").reset_index(drop=True),
        with_current_season.sort_values("game_id").reset_index(drop=True),
    )


def test_flag_is_unchanged_by_a_flipped_outcome_in_the_current_seasons_other_games() -> None:
    """A second current-season game (not the target game) getting a pile of
    NEW turnover events must also leave every 2026 flag untouched -- proving
    the panel this function reads from never incorporates ANY current-season
    play, not just the target game's own."""

    baseline = turnover_under_flag_by_game(_schedule(), _pbp_2025())

    leaky_extra = [_pbp_row("2026_01_TEAMB_TEAMC", "TEAMB", interception=1.0) for _ in range(10)]
    mutated = turnover_under_flag_by_game(_schedule(), _pbp_2025(extra_2026_rows=leaky_extra))

    pd.testing.assert_frame_equal(
        baseline.sort_values("game_id").reset_index(drop=True),
        mutated.sort_values("game_id").reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 2. apply_turnover_luck_rebound_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_onto_the_flagged_team_when_not_already_picked() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_01_TEAMD_TEAMA" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_01_TEAMD_TEAMA")
    assert flip.flagged_team == "TEAMA"
    assert flip.original_pick_team == "TEAMD"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMD_TEAMA", "home_cover_probability"] == pytest.approx(0.42)


def test_overlay_leaves_a_neutral_game_untouched() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_TEAMB_TEAMC" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMB_TEAMC", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_never_flips_a_both_flagged_game() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_TEAMA_TEAME" for flip in result.flips)
    assert "2026_01_TEAMA_TEAME" in result.both_flagged_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMA_TEAME", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_leaves_a_pick_already_on_the_flagged_team_untouched() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_OPPX_TEAMA" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_OPPX_TEAMA", "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_disabled_is_a_no_op() -> None:
    schedule = _schedule()
    predictions = _predictions()
    result = apply_turnover_luck_rebound_tilt_overlay(
        predictions, schedule, _pbp_2025(), enabled=False
    )
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    schedule = _schedule()
    predictions = _predictions()
    result = apply_turnover_luck_rebound_tilt_overlay(predictions, schedule, _pbp_2025())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(
        ["2026_01_TEAMB_TEAMC", "2026_01_TEAMA_TEAME", "2026_01_OPPX_TEAMA"]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason() -> None:
    schedule = _schedule()
    predictions = pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMD_TEAMA"],
            "season": [2026],
            "week": [1],
            "game_type": ["POST"],
            "home_team": ["TEAMD"],
            "away_team": ["TEAMA"],
            "kickoff": ["2026-09-13T17:00:00+00:00"],
            "spread_line": [-2.5],
            "home_cover_probability": [0.58],
        }
    )
    result = apply_turnover_luck_rebound_tilt_overlay(predictions, schedule, _pbp_2025())
    assert result.flip_count == 0
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMD_TEAMA", "home_cover_probability"] == pytest.approx(0.58)


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_turnover_luck_rebound_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _pbp_2025()
        )


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    schedule = _schedule()
    matched_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_01_TEAMB_TEAMC")]
    result = apply_turnover_luck_rebound_tilt_overlay(matched_only, schedule, _pbp_2025())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_turnover_luck_rebound_tilt_overlay(
        _predictions(), schedule, _pbp_2025(), enabled=False
    )
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "TEAMD -> TEAMA" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_turnover_luck_rebound_tilt_challenger_decisions: dual-tracked
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "feature_set": "full_weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMD_TEAMA", "2026_01_TEAMB_TEAMC"],
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "home_team": ["TEAMD", "TEAMB"],
            "away_team": ["TEAMA", "TEAMC"],
            "kickoff": ["2026-09-13T17:00:00+00:00", "2026-09-13T17:00:00+00:00"],
            "spread_line": [-2.5, 1.0],
            "home_cover_probability": [0.58, 0.55],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 1, "2026-09-08T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def _full_pbp_2025() -> pd.DataFrame:
    """The same 2025 turnover events as :func:`_pbp_2025`, widened to every
    column ``nfl_ats.pbp.PBP_REQUIRED_COLUMNS`` demands so it can round-trip
    through :func:`nfl_ats.pbp.write_pbp_snapshot` -> ``canonicalize_pbp`` ->
    ``validate_pbp`` -- the recorder loads its play-by-play from a real
    snapshot on disk, unlike the flag-level tests above which pass a
    DataFrame straight to :func:`turnover_under_flag_by_game`."""

    schedule_2025 = _schedule().loc[lambda f: f["season"].eq(2025)].set_index("game_id")
    rows: list[dict[str, object]] = []
    for play_id, record in enumerate(_pbp_2025().to_dict("records"), start=1):
        game_id = str(record["game_id"])
        home_team = schedule_2025.loc[game_id, "home_team"]
        away_team = schedule_2025.loc[game_id, "away_team"]
        posteam = record["posteam"]
        defteam = away_team if posteam == home_team else home_team
        rows.append(
            {
                "play_id": play_id,
                "game_id": game_id,
                "season": 2025,
                "season_type": "REG",
                "week": int(schedule_2025.loc[game_id, "week"]),
                "home_team": home_team,
                "away_team": away_team,
                "posteam": posteam,
                "defteam": defteam,
                "fixed_drive": 1,
                "down": 1,
                "play_type": "pass",
                "yards_gained": 0,
                "pass_attempt": 1,
                "rush_attempt": 0,
                "sack": 0,
                "qb_hit": 0,
                "epa": 0.0,
                "success": 0,
                "wp": 0.5,
                "interception": record["interception"],
                "fumble_lost": record["fumble_lost"],
            }
        )
    return pd.DataFrame(rows)


def _write_data_root(tmp_path: Path) -> Path:
    """Write a schedules snapshot (nfl_ats.snapshots) and a play-by-play
    snapshot (nfl_ats.pbp) under ``<tmp>/data``, as the recorder expects."""

    from nfl_ats.pbp import write_pbp_snapshot
    from nfl_ats.snapshots import write_snapshot

    data_root = tmp_path / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )
    write_pbp_snapshot({2025: _full_pbp_2025()}, data_root / "pbp" / "raw")
    return data_root


def test_record_turnover_luck_rebound_challenger_decisions_records_the_tilt_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    result = record_turnover_luck_rebound_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_01_TEAMD_TEAMA"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The tilt's own arm diverges from the active model's raw pick (0.58 ->
    # HOME): the tilt flips it to AWAY (TEAMA), onto the flagged team.
    assert ledger.loc["2026_01_TEAMD_TEAMA", "pick_side"] == "AWAY"
    # The neutral game keeps the model's own pick (0.55 -> HOME).
    assert ledger.loc["2026_01_TEAMB_TEAMC", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_turnover_luck_rebound_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_turnover_luck_rebound_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 1, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_turnover_luck_rebound_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_turnover_luck_rebound_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
        )


def test_turnover_luck_rebound_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
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

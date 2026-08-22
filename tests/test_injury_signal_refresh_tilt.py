from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.four_overlay_composition import POLICY_FINGERPRINT, POLICY_ID
from nfl_ats.injury_signal_refresh_tilt import (
    DISAGREEMENT_BOTH_AGREE,
    DISAGREEMENT_BOTH_DISAGREE,
    DISAGREEMENT_INJURY_ONLY,
    DISAGREEMENT_MOVEMENT_ONLY,
    DISAGREEMENT_NEITHER,
    INJURY_NET_THRESHOLD,
    INJURY_SIGNAL_LEDGER_COLUMNS,
    PFT_NET_THRESHOLD,
    SOURCE_NONE,
    SOURCE_OFFICIAL,
    SOURCE_PFT_FALLBACK,
    build_injury_signal_rows,
    classify_disagreement,
    injury_signal_for_game,
    injury_signal_ledger_path,
    load_injury_signal_decisions,
    own_week_tuesday_noon_utc,
    record_injury_signal_refresh_tilt,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MODEL_ONLY,
    MOVEMENT_POLICY_MOVEMENT,
    RefreshedGame,
    RefreshResult,
)

# Thursday-night kickoff used throughout -- own-week Tuesday noon ET falls on
# 2026-09-15 (September is EDT, UTC-4), i.e. 2026-09-15T16:00:00Z.
KICKOFF = pd.Timestamp("2026-09-18T00:15:00+00:00")
TUESDAY_NOON = own_week_tuesday_noon_utc(pd.Series([KICKOFF])).iloc[0]
BEFORE_TUESDAY = TUESDAY_NOON - pd.Timedelta(days=1)
WED_MORNING = TUESDAY_NOON + pd.Timedelta(hours=20)
FRIDAY = TUESDAY_NOON + pd.Timedelta(days=3)
SEASON = 2026
WEEK = 2


def _injury_row(
    *,
    team: str,
    gsis_id: str,
    report_status: str,
    date_modified: pd.Timestamp,
    position: str = "WR",
) -> dict[str, object]:
    return {
        "season": SEASON,
        "week": WEEK,
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "report_status": report_status,
        "date_modified": date_modified,
    }


def _injuries_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


def _pft_frame(rows: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["headline_norm", "lastmod"])
    frame["lastmod"] = pd.to_datetime(frame["lastmod"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# 1. Official-path trigger logic
# ---------------------------------------------------------------------------


def test_official_path_fires_on_a_brand_new_post_tuesday_designation() -> None:
    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    assert reading.source == SOURCE_OFFICIAL
    assert reading.net_score == pytest.approx(4.0)
    assert reading.threshold == INJURY_NET_THRESHOLD
    assert reading.fires is True


def test_official_path_does_not_fire_below_the_predeclared_threshold() -> None:
    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Probable", date_modified=WED_MORNING)]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    assert reading.net_score == pytest.approx(1.0)
    assert reading.fires is False


def test_official_path_nets_against_the_opponents_own_injury_news() -> None:
    injuries = _injuries_frame(
        [
            _injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING),
            _injury_row(
                team="MOV", gsis_id="p2", report_status="Doubtful", date_modified=WED_MORNING
            ),
        ]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    # 4 (Out) - 3 (Doubtful) = 1, below the >=2 bar.
    assert reading.net_score == pytest.approx(1.0)
    assert reading.fires is False


def test_official_path_a_preexisting_designation_with_no_update_contributes_zero() -> None:
    """A player already on the Tuesday-noon report, with no NEWER row, is not
    double-counted as fresh news -- the outer-merge finds the SAME row on
    both sides of the cutoff."""

    injuries = _injuries_frame(
        [
            _injury_row(
                team="TST", gsis_id="p1", report_status="Questionable", date_modified=BEFORE_TUESDAY
            )
        ]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    assert reading.net_score == pytest.approx(0.0)
    assert reading.fires is False


def test_official_path_a_genuine_midweek_worsening_is_captured() -> None:
    """Same player, two rows: Probable before Tuesday noon, Doubtful after --
    the as-of-cutoff logic must pick up the LATER row for the "now" read and
    the EARLIER row for the Tuesday read."""

    injuries = _injuries_frame(
        [
            _injury_row(
                team="TST", gsis_id="p1", report_status="Probable", date_modified=BEFORE_TUESDAY
            ),
            _injury_row(
                team="TST", gsis_id="p1", report_status="Doubtful", date_modified=WED_MORNING
            ),
        ]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    assert reading.net_score == pytest.approx(2.0)  # 3 (Doubtful) - 1 (Probable)
    assert reading.fires is True


def test_official_path_signal_is_zero_before_any_post_tuesday_filing_lands() -> None:
    """A refresh pass run AT Tuesday noon itself (before Wed-Fri filings
    exist) must read zero signal -- the front-running sketch's core premise:
    quiet until real news lands, never a false positive from the frozen
    baseline itself."""

    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=TUESDAY_NOON,
        injuries=injuries,
        pft=None,
    )
    assert reading.net_score == pytest.approx(0.0)
    assert reading.fires is False


# ---------------------------------------------------------------------------
# 2. PFT-headline fallback
# ---------------------------------------------------------------------------


def test_pft_fallback_used_when_season_has_no_official_coverage() -> None:
    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    injuries["season"] = 2019  # a season this game's season (2026) never matches
    pft = _pft_frame(
        [
            ("Cowboys WR questionable for Sunday", WED_MORNING),
            ("Cowboys WR ruled out", FRIDAY - pd.Timedelta(hours=1)),
        ]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="DAL",
        opponent_team="NYG",
        now=FRIDAY,
        injuries=injuries,
        pft=pft,
    )
    assert reading.source == SOURCE_PFT_FALLBACK
    assert reading.net_score == pytest.approx(2.0)
    assert reading.threshold == PFT_NET_THRESHOLD
    assert reading.fires is True


def test_pft_fallback_excludes_headlines_outside_the_tuesday_to_now_window() -> None:
    injuries = None
    pft = _pft_frame(
        [
            ("Cowboys news before Tuesday", BEFORE_TUESDAY),  # excluded: too early
            ("Cowboys news after now", FRIDAY + pd.Timedelta(days=2)),  # excluded: too late
        ]
    )
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="DAL",
        opponent_team="NYG",
        now=FRIDAY,
        injuries=injuries,
        pft=pft,
    )
    assert reading.net_score == pytest.approx(0.0)
    assert reading.fires is False


# ---------------------------------------------------------------------------
# 3. Fail-open
# ---------------------------------------------------------------------------


def test_fails_open_to_no_signal_with_no_injury_data_at_all() -> None:
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=None,
        pft=None,
    )
    assert reading.source == SOURCE_NONE
    assert reading.net_score == 0.0
    assert reading.fires is False


def test_fails_open_when_official_frame_present_but_season_and_pft_both_absent() -> None:
    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    injuries["season"] = 2019
    reading = injury_signal_for_game(
        game_id="g1",
        season=SEASON,
        week=WEEK,
        kickoff=KICKOFF,
        picked_team="TST",
        opponent_team="MOV",
        now=FRIDAY,
        injuries=injuries,
        pft=None,
    )
    assert reading.source == SOURCE_NONE
    assert reading.fires is False


# ---------------------------------------------------------------------------
# 4. Disagreement classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injury_fires,injury_side,movement_policy,movement_side,expected",
    [
        (True, "AWAY", MOVEMENT_POLICY_MODEL_ONLY, "", DISAGREEMENT_INJURY_ONLY),
        (True, "AWAY", MOVEMENT_POLICY_MOVEMENT, "AWAY", DISAGREEMENT_BOTH_AGREE),
        (True, "AWAY", MOVEMENT_POLICY_MOVEMENT, "HOME", DISAGREEMENT_BOTH_DISAGREE),
        (False, "HOME", MOVEMENT_POLICY_MOVEMENT, "AWAY", DISAGREEMENT_MOVEMENT_ONLY),
        (False, "HOME", MOVEMENT_POLICY_MODEL_ONLY, "", DISAGREEMENT_NEITHER),
    ],
)
def test_classify_disagreement_covers_every_branch(
    injury_fires: bool, injury_side: str, movement_policy: str, movement_side: str, expected: str
) -> None:
    assert (
        classify_disagreement(
            injury_fires=injury_fires,
            injury_tilt_pick_side=injury_side,
            movement_policy=movement_policy,
            movement_pick_side=movement_side,
        )
        == expected
    )


# ---------------------------------------------------------------------------
# 5. build_injury_signal_rows: both arms, disagreement metadata, eligibility
# ---------------------------------------------------------------------------


def _game(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    model_only_pick_side: str,
    movement_policy: str,
    movement_pick_side: str,
    eligible: bool = True,
    new_pick_side: str | None = None,
) -> RefreshedGame:
    return RefreshedGame(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        kickoff=KICKOFF,
        deadline=KICKOFF,
        decision_home_spread=-2.5,
        original_recorded_at_utc=BEFORE_TUESDAY,
        previous_pick_side=model_only_pick_side,
        previous_home_cover_probability=None,
        new_pick_side=new_pick_side or model_only_pick_side,
        new_home_cover_probability=0.55 if model_only_pick_side == "HOME" else 0.45,
        decision_policy_id=POLICY_ID,
        decision_policy_fingerprint=POLICY_FINGERPRINT,
        coach_fade_flip=False,
        division_revenge_flip=False,
        player_arrests_flip=False,
        spread_gap_zone_flip=False,
        composed_overlay_flip=False,
        player_arrests_snapshot_id="snapshot-tuesday",
        player_arrests_safe_index_sha256="safe-index-sha",
        movement_policy=movement_policy,
        movement_delta=None if movement_policy == MOVEMENT_POLICY_MODEL_ONLY else 1.5,
        movement_pick_side=movement_pick_side,
        model_only_pick_side=model_only_pick_side,
        eligible=eligible,
        ineligible_reason="" if eligible else "kickoff_passed",
        changed=False,
    )


def _plan(games: tuple[RefreshedGame, ...], *, now: pd.Timestamp = FRIDAY) -> RefreshResult:
    return RefreshResult(
        season=SEASON,
        week=WEEK,
        refresh_run_id="20260918T120000Z",
        computed_at_utc=now,
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=games,
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )


def test_build_injury_signal_rows_records_both_arms_and_excludes_ineligible_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: injuries,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )

    fires_game = _game(
        game_id="g_fires",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    quiet_game = _game(
        game_id="g_quiet",
        home_team="QUI",
        away_team="ETT",
        model_only_pick_side="AWAY",
        movement_policy=MOVEMENT_POLICY_MOVEMENT,
        movement_pick_side="AWAY",
        new_pick_side="AWAY",
    )
    ineligible_game = _game(
        game_id="g_ineligible",
        home_team="XXX",
        away_team="YYY",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
        eligible=False,
    )
    plan = _plan((fires_game, quiet_game, ineligible_game))

    rows = build_injury_signal_rows(plan, data_root=Path("unused"))

    assert set(rows["game_id"]) == {"g_fires", "g_quiet"}
    assert list(rows.columns) == list(INJURY_SIGNAL_LEDGER_COLUMNS)

    fired = rows.set_index("game_id").loc["g_fires"]
    assert fired["hold_pick_side"] == "HOME"
    assert fired["injury_tilt_pick_side"] == "AWAY"  # flipped: TST's own injury news
    assert bool(fired["injury_signal_fires"]) is True
    assert fired["disagreement_type"] == DISAGREEMENT_INJURY_ONLY

    quiet = rows.set_index("game_id").loc["g_quiet"]
    assert quiet["hold_pick_side"] == "AWAY"
    assert quiet["injury_tilt_pick_side"] == "AWAY"  # unchanged: no injury data for QUI/ETT
    assert bool(quiet["injury_signal_fires"]) is False
    assert quiet["disagreement_type"] == DISAGREEMENT_MOVEMENT_ONLY
    assert quiet["played_pick_side"] == "AWAY"


def test_build_injury_signal_rows_is_empty_when_no_games_are_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: (_ for _ in ()).throw(AssertionError("must not load data")),
    )
    plan = _plan(())
    rows = build_injury_signal_rows(plan, data_root=Path("unused"))
    assert rows.empty
    assert list(rows.columns) == list(INJURY_SIGNAL_LEDGER_COLUMNS)


def test_build_injury_signal_rows_fails_open_with_no_data_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: None,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )
    game = _game(
        game_id="g1",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    plan = _plan((game,))
    rows = build_injury_signal_rows(plan, data_root=Path("unused"))
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["injury_signal_source"] == SOURCE_NONE
    assert bool(row["injury_signal_fires"]) is False
    assert row["hold_pick_side"] == row["injury_tilt_pick_side"] == "HOME"
    assert row["disagreement_type"] == DISAGREEMENT_NEITHER


# ---------------------------------------------------------------------------
# 6. record_injury_signal_refresh_tilt: the append-only ledger write
# ---------------------------------------------------------------------------


def _write_original_card(
    artifacts_root: Path, kickoff: pd.Timestamp, *, recorded_at: pd.Timestamp
) -> None:
    frame = pd.DataFrame(
        {
            "recorded_at_utc": [recorded_at],
            "forecast_artifact": ["margin_predictions/test"],
            "forecast_created_at_utc": [recorded_at],
            "model_id": ["model-1"],
            "method": ["market_residual"],
            "decision_policy_id": [POLICY_ID],
            "decision_policy_fingerprint": [POLICY_FINGERPRINT],
            "game_id": ["g1"],
            "season": [SEASON],
            "week": [WEEK],
            "kickoff": [kickoff],
            "away_team": ["MOV"],
            "home_team": ["TST"],
            "model_pick_side": ["HOME"],
            "pre_arrest_pick_side": ["HOME"],
            "former_policy_pick_side": ["HOME"],
            "pick_side": ["HOME"],
            "coach_fade_flip": [False],
            "division_revenge_flip": [False],
            "player_arrests_flip": [False],
            "spread_gap_zone_flip": [False],
            "composed_overlay_flip": [False],
            "player_arrests_home_flag": [False],
            "player_arrests_away_flag": [False],
            "player_arrests_snapshot_id": [""],
            "player_arrests_snapshot_fetched_at_utc": [pd.NaT],
            "player_arrests_safe_index_sha256": [""],
            "schedule_snapshot_id": ["schedule-tuesday"],
            "schedule_parquet_sha256": ["schedule-hash"],
            "bet_side": ["PASS"],
            "decision_home_spread": [-2.5],
            "edge": [float("nan")],
            "is_best_pick": [False],
        }
    )
    frame["recorded_at_utc"] = pd.to_datetime(frame["recorded_at_utc"], utc=True)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame["forecast_created_at_utc"] = pd.to_datetime(frame["forecast_created_at_utc"], utc=True)
    atomic_parquet(frame[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root))


def test_record_skips_the_ledger_write_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root, KICKOFF, recorded_at=BEFORE_TUESDAY)
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: None,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )
    game = _game(
        game_id="g1",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    plan = _plan((game,))
    result = record_injury_signal_refresh_tilt(
        artifacts_root, data_root, plan, record_decisions=False
    )
    assert result == {
        "recorded": 0,
        "skipped": True,
        "reason": (
            "pass --record-decisions to append this pass's injury-signal reading to "
            "the injury-signal refresh ledger"
        ),
    }
    assert not injury_signal_ledger_path(artifacts_root).is_file()


def test_record_appends_both_arms_and_disagreement_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root, KICKOFF, recorded_at=BEFORE_TUESDAY)

    injuries = _injuries_frame(
        [_injury_row(team="TST", gsis_id="p1", report_status="Out", date_modified=WED_MORNING)]
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: injuries,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )
    game = _game(
        game_id="g1",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    plan = _plan((game,), now=WED_MORNING + pd.Timedelta(hours=1))
    result = record_injury_signal_refresh_tilt(
        artifacts_root, data_root, plan, record_decisions=True
    )

    assert result["recorded"] == 1
    assert result["ledger_rows"] == 1
    assert result["injury_signal_fired_game_ids"] == ["g1"]
    assert result["disagreement_game_ids"] == ["g1"]

    ledger = load_injury_signal_decisions(artifacts_root)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["hold_pick_side"] == "HOME"
    assert row["injury_tilt_pick_side"] == "AWAY"
    assert bool(row["injury_signal_fires"]) is True
    assert row["injury_signal_source"] == SOURCE_OFFICIAL
    assert row["disagreement_type"] == DISAGREEMENT_INJURY_ONLY
    assert row["model_id"] == "model-1"
    assert row["feature_table_sha256"] == "feature-sha"


def test_record_is_append_only_across_multiple_refresh_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated passes across a week legitimately append MULTIPLE rows per
    game -- deliberately not deduped (see module docstring)."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root, KICKOFF, recorded_at=BEFORE_TUESDAY)
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: None,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )
    game = _game(
        game_id="g1",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    first_pass = _plan((game,), now=WED_MORNING)
    record_injury_signal_refresh_tilt(artifacts_root, data_root, first_pass, record_decisions=True)

    second_pass = _plan((game,), now=FRIDAY)
    result = record_injury_signal_refresh_tilt(
        artifacts_root, data_root, second_pass, record_decisions=True
    )
    assert result["recorded"] == 1
    assert result["ledger_rows"] == 2

    ledger = load_injury_signal_decisions(artifacts_root)
    assert len(ledger) == 2
    assert set(pd.to_datetime(ledger["revision_recorded_at_utc"], utc=True)) == {
        WED_MORNING,
        FRIDAY,
    }


def test_record_refuses_far_ahead_of_kickoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    far_future_kickoff = pd.Timestamp("2026-12-01T18:00:00+00:00")
    _write_original_card(artifacts_root, far_future_kickoff, recorded_at=BEFORE_TUESDAY)
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_official_injuries_fail_open",
        lambda data_root: None,
    )
    monkeypatch.setattr(
        "nfl_ats.injury_signal_refresh_tilt._latest_pft_index_fail_open",
        lambda data_root: None,
    )
    game = _game(
        game_id="g1",
        home_team="TST",
        away_team="MOV",
        model_only_pick_side="HOME",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_pick_side="",
    )
    plan = _plan((game,), now=BEFORE_TUESDAY)
    with pytest.raises(ValueError, match="injury-signal-refresh"):
        record_injury_signal_refresh_tilt(artifacts_root, data_root, plan, record_decisions=True)
    assert not injury_signal_ledger_path(artifacts_root).is_file()


def test_load_injury_signal_decisions_returns_empty_frame_when_absent(tmp_path: Path) -> None:
    ledger = load_injury_signal_decisions(tmp_path)
    assert ledger.empty
    assert list(ledger.columns) == list(INJURY_SIGNAL_LEDGER_COLUMNS)


def test_load_injury_signal_decisions_raises_on_missing_columns(tmp_path: Path) -> None:
    bad = pd.DataFrame({"game_id": ["g1"]})
    atomic_parquet(bad, injury_signal_ledger_path(tmp_path))
    from nfl_ats.data import DataContractError

    with pytest.raises(DataContractError):
        load_injury_signal_decisions(tmp_path)

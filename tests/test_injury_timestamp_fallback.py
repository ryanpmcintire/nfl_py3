"""ENG-39: leakage-safe timestamp fallback for undated injury revisions.

nflverse's 2025 injuries release drops ``date_modified`` entirely (measured,
docs/injury_timestamp_fallback.md M1), and the historical (default) response
in ``nfl_ats.players.canonicalize_injuries`` -- drop any row without one --
silently zeroes the whole ``home_/away_/diff_injury_*`` feature block for
every 2025+ game (M3). This file pins:

* the default ``timestamp_fallback="drop"`` path stays byte-identical (a
  hash pin, so any accidental change to it fails loudly here);
* the opt-in ``"week_proxy"`` fallback never makes a proxied row visible
  before its own proxy time (the leakage invariant AGENTS.md requires for
  every new feature family);
* the proxy is clamped to never precede the Tuesday that starts the row's
  own NFL week;
* a real ``date_modified`` is never overwritten by the fallback;
* a 2025-shaped frame with no ``date_modified`` column at all survives
  ``"week_proxy"`` (and still raises under the default ``"drop"``);
* ``nfl_ats.prediction_safety``'s new ``injury_feature_presence`` check
  fails a prospective card whose injury feature block is entirely
  null/zero, and passes a healthy one.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from test_players import _games, _pbp, _rosters, _snaps

from nfl_ats.backtest import score_week
from nfl_ats.data import DataContractError
from nfl_ats.outcomes import OUTCOME_METHODS, score_outcome_week
from nfl_ats.players import canonicalize_injuries, enrich_with_player_features
from nfl_ats.prediction_safety import (
    PredictionSafetyError,
    validate_outcome_prediction_card,
    validate_prediction_card,
)

# ---------------------------------------------------------------------------
# Default "drop" mode: byte-identical to the pre-ENG-39 behaviour
# ---------------------------------------------------------------------------


def _hash_pin_fixture() -> pd.DataFrame:
    """A small multi-season, multi-revision, deliberately out-of-order frame.

    Spans 2011 and 2024 (named in the plan this file implements) with a
    duplicate row, an out-of-order revision, and multiple teams/players, so
    the sort/dedup logic in ``canonicalize_injuries`` is actually exercised
    rather than pinning a trivial single-row identity.
    """

    rows: list[dict[str, object]] = []
    for season, week, team, gsis_id, revisions in (
        (
            2011,
            1,
            "A",
            "P1",
            [("Questionable", "2011-09-08T20:00:00Z"), ("Out", "2011-09-06T12:00:00Z")],
        ),
        (2011, 1, "B", "P2", [("Doubtful", "2011-09-07T15:30:00Z")]),
        (2024, 17, "C", "P3", [("Questionable", "2024-12-27T10:00:00Z")] * 2),  # exact duplicate
        (2024, 17, "D", "P4", [("Out", "2024-12-28T18:45:00Z")]),
    ):
        for status, timestamp in revisions:
            rows.append(
                {
                    "season": season,
                    "game_type": "REG",
                    "team": team,
                    "week": week,
                    "gsis_id": gsis_id,
                    "position": "WR",
                    "report_status": status,
                    "practice_status": "Limited Participation in Practice",
                    "date_modified": timestamp,
                }
            )
    # Reversed insertion order: the output must be determined by the sort
    # key, not by row order in the source frame.
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def test_week_proxy_default_drop_mode_is_byte_identical_to_pre_eng39() -> None:
    fixture = _hash_pin_fixture()

    default_result = canonicalize_injuries(fixture)
    explicit_result = canonicalize_injuries(fixture, timestamp_fallback="drop")
    pd.testing.assert_frame_equal(default_result, explicit_result)

    # No new columns in "drop" mode -- exactly the pre-ENG-39 schema.
    assert "effective_observed_at" not in default_result.columns
    assert "observed_at_basis" not in default_result.columns

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(default_result, index=True).to_numpy().tobytes()
    ).hexdigest()
    assert digest == "f4495befd1961cc9556bee59efef564f09d7e62d68a102bce10c64aac5043d3d"


# ---------------------------------------------------------------------------
# week_proxy: schema-tolerant, leakage-safe, never overwrites a real revision
# ---------------------------------------------------------------------------


def _injury_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2024,
        "game_type": "REG",
        "team": "A",
        "week": 2,
        "gsis_id": "QB-A",
        "position": "QB",
        "report_status": "Questionable",
        "practice_status": "Limited Participation in Practice",
        "date_modified": pd.NaT,
    }
    row.update(overrides)
    return row


def _schedule_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2024,
        "week": 2,
        "home_team": "A",
        "away_team": "B",
        "kickoff": pd.Timestamp("2024-09-15T17:00:00Z"),
    }
    row.update(overrides)
    return row


def test_week_proxy_clamps_to_the_games_own_week_tuesday_including_a_thursday_game() -> None:
    # A Thursday game: kickoff Thu 2024-09-12 20:15 ET = 2024-09-13T00:15Z.
    # kickoff-24h lands Wednesday, still after that week's Tuesday floor --
    # this only pins the floor is computed correctly for a Thursday game,
    # not that it clamps (a Thursday game structurally never can: kickoff
    # minus 24h cannot precede a Tuesday two calendar days earlier).
    thursday_kickoff = pd.Timestamp("2024-09-13T00:15:00Z")
    schedule = pd.DataFrame([_schedule_row(kickoff=thursday_kickoff)])
    result = canonicalize_injuries(
        pd.DataFrame([_injury_row()]), timestamp_fallback="week_proxy", schedule=schedule
    )
    naive_proxy = thursday_kickoff - pd.Timedelta(hours=24)
    tuesday_floor = pd.Timestamp("2024-09-10T04:00:00Z")  # Tue 2024-09-10 00:00 ET (EDT, UTC-4)
    assert naive_proxy > tuesday_floor  # sanity: this case does not need the clamp
    assert result.loc[0, "effective_observed_at"] == naive_proxy
    assert result.loc[0, "observed_at_basis"] == "week_proxy"

    # An adversarial synthetic Tuesday kickoff: kickoff-24h (Monday evening)
    # DOES precede that week's own Tuesday floor, so the clamp must engage.
    tuesday_kickoff = pd.Timestamp("2024-09-10T22:00:00Z")  # Tue 2024-09-10 18:00 ET
    schedule2 = pd.DataFrame([_schedule_row(kickoff=tuesday_kickoff)])
    result2 = canonicalize_injuries(
        pd.DataFrame([_injury_row()]), timestamp_fallback="week_proxy", schedule=schedule2
    )
    naive_proxy2 = tuesday_kickoff - pd.Timedelta(hours=24)
    tuesday_floor2 = pd.Timestamp("2024-09-10T04:00:00Z")
    assert naive_proxy2 < tuesday_floor2  # would precede the floor unclamped
    assert result2.loc[0, "effective_observed_at"] == tuesday_floor2
    assert result2.loc[0, "effective_observed_at"] < tuesday_kickoff


def test_week_proxy_never_overwrites_a_real_date_modified() -> None:
    kickoff = pd.Timestamp("2024-09-15T17:00:00Z")
    schedule = pd.DataFrame([_schedule_row(kickoff=kickoff)])
    real_timestamp = "2024-09-14T12:00:00Z"
    injuries = pd.DataFrame(
        [
            _injury_row(gsis_id="QB-A", date_modified=real_timestamp),
            _injury_row(gsis_id="WR-A", date_modified=pd.NaT),
        ]
    )
    result = canonicalize_injuries(injuries, timestamp_fallback="week_proxy", schedule=schedule)

    real_row = result.loc[result["gsis_id"].eq("QB-A")].iloc[0]
    proxy_row = result.loc[result["gsis_id"].eq("WR-A")].iloc[0]
    assert real_row["effective_observed_at"] == pd.Timestamp(real_timestamp)
    assert real_row["observed_at_basis"] == "date_modified"
    assert proxy_row["observed_at_basis"] == "week_proxy"
    assert proxy_row["effective_observed_at"] == kickoff - pd.Timedelta(hours=24)


def test_week_proxy_survives_a_2025_shaped_frame_with_no_date_modified_column() -> None:
    kickoff = pd.Timestamp("2025-09-14T17:00:00Z")
    schedule = pd.DataFrame([_schedule_row(season=2025, kickoff=kickoff)])
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_type": "REG",
                "team": "A",
                "week": 2,
                "gsis_id": "QB-A",
                "position": "QB",
                "report_status": "Questionable",
                "practice_status": "Limited Participation in Practice",
                # No "date_modified" column at all -- the real 2025 nflverse shape.
            }
        ]
    )
    assert "date_modified" not in injuries.columns

    result = canonicalize_injuries(injuries, timestamp_fallback="week_proxy", schedule=schedule)
    assert len(result) == 1
    assert result.loc[0, "observed_at_basis"] == "week_proxy"
    assert result.loc[0, "effective_observed_at"] == kickoff - pd.Timedelta(hours=24)

    # The default mode still requires a real date_modified column -- this is
    # exactly the production failure mode (M1), reproduced and pinned.
    with pytest.raises(DataContractError):
        canonicalize_injuries(injuries)


def test_week_proxy_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="timestamp_fallback"):
        canonicalize_injuries(pd.DataFrame([_injury_row()]), timestamp_fallback="invented")
    with pytest.raises(ValueError, match="requires a schedule"):
        canonicalize_injuries(pd.DataFrame([_injury_row()]), timestamp_fallback="week_proxy")


# ---------------------------------------------------------------------------
# Idempotency: re-canonicalizing an already-canonical (week_proxy) frame
#
# THE GAP this closes (measured by lane S, docs/injury_timestamp_fallback.md):
# a feature-build step reads a snapshot's own injuries.parquet back off disk
# -- already carrying effective_observed_at/observed_at_basis from the
# week_proxy fallback applied once at ingest -- and calls
# canonicalize_injuries on it a SECOND time with the function's own default
# ("drop"). The pre-fix "drop" branch re-derives visibility from the
# still-null date_modified column and silently drops every proxied row,
# even though the caller never asked for "drop" mode to override anything;
# it simply never passed a fallback at all. These tests pin that a frame
# already carrying the snapshot's basis survives re-canonicalization
# regardless of the argument, while a frame that never had that basis
# (every pre-ENG-39 snapshot) is completely unaffected.
# ---------------------------------------------------------------------------


def _week_proxy_snapshot_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A 2025-shaped injuries frame (no ``date_modified`` column) plus one
    row with a real revision, canonicalized once with ``"week_proxy"`` --
    i.e. exactly what a snapshot's own ``injuries.parquet`` looks like on
    disk after ``player-ingest --timestamp-fallback week_proxy``.
    """

    kickoff = pd.Timestamp("2024-09-15T17:00:00Z")
    schedule = pd.DataFrame([_schedule_row(kickoff=kickoff)])
    injuries = pd.DataFrame(
        [
            _injury_row(gsis_id="QB-A", date_modified="2024-09-14T12:00:00Z"),
            _injury_row(gsis_id="WR-A", date_modified=pd.NaT),
        ]
    )
    once_canonicalized = canonicalize_injuries(
        injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    return once_canonicalized, schedule


def test_canonicalize_injuries_default_mode_is_idempotent_on_a_week_proxy_snapshot() -> None:
    once_canonicalized, _schedule = _week_proxy_snapshot_fixture()
    assert len(once_canonicalized) == 2  # sanity: both rows survived the first pass

    # The exact call the production feature-build steps make: no
    # timestamp_fallback argument at all, so the function's own "drop"
    # default applies -- and no schedule, since none was passed either.
    recanonicalized = canonicalize_injuries(once_canonicalized)

    assert len(recanonicalized) == 2  # both rows, including the proxied one, survive
    pd.testing.assert_frame_equal(
        recanonicalized.reset_index(drop=True), once_canonicalized.reset_index(drop=True)
    )
    proxy_row = recanonicalized.loc[recanonicalized["gsis_id"].eq("WR-A")].iloc[0]
    real_row = recanonicalized.loc[recanonicalized["gsis_id"].eq("QB-A")].iloc[0]
    assert proxy_row["observed_at_basis"] == "week_proxy"
    assert real_row["observed_at_basis"] == "date_modified"
    assert real_row["effective_observed_at"] == pd.Timestamp("2024-09-14T12:00:00Z")


def test_canonicalize_injuries_week_proxy_snapshot_ignores_the_argument_and_needs_no_schedule() -> (
    None
):
    once_canonicalized, _schedule = _week_proxy_snapshot_fixture()

    # Explicitly requesting "week_proxy" with schedule=None would normally
    # raise ("requires a schedule frame") -- but since the frame already
    # carries the snapshot's own basis, no schedule is needed and the
    # request is honoured from the existing columns instead.
    recanonicalized = canonicalize_injuries(once_canonicalized, timestamp_fallback="week_proxy")
    pd.testing.assert_frame_equal(
        recanonicalized.reset_index(drop=True), once_canonicalized.reset_index(drop=True)
    )


def test_canonicalize_injuries_records_snapshot_basis_provenance_in_attrs() -> None:
    once_canonicalized, _schedule = _week_proxy_snapshot_fixture()
    recanonicalized = canonicalize_injuries(once_canonicalized)
    provenance = recanonicalized.attrs.get("injury_timestamp_basis_provenance")
    assert provenance is not None
    assert provenance["source"] == "snapshot"
    assert provenance["requested_timestamp_fallback"] == "drop"


def test_canonicalize_injuries_plain_snapshot_re_canonicalization_is_unchanged() -> None:
    """A frame with no basis columns (every pre-ENG-39 snapshot) is untouched
    by the idempotency branch -- re-canonicalizing it is a true no-op, and
    it never gains the new columns.
    """

    fixture = _hash_pin_fixture()
    once = canonicalize_injuries(fixture)
    assert "effective_observed_at" not in once.columns
    twice = canonicalize_injuries(once)
    pd.testing.assert_frame_equal(once, twice)


def test_enrich_with_player_features_default_mode_survives_a_week_proxy_snapshot() -> None:
    """The exact end-to-end gap lane S measured: a feature-build call that
    never passes ``injury_timestamp_fallback`` (production's own call sites,
    pre-fix) must still see a proxied row, because the *snapshot itself*
    (not this call) already carries the fallback's basis.
    """

    games = _games()
    kickoff_week2 = pd.Timestamp(games.loc[games["week"].eq(2), "kickoff"].iloc[0])
    schedule = games.loc[:, ["season", "week", "home_team", "away_team", "kickoff"]]
    raw_injuries = pd.DataFrame([_injury_row(season=2022, week=2, date_modified=pd.NaT)])
    snapshot_injuries = canonicalize_injuries(
        raw_injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    assert snapshot_injuries.loc[0, "observed_at_basis"] == "week_proxy"

    enriched = enrich_with_player_features(
        games,
        snapshot_injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        decision_hours_before_kickoff=1,
        # No injury_timestamp_fallback passed -- this is the default "drop"
        # production used before this fix, on an already-week_proxy'd
        # snapshot frame.
    )
    row = enriched.loc[enriched["week"].eq(2)].iloc[0]
    assert row["home_injury_offense_unavailability"] > 0
    assert row["home_injury_observed_at"] == kickoff_week2 - pd.Timedelta(hours=24)
    assert row["home_injury_observed_at_basis"] == "week_proxy"
    assert row["home_injury_observed_at_is_proxy"]
    assert row["home_injury_proxy_row_count"] == 1
    assert {"season": 2022, "week": 2, "proxy_rows": 1} in enriched.attrs["injury_proxy_provenance"]


def test_qb_availability_canonicalization_is_idempotent_on_a_week_proxy_snapshot() -> None:
    """Mirrors the players.py idempotency fix for
    ``nfl_ats.quarterbacks._canonicalize_qb_availability``, the equivalent
    re-canonicalization site on the named-QB availability path.
    """

    from nfl_ats.quarterbacks import _canonicalize_qb_availability

    kickoff = pd.Timestamp("2024-09-15T17:00:00Z")
    schedule = pd.DataFrame([_schedule_row(kickoff=kickoff)])
    injuries = pd.DataFrame(
        [
            _injury_row(gsis_id="QB-A", date_modified="2024-09-14T12:00:00Z"),
            _injury_row(gsis_id="WR-A", date_modified=pd.NaT),
        ]
    )
    once = _canonicalize_qb_availability(
        injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    assert len(once) == 2

    # Default mode, no schedule: mirrors the players.py case exactly.
    twice = _canonicalize_qb_availability(once)
    assert len(twice) == 2
    pd.testing.assert_frame_equal(twice.reset_index(drop=True), once.reset_index(drop=True))
    proxy_row = twice.loc[twice["gsis_id"].eq("WR-A")].iloc[0]
    assert proxy_row["observed_at_basis"] == "week_proxy"


# ---------------------------------------------------------------------------
# Leakage pin: a proxied row is invisible before its own proxy time
# ---------------------------------------------------------------------------


def test_week_proxy_proxied_row_is_invisible_before_its_own_proxy_time() -> None:
    """AGENTS.md: a proxied row must never be visible before its proxy time.

    Team A's week-2 injury has no real ``date_modified``, so under
    ``timestamp_fallback="week_proxy"`` it is proxied to that game's own
    kickoff minus 24h. A decision cutoff 48h before kickoff sits BEFORE that
    proxy time and must see nothing; a cutoff 1h before kickoff sits AFTER
    it and must see the report.
    """

    games = _games()
    kickoff_week2 = pd.Timestamp(games.loc[games["week"].eq(2), "kickoff"].iloc[0])
    proxy_at = kickoff_week2 - pd.Timedelta(hours=24)

    injuries = pd.DataFrame([_injury_row(season=2022, week=2, date_modified=pd.NaT)])

    too_early = enrich_with_player_features(
        games,
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        injury_timestamp_fallback="week_proxy",
        decision_hours_before_kickoff=48,
    )
    row_too_early = too_early.loc[too_early["week"].eq(2)].iloc[0]
    assert pd.isna(row_too_early["home_injury_offense_unavailability"])
    assert pd.isna(row_too_early["home_injury_observed_at"])
    assert pd.isna(row_too_early["home_injury_observed_at_basis"])
    assert not row_too_early["home_injury_observed_at_is_proxy"]
    assert row_too_early["home_injury_proxy_row_count"] == 0

    visible = enrich_with_player_features(
        games,
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        injury_timestamp_fallback="week_proxy",
        decision_hours_before_kickoff=1,
    )
    row_visible = visible.loc[visible["week"].eq(2)].iloc[0]
    assert row_visible["home_injury_offense_unavailability"] > 0
    assert row_visible["home_injury_observed_at"] == proxy_at
    assert row_visible["home_injury_observed_at_basis"] == "week_proxy"

    # The default "drop" mode never sees this row at all, at either cutoff --
    # only the opt-in fallback unlocks it.
    still_blind = enrich_with_player_features(
        games,
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        decision_hours_before_kickoff=1,
    )
    row_blind = still_blind.loc[still_blind["week"].eq(2)].iloc[0]
    assert pd.isna(row_blind["home_injury_offense_unavailability"])
    assert pd.isna(row_blind["home_injury_observed_at"])


# ---------------------------------------------------------------------------
# prediction_safety: injury_feature_presence
# ---------------------------------------------------------------------------

_INJURY_COLUMNS = (
    "home_injury_offense_unavailability",
    "away_injury_offense_unavailability",
    "diff_injury_offense_unavailability",
)


def test_outcome_card_injury_feature_presence_fails_all_zero_and_passes_healthy(
    model_frame: pd.DataFrame,
) -> None:
    predictions = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)

    zeroed = predictions.copy()
    for column in _INJURY_COLUMNS:
        zeroed[column] = 0.0

    with pytest.raises(PredictionSafetyError, match="injury_feature_presence"):
        validate_outcome_prediction_card(
            zeroed,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
            expected_season=2020,
            expected_week=1,
            prospective=True,
        )

    # A non-prospective (e.g. backtest) call never runs the check -- the
    # exact same all-zero card passes without it.
    non_prospective_audit = validate_outcome_prediction_card(
        zeroed,
        min_edge=0.02,
        expected_methods=OUTCOME_METHODS,
        expected_season=2020,
        expected_week=1,
    )
    assert "injury_feature_presence" not in non_prospective_audit.checks_passed

    # The explicit escape hatch suppresses the failure but still records it.
    allowed_audit = validate_outcome_prediction_card(
        zeroed,
        min_edge=0.02,
        expected_methods=OUTCOME_METHODS,
        expected_season=2020,
        expected_week=1,
        prospective=True,
        allow_empty_injury_block=True,
    )
    assert "injury_feature_presence" in allowed_audit.checks_passed
    assert any("allow_empty_injury_block" in warning for warning in allowed_audit.warnings)

    healthy = predictions.copy()
    home_values = np.linspace(0.05, 0.35, len(healthy))
    away_values = np.linspace(0.35, 0.05, len(healthy))
    healthy["home_injury_offense_unavailability"] = home_values
    healthy["away_injury_offense_unavailability"] = away_values
    healthy["diff_injury_offense_unavailability"] = home_values - away_values
    healthy_audit = validate_outcome_prediction_card(
        healthy,
        min_edge=0.02,
        expected_methods=OUTCOME_METHODS,
        expected_season=2020,
        expected_week=1,
        prospective=True,
    )
    assert "injury_feature_presence" in healthy_audit.checks_passed
    assert healthy_audit.status == "PASS"


def test_direct_ats_card_injury_feature_presence_fails_all_zero_and_passes_healthy(
    model_frame: pd.DataFrame,
) -> None:
    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    predictions["kickoff"] = pd.to_datetime(predictions["gameday"], utc=True) + pd.Timedelta(
        hours=12
    )
    for column in ("home_cover", "result", "ats_margin"):
        predictions[column] = np.nan
    created_at = datetime(2018, 12, 9, tzinfo=UTC)

    zeroed = predictions.copy()
    zeroed["diff_injury_offense_unavailability"] = 0.0

    with pytest.raises(PredictionSafetyError, match="injury_feature_presence"):
        validate_prediction_card(
            zeroed,
            min_edge=0.02,
            prospective=True,
            created_at=created_at,
            feature_columns=["diff_injury_offense_unavailability"],
        )

    # margin-predict never sets allow_empty_injury_block; a non-prospective
    # call (e.g. a research backtest) never runs the check at all.
    non_prospective_audit = validate_prediction_card(
        zeroed,
        min_edge=0.02,
        feature_columns=["diff_injury_offense_unavailability"],
    )
    assert "injury_feature_presence" not in non_prospective_audit.checks_passed

    allowed_audit = validate_prediction_card(
        zeroed,
        min_edge=0.02,
        prospective=True,
        created_at=created_at,
        feature_columns=["diff_injury_offense_unavailability"],
        allow_empty_injury_block=True,
    )
    assert "injury_feature_presence" in allowed_audit.checks_passed

    healthy = zeroed.copy()
    healthy["diff_injury_offense_unavailability"] = np.linspace(-0.3, 0.3, len(healthy))
    healthy_audit = validate_prediction_card(
        healthy,
        min_edge=0.02,
        prospective=True,
        created_at=created_at,
        feature_columns=["diff_injury_offense_unavailability"],
    )
    assert "injury_feature_presence" in healthy_audit.checks_passed


def test_real_revision_replaces_existing_proxy_and_hides_future_report():
    games = _games()
    kickoff = pd.Timestamp(games.loc[games.week.eq(2), "kickoff"].iloc[0])
    snapshot = canonicalize_injuries(
        pd.DataFrame([_injury_row(season=2022, week=2, date_modified=pd.NaT)]),
        timestamp_fallback="week_proxy",
        schedule=games,
    )
    snapshot["date_modified"] = kickoff - pd.Timedelta(minutes=30)
    canonical = canonicalize_injuries(snapshot)
    assert canonical.effective_observed_at.iloc[0] == snapshot.date_modified.iloc[0]
    assert canonical.observed_at_basis.iloc[0] == "date_modified"
    assert not canonical.observed_at_is_proxy.iloc[0]
    enriched = enrich_with_player_features(
        games,
        snapshot,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        decision_hours_before_kickoff=1,
    )
    row = enriched.loc[enriched.week.eq(2)].iloc[0]
    assert pd.isna(row.home_injury_observed_at)
    assert row.home_injury_proxy_row_count == 0


def test_availability_real_revision_wins_over_preexisting_proxy():
    from nfl_ats.availability import build_availability_outcomes

    games = _games()
    kickoff = pd.Timestamp(games.loc[games.week.eq(2), "kickoff"].iloc[0])
    snapshot = canonicalize_injuries(
        pd.DataFrame([_injury_row(season=2022, week=2, date_modified=pd.NaT)]),
        timestamp_fallback="week_proxy",
        schedule=games,
    )
    snaps = pd.DataFrame(
        [
            {
                "season": 2022,
                "week": 2,
                "team": "A",
                "gsis_id": "QB-A",
                "offense_snaps": 1,
                "defense_snaps": 0,
                "st_snaps": 0,
            }
        ]
    )
    proxy_outcomes = build_availability_outcomes(snapshot, snaps, games)
    assert len(proxy_outcomes) == 1
    assert proxy_outcomes.observed_at_is_proxy.iloc[0]
    snapshot["date_modified"] = kickoff - pd.Timedelta(hours=1)
    assert build_availability_outcomes(snapshot, snaps, games).empty
    real_outcomes = build_availability_outcomes(
        snapshot, snaps, games, decision_hours_before_kickoff=0
    )
    assert len(real_outcomes) == 1
    assert not real_outcomes.observed_at_is_proxy.iloc[0]


def test_proxy_lineage_counts_all_contributors_when_latest_revision_is_real():
    games = _games()
    kickoff = pd.Timestamp(games.loc[games.week.eq(2), "kickoff"].iloc[0])
    injuries = pd.DataFrame(
        [
            _injury_row(season=2022, week=2, date_modified=pd.NaT),
            _injury_row(
                season=2022,
                week=2,
                gsis_id="WR-A",
                position="WR",
                date_modified=kickoff - pd.Timedelta(hours=2),
            ),
        ]
    )
    enriched = enrich_with_player_features(
        games,
        injuries,
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        decision_hours_before_kickoff=1,
        injury_timestamp_fallback="week_proxy",
    )
    row = enriched.loc[enriched.week.eq(2)].iloc[0]
    assert row.home_injury_observed_at_basis == "date_modified"
    assert row.home_injury_observed_at_is_proxy
    assert row.home_injury_proxy_row_count == 1

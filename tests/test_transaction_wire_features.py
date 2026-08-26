from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.transaction_wire_features import (
    ALL_CATEGORIES,
    OTHER_CATEGORY,
    TRANSACTION_CATEGORIES,
    attach_transaction_counts,
    build_team_week_population,
    canonical_team,
    classify_transaction_slug,
    explode_dated_transactions,
    kickoff_utc,
    match_transaction_teams,
    own_week_wednesday_freeze_utc,
)

# ---------------------------------------------------------------------------
# 1. Slug classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug,expected",
    [
        ("rams-activate-s-quentin-lake-from-ir", "ir_activation"),
        ("eagles-activate-lb-from-injured-reserve", "ir_activation"),
        ("packers-place-te-on-injured-reserve", "ir_placement"),
        ("jets-place-wr-on-ir", "ir_placement"),
        ("cowboys-elevate-rb-from-practice-squad", "practice_squad_elevation"),
        ("49ers-elevated-two-players-for-sunday", "practice_squad_elevation"),
        ("bears-claim-cb-off-waivers", "waiver_claim"),
        ("titans-release-veteran-ol", "release"),
        ("browns-waived-de-tuesday", "release"),
        ("bills-cut-rb", "release"),
        ("bills-trade-te-lee-smith-to-falcons", "trade"),
        ("49ers-acquire-stevie-johnson-bills", "trade"),
        ("watson-suspension-latest", "suspension"),
        ("patriots-to-sign-nate-washington", "signing"),
        ("falcons-extend-smith-dimitroff-mckay", "signing"),
        ("steelers-extend-troy-polamalus-contract", "signing"),
        ("minor-nfl-transactions-9-23-15", OTHER_CATEGORY),
        ("legarrette-blount-free-agent", OTHER_CATEGORY),
    ],
)
def test_classify_transaction_slug(slug: str, expected: str) -> None:
    assert classify_transaction_slug(slug) == expected


def test_every_category_is_declared() -> None:
    assert set(TRANSACTION_CATEGORIES) == {
        "ir_activation",
        "ir_placement",
        "practice_squad_elevation",
        "waiver_claim",
        "release",
        "trade",
        "suspension",
        "signing",
    }
    assert set(ALL_CATEGORIES) == set(TRANSACTION_CATEGORIES) | {OTHER_CATEGORY}


# ---------------------------------------------------------------------------
# 2. Team-nickname matching
# ---------------------------------------------------------------------------


def test_match_transaction_teams_single_team() -> None:
    assert match_transaction_teams("eagles-extend-jason-peters") == {"PHI"}


def test_match_transaction_teams_zero_teams_for_roundups() -> None:
    assert match_transaction_teams("minor-nfl-transactions-9-23-15") == set()


def test_match_transaction_teams_two_teams_for_a_trade() -> None:
    assert match_transaction_teams("bills-trade-te-lee-smith-to-falcons") == {"BUF", "ATL"}


def test_match_transaction_teams_multi_word_nickname() -> None:
    assert match_transaction_teams("commanders-sign-free-agent-cb") == {"WAS"}
    assert match_transaction_teams("washington-football-team-sign-cb") == {"WAS"}


def test_match_transaction_teams_no_false_positive_substring() -> None:
    # "cardinals" must not fire on an unrelated token that merely contains
    # similar letters; only a whole hyphen-delimited token counts.
    assert match_transaction_teams("chargers-sign-rb") == {"LAC"}
    assert "TEN" not in match_transaction_teams("patriots-sign-te")


def test_canonical_team_maps_relocated_codes() -> None:
    assert canonical_team("OAK") == "LV"
    assert canonical_team("SD") == "LAC"
    assert canonical_team("STL") == "LA"
    assert canonical_team("PHI") == "PHI"


# ---------------------------------------------------------------------------
# 3. Cutoff construction
# ---------------------------------------------------------------------------


def test_own_week_wednesday_freeze_is_the_same_calendar_week_wednesday_noon() -> None:
    # One kickoff per weekday, Wednesday through Monday (2026-09-16 is a
    # Wednesday, verified against the real 2026 calendar), all EDT.
    kickoffs = pd.Series(
        pd.to_datetime(
            [
                "2026-09-16T17:00:00Z",  # Wednesday
                "2026-09-17T00:15:00Z",  # Thursday night (gameday Wed 9/16 in ET... see below)
                "2026-09-20T17:00:00Z",  # Sunday early
                "2026-09-22T00:15:00Z",  # Monday night (ET gameday still Mon 9/21)
            ]
        ),
        dtype="datetime64[ns, UTC]",
    )
    freeze = own_week_wednesday_freeze_utc(kickoffs)
    expected_wednesday_noon = pd.Timestamp("2026-09-16T16:00:00Z")  # noon EDT = 16:00 UTC
    for value in freeze:
        assert value == expected_wednesday_noon
    # Freeze must precede kickoff for every game actually in this week.
    for k in kickoffs:
        assert freeze.iloc[0] <= k


def test_own_week_wednesday_freeze_for_a_tuesday_kickoff_is_the_prior_week() -> None:
    """Documented edge case (see the function's own docstring): a Tuesday
    kickoff's own calendar week's Wednesday has not happened yet, so the
    MOST-RECENT-Wednesday-at-or-before-kickoff definition correctly resolves
    to the PRIOR week -- not a bug, just worth pinning explicitly."""

    tuesday_kickoff = pd.Series(
        pd.to_datetime(["2026-09-15T17:00:00Z"]), dtype="datetime64[ns, UTC]"
    )
    freeze = own_week_wednesday_freeze_utc(tuesday_kickoff)
    assert freeze.iloc[0] == pd.Timestamp("2026-09-09T16:00:00Z")
    assert freeze.iloc[0] < tuesday_kickoff.iloc[0]


def test_kickoff_utc_combines_gameday_and_eastern_gametime() -> None:
    games = pd.DataFrame({"gameday": ["2026-09-17"], "gametime": ["20:15"]})
    result = kickoff_utc(games)
    # September is EDT (UTC-4): 20:15 ET -> 00:15 UTC next day.
    assert result.iloc[0] == pd.Timestamp("2026-09-18T00:15:00Z")


# ---------------------------------------------------------------------------
# 4. Team-week population construction
# ---------------------------------------------------------------------------


def _schedules_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_02_PHI_DAL"],
            "game_type": ["REG"],
            "season": [2026],
            "week": [2],
            "gameday": ["2026-09-17"],
            "gametime": ["20:15"],
            "home_team": ["DAL"],
            "away_team": ["PHI"],
        }
    )


def test_build_team_week_population_has_one_row_per_side() -> None:
    panel = build_team_week_population(_schedules_frame(), season_start=2026, season_end=2026)
    assert len(panel) == 2
    assert set(panel["team"]) == {"DAL", "PHI"}
    home_row = panel.loc[panel["team"] == "DAL"].iloc[0]
    away_row = panel.loc[panel["team"] == "PHI"].iloc[0]
    assert bool(home_row["is_home"]) is True
    assert bool(away_row["is_home"]) is False
    assert home_row["opponent"] == "PHI"
    assert away_row["opponent"] == "DAL"
    assert home_row["kickoff_utc"] == pd.Timestamp("2026-09-18T00:15:00Z")
    assert home_row["window72_start_utc"] == home_row["kickoff_utc"] - pd.Timedelta(hours=72)
    assert home_row["freeze_utc"] < home_row["kickoff_utc"]


def test_build_team_week_population_canonicalizes_relocated_codes() -> None:
    schedules = _schedules_frame()
    schedules["home_team"] = "OAK"
    schedules["away_team"] = "SD"
    panel = build_team_week_population(schedules, season_start=2026, season_end=2026)
    assert set(panel["team"]) == {"LV", "LAC"}


def test_build_team_week_population_filters_to_season_range_and_reg() -> None:
    schedules = pd.concat(
        [
            _schedules_frame(),
            _schedules_frame().assign(game_type="WC", game_id="2026_20_PHI_DAL"),
            _schedules_frame().assign(season=2020, game_id="2020_02_PHI_DAL"),
        ],
        ignore_index=True,
    )
    panel = build_team_week_population(schedules, season_start=2026, season_end=2026)
    assert len(panel) == 2  # only the REG 2026 game survives


# ---------------------------------------------------------------------------
# 5. Leakage regression test: nothing published at/after kickoff can count
# ---------------------------------------------------------------------------


def _dated(rows: list[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["slug", "precise_ts"])
    frame["precise_ts"] = pd.to_datetime(frame["precise_ts"], utc=True)
    return frame


def test_leakage_transaction_published_after_kickoff_is_never_counted() -> None:
    panel = build_team_week_population(_schedules_frame(), season_start=2026, season_end=2026)
    kickoff = panel.loc[panel["team"] == "DAL", "kickoff_utc"].iloc[0]

    dated = _dated(
        [
            # Before kickoff -- a real practice-squad elevation for the home
            # team, published Saturday, well inside the 72h window.
            ("cowboys-elevate-rb-from-practice-squad", str(kickoff - pd.Timedelta(hours=20))),
            # AT kickoff exactly -- must be excluded (right bound is strict).
            ("cowboys-elevate-wr-from-practice-squad", str(kickoff)),
            # AFTER kickoff -- a postgame transaction story that must never
            # leak into a pregame feature.
            ("cowboys-place-lb-on-injured-reserve", str(kickoff + pd.Timedelta(hours=2))),
        ]
    )
    exploded = explode_dated_transactions(dated)
    scored = attach_transaction_counts(panel, exploded)

    dal_row = scored.loc[scored["team"] == "DAL"].iloc[0]
    assert dal_row["n_events_72h"] == 1
    assert dal_row["n_practice_squad_elevation_72h"] == 1
    assert dal_row["n_ir_placement_72h"] == 0
    assert dal_row["n_events_since_freeze"] == 1
    assert dal_row["n_ir_placement_since_freeze"] == 0


def _sunday_schedules_frame() -> pd.DataFrame:
    # A Sunday-afternoon kickoff puts Wed-noon freeze ~4.5 days (108h) before
    # kickoff -- comfortably outside the 72h-before-kickoff window, unlike a
    # Thursday game where the two windows overlap (disclosed separately).
    return pd.DataFrame(
        {
            "game_id": ["2026_02_PHI_DAL"],
            "game_type": ["REG"],
            "season": [2026],
            "week": [2],
            "gameday": ["2026-09-20"],  # a Sunday
            "gametime": ["13:00"],
            "home_team": ["DAL"],
            "away_team": ["PHI"],
        }
    )


def test_leakage_transaction_before_freeze_excluded_from_since_freeze_but_may_count_in_72h() -> (
    None
):
    panel = build_team_week_population(
        _sunday_schedules_frame(), season_start=2026, season_end=2026
    )
    row = panel.loc[panel["team"] == "DAL"].iloc[0]
    kickoff = row["kickoff_utc"]
    freeze = row["freeze_utc"]
    assert freeze < kickoff - pd.Timedelta(hours=72)  # Wed noon well before Sun-72h in this fixture

    dated = _dated(
        [
            # Exactly at the freeze instant -- excluded (left bound strict).
            ("cowboys-sign-rb", str(freeze)),
            # Just after the freeze but outside the 72h-before-kickoff window.
            ("cowboys-sign-wr", str(freeze + pd.Timedelta(hours=1))),
        ]
    )
    exploded = explode_dated_transactions(dated)
    scored = attach_transaction_counts(panel, exploded)
    dal_row = scored.loc[scored["team"] == "DAL"].iloc[0]
    assert dal_row["n_events_since_freeze"] == 1  # only the post-freeze one
    assert dal_row["n_signing_since_freeze"] == 1
    assert dal_row["n_events_72h"] == 0  # neither falls in the 72h-before-kickoff window


def test_thursday_kickoff_freeze_instant_falls_inside_the_72h_window() -> None:
    """Disclosed, not a bug: for a Thursday game, Wed-noon freeze is only
    ~32h before kickoff -- INSIDE the 72h-before-kickoff window, so the two
    features overlap for Thursday games specifically (unlike Sun/Mon games,
    where they do not). A signing right after freeze counts in BOTH windows."""

    panel = build_team_week_population(_schedules_frame(), season_start=2026, season_end=2026)
    row = panel.loc[panel["team"] == "DAL"].iloc[0]  # _schedules_frame is a Thursday game
    assert row["freeze_utc"] > row["window72_start_utc"]

    dated = _dated([("cowboys-sign-rb", str(row["freeze_utc"] + pd.Timedelta(hours=1)))])
    exploded = explode_dated_transactions(dated)
    scored = attach_transaction_counts(panel, exploded)
    dal_row = scored.loc[scored["team"] == "DAL"].iloc[0]
    assert dal_row["n_events_since_freeze"] == 1
    assert dal_row["n_events_72h"] == 1


def test_leakage_bulk_random_events_never_leak_across_kickoff() -> None:
    """A larger randomized check: for many synthetic events straddling many
    team-week kickoffs, the counted set (by direct recomputation) always
    equals the set with ``precise_ts < kickoff_utc`` -- never includes a
    post-kickoff timestamp."""

    # Real team codes/nicknames (DAL/"cowboys") -- a fake code like "TST" would
    # never match any TEAM_NICKNAMES entry and every event would silently be
    # dropped by explode_dated_transactions before this test could exercise
    # the leakage boundary at all.
    schedules = pd.DataFrame(
        {
            "game_id": [f"2026_{w:02d}_DAL_PHI" for w in range(1, 6)],
            "game_type": ["REG"] * 5,
            "season": [2026] * 5,
            "week": list(range(1, 6)),
            "gameday": ["2026-09-10", "2026-09-17", "2026-09-24", "2026-10-01", "2026-10-08"],
            "gametime": ["20:15"] * 5,
            "home_team": ["DAL"] * 5,
            "away_team": ["PHI"] * 5,
        }
    )
    panel = build_team_week_population(schedules, season_start=2026, season_end=2026)
    dal_panel = panel.loc[panel["team"] == "DAL"].reset_index(drop=True)

    rng = np.random.default_rng(20260826)
    base = pd.Timestamp("2026-09-01T00:00:00Z")
    n_events = 200
    offsets_days = rng.uniform(0, 45, size=n_events)
    event_ts = base + pd.to_timedelta(offsets_days, unit="D")
    dated = pd.DataFrame(
        {
            "slug": ["cowboys-sign-rb"] * n_events,
            "precise_ts": event_ts,
        }
    )
    dated["precise_ts"] = pd.to_datetime(dated["precise_ts"], utc=True)
    exploded = explode_dated_transactions(dated)
    scored = attach_transaction_counts(dal_panel, exploded)

    # Naive UTC datetime64[ns] throughout, matching attach_transaction_counts'
    # own internal representation, so every comparison below is dtype-safe.
    event_ts_arr = (
        dated["precise_ts"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    kickoffs = (
        scored["kickoff_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    freezes = (
        scored["freeze_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    window72_starts = (
        scored["window72_start_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    saw_a_post_kickoff_event = bool((event_ts_arr >= kickoffs.min()).any())
    assert saw_a_post_kickoff_event  # sanity: the random draw actually exercised the boundary

    for i in range(len(scored)):
        kickoff = kickoffs[i]
        expected_since_freeze = int(((event_ts_arr > freezes[i]) & (event_ts_arr < kickoff)).sum())
        expected_72h = int(((event_ts_arr > window72_starts[i]) & (event_ts_arr < kickoff)).sum())
        assert scored["n_events_since_freeze"].iloc[i] == expected_since_freeze
        assert scored["n_events_72h"].iloc[i] == expected_72h

        # Directly re-derive the count using ONLY strictly-pregame events and
        # confirm it is unchanged by however many post-kickoff events exist --
        # the leakage property this test exists to pin.
        pregame_only = event_ts_arr[event_ts_arr < kickoff]
        recount = int((pregame_only > freezes[i]).sum())
        assert scored["n_events_since_freeze"].iloc[i] == recount


# ---------------------------------------------------------------------------
# 6. explode_dated_transactions
# ---------------------------------------------------------------------------


def test_explode_dated_transactions_drops_zero_team_rows_and_splits_trades() -> None:
    dated = _dated(
        [
            ("eagles-extend-jason-peters", "2026-09-01T12:00:00Z"),
            ("minor-nfl-transactions-9-23-15", "2026-09-01T12:00:00Z"),
            ("bills-trade-te-lee-smith-to-falcons", "2026-09-01T12:00:00Z"),
        ]
    )
    exploded = explode_dated_transactions(dated)
    assert set(exploded["slug"]) == {
        "eagles-extend-jason-peters",
        "bills-trade-te-lee-smith-to-falcons",
    }
    trade_rows = exploded.loc[exploded["slug"] == "bills-trade-te-lee-smith-to-falcons"]
    assert set(trade_rows["team"]) == {"BUF", "ATL"}
    assert len(trade_rows) == 2

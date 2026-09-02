"""Leakage, additivity and join contracts for the on-production Reddit columns.

Predeclared in ``docs/reddit_attention_on_production.md``. Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no ``data/raw/arctic_shift`` fetch, no processed parquet).

The leakage claim under test is the frozen battery's own: the attention window
ends on the **Tuesday of the game's own week**, and the trailing baseline that
normalizes it is ``shift(1)``-ed before the rolling window, so neither can see
a day at or after kickoff. It is pinned in BOTH directions -- a spike inside
the window must reach the column, and the identical spike moved past kickoff
must not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.reddit_attention_production_feature import (
    REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS,
    REDDIT_AWAY_SPIKE_COLUMN,
    REDDIT_HOME_RATIO_ELEVATED_COLUMN,
    attach_reddit_attention_features,
    derive_reddit_attention_features,
)

# Ten consecutive Sunday game days in one season. The trailing baseline needs
# TRAILING_MIN_GAMES=2 strictly prior games (shift(1) then min_periods=2), so
# a z-score first exists on a team's THIRD game; the spike is placed on the
# tenth, with eight quiet games behind it.
SUNDAYS = (
    [f"2012-{month:02d}-{day:02d}" for month, day in [(9, 9), (9, 16), (9, 23), (9, 30)]]
    + [f"2012-10-{day:02d}" for day in (7, 14, 21, 28)]
    + ["2012-11-04", "2012-11-11"]
)

TARGET_GAMEDAY = pd.Timestamp(SUNDAYS[-1])  # 2012-11-11, a Sunday
# window_end = gameday - ((weekday - 1) mod 7) days; Sunday weekday=6 -> 5 days
TARGET_WINDOW_END = pd.Timestamp("2012-11-06")  # Tuesday of that game week
TARGET_WINDOW_START = pd.Timestamp("2012-10-31")


def _games() -> pd.DataFrame:
    """One team pair playing ten straight weeks, plus a third pair each week so
    the frame looks like a slate rather than a single matchup."""

    rows = []
    for week, gameday in enumerate(SUNDAYS, start=1):
        rows.append(
            {
                "game_id": f"2012_{week:02d}_AAA_BBB",
                "season": 2012,
                "week": week,
                "gameday": gameday,
                "game_type": "REG",
                "home_team": "BBB",
                "away_team": "AAA",
                "spread_line": -3.0,
            }
        )
        rows.append(
            {
                "game_id": f"2012_{week:02d}_CCC_DDD",
                "season": 2012,
                "week": week,
                "gameday": gameday,
                "game_type": "REG",
                "home_team": "DDD",
                "away_team": "CCC",
                "spread_line": 1.5,
            }
        )
    return pd.DataFrame(rows)


#: A trailing z-score needs a non-degenerate trailing standard deviation, so
#: the fixture's daily comment counts carry seeded, deterministic day-to-day
#: variation. Posts stay flat, which keeps the comment-to-post ratio a clean
#: rescaling of the comment series.
_JITTER_SEED = 20260901


def _daily(values: dict[str, float], *, baseline: float, jitter: bool) -> pd.Series:
    """A daily series covering the whole fixture season, overridden on the dates
    in ``values``."""

    index = pd.date_range("2012-08-01", "2012-12-31", freq="D")
    if jitter:
        rng = np.random.default_rng(_JITTER_SEED)
        counts = baseline + rng.integers(-12, 13, size=len(index)).astype(float)
    else:
        counts = np.full(len(index), float(baseline))
    series = pd.Series(counts, index=index, dtype="float64")
    for date, value in values.items():
        series.loc[pd.Timestamp(date)] = value
    return series


def _team_daily(
    *,
    away_comment_spike: dict[str, float] | None = None,
    home_ratio_spike: dict[str, float] | None = None,
) -> dict[str, dict[str, pd.Series]]:
    """Four teams with identical daily post/comment traffic, optionally
    perturbed on named dates.

    ``away_comment_spike`` raises team AAA's comment volume (driving
    ``volume_z``); ``home_ratio_spike`` raises team BBB's comments while its
    posts stay flat (driving ``comment_post_ratio`` z).
    """

    out: dict[str, dict[str, pd.Series]] = {}
    for team in ("AAA", "BBB", "CCC", "DDD"):
        out[team] = {
            "posts": _daily({}, baseline=10.0, jitter=False),
            "comments": _daily({}, baseline=100.0, jitter=True),
        }
    if away_comment_spike:
        out["AAA"]["comments"] = _daily(away_comment_spike, baseline=100.0, jitter=True)
    if home_ratio_spike:
        out["BBB"]["comments"] = _daily(home_ratio_spike, baseline=100.0, jitter=True)
    return out


def _target(derived: pd.DataFrame, column: str) -> float:
    return float(derived.set_index("game_id").loc["2012_10_AAA_BBB", column])


# ---------------------------------------------------------------------------
# Leakage, both directions: the Tuesday-ending window is binding
# ---------------------------------------------------------------------------


def test_away_spike_inside_the_pre_kickoff_window_reaches_the_column() -> None:
    """A volume spike on a day inside the Tuesday-ending window must fire the
    away column. Without this direction the post-kickoff test below would pass
    trivially on a builder that never returns anything."""

    inside = str(TARGET_WINDOW_END.date())  # Tuesday 2012-11-06, the last day in
    assert TARGET_WINDOW_START <= pd.Timestamp(inside) <= TARGET_WINDOW_END
    derived = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(away_comment_spike={inside: 100_000.0})
    )
    assert _target(derived, REDDIT_AWAY_SPIKE_COLUMN) == 1.0


def test_away_spike_after_kickoff_never_reaches_the_column() -> None:
    """The identical spike moved to the day AFTER kickoff must be invisible.

    This is the leakage regression: the column for the 2012-11-11 game must be
    identical to the no-spike case.
    """

    after_kickoff = "2012-11-12"
    assert pd.Timestamp(after_kickoff) > TARGET_GAMEDAY
    derived = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(away_comment_spike={after_kickoff: 100_000.0})
    )
    quiet = derive_reddit_attention_features(_games(), team_daily=_team_daily())
    assert _target(derived, REDDIT_AWAY_SPIKE_COLUMN) == _target(quiet, REDDIT_AWAY_SPIKE_COLUMN)
    assert _target(derived, REDDIT_AWAY_SPIKE_COLUMN) == 0.0


def test_a_spike_between_the_tuesday_cutoff_and_kickoff_is_also_invisible() -> None:
    """The sharper claim: the window closes on TUESDAY, not at kickoff. A spike
    on the Friday before a Sunday game is still pregame, yet the frozen
    construction deliberately excludes it -- so the column must not move."""

    friday = "2012-11-09"
    assert TARGET_WINDOW_END < pd.Timestamp(friday) < TARGET_GAMEDAY
    derived = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(away_comment_spike={friday: 100_000.0})
    )
    assert _target(derived, REDDIT_AWAY_SPIKE_COLUMN) == 0.0


def test_home_ratio_spike_inside_the_window_reaches_the_column() -> None:
    """Mirror of the volume test for the comment-to-post ratio arm."""

    inside = str((TARGET_WINDOW_END - pd.Timedelta(days=2)).date())
    assert TARGET_WINDOW_START <= pd.Timestamp(inside) <= TARGET_WINDOW_END
    derived = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(home_ratio_spike={inside: 100_000.0})
    )
    assert _target(derived, REDDIT_HOME_RATIO_ELEVATED_COLUMN) == 1.0


def test_home_ratio_spike_after_kickoff_never_reaches_the_column() -> None:
    derived = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(home_ratio_spike={"2012-11-13": 100_000.0})
    )
    assert _target(derived, REDDIT_HOME_RATIO_ELEVATED_COLUMN) == 0.0


def test_a_future_weeks_spike_cannot_reach_an_earlier_games_baseline() -> None:
    """The trailing baseline is shift(1)-ed, so a later week's window can never
    normalize an earlier week's z-score. A spike in the last week must leave
    week 5's column untouched."""

    spiked = derive_reddit_attention_features(
        _games(), team_daily=_team_daily(away_comment_spike={"2012-11-06": 100_000.0})
    ).set_index("game_id")
    quiet = derive_reddit_attention_features(_games(), team_daily=_team_daily()).set_index(
        "game_id"
    )
    for week in range(1, 10):
        key = f"2012_{week:02d}_AAA_BBB"
        left = spiked.loc[key, REDDIT_AWAY_SPIKE_COLUMN]
        right = quiet.loc[key, REDDIT_AWAY_SPIKE_COLUMN]
        assert (pd.isna(left) and pd.isna(right)) or left == right


# ---------------------------------------------------------------------------
# Missing coverage is NaN, never False (predeclaration section 2, deviation 1)
# ---------------------------------------------------------------------------


def test_first_games_of_a_season_have_no_baseline_and_stay_missing() -> None:
    """A z-score needs two strictly prior games, so a team's first two games of
    a season come back NaN rather than "not elevated"."""

    derived = derive_reddit_attention_features(_games(), team_daily=_team_daily()).set_index(
        "game_id"
    )
    for column in REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS:
        assert np.isnan(derived.loc["2012_01_AAA_BBB", column])
        assert np.isnan(derived.loc["2012_02_AAA_BBB", column])


def test_a_team_with_no_subreddit_data_stays_missing() -> None:
    """A franchise whose subreddit did not yet exist (zero volume, zero trailing
    standard deviation) self-excludes to NaN -- it is never read as "no chatter,
    therefore no spike"."""

    daily = _team_daily()
    del daily["AAA"]
    derived = derive_reddit_attention_features(_games(), team_daily=daily).set_index("game_id")
    assert np.isnan(derived.loc["2012_10_AAA_BBB", REDDIT_AWAY_SPIKE_COLUMN])
    # the home side of the same game still resolves: the columns are per-side
    assert not np.isnan(derived.loc["2012_10_AAA_BBB", REDDIT_HOME_RATIO_ELEVATED_COLUMN])


def test_playoff_rows_are_never_given_a_value() -> None:
    """Only REG rows carry the column; a playoff row comes back NaN, and is
    never scored anyway (nfl_ats.modeling.regular_season_rows)."""

    games = _games()
    games.loc[len(games)] = {
        "game_id": "2012_19_AAA_BBB",
        "season": 2012,
        "week": 19,
        "gameday": "2013-01-06",
        "game_type": "WC",
        "home_team": "BBB",
        "away_team": "AAA",
        "spread_line": -2.5,
    }
    derived = derive_reddit_attention_features(games, team_daily=_team_daily()).set_index("game_id")
    for column in REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS:
        assert np.isnan(derived.loc["2012_19_AAA_BBB", column])


def test_no_fetched_data_at_all_returns_all_missing_not_all_false() -> None:
    derived = derive_reddit_attention_features(_games(), team_daily={})
    for column in REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS:
        assert derived[column].isna().all()
    assert len(derived) == len(_games())


# ---------------------------------------------------------------------------
# Additivity and join contracts
# ---------------------------------------------------------------------------


def test_attach_is_purely_additive() -> None:
    """Every pre-existing column comes back bit-identical and exactly the two
    named columns are added -- the discipline every candidate feature module in
    this family shares."""

    games = _games()
    games["some_existing_feature"] = np.linspace(-2.0, 2.0, len(games))

    widened = attach_reddit_attention_features(games, team_daily=_team_daily())

    new_columns = sorted(set(widened.columns) - set(games.columns))
    assert new_columns == sorted(REDDIT_ATTENTION_ON_PRODUCTION_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(games, widened[games.columns], check_exact=True)
    assert list(widened.index) == list(games.index)


def test_attach_requires_the_join_key() -> None:
    games = _games().drop(columns=["game_id"])
    with pytest.raises(DataContractError, match="game_id"):
        attach_reddit_attention_features(games, team_daily={})


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    games = _games()
    games[REDDIT_AWAY_SPIKE_COLUMN] = 0.0
    with pytest.raises(DataContractError, match=REDDIT_AWAY_SPIKE_COLUMN):
        attach_reddit_attention_features(games, team_daily={})


def test_derive_requires_every_input_column() -> None:
    games = _games().drop(columns=["game_type"])
    with pytest.raises(DataContractError, match="game_type"):
        derive_reddit_attention_features(games, team_daily={})


def test_join_is_one_to_one() -> None:
    """A duplicated game_id must raise rather than silently fanning out rows --
    ``validate="one_to_one"`` on the merge is what enforces it.

    Two exception types are accepted because which one surfaces is a pandas
    implementation detail, not a contract: merging on a Series ``left_on`` with
    duplicate keys trips an internal index-name check (``TypeError``) before
    ``validate`` reports its own ``MergeError``. The claim under test is that
    it RAISES, never that it raises a particular class.
    """

    games = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises((pd.errors.MergeError, TypeError)):
        attach_reddit_attention_features(games, team_daily={})

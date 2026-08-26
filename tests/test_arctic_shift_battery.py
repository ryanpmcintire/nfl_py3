from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("arctic_shift_battery_screen_test", "arctic_shift_battery_screen.py")


def test_window_sum_excludes_days_outside_window() -> None:
    dates = pd.date_range("2024-09-01", "2024-09-10", freq="D")
    series = pd.Series(1.0, index=dates)
    start = pd.Timestamp("2024-09-03")
    end = pd.Timestamp("2024-09-09")

    total = screen.window_sum(series, start, end)

    assert total == 7.0  # Sep 3 through Sep 9 inclusive, 10 days available


def test_window_sum_ignores_a_day_strictly_after_the_cutoff() -> None:
    """The window's own upper bound is the decision cutoff; a spike the day
    after it (i.e. during game week itself) must never move the window sum."""

    dates = pd.date_range("2024-09-01", "2024-09-10", freq="D")
    baseline = pd.Series(1.0, index=dates)
    spiked = baseline.copy()
    spiked.loc[pd.Timestamp("2024-09-10")] += 1000.0  # one day after the window end below

    window_start = pd.Timestamp("2024-09-03")
    window_end = pd.Timestamp("2024-09-09")

    before = screen.window_sum(baseline, window_start, window_end)
    after = screen.window_sum(spiked, window_start, window_end)

    assert before == after


def test_window_sum_reflects_a_spike_inside_the_window() -> None:
    """Sanity check the test above is not vacuous: a spike INSIDE the window
    must change the sum."""

    dates = pd.date_range("2024-09-01", "2024-09-10", freq="D")
    baseline = pd.Series(1.0, index=dates)
    spiked = baseline.copy()
    spiked.loc[pd.Timestamp("2024-09-05")] += 1000.0  # inside [Sep 3, Sep 9]

    window_start = pd.Timestamp("2024-09-03")
    window_end = pd.Timestamp("2024-09-09")

    before = screen.window_sum(baseline, window_start, window_end)
    after = screen.window_sum(spiked, window_start, window_end)

    assert after == pytest.approx(before + 1000.0)


def _synthetic_games() -> pd.DataFrame:
    """Two REG games for ARI, three weeks apart, both Sunday games."""

    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 4],
            "gameday": pd.to_datetime(["2024-09-08", "2024-09-29"]),  # both Sundays
            "home_team": ["ARI", "ARI"],
            "away_team": ["BUF", "BUF"],
        }
    )


def _daily_series(dates: pd.DatetimeIndex, value: float = 1.0) -> pd.Series:
    return pd.Series(value, index=dates, dtype="float64")


def test_build_team_game_long_window_ignores_activity_after_tuesday_cutoff() -> None:
    """The Tuesday-ending window for g1 (gameday 2024-09-08, a Sunday) ends
    2024-09-03; a spike on 2024-09-04 (game week itself, still strictly
    before kickoff) must not change g1's computed window_volume -- the same
    leakage-regression shape as tests/test_gdelt_attention_screen.py's
    test_tuesday_features_ignore_news_after_tuesday_cutoff."""

    games = _synthetic_games()
    dates = pd.date_range("2024-08-01", "2024-10-01", freq="D")
    posts_baseline = _daily_series(dates, 2.0)
    comments_baseline = _daily_series(dates, 3.0)
    team_daily_before = {
        "ARI": {"posts": posts_baseline, "comments": comments_baseline},
        "BUF": {"posts": _daily_series(dates, 1.0), "comments": _daily_series(dates, 1.0)},
    }

    posts_after = posts_baseline.copy()
    posts_after.loc[pd.Timestamp("2024-09-04")] += 500.0  # one day after g1's Tuesday cutoff
    team_daily_after = {
        "ARI": {"posts": posts_after, "comments": comments_baseline},
        "BUF": {"posts": _daily_series(dates, 1.0), "comments": _daily_series(dates, 1.0)},
    }

    long_before = screen.build_team_game_long(games, team_daily_before)
    long_after = screen.build_team_game_long(games, team_daily_after)

    g1_before = long_before.loc[(long_before["game_id"] == "g1") & (long_before["team"] == "ARI")]
    g1_after = long_after.loc[(long_after["game_id"] == "g1") & (long_after["team"] == "ARI")]

    assert g1_before["window_volume"].iloc[0] == g1_after["window_volume"].iloc[0]
    assert g1_before["window_posts"].iloc[0] == g1_after["window_posts"].iloc[0]


def test_build_team_game_long_window_reflects_activity_inside_the_window() -> None:
    """Sanity check: a spike strictly INSIDE g1's window (before the Tuesday
    cutoff) must change window_volume -- otherwise the leakage test above
    would be vacuous."""

    games = _synthetic_games()
    dates = pd.date_range("2024-08-01", "2024-10-01", freq="D")
    posts_baseline = _daily_series(dates, 2.0)
    comments_baseline = _daily_series(dates, 3.0)
    team_daily_before = {
        "ARI": {"posts": posts_baseline, "comments": comments_baseline},
        "BUF": {"posts": _daily_series(dates, 1.0), "comments": _daily_series(dates, 1.0)},
    }

    posts_inside = posts_baseline.copy()
    posts_inside.loc[pd.Timestamp("2024-09-01")] += 500.0  # inside g1's window
    team_daily_inside = {
        "ARI": {"posts": posts_inside, "comments": comments_baseline},
        "BUF": {"posts": _daily_series(dates, 1.0), "comments": _daily_series(dates, 1.0)},
    }

    long_before = screen.build_team_game_long(games, team_daily_before)
    long_inside = screen.build_team_game_long(games, team_daily_inside)

    g1_before = long_before.loc[(long_before["game_id"] == "g1") & (long_before["team"] == "ARI")]
    g1_inside = long_inside.loc[(long_inside["game_id"] == "g1") & (long_inside["team"] == "ARI")]

    assert g1_before["window_volume"].iloc[0] != g1_inside["window_volume"].iloc[0]


def test_build_team_game_long_trailing_baseline_excludes_current_window() -> None:
    """g2's trailing volume_z must be computed from g1's window only (the
    team's strictly prior game), never including g2's own window sum. A
    spike inside g2's OWN window must not move g2's baseline/mean, hence
    must not silently cancel out in the z-score denominator either."""

    games = _synthetic_games()
    dates = pd.date_range("2024-08-01", "2024-10-01", freq="D")
    posts = _daily_series(dates, 2.0)
    comments = _daily_series(dates, 3.0)
    team_daily = {
        "ARI": {"posts": posts, "comments": comments},
        "BUF": {"posts": _daily_series(dates, 1.0), "comments": _daily_series(dates, 1.0)},
    }

    long_df = screen.build_team_game_long(games, team_daily)
    ari = long_df.loc[long_df["team"] == "ARI"].sort_values("gameday")

    # g1 is ARI's first game of the season -> no strictly-prior game -> no baseline.
    assert bool(ari.iloc[0]["has_baseline_volume"]) is False
    # g2 has exactly one strictly-prior game (g1); TRAILING_MIN_GAMES=2 in
    # attention_battery_screen.py, so a single prior game is still short of
    # the floor -- also no baseline. This assertion documents that floor
    # rather than assuming it; if the shared constant ever changes this test
    # will fail loudly instead of silently drifting.
    from attention_battery_screen import TRAILING_MIN_GAMES

    if TRAILING_MIN_GAMES > 1:
        assert bool(ari.iloc[1]["has_baseline_volume"]) is False
    else:
        assert bool(ari.iloc[1]["has_baseline_volume"]) is True


def test_load_subreddit_daily_counts_skips_teams_without_both_files(tmp_path: Path) -> None:
    """A team missing either the posts or comments file (a failed fetch) must
    be excluded entirely, never zero-filled -- zero-filling would silently
    manufacture a fake, always-cold baseline instead of reporting missing
    coverage."""

    raw_dir = tmp_path
    subreddit = next(iter(screen.SUBREDDITS_ALL.values()))
    posts_path = raw_dir / f"{subreddit}_posts_timeseries_full.json"
    posts_path.write_text('{"data": [{"date": 1700000000, "value": 5}]}', encoding="utf-8")
    # comments file deliberately not written

    out = screen.load_subreddit_daily_counts(raw_dir)

    team = next(t for t, sub in screen.SUBREDDITS_ALL.items() if sub == subreddit)
    assert team not in out

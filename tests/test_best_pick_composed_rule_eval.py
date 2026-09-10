from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import nfl_ats.best_pick_nomination as bpn
from nfl_ats.best_pick_nomination import dispersion_pool_from_frame, week_dispersion_pool


def test_dispersion_pool_from_frame_matches_week_dispersion_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for game_id, lines in {"g1": [1.0, 1.0], "g2": [1.0, 2.0], "g3": [1.0, 3.0]}.items():
        for book_index, line in enumerate(lines):
            rows.append(
                {
                    "nflverse_game_id": game_id,
                    "bookmaker_key": f"book{book_index}",
                    "market": "spreads",
                    "outcome_side": "HOME",
                    "home_spread_line": line,
                    "observed_at_utc": pd.Timestamp("2026-08-18T13:00:00Z"),
                    "commence_time_utc": pd.Timestamp("2026-08-22T13:00:00Z"),
                }
            )
    monkeypatch.setattr(bpn, "load_quote_history", lambda root: pd.DataFrame(rows))
    production = week_dispersion_pool(Path("unused"), ["g1", "g2", "g3", "g4"])
    direct = dispersion_pool_from_frame(production.frame[["game_id", "spread_std"]])
    assert direct.fallback == production.fallback
    assert direct.fallback_reason == production.fallback_reason
    assert direct.n_pool_pass == production.n_pool_pass
    pd.testing.assert_frame_equal(direct.frame, production.frame)

    frame = pd.DataFrame({"game_id": ["a", "b", "c"], "spread_std": [0.0, 0.5, 1.0]})
    pool = dispersion_pool_from_frame(frame)
    assert pool.fallback is False
    assert set(pool.frame.loc[pool.frame["pool_pass"], "game_id"]) == {"a"}
    with pytest.raises(ValueError, match="missing columns"):
        dispersion_pool_from_frame(pd.DataFrame({"game_id": ["a"]}))

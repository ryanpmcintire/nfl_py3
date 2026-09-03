"""Release-blocking tests for the unit-APM reliability screen (no network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.unit_apm_screen import (
    build_unit_lookup,
    fit_unit_coefficients,
    modal_unit_by_player_season,
    split_half_reliability,
)


def _rosters() -> pd.DataFrame:
    rows = []
    # AAA: G in 2020, T in 2021 (a real position switch across seasons).
    rows += [
        {"gsis_id": "AAA", "season": 2020, "position": "G", "team": "H", "week": 1},
        {"gsis_id": "AAA", "season": 2020, "position": "G", "team": "H", "week": 2},
        {"gsis_id": "AAA", "season": 2021, "position": "T", "team": "H", "week": 1},
    ]
    # BBB: WR every season. CCC: K (unmapped specialist).
    for season in (2020, 2021):
        rows.append({"gsis_id": "BBB", "season": season, "position": "WR", "team": "H", "week": 1})
        rows.append({"gsis_id": "CCC", "season": season, "position": "K", "team": "H", "week": 1})
    return pd.DataFrame(rows)


def test_modal_unit_resolves_switches_and_counts_unmapped() -> None:
    units, unmapped = modal_unit_by_player_season(_rosters())
    lookup = build_unit_lookup(units)
    assert lookup[("AAA", 2020)] == "OFF_OL"
    assert lookup[("AAA", 2021)] == "OFF_OL"
    assert lookup[("BBB", 2020)] == "OFF_SKILL"
    assert ("CCC", 2020) not in lookup
    assert unmapped.get("K") == 2


def test_skill_player_never_enters_the_ol_design() -> None:
    table = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 1,
                "game_id": "2020_01_X_Y",
                "play_id": 1,
                "posteam": "H",
                "defteam": "A",
                "epa": 1.0,
                "offense_players": "AAA;BBB",
                "defense_players": "DDD;EEE",
            }
        ]
    )
    lookup = {("AAA", 2020): "OFF_OL", ("BBB", 2020): "OFF_SKILL"}
    coef, counts = fit_unit_coefficients(table, "OFF_OL", lookup)
    assert set(counts) == {"AAA"}
    assert "unit_player::BBB" not in coef


def _rotating_table() -> pd.DataFrame:
    """Randomized pools (fixed seed) with stable per-player latents.

    Deliberately NOT a rotation schedule: a period-8 rotation phase-locks
    pool composition to week parity, which flips odd/even coefficient signs
    as an artifact of the fixture rather than the implementation.
    """

    import numpy as np

    rng = np.random.default_rng(7)
    latent = {f"P{i}": (i - 3.5) * 1.0 for i in range(8)}
    rows = []
    play = 0
    for week in range(1, 21):
        for side in ("H", "A"):
            play += 1
            pool = [f"P{i}" for i in rng.choice(8, size=4, replace=False)]
            # Same +1 coding as the implementation for every row: the target
            # must NOT be negated by side (that would model players hurting
            # their own team on the road and make the halves disagree).
            epa = float(sum(latent[player] for player in pool)) / 4.0
            rows.append(
                {
                    "season": 2020,
                    "week": week,
                    "game_id": f"2020_{play:02d}_X_Y",
                    "play_id": play,
                    "posteam": side,
                    "defteam": "A" if side == "H" else "H",
                    "epa": epa,
                    "offense_players": ";".join(pool),
                    "defense_players": ";".join(f"Q{i}" for i in range(4)),
                }
            )
    return pd.DataFrame(rows)


def _rotating_lookup() -> dict:
    return {(f"P{i}", 2020): "OFF_SKILL" for i in range(8)}


def test_split_half_deterministic_floor_gated_and_bounded() -> None:
    table = _rotating_table()
    lookup = _rotating_lookup()
    first = split_half_reliability(table, "OFF_SKILL", lookup, min_plays_per_half=1)
    second = split_half_reliability(table, "OFF_SKILL", lookup, min_plays_per_half=1)
    assert first["split_half_pearson"] == pytest.approx(second["split_half_pearson"])
    assert first["n"] == 8
    assert first["split_half_pearson"] > 0.5
    gated = split_half_reliability(table, "OFF_SKILL", lookup, min_plays_per_half=10**9)
    assert gated["insufficient"] is True


def test_roster_reader_rejects_missing_keys() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        modal_unit_by_player_season(pd.DataFrame({"gsis_id": ["A"]}))

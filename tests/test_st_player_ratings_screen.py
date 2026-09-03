"""Release-blocking tests for the ST-APM reliability screen (no network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.st_player_ratings_screen import (
    build_st_play_table,
    classify_st_unit,
    fit_st_coefficients,
    play_type_agreement,
    split_half_reliability,
)

FG = "1 C, 1 G, 1 K, 1 LS, 1 P, 4 T, 2 TE"
PUNT = "1 C, 2 G, 1 LS, 1 P, 4 T, 2 TE, 1 WR"
KICK = "1 C, 2 G, 1 K, 5 T, 2 TE"
SCRIMMAGE = "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR"
DEFENSE = "2 CB, 3 DE, 1 DT, 2 ILB, 1 OLB, 1 SS"


def test_token_classifier_splits_units() -> None:
    assert classify_st_unit(FG, DEFENSE) == "FG_XP"
    assert classify_st_unit(DEFENSE, FG) == "FG_XP"
    assert classify_st_unit(PUNT, DEFENSE) == "PUNT"
    assert classify_st_unit(KICK, DEFENSE) == "KICKOFF"
    assert classify_st_unit(SCRIMMAGE, DEFENSE) is None
    assert classify_st_unit(None, None) is None


def _ids(prefix: str, n: int = 11) -> str:
    return ";".join(f"{prefix}{i:02d}" for i in range(n))


def _synthetic_table() -> pd.DataFrame:
    """Rotating player pools with stable per-player latents: odd/even halves
    must recover the same ordering (high split-half correlation), while a
    fixed 22-man lineup would leave every coefficient identical (NaN)."""

    latent_a = {f"A{i:02d}": 0.5 * i for i in range(8)}
    latent_b = {f"B{i:02d}": -0.5 * i for i in range(8)}
    rows = []
    play = 0
    for week in range(1, 9):
        for side_index, side in enumerate(("HOME", "AWAY")):
            play += 1
            pool_a = [f"A{(week + side_index * 3 + i) % 8:02d}" for i in range(4)]
            pool_b = [f"B{(week + side_index * 5 + i) % 8:02d}" for i in range(4)]
            epa = (
                float(
                    sum(latent_a[player] for player in pool_a)
                    - sum(latent_b[player] for player in pool_b)
                )
                / 4.0
            )
            rows.append(
                {
                    "season": 2020,
                    "week": week,
                    "game_id": f"2020_{play:02d}_X_Y",
                    "play_id": play,
                    "posteam": side,
                    "possession_team": side,
                    "epa": epa,
                    "st_unit": "PUNT",
                    "side_a_ids": tuple(pool_a),
                    "side_b_ids": tuple(pool_b),
                }
            )
    return pd.DataFrame(rows)


def _ids_str(prefix: str, n: int = 11) -> str:
    return ";".join(f"99-{prefix}{i:03d}" for i in range(n))


def _synthetic_participation() -> pd.DataFrame:
    base = {
        "season": 2020,
        "old_game_id": "x",
        "possession_team": "HOME",
        "offense_formation": pd.NA,
        "defenders_in_box": pd.NA,
        "defense_personnel": DEFENSE,
        "number_of_pass_rushers": pd.NA,
        "ngs_air_yards": pd.NA,
        "time_to_throw": pd.NA,
        "was_pressure": pd.NA,
        "route": pd.NA,
        "defense_man_zone_type": pd.NA,
        "defense_coverage_type": pd.NA,
    }
    rows = [
        {"game_id": "2020_01_A_B", "play_id": 1, "offense_personnel": PUNT},
        {"game_id": "2020_01_A_B", "play_id": 2, "offense_personnel": SCRIMMAGE},
        {"game_id": "2020_01_A_B", "play_id": 3, "offense_personnel": FG},
    ]
    frames = []
    for row in rows:
        record = dict(base)
        record.update(row)
        record["offense_players"] = _ids_str("O")
        record["defense_players"] = _ids_str("D")
        record["players_on_play"] = record["offense_players"] + ";" + record["defense_players"]
        record["n_offense"] = 11
        record["n_defense"] = 11
        frames.append(record)
    return pd.DataFrame(frames)


def _synthetic_pbp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2020_01_A_B",
                "play_id": 1,
                "season": 2020,
                "week": 1,
                "posteam": "HOME",
                "epa": 0.5,
                "play_type": "punt",
            },
            {
                "game_id": "2020_01_A_B",
                "play_id": 2,
                "season": 2020,
                "week": 1,
                "posteam": "HOME",
                "epa": 0.5,
                "play_type": "pass",
            },
            {
                "game_id": "2020_01_A_B",
                "play_id": 3,
                "season": 2020,
                "week": 1,
                "posteam": "HOME",
                "epa": 0.5,
                "play_type": "extra_point",
            },
        ]
    )


def test_builder_keeps_only_token_st_plays_with_epa() -> None:
    table = build_st_play_table(_synthetic_participation(), _synthetic_pbp())
    assert set(table["st_unit"]) == {"PUNT", "FG_XP"}
    assert len(table) == 2
    agreement = play_type_agreement(table)
    assert agreement["rows"] == 2
    assert agreement["agreement_rate"] == pytest.approx(1.0)


def test_builder_rejects_unmatched_epa() -> None:
    pbp = _synthetic_pbp().iloc[:0].copy()
    with pytest.raises(DataContractError, match="No participation plays match"):
        build_st_play_table(_synthetic_participation(), pbp)


def test_split_half_is_deterministic_and_floor_gated() -> None:
    table = _synthetic_table()
    first = split_half_reliability(table, min_plays_per_half=1)
    second = split_half_reliability(table, min_plays_per_half=1)
    assert first["split_half_pearson"] == pytest.approx(second["split_half_pearson"])
    assert first["n"] == 16
    assert first["split_half_pearson"] > 0.5
    gated = split_half_reliability(table, min_plays_per_half=10**9)
    assert gated["insufficient"] is True


def test_fit_is_deterministic_on_synthetic_table() -> None:
    table = _synthetic_table()
    first_coef, first_counts = fit_st_coefficients(table)
    second_coef, second_counts = fit_st_coefficients(table)
    assert first_counts == second_counts
    assert set(first_coef) == set(second_coef)
    for key in first_coef:
        assert first_coef[key] == pytest.approx(second_coef[key])

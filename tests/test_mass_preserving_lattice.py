"""MOD-18 lane K: the mass-preserving conditional margin read.

The property under test is the one the lane exists for: the integer atoms keep
their ABSOLUTE positions and the model's point enters only as a reweighting, so
the key-number mass at 3/7/10/14 is never translated away (MOD-05's recentring
constraint, which lane K's K2 and lane H's M1 both violated in effect).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mass_preserving_lattice_opener_eval import (
    ARMS,
    MIN_BAND_GAMES,
    THETA_BRACKET,
    band_read,
    build_pool,
    mass_preserving,
    powershell_quote,
    record_argv,
    tilted_atoms,
    verify_replay,
)


def key_number_atoms():
    """A base distribution with real key-number spikes at 3, 7, 10 and 14."""

    margins = np.arange(-21.0, 22.0)
    counts = np.ones_like(margins)
    for key, weight in ((3, 12.0), (7, 7.0), (10, 4.0), (14, 3.0)):
        counts[margins == key] += weight
        counts[margins == -key] += weight
    return margins, counts


def pool_fixture():
    """Prior games whose lines span the band and whose margins are integers."""

    rows = []
    for season in (2014, 2015, 2016, 2017, 2018, 2019):
        for index in range(120):
            rows.append(
                {
                    "game_id": f"{season}-{index}",
                    "season": season,
                    "week": 1 + index % 17,
                    "gameday": pd.Timestamp(f"{season}-09-10") + pd.Timedelta(days=index),
                    "line": float(index % 21 - 10),
                    "result": float((index * 7) % 41 - 20),
                }
            )
    return pd.DataFrame(rows)


def test_tilt_moves_weight_not_positions():
    margins, counts = key_number_atoms()
    mass, theta = tilted_atoms(margins, counts, line=3.0, target=6.0)
    assert np.isclose((margins * mass).sum(), 6.0)
    assert theta > 0.0
    # The support is untouched: every base atom keeps its position and mass.
    assert mass.shape == margins.shape
    assert (mass > 0).all()
    # The key-number spikes are still AT the key numbers, not beside them.
    for key in (3.0, 7.0, 10.0, 14.0):
        index = int(np.flatnonzero(margins == key)[0])
        assert mass[index] > mass[index - 1]
        assert mass[index] > mass[index + 1]


def test_tilt_of_the_base_mean_is_the_base_distribution():
    margins, counts = key_number_atoms()
    base = counts / counts.sum()
    mass, theta = tilted_atoms(margins, counts, line=0.0, target=float((margins * base).sum()))
    assert theta == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(mass, base, atol=1e-12)


def test_tilt_is_monotone_in_the_target():
    margins, counts = key_number_atoms()
    lower, _ = tilted_atoms(margins, counts, line=0.0, target=-4.0)
    upper, _ = tilted_atoms(margins, counts, line=0.0, target=+4.0)
    assert upper[margins > 0].sum() > lower[margins > 0].sum()


def test_unreachable_target_clamps_to_the_declared_bracket():
    margins, counts = key_number_atoms()
    _, theta = tilted_atoms(margins, counts, line=0.0, target=1e6)
    assert theta == THETA_BRACKET
    _, theta = tilted_atoms(margins, counts, line=0.0, target=-1e6)
    assert theta == -THETA_BRACKET


def test_push_mass_sits_on_the_integer_line_and_is_zero_at_a_half_point():
    lines = np.repeat(np.arange(-10.0, 11.0), 40)
    margins = np.tile(key_number_atoms()[0][:40], 21)
    integer = band_read(lines, margins, line=3.0, point=3.0, half_width=2.5)
    half = band_read(lines, margins, line=3.5, point=3.5, half_width=2.5)
    assert integer["push"] > 0.0
    assert half["push"] == 0.0
    for read in (integer, half):
        assert read["cover"] + read["push"] + read["loss"] == pytest.approx(1.0)
        assert read["home_cover_probability"] == pytest.approx(read["cover"] + 0.5 * read["push"])


def test_band_widens_only_until_the_declared_floor():
    lines = np.concatenate([np.full(30, 12.0), np.full(400, 0.0)])
    margins = np.concatenate([np.full(30, 14.0), np.zeros(400)])
    read = band_read(lines, margins, line=12.0, point=12.0, half_width=2.5)
    assert read["band"] > 2.5
    assert read["band_games"] >= MIN_BAND_GAMES
    wide = band_read(lines, margins, line=0.0, point=0.0, half_width=2.5)
    assert wide["band"] == 2.5


def test_future_same_week_and_old_seasons_cannot_leak():
    pool = pool_fixture()
    targets = pd.DataFrame(
        {
            "game_id": ["t1", "t2"],
            "season": [2020, 2020],
            "week": [1, 1],
            "gameday": [pd.Timestamp("2020-09-10"), pd.Timestamp("2020-09-10")],
            "line": [3.0, -3.0],
            "point": [3.0, -3.0],
        }
    )
    expected = mass_preserving(pool, targets, 2.5)
    poisoned = pd.concat(
        [
            pool,
            # Same week as the target, a later week, and a sixth season back.
            pd.DataFrame(
                {
                    "game_id": ["same", "later", "ancient"],
                    "season": [2020, 2020, 2013],
                    "week": [1, 2, 1],
                    "gameday": [
                        pd.Timestamp("2020-09-10"),
                        pd.Timestamp("2020-09-20"),
                        pd.Timestamp("2013-09-10"),
                    ],
                    "line": [3.0, 3.0, 3.0],
                    "result": [40.0, 40.0, 40.0],
                }
            ),
        ],
        ignore_index=True,
    )
    poisoned_read = mass_preserving(poisoned, targets, 2.5)
    pd.testing.assert_frame_equal(expected, poisoned_read)


def test_a_prior_game_completed_before_the_cutoff_does_enter():
    pool = pool_fixture()
    targets = pd.DataFrame(
        {
            "game_id": ["t1"],
            "season": [2020],
            "week": [3],
            "gameday": [pd.Timestamp("2020-09-24")],
            "line": [3.0],
            "point": [3.0],
        }
    )
    before = mass_preserving(pool, targets, 2.5)
    extra = pd.DataFrame(
        {
            "game_id": ["earlier"],
            "season": [2020],
            "week": [1],
            "gameday": [pd.Timestamp("2020-09-10")],
            "line": [3.0],
            "result": [40.0],
        }
    )
    after = mass_preserving(pd.concat([pool, extra], ignore_index=True), targets, 2.5)
    assert after.prior_rows.iloc[0] == before.prior_rows.iloc[0] + 1
    # A margin of 40 is not in the fixture's support, so the completed earlier
    # game adds an atom and the read moves. (It need not move UP: the tilt holds
    # the mean fixed, so a new high atom is paid for by a more negative theta.)
    assert after.atoms.iloc[0] == before.atoms.iloc[0] + 1
    assert after.cover.iloc[0] != before.cover.iloc[0]


def test_build_pool_prefers_the_archived_opener_line(tmp_path, monkeypatch):
    features = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "post"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 1, 1, 20],
            "gameday": [pd.Timestamp("2020-09-10")] * 4,
            "game_type": ["REG", "REG", "REG", "WC"],
            "spread_line": [1.0, 2.0, 3.0, 4.0],
            # The unplayed game and the postseason row must both drop out.
            "result": [7.0, -3.0, np.nan, 5.0],
        }
    )
    path = tmp_path / "features.parquet"
    features.to_parquet(path, index=False)
    monkeypatch.setattr(build_pool.__globals__["common"], "FEATURES", path)
    archive = pd.DataFrame({"game_id": ["a"], "tue_open_home_spread": [4.5]})
    pool = build_pool(archive)
    assert list(pool.game_id) == ["a", "b"]
    assert list(pool.line) == [4.5, 2.0]


def test_replay_gate_fails_closed():
    served = np.array([0.5, 0.6])
    assert verify_replay(served.copy(), served) == 0.0
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_replay(np.array([0.5, 0.6 + 1e-6]), served)
    with pytest.raises(ValueError, match="missing probabilities"):
        verify_replay(np.array([0.5, np.nan]), served)


def test_recorder_argv_is_admissible_and_names_no_closing_ground():
    metrics = {
        "delta": -1.13,
        "lower": -3.09,
        "upper": 0.87,
        "probability_positive": 0.1263,
        "standard_error": 0.95,
        "n": 1503,
        "weeks": 107,
    }
    argv = record_argv("mp1_mp1_overall_standalone", metrics, "accuracy_points", 2020, 2025)
    assert argv[:2] == ["weak-signals", "record"]
    assert argv[argv.index("--classification") + 1] == "unresolved_below_power"
    assert "--closing-ground" not in argv
    assert argv[argv.index("--probability-positive") + 1] == "0.126300000000"
    # Scientific notation would be read by argparse as a flag; fixed point is not.
    assert "e" not in argv[argv.index("--effect") + 1]


def test_powershell_quoting_escapes_apostrophes():
    assert powershell_quote("lane H's M1") == "'lane H''s M1'"


def test_both_declared_arms_are_band_widths():
    assert set(ARMS) == {"MP1", "MP1b"}
    assert ARMS["MP1b"] == 2.0 * ARMS["MP1"]

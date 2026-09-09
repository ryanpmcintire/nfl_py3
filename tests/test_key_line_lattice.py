"""MOD-18 lane T: the key-line restriction of lane K's mass-preserving read.

The property under test is the restriction itself. The discrete read is served
ONLY where the quoted line sits exactly on a key atom (3, 7, 10 or 14, either
sign); every other game keeps the served S3 probability bit-for-bit, so the
restriction can never move a game the mechanism does not name.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mass_preserving_lattice_opener_eval as lane_k
from key_line_lattice_opener_eval import (
    ARMS,
    FAMILY,
    KEY_NUMBERS,
    key_line_mask,
    plain_summary,
    record_argv,
    restrict,
    split_cell,
    units_for,
)

LINES = pd.Series(
    [
        0.0,
        3.0,
        -3.0,
        3.5,
        -2.5,
        6.5,
        6.75,
        7.0,
        -7.0,
        -6.75,
        7.25,
        7.5,
        9.5,
        10.0,
        -10.0,
        10.5,
        13.5,
        14.0,
        -14.0,
        14.5,
    ]
)
EXPECTED_KL1 = {3.0, -3.0, 7.0, -7.0, 10.0, -10.0, 14.0, -14.0}
JARGON = (
    "opener",
    "p+",
    "probability_positive",
    "played card",
    "week-blocked",
    "bootstrap",
    "brier",
    "log loss",
    "snapshot",
    "model id",
    "overlay",
    "raw model",
)


def served_arrays(size):
    """Distinguishable served and mapped reads so a mix-up cannot pass."""

    served = np.linspace(0.10, 0.40, size)
    mapped = np.linspace(0.60, 0.90, size)
    return served, served / 10.0, mapped, mapped / 10.0


def test_key_line_mask_selects_only_lines_quoted_on_an_atom():
    mask = key_line_mask(LINES, KEY_NUMBERS)
    selected = set(LINES[mask])
    assert selected == EXPECTED_KL1
    assert not mask[list(LINES).index(6.75)]
    assert not mask[list(LINES).index(-6.75)]
    assert not mask[list(LINES).index(7.25)]
    assert not mask[list(LINES).index(0.0)]
    assert not mask[list(LINES).index(3.5)]


def test_kl1b_is_a_declared_subset_of_kl1():
    assert ARMS["KL1"] == KEY_NUMBERS
    assert set(ARMS["KL1b"]) < set(ARMS["KL1"])
    assert set(ARMS["KL1b"]) == {3.0, 7.0}
    narrow = key_line_mask(LINES, ARMS["KL1b"])
    wide = key_line_mask(LINES, ARMS["KL1"])
    assert bool(np.all(wide[narrow]))
    assert set(LINES[narrow]) == {3.0, -3.0, 7.0, -7.0}


def test_non_key_lines_reproduce_the_served_read_bit_for_bit():
    served, served_push, mapped, mapped_push = served_arrays(len(LINES))
    for keys in ARMS.values():
        applied = restrict(LINES, served, served_push, mapped, mapped_push, keys)
        untouched = ~applied["touched"]
        assert untouched.any()
        assert list(applied["probability"][untouched]) == list(served[untouched])
        assert list(applied["push"][untouched]) == list(served_push[untouched])


def test_key_lines_take_the_mapped_read():
    served, served_push, mapped, mapped_push = served_arrays(len(LINES))
    applied = restrict(LINES, served, served_push, mapped, mapped_push, KEY_NUMBERS)
    touched = applied["touched"]
    assert int(touched.sum()) == len(EXPECTED_KL1)
    assert list(applied["probability"][touched]) == list(mapped[touched])
    assert list(applied["push"][touched]) == list(mapped_push[touched])


def test_a_card_with_no_key_line_is_left_entirely_alone():
    lines = pd.Series([1.5, -2.5, 4.5, -6.5, 8.5, -11.5])
    served, served_push, mapped, mapped_push = served_arrays(len(lines))
    applied = restrict(lines, served, served_push, mapped, mapped_push, KEY_NUMBERS)
    assert not applied["touched"].any()
    assert list(applied["probability"]) == list(served)


def test_the_s3_replay_gate_fails_closed():
    """The gate lane T's replay stage runs before any candidate is built."""

    served = np.array([0.5, 0.6])
    assert lane_k.verify_replay(served.copy(), served) == 0.0
    with pytest.raises(ValueError, match="replay mismatch"):
        lane_k.verify_replay(np.array([0.5, 0.6 + 1e-6]), served)
    with pytest.raises(ValueError, match="missing probabilities"):
        lane_k.verify_replay(np.array([0.5, np.nan]), served)


@pytest.mark.parametrize(
    ("name", "arm", "label", "kind"),
    [
        ("kl1_kl1_overall_standalone", "KL1", "overall", "standalone"),
        ("kl1_kl1_season_2023_card", "KL1", "season_2023", "card"),
        ("kl1_kl1_key_line_7_brier", "KL1", "key_line_7", "brier"),
        ("kl1_kl1_touched_log_loss", "KL1", "touched", "log_loss"),
        ("kl1_kl1_push_at_3_push_brier", "KL1", "push_at_3", "push_brier"),
        ("kl1_kl1b_overall_standalone", "KL1b", "overall", "standalone"),
        ("kl1_kl1b_key_line_3_log_loss", "KL1b", "key_line_3", "log_loss"),
        ("kl1_kl1b_push_at_7_push_brier", "KL1b", "push_at_7", "push_brier"),
    ],
)
def test_every_cell_name_shape_splits_and_earns_a_plain_summary(name, arm, label, kind):
    assert split_cell(name) == (arm, label, kind)
    summary = plain_summary(arm, label, kind)
    assert summary.strip()
    assert not any(token in summary.lower() for token in JARGON)


def test_an_unrecognised_cell_name_is_refused():
    with pytest.raises(ValueError, match="Unrecognised cell name"):
        split_cell("kl1_kl1_overall_mystery")


def test_units_follow_the_metric():
    assert units_for("kl1_kl1_overall_standalone") == "accuracy_points"
    assert units_for("kl1_kl1_overall_card") == "accuracy_points"
    assert units_for("kl1_kl1_overall_brier") == "brier_improvement"
    assert units_for("kl1_kl1_push_at_3_push_brier") == "brier_improvement"
    assert units_for("kl1_kl1_overall_log_loss") == "log_loss_improvement"


def test_recorder_argv_is_admissible_and_carries_a_plain_summary():
    metrics = {
        "delta": -1.13,
        "lower": -3.09,
        "upper": 0.87,
        "probability_positive": 0.1263,
        "standard_error": 0.95,
        "n": 1503,
        "weeks": 107,
    }
    argv = record_argv("kl1_kl1_overall_standalone", metrics, "accuracy_points", 2020, 2025)
    assert argv[:2] == ["weak-signals", "record"]
    assert argv[argv.index("--classification") + 1] == "unresolved_below_power"
    assert "--closing-ground" not in argv
    assert argv[argv.index("--family") + 1] == FAMILY
    name = argv[argv.index("--name") + 1]
    assert name == f"{FAMILY}_kl1_kl1_overall_standalone_2020_2025"
    assert name.startswith(f"{FAMILY}_kl1_")
    summary = argv[argv.index("--plain-summary") + 1]
    assert summary and not any(token in summary.lower() for token in JARGON)
    assert argv[argv.index("--probability-positive") + 1] == "0.126300000000"
    assert "e" not in argv[argv.index("--effect") + 1]
    assert "--replace" in argv


def test_the_restriction_reaches_the_declared_atoms_through_lane_ks_own_read():
    """End to end on a synthetic pool: key lines move, half-point lines do not."""

    pool_lines = np.repeat(np.arange(-14.0, 15.0), 40)
    pool_margins = np.tile(np.arange(-19.0, 21.0), 29)
    reads = {
        line: lane_k.band_read(pool_lines, pool_margins, line, line + 0.5, 2.5)
        for line in (3.0, 3.5, 7.0)
    }
    assert reads[3.0]["push"] > 0.0
    assert reads[7.0]["push"] > 0.0
    assert reads[3.5]["push"] == 0.0
    lines = pd.Series([3.0, 3.5, 7.0])
    served = np.array([0.5, 0.5, 0.5])
    mapped = np.array([reads[line]["home_cover_probability"] for line in lines])
    applied = restrict(lines, served, np.zeros(3), mapped, np.zeros(3), KEY_NUMBERS)
    assert list(applied["touched"]) == [True, False, True]
    assert applied["probability"][1] == 0.5
    assert applied["probability"][0] == mapped[0]
    assert applied["probability"][2] == mapped[2]

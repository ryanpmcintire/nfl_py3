"""MOD-18 lane V: the out-of-sample declaration of two post-hoc candidates.

The properties under test are the two declaration rules and the held-out
restriction. A rule that could look at a held-out season, or that could select
an atom or a bucket its own stated condition does not name, would silently give
back the discount this lane exists to remove.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import key_line_lattice_opener_eval as lane_t
import mass_preserving_lattice_opener_eval as lane_k
from out_of_sample_declaration_opener_eval import (
    ACTIVE_MODEL_ID,
    ARM_FAMILY,
    ARMS,
    KEY_ATOMS,
    SLOPE_CEILING,
    WINDOWS,
    apply_declaration,
    arm_sets,
    candidate_frame,
    cell_name,
    declare_atoms,
    declare_buckets,
    held_out_groups,
    plain_summary,
    record_argv,
    seasons_for,
    selected_atoms,
    selected_buckets,
    split_cell,
    write_research_artifact,
)

from nfl_ats.spread_regime import spread_bucket

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
    "probability rule",
    "pick_road",
    "pick_home",
    "standalone",
    "10.5+",
    "7.5-10",
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal scored frame carrying both frozen reads and the incumbent."""

    frame = pd.DataFrame(rows)
    frame["game_id"] = [f"g{i:03d}" for i in range(len(frame))]
    frame["push_S3"] = 0.02
    frame["push_MP1"] = 0.09
    frame["offset_S3"] = 0.1
    frame["shift_R1"] = 0.4
    frame["residual_S3"] = 1.0
    frame["residual_R1"] = 0.3
    frame["residual_at_open"] = 1.0
    frame["bucket"] = spread_bucket(frame.tue_open_home_spread)
    return frame


def _atom_rows(season: int, week: int, line: float, favours: str, count: int) -> list[dict]:
    """``count`` games on one atom where the named read makes the right pick.

    S3 always states 0.60 for the home side and lane K's read always states
    0.40, so the two disagree on every one of these games and the winner is
    decided purely by which way the game actually went.
    """

    margin = 4.0 if favours == "S3" else -4.0
    return [
        {
            "season": season,
            "week": week + index,
            "tue_open_home_spread": line,
            "p_S3": 0.60,
            "p_MP1": 0.40,
            "p_R1": 0.60,
            "margin_vs_open": margin,
        }
        for index in range(count)
    ]


def declaration_frame() -> pd.DataFrame:
    """2020 declares; 2021 is held out and would flip the 3 atom if it leaked."""

    rows: list[dict] = []
    # Atom 3: the incumbent is right three times in four -> negative delta.
    rows += _atom_rows(2020, 1, 3.0, "S3", 3)
    rows += _atom_rows(2020, 4, 3.0, "MP1", 1)
    # Atom 7: lane K's read is right twice -> strictly positive delta.
    rows += _atom_rows(2020, 1, 7.0, "MP1", 2)
    # Atom 10: the reads agree, so no pick moves -> delta exactly zero.
    rows += [
        {
            "season": 2020,
            "week": 1 + index,
            "tue_open_home_spread": 10.0,
            "p_S3": 0.60,
            "p_MP1": 0.60,
            "p_R1": 0.60,
            "margin_vs_open": 4.0,
        }
        for index in range(2)
    ]
    # Atom 14: one pick gained, one lost -> delta exactly zero, not positive.
    rows += _atom_rows(2020, 1, 14.0, "MP1", 1)
    rows += _atom_rows(2020, 2, 14.0, "S3", 1)
    # Untouched padding on a half-point line.
    rows += _atom_rows(2020, 1, 4.5, "S3", 4)
    # The held-out season, where the 3 atom goes the other way.
    rows += _atom_rows(2021, 1, 3.0, "MP1", 8)
    return _frame(rows)


def slope_stream() -> pd.DataFrame:
    """Exact linear relations, so every bootstrap draw returns the same slope."""

    rows = []
    for week in range(1, 13):
        for offset, line in ((1.0, 1.0), (2.0, 5.0), (3.0, 12.0)):
            residual = offset + week / 10.0
            slope = {1.0: 1.0, 5.0: SLOPE_CEILING, 12.0: -1.0}[line]
            rows.append(
                {
                    "season": 2020 + week % 3,
                    "week": week,
                    "spread_line": line,
                    "point_incumbent": line + residual,
                    "result": line + slope * residual,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[frame.season.eq(2023), :]  # no-op guard: 2023 rows are never built
    return frame


# ---------------------------------------------------------------------------
# Rule A: the atom set
# ---------------------------------------------------------------------------


def test_rule_a_selects_exactly_the_atoms_with_a_strictly_positive_delta():
    rows = declare_atoms(declaration_frame(), (2020, 2020))
    by_atom = {row["atom"]: row for row in rows}
    assert set(by_atom) == set(KEY_ATOMS)
    assert by_atom[3.0]["delta"] < 0.0 and not by_atom[3.0]["selected"]
    assert by_atom[7.0]["delta"] > 0.0 and by_atom[7.0]["selected"]
    # An atom whose reads agree, and an atom that gains one pick and loses one,
    # both score exactly zero. Zero is not positive, so neither is selected.
    assert by_atom[10.0]["delta"] == 0.0 and not by_atom[10.0]["selected"]
    assert by_atom[10.0]["flips"] == 0
    assert by_atom[14.0]["delta"] == 0.0 and not by_atom[14.0]["selected"]
    assert by_atom[14.0]["flips"] == 2
    assert selected_atoms(rows) == (7.0,)


def test_rule_a_never_reads_a_held_out_season():
    frame = declaration_frame()
    declared = declare_atoms(frame, (2020, 2020))
    leaked = declare_atoms(frame, (2020, 2021))
    # The held-out season alone would carry the 3 atom into the declared set.
    assert selected_atoms(declared) == (7.0,)
    assert selected_atoms(leaked) == (3.0, 7.0)


def test_both_rules_hold_only_their_own_declaration_seasons():
    frame = declaration_frame()
    for seasons in ((2020, 2020), (2020, 2021)):
        block = frame.loc[frame.season.between(*seasons)]
        assert set(block.season) <= set(range(seasons[0], seasons[1] + 1))
    # An empty declaration block is refused rather than silently selecting
    # nothing, which would look identical to a rule that found nothing.
    with pytest.raises(ValueError, match="holds no games"):
        declare_atoms(frame, (2019, 2019))
    with pytest.raises(ValueError, match="holds no completed games"):
        declare_buckets(slope_stream(), (2030, 2031))


# ---------------------------------------------------------------------------
# Rule B: the bucket set
# ---------------------------------------------------------------------------


def test_rule_b_selects_exactly_the_buckets_wholly_below_the_ceiling():
    rows = declare_buckets(slope_stream(), (2020, 2022))
    by_bucket = {row["bucket"]: row for row in rows}
    assert by_bucket["0-3"]["upper"] == pytest.approx(1.0)
    assert not by_bucket["0-3"]["selected"]
    # Exactly at the ceiling is NOT wholly below it.
    assert by_bucket["3.5-6.5"]["upper"] == pytest.approx(SLOPE_CEILING)
    assert not by_bucket["3.5-6.5"]["selected"]
    assert by_bucket["10.5+"]["upper"] == pytest.approx(-1.0)
    assert by_bucket["10.5+"]["selected"]
    # A bucket with no games has no estimable slope and is never selected.
    assert by_bucket["7"]["n"] == 0 and not by_bucket["7"]["selected"]
    assert selected_buckets(rows) == ("10.5+",)


def test_rule_b_never_reads_a_held_out_season():
    stream = slope_stream()
    held_out = stream.loc[stream.season.eq(2022)].copy()
    held_out["result"] = held_out.spread_line + 3.0 * (
        held_out.point_incumbent - held_out.spread_line
    )
    leaked = pd.concat([stream, held_out.assign(season=2021)], ignore_index=True)
    declared = declare_buckets(leaked, (2020, 2020))
    assert {row["bucket"] for row in declared if row["selected"]} == {"10.5+"}
    by_bucket = {row["bucket"]: row for row in declared}
    assert by_bucket["10.5+"]["upper"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The declared arms
# ---------------------------------------------------------------------------


def test_untouched_games_reproduce_the_incumbent_bit_for_bit():
    frame = declaration_frame()
    applied = apply_declaration(frame, (7.0,), ("10.5+",))
    untouched = ~applied["touched"]
    assert untouched.any()
    assert list(applied["probability"][untouched]) == list(frame.p_S3.to_numpy()[untouched])
    assert list(applied["offset"][untouched]) == list(frame.offset_S3.to_numpy()[untouched])
    assert list(applied["push"][untouched]) == list(frame.push_S3.to_numpy()[untouched])


def test_an_empty_declaration_is_the_incumbent_exactly():
    frame = declaration_frame()
    applied = apply_declaration(frame, (), ())
    assert not applied["touched"].any()
    assert list(applied["probability"]) == list(frame.p_S3.to_numpy())


def test_the_key_line_read_takes_precedence_over_the_bucket_read():
    """A line of exactly 14 is on an atom AND inside the 10.5+ bucket."""

    frame = _frame(_atom_rows(2020, 1, 14.0, "S3", 1) + _atom_rows(2020, 2, 11.5, "S3", 1))
    both = apply_declaration(frame, (14.0,), ("10.5+",))
    assert list(both["on_atom"]) == [True, False]
    assert list(both["in_bucket"]) == [False, True]
    assert both["probability"][0] == frame.p_MP1.iloc[0]
    assert both["probability"][1] == frame.p_R1.iloc[1]
    # The bucket read alone would take the 14 game instead.
    bucket_only = apply_declaration(frame, (), ("10.5+",))
    assert bucket_only["probability"][0] == frame.p_R1.iloc[0]


def test_atoms_are_matched_exactly_and_never_by_bucket():
    frame = _frame(
        _atom_rows(2020, 1, 7.0, "S3", 1)
        + _atom_rows(2020, 2, 6.75, "S3", 1)
        + _atom_rows(2020, 3, 7.25, "S3", 1)
        + _atom_rows(2020, 4, -7.0, "S3", 1)
    )
    applied = apply_declaration(frame, (7.0,), ())
    assert list(applied["on_atom"]) == [True, False, False, True]


def test_arm_sets_routes_each_arm():
    atoms, buckets = (3.0, 7.0), ("10.5+",)
    assert arm_sets("KL", atoms, buckets) == (atoms, ())
    assert arm_sets("RS", atoms, buckets) == ((), buckets)
    assert arm_sets("BOTH", atoms, buckets) == (atoms, buckets)
    assert set(ARMS) == {"KL", "RS", "BOTH"}


def test_the_s3_replay_gate_fails_closed():
    """The gate this lane runs on both frozen lanes before anything is built."""

    served = np.array([0.5, 0.6])
    assert lane_k.verify_replay(served.copy(), served) == 0.0
    with pytest.raises(ValueError, match="replay mismatch"):
        lane_k.verify_replay(np.array([0.5, 0.6 + 1e-6]), served)
    with pytest.raises(ValueError, match="missing probabilities"):
        lane_k.verify_replay(np.array([0.5, np.nan]), served)


# ---------------------------------------------------------------------------
# The held-out restriction and the research artifact
# ---------------------------------------------------------------------------


def test_held_out_groups_cover_the_seasons_and_the_touched_games():
    frame = declaration_frame()
    applied = apply_declaration(frame, (3.0,), ())
    frame["touched_W1_KL"] = applied["touched"]
    held = frame.loc[frame.season.eq(2021)].reset_index(drop=True)
    labels = [label for label, _ in held_out_groups(held, "W1_KL")]
    assert labels == ["overall", "season_2021", "touched"]
    sizes = {label: len(group) for label, group in held_out_groups(held, "W1_KL")}
    assert sizes["overall"] == sizes["season_2021"] == 8
    assert sizes["touched"] == 8


def test_candidate_frame_matches_lane_ts_research_per_game(tmp_path, monkeypatch):
    """The candidate columns are lane T's, not a lookalike written here."""

    frame = declaration_frame()
    applied = apply_declaration(frame, (7.0,), ("10.5+",))
    frame["p_W1_BOTH"] = applied["probability"]
    frame["offset_W1_BOTH"] = applied["offset"]
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "metadata.json").write_text(
        json.dumps({"active_model_id": "incumbent", "active_model_config": {"model_id": "inc"}})
    )
    monkeypatch.setattr(lane_t, "OUT", tmp_path / "laneT")
    directory = lane_t.research_per_game(frame, "W1_BOTH", archive)
    theirs = pd.read_parquet(directory / "per_game.parquet")
    mine = candidate_frame(frame, "W1_BOTH")
    for column in (
        "home_cover_probability_at_open",
        "home_side_offset_at_open",
        "pick_home_at_open",
        "pick_home_at_open_probability_rule",
        "correct_at_open",
        "correct_at_open_probability_rule",
    ):
        assert list(mine[column]) == list(theirs[column]), column


def test_the_research_artifact_never_claims_the_active_identity(tmp_path):
    frame = declaration_frame()
    applied = apply_declaration(frame, (7.0,), ())
    frame["p_W1_KL"] = applied["probability"]
    frame["offset_W1_KL"] = applied["offset"]
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "metadata.json").write_text(
        json.dumps(
            {"active_model_id": "a4c757efd2525add", "active_model_config": {"model_id": "a"}}
        )
    )
    import out_of_sample_declaration_opener_eval as lane_v

    original = lane_v.OUT
    lane_v.OUT = tmp_path / "laneV"
    try:
        directory = write_research_artifact(candidate_frame(frame, "W1_KL"), "W1_KL", archive)
    finally:
        lane_v.OUT = original
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["active_model_id"] == ACTIVE_MODEL_ID == "research_laneV_oos1"
    assert metadata["incumbent_model_id"] == "a4c757efd2525add"
    assert metadata["research_arm"] == "W1_KL"
    assert metadata["active_model_config"]["model_id"] == ACTIVE_MODEL_ID


# ---------------------------------------------------------------------------
# Registry rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "window", "arm", "label", "kind"),
    [
        ("oos1_w1_kl_overall_standalone", "W1", "KL", "overall", "standalone"),
        ("oos1_w1_rs_overall_card", "W1", "RS", "overall", "card"),
        ("oos1_w1_both_season_2024_card", "W1", "BOTH", "season_2024", "card"),
        ("oos1_w2_kl_touched_standalone", "W2", "KL", "touched", "standalone"),
        ("oos1_w2_rs_overall_brier", "W2", "RS", "overall", "brier"),
        ("oos1_w2_both_overall_log_loss", "W2", "BOTH", "overall", "log_loss"),
    ],
)
def test_every_cell_name_shape_splits_and_earns_a_plain_summary(name, window, arm, label, kind):
    assert split_cell(name) == (window, arm, label, kind)
    assert cell_name(window, arm, label, kind) == name
    summary = plain_summary(window, arm, label, kind, (3.0, 7.0), ("10.5+",))
    assert summary.strip()
    assert not any(token in summary.lower() for token in JARGON)


def test_an_unrecognised_cell_name_is_refused():
    for junk in ("oos1_w9_kl_overall_card", "oos1_w1_zz_overall_card", "oos1_w1_kl_overall_x"):
        with pytest.raises(ValueError, match="Unrecognised cell name"):
            split_cell(junk)


def test_a_cell_is_stamped_with_the_seasons_it_was_graded_on():
    assert seasons_for("W1", "overall") == (2024, 2025)
    assert seasons_for("W2", "overall") == (2023, 2025)
    assert seasons_for("W1", "season_2024") == (2024, 2024)
    assert seasons_for("W2", "touched") == (2023, 2025)
    # No held-out block may reach back into its own declaration seasons.
    for window, spec in WINDOWS.items():
        assert spec["holdout"][0] > spec["declare"][1]
        assert seasons_for(window, "overall")[0] > spec["declare"][1]


def test_a_summary_names_the_numbers_the_rule_actually_picked():
    only_seven = plain_summary("W1", "KL", "overall", "card", (7.0,), ())
    assert "exactly 7 points" in only_seven
    assert "exactly 3" not in only_seven
    assert "3, 7" not in only_seven
    empty = plain_summary("W1", "KL", "overall", "card", (), ())
    assert "picked nothing at all" in empty
    both = plain_summary("W2", "BOTH", "overall", "card", (3.0, 7.0), ("10.5+",))
    assert "Both changes at once" in both
    assert "10.5 points or more" in both


def test_recorder_argv_is_admissible_and_carries_a_plain_summary():
    metrics = {
        "delta": 0.5629,
        "lower": -0.1887,
        "upper": 1.3308,
        "probability_positive": 0.8798,
        "standard_error": 0.39,
        "n": 533,
        "weeks": 36,
    }
    argv = record_argv("oos1_w1_kl_overall_card", metrics, (7.0,), ("10.5+",))
    assert argv[:2] == ["weak-signals", "record"]
    assert argv[argv.index("--classification") + 1] == "unresolved_below_power"
    assert "--closing-ground" not in argv
    family = ARM_FAMILY["KL"]
    assert argv[argv.index("--family") + 1] == family
    name = argv[argv.index("--name") + 1]
    assert name == f"{family}_oos1_w1_kl_overall_card_2024_2025"
    assert "_oos1_" in name
    assert argv[argv.index("--effect-units") + 1] == "accuracy_points"
    assert argv[argv.index("--season-start") + 1] == "2024"
    summary = argv[argv.index("--plain-summary") + 1]
    assert summary and not any(token in summary.lower() for token in JARGON)
    # Scientific notation would be read by argparse as a flag; fixed point is not.
    for flag in ("--effect", "--interval-low", "--interval-high", "--probability-positive"):
        assert "e" not in argv[argv.index(flag) + 1]
    assert "--replace" in argv


def test_the_slope_arm_is_recorded_in_the_other_family():
    metrics = {
        "delta": -0.1876,
        "lower": -0.7708,
        "upper": 0.3795,
        "probability_positive": 0.1825,
        "standard_error": 0.3,
        "n": 533,
        "weeks": 36,
    }
    argv = record_argv("oos1_w1_rs_overall_standalone", metrics, (7.0,), ("10.5+",))
    assert argv[argv.index("--family") + 1] == ARM_FAMILY["RS"]
    assert ARM_FAMILY["RS"] != ARM_FAMILY["KL"]
    assert ARM_FAMILY["BOTH"] == ARM_FAMILY["KL"]
    assert argv[argv.index("--classification") + 1] == "unresolved_below_power"
    # A negative reading is still unresolved: no closing ground is ever claimed.
    assert "--closing-ground" not in argv


def test_every_emitted_command_parses_against_the_real_recorder():
    from nfl_ats.cli import build_parser

    parser = build_parser()
    metrics = {
        "delta": -0.000049,
        "lower": -0.0016,
        "upper": 0.0015,
        "probability_positive": 0.4735,
        "standard_error": 0.0008,
        "n": 533,
        "weeks": 36,
    }
    for name in ("oos1_w1_kl_overall_log_loss", "oos1_w2_both_touched_card"):
        argv = record_argv(name, metrics, (3.0, 7.0), ("10.5+",))
        parsed = parser.parse_args(argv)
        assert parsed.classification == "unresolved_below_power"
        assert parsed.closing_ground is None
        assert parsed.plain_summary

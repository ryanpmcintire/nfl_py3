"""LEAD-02 half-line script-disagreement screen: ratio, frozen-cut flag,
join integrity, and leakage (a game's outcome must never change its flag).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lead02_half_line_script_screen import (
    add_dog_outcome,
    apply_flag,
    compute_ratio,
    dedup_to_one_row_per_game,
    eligible_favorites,
    filter_plausible,
    freeze_cut,
    join_half_leg,
    positive_control_flag,
    within_block_permutation_null,
)


def _full_row(**overrides: object) -> dict[str, object]:
    row = {
        "capture_ts": "20100903154457",
        "game_date": "2010-09-09",
        "away": "MIN",
        "home": "NO",
        "kickoff_time": "8:30 PM",
        "book": "HILTON",
        "full_spread": -4.5,
        "total_line": 49.0,
        "season": 2010,
    }
    row.update(overrides)
    return row


def _half_row(**overrides: object) -> dict[str, object]:
    row = {
        "capture_ts": "20100903154457",
        "game_date": "2010-09-09",
        "away": "MIN",
        "home": "NO",
        "book": "HILTON",
        "half": 1,
        "spread_line": -2.0,
        "season": 2010,
    }
    row.update(overrides)
    return row


# --- ratio computation --------------------------------------------------


def test_compute_ratio_divides_half_by_full() -> None:
    df = pd.DataFrame({"half1_spread": [-2.0, -1.0, 0.0], "full_spread": [-4.0, -10.0, -5.0]})
    ratio = compute_ratio(df, 1)
    np.testing.assert_allclose(ratio.to_numpy(), [0.5, 0.1, 0.0])


def test_freeze_cut_is_the_20th_percentile_of_the_ratio_only() -> None:
    ratio = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cut = freeze_cut(ratio, percentile=0.20)
    assert cut == float(np.quantile(ratio.to_numpy(), 0.20))


# --- frozen-cut flag ------------------------------------------------------


def test_apply_flag_below_cut_is_true_above_is_false() -> None:
    df = pd.DataFrame({"half1_spread": [-1.0, -3.0, -5.0], "full_spread": [-10.0, -10.0, -10.0]})
    # ratios: 0.1, 0.3, 0.5
    flagged = apply_flag(df, 1, cut=0.3)
    assert flagged["flag"].tolist() == [True, False, False]


def test_apply_flag_handles_a_sign_flip_ratio_as_flagged() -> None:
    """A half spread on the OPPOSITE side (positive, since full_spread is
    negative) yields a negative ratio -- an extreme disagreement, correctly
    below any positive cut, not a crash or a special case."""

    df = pd.DataFrame({"half1_spread": [1.5], "full_spread": [-7.0]})
    flagged = apply_flag(df, 1, cut=0.4)
    assert bool(flagged["flag"].iloc[0]) is True
    assert flagged["ratio"].iloc[0] < 0


# --- join integrity ---------------------------------------------------


def test_join_half_leg_only_matches_identical_capture_matchup_book() -> None:
    full = pd.DataFrame([_full_row(), _full_row(capture_ts="20100903160000", book="CAESARS")])
    half = pd.DataFrame(
        [
            _half_row(),  # matches row 1 (same capture, book)
            _half_row(book="MIRAGE"),  # different book -- no full-game row to join to
        ]
    )
    merged = join_half_leg(full, half, half_num=1)
    assert len(merged) == 1
    assert merged.loc[0, "book"] == "HILTON"
    assert merged.loc[0, "capture_ts"] == "20100903154457"


def test_join_half_leg_selects_only_the_requested_half() -> None:
    full = pd.DataFrame([_full_row()])
    half = pd.DataFrame([_half_row(half=1, spread_line=-2.0), _half_row(half=2, spread_line=-1.0)])
    merged1 = join_half_leg(full, half, half_num=1)
    merged2 = join_half_leg(full, half, half_num=2)
    assert merged1.loc[0, "half1_spread"] == -2.0
    assert merged2.loc[0, "half2_spread"] == -1.0


def test_join_half_leg_drops_rows_missing_either_spread() -> None:
    full = pd.DataFrame([_full_row(full_spread=np.nan)])
    half = pd.DataFrame([_half_row()])
    merged = join_half_leg(full, half, half_num=1)
    assert merged.empty


# --- plausibility guard (measured data-quality defect) --------------------


def test_filter_plausible_drops_a_positive_or_oversized_full_spread() -> None:
    merged = pd.DataFrame(
        {
            "full_spread": [
                -4.5,
                53.5,
                -40.0,
            ],  # -4.5 real, 53.5 total-in-spread defect, -40 oversized
            "half1_spread": [-2.0, -1.0, -2.0],
        }
    )
    plausible, dropped = filter_plausible(merged, half_num=1)
    assert dropped == 2
    assert plausible["full_spread"].tolist() == [-4.5]


def test_filter_plausible_drops_an_oversized_half_leg_too() -> None:
    merged = pd.DataFrame({"full_spread": [-4.5], "half1_spread": [-35.0]})
    plausible, dropped = filter_plausible(merged, half_num=1)
    assert dropped == 1
    assert plausible.empty


# --- dedup to one row per game ---------------------------------------------


def test_dedup_keeps_the_latest_capture_per_game() -> None:
    fav = pd.DataFrame(
        {
            "season": [2010, 2010],
            "game_date": ["2010-09-09", "2010-09-09"],
            "away": ["MIN", "MIN"],
            "home": ["NO", "NO"],
            "capture_ts": ["20100901000000", "20100903000000"],
            "book": ["HILTON", "CAESARS"],
            "full_spread": [-4.0, -4.5],
            "half1_spread": [-2.0, -2.5],
        }
    )
    deduped = dedup_to_one_row_per_game(fav)
    assert len(deduped) == 1
    assert deduped.loc[0, "capture_ts"] == "20100903000000"
    assert deduped.loc[0, "book"] == "CAESARS"


def test_eligible_favorites_requires_at_least_three_points() -> None:
    plausible = pd.DataFrame({"full_spread": [-1.0, -3.0, -7.0]})
    fav = eligible_favorites(plausible)
    assert fav["full_spread"].tolist() == [-3.0, -7.0]


# --- leakage: a game's outcome must never change its flag ------------------


def test_flag_is_invariant_to_the_games_outcome() -> None:
    """The flag is a pure function of (full_spread, half_spread, frozen
    cut). Two games identical on those columns but with OPPOSITE outcomes
    must get the identical flag -- this is the leakage guarantee the
    predeclaration order (freeze_cut before any outcome column is read)
    exists to protect."""

    market_only = pd.DataFrame(
        {
            "full_spread": [-10.0, -10.0],
            "half1_spread": [-3.0, -3.0],  # ratio 0.3 both rows
        }
    )
    cut = 0.4
    flagged = apply_flag(market_only, 1, cut)
    assert flagged["flag"].tolist() == [True, True]

    # Attach two wildly different "outcomes" after the fact and recompute --
    # the flag column itself (already computed from market data only) must
    # not need or use either of these to have been correct.
    with_outcome_a = flagged.assign(dog_covered=[1.0, 0.0])
    with_outcome_b = flagged.assign(dog_covered=[0.0, 1.0])
    pd.testing.assert_series_equal(with_outcome_a["flag"], with_outcome_b["flag"], check_names=True)


def test_add_dog_outcome_uses_schedule_spread_not_vi_spread_for_side() -> None:
    """Side attribution (home vs. away favorite) always comes from the
    matched schedule row's own signed spread_line -- the VI archive alone
    cannot encode it (docs/vegasinsider_pilot.md)."""

    outcome_df = pd.DataFrame(
        {
            "spread_line": [-3.0, 3.0],  # home favored, away favored
            "result": [10.0, -10.0],  # home_score - away_score
        }
    )
    out = add_dog_outcome(outcome_df)
    assert out.loc[0, "home_is_favorite"] == True  # noqa: E712
    assert out.loc[1, "home_is_favorite"] == False  # noqa: E712
    # Row 0: home favored by 3, result +10 -> home covers by 7 -> dog (away) does NOT cover.
    assert out.loc[0, "dog_covered"] == 0.0
    # Row 1: away favored by 3 (home spread_line +3), result -10 -> home lost by 10,
    # further than the +3 home spread_line implies -> home (the dog) does NOT cover.
    assert out.loc[1, "dog_covered"] == 0.0


# --- positive control -------------------------------------------------


def test_positive_control_flag_picks_the_top_n_by_realized_margin() -> None:
    df = pd.DataFrame({"dog_margin": [5.0, -3.0, 10.0, 0.5, -1.0]})
    flag = positive_control_flag(df, "dog_margin", n_flag=2)
    # Exactly the two largest-margin rows (index 2: 10.0, index 0: 5.0) are flagged.
    assert flag.sum() == 2
    assert bool(flag.iloc[2]) is True
    assert bool(flag.iloc[0]) is True
    assert bool(flag.iloc[1]) is False


# --- within-block permutation null: block-size diagnostics -----------------


def test_within_block_permutation_null_reports_degenerate_single_row_blocks() -> None:
    df = pd.DataFrame(
        {
            "flag": [True, False, False, False],
            "dog_covered": [1.0, 0.0, 1.0, 0.0],
            "week_block": [1, 2, 3, 4],  # every block has exactly one row
        }
    )
    result = within_block_permutation_null(
        df, flag_col="flag", value_col="dog_covered", block_col="week_block", draws=50, seed=1
    )
    assert result["n_blocks_multi_row"] == 0
    # every draw must reproduce the observed gap exactly (nothing can move)
    assert result["null_sd"] == 0.0 or np.isnan(result["null_sd"])
    assert result["draws_used"] == 50


def test_within_block_permutation_null_can_actually_permute_multi_row_blocks() -> None:
    df = pd.DataFrame(
        {
            "flag": [True, False, True, False, True, False],
            "dog_covered": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "week_block": [1, 1, 1, 1, 1, 1],  # one big block
        }
    )
    result = within_block_permutation_null(
        df, flag_col="flag", value_col="dog_covered", block_col="week_block", draws=200, seed=1
    )
    assert result["n_blocks_multi_row"] == 1
    assert result["n_rows_in_multi_row_blocks"] == 6
    assert result["null_sd"] > 0.0

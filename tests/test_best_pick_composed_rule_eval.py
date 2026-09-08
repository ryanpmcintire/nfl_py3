"""POL-09 composed-rule scoring (``docs/best_pick_composed_rule.md``).

Three things are pinned:

1. The archive nominator reproduces the PRODUCTION nominee: on a synthetic
   week it agrees with ``nfl_ats.best_pick_nomination.select_nominee`` on the
   production pool, the big-spread discount excludes what
   ``apply_big_spread_eligibility`` excludes, and v1 comes from
   ``nfl_ats.best_pick.select_best_pick`` on the sweep.
2. Tie detection: a tie at the top of the filtered pool is reported with the
   tie-break that actually decided it (dispersion vs game_id), and the
   alphabetical fallback (``select_nominee_v3``) is carried alongside.
3. Leakage: a later week's rows can never change an earlier week's nominee.

Plus: ``dispersion_pool_from_frame`` (the helper factored out this session)
gives the same pool ``week_dispersion_pool`` gives production.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import nfl_ats.best_pick_nomination as bpn
from nfl_ats.best_pick_nomination import dispersion_pool_from_frame, week_dispersion_pool

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluator() -> ModuleType:
    path = REPO / "scripts" / "best_pick_composed_rule_eval.py"
    spec = importlib.util.spec_from_file_location("best_pick_composed_rule_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _week(
    season: int,
    week: int,
    rows: list[tuple[str, float, float, float, float]],
) -> pd.DataFrame:
    """rows: (game_id, candidate_dist, spread_std, tue_open_home_spread,
    baseline_correct_open). The candidate arm's correctness is set to the
    opposite of the baseline's so the two readings are distinguishable."""

    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "candidate_dist",
            "spread_std",
            "tue_open_home_spread",
            "baseline_correct_open",
        ],
    )
    frame["season"] = season
    frame["week"] = week
    frame["candidate_correct_open"] = 1.0 - frame["baseline_correct_open"]
    frame["abs_residual"] = frame["candidate_dist"] * 10.0
    frame["residual_at_open"] = frame["abs_residual"]
    frame["baseline_prob_open"] = 0.5 + frame["candidate_dist"]
    return frame


def _work(*weeks: pd.DataFrame, evaluator: ModuleType) -> pd.DataFrame:
    return evaluator.attach_production_pool(pd.concat(weeks, ignore_index=True))


# ---------------------------------------------------------------------------
# 1. Reproduction of the production nominee
# ---------------------------------------------------------------------------


def test_v2_nominee_is_the_production_select_nominee_on_the_production_pool(
    evaluator: ModuleType,
) -> None:
    # std: a=0.0, b=0.5, c=1.0, d=2.0 -> median 0.75 -> pool {a, b}. d has the
    # largest distance but is OUT of the pool; b wins inside it.
    week = _week(
        2021,
        3,
        [
            ("2021_03_A_B", 0.10, 0.0, 3.0, 1.0),
            ("2021_03_C_D", 0.20, 0.5, 6.5, 0.0),
            ("2021_03_E_F", 0.25, 1.0, 1.0, 1.0),
            ("2021_03_G_H", 0.30, 2.0, 4.0, 1.0),
        ],
    )
    work = _work(week, evaluator=evaluator)
    assert set(work.loc[work["pool_pass"], "game_id"]) == {"2021_03_A_B", "2021_03_C_D"}
    weekly = evaluator.nominate_composed(work)
    assert len(weekly) == 1
    row = weekly.iloc[0]
    assert row["v2_game_id"] == "2021_03_C_D"
    assert row["v2_n_tied"] == 1
    assert row["v2_tie_break"] == "none"
    # Scored on the ACTIVE model's pick, never the candidate arm's.
    assert row["v2_correct"] == 0.0
    assert row["v2_correct_candidate_arm"] == 1.0
    # The unfiltered choosers take the out-of-pool game.
    assert row["ch4_game_id"] == "2021_03_G_H"
    assert row["ch8_game_id"] == "2021_03_G_H"
    assert row["n_pool"] == 2
    # Direct cross-check against the production function on the same pool.
    pool = work.loc[work["pool_pass"], ["game_id", "candidate_dist", "spread_std"]]
    assert bpn.select_nominee(pool)[0] == row["v2_game_id"]


def test_big_spread_discount_excludes_ten_plus_and_falls_back_when_empty(
    evaluator: ModuleType,
) -> None:
    # std 0.0/0.5/2.0/3.0 -> median 1.25 -> pool {A_B (spread 10.0), C_D
    # (spread 9.5)}. A_B is the v2 nominee but sits exactly on the 10-point
    # boundary, so the discount moves to C_D.
    week = _week(
        2022,
        5,
        [
            ("2022_05_A_B", 0.30, 0.0, 10.0, 1.0),
            ("2022_05_C_D", 0.20, 0.5, 9.5, 0.0),
            ("2022_05_E_F", 0.10, 2.0, 1.0, 1.0),
            ("2022_05_G_H", 0.05, 3.0, 1.0, 1.0),
        ],
    )
    weekly = evaluator.nominate_composed(_work(week, evaluator=evaluator))
    row = weekly.iloc[0]
    assert row["v2_game_id"] == "2022_05_A_B"
    assert row["u2_game_id"] == "2022_05_C_D"
    assert row["u2_n_excluded"] == 1
    assert not row["u2_fallback_to_v2"]
    assert row["u2_correct"] == 0.0

    # Every pool game ({A_B, C_D}) is 10+: fall back to the unmodified v2 pool.
    week = _week(
        2022,
        6,
        [
            ("2022_06_A_B", 0.30, 0.0, -13.5, 1.0),
            ("2022_06_C_D", 0.20, 0.5, 10.0, 0.0),
            ("2022_06_E_F", 0.10, 2.0, 1.0, 1.0),
            ("2022_06_G_H", 0.05, 3.0, 1.0, 1.0),
        ],
    )
    row = evaluator.nominate_composed(_work(week, evaluator=evaluator)).iloc[0]
    assert row["u2_game_id"] == row["v2_game_id"] == "2022_06_A_B"
    assert row["u2_fallback_to_v2"]


def test_v1_comes_from_the_frozen_select_best_pick_on_the_sweep(evaluator: ModuleType) -> None:
    week = _week(
        2023,
        1,
        [
            ("2023_01_A_B", 0.10, 0.0, 3.0, 0.0),
            ("2023_01_C_D", 0.20, 0.5, 6.5, 1.0),
        ],
    )
    # A_B's pick survives a 2-point run around the quote; C_D only 1 point.
    offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    sweep = pd.DataFrame(
        [
            {"game_id": "2023_01_A_B", "line_offset": o, "home_cover_probability": p}
            for o, p in zip(offsets, [0.55, 0.6, 0.65, 0.6, 0.55], strict=True)
        ]
        + [
            {"game_id": "2023_01_C_D", "line_offset": o, "home_cover_probability": p}
            for o, p in zip(offsets, [0.4, 0.45, 0.7, 0.45, 0.4], strict=True)
        ]
    )
    sweep["season"] = 2023
    sweep["week"] = 1
    v1 = evaluator.nominate_v1(week, sweep)
    assert v1.iloc[0]["v1_game_id"] == "2023_01_A_B"
    assert v1.iloc[0]["v1_n_tied"] == 1
    assert v1.iloc[0]["v1_correct"] == 0.0


# ---------------------------------------------------------------------------
# 2. Tie detection
# ---------------------------------------------------------------------------


def test_a_tie_at_the_top_of_the_pool_is_reported_with_the_deciding_tie_break(
    evaluator: ModuleType,
) -> None:
    # std 0.5/0.0/1.0/2.0/3.0 -> median 1.0 -> pool {A_B, C_D}, which tie on
    # candidate_dist; C_D has the lower dispersion, A_B the earlier game_id.
    week = _week(
        2024,
        9,
        [
            ("2024_09_A_B", 0.30, 0.5, 3.0, 0.0),
            ("2024_09_C_D", 0.30, 0.0, 6.5, 1.0),
            ("2024_09_E_F", 0.10, 1.0, 1.0, 1.0),
            ("2024_09_G_H", 0.05, 2.0, 4.0, 1.0),
            ("2024_09_I_J", 0.05, 3.0, 4.0, 1.0),
        ],
    )
    row = evaluator.nominate_composed(_work(week, evaluator=evaluator)).iloc[0]
    assert row["v2_n_tied"] == 2
    assert row["v2_tie_break"] == "dispersion"
    assert row["v2_game_id"] == "2024_09_C_D"  # lower spread_std wins the tie
    assert row["v3_game_id"] == "2024_09_A_B"  # alphabetical fallback
    assert row["v3_tie_break"] == "game_id"
    audit = evaluator.tie_break_audit(evaluator.nominate_composed(_work(week, evaluator=evaluator)))
    assert audit["weeks_tied_at_top_of_pool"] == 1
    assert audit["weeks_dispersion_decided"] == 1
    assert audit["weeks_dispersion_nominee_differs_from_alphabetical"] == 1

    # Same dispersion on both tied games (and an empty strict filter, so the
    # whole week is the pool): falls through to game_id.
    week = _week(
        2024,
        10,
        [
            ("2024_10_A_B", 0.30, 0.0, 3.0, 0.0),
            ("2024_10_C_D", 0.30, 0.0, 6.5, 1.0),
            ("2024_10_E_F", 0.10, 2.0, 1.0, 1.0),
        ],
    )
    row = evaluator.nominate_composed(_work(week, evaluator=evaluator)).iloc[0]
    assert row["v2_n_tied"] == 2
    assert row["v2_tie_break"] == "game_id"
    assert row["v2_game_id"] == row["v3_game_id"] == "2024_10_A_B"


# ---------------------------------------------------------------------------
# 3. Leakage: later weeks cannot move an earlier nominee
# ---------------------------------------------------------------------------


def test_a_later_weeks_rows_cannot_change_an_earlier_weeks_nominee(evaluator: ModuleType) -> None:
    early = _week(
        2025,
        2,
        [
            ("2025_02_A_B", 0.10, 0.0, 3.0, 1.0),
            ("2025_02_C_D", 0.20, 0.5, 6.5, 0.0),
            ("2025_02_E_F", 0.25, 1.0, 1.0, 1.0),
            ("2025_02_G_H", 0.30, 2.0, 4.0, 1.0),
        ],
    )
    before = evaluator.nominate_composed(_work(early, evaluator=evaluator))
    # A later week with extreme values in every column the rule reads.
    late = _week(
        2025,
        3,
        [
            ("2025_03_A_B", 0.49, 0.0, 14.0, 0.0),
            ("2025_03_C_D", 0.49, 0.0, -14.0, 0.0),
            ("2025_03_E_F", 0.01, 9.0, 0.0, 0.0),
        ],
    )
    after = evaluator.nominate_composed(_work(early, late, evaluator=evaluator))
    first = after.loc[(after["season"] == 2025) & (after["week"] == 2)].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, before)
    # And the pool itself was built week-by-week (the later week's zero-std
    # games did not drag the earlier week's median down).
    work = _work(early, late, evaluator=evaluator)
    early_pool = set(work.loc[(work["week"] == 2) & work["pool_pass"], "game_id"])
    assert early_pool == {"2025_02_A_B", "2025_02_C_D"}


# ---------------------------------------------------------------------------
# 4. The factored-out pool helper matches production's pool
# ---------------------------------------------------------------------------


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

    # No missing data: strict below-median filter.
    frame = pd.DataFrame({"game_id": ["a", "b", "c"], "spread_std": [0.0, 0.5, 1.0]})
    pool = dispersion_pool_from_frame(frame)
    assert pool.fallback is False
    assert set(pool.frame.loc[pool.frame["pool_pass"], "game_id"]) == {"a"}
    with pytest.raises(ValueError, match="missing columns"):
        dispersion_pool_from_frame(pd.DataFrame({"game_id": ["a"]}))

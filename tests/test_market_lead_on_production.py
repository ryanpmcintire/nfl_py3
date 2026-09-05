"""Contracts for the two Phase 12 market-microstructure leads (LEAD-05,
LEAD-03): the shared opener-confirmation harness wiring
(``scripts/market_lead_on_production.py``, mirroring
``scripts/on_production_opener_confirmation.py``'s own tested contracts) and
the two candidate-column builders in ``nfl_ats.market_lead_features``.

Every fixture is built in memory (synthetic quote frames monkeypatched in
place of ``nfl_ats.clv.load_decision_quotes``): these tests must pass in a
fresh clone with no local ``data/market/raw`` snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import market_lead_on_production as harness  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats import market_lead_features as mlf  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402

UTC = "UTC"


# ---------------------------------------------------------------------------
# Shared harness wiring (mirrors tests/test_on_production_opener_confirmation.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_key", ["opener_softness", "ml_divergence"])
def test_profile_identity_is_production_plus_the_declared_one_column(candidate_key: str) -> None:
    candidate = harness.CANDIDATES[candidate_key]
    columns = set(margin_feature_columns("market_residual", candidate.profile))
    frame = pd.DataFrame({column: [0.0] for column in columns})

    observed = harness.profile_identity(candidate, frame)

    assert observed == {
        "baseline_columns": len(margin_feature_columns("market_residual", "weak_stack")),
        "candidate_columns": len(columns),
        "only_added_column": candidate.column,
    }


def test_scoped_window_refuses_training_that_reaches_the_screen(monkeypatch) -> None:
    training = pd.DataFrame({"season": [2019], "gameday": ["2020-09-01"], "result": [1]})
    window = pd.DataFrame({"season": [2020], "gameday": ["2020-09-01"], "result": [1]})
    monkeypatch.setattr(harness, "confirmation_split", lambda *_: (training, window))

    try:
        harness.scoped_window_frame(pd.DataFrame(), object(), "family")
    except ValueError as error:
        assert "leaked" in str(error)
    else:  # pragma: no cover - an assertion gives a clearer failure than a silent pass
        raise AssertionError("same-day training must be rejected")


@pytest.mark.parametrize("candidate_key", ["opener_softness", "ml_divergence"])
def test_only_positive_control_replaces_the_candidate_column(
    candidate_key: str, monkeypatch
) -> None:
    candidate = harness.CANDIDATES[candidate_key]
    source = pd.DataFrame({"season": [2020], "ats_margin": [7.0], candidate.column: [0.0]})
    captured: list[pd.DataFrame] = []

    def fake_evaluation(_market_root, frame, **_kwargs):
        captured.append(frame.copy())
        return pd.DataFrame({"season": [2020]})

    monkeypatch.setattr(harness, "opener_pick_evaluation", fake_evaluation)
    harness.run_arm(
        source,
        candidate,
        market_root=Path("market"),
        profile=candidate.profile,
        seasons=(2020,),
        min_train_games=1,
        leak=False,
    )
    harness.run_arm(
        source,
        candidate,
        market_root=Path("market"),
        profile=candidate.profile,
        seasons=(2020,),
        min_train_games=1,
        leak=True,
    )

    assert captured[0][candidate.column].tolist() == [0.0]
    assert captured[1][candidate.column].tolist() == [7.0]
    assert source[candidate.column].tolist() == [0.0]


# ---------------------------------------------------------------------------
# LEAD-03's >= 3 percentage-point threshold (pure function, no fixtures)
# ---------------------------------------------------------------------------


def test_divergence_threshold_maps_to_signed_signal() -> None:
    divergence = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, np.nan])

    signal = mlf._divergence_to_signal(divergence, threshold=0.03)

    # The boundary itself counts as a divergence (LEAD-03's ">=" / "<=" rule).
    assert signal.tolist()[:-1] == [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert np.isnan(signal[-1])


def test_divergence_threshold_is_symmetric_and_configurable() -> None:
    divergence = np.array([-0.10, -0.05, 0.05, 0.10])

    signal = mlf._divergence_to_signal(divergence, threshold=0.05)

    assert signal.tolist() == [-1.0, -1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Softest-book walk-forward identification (pure function, no fixtures)
# ---------------------------------------------------------------------------


def _errors_row(
    game_id: str, season: int, week: int, gameday: str, book: str, error: float
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "gameday": pd.Timestamp(gameday, tz=UTC),
        "bookmaker_key": book,
        "abs_error": error,
    }


def test_walk_forward_softest_book_uses_only_strictly_prior_games() -> None:
    # Week 1: book B is far softer than book A. Week 2's identification must
    # be based ONLY on week 1 (strictly earlier gameday); week 3's on weeks
    # 1-2. A book needs >= min_history_games PRIOR observations to be named.
    errors = pd.DataFrame(
        [
            _errors_row("G1", 2020, 1, "2020-09-10", "A", 0.5),
            _errors_row("G1", 2020, 1, "2020-09-10", "B", 5.0),
            _errors_row("G2", 2020, 2, "2020-09-17", "A", 0.5),
            _errors_row("G2", 2020, 2, "2020-09-17", "B", 5.0),
            _errors_row("G3", 2020, 3, "2020-09-24", "A", 0.5),
            _errors_row("G3", 2020, 3, "2020-09-24", "B", 5.0),
        ]
    )

    result = mlf.walk_forward_softest_book(errors, min_history_games=1)
    by_week = result.set_index("week")["softest_book"]

    # Week 1 has no strictly-prior history at all: nobody is eligible yet.
    assert pd.isna(by_week.loc[1])
    # Weeks 2 and 3 see book B's much larger prior error and name it softest.
    assert by_week.loc[2] == "B"
    assert by_week.loc[3] == "B"


def test_walk_forward_softest_book_respects_min_history_threshold() -> None:
    # Only ONE prior observation exists by week 2; requiring 2 must refuse to
    # name a softest book even though book B's single observation is huge.
    errors = pd.DataFrame(
        [
            _errors_row("G1", 2020, 1, "2020-09-10", "A", 0.5),
            _errors_row("G1", 2020, 1, "2020-09-10", "B", 5.0),
            _errors_row("G2", 2020, 2, "2020-09-17", "A", 0.5),
            _errors_row("G2", 2020, 2, "2020-09-17", "B", 5.0),
        ]
    )

    result = mlf.walk_forward_softest_book(errors, min_history_games=2)
    by_week = result.set_index("week")["softest_book"]

    assert pd.isna(by_week.loc[2])  # only 1 prior observation per book so far


def test_walk_forward_softest_book_never_uses_the_same_or_a_later_week() -> None:
    """A later week's games must never influence an earlier week's identification."""

    base = [
        _errors_row("G1", 2020, 1, "2020-09-10", "A", 0.5),
        _errors_row("G1", 2020, 1, "2020-09-10", "B", 1.0),
        _errors_row("G2", 2020, 2, "2020-09-17", "A", 0.5),
        _errors_row("G2", 2020, 2, "2020-09-17", "B", 1.0),
    ]
    before = mlf.walk_forward_softest_book(pd.DataFrame(base), min_history_games=1)
    week1_before = before.set_index("week")["softest_book"].loc[1]

    # A huge future (week 3) observation for book A must not reach back and
    # change week 1's identification (there IS no history before week 1).
    augmented = pd.DataFrame([*base, _errors_row("G3", 2020, 3, "2020-09-24", "A", 99.0)])
    after = mlf.walk_forward_softest_book(augmented, min_history_games=1)
    week1_after = after.set_index("week")["softest_book"].loc[1]

    assert pd.isna(week1_before) and pd.isna(week1_after)


# ---------------------------------------------------------------------------
# Synthetic quote-frame fixtures shared by the two flag-construction tests
# ---------------------------------------------------------------------------

COMMENCE = {
    "G1": pd.Timestamp("2020-09-10T17:00:00", tz=UTC),
    "G2": pd.Timestamp("2020-09-17T17:00:00", tz=UTC),
    "G3": pd.Timestamp("2020-09-24T17:00:00", tz=UTC),
}
OBSERVED = {
    "G1": pd.Timestamp("2020-09-08T09:00:00", tz=UTC),
    "G2": pd.Timestamp("2020-09-15T09:00:00", tz=UTC),
    "G3": pd.Timestamp("2020-09-22T09:00:00", tz=UTC),
}
GAME_WEEK = {"G1": 1, "G2": 2, "G3": 3}


def _quote_common(game_id: str, book: str) -> dict:
    return {
        "nflverse_game_id": game_id,
        "bookmaker_key": book,
        "observed_at_utc": OBSERVED[game_id],
        "commence_time_utc": COMMENCE[game_id],
        "season": 2020,
        "week": GAME_WEEK[game_id],
        "decision_label": "tue_open",
        "snapshot_timestamp_utc": OBSERVED[game_id],
        "capture_kind": "historical_backfill",
    }


def _spread_rows(game_id: str, book: str, home_spread: float) -> list[dict]:
    home_price, away_price = -110, -110
    return [
        {
            **_quote_common(game_id, book),
            "market": "spreads",
            "outcome_side": "HOME",
            "line": home_spread,
            "price": home_price,
            "home_spread_line": home_spread,
        },
        {
            **_quote_common(game_id, book),
            "market": "spreads",
            "outcome_side": "AWAY",
            "line": -home_spread,
            "price": away_price,
            "home_spread_line": home_spread,
        },
    ]


def _moneyline_rows(game_id: str, book: str, home_price: int, away_price: int) -> list[dict]:
    return [
        {
            **_quote_common(game_id, book),
            "market": "h2h",
            "outcome_side": "HOME",
            "line": np.nan,
            "price": home_price,
            "home_spread_line": np.nan,
        },
        {
            **_quote_common(game_id, book),
            "market": "h2h",
            "outcome_side": "AWAY",
            "line": np.nan,
            "price": away_price,
            "home_spread_line": np.nan,
        },
    ]


def _install_fake_quotes(monkeypatch, frame: pd.DataFrame) -> None:
    def fake_load_decision_quotes(_root, *, capture_kind=None, labels=None, seasons=None):
        working = frame
        if labels is not None:
            working = working.loc[working["decision_label"].isin(set(labels))]
        return working.reset_index(drop=True)

    monkeypatch.setattr(mlf, "load_decision_quotes", fake_load_decision_quotes)
    monkeypatch.setattr(clv, "load_decision_quotes", fake_load_decision_quotes)


def _schedule(games: dict[str, float]) -> pd.DataFrame:
    """``(game_id, season, week, gameday, spread_line)`` for the given games.

    ``spread_line`` doubles as the CLOSE fallback (no close-label quotes are
    supplied in these fixtures, so ``close_reference_table`` always falls
    back to the schedule).
    """

    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2020,
                "week": GAME_WEEK[game_id],
                "gameday": COMMENCE[game_id],
                "spread_line": close_spread,
            }
            for game_id, close_spread in games.items()
        ]
    )


def test_opener_softness_flag_construction_on_synthetic_quote_frames(monkeypatch) -> None:
    # Week 1 (G1): books A and C agree with the eventual close (-3.0); book B
    # misses it badly (its opener is +2.0), building up book B's history as
    # the softest book. Week 2 (G2): the CONSENSUS opener favors home
    # (median of -3.0(A), +2.0(B), -3.0(C) == -3.0), but the now-identified
    # softest book (B) posts an AWAY-favoring opener (+2.0) for this game --
    # a side disagreement, so the predeclared fade fires FOR the consensus's
    # side (home, +1.0).
    frame = pd.DataFrame(
        [
            *_spread_rows("G1", "A", -3.0),
            *_spread_rows("G1", "B", 2.0),
            *_spread_rows("G1", "C", -3.0),
            *_spread_rows("G2", "A", -3.0),
            *_spread_rows("G2", "B", 2.0),
            *_spread_rows("G2", "C", -3.0),
        ]
    )
    _install_fake_quotes(monkeypatch, frame)
    features = _schedule({"G1": -3.0, "G2": -3.0})

    result = mlf.derive_opener_softness_fade_features(
        features, market_root=Path("unused"), min_history_games=1
    )
    by_game = result.set_index("game_id")[mlf.OPENER_SOFTNESS_FADE_COLUMN]

    assert np.isnan(by_game.loc["G1"])  # no history yet: no softest book identified
    assert by_game.loc["G2"] == 1.0  # fades book B, siding with the home consensus


def test_opener_softness_flag_is_zero_on_agreement_and_nan_without_a_quote(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            *_spread_rows("G1", "A", -3.0),
            *_spread_rows("G1", "B", 2.0),
            *_spread_rows("G1", "C", -3.0),
            # G2: the softest book (B) AGREES with the consensus favorite (both home).
            *_spread_rows("G2", "A", -1.0),
            *_spread_rows("G2", "B", -4.0),
            *_spread_rows("G2", "C", -1.0),
            # G3: the softest book (B) never quotes this game at all.
            *_spread_rows("G3", "A", -1.0),
            *_spread_rows("G3", "C", -1.0),
        ]
    )
    _install_fake_quotes(monkeypatch, frame)
    features = _schedule({"G1": -3.0, "G2": -1.0, "G3": -1.0})

    result = mlf.derive_opener_softness_fade_features(
        features, market_root=Path("unused"), min_history_games=1
    )
    by_game = result.set_index("game_id")[mlf.OPENER_SOFTNESS_FADE_COLUMN]

    assert by_game.loc["G2"] == 0.0  # agreement: no fade
    assert np.isnan(by_game.loc["G3"])  # softest book has no quote for this game


def test_ml_spread_divergence_flag_construction_on_synthetic_quote_frames(monkeypatch) -> None:
    # Six training games (weeks 1-3, two per week) establish a clean walk
    # -forward logistic: home_spread <= -5 -> home always wins; >= +5 -> away
    # always wins, and every training game is priced -110/-110 (a 0.5
    # no-vig moneyline probability -- the moneyline carries no lean at all).
    # Week 4's game (S1) has home_spread == -5 (spread implies a heavy home
    # favorite once the logistic has seen the training games) but a
    # moneyline priced near even money: the moneyline implies home MUCH
    # WEAKER than the spread does, so the predeclared rule sides WITH the
    # moneyline and fades home (-1.0, away).
    frame_rows: list[dict] = []
    for game_id, week, spread in [
        ("T1", 1, -8.0),
        ("T2", 1, 8.0),
        ("T3", 2, -8.0),
        ("T4", 2, 8.0),
        ("T5", 3, -8.0),
        ("T6", 3, 8.0),
    ]:
        COMMENCE[game_id] = pd.Timestamp(f"2020-09-{10 + 7 * (week - 1)}T17:00:00", tz=UTC)
        OBSERVED[game_id] = pd.Timestamp(f"2020-09-{8 + 7 * (week - 1)}T09:00:00", tz=UTC)
        GAME_WEEK[game_id] = week
        frame_rows += _spread_rows(game_id, "A", spread)
        frame_rows += _moneyline_rows(game_id, "A", -110, -110)
    COMMENCE["S1"] = pd.Timestamp("2020-10-08T17:00:00", tz=UTC)
    OBSERVED["S1"] = pd.Timestamp("2020-10-06T09:00:00", tz=UTC)
    GAME_WEEK["S1"] = 4
    frame_rows += _spread_rows("S1", "A", -5.0)
    frame_rows += _moneyline_rows("S1", "A", -105, -115)  # near even money

    frame = pd.DataFrame(frame_rows)
    _install_fake_quotes(monkeypatch, frame)

    results = {
        "T1": True,
        "T2": False,
        "T3": True,
        "T4": False,
        "T5": True,
        "T6": False,
        "S1": True,
    }
    schedule_rows = []
    for game_id, home_won in results.items():
        schedule_rows.append(
            {
                "game_id": game_id,
                "season": 2020,
                "week": GAME_WEEK[game_id],
                "gameday": COMMENCE[game_id],
                "result": 7.0 if home_won else -7.0,
            }
        )
    features = pd.DataFrame(schedule_rows)

    result = mlf.derive_ml_spread_divergence_features(
        features, market_root=Path("unused"), min_train_games=2
    )
    by_game = result.set_index("game_id")[mlf.ML_SPREAD_DIVERGENCE_COLUMN]

    assert np.isnan(by_game.loc["T1"])  # week 1: no strictly-prior training games exist yet
    # Moneyline near even (~0.52 home) vs a spread-implied home win
    # probability the logistic pushes well above that once it has seen 6
    # perfectly separating training games -- a wide divergence with the
    # moneyline implying home MUCH weaker than the spread does, so the
    # predeclared rule sides WITH the moneyline and fades home.
    assert by_game.loc["S1"] == -1.0


# ---------------------------------------------------------------------------
# Leakage: a game's own outcome never changes its own flag
# ---------------------------------------------------------------------------


def test_ml_divergence_leakage_a_games_own_result_never_changes_its_own_flag() -> None:
    """``_walk_forward_spread_implied_home_win_probability`` is fit per week
    on STRICTLY earlier weeks only, so flipping a game's own ``result`` must
    never change that same game's own predicted probability (it can only
    change LATER weeks', which use it as training data)."""

    frame = pd.DataFrame(
        {
            "game_id": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "season": [2020] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "gameday": pd.to_datetime(
                [
                    "2020-09-10",
                    "2020-09-10",
                    "2020-09-17",
                    "2020-09-17",
                    "2020-09-24",
                    "2020-09-24",
                ],
                utc=True,
            ),
            "home_spread": [-8.0, 8.0, -8.0, 8.0, -8.0, 8.0],
            "result": [7.0, -7.0, 7.0, -7.0, 7.0, -7.0],
        }
    )

    baseline = mlf._walk_forward_spread_implied_home_win_probability(frame, min_train_games=2)

    flipped = frame.copy()
    # Flip week 2's own results (B1/B2) -- week 2's model was trained only on
    # week 1, so week 2's OWN predictions must be unaffected.
    flipped.loc[flipped["game_id"].isin(["B1", "B2"]), "result"] *= -1.0
    after = mlf._walk_forward_spread_implied_home_win_probability(flipped, min_train_games=2)

    week2_positions = frame.index[frame["week"].eq(2)].to_numpy()
    np.testing.assert_array_equal(baseline[week2_positions], after[week2_positions])

    # Week 1 (no training data exists before it either way) is also unaffected.
    week1_positions = frame.index[frame["week"].eq(1)].to_numpy()
    np.testing.assert_array_equal(baseline[week1_positions], after[week1_positions])

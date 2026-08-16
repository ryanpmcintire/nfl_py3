from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.data import DataContractError
from nfl_ats.margin import margin_feature_columns
from nfl_ats.market_decomposition import (
    FAMILY_PHRASES,
    INTERCEPT_FAMILY,
    MarketDecompositionResult,
    OpenerVariantResult,
    WalkForwardDecomposition,
    attribute_predictions,
    build_family_map,
    classify_families,
    classify_family,
    decomposition_feature_columns,
    explain_game,
    explain_game_structured,
    family_weights_table,
    latest_open_close_games_path,
    market_decomposition_markdown,
    opener_variant_decomposition,
    r_squared_table,
    reconciliation_summary,
    walk_forward_decomposition,
)

# ---------------------------------------------------------------------------
# Synthetic "planted family" fixture: `known` truly drives both margin and
# the market's spread; `hidden` truly drives margin but the market ignores
# it; `overpriced` drives the market's spread but has no true margin effect;
# `noise` drives neither. The signal-to-noise ratio is deliberately tight
# (small residual scales) so ridge alpha=10 recovers this structure reliably
# with a fixed seed -- a known-good configuration, not tuned per assertion.
# ---------------------------------------------------------------------------

SYNTHETIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "known": ("feat_known",),
    "hidden": ("feat_hidden",),
    "overpriced": ("feat_over",),
    "noise": ("feat_noise",),
}
SYNTHETIC_FEATURE_COLUMNS = ("feat_known", "feat_hidden", "feat_over", "feat_noise")
SYNTHETIC_SEASONS = (2020, 2021, 2022)
SYNTHETIC_WEEKS = 12
SYNTHETIC_GAMES_PER_WEEK = 16
SYNTHETIC_MIN_TRAIN_GAMES = 150


def _synthetic_decomposition_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    rows: list[dict[str, object]] = []
    index = 0
    for season in SYNTHETIC_SEASONS:
        base = pd.Timestamp(f"{season}-09-03")
        for week in range(1, SYNTHETIC_WEEKS + 1):
            gameday = base + pd.Timedelta(days=7 * (week - 1))
            for _ in range(SYNTHETIC_GAMES_PER_WEEK):
                known = float(rng.normal())
                hidden = float(rng.normal())
                over = float(rng.normal())
                noise = float(rng.normal())
                true_margin = 5.0 * known + 4.0 * hidden + float(rng.normal(scale=3.0))
                spread = 5.0 * known + 4.0 * over + float(rng.normal(scale=0.5))
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{index}",
                        "season": season,
                        "week": week,
                        "gameday": gameday,
                        "home_team": "HME",
                        "away_team": "AWY",
                        "result": true_margin,
                        "spread_line": spread,
                        "feat_known": known,
                        "feat_hidden": hidden,
                        "feat_over": over,
                        "feat_noise": noise,
                    }
                )
                index += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return _synthetic_decomposition_frame()


@pytest.fixture(scope="module")
def synthetic_walk_forward(synthetic_frame: pd.DataFrame) -> WalkForwardDecomposition:
    return walk_forward_decomposition(
        synthetic_frame,
        feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        start_season=2021,
        end_season=2022,
        min_train_games=SYNTHETIC_MIN_TRAIN_GAMES,
        families=SYNTHETIC_FAMILIES,
    )


# ---------------------------------------------------------------------------
# Family registry plumbing
# ---------------------------------------------------------------------------


def test_build_family_map_covers_real_market_family() -> None:
    mapping = build_family_map(["spread_line", "total_line"])
    assert mapping == {"spread_line": "market", "total_line": "market"}


def test_build_family_map_rejects_uncovered_feature() -> None:
    with pytest.raises(ValueError, match="family assignment"):
        build_family_map(["not_a_real_feature"])


def test_build_family_map_rejects_doubly_claimed_feature() -> None:
    with pytest.raises(ValueError, match="claimed by both"):
        build_family_map(["dup"], {"a": ("dup",), "b": ("dup",)})


def test_family_phrases_cover_every_registry_family() -> None:
    missing = set(FEATURE_FAMILIES) - set(FAMILY_PHRASES)
    assert not missing, f"families without a plain-English phrase: {sorted(missing)}"
    assert INTERCEPT_FAMILY in FAMILY_PHRASES


def test_decomposition_feature_columns_excludes_market_family() -> None:
    columns = decomposition_feature_columns("player")
    assert "spread_line" not in columns
    assert "total_line" not in columns
    assert columns == margin_feature_columns("margin", "player")


# ---------------------------------------------------------------------------
# Walk-forward matched regressions, reconciliation, R^2
# ---------------------------------------------------------------------------


def test_walk_forward_decomposition_reconciles_within_tolerance(
    synthetic_walk_forward: WalkForwardDecomposition,
) -> None:
    summary = reconciliation_summary(synthetic_walk_forward.reconciliation)
    assert summary["games"] == len(synthetic_walk_forward.reconciliation)
    assert summary["max_abs_error"] < 1e-6
    assert summary["mean_abs_error"] < 1e-6


def test_planted_families_classify_correctly(
    synthetic_walk_forward: WalkForwardDecomposition,
) -> None:
    weights = family_weights_table(synthetic_walk_forward.coefficients)
    classification = classify_families(weights).set_index("family")
    assert classification.loc["known", "classification"] == "priced"
    assert classification.loc["hidden", "classification"] == "unpriced_predictive"
    assert classification.loc["overpriced", "classification"] == "overpriced"
    assert classification.loc["noise", "classification"] == "noise"
    # The planted "hidden" family has real margin weight but ~zero spread weight.
    assert (
        classification.loc["hidden", "margin_share"] > classification.loc["hidden", "spread_share"]
    )
    # The planted "overpriced" family has real spread weight but ~zero margin weight.
    assert (
        classification.loc["overpriced", "spread_share"]
        > classification.loc["overpriced", "margin_share"]
    )


def test_r_squared_table_high_for_spread_lower_for_margin(
    synthetic_walk_forward: WalkForwardDecomposition,
) -> None:
    r_squared = r_squared_table(synthetic_walk_forward.predictions).set_index("target")
    # The spread target is nearly deterministic given `known`/`over` (residual
    # scale 0.5), so its out-of-sample R^2 should be high.
    assert r_squared.loc["spread", "r_squared"] > 0.9
    # The margin target has a much noisier generating process; it should
    # still be well above zero (the features are genuinely informative) but
    # far short of the spread target's fit.
    assert 0.0 < r_squared.loc["margin", "r_squared"] < r_squared.loc["spread", "r_squared"]


def test_r_squared_table_perfect_fit_scores_one() -> None:
    predictions = pd.DataFrame(
        {
            "target": ["margin", "margin", "margin"],
            "predicted": [1.0, 2.0, 3.0],
            "actual": [1.0, 2.0, 3.0],
        }
    )
    table = r_squared_table(predictions)
    assert table.loc[0, "r_squared"] == pytest.approx(1.0)


def test_family_weights_table_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        family_weights_table(
            pd.DataFrame(columns=["season", "week", "target", "family", "coefficient"])
        )


def test_reconciliation_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        reconciliation_summary(pd.DataFrame(columns=["season", "week", "game_id", "error"]))


def test_classify_families_requires_all_targets() -> None:
    incomplete = pd.DataFrame(
        {
            "target": ["margin"],
            "family": ["known"],
            "mean_abs_weight": [1.0],
            "share": [1.0],
            "mean_signed_weight": [1.0],
            "refit_std_abs_weight": [0.0],
            "season_std_abs_weight": [0.0],
        }
    )
    with pytest.raises(ValueError, match="missing targets"):
        classify_families(incomplete)


# ---------------------------------------------------------------------------
# classify_family boundary behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spread_share", "margin_share", "expected"),
    [
        (0.01, 0.01, "noise"),
        (0.01, 0.20, "unpriced_predictive"),
        (0.20, 0.01, "overpriced"),
        (0.15, 0.05, "overpriced"),  # ratio 3.0 >= default 1.5 threshold
        (0.10, 0.09, "priced"),  # ratio ~1.11 < default 1.5 threshold
    ],
)
def test_classify_family_buckets(spread_share: float, margin_share: float, expected: str) -> None:
    assert classify_family(spread_share, margin_share) == expected


def test_classify_family_rejects_out_of_range_shares() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        classify_family(1.5, 0.1)


# ---------------------------------------------------------------------------
# Per-game attribution
# ---------------------------------------------------------------------------


def test_attribute_predictions_sums_exactly_to_predicted_residual(
    synthetic_frame: pd.DataFrame,
) -> None:
    attribution = attribute_predictions(
        synthetic_frame,
        season=2022,
        week=6,
        feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        min_train_games=SYNTHETIC_MIN_TRAIN_GAMES,
        families=SYNTHETIC_FAMILIES,
    )
    assert set(attribution["family"].unique()) == {*SYNTHETIC_FAMILIES, INTERCEPT_FAMILY}
    for game_id, rows in attribution.groupby("game_id"):
        total = rows["contribution"].sum()
        predicted = rows["predicted_residual"].iloc[0]
        assert total == pytest.approx(predicted, abs=1e-6), game_id
        # The rendered explanation is identical across every family row for
        # the same game (it is a per-game, not a per-family, artifact).
        assert rows["explanation"].nunique() == 1


def test_attribute_predictions_raises_without_target_games(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No games found"):
        attribute_predictions(
            synthetic_frame,
            season=1999,
            week=1,
            feature_columns=SYNTHETIC_FEATURE_COLUMNS,
            min_train_games=SYNTHETIC_MIN_TRAIN_GAMES,
            families=SYNTHETIC_FAMILIES,
        )


def test_attribute_predictions_requires_a_spread(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    target_index = broken.loc[broken["season"].eq(2022) & broken["week"].eq(6)].index[0]
    broken.loc[target_index, "spread_line"] = np.nan
    with pytest.raises(DataContractError, match="target spread is missing"):
        attribute_predictions(
            broken,
            season=2022,
            week=6,
            feature_columns=SYNTHETIC_FEATURE_COLUMNS,
            min_train_games=SYNTHETIC_MIN_TRAIN_GAMES,
            families=SYNTHETIC_FAMILIES,
        )


# ---------------------------------------------------------------------------
# Plain-English explanations
# ---------------------------------------------------------------------------


def test_explain_game_single_driver_home_pick() -> None:
    explanation = explain_game_structured(
        game_id="g1",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=2.15,
        family_contributions={"known": 2.0, "hidden": 0.1, "noise": 0.05},
    )
    assert explanation.pick_side == "HOME"
    assert explanation.pick_team == "BUF"
    assert len(explanation.drivers) == 1
    assert explanation.drivers[0].family == "known"
    assert not explanation.offsets
    assert "BUF" in explanation.sentence
    assert "known" in explanation.sentence
    assert "hidden" not in explanation.sentence


def test_explain_game_driver_with_offset() -> None:
    sentence = explain_game(
        game_id="g2",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=1.0,
        family_contributions={"known": 2.0, "hidden": -1.0},
    )
    assert "partly offset by" in sentence
    assert "known (+2.0)" in sentence
    assert "hidden (-1.0)" in sentence


def test_explain_game_negligible_gap_degrades_gracefully() -> None:
    explanation = explain_game_structured(
        game_id="g3",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=0.2,
        family_contributions={"known": 3.0},
    )
    assert not explanation.drivers
    assert not explanation.offsets
    assert "essentially agrees with the market" in explanation.sentence


def test_explain_game_away_side_pick_reorients_sign() -> None:
    explanation = explain_game_structured(
        game_id="g4",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=-3.0,
        family_contributions={"known": -2.5},
    )
    assert explanation.pick_side == "AWAY"
    assert explanation.pick_team == "MIA"
    assert explanation.drivers[0].family == "known"
    assert explanation.drivers[0].points == pytest.approx(2.5)
    assert "MIA" in explanation.sentence
    assert "BUF" not in explanation.sentence


def test_explain_game_no_driver_clears_materiality_bar() -> None:
    explanation = explain_game_structured(
        game_id="g5",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=0.6,
        family_contributions={"known": 0.1},
    )
    assert not explanation.drivers
    assert "no single feature family clears" in explanation.sentence


def test_explain_game_caps_drivers_and_offsets() -> None:
    explanation = explain_game_structured(
        game_id="g6",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=10.0,
        family_contributions={
            "a": 5.0,
            "b": 4.0,
            "c": 3.0,
            "d": 2.0,
            "e": -1.0,
            "f": -2.0,
        },
    )
    assert [driver.family for driver in explanation.drivers] == ["a", "b", "c"]
    assert [offset.family for offset in explanation.offsets] == ["f"]


def test_phrase_fallback_for_unmapped_family_is_readable() -> None:
    sentence = explain_game(
        game_id="g7",
        home_team="BUF",
        away_team="MIA",
        predicted_residual=1.0,
        family_contributions={"some_custom_family": 1.0},
    )
    assert "some custom family" in sentence


# ---------------------------------------------------------------------------
# Opener variant
# ---------------------------------------------------------------------------


def _synthetic_opener_games(n: int = 80, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    game_ids = [f"open_{index}" for index in range(n)]
    feat_open = rng.normal(size=n)
    feat_absorb = rng.normal(size=n)
    opener = 3.0 * feat_open + rng.normal(scale=0.2, size=n)
    close = opener + 2.0 * feat_absorb + rng.normal(scale=0.2, size=n)
    opener_games = pd.DataFrame(
        {
            "nflverse_game_id": game_ids,
            "opening_home_spread": opener,
            "consensus_closing_home_spread": close,
        }
    )
    features = pd.DataFrame(
        {
            "game_id": game_ids,
            "feat_open_family": feat_open,
            "feat_absorb_family": feat_absorb,
        }
    )
    return opener_games, features


def test_opener_variant_decomposition_available() -> None:
    opener_games, features = _synthetic_opener_games()
    result = opener_variant_decomposition(
        opener_games,
        features,
        feature_columns=("feat_open_family", "feat_absorb_family"),
        min_games=50,
        families={"opener_family": ("feat_open_family",), "absorb_family": ("feat_absorb_family",)},
    )
    assert result.available
    assert result.games == 80
    assert result.family_weights is not None
    weights = result.family_weights.set_index(["target", "family"])
    assert (
        weights.loc[("open", "opener_family"), "share"]
        > weights.loc[("open", "absorb_family"), "share"]
    )
    assert (
        weights.loc[("absorption", "absorb_family"), "share"]
        > weights.loc[("absorption", "opener_family"), "share"]
    )


def test_opener_variant_decomposition_missing_columns() -> None:
    result = opener_variant_decomposition(
        pd.DataFrame({"nflverse_game_id": ["a"]}),
        pd.DataFrame({"game_id": ["a"]}),
        feature_columns=("feat",),
    )
    assert not result.available
    assert result.reason is not None
    assert "missing columns" in result.reason


def test_opener_variant_decomposition_too_few_games() -> None:
    opener_games, features = _synthetic_opener_games(n=10)
    result = opener_variant_decomposition(
        opener_games,
        features,
        feature_columns=("feat_open_family", "feat_absorb_family"),
        min_games=50,
        families={"opener_family": ("feat_open_family",), "absorb_family": ("feat_absorb_family",)},
    )
    assert not result.available
    assert result.games == 10
    assert result.reason is not None and "need at least" in result.reason


def test_latest_open_close_games_path(tmp_path: Path) -> None:
    assert latest_open_close_games_path(tmp_path / "missing") is None

    early = tmp_path / "20250101T000000Z"
    late = tmp_path / "20250601T000000Z"
    early.mkdir()
    late.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(early / "games.parquet")
    pd.DataFrame({"a": [2]}).to_parquet(late / "games.parquet")

    found = latest_open_close_games_path(tmp_path)
    assert found == late / "games.parquet"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_market_decomposition_markdown_renders_all_sections(
    synthetic_walk_forward: WalkForwardDecomposition,
) -> None:
    weights = family_weights_table(synthetic_walk_forward.coefficients)
    classification = classify_families(weights)
    r_squared = r_squared_table(synthetic_walk_forward.predictions)
    reconciliation = reconciliation_summary(synthetic_walk_forward.reconciliation)
    result = MarketDecompositionResult(
        feature_profile="synthetic",
        feature_columns=SYNTHETIC_FEATURE_COLUMNS,
        start_season=2021,
        end_season=2022,
        ridge_alpha=10.0,
        min_train_games=SYNTHETIC_MIN_TRAIN_GAMES,
        refit_weeks=synthetic_walk_forward.refit_weeks,
        coefficients=synthetic_walk_forward.coefficients,
        family_weights=weights,
        classification=classification,
        r_squared=r_squared,
        reconciliation=reconciliation,
        thresholds={"noise_share_threshold": 0.03, "overpriced_ratio_threshold": 1.5},
        opener_variant=OpenerVariantResult(
            available=False,
            reason="no opener sample supplied",
            games=0,
            coefficients=None,
            family_weights=None,
            r_squared=None,
        ),
    )
    attribution = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "family": ["known", INTERCEPT_FAMILY],
            "contribution": [1.0, 0.1],
            "explanation": ["The model leans HME 1.1 points more than the market."] * 2,
        }
    )
    markdown = market_decomposition_markdown(result, attribution=attribution)
    assert "# Market decomposition" in markdown
    assert "## R^2 accounting" in markdown
    assert "## Family classification" in markdown
    assert "noise_share_threshold" in markdown
    assert "known" in markdown
    assert "Not available: no opener sample supplied" in markdown
    assert "The model leans HME 1.1 points more than the market." in markdown
    assert "unpriced_predictive" in markdown

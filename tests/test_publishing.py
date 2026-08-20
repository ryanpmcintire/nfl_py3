from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_ats.publishing as publishing_module
from nfl_ats.best_pick_nomination import (
    NOMINATION_V2_METHOD_SENTENCE,
    DispersionPool,
    NominationV2Result,
)
from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from nfl_ats.publishing import BEST_PICK_MARK, publish_active_predictions
from nfl_ats.snapshots import write_snapshot


def _write_line_sweep(forecast: Path, widths: dict[str, float]) -> None:
    """A sweep where each game's pick holds >= 0.50 across ``2 * width`` points.

    Both fixture games are AWAY picks (``home_cover_probability`` below 0.5), and
    the sweep always reports the HOME probability, so the run is written as the
    complement.
    """

    offsets = np.arange(-4.0, 4.5, 0.5)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "game_id": game_id,
                    "line_offset": offsets,
                    "home_cover_probability": np.where(np.abs(offsets) <= width, 0.4, 0.6),
                    "method": "market_residual",
                }
            )
            for game_id, width in widths.items()
        ],
        ignore_index=True,
    ).to_parquet(forecast / "line_sweep.parquet")


def _write_active_publication_fixture(root: Path) -> tuple[Path, Path]:
    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["later", "earlier"],
            "gameday": ["2026-09-13", "2026-09-10"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [10.5, -3.5],
            "home_cover_probability": [0.38, 0.46],
            "method": ["market_residual", "market_residual"],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Project\n\nDescription.\n\n## Details\n", encoding="utf-8")
    return forecast, readme


def test_publish_active_predictions_updates_github_markdown_idempotently(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 8, 12, tzinfo=UTC)

    result = publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, published_at=instant
    )
    first_readme = readme.read_text(encoding="utf-8")
    publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, published_at=instant
    )

    assert result["model_id"] == "model-123"
    assert result["games"] == 2
    assert readme.read_text(encoding="utf-8") == first_readme
    assert first_readme.count("<!-- CURRENT_PREDICTIONS:START -->") == 1
    assert "**1,080 of 2,075 non-push games correctly (52.05%)**" in first_readme
    assert "distinct close-graded chronological" in first_readme
    assert "separate opener-graded production rule" in first_readme
    assert first_readme.index("SF at LA") < first_readme.index("ARI at LAC")
    assert "SF -3.5" in first_readme
    assert "ARI +10.5" in first_readme
    assert "Published from synchronized model `model-123`" in destination.read_text(
        encoding="utf-8"
    )


def test_published_card_marks_the_week_best_pick(tmp_path: Path) -> None:
    """POL-10: the card the user reads at pick time must name the Best Pick.

    The ledger answers "what did we choose?" months later; only the card
    answers "which one do I enter today?".
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_game_id"] == "later"
    card = destination.read_text(encoding="utf-8")
    assert f"{BEST_PICK_MARK}ARI +10.5" in card
    assert "Best Pick of the week" in card
    assert "ARI +10.5 in ARI at LAC" in card
    # Exactly one row is marked (the other occurrence is the note's legend).
    assert card.count(BEST_PICK_MARK) == 1
    assert card.count(BEST_PICK_MARK.strip()) == 2


def test_published_card_discloses_a_tied_best_pick(tmp_path: Path) -> None:
    """POL-09/POL-10: an undisclosed tie is not a lean.

    The dashboard (nfl_ats.dashboard.app_pages.picks) already shows this
    disclosure; the published card must show the identical sentence via the
    same nfl_ats.best_pick.best_pick_tie_note the dashboard calls, so the two
    surfaces cannot silently disagree about whether a nomination is arbitrary.
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    # "later" and "earlier" both hold to the same width -- a two-way tie.
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 3.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_tied"] is True
    card = destination.read_text(encoding="utf-8")
    assert "2 games tie at the top of that signal" in card
    assert "reproducible, but not a lean" in card
    readme_text = readme.read_text(encoding="utf-8")
    assert "2 games tie at the top of that signal" in readme_text


def test_published_card_does_not_disclose_an_unambiguous_best_pick(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_tied"] is False
    assert "tie at the top" not in destination.read_text(encoding="utf-8")


def test_published_card_without_a_sweep_names_no_best_pick(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)
    assert result["best_pick_game_id"] is None
    assert BEST_PICK_MARK not in destination.read_text(encoding="utf-8")


def _tenure_schedules_for_overlay() -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
    ]
    rows = [
        ("2025_01_KEEP_OPP", 2025, "REG", 1, "KEEP", "OPP", "Steady", "OppC"),
        ("2025_01_YR1_OPP2", 2025, "REG", 1, "YR1", "OPP2", "Old1", "OppC2"),
        ("2026_01_KEEP_YR1", 2026, "REG", 1, "KEEP", "YR1", "Steady", "New1"),
        ("2026_01_OTHER1_OTHER2", 2026, "REG", 1, "OTHER1", "OTHER2", "X", "Y"),
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_overlay_publication_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Like ``_write_active_publication_fixture``, plus the season/week/
    game_type columns and a local schedule snapshot the overlay needs. One
    game (KEEP hosting YR1, a year-1 coach's team) is the clean flip
    candidate; the other is an unrelated control game the overlay must not
    touch."""

    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)
    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["2026_01_KEEP_YR1", "2026_01_OTHER1_OTHER2"],
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "gameday": ["2026-09-10", "2026-09-10"],
            "away_team": ["YR1", "OTHER2"],
            "home_team": ["KEEP", "OTHER1"],
            "spread_line": [-3.5, 2.5],
            # KEEP (home, kept coach) is NOT picked -- YR1 (away, year-1) is.
            "home_cover_probability": [0.35, 0.55],
            "method": ["market_residual", "market_residual"],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "player",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Project\n\nDescription.\n\n## Details\n", encoding="utf-8")

    data_root = root / "data"
    write_snapshot(
        _tenure_schedules_for_overlay(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )
    return forecast, readme, data_root


def test_published_card_applies_and_discloses_the_coach_fade_overlay(tmp_path: Path) -> None:
    """The overlay (docs/coach_fade_overlay.md) flips the clean-case pick and
    discloses it in the card's provenance, the same plain way Best Pick ties
    are disclosed."""

    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, data_root=data_root
    )

    assert result["overlay_enabled"] is True
    assert result["overlay_flip_count"] == 1
    assert result["overlay_flipped_game_ids"] == ["2026_01_KEEP_YR1"]

    card = destination.read_text(encoding="utf-8")
    assert "**Overlay applied: 1 pick flipped**" in card
    assert "YR1 -> KEEP" in card
    assert "KEEP +3.5" in card
    assert "YR1 -3.5" not in card
    readme_text = readme.read_text(encoding="utf-8")
    assert "**Overlay applied: 1 pick flipped**" in readme_text


def test_published_card_without_a_data_root_leaves_the_overlay_off(tmp_path: Path) -> None:
    """``data_root`` is the overlay's explicit opt-in: omit it and the card
    publishes exactly as it would with no overlay wired in at all."""

    _, readme, _data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["overlay_enabled"] is False
    assert result["overlay_flip_count"] == 0
    card = destination.read_text(encoding="utf-8")
    assert "Overlay applied" not in card
    assert "YR1 -3.5" in card


def test_publish_rejects_weekly_model_id_mismatch(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    metadata["active_model_id"] = "wrong-model"
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model ID does not match"):
        publish_active_predictions(
            tmp_path,
            destination=tmp_path / "CURRENT_PREDICTIONS.md",
            readme_path=readme,
        )


# ---------------------------------------------------------------------------
# POL-09 2026-08-18: the v2 Best Pick nomination rule
# ---------------------------------------------------------------------------


def _write_v2_capable_fixture(root: Path) -> tuple[Path, Path, Path, list[str]]:
    """A real, walk-forward-fittable feature table plus a matching card and
    a local Tuesday-opener market snapshot with genuinely DIFFERENT
    cross-book dispersion per game -- everything ``_nomination_v2`` needs to
    compute v2 for real, end to end, no mocking of the model fit itself."""

    forecast = root / "margin_predictions" / "forecast"
    forecast.mkdir(parents=True)

    game_ids = ["2026_01_AAA_BBB", "2026_01_CCC_DDD", "2026_01_EEE_FFF"]
    train_rows = 150
    total = train_rows + len(game_ids)
    start = date(2019, 9, 1)
    index = np.arange(total)
    features = pd.DataFrame(
        {
            "game_id": [f"train_{v:03d}" for v in range(train_rows)] + game_ids,
            "season": np.where(index < train_rows, 2019, 2026),
            "week": np.where(index < train_rows, (index // 15) + 1, 1),
            "gameday": [start + timedelta(days=int(v)) for v in range(train_rows)]
            + [date(2026, 9, 10)] * len(game_ids),
            "away_team": "AWY",
            "home_team": "HME",
        }
    )
    all_features = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)
    for feature_index, column in enumerate(all_features, start=1):
        features[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    features["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    rng = np.random.default_rng(20260818)
    features["ats_margin"] = rng.normal(loc=0.0, scale=8.0, size=total)
    features["home_cover"] = (features["ats_margin"] > 0).astype(float)
    features["result"] = features["spread_line"] + features["ats_margin"]
    features.loc[index >= train_rows, ["home_cover", "ats_margin", "result"]] = np.nan
    features_path = root / "v2_features.parquet"
    features.to_parquet(features_path)

    metadata = {
        "active_model_id": "model-123",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "feature_profile": "base",
        "regressor": "ridge",
        "min_train_games": 100,
        "provenance": {"feature_table": {"path": str(features_path)}},
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    predictions = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2026,
            "week": 1,
            "game_type": "REG",
            "gameday": ["2026-09-10"] * len(game_ids),
            "away_team": ["AWY1", "AWY2", "AWY3"],
            "home_team": ["HME1", "HME2", "HME3"],
            "spread_line": [2.5, -2.5, 2.5],
            "home_cover_probability": [0.30, 0.60, 0.45],
            "method": "market_residual",
        }
    )
    predictions.to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "target": "ats_classification",
        "model_id": "model-123",
        "method": "market_residual",
        "feature_profile": "base",
        "regressor": "ridge",
        "historical_evaluation": {
            "artifact": "margins/evaluation",
            "accuracy": 0.5205,
            "correct": 1080,
            "games": 2075,
            "intervals": {"week": {"lower": 0.4985, "upper": 0.5425}},
        },
        "weekly_forecast": {
            "artifact": "margin_predictions/forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (root / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Project\n\nDescription.\n\n## Details\n", encoding="utf-8")

    data_root = root / "data"
    snapshot_dir = data_root / "market" / "raw" / "20260818T130000Z"
    snapshot_dir.mkdir(parents=True)
    tuesday = pd.Timestamp("2026-08-18T13:00:00Z")
    kickoff = pd.Timestamp("2026-09-10T17:00:00Z")
    # Genuinely different dispersion per game, so the filter is real, not a
    # missing-data fallback.
    book_lines = {
        game_ids[0]: [2.5, 2.5],
        game_ids[1]: [-2.5, -3.0],
        game_ids[2]: [2.5, 4.5],
    }
    quotes_rows = [
        {
            "nflverse_game_id": game_id,
            "provider_event_id": game_id,
            "bookmaker_key": f"book{i}",
            "market": "spreads",
            "outcome_side": "HOME",
            "home_spread_line": line,
            "observed_at_utc": tuesday,
            "commence_time_utc": kickoff,
        }
        for game_id, lines in book_lines.items()
        for i, line in enumerate(lines)
    ]
    pd.DataFrame(quotes_rows).to_parquet(snapshot_dir / "quotes.parquet")
    return forecast, readme, data_root, game_ids


def test_published_card_uses_v2_nomination_end_to_end(tmp_path: Path) -> None:
    """The real thing: a real walk-forward alpha=2000 fit, a real dispersion
    pool from a local market snapshot, no mocking anywhere in the chain."""

    _, readme, data_root, game_ids = _write_v2_capable_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, data_root=data_root
    )

    assert result["best_pick_nomination_rule"] == "v2"
    assert result["best_pick_nomination_v2_available"] is True
    assert result["best_pick_nomination_v2_game_id"] in game_ids
    assert result["best_pick_game_id"] == result["best_pick_nomination_v2_game_id"]
    # v1's own (unchanged) nomination is still reported for the old-vs-new
    # audit even though it is not the one marked on the card this week.
    assert (
        result["best_pick_nomination_v1_game_id"] is None
    )  # no line_sweep.parquet in this fixture

    card = destination.read_text(encoding="utf-8")
    assert NOMINATION_V2_METHOD_SENTENCE in card
    assert "This pick was nominated by calibrated probability among low-disagreement games" in card


def test_published_card_falls_back_to_v1_when_v2_infrastructure_is_absent(
    tmp_path: Path,
) -> None:
    """No data_root at all -- the same degrade contract the overlay uses."""

    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = publish_active_predictions(tmp_path, destination=destination, readme_path=readme)

    assert result["best_pick_nomination_rule"] == "v1"
    assert result["best_pick_nomination_v2_available"] is False
    assert result["best_pick_nomination_v2_game_id"] is None


def test_v2_nomination_and_the_coach_fade_overlay_do_not_interfere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independence property, end to end: the overlay flips a DIFFERENT
    game's side while v2 (mocked here to a fixed nominee, since the ranking
    rule itself is pinned unit-by-unit in test_best_pick_nomination.py)
    nominates its own game -- neither lever's output moves because the other
    exists. Best Pick selection runs on the UN-overlaid card either way."""

    forecast, readme, data_root, game_ids = _write_v2_capable_fixture(tmp_path)
    # Overwrite the card so ONE game is a clean year-1-coach fade candidate,
    # distinct from every game v2 will ever be told to nominate.
    predictions = pd.read_csv(forecast / "recommendations.csv")
    predictions["home_team"] = ["KEEP", "HME2", "HME3"]
    predictions["away_team"] = ["YR1", "AWY2", "AWY3"]
    predictions["home_cover_probability"] = [0.35, 0.60, 0.45]  # KEEP@home is NOT picked -> flips
    predictions.to_csv(forecast / "recommendations.csv", index=False)

    schedules = pd.DataFrame(
        [
            ("2025_01_YR1_OPP", 2025, "REG", 1, "YR1", "OPP", "Old1", "OppC"),
            (game_ids[0], 2026, "REG", 1, "KEEP", "YR1", "Steady", "New1"),
        ],
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
        ],
    )
    write_snapshot(
        schedules,
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )

    fixed_v2 = NominationV2Result(
        game_id=game_ids[2],
        n_tied_at_max=1,
        tie_break="none",
        probability_table=pd.DataFrame(),
        dispersion=DispersionPool(pd.DataFrame(), False, None, 3, 0, 2),
    )
    monkeypatch.setattr(publishing_module, "nominate_v2", lambda *a, **k: fixed_v2)

    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    result = publish_active_predictions(
        tmp_path, destination=destination, readme_path=readme, data_root=data_root
    )

    # v2 nominated its own game, unmoved by the overlay flipping a different one.
    assert result["best_pick_nomination_v2_game_id"] == game_ids[2]
    assert result["best_pick_game_id"] == game_ids[2]
    # The overlay flipped the year-1-coach game, unmoved by v2 existing.
    assert result["overlay_flip_count"] == 1
    assert result["overlay_flipped_game_ids"] == [game_ids[0]]

    card = destination.read_text(encoding="utf-8")
    assert "Overlay applied: 1 pick flipped" in card
    assert NOMINATION_V2_METHOD_SENTENCE in card

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
from nfl_ats.clv import load_paper_decisions, record_paper_decisions
from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from nfl_ats.dashboard import findings_content
from nfl_ats.data import DataContractError
from nfl_ats.provenance import sha256_file
from nfl_ats.publishing import BEST_PICK_MARK, publish_active_predictions
from nfl_ats.snapshots import latest_snapshot, write_snapshot


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
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
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


def test_publication_exposes_frozen_best_pick_inputs_without_a_shadow_card(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    result = _publish_with_fresh_empty_arrest(tmp_path, destination=destination, readme_path=readme)
    inputs = result["best_pick_prospective_input"]
    probabilities = pd.DataFrame(inputs["predictions"])
    assert set(probabilities["game_id"]) == {"later", "earlier"}
    assert probabilities["home_cover_probability"].between(0, 1).all()
    assert set(pd.DataFrame(inputs["pool"])["game_id"]) == {"later", "earlier"}
    assert result["best_pick_game_id"] == "later"
    text = destination.read_text(encoding="utf-8")
    assert "shade" not in text.lower()
    assert "re-nomination" not in text.lower()
    assert not (tmp_path / "prospective" / "best_pick_refresh_decisions.parquet").exists()


def test_publish_active_predictions_updates_github_markdown_idempotently(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 8, 12, tzinfo=UTC)

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=instant,
    )
    first_readme = readme.read_text(encoding="utf-8")
    _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=instant,
    )

    assert result["model_id"] == "model-123"
    assert result["games"] == 2
    assert readme.read_text(encoding="utf-8") == first_readme
    assert first_readme.count("<!-- CURRENT_PREDICTIONS:START -->") == 1
    assert "**1,080 of 2,075 non-push games correctly (52.05%)**" in first_readme
    assert "distinct close-graded chronological" in first_readme
    assert "separate opener-graded accuracy rule" in first_readme
    assert "**Production policy active:**" in first_readme
    assert "four situational rules" in first_readme
    # Pins the SUBSTANCE of the disclosure, not one phrasing: the archive
    # score must be named a ceiling, and the card must carry the same
    # de-inflated planning estimate the rest of the site publishes rather
    # than a second number of its own.
    assert "best of 127 similar combinations" in first_readme
    assert "never an expectation" in first_readme
    assert findings_content.PLAYED_CARD_EXPECTATION_HERO in first_readme
    assert first_readme.index("SF at LA") < first_readme.index("ARI at LAC")
    assert "SF -3.5" in first_readme
    assert "ARI +10.5" in first_readme
    # No raw model id/hash in the card's own header (owner mandate,
    # 2026-09-05) -- the humanized method label instead.
    assert "Published from the synchronized player model" in destination.read_text(encoding="utf-8")


def test_publish_active_predictions_persists_source_policy_json(tmp_path: Path) -> None:
    """ENG-34 follow-up: the ENG-14 ``source_policy`` block is persisted
    additively as ``source_policy.json`` beside the forecast artifact (next
    to ``explanations.json``) AND beside the published card (next to
    ``lineage.json``) -- and the forecast's own ``metadata.json`` is left
    byte-identical, since its digest is pinned by the lock-day package and
    replay."""

    forecast, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    metadata_before = (forecast / "metadata.json").read_text(encoding="utf-8")

    _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    forecast_copy = forecast / "source_policy.json"
    card_copy = destination.parent / "source_policy.json"
    assert forecast_copy.is_file()
    assert card_copy.is_file()
    forecast_block = json.loads(forecast_copy.read_text(encoding="utf-8"))
    card_block = json.loads(card_copy.read_text(encoding="utf-8"))
    assert forecast_block == card_block
    assert forecast_block["state"] in {"complete", "degraded", "blocked"}
    assert "sources" in forecast_block
    assert "evaluated_at_utc" in forecast_block
    assert (forecast / "metadata.json").read_text(encoding="utf-8") == metadata_before


def test_publish_active_predictions_also_refreshes_readme_state_blocks(tmp_path: Path) -> None:
    """publish-predictions owns the README, so it must refresh ALL of its
    generated blocks in the same write, not just CURRENT_PREDICTIONS."""

    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    registry_root = tmp_path / "registry"

    _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
        registry_root=registry_root,
    )

    readme_text = readme.read_text(encoding="utf-8")
    assert readme_text.count("<!-- ACTIVE_MODEL_STATE:START -->") == 1
    assert readme_text.count("<!-- RESEARCH_STATE:START -->") == 1
    assert "`model-123`" in readme_text
    # No registry files exist under registry_root in this fixture, so the
    # research-state block must degrade honestly rather than fabricate counts.
    assert "0 results recorded yet" in readme_text


def test_published_card_marks_the_week_best_pick(tmp_path: Path) -> None:
    """POL-10: the card the user reads at pick time must name the Best Pick.

    The ledger answers "what did we choose?" months later; only the card
    answers "which one do I enter today?".
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 1.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )

    assert result["best_pick_game_id"] == "later"
    card = destination.read_text(encoding="utf-8")
    assert f"{BEST_PICK_MARK}ARI +10.5" in card
    assert "Best Pick of the week" in card
    assert "ARI +10.5 in ARI at LAC" in card
    # Exactly one row is marked (the other occurrence is the note's legend).
    assert card.count(BEST_PICK_MARK) == 1
    assert card.count(BEST_PICK_MARK.strip()) == 2


def test_published_card_discloses_a_tied_best_pick(tmp_path: Path) -> None:
    """POL-09/POL-10: an undisclosed tie is not a lean. The published card and the
    public site must show the identical sentence via the same
    nfl_ats.best_pick.best_pick_tie_note, so the surfaces cannot silently
    disagree about whether a nomination is arbitrary.
    """

    forecast, readme = _write_active_publication_fixture(tmp_path)
    # "later" and "earlier" both hold to the same width -- a two-way tie.
    _write_line_sweep(forecast, {"later": 3.0, "earlier": 3.0})
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )

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

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )

    assert result["best_pick_tied"] is False
    assert "tie at the top" not in destination.read_text(encoding="utf-8")


def test_published_card_without_a_sweep_names_no_best_pick(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )
    assert result["best_pick_game_id"] is None
    assert BEST_PICK_MARK not in destination.read_text(encoding="utf-8")


def _tenure_schedules_for_overlay() -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
        "result",
    ]
    rows = [
        ("2025_01_KEEP_OPP", 2025, "REG", 1, "2025-09-07", "KEEP", "OPP", "Steady", "OppC", 3.0),
        ("2025_01_YR1_OPP2", 2025, "REG", 1, "2025-09-07", "YR1", "OPP2", "Old1", "OppC2", -3.0),
        ("2026_01_KEEP_YR1", 2026, "REG", 1, "2026-09-10", "KEEP", "YR1", "Steady", "New1", np.nan),
        (
            "2026_01_OTHER1_OTHER2",
            2026,
            "REG",
            1,
            "2026-09-10",
            "OTHER1",
            "OTHER2",
            "X",
            "Y",
            np.nan,
        ),
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
            "kickoff": ["2026-09-10T17:00:00+00:00", "2026-09-10T20:00:00+00:00"],
            "away_team": ["YR1", "OTHER2"],
            "home_team": ["KEEP", "OTHER1"],
            "spread_line": [-3.5, 2.5],
            # KEEP (home, kept coach) is NOT picked -- YR1 (away, year-1) is.
            "home_cover_probability": [0.35, 0.55],
            "bet_side": ["AWAY", "HOME"],
            "edge": [0.15, 0.05],
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


def _write_arrest_snapshot(
    data_root: Path,
    *,
    snapshot_id: str,
    fetched_at_utc: str,
    incidents: pd.DataFrame,
) -> Path:
    directory = data_root / "raw" / "player_arrests" / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    safe = directory / "incidents_point_in_time.parquet"
    incidents.to_parquet(safe, index=False)
    manifest = {
        "snapshot_id": snapshot_id,
        "fetched_at_utc": fetched_at_utc,
        "complete": True,
        "rows_cached": len(incidents),
        "point_in_time_policy": {"safe_index": safe.name},
        "files": {safe.name: sha256_file(safe)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def _publish_with_fresh_empty_arrest(
    artifacts_root: Path,
    *,
    destination: Path,
    readme_path: Path,
    data_root: Path | None = None,
    published_at: datetime | None = None,
    registry_root: Path | None = None,
) -> dict[str, object]:
    """Exercise production publishing with an explicit no-incident snapshot."""

    instant = published_at or datetime.now(UTC)
    resolved_data_root = data_root or artifacts_root / "test-data"
    snapshot_id = instant.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    _write_arrest_snapshot(
        resolved_data_root,
        snapshot_id=snapshot_id,
        fetched_at_utc=instant.astimezone(UTC).isoformat(),
        incidents=pd.DataFrame(columns=["record_id", "incident_date", "team"]),
    )
    if not any((resolved_data_root / "raw").glob("*/schedules.parquet")):
        forecast = artifacts_root / "margin_predictions" / "forecast" / "recommendations.csv"
        card = pd.read_csv(forecast)
        current = card[
            ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"]
        ].copy()
        current["gameday"] = pd.to_datetime(current["gameday"]).dt.date
        current["home_coach"] = current["home_team"].astype(str) + " Coach"
        current["away_coach"] = current["away_team"].astype(str) + " Coach"
        current["result"] = np.nan
        prior = current.copy()
        prior["season"] = prior["season"].astype(int) - 1
        prior["game_id"] = "prior_" + prior["game_id"].astype(str)
        prior["gameday"] = (pd.to_datetime(prior["gameday"]) - pd.DateOffset(years=1)).dt.date
        prior["result"] = 1.0
        write_snapshot(
            pd.concat([prior, current], ignore_index=True),
            pd.DataFrame({"game_id": [], "team": []}),
            seasons=sorted(set(prior["season"]) | set(current["season"])),
            raw_root=resolved_data_root / "raw",
        )
    return publish_active_predictions(
        artifacts_root,
        destination=destination,
        readme_path=readme_path,
        data_root=resolved_data_root,
        published_at=instant,
        registry_root=registry_root,
    )


def test_published_card_applies_and_discloses_the_coach_fade_overlay(tmp_path: Path) -> None:
    """The overlay (docs/coach_fade_overlay.md) flips the clean-case pick and
    discloses it in the card's provenance, the same plain way Best Pick ties
    are disclosed."""

    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
    )

    assert result["overlay_enabled"] is True
    assert result["overlay_flip_count"] == 1
    assert result["overlay_flipped_game_ids"] == ["2026_01_KEEP_YR1"]

    card = destination.read_text(encoding="utf-8")
    assert "**Production policy active:**" in card
    assert "KEEP +3.5" in card
    assert "YR1 -3.5" not in card
    readme_text = readme.read_text(encoding="utf-8")
    assert "**Production policy active:**" in readme_text


def test_publication_helper_builds_required_production_sources(tmp_path: Path) -> None:

    _, readme, _data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )

    assert result["decision_policy_id"] == (
        "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
    )
    assert result["overlay_enabled"] is True
    assert result["overlay_flip_count"] == 0
    card = destination.read_text(encoding="utf-8")
    assert "Production policy active" in card
    assert "YR1 -3.5" in card


def test_production_composes_coach_then_arrest_and_requires_fresh_source(
    tmp_path: Path,
) -> None:
    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    _write_arrest_snapshot(
        data_root,
        snapshot_id="20260908T150000Z",
        fetched_at_utc="2026-09-08T15:00:00+00:00",
        incidents=pd.DataFrame(
            {
                "record_id": [1],
                "incident_date": ["2026-09-01"],
                "team": ["YR1"],
            }
        ),
    )

    result = publish_active_predictions(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
        published_at=instant,
    )

    assert result["overlay_flipped_game_ids"] == ["2026_01_KEEP_YR1"]
    assert result["production_overlay_overlap_game_ids"] == []
    card = destination.read_text(encoding="utf-8")
    assert "KEEP +3.5" in card
    assert "YR1 -3.5" not in card


def test_production_refuses_a_schedule_manifest_hash_mismatch(tmp_path: Path) -> None:
    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    snapshot = latest_snapshot(data_root / "raw")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["schedules.parquet"]["sha256"] = "0" * 64
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_arrest_snapshot(
        data_root,
        snapshot_id="20260908T150000Z",
        fetched_at_utc="2026-09-08T15:00:00+00:00",
        incidents=pd.DataFrame({"record_id": [], "incident_date": [], "team": []}),
    )

    with pytest.raises(ValueError, match="payload hash mismatch"):
        publish_active_predictions(
            tmp_path,
            destination=destination,
            readme_path=readme,
            data_root=data_root,
            published_at=datetime(2026, 9, 8, 16, 0, tzinfo=UTC),
        )
    assert not destination.exists()


def test_production_stale_arrest_source_writes_neither_publication_file(
    tmp_path: Path,
) -> None:
    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    original_readme = readme.read_text(encoding="utf-8")
    _write_arrest_snapshot(
        data_root,
        snapshot_id="20260906T000000Z",
        fetched_at_utc="2026-09-06T00:00:00+00:00",
        incidents=pd.DataFrame(columns=["record_id", "incident_date", "team"]),
    )

    with pytest.raises(DataContractError, match="stale"):
        publish_active_predictions(
            tmp_path,
            destination=destination,
            readme_path=readme,
            data_root=data_root,
            published_at=datetime(2026, 9, 8, 16, 0, tzinfo=UTC),
        )

    assert not destination.exists()
    assert readme.read_text(encoding="utf-8") == original_readme


def test_paper_ledger_records_final_side_and_frozen_arrest_provenance(
    tmp_path: Path,
) -> None:
    _, _readme, data_root = _write_overlay_publication_fixture(tmp_path)
    instant = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    snapshot = _write_arrest_snapshot(
        data_root,
        snapshot_id="20260908T150000Z",
        fetched_at_utc="2026-09-08T15:00:00+00:00",
        incidents=pd.DataFrame(
            {
                "record_id": [1],
                "incident_date": ["2026-09-01"],
                "team": ["YR1"],
            }
        ),
    )

    result = record_paper_decisions(
        tmp_path,
        data_root=data_root,
        now=instant,
    )
    ledger = load_paper_decisions(tmp_path).set_index("game_id")
    row = ledger.loc["2026_01_KEEP_YR1"]

    assert result["recorded"] == 2
    assert row["model_pick_side"] == "AWAY"
    assert row["pre_arrest_pick_side"] == "HOME"
    assert row["former_policy_pick_side"] == "AWAY"
    assert row["pick_side"] == "HOME"
    assert bool(row["coach_fade_flip"])
    assert not bool(row["player_arrests_flip"])
    assert bool(row["composed_overlay_flip"])
    assert not bool(row["player_arrests_home_flag"])
    assert bool(row["player_arrests_away_flag"])
    assert row["player_arrests_snapshot_id"] == "20260908T150000Z"
    assert row["player_arrests_safe_index_sha256"] == sha256_file(
        snapshot / "incidents_point_in_time.parquet"
    )
    assert row["bet_side"] == "PASS"


def test_publish_rejects_weekly_model_id_mismatch(tmp_path: Path) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    metadata["active_model_id"] = "wrong-model"
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="model ID does not match"):
        _publish_with_fresh_empty_arrest(
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

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
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

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
    )

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
            (
                "2025_01_YR1_OPP",
                2025,
                "REG",
                1,
                "2025-09-07",
                "YR1",
                "OPP",
                "Old1",
                "OppC",
                -3.0,
            ),
            (
                game_ids[0],
                2026,
                "REG",
                1,
                "2026-09-10",
                "KEEP",
                "YR1",
                "Steady",
                "New1",
                np.nan,
            ),
        ],
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_coach",
            "away_coach",
            "result",
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
    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
    )

    # v2 nominated its own game, unmoved by the overlay flipping a different one.
    assert result["best_pick_nomination_v2_game_id"] == game_ids[2]
    assert result["best_pick_game_id"] == game_ids[2]
    # The overlay flipped the year-1-coach game, unmoved by v2 existing.
    assert result["overlay_flip_count"] == 1
    assert result["overlay_flipped_game_ids"] == [game_ids[0]]

    card = destination.read_text(encoding="utf-8")
    assert "Production policy active" in card
    assert NOMINATION_V2_METHOD_SENTENCE in card


# ---------------------------------------------------------------------------
# POL-12 (2026-09-05 owner mandate): "our project over/under total needs to
# line up with our spread prediction" -- tiebreaker.json persistence and the
# CURRENT_PREDICTIONS.md card line, both read straight off the SAME
# TiebreakerReport publish_active_predictions computes once.
# ---------------------------------------------------------------------------


def _fixed_tiebreaker_report() -> object:
    from nfl_ats.tiebreaker import MarketConsensus, TiebreakerReport

    consensus = MarketConsensus(
        game_id="later", home_expected_margin=3.0, total_line=43.0, source="test"
    )
    return TiebreakerReport(
        game_id="later",
        home="LAC",
        away="ARI",
        consensus=consensus,
        model_view=None,
        totals_view=None,
        guess_margin=3.19,
        guess_total_line=43.0421,
        served_total_method="blend_k01",
        comparison_total_blend_k01=43.0421,
        implied_home=23.02,
        implied_away=20.02,
        neighborhood_games=221,
        neighborhood_window="test",
        median_total=43.0,
        median_home_margin=3.0,
        guess_home=23,
        guess_away=19,
        common_scores=((24, 20, 12.0),),
        pick_side="HOME",
        pick_spread_line=3.0,
        pick_cover_probability=0.42,
        pick_push_probability=0.15,
        consistency_note="consistent with the LAC -3 pick",
        total_mae=10.5,
        total_median_ae=9.0,
        total_bias=0.5,
        implied_score_mae=7.4,
    )


def test_publish_active_predictions_writes_tiebreaker_json_and_card_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forecast, readme = _write_active_publication_fixture(tmp_path)
    monkeypatch.setattr(
        publishing_module, "published_tiebreaker_guess", lambda *a, **k: _fixed_tiebreaker_report()
    )
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert result["tiebreaker_json_path"] is not None
    assert result["tiebreaker_skip_reason"] is None
    forecast_copy = forecast / "tiebreaker.json"
    card_copy = destination.parent / "tiebreaker.json"
    assert forecast_copy.is_file()
    assert card_copy.is_file()
    forecast_block = json.loads(forecast_copy.read_text(encoding="utf-8"))
    card_block = json.loads(card_copy.read_text(encoding="utf-8"))
    assert forecast_block == card_block
    assert forecast_block["home"] == "LAC"
    assert forecast_block["away"] == "ARI"
    assert forecast_block["guess_home"] == 23
    assert forecast_block["guess_away"] == 19
    # implied_margin is guess_home - guess_away (the SCORE's own margin),
    # never the pre-lattice guess_margin (3.19) -- the whole point of the
    # fix is that the two can never disagree once persisted.
    assert forecast_block["implied_margin"] == 4
    assert forecast_block["market_total"] == pytest.approx(43.0)
    assert forecast_block["blended_total"] == pytest.approx(43.0421)
    # MOD-17 (docs/tiebreaker.md "one lattice, one margin, one total";
    # nfl_ats.served_total): the JSON also names WHICH method served and
    # always carries the comparison arm's own number.
    assert forecast_block["served_total"] == pytest.approx(43.0421)
    assert forecast_block["served_total_method"] == "blend_k01"
    assert forecast_block["comparison_total_blend_k01"] == pytest.approx(43.0421)

    card = destination.read_text(encoding="utf-8")
    assert "**Tiebreaker (last game, ARI at LAC):** LAC 23 - ARI 19, total 42" in card
    assert "market total 43" in card
    assert "consistent with the LAC -3 pick" in card


def test_publish_active_predictions_refuses_tiebreaker_artifacts_on_consistency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``TiebreakerConsistencyError`` refuses ``tiebreaker.json``/the card
    line ONLY -- the pool's card must still publish regardless."""

    from nfl_ats.tiebreaker import TiebreakerConsistencyError

    forecast, readme = _write_active_publication_fixture(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise TiebreakerConsistencyError("no lattice cell on the pick's side")

    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    monkeypatch.setattr(
        publishing_module, "published_tiebreaker_guess", lambda *a, **k: _fixed_tiebreaker_report()
    )
    _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert (forecast / "tiebreaker.json").is_file()
    assert (destination.parent / "tiebreaker.json").is_file()
    monkeypatch.setattr(publishing_module, "published_tiebreaker_guess", _raise)

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert result["tiebreaker_json_path"] is None
    assert result["tiebreaker_skip_reason"] is not None
    assert "consistency check refused" in str(result["tiebreaker_skip_reason"])
    assert not (forecast / "tiebreaker.json").is_file()
    assert not (destination.parent / "tiebreaker.json").is_file()
    from nfl_ats.board_content import _load_tiebreaker_view

    assert not _load_tiebreaker_view(
        forecast, json.loads((forecast / "metadata.json").read_text())
    ).recorded
    card = destination.read_text(encoding="utf-8")
    assert "Tiebreaker (last game" not in card
    # The card itself still published -- a refused tiebreaker never blocks it.
    assert result["games"] == 2


def test_publish_active_predictions_degrades_gracefully_without_tiebreaker_data(
    tmp_path: Path,
) -> None:
    """The real (unmocked) path: the fixture's local schedule snapshot has
    no spread_line/total_line/score columns at all, so the real
    ``tiebreaker_report`` degrades to "unavailable" the same fail-open way
    every other optional artifact on this publish path already does --
    never a crash, never a half-written tiebreaker.json."""

    forecast, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert result["tiebreaker_json_path"] is None
    assert result["tiebreaker_skip_reason"]
    assert not (forecast / "tiebreaker.json").is_file()
    card = destination.read_text(encoding="utf-8")
    assert "Tiebreaker (last game" not in card


def test_published_tiebreaker_passes_final_card_side_and_exact_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "game_id": "2026_01_DEN_KC",
        "home_team": "KC",
        "away_team": "DEN",
        "season": 2026,
        "week": 1,
        "game_type": "REG",
        "gameday": "2026-09-14",
        "spread_line": 3.0,
        "predicted_margin": 3.19,
        "predicted_market_residual": 0.19,
        "home_cover_probability": 0.49,
    }
    # This is the resolved overlay-flipped card: raw residual HOME, final probability AWAY.
    frame = pd.DataFrame([row])
    raw = tmp_path / "raw" / "fixture"
    raw.mkdir(parents=True)
    frame.to_parquet(raw / "schedules.parquet")
    captured = {}

    def fake_report(*args, **kwargs):
        captured.update(kwargs)
        return _fixed_tiebreaker_report()

    monkeypatch.setattr(publishing_module, "tiebreaker_report", fake_report)
    publishing_module.published_tiebreaker_guess(
        tmp_path,
        artifacts_root=tmp_path / "artifacts",
        active={"model_id": "active"},
        metadata={"season": 2026, "week": 1, "active_model_id": "active"},
        predictions=frame,
    )
    assert captured["published_pick_side"] == "AWAY"
    assert captured["frozen_spread"] == 3.0
    pd.testing.assert_series_equal(captured["forecast_row"], frame.iloc[0])
    assert captured["model_id"] == captured["forecast_model_id"] == "active"

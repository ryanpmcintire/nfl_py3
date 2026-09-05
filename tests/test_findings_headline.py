"""The findings baseline follows the same validated artifact as the board."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nfl_ats.active_model import matching_opener_evaluation
from nfl_ats.board_site_content import _load_findings_content
from nfl_ats.dashboard import findings_content
from nfl_ats.public_board import load_baseline_measurement


def test_findings_hero_follows_board_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _board_content_fixtures import build_fixture_content

    monkeypatch.setattr(findings_content, "FINDINGS", ())
    monkeypatch.setattr(findings_content, "LEAD_BLURBS", ())
    board = build_fixture_content()
    for value, caption in (
        ("57.0%", "95% range [51.00%, 62.00%]."),
        ("61.0%", "95% range [55.00%, 66.00%]."),
    ):
        changed = replace(
            board,
            headline=replace(board.headline, raw_model_value_text=value, raw_model_caption=caption),
        )
        content = _load_findings_content(
            [], registry_root=tmp_path, generated_at=datetime(2026, 9, 5, tzinfo=UTC), board=changed
        )
        assert content.hero_tiles[0].value == value
        assert content.hero_tiles[0].context == caption


def test_handoff_uses_validated_baseline(tmp_path: Path) -> None:
    import json

    from test_board_content import _headline_artifacts

    active, path = _headline_artifacts(tmp_path)
    baseline = load_baseline_measurement(tmp_path, active)
    matched = matching_opener_evaluation(tmp_path, active)
    assert matched is not None
    assert matched[0] == baseline.directory
    assert matched[1]["metrics"]["opener_accuracy_probability_rule"] == baseline.accuracy
    metadata = json.loads(path.read_text())
    metadata["active_model_config"]["probability_method"] = "ecdf"
    path.write_text(json.dumps(metadata))
    assert matching_opener_evaluation(tmp_path, active) is None
    with pytest.raises(ValueError, match="No opener-evaluation"):
        load_baseline_measurement(tmp_path, active)


def test_no_static_active_measurement_in_findings() -> None:
    assert not hasattr(findings_content, "HEADLINE")
    assert findings_content.HERO_TILES[0].value == "--"
    assert "53.35994677312043" not in Path(findings_content.__file__).read_text()


def test_missing_provenance_note_never_exposes_model_identity(tmp_path: Path) -> None:
    from test_board_content import _headline_artifacts

    from nfl_ats.board_site_content import _number_provenance_rows

    active, _ = _headline_artifacts(tmp_path)
    # The fixture has no forecast, so verification cannot finish.
    rows, note = _number_provenance_rows(tmp_path, active)
    assert rows == ()
    assert note == "The current model's archive scores have not all been verified yet."
    assert active["model_id"] not in note

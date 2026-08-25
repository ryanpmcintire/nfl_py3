"""Tests for the UI-08 model-explanation view (``nfl_ats.model_explanation``).

The section reads the latest ``market-decomposition`` artifact's
``classification.csv`` and renders family weights with stability labels,
per-bucket caveat captions, honesty notes, and provenance/trust lines --
fail-open in both directions (quiet empty state when nothing was ever run;
visible warning box when a run exists but cannot be parsed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_public_board import assert_public_safe

from nfl_ats.model_explanation import (
    STABILITY_JUMPY_RATIO,
    load_model_explanation,
    load_model_explanation_html,
    market_decomposition_run_count,
)
from nfl_ats.public_board import render_models_page

_STEADY_RATIO = STABILITY_JUMPY_RATIO / 2
_JUMPY_RATIO = STABILITY_JUMPY_RATIO * 2


def _classification_csv(families: list[dict[str, Any]]) -> str:
    header = [
        "family",
        "weight_in_margin",
        "margin_share",
        "net_signed_in_margin",
        "refit_std_in_margin",
        "season_std_in_margin",
        "weight_in_spread",
        "spread_share",
        "net_signed_in_spread",
        "refit_std_in_spread",
        "season_std_in_spread",
        "weight_in_residual",
        "residual_share",
        "net_signed_in_residual",
        "refit_std_in_residual",
        "season_std_in_residual",
        "classification",
    ]
    lines = [",".join(header)]
    for family in families:
        name = family["family"]
        row = [name] + ["0.0"] * 15 + [family["classification"]]
        # weight_in_spread (index 6), refit_std_in_spread (index 9).
        row[6] = f"{family['weight_in_spread']:.6f}"
        row[9] = f"{family['refit_std_in_spread']:.6f}"
        row[2] = f"{family['margin_share']:.6f}"
        row[7] = f"{family['spread_share']:.6f}"
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _write_decomposition_run(
    tmp_path: Path,
    run_id: str,
    *,
    feature_sha256: str | None = "f" * 64,
    families: list[dict[str, Any]] | None = None,
    metadata_extra: dict[str, Any] | None = None,
    raw_classification: str | None = None,
) -> Path:
    directory = tmp_path / "market_decomposition" / run_id
    directory.mkdir(parents=True)
    if raw_classification is not None:
        (directory / "classification.csv").write_text(raw_classification, encoding="utf-8")
    else:
        payload = families or []
        (directory / "classification.csv").write_text(
            _classification_csv(payload), encoding="utf-8"
        )
    metadata: dict[str, Any] = {
        "created_at_utc": "2026-08-25T12:00:00+00:00",
        "feature_profile": "weak_stack",
        "ridge_alpha": 10.0,
        "start_season": 2013,
        "end_season": 2025,
        "refit_weeks": 142,
        **(metadata_extra or {}),
    }
    if feature_sha256 is not None:
        metadata["provenance"] = {"feature_table": {"sha256": feature_sha256}}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


_DEFAULT_FAMILIES = [
    {
        "family": "defense",
        "margin_share": 0.20,
        "spread_share": 0.19,
        "weight_in_spread": 4.0,
        "refit_std_in_spread": 4.0 * _STEADY_RATIO,
        "classification": "priced",
    },
    {
        "family": "player_values",
        "margin_share": 0.06,
        "spread_share": 0.01,
        "weight_in_spread": 1.0,
        "refit_std_in_spread": 1.0 * _JUMPY_RATIO,
        "classification": "unpriced_predictive",
    },
]


def test_missing_run_yields_honest_empty_state(tmp_path: Path) -> None:
    """No run ever saved: the section explains itself instead of vanishing."""

    html = load_model_explanation_html(tmp_path)
    assert "No model-explanation run saved yet" in html
    assert "nfl-ats market-decomposition" in html
    assert load_model_explanation(tmp_path) is None
    assert market_decomposition_run_count(tmp_path) == 0


def test_family_table_renders_stability_labels_and_caveat_captions(
    tmp_path: Path,
) -> None:
    _write_decomposition_run(tmp_path, "20260825T000000Z", families=_DEFAULT_FAMILIES)
    html = load_model_explanation_html(tmp_path)

    # Human family phrase from FAMILY_PHRASES, not the raw registry name.
    assert "recent defensive performance" in html
    # Shares rendered as percentages.
    assert "20.0%" in html
    assert "1.0%" in html
    # Caveat caption for the bucket that most resembles a lead says
    # "unconfirmed" out loud -- never "edge".
    assert "unconfirmed" in html
    assert "found an edge" in html  # inside the honesty note, negated
    # Stability labels split exactly at the declared ratio.
    assert "steady across refits" in html
    assert "jumps around between refits" in html
    # Exact stability numbers stay in the small print, off the headline word.
    assert "standard deviation" in html


def test_staleness_warning_tracks_the_active_manifest(
    tmp_path: Path,
) -> None:
    _write_decomposition_run(
        tmp_path, "20260825T000000Z", families=_DEFAULT_FAMILIES, feature_sha256="a" * 64
    )
    manifest = tmp_path / "active_ats_model.json"
    manifest.write_text(json.dumps({"feature_table_sha256": "b" * 64}), encoding="utf-8")

    stale = load_model_explanation_html(tmp_path)
    assert "earlier build of the model&#8217;s inputs" in stale

    manifest.write_text(json.dumps({"feature_table_sha256": "a" * 64}), encoding="utf-8")
    current = load_model_explanation_html(tmp_path)
    assert "earlier build of the model&#8217;s inputs" not in current

    # No manifest at all: no claim either way, numbers still render.
    manifest.unlink()
    unclaimed = load_model_explanation_html(tmp_path)
    assert "earlier build of the model&#8217;s inputs" not in unclaimed
    assert "steady across refits" in unclaimed


def test_unreadable_runs_show_a_visible_warning_not_a_crash(tmp_path: Path) -> None:
    _write_decomposition_run(
        tmp_path,
        "20260825T000000Z",
        raw_classification="this is not,a valid\nbroken csv\x00\x01",
    )
    html = load_model_explanation_html(tmp_path)
    assert "MODEL EXPLANATION UNAVAILABLE" in html
    assert "none readable" in html


def test_torn_newest_run_falls_back_to_the_previous_complete_run(
    tmp_path: Path,
) -> None:
    _write_decomposition_run(
        tmp_path,
        "20260824T000000Z",
        families=_DEFAULT_FAMILIES,
        metadata_extra={"refit_weeks": 141},
    )
    _write_decomposition_run(tmp_path, "20260825T000000Z", raw_classification=",bad header only\n")
    explanation = load_model_explanation(tmp_path)
    assert explanation is not None
    assert explanation.run_directory == "20260824T000000Z"


def test_provenance_line_carries_window_method_and_timestamp(
    tmp_path: Path,
) -> None:
    _write_decomposition_run(tmp_path, "20260825T000000Z", families=_DEFAULT_FAMILIES)
    html = load_model_explanation_html(tmp_path)
    assert "fit on 2013&ndash;2025 completed games" in html
    assert "142 weekly refits" in html
    assert "ridge alpha 10" in html
    assert "feature profile weak_stack" in html
    assert "artifact written 2026-08-25 12:00 UTC" in html


def test_metadata_free_run_still_renders_weights(tmp_path: Path) -> None:
    directory = tmp_path / "market_decomposition" / "20260825T000000Z"
    directory.mkdir(parents=True)
    (directory / "classification.csv").write_text(
        _classification_csv(_DEFAULT_FAMILIES), encoding="utf-8"
    )
    html = load_model_explanation_html(tmp_path)
    assert "steady across refits" in html
    assert "Provenance:" not in html  # nothing invented when metadata is absent


def test_section_wires_into_the_models_page(tmp_path: Path) -> None:
    _write_decomposition_run(tmp_path, "20260825T000000Z", families=_DEFAULT_FAMILIES)
    section = load_model_explanation_html(tmp_path)
    page = render_models_page(None, explanation_section=section)
    assert "WHAT THE MODEL DECIDES" not in page  # kicker text is exact below
    assert "HOW THE MODEL DECIDES" in page
    assert "Read this table honestly" in page
    # Page-level guardrails still hold with the new section aboard.
    assert_public_safe(page)


def test_models_page_without_section_keeps_prior_shape() -> None:
    page = render_models_page(None)
    assert "Ledger unavailable right now" in page
    assert "HOW THE MODEL DECIDES" not in page
    assert_public_safe(page)


@pytest.mark.parametrize(
    ("std", "expected"),
    [
        (_STEADY_RATIO, "steady across refits"),
        (_JUMPY_RATIO, "jumps around between refits"),
    ],
)
def test_stability_threshold_is_a_declared_constant(
    tmp_path: Path, std: float, expected: str
) -> None:
    _write_decomposition_run(
        tmp_path,
        "20260825T000000Z",
        families=[
            {
                "family": "elo",
                "margin_share": 0.3,
                "spread_share": 0.3,
                "weight_in_spread": 1.0,
                "refit_std_in_spread": std,
                "classification": "priced",
            }
        ],
    )
    html = load_model_explanation_html(tmp_path)
    assert expected in html

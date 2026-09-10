from __future__ import annotations

import io
import tokenize
from pathlib import Path

from nfl_ats import model_ledger, public_board
from nfl_ats.dashboard import findings_content
from nfl_ats.dashboard import viz as dashboard_viz

CANONICAL_FIGURE_TOKENS: tuple[str, ...] = (
    "53.4",
    "52.1",
    "53.76",
    "53.36",
    "54.2",
    "55.42",
    "1.2641",
    "0.8571",
    "0.8562",
    "0.493",
    "55-56",
    "57-58",
    "54-55",
    "51.1",
    "52.14",
    "52.24",
    "51.7",
)

_REGION_END_MARKER = 'PINNED_NUMBER_REGION_END = "End of the pinned-number region"'

_PROSE_MODULES = (
    public_board,
    model_ledger,
    dashboard_viz,
)


def _module_source(module: object) -> str:
    return Path(str(module.__file__)).read_text(encoding="utf-8")  # type: ignore[attr-defined]


def _blank_comment_tokens(source: str) -> str:

    spans: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append((token.start, token.end))
    lines = source.splitlines()
    for (start_row, start_col), (_end_row, end_col) in spans:
        row = lines[start_row - 1]
        end = min(end_col, len(row))
        lines[start_row - 1] = row[:start_col] + " " * (end - start_col) + row[end:]
    return "\n".join(lines)


def _prose_code() -> dict[str, str]:

    raw = _module_source(findings_content)
    assert _REGION_END_MARKER in raw, (
        "findings_content lost its 'End of the pinned-number region' marker; "
        "the literal-ban guard has no boundary to enforce."
    )
    _, _, findings_tail = raw.partition(_REGION_END_MARKER)
    scanned: dict[str, str] = {
        "findings_content (below the region marker)": findings_tail,
    }
    scanned.update(
        {module.__name__: _module_source(module) for module in _PROSE_MODULES}  # type: ignore[attr-defined]
    )
    return {name: _blank_comment_tokens(text) for name, text in scanned.items()}


def test_canonical_figures_appear_only_inside_the_pinned_number_region() -> None:

    for token in CANONICAL_FIGURE_TOKENS:
        for name, code in _prose_code().items():
            assert token not in code, (
                f"canonical figure {token!r} is typed into {name}. Compose it "
                "from the named constant in "
                "nfl_ats.dashboard.findings_content instead -- every accuracy "
                "figure on the site comes from ONE variable."
            )


def test_the_pinned_region_actually_pins_the_constants() -> None:

    raw = _module_source(findings_content)
    head, _, _tail = raw.partition(_REGION_END_MARKER)
    head_code = _blank_comment_tokens(head)
    for token in (
        "52.14",
        "52.24",
        "51.1",
        "51.7",
        "55-56",
        "57-58",
        "1.2641",
        "0.85715",
        "0.4930",
    ):
        assert token in head_code, f"expected pinned constant {token!r} in the region"


def test_ceiling_band_constants_are_frozen() -> None:

    assert findings_content.PRACTICAL_CEILING_LOW_PCT == 54.0
    assert findings_content.PRACTICAL_CEILING_HIGH_PCT == 55.0
    assert findings_content.BETTORS_VS_CLOSE_BAND == "55-56"
    assert findings_content.MEASURED_CEILING_PCT == 56
    assert findings_content.PREMEASUREMENT_GUESS_BAND == "57-58"
    assert findings_content.ORACLE_FROZEN_LINE_PCT == 57
    assert findings_content.CEILING_BUG_MARK_PCT == 60


def test_headline_ceiling_is_derived_from_the_practical_band() -> None:

    assert (
        str(int(findings_content.PRACTICAL_CEILING_LOW_PCT)) in findings_content.HERO_TILES[1].value
    )
    assert (
        str(int(findings_content.PRACTICAL_CEILING_HIGH_PCT))
        in findings_content.HERO_TILES[1].value
    )


def test_player_study_constants_are_frozen() -> None:

    assert findings_content.MARKET_TEAM_FORM_MODEL_PCT == 51.1
    assert findings_content.FULL_PLAYER_LAYER_PCT == 52.1
    assert findings_content.INJURY_ONLY_MODEL_PCT == 51.3
    assert findings_content.LEARNED_AVAILABILITY_BEFORE_PCT == 52.14
    assert findings_content.LEARNED_AVAILABILITY_AFTER_PCT == 52.24
    assert findings_content.PARTICIPATION_RAPM_MODEL_PCT == 51.7


def test_ladder_and_cards_compose_the_bands_not_retype_them() -> None:

    rungs = findings_content.ladder_rungs(None)
    ceiling_rung = rungs[-2]
    assert "roughly 55-56% against the close" in ceiling_rung
    assert "about 56% (total-leak control" in ceiling_rung
    assert "the older 57-58% band was the pre-measurement guess" in ceiling_rung

"""The enforceable "everything is tied to a variable" law (owner directive).

Every accuracy figure rendered on the site must come from ONE named constant;
no hand-typed number literals in prose; and each canonical stat renders as a
figure ONLY on its home page. This module enforces the source half of that
law: the canonical figures -- headline grades, chain/overlay evidence, and
the per-card study numbers -- may appear as SOURCE LITERALS only inside
``nfl_ats.dashboard.findings_content``'s pinned-number region (the constants
and ``HEADLINE`` definition block ending at its "End of the pinned-number
region" marker). Anywhere else in the dashboard modules they must be
composed from those constants, so hand-typing a figure into prose fails CI.

Mechanics: each file is tokenized with :mod:`tokenize` and COMMENT tokens are
blanked out before scanning, so provenance comments never trip the scan
while string literals (which DO reach rendered pages) always do. The
rendered-page half of the law lives in ``tests/test_public_board.py``
(canonical figures render only on their home page's default view).

Canonical stats and their homes: opener baseline 53.4 / close 52.1 / arrest
evaluation 53.76 vs 53.36 / season range -> track_record.html; ≈55%
expectation + 54.2% chain history + collapsed ladder -> index.html; per-card
effect sizes/intervals -> findings.html; per-row track records ->
models.html.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

from nfl_ats import model_ledger, public_board
from nfl_ats.dashboard import findings_content
from nfl_ats.dashboard import viz as dashboard_viz

#: Every canonical accuracy figure, in one list. Substring matching is
#: deliberate: "52.14" must be caught inside "52.145", and the band strings
#: ("55-56") must match however they are embedded in prose.
CANONICAL_FIGURE_TOKENS: tuple[str, ...] = (
    # Headline grades + arrest evaluation (home: track_record.html).
    "53.4",
    "52.1",
    "53.76",
    "53.36",
    # Played-chain history (home: index.html).
    "54.2",
    # Overlay-union evidence constants (pinned in findings_content).
    "55.42",
    "1.2641",
    "0.8571",
    "0.8562",
    "0.493",
    # Ceiling bands (pinned in findings_content).
    "55-56",
    "57-58",
    "54-55",
    # Per-card study numbers that collide with canonical grades (pinned in
    # findings_content by study name).
    "51.1",
    "52.14",
    "52.24",
    "51.7",
)

#: The exact comment text that ends findings_content's allowed region.
_REGION_END_MARKER = "End of the pinned-number region"

_PROSE_MODULES = (
    public_board,
    model_ledger,
    dashboard_viz,
)


def _module_source(module: object) -> str:
    return Path(str(module.__file__)).read_text(encoding="utf-8")  # type: ignore[attr-defined]


def _blank_comment_tokens(source: str) -> str:
    """The source with every ``#...`` comment replaced by spaces.

    Comments are documentation, not rendered copy, so they cannot smuggle a
    number onto the site -- but string tokens survive intact, which is what
    matters: a band typed into a prose string is exactly the failure mode
    this guard exists to catch (and it also keeps CSS hex colors inside
    strings from being mangled by a naive ``#`` split).
    """

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
    """Comment-stripped code for everything OUTSIDE the allowed region."""

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
    """THE LAW: no canonical figure as a source literal outside the region.

    A hit anywhere in ``public_board``/``model_ledger``/``viz``, or below
    findings_content's region marker, means a number was hand-typed into
    prose instead of composed from the named constants.
    """

    for token in CANONICAL_FIGURE_TOKENS:
        for name, code in _prose_code().items():
            assert token not in code, (
                f"canonical figure {token!r} is typed into {name}. Compose it "
                "from the named constant in "
                "nfl_ats.dashboard.findings_content instead -- every accuracy "
                "figure on the site comes from ONE variable."
            )


def test_the_pinned_region_actually_pins_the_constants() -> None:
    """Positive control for the scanner: the region really does contain the
    constants this law protects, so the ban above can never pass vacuously.

    The learned-availability pair (52.14% -> 52.24%), the full-player layer
    grade (52.1), and the bettors-vs-close band ("55-56") must exist as code
    literals inside the region.
    """

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
    """All measured (from doc): docs/pool_edge_plan.md ceiling section and
    docs/leak_ceiling_control.md's total-leak positive control."""

    assert findings_content.PRACTICAL_CEILING_LOW_PCT == 54.0
    assert findings_content.PRACTICAL_CEILING_HIGH_PCT == 55.0
    assert findings_content.BETTORS_VS_CLOSE_BAND == "55-56"
    assert findings_content.MEASURED_CEILING_PCT == 56
    assert findings_content.PREMEASUREMENT_GUESS_BAND == "57-58"
    assert findings_content.ORACLE_FROZEN_LINE_PCT == 57
    assert findings_content.CEILING_BUG_MARK_PCT == 60


def test_headline_ceiling_is_derived_from_the_practical_band() -> None:
    """HEADLINE's ceiling interval IS the practical band -- one variable, so
    the findings hero tile and every verbal repeat can never drift apart."""

    assert findings_content.HEADLINE.ceiling_low is findings_content.PRACTICAL_CEILING_LOW_PCT
    assert findings_content.HEADLINE.ceiling_high is findings_content.PRACTICAL_CEILING_HIGH_PCT


def test_player_study_constants_are_frozen() -> None:
    """All measured (from doc): docs/modeling.md, ROADMAP.md PER-05 ablation,
    docs/data_feasibility.md participation-rating screen."""

    assert findings_content.MARKET_TEAM_FORM_MODEL_PCT == 51.1
    assert findings_content.FULL_PLAYER_LAYER_PCT == 52.1
    assert findings_content.INJURY_ONLY_MODEL_PCT == 51.3
    assert findings_content.LEARNED_AVAILABILITY_BEFORE_PCT == 52.14
    assert findings_content.LEARNED_AVAILABILITY_AFTER_PCT == 52.24
    assert findings_content.PARTICIPATION_RAPM_MODEL_PCT == 51.7


def test_ladder_and_cards_compose_the_bands_not_retype_them() -> None:
    """The composed sentences still carry the doc-measured bands end to end:
    if a constant moves, these renderings move with it or fail here."""

    rungs = findings_content.ladder_rungs(None)
    ceiling_rung = rungs[-2]
    assert "roughly 55-56% against the close" in ceiling_rung
    assert "about 56% (total-leak control" in ceiling_rung
    assert "the older 57-58% band was the pre-measurement guess" in ceiling_rung

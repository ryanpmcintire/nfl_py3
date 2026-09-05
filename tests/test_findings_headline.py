"""The findings page must never hardcode the active model's grades again.

Promoting ``weak_stack`` over ``player`` on 2026-08-18 changed every headline
number, and the page kept quoting the old model for several commits because the
figures were typed into the prose. These tests make that failure mode loud.
"""

from __future__ import annotations

from dataclasses import replace

from nfl_ats.dashboard import findings_content
from nfl_ats.dashboard.findings_content import HEADLINE, HERO_PARAGRAPHS, HERO_TILES

#: Places where the active model's grade may legitimately appear as a literal
#: because the sentence is about a DIFFERENT measurement that happens to share
#: the value. Each entry must name which measurement it is. (The former
#: player-layer entries -- "52.1% with the full player", "51.7% against
#: 52.1%" -- are gone: those study figures are now named constants in
#: findings_content, per tests/test_number_variables.py.)
LITERAL_EXEMPTIONS = (
    "52.5-52.8%",  # the mined-era range of many variants, not the active grade
    "51.6% of 8,933 games",  # the CFB benchmark model, not our NFL close grade
)


def test_hero_tiles_render_the_headline_object() -> None:
    values = [tile.value for tile in HERO_TILES]
    assert HEADLINE.opener in values
    assert HEADLINE.ceiling in values
    # Home law (owner, 2026-08-23): the close grade's figure is home on
    # The History page owns row-level outcomes -- findings must not re-tile it.
    assert HEADLINE.close not in values


def test_hero_paragraphs_quote_the_headline_not_a_literal() -> None:
    blob = " ".join(HERO_PARAGRAPHS)
    assert HEADLINE.opener in blob
    assert HEADLINE.games in blob
    assert str(HEADLINE.extra_correct_per_season) in blob
    assert "86% likely real" in blob
    # The arrest evaluation's accuracy pair belongs to the model record;
    # the findings hero names it verbally instead (home law, 2026-08-23).
    from nfl_ats.player_arrests_back_side_overlay import (
        POLICY_BASELINE_OPENER_ACCURACY,
        POLICY_OPENER_ACCURACY,
    )

    assert f"{POLICY_OPENER_ACCURACY:.2%}" not in blob
    assert f"{POLICY_BASELINE_OPENER_ACCURACY:.2%}" not in blob


def test_edge_points_and_extra_picks_follow_the_accuracy() -> None:
    assert HEADLINE.edge_points == f"{HEADLINE.opener_accuracy - 50.0:.1f}"
    assert HEADLINE.extra_correct_per_season == round(
        (HEADLINE.opener_accuracy - 50.0) / 100.0 * 285
    )


def test_active_model_grades_are_never_typed_into_the_prose() -> None:
    """The exact failure of 2026-08-18: grades typed in, then left stale.

    Other measurements may be literals -- the page cites dozens of them. What
    must never be a literal is a number that CHANGES when the active model
    changes, because that is the one nobody remembers to update.
    """

    source = findings_content.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    # The dataclass and HEADLINE definition are where literals belong.
    _, _, body = text.partition('    source="docs/opener_evaluation.md",\n)')
    for exemption in LITERAL_EXEMPTIONS:
        body = body.replace(exemption, "")

    for label, literal in (("opener", HEADLINE.opener), ("close", HEADLINE.close)):
        assert literal not in body, (
            f"The active model's {label} grade {literal} is typed into the findings "
            f"prose. Use HEADLINE.{label} so it tracks the model. If this sentence is "
            "genuinely about a different measurement, add it to LITERAL_EXEMPTIONS "
            "with a comment naming that measurement."
        )


def test_headline_drives_rendered_text() -> None:
    """Changing HEADLINE must change what the page says."""

    moved = replace(HEADLINE, opener_accuracy=HEADLINE.opener_accuracy + 1.0)
    assert moved.opener != HEADLINE.opener
    assert moved.edge_points != HEADLINE.edge_points
    assert moved.extra_correct_per_season > HEADLINE.extra_correct_per_season

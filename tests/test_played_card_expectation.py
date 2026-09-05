"""The played-card expectation numbers must never drift silently.

2026-08-23, owner question "what edge am I playing": the dashboard's crowned
hero is now a PLANNING synthesis pinned in
:data:`nfl_ats.dashboard.findings_content.PLAYED_CARD_EXPECTATION_HERO`, not a
measured figure, so every value behind it is frozen here with its provenance.
These tests follow ``tests/test_findings_headline.py``: if a constant moves,
the build fails until a human re-derives it from the named source.

Provenance (all verified against the named sources when these tests were
written):

- docs/overlay_subset_composition.md -- paired +1.2641 accuracy points over
  the coach->arrest chain on the same paired opener archive, P+ 0.85715;
  selection-inflated archive score 55.4225% chosen as best of 127 subsets;
  "[Inferred] Decision expectation" section for the de-inflated planning
  synthesis (~55%, roughly +1 point over the chain).
- registry/weak_signals.json entry ``redteam_overlay_subset_loso_cv``
  (docs/edge_audit_redteam.md) -- out-of-sample re-check of the SELECTION
  step: 0.0000 pts, P+ 0.4930.
- registry/weak_signals.json entry ``movement_rule_composed_chain``
  (docs/movement_composition_eval.md results table) -- movement rule composed
  on top of the played chain: paired +1.5303 accuracy points on the same
  1,503 reused games; week-blocked P+ 0.8942, season-blocked P+ 0.9297.

Consolidation law (2026-08-23, owner, binding): these numbers render ONLY
inside the picks page's ONE collapsed ladder ``<details>`` (or the models
page's one-sentence pointer); the index default view carries exactly two
accuracy stats -- the ≈55% hero and the measured chain history.
"""

from __future__ import annotations

from nfl_ats.card_explanation import BANNED_BOILERPLATE
from nfl_ats.dashboard import findings_content

_CHAIN = 0.541583499667332


def test_expectation_hero_is_the_approx_sign_on_55_percent() -> None:
    """The hero must read as an APPROXIMATION (planning estimate, never a
    measured figure) at exactly 55 percent."""

    assert findings_content.PLAYED_CARD_EXPECTATION_PERCENT == 55
    assert findings_content.PLAYED_CARD_EXPECTATION_HERO == "\u224855%"
    # The approx sign must be the real character, not an HTML entity: the
    # models-page summary path escapes its text, which would mangle "&#8776;".
    assert "&#" not in findings_content.PLAYED_CARD_EXPECTATION_HERO


def test_overlay_union_paired_effect_constants_are_frozen() -> None:
    """docs/overlay_subset_composition.md: paired +1.2641383899 accuracy
    points, probability_positive 0.85715."""

    assert findings_content.OVERLAY_UNION_PAIRED_EFFECT_POINTS == 1.2641
    assert findings_content.OVERLAY_UNION_PAIRED_PROBABILITY_POSITIVE == 0.85715


def test_selection_inflation_constants_are_frozen() -> None:
    """docs/overlay_subset_composition.md: candidate accuracy 55.4225%
    selected as the maximum over 127 correlated subsets."""

    assert findings_content.OVERLAY_UNION_ARCHIVE_SCORE_FRACTION == 0.554225
    assert findings_content.OVERLAY_UNION_SUBSET_COUNT == 127


def test_selection_recheck_constants_are_frozen() -> None:
    """registry ``redteam_overlay_subset_loso_cv`` / docs/edge_audit_redteam.md:
    leave-one-season-out CV of the selection step measured 0.0000 pts,
    P+ 0.4930."""

    assert findings_content.OVERLAY_SELECTION_RECHECK_POINTS == 0.0
    assert findings_content.OVERLAY_SELECTION_RECHECK_P_PLUS == 0.4930


def test_movement_composed_constants_are_frozen() -> None:
    """registry ``movement_rule_composed_chain`` / docs/movement_composition_eval.md
    results table: paired +1.5303 accuracy points over the played chain on the
    same reused games; week-blocked P+ 0.8942, season-blocked P+ 0.9297."""

    assert findings_content.MOVEMENT_COMPOSED_EFFECT_POINTS == 1.5303
    assert findings_content.MOVEMENT_COMPOSED_WEEK_P_PLUS == 0.8942
    assert findings_content.MOVEMENT_COMPOSED_SEASON_P_PLUS == 0.9297


def test_ladder_rungs_render_the_pinned_sentences_in_fixed_order() -> None:
    """The collapsed ladder's entire content: one sentence per rung, this
    order, no others (the owner's consolidation law). The measured played-
    chain rung appears only when a chain artifact was actually read."""

    with_chain = findings_content.ladder_rungs(_CHAIN)
    assert with_chain == (
        (
            "Coin flip: 50%. The model on its own, before any situational rules: "
            "53.4% at the opener (1,537 games, 2020-2025); 52.1% at the sharper close."
        ),
        (
            "Played chain (model alone \u2192 coach fade \u2192 arrests): 54.2% measured "
            "on 1,503 paired games \u2014 the measured history under the crowned "
            "expectation."
        ),
        (
            "Fix-up rules: paired +1.26 points on reused data "
            "(86% likely real); its 55.4% archive score is inflated by picking the "
            "best of 127 similar combinations, and a fair out-of-sample re-check of "
            "that pick found only 0.00 pts (49% likely real) \u2014 already discounted "
            "in the \u224855% expectation."
        ),
        (
            "Movement rule (market-follow on >=1pt moves via refresh): "
            "composed +1.53 points (week to week 89% likely real, season to season "
            "93% likely real) \u2014 an attribution upper "
            "bound on already-looked-at data."
        ),
        (
            "Best documented long-run bettors: roughly 55-56% against the "
            "close. Measured pregame ceiling: about 56% (total-leak control, "
            "docs/leak_ceiling_control.md); the older 57-58% band was the "
            "pre-measurement guess."
        ),
        (
            "A small step above a coin flip could easily be erased by sportsbook "
            "vig alone. These are forced paper picks \u2014 not a game-level "
            "probability."
        ),
    )
    # Without a chain artifact the played rung omits itself; nothing else moves.
    without_chain = findings_content.ladder_rungs(None)
    assert without_chain == (
        with_chain[0],
        *with_chain[2:],
    )


def test_composed_sentences_render_the_pinned_values_exactly() -> None:
    """The prose constants are built from the pinned numbers above, so the
    rendered copy cannot drift away from the sources without this file
    failing alongside them."""

    assert findings_content.PLAYED_CARD_EXPECTATION_DEK == (
        "Planning estimate for the played card."
    )
    assert findings_content.LEDGER_PROMOTED_CAVEAT == (
        "Archive score was selection-inflated; played-card expectation "
        "\u224855% \u2014 full ladder on The Model page."
    )


def test_caveat_never_states_the_planning_estimate_as_measured() -> None:
    """AGENTS.md binding framing: measured and inferred must be distinguishable
    at a glance, and historical accuracy is never proof of a profitable edge --
    the composed sentences must keep their hedges. 2026-09-05 (owner, verbatim:
    "ive told you repeatedly to drop these fucking legal bullshit words"): the
    hedge stays a plain, honest fact (vig would likely erase a small edge)
    without the "not proof of a profitable or stable edge" legalese phrase."""

    blob = " ".join(
        (
            findings_content.PLAYED_CARD_EXPECTATION_DEK,
            findings_content.LEDGER_PROMOTED_CAVEAT,
            *findings_content.ladder_rungs(_CHAIN),
        )
    )
    assert "Planning estimate" in blob
    assert "selection-inflated" in blob
    assert "reused data" in blob
    assert "erased by sportsbook vig alone" in blob
    assert "not proof of a stable, profitable edge" not in blob
    for phrase in BANNED_BOILERPLATE:
        assert phrase not in blob.lower()

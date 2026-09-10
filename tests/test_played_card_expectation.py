from __future__ import annotations

from nfl_ats.card_explanation import BANNED_BOILERPLATE
from nfl_ats.dashboard import findings_content

_CHAIN = 0.541583499667332


def test_expectation_hero_is_the_approx_sign_on_55_percent() -> None:

    assert findings_content.PLAYED_CARD_EXPECTATION_PERCENT == 55
    assert findings_content.PLAYED_CARD_EXPECTATION_HERO == "\u224855%"
    assert "&#" not in findings_content.PLAYED_CARD_EXPECTATION_HERO


def test_overlay_union_paired_effect_constants_are_frozen() -> None:

    assert findings_content.OVERLAY_UNION_PAIRED_EFFECT_POINTS == 1.2641
    assert findings_content.OVERLAY_UNION_PAIRED_PROBABILITY_POSITIVE == 0.85715


def test_selection_inflation_constants_are_frozen() -> None:

    assert findings_content.OVERLAY_UNION_ARCHIVE_SCORE_FRACTION == 0.554225
    assert findings_content.OVERLAY_UNION_SUBSET_COUNT == 127


def test_selection_recheck_constants_are_frozen() -> None:

    assert findings_content.OVERLAY_SELECTION_RECHECK_POINTS == 0.0
    assert findings_content.OVERLAY_SELECTION_RECHECK_P_PLUS == 0.4930


def test_movement_composed_constants_are_frozen() -> None:

    assert findings_content.MOVEMENT_COMPOSED_EFFECT_POINTS == 1.5303
    assert findings_content.MOVEMENT_COMPOSED_WEEK_P_PLUS == 0.8942
    assert findings_content.MOVEMENT_COMPOSED_SEASON_P_PLUS == 0.9297


def test_ladder_rungs_render_the_pinned_sentences_in_fixed_order() -> None:

    with_chain = findings_content.ladder_rungs(_CHAIN)
    assert with_chain == (
        "Coin flip: 50%. The model's current opener and close grades are on The Model page.",
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
    without_chain = findings_content.ladder_rungs(None)
    assert without_chain == (
        with_chain[0],
        *with_chain[2:],
    )


def test_composed_sentences_render_the_pinned_values_exactly() -> None:

    assert findings_content.PLAYED_CARD_EXPECTATION_DEK == (
        "Planning estimate for the played card."
    )
    assert findings_content.LEDGER_PROMOTED_CAVEAT == (
        "Archive score was selection-inflated; played-card expectation "
        "\u224855% \u2014 full ladder on The Model page."
    )


def test_caveat_never_states_the_planning_estimate_as_measured() -> None:

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

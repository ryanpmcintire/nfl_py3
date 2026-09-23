from __future__ import annotations

from nfl_ats.card_explanation import BANNED_BOILERPLATE
from nfl_ats.dashboard import findings_content

_CHAIN = 0.541583499667332


def test_expectation_hero_is_the_approx_sign_on_55_percent() -> None:

    assert findings_content.PLAYED_CARD_EXPECTATION_PERCENT == 55
    assert findings_content.PLAYED_CARD_EXPECTATION_HERO == "\u224855%"
    assert "&#" not in findings_content.PLAYED_CARD_EXPECTATION_HERO


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

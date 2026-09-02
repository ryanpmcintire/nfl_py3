from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.preseason_prior import (
    COMPONENTS,
    PreseasonPriorConfig,
    PriorComponentRule,
    build_transaction_aware_preseason_prior,
)


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "team": ["OAK"],
            "decision_at_utc": ["2026-09-01T16:00:00Z"],
            "game_id": ["2026_01_LV_DEN"],
        }
    )


def _adjustment(
    *,
    component: str = "qb",
    adjustment: float = 2.0,
    uncertainty: float = 0.5,
    source_id: str = "source-a",
    observed: str = "2026-08-01T16:00:00Z",
    effective: str = "2026-08-02T16:00:00Z",
    application: str = "additive",
    priority: int = 0,
) -> dict[str, object]:
    return {
        "season": 2026,
        "team": "LV",
        "component": component,
        "adjustment": adjustment,
        "uncertainty": uncertainty,
        "units": "rating_points",
        "source_id": source_id,
        "source_observed_at_utc": observed,
        "effective_at_utc": effective,
        "application": application,
        "override_priority": priority,
    }


def test_default_configuration_is_neutral_but_retains_source_audit() -> None:
    result = build_transaction_aware_preseason_prior(_decisions(), pd.DataFrame([_adjustment()]))

    prior = result.priors.iloc[0]
    assert prior["team"] == "LV"
    assert prior["game_id"] == "2026_01_LV_DEN"
    assert prior["prior_value"] == 0.0
    assert prior["prior_uncertainty"] == 0.0
    assert prior["visible_source_count"] == 1
    assert result.source_audit.iloc[0]["configured_weight"] == 0.0
    assert bool(result.source_audit.iloc[0]["selected"])


def test_explicit_weights_apply_half_life_decay_and_quadrature_uncertainty() -> None:
    adjustments = pd.DataFrame(
        [
            _adjustment(
                adjustment=2.0,
                uncertainty=0.4,
                effective="2026-08-02T16:00:00Z",  # exactly 30 days old
            ),
            _adjustment(
                component="roster",
                adjustment=-1.0,
                uncertainty=0.3,
                source_id="source-b",
                effective="2026-08-22T16:00:00Z",  # exactly 10 days old
            ),
        ]
    )
    config = PreseasonPriorConfig(
        baseline_prior=1.0,
        qb=PriorComponentRule(weight=1.5, half_life_days=30.0),
        roster=PriorComponentRule(weight=2.0, half_life_days=None),
    )

    result = build_transaction_aware_preseason_prior(_decisions(), adjustments, config=config)
    prior = result.priors.iloc[0]

    assert prior["qb_contribution"] == pytest.approx(1.5)
    assert prior["roster_contribution"] == pytest.approx(-2.0)
    assert prior["prior_adjustment"] == pytest.approx(-0.5)
    assert prior["prior_value"] == pytest.approx(0.5)
    assert prior["prior_uncertainty"] == pytest.approx((0.3**2 + 0.6**2) ** 0.5)
    qb_audit = result.source_audit.loc[result.source_audit["component"] == "qb"].iloc[0]
    assert qb_audit["age_days"] == pytest.approx(30.0)
    assert qb_audit["decay_factor"] == pytest.approx(0.5)


def test_override_winner_is_deterministic_and_suppresses_additive_sources() -> None:
    adjustments = pd.DataFrame(
        [
            _adjustment(source_id="additive", adjustment=9.0),
            _adjustment(
                source_id="override-z",
                adjustment=3.0,
                application="override",
                priority=4,
                observed="2026-08-20T16:00:00Z",
            ),
            _adjustment(
                source_id="override-a",
                adjustment=-2.0,
                application="override",
                priority=4,
                observed="2026-08-20T16:00:00Z",
            ),
        ]
    )
    config = PreseasonPriorConfig(qb=PriorComponentRule(weight=1.0))

    first = build_transaction_aware_preseason_prior(_decisions(), adjustments, config=config)
    second = build_transaction_aware_preseason_prior(
        _decisions(), adjustments.sample(frac=1.0, random_state=7), config=config
    )

    assert first.priors.iloc[0]["prior_value"] == -2.0
    pd.testing.assert_frame_equal(first.priors, second.priors)
    first_audit = first.source_audit.set_index("source_id")
    assert bool(first_audit.loc["override-a", "selected"])
    assert first_audit.loc["override-a", "selection_reason"] == "override_winner"
    assert not bool(first_audit.loc["override-z", "selected"])
    assert first_audit.loc["override-z", "selection_reason"] == "lower_ranked_override"
    assert first_audit.loc["additive", "selection_reason"] == "suppressed_by_override"


def test_post_cutoff_mutations_cannot_change_prior_or_source_audit() -> None:
    pre_cutoff = _adjustment(source_id="visible", adjustment=1.25)
    future = _adjustment(
        component="draft",
        source_id="future",
        adjustment=100.0,
        observed="2026-09-01T16:00:01Z",
        effective="2026-08-01T16:00:00Z",
    )
    config = PreseasonPriorConfig(
        qb=PriorComponentRule(weight=1.0),
        draft=PriorComponentRule(weight=1.0),
    )
    baseline = build_transaction_aware_preseason_prior(
        _decisions(), pd.DataFrame([pre_cutoff, future]), config=config
    )

    mutated_future = dict(future)
    mutated_future["adjustment"] = -999_999.0
    extra_future = _adjustment(
        component="coaching",
        source_id="also-future",
        adjustment=777.0,
        observed="2026-09-02T16:00:00Z",
        effective="2026-09-02T16:00:00Z",
    )
    mutated = build_transaction_aware_preseason_prior(
        _decisions(),
        pd.DataFrame([pre_cutoff, mutated_future, extra_future]),
        config=config,
    )

    pd.testing.assert_frame_equal(baseline.priors, mutated.priors)
    pd.testing.assert_frame_equal(baseline.source_audit, mutated.source_audit)
    assert baseline.priors.iloc[0]["visible_source_count"] == 1


def test_future_effective_timestamp_is_not_visible_even_if_already_observed() -> None:
    adjustment = _adjustment(
        observed="2026-08-01T16:00:00Z",
        effective="2026-09-02T16:00:00Z",
    )
    config = PreseasonPriorConfig(qb=PriorComponentRule(weight=1.0))

    result = build_transaction_aware_preseason_prior(
        _decisions(), pd.DataFrame([adjustment]), config=config
    )

    assert result.priors.iloc[0]["prior_value"] == 0.0
    assert result.priors.iloc[0]["visible_source_count"] == 0
    assert result.source_audit.empty


def test_each_named_component_has_an_explicit_output() -> None:
    result = build_transaction_aware_preseason_prior(
        _decisions(),
        pd.DataFrame(
            [_adjustment(component=component, source_id=component) for component in COMPONENTS]
        ),
    )

    for component in COMPONENTS:
        assert f"{component}_contribution" in result.priors
        assert f"{component}_uncertainty" in result.priors


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"component": "weather"}, "unsupported adjustment components"),
        ({"units": "probability"}, "configured units"),
        ({"uncertainty": -0.01}, "non-negative"),
        ({"application": "replace-ish"}, "additive or override"),
    ],
)
def test_invalid_adjustment_contract_is_rejected(mutation: dict[str, object], match: str) -> None:
    row = _adjustment()
    row.update(mutation)
    with pytest.raises(ValueError, match=match):
        build_transaction_aware_preseason_prior(_decisions(), pd.DataFrame([row]))


def test_component_rule_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="positive"):
        PriorComponentRule(weight=1.0, half_life_days=0.0)

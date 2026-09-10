from __future__ import annotations

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.margin import margin_feature_columns
from nfl_ats.qb_identity_features import QB_REVENGE_COLUMN
from nfl_ats.transaction_flag_features import DEADLINE_INTEGRATION_DRAG_COLUMN


def test_stacked_profile_is_weak_stack_plus_both_columns_exactly() -> None:
    for target in ("margin", "market_residual"):
        base = set(margin_feature_columns(target, "weak_stack"))
        qb_revenge_only = set(margin_feature_columns(target, "weak_stack_qb_revenge"))
        deadline_drag_only = set(margin_feature_columns(target, "weak_stack_deadline_drag"))
        stacked = set(margin_feature_columns(target, "weak_stack_qb_revenge_deadline_drag"))

        assert qb_revenge_only - base == {QB_REVENGE_COLUMN}
        assert deadline_drag_only - base == {DEADLINE_INTEGRATION_DRAG_COLUMN}
        assert stacked - base == {QB_REVENGE_COLUMN, DEADLINE_INTEGRATION_DRAG_COLUMN}
        assert base - stacked == set()
        assert stacked == base | {QB_REVENGE_COLUMN, DEADLINE_INTEGRATION_DRAG_COLUMN}
        assert stacked == qb_revenge_only | deadline_drag_only


def test_stacked_profile_columns_are_claimed_by_the_two_existing_families() -> None:
    assert QB_REVENGE_COLUMN in FEATURE_FAMILIES["qb_revenge_on_production"]
    assert (
        DEADLINE_INTEGRATION_DRAG_COLUMN
        in FEATURE_FAMILIES["deadline_integration_drag_on_production"]
    )
    assert "weak_stack_qb_revenge_deadline_drag" not in FEATURE_FAMILIES

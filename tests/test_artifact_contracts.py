from __future__ import annotations

import pytest

from nfl_ats.artifact_contracts import (
    ARTIFACT_KINDS,
    CODE_LEGACY_UNVERSIONED,
    CODE_MISSING_COLUMNS,
    CODE_UNKNOWN_FORECAST_SCHEMA,
    CODE_VERSION_MISMATCH,
    CONTRACT_KEY,
    KIND_CARD,
    KIND_DECISION_LEDGER,
    KIND_FEATURE_TABLE,
    KIND_FORECAST,
    KIND_LOCKDAY_PACKAGE,
    KIND_PICK_REVISION_LEDGER,
    ArtifactContractError,
    check_compatible,
    check_ledger,
    read_contract,
    stamp,
)
from nfl_ats.prediction_safety import (
    PredictionSafetyError,
    validate_outcome_prediction_card,
    validate_prediction_card,
    validate_prediction_compatibility,
)

# ---------------------------------------------------------------------------
# stamp() / read_contract() round trip, one per registered kind.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(ARTIFACT_KINDS))
def test_stamp_round_trips_for_every_registered_kind(kind: str) -> None:
    spec = ARTIFACT_KINDS[kind]
    original = {"some_field": 1, "destination": "x"}
    stamped = stamp(kind, original)

    # Additive: original keys survive untouched, and the input is not mutated.
    assert original == {"some_field": 1, "destination": "x"}
    assert stamped["some_field"] == 1
    assert stamped["destination"] == "x"
    assert CONTRACT_KEY in stamped

    contract = read_contract(stamped)
    assert contract.legacy is False
    assert contract.kind == spec.kind
    assert contract.schema_version == spec.schema_version
    assert contract.builder_version == spec.builder_version
    assert contract.builder_module == spec.builder_module


def test_stamp_rejects_unknown_kind() -> None:
    with pytest.raises(ArtifactContractError):
        stamp("not_a_registered_kind", {})


def test_read_contract_accepts_a_path(tmp_path) -> None:
    import json

    stamped = stamp(KIND_FEATURE_TABLE, {"rows": 10})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(stamped), encoding="utf-8")

    contract = read_contract(path)
    assert contract.legacy is False
    assert contract.kind == KIND_FEATURE_TABLE


def test_read_contract_reports_legacy_for_a_block_with_no_contract() -> None:
    contract = read_contract({"rows": 10})
    assert contract.legacy is True
    assert contract.kind is None
    assert contract.schema_version is None


# ---------------------------------------------------------------------------
# check_compatible: legacy_unversioned warns, version_mismatch refuses.
# ---------------------------------------------------------------------------


def test_check_compatible_with_no_active_model_has_no_issues() -> None:
    feature_table = stamp(KIND_FEATURE_TABLE, {"rows": 1})
    report = check_compatible(None, feature_table)
    assert report.compatible
    assert report.issues == ()


def test_check_compatible_legacy_unversioned_is_a_warning_not_a_failure() -> None:
    # Neither side carries an artifact_contract block -- exactly today's real
    # active_ats_model.json / pre-ENG-09 feature-table manifests.
    model_manifest = {"model_id": "abc"}
    feature_table = {"rows": 1}
    report = check_compatible(model_manifest, feature_table)
    assert report.compatible
    assert not report.hard_failures
    assert report.warnings
    assert report.warnings[0].code == CODE_LEGACY_UNVERSIONED


def test_check_compatible_matching_versions_are_compatible() -> None:
    feature_table = stamp(KIND_FEATURE_TABLE, {"rows": 1})
    contract = feature_table[CONTRACT_KEY]
    model_manifest = {
        "model_id": "abc",
        "feature_table_schema_version": contract["schema_version"],
        "feature_table_builder_version": contract["builder_version"],
    }
    report = check_compatible(model_manifest, feature_table)
    assert report.compatible
    assert report.issues == ()


def test_check_compatible_schema_mismatch_is_a_hard_failure() -> None:
    feature_table = stamp(KIND_FEATURE_TABLE, {"rows": 1})
    contract = feature_table[CONTRACT_KEY]
    model_manifest = {
        "model_id": "abc",
        "feature_table_schema_version": contract["schema_version"] + 1,
        "feature_table_builder_version": contract["builder_version"],
    }
    report = check_compatible(model_manifest, feature_table)
    assert not report.compatible
    assert report.hard_failures[0].code == CODE_VERSION_MISMATCH
    with pytest.raises(ArtifactContractError):
        report.refuse_if_incompatible(action="fit a model on this feature table")


def test_check_compatible_builder_version_mismatch_is_a_hard_failure() -> None:
    feature_table = stamp(KIND_FEATURE_TABLE, {"rows": 1})
    contract = feature_table[CONTRACT_KEY]
    model_manifest = {
        "model_id": "abc",
        "feature_table_schema_version": contract["schema_version"],
        "feature_table_builder_version": "some-other-builder-version",
    }
    report = check_compatible(model_manifest, feature_table)
    assert not report.compatible
    assert report.hard_failures[0].code == CODE_VERSION_MISMATCH


def test_check_compatible_forecast_legacy_unversioned_warns() -> None:
    report = check_compatible(None, None, forecast_metadata={"created_at_utc": "x"})
    assert report.compatible
    assert report.warnings[0].code == CODE_LEGACY_UNVERSIONED


def test_check_compatible_forecast_unknown_schema_version_is_a_hard_failure() -> None:
    forecast = stamp(KIND_FORECAST, {"created_at_utc": "x"})
    forecast[CONTRACT_KEY]["schema_version"] = 999
    report = check_compatible(None, None, forecast_metadata=forecast)
    assert not report.compatible
    assert report.hard_failures[0].code == CODE_UNKNOWN_FORECAST_SCHEMA


def test_check_compatible_forecast_recognized_schema_version_is_fine() -> None:
    forecast = stamp(KIND_FORECAST, {"created_at_utc": "x"})
    report = check_compatible(None, None, forecast_metadata=forecast)
    assert report.compatible
    assert report.issues == ()


def test_refuse_if_incompatible_is_a_no_op_when_compatible() -> None:
    report = check_compatible(None, stamp(KIND_FEATURE_TABLE, {}))
    report.refuse_if_incompatible(action="do anything")  # must not raise


# ---------------------------------------------------------------------------
# check_ledger: missing required columns always refuses.
# ---------------------------------------------------------------------------


def test_check_ledger_passes_with_every_required_column_present() -> None:
    columns = ARTIFACT_KINDS[KIND_PICK_REVISION_LEDGER].required
    report = check_ledger(KIND_PICK_REVISION_LEDGER, columns)
    assert report.compatible
    assert report.issues == ()


def test_check_ledger_refuses_on_a_missing_column() -> None:
    columns = list(ARTIFACT_KINDS[KIND_DECISION_LEDGER].required)
    columns.remove("pick_side")
    report = check_ledger(KIND_DECISION_LEDGER, columns)
    assert not report.compatible
    assert report.hard_failures[0].code == CODE_MISSING_COLUMNS
    assert "pick_side" in report.hard_failures[0].message
    with pytest.raises(ArtifactContractError):
        report.refuse_if_incompatible(action="load this ledger")


# ---------------------------------------------------------------------------
# The registry's ledger column lists must not drift from their source of
# truth. artifact_contracts.py cannot import nfl_ats.clv / nfl_ats.pick_refresh
# directly (see the module docstring on the circular-import hazard); this
# test is the mechanical backstop that catches drift instead.
# ---------------------------------------------------------------------------


def test_decision_ledger_columns_match_clv_source_of_truth() -> None:
    from nfl_ats.clv import PAPER_DECISION_COLUMNS

    assert ARTIFACT_KINDS[KIND_DECISION_LEDGER].required == PAPER_DECISION_COLUMNS


def test_pick_revision_ledger_columns_match_pick_refresh_source_of_truth() -> None:
    from nfl_ats.pick_refresh import PICK_REVISION_COLUMNS

    assert ARTIFACT_KINDS[KIND_PICK_REVISION_LEDGER].required == PICK_REVISION_COLUMNS


def test_lockday_package_schema_version_matches_its_source_of_truth() -> None:
    from nfl_ats.lockday_package import PACKAGE_SCHEMA_VERSION

    # lockday_package.build_manifest owns its own "schema_version" key with
    # narrower semantics (the package format's own version) and is
    # deliberately not wired to stamp() -- see ARTIFACT_KINDS[KIND_LOCKDAY_PACKAGE]'s
    # description. This just keeps the registry's number honest.
    assert ARTIFACT_KINDS[KIND_LOCKDAY_PACKAGE].schema_version == PACKAGE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# prediction_safety integration: the one release-blocking contract check.
# ---------------------------------------------------------------------------


def test_validate_prediction_compatibility_passes_on_a_compatible_report() -> None:
    report = check_compatible(None, stamp(KIND_FEATURE_TABLE, {}))
    audit = validate_prediction_compatibility(report)
    assert audit.status == "PASS"
    assert audit.checks_passed == ("artifact_contract",)


def test_validate_prediction_compatibility_warns_on_legacy_unversioned() -> None:
    report = check_compatible({"model_id": "x"}, {"rows": 1})
    audit = validate_prediction_compatibility(report)
    assert audit.status == "PASS_WITH_WARNINGS"
    assert audit.warnings
    assert CODE_LEGACY_UNVERSIONED in audit.warnings[0]


def test_validate_prediction_card_fails_closed_on_a_contract_mismatch(model_frame) -> None:
    from nfl_ats.backtest import score_week

    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    feature_table = stamp(KIND_FEATURE_TABLE, {})
    contract = feature_table[CONTRACT_KEY]
    model_manifest = {
        "feature_table_schema_version": contract["schema_version"] + 1,
        "feature_table_builder_version": contract["builder_version"],
    }
    mismatch = check_compatible(model_manifest, feature_table)
    with pytest.raises(PredictionSafetyError):
        validate_prediction_card(
            predictions,
            min_edge=0.02,
            expected_season=2020,
            expected_week=1,
            compatibility=mismatch,
        )


def test_validate_prediction_card_passes_with_a_compatible_report(model_frame) -> None:
    from nfl_ats.backtest import score_week

    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    report = check_compatible(None, stamp(KIND_FEATURE_TABLE, {}))
    audit = validate_prediction_card(
        predictions,
        min_edge=0.02,
        expected_season=2020,
        expected_week=1,
        compatibility=report,
    )
    assert audit.status == "PASS"
    assert "artifact_contract" in audit.checks_passed


def test_validate_outcome_prediction_card_fails_closed_on_a_contract_mismatch(model_frame) -> None:
    from nfl_ats.outcomes import OUTCOME_METHODS, score_outcome_week

    predictions = score_outcome_week(model_frame, season=2020, week=1, min_train_games=80)
    feature_table = stamp(KIND_FEATURE_TABLE, {})
    contract = feature_table[CONTRACT_KEY]
    model_manifest = {
        "feature_table_schema_version": contract["schema_version"],
        "feature_table_builder_version": "not-the-real-builder-version",
    }
    mismatch = check_compatible(model_manifest, feature_table)
    with pytest.raises(PredictionSafetyError):
        validate_outcome_prediction_card(
            predictions,
            min_edge=0.02,
            expected_methods=OUTCOME_METHODS,
            expected_season=2020,
            expected_week=1,
            compatibility=mismatch,
        )


def test_card_kind_is_registered_and_stampable() -> None:
    # nfl_ats.publishing stamps the publish-predictions summary with this
    # kind; asserted here (rather than by driving the full publish path,
    # which requires a synchronized active model and live snapshots) so the
    # contract itself is covered without rebuilding that whole fixture.
    stamped = stamp(KIND_CARD, {"model_id": "abc", "season": 2026, "week": 1})
    contract = read_contract(stamped)
    assert contract.kind == KIND_CARD
    assert contract.legacy is False

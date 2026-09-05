"""ENG-14: per-source freshness budgets, degraded-mode fallbacks, and the
card-level roll-up.

One test per state (complete / degraded-via-allowed-fallback / blocked) per
source class, the overall roll-up, and a test that the published card metadata
carries the block. Every fixture is synthetic and written under ``tmp_path``;
nothing here reads or writes the real ``data/`` or ``artifacts/`` trees.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_publishing import (  # cross-test fixture reuse, as test_reporting does
    _publish_with_fresh_empty_arrest,
    _write_active_publication_fixture,
)
from test_weekly import _Recorder, _write_active_model, _write_data_root

from nfl_ats.player_arrests_back_side_overlay import MAX_SNAPSHOT_AGE
from nfl_ats.publishing import publish_active_predictions
from nfl_ats.source_freshness_policy import (
    BLOCKED,
    COMPLETE,
    DEGRADED,
    SOURCE_FRESHNESS_POLICIES,
    SourceFreshnessError,
    SourceObservation,
    evaluate_sources,
    observe_from_disk,
    policy_table,
    report_for_publication,
)
from nfl_ats.weekly import run_weekly

NOW = datetime(2026, 9, 8, 15, 30, tzinfo=UTC)
SOURCE_IDS = tuple(SOURCE_FRESHNESS_POLICIES)

#: Sources whose consumer is ALREADY fail-closed. The policy layer may never
#: grow this set without an owner decision -- that is the whole point of the
#: "degraded is the strongest state" rule in the module docstring.
FAIL_CLOSED_ON_ABSENCE = {"player_arrests"}
#: Sources with an existing anti-backdating gate (prediction_safety's
#: ``market_timing`` failure, and the arrest loader's negative-age refusal).
FAIL_CLOSED_ON_FUTURE_DATING = {"player_arrests", "odds_opener", "odds_refresh"}


def _stamp(instant: datetime) -> str:
    return instant.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# The table: budgets are derived from SCHEDULE, never typed in
# ---------------------------------------------------------------------------


def test_every_budget_matches_the_capture_schedule_arithmetic() -> None:
    """Hand-computed from ``scripts/capture_scheduler.py``'s ``SCHEDULE``:
    longest gap over one weekly cycle plus the grace of the job that closes it.
    A cadence change in that file must fail here rather than silently move a
    freshness budget."""

    expected = {
        # odds_tue_open alone: weekly, 10080 + 180 grace.
        "odds_opener": (10080, 180, 10260),
        # Six odds jobs; the longest gap is Tue 09:00 -> Thu 18:00 (3420 min),
        # closed by odds_thu_tnf's 90-minute grace.
        "odds_refresh": (3420, 90, 3510),
        # weekly_lock (Tue 09:15, grace 120) runs weekly-run step 1 ingest.
        "injuries_nflverse": (10080, 120, 10200),
        # ENG-39: same weekly_lock cadence as injuries_nflverse above -- this
        # row watches whether the CONSUMED player snapshot's date_modified is
        # real, not whether a capture landed.
        "injuries_nflverse_timestamps": (10080, 120, 10200),
        # Sat 10:00 -> Wed 17:30 = 6210 min, closed by a 240-minute grace.
        "injuries_sportradar": (6210, 240, 6450),
        # Sun 14:40 -> Thu 11:35 = 5575 min, closed by a 15-minute grace.
        "inactives": (5575, 15, 5590),
        # Daily noon capture: 1440 + 180 grace.
        "projected_lineups": (1440, 180, 1620),
        "referee_assignments": (10080, 240, 10320),
        # Cadence would allow 10080 + 90; the ALREADY-ENFORCED 36-hour
        # production gate tightens it to 2160.
        "player_arrests": (10080, 90, 2160),
        # Sat 07:00 -> Wed 07:00 = 5760 min, closed by a 120-minute grace.
        "pfr_transactions": (5760, 120, 5880),
        "airnow_weather": (10080, 15, 10095),
    }
    assert set(expected) == set(SOURCE_IDS)
    actual = {
        row["source_id"]: (
            row["recurrence_minutes"],
            row["grace_minutes"],
            row["budget_minutes"],
        )
        for row in policy_table()
    }
    assert actual == expected


def test_the_only_tightened_budget_is_the_constant_production_already_enforces() -> None:
    overridden = {
        policy.source_id
        for policy in SOURCE_FRESHNESS_POLICIES.values()
        if policy.budget_override_minutes is not None
    }
    assert overridden == {"player_arrests"}
    arrests = SOURCE_FRESHNESS_POLICIES["player_arrests"]
    assert arrests.budget_minutes == int(MAX_SNAPSHOT_AGE.total_seconds() // 60)
    # A derived budget is only ever tightened, never loosened.
    assert arrests.budget_minutes < arrests.recurrence_minutes + arrests.grace_minutes
    assert "MAX_SNAPSHOT_AGE" in arrests.budget_derivation


def test_no_currently_permitted_publish_path_becomes_newly_blockable() -> None:
    """The binding constraint on this module: only a source whose consumer is
    already fail-closed may reach ``blocked``."""

    blocking_on_absence = {
        policy.source_id
        for policy in SOURCE_FRESHNESS_POLICIES.values()
        if BLOCKED in (policy.on_absent, policy.on_stale)
    }
    assert blocking_on_absence == FAIL_CLOSED_ON_ABSENCE
    blocking_on_future = {
        policy.source_id
        for policy in SOURCE_FRESHNESS_POLICIES.values()
        if policy.on_future_dated == BLOCKED
    }
    assert blocking_on_future == FAIL_CLOSED_ON_FUTURE_DATING
    for policy in SOURCE_FRESHNESS_POLICIES.values():
        # Every degrading source must name what it falls back to; a degraded
        # state with no stated fallback is an outage in disguise.
        if not policy.fail_closed:
            assert policy.fallback and "none" not in policy.fallback[:5]
        assert policy.enforced_by


# ---------------------------------------------------------------------------
# One test per state, per source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", SOURCE_IDS)
def test_state_complete_when_the_snapshot_is_inside_budget(source_id: str) -> None:
    policy = SOURCE_FRESHNESS_POLICIES[source_id]
    fresh = NOW - timedelta(minutes=policy.budget_minutes / 2.0)
    report = evaluate_sources([SourceObservation(source_id, fresh)], NOW)

    assert report.sources[0].state == COMPLETE
    assert report.complete == (source_id,)
    assert report.state == COMPLETE
    assert "inside the" in report.sources[0].reason


@pytest.mark.parametrize("source_id", SOURCE_IDS)
def test_state_on_a_stale_snapshot_matches_the_declared_fallback(source_id: str) -> None:
    """Degraded sources fall back; the fail-closed source blocks. The boundary
    is exclusive: exactly-at-budget is still complete."""

    policy = SOURCE_FRESHNESS_POLICIES[source_id]
    at_budget = NOW - timedelta(minutes=policy.budget_minutes)
    just_over = NOW - timedelta(minutes=policy.budget_minutes + 1)

    assert evaluate_sources([SourceObservation(source_id, at_budget)], NOW).state == COMPLETE

    report = evaluate_sources([SourceObservation(source_id, just_over)], NOW)
    row = report.sources[0]
    assert row.state == policy.on_stale
    assert row.state == (BLOCKED if source_id in FAIL_CLOSED_ON_ABSENCE else DEGRADED)
    assert row.fallback == policy.fallback
    assert f"over the {policy.budget_minutes} min budget" in row.reason


@pytest.mark.parametrize("source_id", SOURCE_IDS)
def test_state_on_an_absent_snapshot_matches_the_declared_fallback(source_id: str) -> None:
    policy = SOURCE_FRESHNESS_POLICIES[source_id]
    report = evaluate_sources([SourceObservation(source_id, None)], NOW)
    row = report.sources[0]

    assert row.state == policy.on_absent
    assert row.age_minutes is None
    assert "no snapshot present" in row.reason
    if source_id in FAIL_CLOSED_ON_ABSENCE:
        assert row.state == BLOCKED
        assert report.state == BLOCKED
    else:
        assert row.state == DEGRADED
        assert report.state == DEGRADED


@pytest.mark.parametrize("source_id", SOURCE_IDS)
def test_state_on_a_future_dated_snapshot(source_id: str) -> None:
    policy = SOURCE_FRESHNESS_POLICIES[source_id]
    report = evaluate_sources([SourceObservation(source_id, NOW + timedelta(minutes=7))], NOW)
    row = report.sources[0]

    assert row.state == policy.on_future_dated
    assert "future-dated by 7.0 min" in row.reason
    expected = BLOCKED if source_id in FAIL_CLOSED_ON_FUTURE_DATING else DEGRADED
    assert row.state == expected


# ---------------------------------------------------------------------------
# The overall roll-up
# ---------------------------------------------------------------------------


def test_rollup_is_complete_only_when_every_observed_source_is_inside_budget() -> None:
    report = evaluate_sources(
        {
            source_id: NOW - timedelta(minutes=policy.budget_minutes / 2.0)
            for source_id, policy in SOURCE_FRESHNESS_POLICIES.items()
        },
        NOW,
    )
    assert report.state == COMPLETE
    assert set(report.complete) == set(SOURCE_IDS)
    assert report.degraded == ()
    assert report.blocked == ()
    assert report.unobserved == ()


def test_rollup_degrades_on_one_stale_fallback_source() -> None:
    observations = {
        source_id: NOW - timedelta(minutes=policy.budget_minutes / 2.0)
        for source_id, policy in SOURCE_FRESHNESS_POLICIES.items()
    }
    observations["referee_assignments"] = NOW - timedelta(days=30)
    report = evaluate_sources(observations, NOW)

    assert report.state == DEGRADED
    assert report.degraded == ("referee_assignments",)
    assert report.blocked == ()
    assert report.blocking_reasons == ()
    assert "DEGRADED" in report.summary_line()


def test_rollup_blocks_when_the_fail_closed_source_breaches() -> None:
    observations = {
        source_id: NOW - timedelta(minutes=policy.budget_minutes / 2.0)
        for source_id, policy in SOURCE_FRESHNESS_POLICIES.items()
    }
    observations["referee_assignments"] = None  # degraded, and outranked
    observations["player_arrests"] = NOW - timedelta(hours=48)
    report = evaluate_sources(observations, NOW)

    assert report.state == BLOCKED
    assert report.blocked == ("player_arrests",)
    assert len(report.blocking_reasons) == 1
    reason = report.blocking_reasons[0]
    # The message must name the SOURCE and the RULE, not just "stale".
    assert reason.startswith("player_arrests: ")
    assert "budget 2160 min, fail-closed" in reason
    assert "load_latest_complete_arrest_snapshot" in reason
    assert "player_arrests" in report.block_message()


def test_an_empty_report_is_degraded_not_complete() -> None:
    """Nothing was looked at, so nothing may be claimed complete."""

    report = evaluate_sources([], NOW)
    assert report.state == DEGRADED
    assert report.sources == ()
    assert set(report.unobserved) == set(SOURCE_IDS)


def test_unobserved_sources_never_contribute_to_the_rollup() -> None:
    report = evaluate_sources({"odds_opener": NOW - timedelta(minutes=5)}, NOW)

    assert report.state == COMPLETE
    assert report.complete == ("odds_opener",)
    assert "player_arrests" in report.unobserved
    assert report.blocked == ()


def test_an_unknown_source_id_is_refused_rather_than_ignored() -> None:
    with pytest.raises(KeyError, match="Unknown source policy id"):
        evaluate_sources([SourceObservation("odds_openr", NOW)], NOW)


def test_summary_line_names_all_three_buckets() -> None:
    report = evaluate_sources(
        {
            "odds_opener": NOW - timedelta(minutes=5),
            "referee_assignments": None,
            "player_arrests": None,
        },
        NOW,
    )
    line = report.summary_line()
    assert "**Source freshness: BLOCKED.**" in line
    # Source ids render as words, not raw snake_case (owner mandate,
    # 2026-09-05: "this is for humans not the opus autist").
    assert "Complete: odds opener." in line
    assert "Degraded (allowed fallback): referee assignments." in line
    assert "Blocked: player arrests." in line
    assert "docs/source_freshness_policy.md" in line


def test_metadata_block_is_json_serialisable_and_carries_every_state() -> None:
    report = evaluate_sources({"odds_opener": NOW - timedelta(minutes=5), "inactives": None}, NOW)
    payload = report.to_metadata()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["state"] == DEGRADED
    assert payload["sources"]["inactives"]["state"] == DEGRADED
    assert payload["sources"]["inactives"]["fallback"].startswith("SOURCE_NO_SNAPSHOT")
    assert payload["sources"]["odds_opener"]["budget_minutes"] == 10260


# ---------------------------------------------------------------------------
# The disk observer (ENG-03 join point)
# ---------------------------------------------------------------------------


def test_observe_from_disk_reads_stamped_directory_names_not_mtimes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    newest = NOW - timedelta(minutes=30)
    for offset in (600, 30, 5000):
        (data_root / "raw" / "player_arrests" / _stamp(NOW - timedelta(minutes=offset))).mkdir(
            parents=True
        )
    # An unparseable sibling must be ignored, not crash the scan.
    (data_root / "raw" / "player_arrests" / "not-a-stamp").mkdir(parents=True)

    observations = {
        observation.source_id: observation
        for observation in observe_from_disk(
            data_root=data_root, artifacts_root=None, source_ids=["player_arrests", "inactives"]
        )
    }
    assert observations["player_arrests"].observed_at == newest
    # A directory that does not exist is ABSENT (we looked, nothing there).
    assert observations["inactives"].observed_at is None


def test_observe_from_disk_reads_the_lineups_json_timestamp(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    lineups = artifacts_root / "lineups" / "current"
    lineups.mkdir(parents=True)
    (lineups / "lineups.json").write_text(
        json.dumps({"generated_at": _stamp(NOW - timedelta(hours=3)), "games": []}),
        encoding="utf-8",
    )

    (observation,) = observe_from_disk(
        data_root=None, artifacts_root=artifacts_root, source_ids=["projected_lineups"]
    )
    assert observation.observed_at == NOW - timedelta(hours=3)
    assert evaluate_sources([observation], NOW).state == COMPLETE


def test_a_missing_root_leaves_a_source_unobserved_rather_than_absent(tmp_path: Path) -> None:
    """'We could not look' is not 'there is nothing there' -- conflating them
    is how a fail-closed source would start blocking a rendering path."""

    observed = observe_from_disk(data_root=None, artifacts_root=tmp_path)
    ids = {observation.source_id for observation in observed}
    assert "player_arrests" not in ids
    assert evaluate_sources(observed, NOW).state != BLOCKED


def test_report_for_publication_uses_the_verified_arrest_instant(tmp_path: Path) -> None:
    """The arrest observation comes from the loader's manifest, never from a
    directory scan: a newer but UNVERIFIED directory must not make it look
    fresher than the gate accepted."""

    data_root = tmp_path / "data"
    (data_root / "raw" / "player_arrests" / _stamp(NOW)).mkdir(parents=True)

    stale = report_for_publication(
        data_root=data_root,
        artifacts_root=tmp_path / "artifacts",
        now=NOW,
        arrest_snapshot_at=NOW - timedelta(hours=48),
        arrest_snapshot_id="verified-old",
    )
    assert stale.state == BLOCKED
    assert stale.blocked == ("player_arrests",)
    assert "verified-old" in stale.to_metadata()["sources"]["player_arrests"]["detail"]

    # No verified instant at all: unobserved, not blocked.
    unverified = report_for_publication(
        data_root=data_root, artifacts_root=tmp_path / "artifacts", now=NOW
    )
    assert "player_arrests" in unverified.unobserved
    assert unverified.state != BLOCKED


# ---------------------------------------------------------------------------
# The published card carries the block
# ---------------------------------------------------------------------------


def test_published_card_metadata_and_markdown_carry_the_source_policy(tmp_path: Path) -> None:
    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 8, 12, tzinfo=UTC)

    result = _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        published_at=instant,
    )

    block = result["source_policy"]
    assert isinstance(block, dict)
    # The fixture writes one fresh, hash-verified arrest snapshot and nothing
    # else, so the card is DEGRADED on the fallback sources and complete on the
    # only fail-closed one.
    assert block["state"] == DEGRADED
    assert block["complete"] == ["player_arrests"]
    assert "odds_opener" in block["degraded"]
    assert block["blocked"] == []
    assert block["evaluated_at_utc"] == instant.isoformat()
    assert json.loads(json.dumps(block)) == block

    card = destination.read_text(encoding="utf-8")
    assert "**Source freshness: DEGRADED.**" in card
    # Rendered as words in card prose (owner mandate, 2026-09-05) -- the
    # JSON metadata block above keeps the raw source id.
    assert "Complete: player arrests." in card
    assert "docs/source_freshness_policy.md" in card


def test_publish_refuses_when_a_fail_closed_source_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal names the source and the rule. Reached here by forcing a
    blocked report -- the arrest gate ordinarily raises first, which is exactly
    why this layer adds no newly blockable path."""

    _, readme = _write_active_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"
    instant = datetime(2026, 8, 12, tzinfo=UTC)
    # Build the fixture (fresh arrest snapshot, schedule snapshot) via a first,
    # successful publish so the failure below is attributable to the policy.
    _publish_with_fresh_empty_arrest(
        tmp_path, destination=destination, readme_path=readme, published_at=instant
    )
    before = destination.read_text(encoding="utf-8")

    def _blocked(**_: object) -> object:
        return evaluate_sources({"player_arrests": None}, instant)

    monkeypatch.setattr("nfl_ats.publishing.report_for_publication", _blocked)

    with pytest.raises(SourceFreshnessError) as error:
        publish_active_predictions(
            tmp_path,
            destination=destination,
            readme_path=readme,
            data_root=tmp_path / "test-data",
            published_at=instant,
        )
    message = str(error.value)
    assert "publication refused by source policy" in message
    assert "player_arrests" in message
    assert "fail-closed" in message
    # A refused publish must not have half-written the card.
    assert destination.read_text(encoding="utf-8") == before


def test_weekly_run_summary_lifts_the_publish_steps_source_policy(tmp_path: Path) -> None:
    """weekly-run answers "which sources fed this card" without re-walking the
    step records. It copies the publish step's block and never re-evaluates."""

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    block = evaluate_sources({"odds_opener": NOW, "referee_assignments": None}, NOW).to_metadata()
    runner = _Recorder(
        **{
            "margin-predict": {"synchronization_status": "SYNCHRONIZED"},
            "publish-predictions": {"source_policy": block},
        }
    )

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_prospective=True,
        runner=runner,
        progress=False,
    )

    assert summary["published"] is True
    assert summary["source_policy"] == block
    assert summary["source_policy"]["state"] == DEGRADED
    assert summary["source_policy"]["degraded"] == ["referee_assignments"]


def test_weekly_run_summary_omits_the_block_when_publish_did_not_report_one(
    tmp_path: Path,
) -> None:
    """No fabricated block: a publish step that returned nothing leaves the key
    absent rather than asserting a state nobody measured."""

    data_root = _write_data_root(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    _write_active_model(artifacts_root, season=2026, week=1, status="SYNCHRONIZED")
    runner = _Recorder(**{"margin-predict": {"synchronization_status": "SYNCHRONIZED"}})

    summary = run_weekly(
        season=2026,
        week=1,
        data_root=data_root,
        artifacts_root=artifacts_root,
        skip_prospective=True,
        runner=runner,
        progress=False,
    )

    assert "source_policy" not in summary

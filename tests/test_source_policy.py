from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pytest

from nfl_ats.odds_backfill import (
    DEFAULT_QUOTA_FLOOR,
    HISTORICAL_CREDITS_PER_MARKET_REGION,
)
from nfl_ats.source_policy import (
    SourcePolicyError,
    load_source_policies,
    require_acquisition,
    require_private_raw_destination,
    require_raw_redistribution,
    validate_review_currency,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.capture_scheduler as capture_scheduler  # noqa: E402
import scripts.ingest_nflcom_injuries as nflcom  # noqa: E402


def test_registry_covers_the_mkt09_sources_and_is_current() -> None:
    policies = load_source_policies()

    assert {
        "the_odds_api",
        "internet_archive_vegasinsider",
        "vegasinsider_content",
        "nfl_com_injuries",
        "nflverse",
        "sec_availability_sheet",
        "sbr_odds_archive",
        "spreadspoke_kaggle",
    }.issubset(policies)
    validate_review_currency(as_of=date(2026, 9, 2))


def test_odds_quota_contract_matches_the_enforced_executor_constants() -> None:
    quota = load_source_policies()["the_odds_api"].quota

    assert quota["historical_credits_per_market_region"] == (HISTORICAL_CREDITS_PER_MARKET_REGION)
    assert quota["historical_minimum_remaining"] == DEFAULT_QUOTA_FLOOR


def test_nflcom_new_acquisition_is_denied_before_output_or_network(tmp_path: Path) -> None:
    args = argparse.Namespace(out=tmp_path / "must-not-exist")

    with pytest.raises(SourcePolicyError, match="New acquisition is disabled"):
        nflcom.run_ingest(args)

    assert not args.out.exists()


def test_all_nflcom_injury_scheduler_jobs_are_paused() -> None:
    jobs = [job for job in capture_scheduler.SCHEDULE if job.name.startswith("injuries_")]

    assert len(jobs) == 4
    assert all(not job.enabled for job in jobs)
    assert all("PAUSED by MKT-09" in job.why for job in jobs)


def test_private_raw_policy_rejects_tracked_repo_destination_and_allows_external_root(
    tmp_path: Path,
) -> None:
    require_private_raw_destination("the_odds_api", ROOT / "data" / "market" / "raw")
    require_private_raw_destination("the_odds_api", tmp_path)

    with pytest.raises(SourcePolicyError, match="gitignored private data root"):
        require_private_raw_destination("the_odds_api", ROOT / "docs" / "raw-odds")


def test_unregistered_or_disallowed_actions_fail_closed() -> None:
    with pytest.raises(SourcePolicyError, match="not registered"):
        require_acquisition("new_mystery_feed")
    with pytest.raises(SourcePolicyError, match="disabled"):
        require_acquisition("vegasinsider_content")
    with pytest.raises(SourcePolicyError, match="redistribution is prohibited"):
        require_raw_redistribution("the_odds_api")


def test_review_currency_validator_makes_annual_reaudit_enforceable() -> None:
    with pytest.raises(SourcePolicyError, match="terms review is stale"):
        validate_review_currency(as_of=date(2027, 8, 26))

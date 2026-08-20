from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_player_arrests import (
    POINT_IN_TIME_COLUMNS,
    PlayerArrestsIngestError,
    ingest,
    normalize_rows,
    parse_sitedata,
    parse_table_page,
    point_in_time_view,
    sanitize_landing_html,
    table_post_fields,
)


def _landing(nonce: str = "nonce-123") -> bytes:
    payload = {
        "ajax_url": "https://databases.usatoday.com/wp-admin/admin-ajax.php",
        "ajax_nonce": nonce,
        "pageID": "10",
        "sortBy": "Date",
        "sortOrder": "desc",
    }
    return f"<script>var sitedata = {json.dumps(payload)};</script>".encode()


def _source_row(record_id: int, date: str) -> dict[str, object]:
    return {
        "PK_ID": record_id,
        "Date": date,
        "First_name": " Test ",
        "Last_name": f"Player{record_id}",
        "Team": "DET",
        "Position": "CB",
        "Case_1": "Arrested",
        "Category": "Example",
        "Description": f"description {record_id}",
        "Outcome": f"resolution {record_id}",
        "F12": "",
        "F13": "",
        "F11": "",
        "F14": "",
        "Links": "<a href='https://example.com'>source</a>",
    }


def _page(page: int, *, total: int = 3, page_size: int = 1) -> bytes:
    date = f"2025-01-{4 - page:02d}T00:00:00"
    payload = {
        "success": True,
        "data": {
            "Result": [_source_row(100 + page, date)],
            "defParams": {
                "q.pageNumber": page,
                "q.orderBy": "Date%20desc",
                "q.pageSize": page_size,
            },
            "totalResults": total,
        },
    }
    return json.dumps(payload).encode()


def test_landing_parser_and_post_contract_match_public_application() -> None:
    config = parse_sitedata(_landing())
    assert config["ajax_nonce"] == "nonce-123"
    assert table_post_fields("nonce-123", 7) == {
        "action": "cspFetchTable",
        "security": "nonce-123",
        "pageID": "10",
        "blogID": "",
        "sortBy": "Date",
        "sortOrder": "desc",
        "page": "7",
        "searches": "{}",
        "heads": "true",
    }


def test_landing_parser_refuses_changed_sort_contract() -> None:
    changed = _landing().replace(b'"sortOrder": "desc"', b'"sortOrder": "asc"')
    with pytest.raises(PlayerArrestsIngestError, match="default sort changed"):
        parse_sitedata(changed)


def test_landing_nonce_is_sanitized_before_persistence() -> None:
    sanitized = sanitize_landing_html(_landing(), "nonce-123")
    assert b"nonce-123" not in sanitized
    assert b"[REDACTED_EPHEMERAL_NONCE]" in sanitized


def test_table_parser_validates_page_and_pagination_metadata() -> None:
    rows, metadata = parse_table_page(_page(2), expected_page=2)
    assert rows[0]["PK_ID"] == 102
    assert metadata == {"total_results": 3, "page_size": 1, "total_pages": 3}

    with pytest.raises(PlayerArrestsIngestError, match="response reports page 2"):
        parse_table_page(_page(2), expected_page=1)


def test_outcome_and_resolution_text_cannot_enter_point_in_time_view() -> None:
    baseline = normalize_rows([_source_row(1, "2025-01-01T00:00:00")])
    mutated_rows = [_source_row(1, "2025-01-01T00:00:00")]
    mutated_rows[0]["Outcome"] = "retrospective resolution changed"
    mutated_rows[0]["Description"] = "retrospective narrative changed"
    mutated_rows[0]["Links"] = "retrospective link changed"
    mutated = normalize_rows(mutated_rows)

    pd.testing.assert_frame_equal(point_in_time_view(baseline), point_in_time_view(mutated))
    assert tuple(point_in_time_view(baseline).columns) == POINT_IN_TIME_COLUMNS
    assert not any("outcome" in column for column in POINT_IN_TIME_COLUMNS)
    assert not any("description" in column for column in POINT_IN_TIME_COLUMNS)


def test_ingest_resumes_only_missing_pages_and_records_hashes(tmp_path: Path) -> None:
    snapshot = tmp_path / "20260820T160000Z"
    calls: list[int] = []

    def fetch_page(_nonce: str, page: int) -> bytes:
        calls.append(page)
        return _page(page)

    partial = ingest(
        snapshot,
        max_pages=2,
        delay_seconds=0,
        landing_fetcher=_landing,
        page_fetcher=fetch_page,
        sleeper=lambda _seconds: None,
    )
    assert calls == [1, 2]
    assert partial["complete"] is False
    assert partial["cached_pages"] == [1, 2]
    assert partial["resume_command"].endswith("--snapshot 20260820T160000Z")

    calls.clear()
    complete = ingest(
        snapshot,
        max_pages=None,
        delay_seconds=0,
        landing_fetcher=_landing,
        page_fetcher=fetch_page,
        sleeper=lambda _seconds: None,
    )
    assert calls == [3]
    assert complete["complete"] is True
    assert complete["cached_pages"] == [1, 2, 3]
    assert complete["rows_cached"] == 3
    assert complete["resume_command"] is None
    assert complete["access"]["nonce_stored"] is False
    assert set(complete["files"]["raw_pages_sha256"]) == {
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    }
    safe = pd.read_parquet(snapshot / "incidents_point_in_time.parquet")
    assert tuple(safe.columns) == POINT_IN_TIME_COLUMNS
    assert safe["incident_date"].is_monotonic_decreasing
    for landing_check in (snapshot / "landing_checks").glob("*.html"):
        assert b"nonce-123" not in landing_check.read_bytes()

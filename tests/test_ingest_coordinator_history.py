"""Offline source, cache, failure and historical-revision leakage contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.coordinator_changes import build_coordinator_history_features
from nfl_ats.data import DataContractError

SPEC = importlib.util.spec_from_file_location(
    "ingest_coordinator_history",
    Path(__file__).parents[1] / "scripts/ingest_coordinator_history.py",
)
assert SPEC and SPEC.loader
ingest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ingest)
FIXTURE = Path(__file__).parent / "fixtures/coordinators/revision.json"


def rows(payload=None, cutoff="2019-09-01T00:00:00Z"):
    return ingest.parse_revision(
        payload or json.loads(FIXTURE.read_bytes()),
        season=2019,
        team="KC",
        cutoff=cutoff,
        retrieved_at="2026-09-07T00:00:00Z",
    )


def games():
    return pd.DataFrame(
        [
            {
                "game_id": f"g{week}",
                "season": 2019,
                "week": week,
                "home_team": "KC",
                "away_team": "GB",
                "decision_at": f"2019-09-{day:02}T00:00:00Z",
                "kickoff": f"2019-09-{day + 1:02}T00:00:00Z",
            }
            for week, day in [(1, 2), (2, 9)]
        ]
    )


def test_dated_roles_and_no_current_transclusion_expansion():
    parsed = rows()
    assert {r["role"] for r in parsed} == {"HC", "OC", "DC"}
    assert {r["effective_observed_at"] for r in parsed} == {"2019-03-25T17:56:25+00:00"}
    assert all(r["source_url"].endswith("oldid=123") for r in parsed)


@pytest.mark.parametrize("stamp", [None, "not-a-date", "2020-01-01T00:00:00Z"])
def test_invalid_or_later_revision_rejected(stamp):
    payload = json.loads(FIXTURE.read_bytes())
    payload["query"]["pages"][0]["revisions"][0]["timestamp"] = stamp
    with pytest.raises(ValueError, match="real timestamp"):
        rows(payload)


def test_ambiguous_roles_and_unexpanded_templates_excluded():
    payload = json.loads(FIXTURE.read_bytes())
    payload["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"] = (
        "*Offensive coordinator \u2013 [[One]] and [[Two]]\n"
        "*Defensive coordinator \u2013 [[First]]\n*Defensive coordinator \u2013 [[Second]]\n"
        "|coach = {{Current coach}}\n{{Chiefs staff}}"
    )
    assert rows(payload) == []


def test_later_revision_cannot_rewrite_earlier_decision():
    old = rows()
    baseline = build_coordinator_history_features(games(), pd.DataFrame(old))
    late = copy.deepcopy(old)
    for row in late:
        row.update(person="Correction", effective_observed_at="2019-09-10T00:00:00Z")
    actual = build_coordinator_history_features(games(), pd.DataFrame(old + late))
    pd.testing.assert_frame_equal(baseline, actual)
    assert actual.loc[1, "home_oc_changed"] == 0
    assert actual.loc[1, "home_oc_name"] == "Old OC"


def test_undated_and_other_season_excluded():
    history = rows()
    for row in history:
        row["observed_at_basis"] = "season_undated"
        row["effective_observed_at"] = None
    other = rows()
    for row in other:
        row["season"] = 2018
    actual = build_coordinator_history_features(games(), pd.DataFrame(history + other))
    assert actual["home_oc_changed"].isna().all()
    assert actual["home_oc_name"].isna().all()


def test_dated_row_with_missing_timestamp_fails_closed():
    history = rows()
    history[1]["effective_observed_at"] = None
    with pytest.raises(DataContractError):
        build_coordinator_history_features(games(), pd.DataFrame(history))


def test_resume_reuses_success_without_modifying_prior_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    calls = []

    def fetch(url):
        calls.append(url)
        return 200, FIXTURE.read_bytes()

    monkeypatch.setattr(ingest, "fetch", fetch)
    first = ingest.capture({"KC": "Template:Example staff"}, [2019])
    original = {p.name: p.read_bytes() for p in first.iterdir()}
    second = ingest.capture({"KC": "Template:Example staff"}, [2019])
    assert len(calls) == 1
    assert first != second
    assert original == {p.name: p.read_bytes() for p in first.iterdir()}
    assert json.loads((second / "manifest.json").read_bytes())["network_requests"] == 0


@pytest.mark.parametrize(
    "status,body", [(429, b"slow down"), (200, b'{"error":{"code":"maxlag"}}')]
)
def test_failure_preserved_and_stops_crawl(tmp_path, monkeypatch, status, body):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    calls = []

    def fetch(url):
        calls.append(url)
        return status, body

    monkeypatch.setattr(ingest, "fetch", fetch)
    destination = ingest.capture({"KC": "Template:Example staff"}, [2019, 2020])
    manifest = json.loads((destination / "manifest.json").read_bytes())
    assert len(calls) == 1
    assert manifest["stopped"] == "source_error"
    assert next(iter(destination.glob("*.response"))).read_bytes() == body


def test_budget_and_polite_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    monkeypatch.setattr(ingest, "fetch", lambda _: (200, FIXTURE.read_bytes()))
    with pytest.raises(ValueError, match="delay"):
        ingest.capture({"KC": "Template:Example staff"}, [2019], delay=0)
    destination = ingest.capture({"KC": "Template:Example staff"}, [2019, 2020], max_requests=1)
    manifest = json.loads((destination / "manifest.json").read_bytes())
    assert manifest["stopped"] == "request_budget"
    assert manifest["network_requests"] == 1

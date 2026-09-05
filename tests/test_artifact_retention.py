from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.artifact_retention as artifact_retention

# ---------------------------------------------------------------------------
# extract_path_refs / protected_node_for_ref / iter_json_strings
# ---------------------------------------------------------------------------


def test_extract_path_refs_forward_slash() -> None:
    text = "See artifacts/altitude_screen/20260821T182533Z/results.json for details."
    assert artifact_retention.extract_path_refs(text) == [
        "artifacts/altitude_screen/20260821T182533Z/results.json"
    ]


def test_extract_path_refs_windows_backslash_is_normalized() -> None:
    text = r"F:\Repos\nfl_py3\artifacts\altitude_screen\20260821T182533Z"
    assert artifact_retention.extract_path_refs(text) == [
        "artifacts/altitude_screen/20260821T182533Z"
    ]


def test_extract_path_refs_semicolon_and_paren_separated_list() -> None:
    text = (
        "docs/fluview_opener_look.md; "
        "artifacts/fam/20260831T164546Z/results.json (screen); "
        "artifacts/fam/20260831T164207Z/results.json (positive-control)"
    )
    refs = artifact_retention.extract_path_refs(text)
    assert refs == [
        "artifacts/fam/20260831T164546Z/results.json",
        "artifacts/fam/20260831T164207Z/results.json",
    ]


def test_extract_path_refs_rejects_bare_root() -> None:
    # A bare tree-root token is never a meaningful reference on its own.
    assert artifact_retention.extract_path_refs("the artifacts directory") == []
    assert artifact_retention.extract_path_refs("see data for details") == []


def test_extract_path_refs_rejects_template_placeholder() -> None:
    # Regression: docs/hc_year_one_fade.md contains a CLI usage template
    # "<artifacts/.../<confirmation-run-id>>" whose literal ".." segment,
    # once separator punctuation is stripped, must never collapse to the
    # bare root "artifacts" (which would wholesale-protect the entire tree).
    text = "  --artifact <artifacts/.../<confirmation-run-id>> `"
    assert artifact_retention.extract_path_refs(text) == []


def test_extract_path_refs_does_not_match_metadata_or_database() -> None:
    text = "metadata/foo and database/bar are not artifacts or data paths"
    assert artifact_retention.extract_path_refs(text) == []


def test_extract_path_refs_trailing_sentence_punctuation_stripped() -> None:
    text = "outputs land under artifacts/rehearsal_lockday."
    assert artifact_retention.extract_path_refs(text) == ["artifacts/rehearsal_lockday"]


def test_protected_node_for_ref_collapses_to_run_boundary() -> None:
    assert (
        artifact_retention.protected_node_for_ref(
            "artifacts/family/20260101T000000Z/nested/results.json"
        )
        == "artifacts/family/20260101T000000Z"
    )


def test_protected_node_for_ref_no_timestamp_returned_unchanged() -> None:
    assert (
        artifact_retention.protected_node_for_ref("artifacts/combined_stacker_look/result.json")
        == "artifacts/combined_stacker_look/result.json"
    )
    assert (
        artifact_retention.protected_node_for_ref("artifacts/rehearsal_lockday")
        == "artifacts/rehearsal_lockday"
    )


def test_iter_json_strings_walks_nested_structures() -> None:
    data = {
        "a": "one",
        "b": [1, "two", {"c": "three"}],
        "d": {"e": ["four", None, 5.0]},
    }
    assert set(artifact_retention.iter_json_strings(data)) == {"one", "two", "three", "four"}


# ---------------------------------------------------------------------------
# Synthetic repo fixture
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _set_mtime(path: Path, days_ago: float) -> None:
    import os
    import time

    now = time.time()
    target = now - days_ago * 86400
    os.utime(path, (target, target))


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A miniature repo exercising every protection path the tool supports."""

    repo = tmp_path / "repo"

    # -- artifacts/: one family with two runs (older unreferenced, newer
    #    unreferenced -- newest must survive on the newest-run guard alone)
    _touch(repo / "artifacts" / "family_a" / "20260101T000000Z" / "results.json", b"old-run")
    _touch(repo / "artifacts" / "family_a" / "20260601T000000Z" / "results.json", b"newest-run")

    # -- artifacts/: a second family with an old run explicitly cited by a doc
    #    (must be protected regardless of age or newest-status)
    _touch(repo / "artifacts" / "family_b" / "20260101T000000Z" / "results.json", b"cited-run")
    _touch(repo / "artifacts" / "family_b" / "20260201T000000Z" / "results.json", b"newer-run")

    # -- artifacts/: a family mentioned only by bare name in a doc (no
    #    timestamp) -- every run inside it must be protected via the
    #    ancestor-inclusive lookup, without collapsing the family to one node.
    _touch(repo / "artifacts" / "family_c" / "20260101T000000Z" / "out.csv", b"c1")
    _touch(repo / "artifacts" / "family_c" / "20260201T000000Z" / "out.csv", b"c2")

    # -- artifacts/: a flat family with no timestamped subdirectory (loose
    #    files directly inside it)
    _touch(repo / "artifacts" / "flat_family" / "result.json", b"flat-result")
    _touch(repo / "artifacts" / "flat_family" / "extra.parquet", b"flat-extra-data")

    # -- artifacts/: a loose top-level file (no subdirectory at all)
    _touch(repo / "artifacts" / "loose_top_file.csv", b"loose")

    # -- artifacts/prospective/ -- always protected regardless of content
    _touch(repo / "artifacts" / "prospective" / "challengers.json", b"{}")
    _touch(repo / "artifacts" / "prospective" / "20260101T000000Z" / "decisions.parquet", b"p")

    # -- artifacts/clv_ledger/ -- always protected
    _touch(repo / "artifacts" / "clv_ledger" / "decisions.parquet", b"ledger")

    # -- a family referenced ONLY through active_ats_model.json's bare
    #    "<family>/<stamp>" schema shape (decoupled from family_a so the
    #    newest-run-guard test below stays a clean, single-cause check).
    _touch(repo / "artifacts" / "family_d" / "20260101T000000Z" / "out.json", b"d1")

    # -- artifacts/active_ats_model.json -- always protected, and its content
    #    protects "family_d/<stamp>" / "family_b/<stamp>" via the bare
    #    family/stamp schema shape.
    active_model = {
        "historical_evaluation": {"artifact": "family_d/20260101T000000Z"},
        "weekly_forecast": {"artifact": "family_b/20260201T000000Z"},
    }
    _touch(
        repo / "artifacts" / "active_ats_model.json",
        json.dumps(active_model).encode("utf-8"),
    )

    # -- artifacts/rehearsal_lockday/ -- must never be decomposed (coarse,
    #    referenced wholesale by a doc)
    _touch(
        repo / "artifacts" / "rehearsal_lockday" / "sim" / "data" / "raw" / "x.parquet",
        b"y" * 1000,
    )

    # -- data/raw and data/processed
    _touch(repo / "data" / "raw" / "20260101T000000Z" / "schedules.parquet", b"raw-old")
    _touch(repo / "data" / "raw" / "20260601T000000Z" / "schedules.parquet", b"raw-new")
    _touch(repo / "data" / "processed" / "game_features.parquet", b"processed")
    _touch(repo / "data" / "processed" / "game_features.manifest.json", b"{}")

    # -- data/market and data/players (unreferenced entirely)
    _touch(repo / "data" / "market" / "raw" / "20260101T000000Z" / "quotes.parquet", b"m1")
    _touch(repo / "data" / "players" / "raw" / "20260101T000000Z" / "rosters.parquet", b"pl1")

    # -- data/other (cfb tree, unreferenced)
    _touch(repo / "data" / "cfb" / "pbp" / "raw" / "20260101T000000Z" / "pbp.parquet", b"cfb1")

    # -- registry sources
    registry_dir = repo / "registry"
    _touch(
        registry_dir / "weak_signals.json",
        json.dumps(
            {"signals": {"sig_one": {"source": "artifacts/family_b/20260101T000000Z/results.json"}}}
        ).encode("utf-8"),
    )
    _touch(registry_dir / "rotation_registry.json", json.dumps({}).encode("utf-8"))
    _touch(
        registry_dir / "experiments" / "some-run" / "20260101T000000Z.json",
        json.dumps(
            {"artifact_directory": r"C:\fake\repo\artifacts\family_b\20260101T000000Z"}
        ).encode("utf-8"),
    )

    # -- docs
    _touch(
        repo / "docs" / "family_c.md",
        b"Outputs from this screen land under `artifacts/family_c/` once scored.",
    )
    _touch(
        repo / "docs" / "week1_readiness.md",
        b"Rehearsal artifacts are kept at `artifacts/rehearsal_lockday/`.",
    )
    for name in ("README.md", "ROADMAP.md", "HANDOFF.md", "CURRENT_PREDICTIONS.md"):
        _touch(repo / name, b"nothing relevant here")

    # -- ages: mark everything 60 days old by default (stamped runs ignore
    #    this -- their age comes from the stamp in their own name, by
    #    design, so mtime games never move them across a threshold).
    for path in repo.rglob("*"):
        if path.is_file():
            _set_mtime(path, days_ago=60)

    # flat_family has no timestamps in either file's name, so both runs'
    # ages come purely from mtime. Keep extra.parquet strictly newer than
    # result.json so it always wins the newest-of-group guard regardless of
    # any age threshold, leaving result.json as a clean, age-filter-only
    # candidate for the threshold test below.
    _set_mtime(repo / "artifacts" / "flat_family" / "extra.parquet", days_ago=45)

    return repo


# ---------------------------------------------------------------------------
# collect_protected_refs
# ---------------------------------------------------------------------------


def test_collect_protected_refs_hardcoded_always_protected(synthetic_repo: Path) -> None:
    refs = artifact_retention.collect_protected_refs(synthetic_repo)
    assert "artifacts/active_ats_model.json" in refs
    assert "artifacts/prospective" in refs
    assert "artifacts/clv_ledger" in refs


def test_collect_protected_refs_doc_and_registry_citations(synthetic_repo: Path) -> None:
    refs = artifact_retention.collect_protected_refs(synthetic_repo)
    assert "artifacts/family_b/20260101T000000Z" in refs
    assert "artifacts/family_c" in refs
    assert "artifacts/rehearsal_lockday" in refs


def test_collect_protected_refs_active_model_bare_family_stamp_shape(
    synthetic_repo: Path,
) -> None:
    refs = artifact_retention.collect_protected_refs(synthetic_repo)
    assert "artifacts/family_d/20260101T000000Z" in refs
    assert "artifacts/family_b/20260201T000000Z" in refs


def test_collect_protected_refs_windows_absolute_path_in_registry_json(
    synthetic_repo: Path,
) -> None:
    refs = artifact_retention.collect_protected_refs(synthetic_repo)
    # The experiment registry entry's artifact_directory is an absolute
    # Windows path pointing at a *different* fake repo root -- only the
    # "artifacts/..." suffix should be captured, not the fake drive prefix.
    assert "artifacts/family_b/20260101T000000Z" in refs


# ---------------------------------------------------------------------------
# discover_runs / build_report
# ---------------------------------------------------------------------------


def test_report_family_c_is_decomposed_not_collapsed(synthetic_repo: Path) -> None:
    """A bare doc mention protects every run, but must not collapse the
    family into a single opaque node the way COARSE_NO_DESCEND does."""

    report = artifact_retention.build_report(synthetic_repo)
    family_c = next(row for row in report.family_rows if row.name == "family_c")
    assert family_c.run_count == 2
    assert family_c.protected_run_count == 2


def test_report_rehearsal_lockday_is_one_coarse_node(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    rehearsal = next(row for row in report.family_rows if row.name == "rehearsal_lockday")
    assert rehearsal.run_count == 1
    assert rehearsal.protected_run_count == 1
    assert rehearsal.total_bytes == 1000


def test_report_byte_accounting(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    family_a = next(row for row in report.family_rows if row.name == "family_a")
    assert family_a.run_count == 2
    assert family_a.total_bytes == len(b"old-run") + len(b"newest-run")
    assert family_a.largest_file_bytes == len(b"newest-run")
    assert family_a.oldest_stamp == "20260101T000000Z"
    assert family_a.newest_stamp == "20260601T000000Z"


def test_report_flat_family_has_two_loose_runs(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    flat = next(row for row in report.family_rows if row.name == "flat_family")
    assert flat.run_count == 2
    assert flat.total_bytes == len(b"flat-result") + len(b"flat-extra-data")


def test_report_root_files_bucket(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    root_files = next(row for row in report.family_rows if row.name == "(root files)")
    # loose_top_file.csv and active_ats_model.json both live directly under
    # artifacts/ with no subdirectory.
    assert root_files.run_count >= 2


def test_report_tree_rows_cover_expected_names(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    names = {row.name for row in report.tree_rows}
    assert names == {
        "artifacts",
        "data/raw",
        "data/processed",
        "data/market",
        "data/players",
        "data/other",
    }


# ---------------------------------------------------------------------------
# build_plan -- the core safety property
# ---------------------------------------------------------------------------


def test_plan_never_lists_a_protected_path(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    protected = artifact_retention.collect_protected_refs(synthetic_repo)

    for candidate in plan.candidates:
        assert artifact_retention._protection_for(candidate.rel, protected) is None, (
            f"{candidate.rel} is protected but appeared in the plan"
        )


def test_plan_never_lists_the_newest_run_of_a_group(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    candidate_rels = {c.rel for c in plan.candidates}

    # family_a's newest run has no doc/registry citation at all -- it must
    # survive purely on the newest-run guard.
    assert "artifacts/family_a/20260601T000000Z" not in candidate_rels
    # family_a's OLDER run is unreferenced and not the newest -- it must be
    # the one candidate this fixture produces for that family.
    assert "artifacts/family_a/20260101T000000Z" in candidate_rels


def test_plan_excludes_cited_and_wholesale_protected_families(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    candidate_rels = {c.rel for c in plan.candidates}

    for protected_rel in (
        "artifacts/family_b/20260101T000000Z",  # cited by weak_signals.json + registry experiment
        "artifacts/family_b/20260201T000000Z",  # cited by active_ats_model.json weekly_forecast
        "artifacts/family_c/20260101T000000Z",  # inside a bare-cited family
        "artifacts/family_c/20260201T000000Z",
        "artifacts/rehearsal_lockday",  # coarse + wholesale cited
        "artifacts/prospective/20260101T000000Z",
        "artifacts/clv_ledger/decisions.parquet",
        "artifacts/active_ats_model.json",
    ):
        assert protected_rel not in candidate_rels, f"{protected_rel} should never be a candidate"


def test_plan_respects_older_than_days_threshold(synthetic_repo: Path) -> None:
    # result.json is unstamped (age comes from its 60-day mtime) and never
    # the newest of its group (extra.parquet is kept at 45 days), so it is a
    # clean, single-cause probe of the age-threshold filter alone.
    target = "artifacts/flat_family/result.json"

    plan_strict = artifact_retention.build_plan(synthetic_repo, older_than_days=90)
    assert target not in {c.rel for c in plan_strict.candidates}  # only 60 days old

    plan_loose = artifact_retention.build_plan(synthetic_repo, older_than_days=30)
    assert target in {c.rel for c in plan_loose.candidates}  # 60 days > 30-day threshold


def test_plan_total_bytes_matches_candidate_sum(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    assert plan.total_bytes == sum(c.size_bytes for c in plan.candidates)


def test_plan_data_raw_old_run_is_never_a_candidate_even_when_unreferenced(
    synthetic_repo: Path,
) -> None:
    # ENG-19 gap-close: data/raw/20260101T000000Z is not cited anywhere and
    # is not the newest of its group (data/raw/20260601T000000Z is newer) --
    # unlike the real repo's README.md, this synthetic repo's README does
    # not mention "data/raw" at all. Before ENG-19 that made it a candidate
    # (see docs/artifact_retention.md Safety rule 3, which flagged this as
    # incidental, doc-reference-only protection); build_plan now excludes
    # every data/raw run unconditionally via
    # retention_policy.is_point_in_time_capture, so NEITHER run in this
    # family is ever a candidate, referenced or not.
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    candidate_rels = {c.rel for c in plan.candidates}
    assert "data/raw/20260101T000000Z" not in candidate_rels
    assert "data/raw/20260601T000000Z" not in candidate_rels


# ---------------------------------------------------------------------------
# Rendering + CLI
# ---------------------------------------------------------------------------


def test_report_to_json_round_trips(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    payload = artifact_retention.report_to_json(report)
    json.dumps(payload)  # must be JSON-serializable
    assert payload["mode"] == "report"
    assert {row["name"] for row in payload["by_top_level_tree"]} == {
        row.name for row in report.tree_rows
    }


def test_plan_to_json_round_trips(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    payload = artifact_retention.plan_to_json(plan)
    json.dumps(payload)
    assert payload["mode"] == "plan_dry_run_no_delete"
    assert payload["candidate_count"] == len(plan.candidates)


def test_render_report_text_smoke(synthetic_repo: Path) -> None:
    report = artifact_retention.build_report(synthetic_repo)
    text = artifact_retention.render_report_text(report)
    assert "By top-level tree" in text
    assert "By artifact family" in text


def test_render_plan_text_smoke(synthetic_repo: Path) -> None:
    plan = artifact_retention.build_plan(synthetic_repo, older_than_days=1)
    text = artifact_retention.render_plan_text(plan)
    assert "dry run" in text
    assert "no delete mode exists" in text


def test_main_report_json_cli(synthetic_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = artifact_retention.main(["--report", "--json", "--root", str(synthetic_repo)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "report"


def test_main_plan_json_cli(synthetic_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = artifact_retention.main(
        ["--plan", "--older-than-days", "1", "--json", "--root", str(synthetic_repo)]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "plan_dry_run_no_delete"
    rels = {c["rel"] for c in payload["candidates"]}
    assert "artifacts/family_b/20260101T000000Z" not in rels


def test_main_rejects_report_and_plan_together(synthetic_repo: Path) -> None:
    with pytest.raises(SystemExit):
        artifact_retention.main(["--report", "--plan", "--root", str(synthetic_repo)])


def test_main_has_no_delete_flag() -> None:
    # There is no delete mode in this tool -- guard against one being added
    # without a corresponding, explicitly-approved policy change.
    parser_help = artifact_retention.main.__module__
    assert parser_help  # module imports cleanly
    assert not hasattr(artifact_retention, "delete_candidates")
    assert not hasattr(artifact_retention, "prune")
    assert not hasattr(artifact_retention, "apply_plan")

"""Tests for the ENG-19 retention-class and disk-budget policy.

Covers `src/nfl_ats/artifact_retention_policy.py` in isolation (pure
classification + budget-math + read-only filesystem probes) and its wiring
into `scripts/artifact_retention.py` (`build_plan`'s point-in-time-capture
exclusion, `retention_class` on every `PlanCandidate`, and the `--budget-check`
CLI mode). Every filesystem-touching test uses a synthetic `tmp_path` repo,
never the real `artifacts/`/`data/` trees.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scripts.artifact_retention as artifact_retention
from nfl_ats import artifact_retention_policy as retention_policy

RetentionClass = retention_policy.RetentionClass

# ---------------------------------------------------------------------------
# classify() / is_scratch / is_point_in_time_capture -- pure unit tests
# ---------------------------------------------------------------------------


def test_classify_protected_is_always_evidence() -> None:
    # Evidence beats every other rule -- a doc-cited raw capture is still
    # "evidence" in the class vocabulary, not double-labelled as
    # point_in_time_capture (both are never-prune; the label should still
    # reflect *why*).
    assert (
        retention_policy.classify("data/raw", "data/raw/20260101T000000Z", protected=True)
        == RetentionClass.EVIDENCE
    )
    assert (
        retention_policy.classify("artifacts", "artifacts/.uv-cache/x", protected=True)
        == RetentionClass.EVIDENCE
    )


def test_classify_data_raw_market_players_are_point_in_time_capture() -> None:
    for tree, rel in (
        ("data/raw", "data/raw/20260101T000000Z"),
        ("data/market", "data/market/raw/20260101T000000Z-ncaaf"),
        ("data/players", "data/players/referee_assignments/20260901T193026Z"),
    ):
        assert retention_policy.classify(tree, rel, protected=False) == (
            RetentionClass.POINT_IN_TIME_CAPTURE
        ), rel


def test_classify_raw_segment_inside_mixed_tree_is_point_in_time_capture() -> None:
    # data/cfb/pbp/raw/... lives under the mixed "data/other" bucket -- no
    # whole-tree rule covers it, only the literal "raw" path segment.
    assert (
        retention_policy.classify(
            "data/other", "data/cfb/pbp/raw/20260101T000000Z", protected=False
        )
        == RetentionClass.POINT_IN_TIME_CAPTURE
    )


def test_classify_refresh_triggers_is_point_in_time_capture_without_raw_segment() -> None:
    # ENG-08's artifacts/refresh_triggers/ has no literal "raw" segment, so
    # this needs the explicit POINT_IN_TIME_ARTIFACT_PREFIXES entry, not the
    # generic path heuristic.
    assert (
        retention_policy.classify(
            "artifacts", "artifacts/refresh_triggers/20260901T000000Z", protected=False
        )
        == RetentionClass.POINT_IN_TIME_CAPTURE
    )
    assert "artifacts/refresh_triggers" in retention_policy.POINT_IN_TIME_ARTIFACT_PREFIXES
    # Ancestor-inclusive: a file nested deeper than the immediate stamp dir
    # still matches.
    assert retention_policy.is_point_in_time_capture(
        "artifacts", "artifacts/refresh_triggers/20260901T000000Z/log.json"
    )
    # A sibling family with a similar name must NOT match by accident.
    assert not retention_policy.is_point_in_time_capture(
        "artifacts", "artifacts/refresh_triggers_unrelated/x"
    )


def test_classify_prospective_scorecards_is_reproducible_by_default() -> None:
    # ENG-06's derived summary output is re-derivable from the
    # evidence-protected artifacts/prospective/ ledgers, so it gets no
    # special-case protection and falls through to "reproducible".
    assert (
        retention_policy.classify(
            "artifacts", "artifacts/prospective_scorecards/20260901T000000Z", protected=False
        )
        == RetentionClass.REPRODUCIBLE
    )


def test_classify_scratch_paths() -> None:
    for rel in (
        "artifacts/.uv-cache/CACHEDIR.TAG",
        "data/tmp/uv-cache/.lock",
        "data/tmp/uv-cache/sdists-v9/.gitignore",
        "artifacts/foo/__pycache__/bar.pyc",
        "artifacts/foo/.pytest_cache/v/cache",
    ):
        assert retention_policy.classify("artifacts", rel, protected=False) == (
            RetentionClass.SCRATCH
        ), rel


def test_classify_default_is_reproducible() -> None:
    assert (
        retention_policy.classify(
            "artifacts", "artifacts/margins/20260101T000000Z", protected=False
        )
        == RetentionClass.REPRODUCIBLE
    )
    assert (
        retention_policy.classify(
            "data/processed", "data/processed/game_features.parquet", protected=False
        )
        == RetentionClass.REPRODUCIBLE
    )


def test_point_in_time_trees_are_exactly_raw_market_players() -> None:
    # Locks down the whole-tree rule's membership -- a change here is a
    # policy change, not a refactor, and should be visible in a diff.
    assert (
        frozenset({"data/raw", "data/market", "data/players"})
        == retention_policy.POINT_IN_TIME_TREES
    )


def test_retention_classes_table_matches_prunability_contract() -> None:
    classes = retention_policy.RETENTION_CLASSES
    assert classes[RetentionClass.EVIDENCE].prunable is False
    assert classes[RetentionClass.EVIDENCE].min_age_days is None
    assert classes[RetentionClass.POINT_IN_TIME_CAPTURE].prunable is False
    assert classes[RetentionClass.POINT_IN_TIME_CAPTURE].min_age_days is None
    assert classes[RetentionClass.SCRATCH].prunable is True
    assert classes[RetentionClass.SCRATCH].min_age_days == 0
    assert classes[RetentionClass.REPRODUCIBLE].prunable is True
    assert classes[RetentionClass.REPRODUCIBLE].min_age_days == 30
    assert retention_policy.REPRODUCIBLE_MIN_AGE_DAYS == 30


# ---------------------------------------------------------------------------
# Disk budget math
# ---------------------------------------------------------------------------


def test_budget_bytes_for_tree_applies_multiplier() -> None:
    baseline = retention_policy.BUDGET_BASELINE_BYTES["artifacts"]
    assert retention_policy.budget_bytes_for_tree("artifacts", multiplier=2.0) == baseline * 2


def test_budget_bytes_for_tree_default_multiplier() -> None:
    baseline = retention_policy.BUDGET_BASELINE_BYTES["data/market"]
    expected = round(baseline * retention_policy.DEFAULT_BUDGET_MULTIPLIER)
    assert retention_policy.budget_bytes_for_tree("data/market") == expected


def test_budget_bytes_for_tree_unknown_tree_is_none() -> None:
    assert retention_policy.budget_bytes_for_tree("nonexistent-tree") is None


def test_budget_baseline_covers_every_top_level_tree_name() -> None:
    # top_level_tree_specs() in scripts/artifact_retention.py always
    # produces exactly these six tree names -- the budget baseline must
    # have an entry for each one or --budget-check silently skips a tree.
    expected_trees = {
        "artifacts",
        "data/raw",
        "data/processed",
        "data/market",
        "data/players",
        "data/other",
    }
    assert set(retention_policy.BUDGET_BASELINE_BYTES) == expected_trees


# ---------------------------------------------------------------------------
# Read-only filesystem probes
# ---------------------------------------------------------------------------


def test_measure_free_space_returns_positive_totals(tmp_path: Path) -> None:
    usage = retention_policy.measure_free_space(tmp_path)
    assert usage is not None
    assert usage.total_bytes > 0
    assert usage.free_bytes >= 0
    assert usage.used_bytes >= 0


def test_measure_free_space_missing_drive_returns_none() -> None:
    # A drive letter that (almost certainly) does not exist on this machine.
    assert retention_policy.measure_free_space(Path("Z:/definitely/not/a/real/path")) is None


def test_read_mirror_manifest_missing_returns_none(tmp_path: Path) -> None:
    assert retention_policy.read_mirror_manifest(tmp_path / "does_not_exist") is None


def test_read_mirror_manifest_reads_valid_json(tmp_path: Path) -> None:
    dest = tmp_path / "mirror"
    dest.mkdir()
    (dest / retention_policy.MIRROR_MANIFEST_NAME).write_text(
        json.dumps({"generated_utc": "2026-09-04T00:00:00+00:00"}), encoding="utf-8"
    )
    manifest = retention_policy.read_mirror_manifest(dest)
    assert manifest is not None
    assert manifest["generated_utc"] == "2026-09-04T00:00:00+00:00"


def test_read_mirror_manifest_corrupt_json_returns_none(tmp_path: Path) -> None:
    dest = tmp_path / "mirror"
    dest.mkdir()
    (dest / retention_policy.MIRROR_MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    assert retention_policy.read_mirror_manifest(dest) is None


# ---------------------------------------------------------------------------
# Integration: scripts/artifact_retention.py wiring
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _set_mtime(path: Path, days_ago: float) -> None:
    now = time.time()
    target = now - days_ago * 86400
    os.utime(path, (target, target))


@pytest.fixture
def policy_repo(tmp_path: Path) -> Path:
    """A small synthetic repo exercising all four retention classes."""

    repo = tmp_path / "repo"

    # evidence: doc-cited, older run of a two-run family
    _touch(repo / "artifacts" / "cited_family" / "20260101T000000Z" / "out.json", b"cited")
    _touch(repo / "artifacts" / "cited_family" / "20260601T000000Z" / "out.json", b"newer-uncited")
    _touch(
        repo / "docs" / "some_doc.md",
        b"See `artifacts/cited_family/20260101T000000Z` for the frozen baseline.",
    )

    # point_in_time_capture: whole-tree rule (data/raw), unreferenced,
    # older non-newest run of a two-run family
    _touch(repo / "data" / "raw" / "20260101T000000Z" / "schedules.parquet", b"raw-old")
    _touch(repo / "data" / "raw" / "20260601T000000Z" / "schedules.parquet", b"raw-new")

    # point_in_time_capture: literal "raw" segment inside a mixed tree
    _touch(
        repo / "data" / "cfb" / "pbp" / "raw" / "20260101T000000Z" / "pbp.parquet", b"cfb-raw-old"
    )
    _touch(
        repo / "data" / "cfb" / "pbp" / "raw" / "20260601T000000Z" / "pbp.parquet", b"cfb-raw-new"
    )

    # scratch: a stray uv cache under artifacts/, two files so the
    # newest-of-group guard does not swallow the older one
    _touch(repo / "artifacts" / ".uv-cache" / "CACHEDIR.TAG", b"cache-tag")
    _touch(repo / "artifacts" / ".uv-cache" / ".lock", b"cache-lock")

    # reproducible: an old, unreferenced, non-newest experiment run
    _touch(repo / "artifacts" / "margins" / "20260101T000000Z" / "results.json", b"old-margin-run")
    _touch(repo / "artifacts" / "margins" / "20260601T000000Z" / "results.json", b"new-margin-run")

    for name in ("README.md", "ROADMAP.md", "HANDOFF.md", "CURRENT_PREDICTIONS.md"):
        _touch(repo / name, b"nothing relevant here")
    _touch(repo / "registry" / "weak_signals.json", json.dumps({}).encode("utf-8"))
    _touch(repo / "registry" / "rotation_registry.json", json.dumps({}).encode("utf-8"))

    for path in repo.rglob("*"):
        if path.is_file():
            _set_mtime(path, days_ago=90)
    # Make the two scratch files unambiguously ordered: .lock is the
    # newest-of-group survivor, CACHEDIR.TAG stays a clean candidate.
    _set_mtime(repo / "artifacts" / ".uv-cache" / ".lock", days_ago=5)

    return repo


def test_build_plan_classifies_every_candidate(policy_repo: Path) -> None:
    plan = artifact_retention.build_plan(policy_repo, older_than_days=1)
    by_rel = {c.rel: c.retention_class for c in plan.candidates}
    assert by_rel["artifacts/margins/20260101T000000Z"] == "reproducible"
    assert by_rel["artifacts/.uv-cache/CACHEDIR.TAG"] == "scratch"


def test_point_in_time_capture_never_appears_in_a_prune_plan(policy_repo: Path) -> None:
    # The binding ENG-19 invariant. older_than_days=0 is the loosest
    # possible age threshold, so the only thing that could keep a run out
    # of the plan here is the class-based exclusion, not the age filter.
    plan = artifact_retention.build_plan(policy_repo, older_than_days=0)
    for candidate in plan.candidates:
        assert candidate.retention_class != "point_in_time_capture", candidate.rel

    # Confirm the fixture actually contains unreferenced, non-newest
    # point-in-time runs that a pre-ENG-19 plan WOULD have listed --
    # otherwise this test would pass trivially, with nothing to exclude.
    candidate_rels = {c.rel for c in plan.candidates}
    assert "data/raw/20260101T000000Z" not in candidate_rels
    assert "data/cfb/pbp/raw/20260101T000000Z" not in candidate_rels


def test_evidence_never_appears_in_a_prune_plan(policy_repo: Path) -> None:
    plan = artifact_retention.build_plan(policy_repo, older_than_days=0)
    candidate_rels = {c.rel for c in plan.candidates}
    assert "artifacts/cited_family/20260101T000000Z" not in candidate_rels
    assert not any(c.retention_class == "evidence" for c in plan.candidates)


def test_build_budget_check_reclaimable_matches_plan(policy_repo: Path) -> None:
    check = artifact_retention.build_budget_check(policy_repo, multiplier=5.0)
    plan = artifact_retention.build_plan(
        policy_repo, older_than_days=retention_policy.REPRODUCIBLE_MIN_AGE_DAYS
    )
    reclaimable_by_tree: dict[str, int] = {}
    for candidate in plan.candidates:
        reclaimable_by_tree[candidate.tree] = (
            reclaimable_by_tree.get(candidate.tree, 0) + candidate.size_bytes
        )
    for row in check.rows:
        assert row.reclaimable_bytes == reclaimable_by_tree.get(row.tree, 0)
    # The synthetic repo is a few hundred bytes per tree; the real,
    # GB-scale measured baseline makes every tree trivially under budget.
    assert check.any_over_budget is False


def test_build_budget_check_disk_usage_present(policy_repo: Path) -> None:
    check = artifact_retention.build_budget_check(policy_repo)
    assert check.disk_total_bytes is not None
    assert check.disk_free_bytes is not None


def test_budget_check_cli_under_budget_exit_code_zero(
    policy_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = artifact_retention.main(["--budget-check", "--root", str(policy_repo)])
    assert exit_code == 0
    assert "exit code 0" in capsys.readouterr().out.lower()


def test_budget_check_cli_over_budget_exit_code_one(
    policy_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = artifact_retention.main(
        ["--budget-check", "--budget-multiplier", "0", "--root", str(policy_repo)]
    )
    assert exit_code == 1
    assert "OVER BUDGET" in capsys.readouterr().out


def test_budget_check_cli_json_reports_over_budget(
    policy_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = artifact_retention.main(
        ["--budget-check", "--budget-multiplier", "0", "--json", "--root", str(policy_repo)]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "budget_check_dry_run_no_delete"
    assert payload["any_over_budget"] is True
    assert any(row["over_budget"] for row in payload["trees"])
    assert all(
        "retention_class" not in row for row in payload["trees"]
    )  # tree rows, not candidates


def test_main_rejects_plan_and_budget_check_together(policy_repo: Path) -> None:
    with pytest.raises(SystemExit):
        artifact_retention.main(["--plan", "--budget-check", "--root", str(policy_repo)])


def test_main_rejects_report_and_budget_check_together(policy_repo: Path) -> None:
    with pytest.raises(SystemExit):
        artifact_retention.main(["--report", "--budget-check", "--root", str(policy_repo)])


def test_no_delete_prune_or_apply_function_exists_anywhere() -> None:
    # Same guarantee as tests/test_artifact_retention.py::
    # test_main_has_no_delete_flag, re-asserted against both modules this
    # ENG-19 pass touches -- budget-check is dry-run/read-only too.
    for module in (artifact_retention, retention_policy):
        assert not hasattr(module, "delete_candidates")
        assert not hasattr(module, "prune")
        assert not hasattr(module, "apply_plan")
        assert not hasattr(module, "apply_budget")

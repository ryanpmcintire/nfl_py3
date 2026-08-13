from __future__ import annotations

from pathlib import Path


def test_handoff_automation_policy_is_tracked() -> None:
    root = Path(__file__).resolve().parents[1]
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    pre_commit = (root / ".githooks/pre-commit").read_text(encoding="utf-8")
    pre_push = (root / ".githooks/pre-push").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Never ask the user to run the handoff command" in agents
    assert "nfl-ats handoff" in pre_commit
    assert "git add -- HANDOFF.md" in pre_commit
    assert "refs/heads/master" in pre_push
    assert "nfl-ats handoff --check" in pre_push
    assert "Require handoff in every master tip" in workflow

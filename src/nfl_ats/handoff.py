"""Generate a tracked, human-readable session handoff from authoritative state."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.io import atomic_text

HANDOFF_VERSION = 1


@dataclass(frozen=True)
class RepositoryState:
    """Small Git snapshot captured immediately before the handoff is written."""

    branch: str
    commit: str
    subject: str
    changes: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.changes


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ValueError(f"Unable to inspect repository: {detail}")
    return result.stdout.rstrip("\r\n")


def inspect_repository(repo_root: Path) -> RepositoryState:
    """Read the current branch, commit, and worktree without changing Git state."""

    branch = _git(repo_root, "branch", "--show-current") or "DETACHED"
    commit = _git(repo_root, "rev-parse", "--short=12", "HEAD")
    subject = _git(repo_root, "log", "-1", "--pretty=%s")
    status = _git(repo_root, "status", "--short", "--untracked-files=all")
    return RepositoryState(
        branch=branch,
        commit=commit,
        subject=subject,
        changes=tuple(line for line in status.splitlines() if line),
    )


def _roadmap_priorities(roadmap_path: Path, limit: int = 6) -> list[str]:
    if not roadmap_path.is_file():
        return []
    text = roadmap_path.read_text(encoding="utf-8")
    marker = "## Recommended execution order"
    if marker not in text:
        return []
    section = text.split(marker, maxsplit=1)[1]
    priorities: list[str] = []
    current: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^\d+\.\s+(.*)", line)
        if match:
            if current:
                priorities.append(" ".join(current))
            current = [match.group(1).strip()]
            if len(priorities) >= limit:
                break
        elif current and line.startswith("   "):
            current.append(line.strip())
        elif current and not line.strip():
            continue
        elif current:
            break
    if current and len(priorities) < limit:
        priorities.append(" ".join(current))
    return priorities[:limit]


def _tracked_publication(predictions_path: Path) -> dict[str, str] | None:
    if not predictions_path.is_file():
        return None
    text = predictions_path.read_text(encoding="utf-8")
    title = re.search(r"^# NFL ATS predictions: (\d+) Week (\d+)$", text, re.MULTILINE)
    model = re.search(r"Published from synchronized model `([^`]+)` at `([^`]+)`", text)
    if title is None or model is None:
        return None
    return {
        "season": title.group(1),
        "week": title.group(2),
        "model_id": model.group(1),
        "published_at_utc": model.group(2),
    }


def _local_inventory(repo_root: Path, artifacts_root: Path) -> list[tuple[str, Path]]:
    return [
        ("canonical team features", repo_root / "data/processed/game_features.parquet"),
        ("play-by-play features", repo_root / "data/processed/game_features_pbp.parquet"),
        ("player features", repo_root / "data/processed/game_features_player.parquet"),
        ("active model manifest", artifacts_root / "active_ats_model.json"),
    ]


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _model_markdown(artifacts_root: Path) -> tuple[str, dict[str, Any] | None]:
    active = load_active_ats_model(artifacts_root)
    if active is None:
        return (
            "Local active-model artifacts are unavailable. This is expected in a fresh clone; "
            "use the tracked forecast below as the last published state and regenerate local "
            "artifacts before changing model claims.",
            None,
        )
    historical = active["historical_evaluation"]
    weekly = active["weekly_forecast"]
    evaluation_path = active_artifact_path(artifacts_root, active, "historical_evaluation")
    forecast_path = active_artifact_path(artifacts_root, active, "weekly_forecast")
    linked = bool(
        evaluation_path is not None
        and evaluation_path.is_dir()
        and forecast_path is not None
        and forecast_path.is_dir()
    )
    text = (
        f"- Status: **{active['status']}**; linked artifacts present: **{str(linked).lower()}**\n"
        f"- Model ID: `{active['model_id']}`\n"
        f"- Method/profile/regressor: `{active['method']}` / "
        f"`{active['feature_profile']}` / `{active['regressor']}`\n"
        f"- Historical ATS classification: **{historical['correct']:,} / "
        f"{historical['games']:,} ({historical['accuracy']:.2%})**\n"
        f"- Linked forecast: **{weekly['season']} Week {weekly['week']}**, created "
        f"`{weekly['created_at_utc']}`"
    )
    return text, active


def _changes_markdown(state: RepositoryState) -> str:
    if state.clean:
        return "none"
    preview = "\n".join(f"  - `{line}`" for line in state.changes[:20])
    remainder = len(state.changes) - 20
    suffix = f"\n  - ...and {remainder} more" if remainder > 0 else ""
    return f"{len(state.changes)} paths\n{preview}{suffix}"


def check_session_handoff(
    repo_root: Path,
    artifacts_root: Path,
    handoff_path: Path,
) -> dict[str, Any]:
    """Fail when the tracked handoff disagrees with current durable state."""

    path = handoff_path if handoff_path.is_absolute() else repo_root / handoff_path
    if not path.is_file():
        raise ValueError(f"Session handoff is missing: {path}")
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    if f"Handoff schema: `{HANDOFF_VERSION}`" not in text:
        failures.append("handoff schema is missing or unsupported")

    publication = _tracked_publication(repo_root / "CURRENT_PREDICTIONS.md")
    if publication is None:
        failures.append("tracked weekly publication is missing or invalid")
    else:
        expected_publication = (
            f"**{publication['season']} Week {publication['week']}** from model "
            f"`{publication['model_id']}`"
        )
        if expected_publication not in text:
            failures.append("tracked weekly publication is not reflected in the handoff")

    active = load_active_ats_model(artifacts_root)
    if active is not None:
        if f"Model ID: `{active['model_id']}`" not in text:
            failures.append("local active model is not reflected in the handoff")
        if publication is not None and active["model_id"] != publication["model_id"]:
            failures.append("local active model and tracked weekly publication do not match")

    priorities = _roadmap_priorities(repo_root / "ROADMAP.md")
    missing_priorities = [priority for priority in priorities if priority not in text]
    if missing_priorities:
        failures.append("roadmap execution priorities are not reflected in the handoff")
    if failures:
        raise ValueError("Stale session handoff: " + "; ".join(failures))
    return {
        "version": HANDOFF_VERSION,
        "handoff": str(path),
        "status": "CURRENT",
        "active_model_id": None if active is None else active["model_id"],
        "published_model_id": None if publication is None else publication["model_id"],
        "priorities": len(priorities),
    }


def render_handoff(
    repo_root: Path,
    artifacts_root: Path,
    state: RepositoryState,
    *,
    generated_at: datetime,
) -> tuple[str, dict[str, Any]]:
    """Render the handoff and return machine-readable headline facts."""

    model_text, active = _model_markdown(artifacts_root)
    publication = _tracked_publication(repo_root / "CURRENT_PREDICTIONS.md")
    priorities = _roadmap_priorities(repo_root / "ROADMAP.md")
    inventory = _local_inventory(repo_root, artifacts_root)
    publication_text = (
        "No valid tracked weekly publication was found."
        if publication is None
        else (
            f"[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains "
            f"**{publication['season']} Week {publication['week']}** from model "
            f"`{publication['model_id']}`, published `{publication['published_at_utc']}`. "
            "It is an early, mutable research preview."
        )
    )
    if (
        active is not None
        and publication is not None
        and active["model_id"] != publication["model_id"]
    ):
        publication_text += (
            " **Warning:** the tracked publication does not match the local active model; "
            "run `nfl-ats publish-predictions` before publishing model claims."
        )
    priorities_text = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(priorities, start=1))
        if priorities
        else "See [ROADMAP.md](ROADMAP.md); no ordered priority list could be parsed."
    )
    inventory_text = "\n".join(
        f"- {label}: **{'present' if path.is_file() else 'missing'}** "
        f"(`{_display_path(path, repo_root)}`)"
        for label, path in inventory
    )
    timestamp = generated_at.astimezone(UTC).isoformat()
    markdown = f"""# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `{HANDOFF_VERSION}`

Refreshed at: `{timestamp}`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\\.tools\\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `{state.branch}`
- Baseline commit: `{state.commit}` — {state.subject}
- Pending change set: {_changes_markdown(state)}

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

{model_text}

The 52.05% figure is historical forced-pick ATS classification accuracy, not a
game-specific probability and not proof of a profitable or stable market edge.

## Last tracked weekly publication

{publication_text}

## Local reproducibility inventory

{inventory_text}

Raw data, processed features, fitted models, and evaluation artifacts are intentionally
ignored by Git. A fresh clone therefore starts with documentation, source, tests, and
the last published Markdown forecast but must rebuild or transfer local artifacts.

## Highest-priority work

{priorities_text}

The roadmap is authoritative. Negative results remain part of the evidence base and
must not be silently removed or retuned away.

## Commands that matter

```powershell
# Manual diagnostic/recovery only; the agent and Git hooks own normal refreshes
.\\.tools\\uv.exe run nfl-ats handoff --check

# Launch the local dashboard
.\\.tools\\uv.exe run nfl-ats dashboard

# Quality gates
.\\.tools\\uv.exe run ruff format --check .
.\\.tools\\uv.exe run ruff check .
.\\.tools\\uv.exe run mypy src
.\\.tools\\uv.exe run pytest
```

## Automatic end-of-session contract

1. Reconcile completed work and new evidence with `ROADMAP.md` and relevant docs.
2. If the synchronized weekly forecast changed, run `nfl-ats publish-predictions`.
3. Run all quality gates and record the result in the final response.
4. The agent refreshes the handoff automatically before a handoff, commit, or push
   to `master`; it must never delegate this command to the user.
5. Check `git status`; never commit ignored data, credentials, or fitted models.
6. Commit or push only when the user explicitly asks, and report the exact branch/hash.
"""
    result: dict[str, Any] = {
        "version": HANDOFF_VERSION,
        "destination": "HANDOFF.md",
        "refreshed_at_utc": timestamp,
        "branch": state.branch,
        "baseline_commit": state.commit,
        "worktree_clean_before_refresh": state.clean,
        "worktree_changes_before_refresh": len(state.changes),
        "active_model_id": None if active is None else active["model_id"],
        "published_model_id": None if publication is None else publication["model_id"],
        "priorities": len(priorities),
    }
    return markdown, result


def write_session_handoff(
    repo_root: Path,
    artifacts_root: Path,
    destination: Path,
    *,
    generated_at: datetime | None = None,
    state: RepositoryState | None = None,
) -> dict[str, Any]:
    """Inspect current state and atomically refresh the tracked handoff."""

    root = repo_root.resolve()
    git_state = state or inspect_repository(root)
    markdown, result = render_handoff(
        root,
        artifacts_root.resolve(),
        git_state,
        generated_at=generated_at or datetime.now(UTC),
    )
    output = destination if destination.is_absolute() else root / destination
    atomic_text(markdown, output)
    result["destination"] = str(output)
    return result

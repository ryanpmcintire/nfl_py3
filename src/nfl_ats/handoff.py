"""Generate a tracked, human-readable session handoff from authoritative state."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.io import atomic_text
from nfl_ats.player_arrests_back_side_overlay import (
    POLICY_BASELINE_OPENER_ACCURACY,
    POLICY_EFFECT_ACCURACY_POINTS,
    POLICY_GRADED_GAMES,
    POLICY_OPENER_ACCURACY,
    POLICY_PROBABILITY_POSITIVE,
)

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
    def latest_file(root: Path, name: str) -> Path:
        runs = (
            sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
            if root.is_dir()
            else []
        )
        return runs[0] / name if runs else root / f"LATEST_{name.upper()}_MISSING"

    participation_source_root = repo_root / "data/players/participation/raw"
    participation_sources = (
        sorted(
            (path for path in participation_source_root.iterdir() if path.is_dir()),
            reverse=True,
        )
        if participation_source_root.is_dir()
        else []
    )
    latest_participation_source = (
        participation_sources[0] / "manifest.json"
        if participation_sources
        else participation_source_root / "LATEST_MANIFEST_MISSING.json"
    )
    return [
        ("canonical team features", repo_root / "data/processed/game_features.parquet"),
        ("play-by-play features", repo_root / "data/processed/game_features_pbp.parquet"),
        ("player features", repo_root / "data/processed/game_features_player.parquet"),
        (
            "player-value research features",
            repo_root / "data/processed/game_features_player_value.parquet",
        ),
        ("participation source snapshot", latest_participation_source),
        (
            "participation-rating research features",
            repo_root / "data/processed/game_features_player_participation.parquet",
        ),
        (
            "learned-availability research features",
            repo_root / "data/processed/game_features_player_learned_availability.parquet",
        ),
        (
            "frozen player-model selection",
            latest_file(artifacts_root / "player_model_selection", "metadata.json"),
        ),
        (
            "participation-rating experiment",
            latest_file(artifacts_root / "participation_experiments", "metadata.json"),
        ),
        (
            "learned-availability experiment",
            latest_file(artifacts_root / "availability_experiments", "metadata.json"),
        ),
        ("active model manifest", artifacts_root / "active_ats_model.json"),
    ]


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _matching_opener_evaluation(
    artifacts_root: Path, active: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest opener evaluation matching the active model recipe."""

    root = artifacts_root / "opener_evaluation"
    runs = (
        sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
        if root.is_dir()
        else []
    )
    for run in runs:
        metadata_path = run / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        config = metadata.get("active_model_config", {})
        expected = {
            "feature_profile": active.get("feature_profile"),
            "regressor": active.get("regressor"),
            "ridge_alpha": active.get("ridge_alpha", 10.0),
            "target": active.get("method"),
        }
        if config != expected:
            continue
        metrics = metadata.get("metrics", {})
        if not isinstance(metrics.get("opener_accuracy_probability_rule"), (int, float)):
            continue
        if not isinstance(metadata.get("games"), int):
            continue
        return run, metadata
    return None


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
    opener = _matching_opener_evaluation(artifacts_root, active)
    opener_text = (
        "- Raw-model baseline (opener-graded probability rule): **unavailable in local artifacts**"
        if opener is None
        else (
            "- Raw-model baseline (opener-graded probability rule): "
            f"**{opener[1]['metrics']['opener_accuracy_probability_rule']:.2%}** on "
            f"**{opener[1]['games']:,} games** "
            f"(`opener_evaluation/{opener[0].name}`)"
        )
    )
    production_policy_text = (
        "- Promoted player-arrest policy component (opener-graded): "
        f"**{POLICY_OPENER_ACCURACY:.2%}** versus "
        f"**{POLICY_BASELINE_OPENER_ACCURACY:.2%}** on "
        f"**{POLICY_GRADED_GAMES:,} games** "
        f"(+{POLICY_EFFECT_ACCURACY_POINTS:.3f} accuracy points; "
        f"`probability_positive={POLICY_PROBABILITY_POSITIVE:.4f}`); the live card "
        "applies this after the coach policy, while paired prospective tracking continues"
    )
    text = (
        f"- Status: **{active['status']}**; linked artifacts present: **{str(linked).lower()}**\n"
        f"- Model ID: `{active['model_id']}`\n"
        f"- Method/profile/regressor/alpha/calibration: `{active['method']}` / "
        f"`{active['feature_profile']}` / `{active['regressor']}` / "
        f"`{active.get('ridge_alpha', 10.0)}` / "
        f"`{active.get('calibration_method', 'none')}`\n"
        f"{opener_text}\n"
        f"{production_policy_text}\n"
        f"- Secondary close-grade historical classification: **{historical['correct']:,} / "
        f"{historical['games']:,} ({historical['accuracy']:.2%})**\n"
        f"- Linked forecast: **{weekly['season']} Week {weekly['week']}**, created "
        f"`{weekly['created_at_utc']}`"
    )
    return text, active


def _accuracy_disclaimer(active: dict[str, Any] | None) -> str:
    """Derive the historical-accuracy disclaimer from the active model, not a literal.

    A hardcoded figure here drifts from reality the moment the active model
    changes (it once read "52.05%" while the model evidence above it reported
    51.57%). The number is now read from `historical_evaluation` every render;
    the disclaimer's WARNING never changes (AGENTS.md forbids describing this
    accuracy as proof of a profitable or stable market edge).
    """

    if active is None:
        return (
            "Historical forced-pick ATS classification accuracy (see "
            "`artifacts/active_ats_model.json` once local artifacts exist) is not a "
            "game-specific probability and not proof of a profitable or stable market edge."
        )
    accuracy = active["historical_evaluation"]["accuracy"]
    return (
        f"The {accuracy:.2%} figure is the distinct secondary close-grade historical "
        "classification, not the raw-model opener baseline, the promoted player-arrest "
        "policy evaluation, a game-specific "
        "probability, or proof of a profitable or stable market edge."
    )


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
        opener = _matching_opener_evaluation(artifacts_root, active)
        if opener is not None:
            opener_accuracy = opener[1]["metrics"]["opener_accuracy_probability_rule"]
            if f"**{opener_accuracy:.2%}**" not in text:
                failures.append("opener-grade raw-model baseline is not reflected in the handoff")

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
    accuracy_disclaimer = _accuracy_disclaimer(active)
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

{accuracy_disclaimer}

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

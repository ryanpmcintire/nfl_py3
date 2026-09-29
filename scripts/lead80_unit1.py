from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_NAME = re.compile(
    r"pool|splash|tiebreak|tie_break|opponent|contest|entrant|guess|field|card|entr(?:y|ies)",
    re.IGNORECASE,
)
SEASONS = tuple(range(2020, 2026))


def relative_name(path: Path) -> str:
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def inventory(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": relative_name(root),
        "exists": root.is_dir(),
        "files": 0,
        "candidates": [],
        "errors": [],
        "skipped_links": 0,
    }
    if not result["exists"]:
        return result
    for directory, folders, filenames in os.walk(
        root, onerror=lambda error: result["errors"].append(str(error)), followlinks=False
    ):
        base = Path(directory)
        kept = []
        for folder in folders:
            child = base / folder
            if child.is_symlink() or child.is_junction():
                result["skipped_links"] += 1
            else:
                kept.append(folder)
        folders[:] = sorted(kept)
        for filename in sorted(filenames):
            path = base / filename
            result["files"] += 1
            if SOURCE_NAME.search(path.relative_to(root).as_posix()):
                result["candidates"].append(path)
    return result


def schema_fields(path: Path, name: str) -> tuple[int, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    fields = [
        item.target.id
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    return node.lineno, fields


def observation_metadata(paths: list[Path]) -> tuple[Counter[int], Counter[str], list[str]]:
    seasons: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    errors = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            season, kind = payload["season"], payload["kind"]
            if type(season) is not int or not isinstance(kind, str):
                raise ValueError("invalid season/kind metadata")
            seasons[season] += 1
            kinds[kind] += 1
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{relative_name(path)}: {error}")
    return seasons, kinds, errors


def build_report(roots: list[Path]) -> tuple[str, dict[str, Any]]:
    inventories = [inventory(root) for root in roots]
    candidates = sorted({path for item in inventories for path in item["candidates"]})
    observations = [
        path
        for path in candidates
        if "pool_observables" in path.parts and path.name == "observations.json"
    ]
    seasons, kinds, metadata_errors = observation_metadata(observations)
    errors = [error for item in inventories for error in item["errors"]] + metadata_errors
    groups: dict[str, list[Path]] = {}
    for path in candidates:
        groups.setdefault(path.name, []).append(path)
    lines = [
        "# LEAD-80 unit 1: local source inventory",
        "",
        "**Measured:** inventory only; no outcome scoring, model fit, field "
        "simulation or registry write.",
        "Declaration: `docs/lanes/lead80.md`; complete source: `docs/lead80_protocol.md`.",
        "",
        "## Inventory boundary",
        "",
        "Inspected file names under configured data/artifact roots, including ignored files.",
        "Candidate names match pool, Splash, tiebreak, opponent, contest, "
        "field, card, entrant, entry or guess terms.",
        "Only registered pool-observation season/kind metadata and Python "
        "schema declarations were read.",
        "Generic files, opaque archives and external/browser-held sources are not authenticated.",
        "Name matching establishes candidates, not authenticity, historical "
        "coverage or absence elsewhere.",
        "",
        "| Root | Exists | File names inspected | Candidate files | Links skipped | Read errors |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in inventories:
        lines.append(
            f"| `{item['root']}` | {item['exists']} | {item['files']} | "
            f"{len(item['candidates'])} | {item['skipped_links']} | {len(item['errors'])} |"
        )
    lines.extend(
        [
            "",
            f"**Measured:** {len(observations)} registered pool-observation files; "
            f"{sum(seasons[season] for season in SEASONS)} dated to 2020-2025; "
            f"{len(metadata_errors)} metadata errors.",
            f"Observation kinds: {dict(sorted(kinds.items())) or 'none'}.",
            "",
            "| Selected candidate basename | Files | Example path (full counts above) |",
            "|---|---:|---|",
        ]
    )
    for name, paths in sorted(groups.items()):
        if not any("splash" in path.parts for path in paths) and name not in (
            "tiebreaker.json",
            "wp12_pool_rules.md",
        ):
            continue
        lines.append(f"| `{name}` | {len(paths)} | `{relative_name(paths[0])}` |")
    if not groups:
        lines.append("| None | 0 | No candidate names under the inspected roots |")
    lines.extend(["", "## Source gates", ""])
    for name in ("FieldObservation", "DistributionObservation"):
        path = Path("src/nfl_ats/pool_observables.py")
        line, fields = schema_fields(path, name)
        lines.extend(
            [
                f"**Read:** `{path.as_posix()}:{line}` `{name}` exposes:",
                ", ".join(f"`{field}`" for field in fields) + ".",
                "",
            ]
        )
    lines.extend(
        [
            "**Read:** these schemas supply field size/prizes and aggregate per-game side shares.",
            "They omit entrant-level weekly cards, Best Pick nominations and "
            "submitted tiebreak guesses.",
            "`docs/pool_observables.md:31` identifies the registered snapshot location.",
            "",
            "**Read:** `docs/tiebreaker.md:3` says final score of the week's last game.",
            "`src/nfl_ats/pool_workbench.py:37` defaults to 1/0 "
            "correct/incorrect points, 0.5 push points,",
            "and a +1/0 Best Pick bonus/penalty; line 52 names `final_score_last_game`.",
            "`docs/pool_rules.md:13` delegates pick locking to min(kickoff, Sunday 16:00 ET).",
            "These current descriptions are not authenticated season-specific 2020-2025 rules.",
            "Closest-total versus exact-score loss, residual tie handling and "
            "guess deadline are unverified.",
            "",
            "**Read:** `docs/prospective_bestpick_tiebreaker.md:26` describes "
            "the project's own 2026 guesses.",
            "Those forecasts cannot substitute for historical opponent "
            "submissions or field outcomes.",
            "",
            "**Inferred:** the replay source gate is not cleared. No "
            "conditional field fit is authorized.",
            "Authentic opener/total issuance and ingestion against the deadline remain unchecked",
            "because the rules/field prerequisites are unmet.",
            "",
            "## Prespecified results availability",
            "",
            "**Measured:** evaluated decisive games = 0; record unavailable (no games scored).",
            "IS/OOS accuracy, Brier, log loss, margin MAE, joint-score log "
            "score, actual tiebreak utility,",
            "IS-minus-OOS gaps, season-block 95% intervals and "
            "`probability_positive` are not estimable.",
            "Fold coefficients, coefficient stability and reliability bands are "
            "unavailable; no fit ran.",
            "",
            "| Held-out season | Registered observation files | Fold "
            "coefficients / IS / OOS / gap |",
            "|---|---:|---|",
        ]
    )
    for season in SEASONS:
        lines.append(f"| {season} | {seasons[season]} | Not fitted: source gate |")
    lines.extend(
        [
            "",
            "**Read:** B=3, F=6, E=2; declared looks = (3+36+16)*7+25+35 = 445.",
            "**Measured:** executed outcome looks = 0; inventory counts are not "
            "performance estimates.",
            "**Inferred:** missing-source status leaves the mechanism open; no "
            "negative or terminal verdict.",
            "No effect estimate exists to register. The orchestrator has no "
            "record command from this unit.",
            "",
            "## Next source unit",
            "",
            "Obtain season-specific official rules and entrant cards/Best "
            "Picks/guesses for 2020-2025,",
            "with field size, contest/week IDs, tiebreak game, effective dates "
            "and submission deadlines.",
            "Keep held-out opponent submissions as outcomes. Authenticate "
            "frozen opener/total captures",
            "and pre-deadline issuance/ingestion before chronological "
            "conditional/unconditional replay.",
        ]
    )
    if errors:
        lines.extend(["", "## Inventory errors", ""] + [f"- {error}" for error in errors])
    summary = {
        "status": "inventory_only_source_gate_not_cleared",
        "files_inspected": sum(item["files"] for item in inventories),
        "candidate_files": len(candidates),
        "registered_observations": len(observations),
        "observations_2020_2025": sum(seasons[season] for season in SEASONS),
        "errors": len(errors),
        "declared_looks": 445,
        "executed_outcome_looks": 0,
        "folds_fitted": 0,
        "probability_positive": None,
    }
    return "\n".join(lines) + "\n", summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory LEAD-80 sources without fitting or scoring."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts")),
    )
    args = parser.parse_args()
    report, summary = build_report([args.data_root, args.artifacts_root])
    path = Path("docs/lead80_inventory.md")
    path.write_text(report, encoding="utf-8")
    summary["report"] = path.as_posix()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

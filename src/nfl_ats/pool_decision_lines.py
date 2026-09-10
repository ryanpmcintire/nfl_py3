from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nfl_ats.features import (
    DECISION_LINE_VERSION,
    AppliedDecisionLines,
    DecisionLineOverride,
)
from nfl_ats.splash_lines import (
    SPLASH_SOURCE,
    SPLASH_SUBDIRECTORY,
    load_splash_capture,
    splash_decision_lines,
)

DECISION_LINE_POLICY = "pool_capture"

BUILDER_MODULE = "nfl_ats.pool_decision_lines"

_CAPTURE_FILENAME = re.compile(r"^(?P<season>\d{4})_week(?P<week>\d{2})_.+\.json$")


def captured_weeks(data_root: Path | str) -> tuple[tuple[int, int], ...]:

    directory = Path(data_root) / SPLASH_SUBDIRECTORY
    if not directory.is_dir():
        return ()
    found: set[tuple[int, int]] = set()
    for path in directory.glob("*.json"):
        match = _CAPTURE_FILENAME.match(path.name)
        if match is None:
            continue
        found.add((int(match.group("season")), int(match.group("week"))))
    return tuple(sorted(found))


def splash_decision_line_overrides(data_root: Path | str) -> tuple[DecisionLineOverride, ...]:

    root = Path(data_root)
    overrides: list[DecisionLineOverride] = []
    for season, week in captured_weeks(root):
        capture = load_splash_capture(root, season, week)
        if capture is None:  # pragma: no cover - captured_weeks just saw the file
            continue
        overrides.append(
            DecisionLineOverride(
                season=capture.season,
                week=capture.week,
                lines=splash_decision_lines(capture),
                source=capture.source or SPLASH_SOURCE,
                capture_id=(
                    capture.path.stem if capture.path is not None else f"{season}_week{week:02d}"
                ),
                captured_at_utc=capture.captured_at_et.isoformat(),
                captured_at=capture.captured_at_et,
            )
        )
    return tuple(overrides)


def decision_lines_manifest_block(applied: Sequence[AppliedDecisionLines]) -> dict[str, Any]:

    return {
        "policy": DECISION_LINE_POLICY,
        "builder_module": BUILDER_MODULE,
        "builder_version": DECISION_LINE_VERSION,
        "weeks": [
            {
                "season": entry.override.season,
                "week": entry.override.week,
                "source": entry.override.source,
                "capture_id": entry.override.capture_id,
                "captured_at_utc": entry.override.captured_at_utc,
                "games": len(entry.game_ids),
                "changed_games": len(entry.changed_game_ids),
                "changed_game_ids": list(entry.changed_game_ids),
            }
            for entry in applied
        ],
    }


__all__ = [
    "BUILDER_MODULE",
    "DECISION_LINE_POLICY",
    "captured_weeks",
    "decision_lines_manifest_block",
    "splash_decision_line_overrides",
]

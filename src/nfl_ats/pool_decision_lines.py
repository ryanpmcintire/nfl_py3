"""Turn captured pool boards into the feature build's decision line.

``nfl_ats.splash_lines`` reads and validates a Splash Sports contest board;
``nfl_ats.features.apply_decision_lines`` applies a week's lines to the
schedules frame and refuses everything that would grade a game on a number
nobody saw.  This module is the thin adapter between them: it finds the
captures on disk, converts each into a
:class:`~nfl_ats.features.DecisionLineOverride`, and turns the result of an
application into the provenance block the feature-table manifest carries.

It exists as its own module so ``nfl_ats.features`` -- the foundational
one-row-per-game builder that almost everything imports -- does not have to
pull in the capture layer's own import graph.  Same reasoning as the
"ported, not imported" note on ``features.add_surface_switch_features``.

The card side reads this block back through
``nfl_ats.feature_manifest.decision_line_week``; nothing here is imported by
``nfl_ats.lineage``, so the dependency only ever points one way.
"""

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
    """Every ``(season, week)`` with at least one capture file, ascending.

    Filename-driven on purpose: it answers "which weeks does the pool board
    cover" without opening (and therefore without validating) any file, so the
    caller decides which weeks it actually needs before paying for validation.
    """

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
    """One override per captured week, newest capture per week, fully validated.

    Absence is not an error: a repository with no ``data/splash/`` directory
    (every clone before 2026-09-08, and every fresh clone today) yields an
    empty tuple, and a build handed an empty tuple is bit-identical to one
    that never heard of this module.  A capture that IS present but malformed
    raises :class:`~nfl_ats.data.DataContractError` out of
    :func:`~nfl_ats.splash_lines.load_splash_capture`; a board that cannot be
    trusted is a defect to fix, not a file to skip past.
    """

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
    """The provenance block a feature-table manifest records for these weeks.

    Shape (read back by ``nfl_ats.feature_manifest.decision_line_week`` and
    surfaced on the card's ``market_line`` lineage record)::

        {"policy": "pool_capture",
         "builder_module": "nfl_ats.pool_decision_lines",
         "builder_version": "v2",
         "weeks": [{"season": 2026, "week": 1, "source": "splashsports.com",
                    "capture_id": "2026_week01_20260908_noon",
                    "captured_at_utc": "2026-09-08T12:45:00-04:00",
                    "games": 16, "changed_games": 8,
                    "changed_game_ids": [...]}]}

    ``changed_games`` is the count that moved off the nflverse close -- the
    honest measure of how much the proxy was costing, kept per build rather
    than quoted from a doc.
    """

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

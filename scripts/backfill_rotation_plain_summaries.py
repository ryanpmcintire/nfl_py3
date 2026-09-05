"""Backfill recent rotation summaries through the public CLI, with an exact diff audit.

Matching weak-signal summaries are reused when they meet the reader-text contract.
The reviewed alternatives below translate technical mirrors or describe unmirrored
families. No results, classifications, or research accounting are changed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nfl_ats import cli
from nfl_ats.findings_registry import (
    STORE_ROTATION,
    load_rotation_registry,
    load_weak_signal_registry,
    recent_registry_activity,
)
from nfl_ats.rotation import default_registry_path

# Reviewed situation-and-action descriptions, used only when a mirrored summary
# is absent or needs translation. Keep new families explicit: never manufacture
# reader prose from a registry identifier.
ROTATION_PLAIN_SUMMARIES = {
    "ats_streak_regress_on_production": (
        "After a team fails to cover the spread in at least three straight games, this rule "
        "backs it to bounce back."
    ),
    "backup_tenure_gap_on_production": (
        "When a backup quarterback starts, this rule backs teams whose backup has been with "
        "them for at least two seasons and goes against teams using a newly arrived backup."
    ),
    "crew_second_meeting_favorite_on_production": (
        "When an officiating crew has already worked a game involving either team this "
        "season, this rule backs the favorite."
    ),
    "deadline_integration_drag_on_production": (
        "After a team trades for a heavily used player, this rule goes against that team "
        "during his first three games as he settles in."
    ),
    "division_dog_on_production": ("When two division rivals meet, this rule backs the underdog."),
    "dome_shootout_favorite_on_production": (
        "In a dome game with a high expected score and a small point spread, this rule backs "
        "the favorite."
    ),
    "fluview_elevated_on_production": (
        "When flu activity rises near either team's home city, this approach uses that "
        "illness information to adjust the picks."
    ),
    "fluview_home_elevated_opener": (
        "When flu activity rises near the home team's city, this approach adjusts the pick "
        "using that illness information and checks it against the opening spread."
    ),
    "fourth_down_aggression_interaction_on_production": (
        "When an underdog often goes for it on fourth down, this rule backs that aggressive team."
    ),
    "graph_def_ypp_on_production": (
        "This approach adjusts picks using how many yards a defense allows per play, "
        "accounting for the strength of the opponents it faced."
    ),
    "graph_off_rush_epa_on_production": (
        "This approach adjusts picks using how much a team's running game helps its offense, "
        "accounting for the strength of the opponents it faced."
    ),
    "graph_off_sack_rate_on_production": (
        "This approach adjusts picks using how often an offense gives up sacks, accounting "
        "for the strength of the opponents it faced."
    ),
    "holdout_slow_start_on_production": (
        "After a starter ends a training-camp holdout, this rule goes against his team during "
        "the first month of games."
    ),
    "home_thursday_on_production": ("In Thursday games, this rule backs the home team."),
    "illness_away_active_ge1_on_production_opener": (
        "When at least one visiting player is available to play despite an illness, this "
        "approach uses that information to adjust the pick against the opening spread."
    ),
    "illness_on_production": (
        "When visiting players are playing through illness or multiple home players are "
        "listed as ill, this approach uses those reports to adjust the picks."
    ),
    "inactives_channel_historical_proxy_v1": (
        "This approach uses past player-availability patterns to anticipate surprise absences "
        "before kickoff and adjust the Tuesday picks."
    ),
    "ir_return_bump_on_production": (
        "When a starter returns from a long injury absence in Weeks 5 through 8, this rule "
        "backs his team in case the spread undervalues his return."
    ),
    "kicker_change_underdog_on_production": (
        "When either team has just changed its placekicker, this rule backs the underdog."
    ),
    "low_total_div_home_dog_on_production": (
        "In a division game with a low expected score, this rule backs the home underdog."
    ),
    "missingness_availability_flags": (
        "When historical lineup information is unavailable, this approach gives the "
        "prediction model one shared missing-information marker instead of treating each "
        "missing statistic separately."
    ),
    "ml_spread_divergence_on_production": (
        "When the odds of winning outright and the point spread disagree about the home "
        "team's strength, this rule follows the outright-win odds."
    ),
    "mnf_road_short_week_on_production": (
        "When a team follows a Monday road game with a short turnaround, this rule goes "
        "against that team."
    ),
    "movement_expansion_v1": (
        "When the spread moves by at least two points, this approach follows the move and "
        "compares acting on Thursday with acting on Saturday."
    ),
    "movement_leads_v1": (
        "This approach compares following spread changes at different points during the week. "
        "It also tries backing the underdog when the expected total score rises but the "
        "spread stays steady."
    ),
    "new_stadium_home_on_production": (
        "During a new stadium's first two seasons, this rule backs the home team."
    ),
    "open_corner_wind_dog_on_production": (
        "In strong winds at an open, exposed stadium, this rule backs the underdog."
    ),
    "opener_softness_fade_on_production": (
        "When one sportsbook's opening line disagrees with the other books about the "
        "favorite, this rule sides with the other books."
    ),
    "opening_drive_script_on_production": (
        "This rule favors the team with a stronger recent record on its opening offensive drive."
    ),
    "per13_durability_on_production": (
        "When accounting for injuries, this approach uses players' past durability to "
        "estimate whether they will play and adjusts picks using those revised availability "
        "estimates."
    ),
    "per13_durability_on_production_opener": (
        "When accounting for injuries, this approach uses players' past durability to "
        "estimate whether they will play. It checks the resulting picks against the opening "
        "spread."
    ),
    "post_ot_fatigue_on_production": (
        "After a team plays an overtime game, this rule goes against it in its next game."
    ),
    "q3_adjustment_on_production": (
        "This rule favors the team with a stronger recent record in the third quarter, "
        "looking for an advantage from halftime adjustments."
    ),
    "qb_revenge_on_production": (
        "When a quarterback faces the team that drafted him, this rule backs his current team."
    ),
    "rain_on_grass_dog_on_production": (
        "When rain is likely on a grass field, this rule backs the underdog."
    ),
    "reddit_attention_on_production": (
        "When fan-forum activity is unusually high for either team, this approach uses that "
        "attention to adjust the picks."
    ),
    "reddit_home_comment_ratio_elevated_on_production_opener": (
        "When the home team's fan forum draws an unusually large share of comments, this "
        "approach uses that attention to adjust the pick against the opening spread."
    ),
    "redzone_reversion_on_production": (
        "After a team converts unusually many third downs, this approach lowers expectations "
        "that it will keep up that success and adjusts the picks."
    ),
    "redzone_third_down_over_fade_on_production_opener": (
        "After a team converts unusually many third downs, this rule goes against it in picks "
        "made against the opening spread."
    ),
    "road_fav_big_fade_on_production": (
        "When a team is favored by at least seven points, this rule backs its opponent."
    ),
    "rookie_crew_underdog_on_production": (
        "When the assigned referee crew is in its first or second year, this rule backs the "
        "underdog."
    ),
    "rookie_qb_debut_fade_on_production": (
        "When a rookie quarterback makes his first career start, this rule backs the opposing team."
    ),
    "rookie_wall_dependence_on_production": (
        "Late in the season, this rule goes against teams relying heavily on rookies drafted "
        "early, looking for signs that a long season wears them down."
    ),
    "sept_heat_home_on_production": (
        "In September heat, this rule backs a home team used to hot weather against a visitor "
        "from a colder climate."
    ),
    "specialist_absence_fade_on_production": (
        "When a team is missing its long snapper or punter, this rule backs its opponent."
    ),
    "suspension_return_rust_on_production": (
        "After a player returns from a suspension of at least six games, this rule goes "
        "against his team for his first two games back."
    ),
    "team_style_pace_mismatch_on_production_opener": (
        "When one team usually plays much faster than the other, this approach uses that pace "
        "mismatch to adjust the pick against the opening spread."
    ),
    "team_style_pace_on_production": (
        "When one team usually plays much faster than the other, this approach uses that pace "
        "mismatch to adjust the picks."
    ),
    "week1_dog_on_production": ("In Week 1, this rule backs the underdog."),
    "xlg05_transfer_prior": (
        "This approach gives NFL predictions a starting point learned from college-football "
        "patterns, then adjusts those patterns using NFL results."
    ),
}

# Mirrors containing technical result prose need the reviewed translation above.
TECHNICAL_TEXT = re.compile(
    r"_|P\+|\b(?:window\w*|production|sample|resamples|interval|range|"
    r"graph-adjusted|schedule-adjusted|signal|flag|feature|predeclared|"
    r"fade|opener|two-season|archive|accuracy|probability|allowed|"
    r"unresolved|recorded|model|third-down over-performance)\b|\b20\d{2}\b",
    re.IGNORECASE,
)


def reader_ready(summary: str) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    return (
        bool(summary.strip())
        and summary.rstrip()[-1] in ".!?"
        and 1 <= len(sentences) <= 2
        and not TECHNICAL_TEXT.search(summary)
    )


def summary_plan(registry_dir: Path, as_of: date, days: int = 7) -> dict[str, tuple[str, str]]:
    rotation = load_rotation_registry(registry_dir)
    weak = load_weak_signal_registry(registry_dir)
    activity = recent_registry_activity(weak, rotation, as_of, days=days)
    names = sorted(
        {
            entry.key.split(":", 1)[1]
            for _, entries in activity.entries_by_category
            for entry in entries
            if entry.store == STORE_ROTATION
        }
    )
    plan = {}
    for name in names:
        family = rotation.families[name]
        mirror = weak.signals.get(family.coverage_weak_signal_family or name)
        summary = mirror.plain_summary if mirror else None
        if summary and reader_ready(summary):
            plan[name] = (summary, "mirrored weak-signal summary")
        elif name in ROTATION_PLAIN_SUMMARIES:
            plan[name] = (ROTATION_PLAIN_SUMMARIES[name], "reviewed translation")
        elif family.plain_summary and reader_ready(family.plain_summary):
            plan[name] = (family.plain_summary, "existing rotation summary")
        else:
            raise ValueError(f"Write a reviewed summary for {name!r} before applying this plan")
    return plan


def without_summaries(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    for family in result["families"].values():
        family.pop("plain_summary", None)
    return result


def without_summary_bytes(raw: bytes) -> bytes:
    # The standard serializer writes each summary as a single JSON string line.
    # Remove that line and its adjacent separator to compare every other byte.
    string = rb'"plain_summary": "(?:[^"\\]|\\.)*"'
    raw = re.sub(rb"[ \t]*" + string + rb",\r?\n", b"", raw)
    return re.sub(rb",\r?\n[ \t]*" + string, b"", raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--missing-plain-summary", action="store_true")
    parser.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    path = default_registry_path()
    plan = summary_plan(path.parent, args.as_of, args.days)
    before_bytes = path.read_bytes()
    before = json.loads(before_bytes)
    changes = {
        name: {
            "before": before["families"][name].get("plain_summary"),
            "after": summary,
            "source": source,
        }
        for name, (summary, source) in plan.items()
        if before["families"][name].get("plain_summary") != summary
    }
    if args.missing_plain_summary:
        print(
            json.dumps(
                {
                    "target_count": len(plan),
                    "missing_plain_summary": [
                        name for name in plan if not before["families"][name].get("plain_summary")
                    ],
                },
                indent=2,
            )
        )
        return
    if not args.dry_run:
        for name, change in changes.items():
            with contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(
                    [
                        "rotation",
                        "set-plain-summary",
                        "--name",
                        name,
                        "--plain-summary",
                        change["after"],
                    ]
                )
            if result != 0:
                raise RuntimeError(f"CLI failed for {name}: {result}")
        after_bytes = path.read_bytes()
        after = json.loads(after_bytes)
        if without_summaries(before) != without_summaries(after):
            raise RuntimeError("Non-summary JSON fields changed during the backfill")
        if without_summary_bytes(before_bytes) != without_summary_bytes(after_bytes):
            raise RuntimeError("Non-summary bytes changed during the backfill")
        for name, (summary, _) in plan.items():
            if after["families"][name].get("plain_summary") != summary:
                raise RuntimeError(f"Summary not saved for {name}")
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "target_count": len(plan),
                "changed_count": len(changes),
                "changes": changes,
                "non_summary_fields_byte_identical": None if args.dry_run else True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

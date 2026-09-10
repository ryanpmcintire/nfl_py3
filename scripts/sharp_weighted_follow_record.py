"""Emit the weak-signals record argv lists for the sharp-weighted-follow lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nfl_ats.provenance import write_stamped_artifact

DATA_REPO = Path("F:/Repos/nfl_py3")
OUTPUT = DATA_REPO / "artifacts/sharp_weighted_follow"
FROZEN = "20260909T233606Z"
CURRENT = "20260909T233611Z"
FAMILY = "sharp_weighted_follow"
DOC = "docs/sharp_weighted_follow.md"

GROUND = (
    "unresolved_below_power: the interval crossing zero is not a closing ground (AGENTS.md); "
    "the fire-flag split-half reliability is non-zero, so no_split_half_reliability is "
    "inadmissible; no interval sits wholly on the wrong side of zero, so wrong_sign_resolved is "
    "inadmissible; and the perfect-foresight positive control resolves a ~45-47 point effect, "
    "which does not prove the instrument can detect a 0.1-3 point one, so positive_control_bound "
    "is inadmissible too."
)


def cells(stamp: str) -> dict[str, Any]:
    return json.loads((OUTPUT / stamp / "metadata.json").read_text(encoding="utf-8"))


def row(
    *,
    name: str,
    description: str,
    plain: str,
    stamp: str,
    block: dict[str, Any],
    reliability: float | None,
    notes: str,
) -> list[str]:
    argv = [
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        description,
        "--source",
        f"artifacts/sharp_weighted_follow/{stamp}/metadata.json ({DOC})",
        "--effect",
        repr(block["effect"]),
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2023",
        "--season-end",
        "2025",
        "--interval-low",
        repr(block["interval_low"]),
        "--interval-high",
        repr(block["interval_high"]),
        "--probability-positive",
        repr(block["probability_positive"]),
        "--sample-games",
        str(block["games"]),
        "--sample-blocks",
        str(block["blocks"]),
        "--family",
        FAMILY,
        "--classification-evidence",
        GROUND,
        "--plain-summary",
        plain,
        "--category",
        "market",
        "--notes",
        notes,
        "--replace",
    ]
    if reliability is not None:
        argv.extend(["--reliability", repr(reliability)])
    return argv


def main() -> None:
    frozen = cells(FROZEN)
    current = cells(CURRENT)
    commands: list[list[str]] = []

    arm_text = {
        "s1_leader_only": (
            "leader-only follow: median Wednesday-to-deadline net spread move across Bovada, "
            "William Hill (US) and MyBookie, followed at 0.5 points against the Tuesday pick"
        ),
        "s2_leader_weighted": (
            "leader-weighted follow: the twelve-book Wednesday-to-deadline net move with the "
            "three leading books weighted 2 and the other nine weighted 1, followed at 0.5 points"
        ),
        "s3_leader_first": (
            "leader-first follow: follow the three leading books when they have moved 0.5 or "
            "more, otherwise follow the equal-book move at 0.5"
        ),
        "s4_served_equal": (
            "served rule replay: equal-book mean Wednesday-to-deadline net move followed at 0.5 "
            "points, the rule production has served since 2026-09-06"
        ),
        "pc_oracle": (
            "positive control: perfect-foresight switch on every game the archive can reach, "
            "bounding what any follow rule on this archive could buy"
        ),
    }
    plain_text = {
        "s1_leader_only": (
            "Three sportsbooks move their number first and the rest copy them. This rule watches "
            "only those three between Wednesday and the pick deadline, and switches the pick to "
            "whichever side they moved toward when they moved at least half a point."
        ),
        "s2_leader_weighted": (
            "Same late-week line-move rule the picks already use, but the three books that move "
            "first count double and the other nine count once."
        ),
        "s3_leader_first": (
            "Listen to the three books that move first; if they have not moved enough, fall back "
            "to what all twelve books did on average."
        ),
        "s4_served_equal": (
            "The rule the picks already use: average what all twelve sportsbooks did to the line "
            "between Wednesday and the deadline, and switch the pick when that average moved at "
            "least half a point."
        ),
        "pc_oracle": (
            "A cheating benchmark that already knows the result, included only to prove the "
            "measuring stick can see a real difference when one exists."
        ),
    }

    for arm, cell in frozen["cells"].items():
        commands.append(
            row(
                name=f"{FAMILY}_{arm}_2023_2025",
                description=f"{arm_text[arm]}; paired against the raw production opener card.",
                plain=plain_text[arm],
                stamp=FROZEN,
                block=cell["week"],
                reliability=cell["fire_flag_reliability"]["correlation"],
                notes=(
                    f"Week-blocked, 20000 samples, seed 20260821. Fires {cell['fires']} of 816 "
                    f"archive games, switches {cell['switches_vs_tuesday']} picks, "
                    f"{cell['sides_differing_from_s4']} served sides differ from the served "
                    f"equal-book rule. Season-blocked "
                    f"[{cell['season']['interval_low']:.4f}, {cell['season']['interval_high']:.4f}]"
                    f" probability_positive {cell['season']['probability_positive']:.4f}. "
                    f"Per-season {cell['week']['season_effects']}. Baseline card is the frozen "
                    f"2026-09-05 opener evaluation (model ab29832a4e099766)."
                ),
            )
        )
        if "vs_s4" in cell:
            commands.append(
                row(
                    name=f"{FAMILY}_{arm}_minus_s4_2023_2025",
                    description=(
                        f"Head-to-head: {arm} minus the served equal-book follow, same games, "
                        "same blocks. Positive favours the challenger aggregation."
                    ),
                    plain=(
                        "Head-to-head between this way of reading the late-week line move and the "
                        "one the picks already use, on the same games."
                    ),
                    stamp=FROZEN,
                    block=cell["vs_s4"]["week"],
                    reliability=cell["fire_flag_reliability"]["correlation"],
                    notes=(
                        f"Week-blocked, 20000 samples, seed 20260821. "
                        f"{cell['sides_differing_from_s4']} of 816 served sides differ. "
                        f"Season-blocked "
                        f"[{cell['vs_s4']['season']['interval_low']:.4f}, "
                        f"{cell['vs_s4']['season']['interval_high']:.4f}] probability_positive "
                        f"{cell['vs_s4']['season']['probability_positive']:.4f}. Per-season "
                        f"{cell['vs_s4']['week']['season_effects']}."
                    ),
                )
            )

    for name, gate in frozen["gate_matched_diagnostic"].items():
        commands.append(
            row(
                name=f"{FAMILY}_{name}_minus_{gate['vs_peer']['peer']}_2023_2025",
                description=(
                    f"Post-hoc gate-matched diagnostic (not predeclared): {name} at threshold "
                    f"{gate['threshold']} fires {gate['fires']} times against its peer's "
                    f"{gate['target_fires']}, isolating leadership content from gate looseness."
                ),
                plain=(
                    "A fairness check: the two ways of reading the line move are compared after "
                    "being tuned to change the same number of picks, so any difference is about "
                    "which books are listened to rather than how often the rule fires."
                ),
                stamp=FROZEN,
                block=gate["vs_peer"]["week"],
                reliability=None,
                notes=(
                    f"Week-blocked, 20000 samples, seed 20260821. Threshold {gate['threshold']} "
                    f"chosen post hoc to match the peer's fire count; season-blocked "
                    f"[{gate['vs_peer']['season']['interval_low']:.4f}, "
                    f"{gate['vs_peer']['season']['interval_high']:.4f}] probability_positive "
                    f"{gate['vs_peer']['season']['probability_positive']:.4f}."
                ),
            )
        )

    for arm, cell in current["cells"].items():
        commands.append(
            row(
                name=f"{FAMILY}_{arm}_current_model_2023_2025",
                description=(
                    f"{arm_text[arm]}; replicated against the currently active card "
                    "(model c657058903f3232b, gaussian_median, served home-side offset)."
                ),
                plain=plain_text[arm],
                stamp=CURRENT,
                block=cell["week"],
                reliability=cell["fire_flag_reliability"]["correlation"],
                notes=(
                    f"Week-blocked, 20000 samples, seed 20260821. Same 816-game archive "
                    f"population, baseline is artifacts/opener_evaluation/20260909T183120Z. "
                    f"Fires {cell['fires']}, switches {cell['switches_vs_tuesday']}, "
                    f"{cell['sides_differing_from_s4']} served sides differ from the served "
                    f"equal-book rule. Season-blocked "
                    f"[{cell['season']['interval_low']:.4f}, {cell['season']['interval_high']:.4f}]"
                    f" probability_positive {cell['season']['probability_positive']:.4f}."
                ),
            )
        )
        if "vs_s4" in cell:
            commands.append(
                row(
                    name=f"{FAMILY}_{arm}_minus_s4_current_model_2023_2025",
                    description=(
                        f"Head-to-head on the currently active card: {arm} minus the served "
                        "equal-book follow, same games, same blocks."
                    ),
                    plain=(
                        "Head-to-head between this way of reading the late-week line move and the "
                        "one the picks already use, on today's model."
                    ),
                    stamp=CURRENT,
                    block=cell["vs_s4"]["week"],
                    reliability=cell["fire_flag_reliability"]["correlation"],
                    notes=(
                        f"Week-blocked, 20000 samples, seed 20260821. Season-blocked "
                        f"[{cell['vs_s4']['season']['interval_low']:.4f}, "
                        f"{cell['vs_s4']['season']['interval_high']:.4f}] probability_positive "
                        f"{cell['vs_s4']['season']['probability_positive']:.4f}. Per-season "
                        f"{cell['vs_s4']['week']['season_effects']}."
                    ),
                )
            )

    for name, gate in current["gate_matched_diagnostic"].items():
        commands.append(
            row(
                name=f"{FAMILY}_{name}_minus_{gate['vs_peer']['peer']}_current_model_2023_2025",
                description=(
                    f"Post-hoc gate-matched diagnostic on the currently active card: {name} at "
                    f"threshold {gate['threshold']} against its peer at matched fire count."
                ),
                plain=(
                    "The same fairness check on today's model: both ways of reading the line move "
                    "tuned to change the same number of picks."
                ),
                stamp=CURRENT,
                block=gate["vs_peer"]["week"],
                reliability=None,
                notes=(
                    f"Week-blocked, 20000 samples, seed 20260821. Threshold {gate['threshold']} "
                    f"chosen post hoc; season-blocked "
                    f"[{gate['vs_peer']['season']['interval_low']:.4f}, "
                    f"{gate['vs_peer']['season']['interval_high']:.4f}] probability_positive "
                    f"{gate['vs_peer']['season']['probability_positive']:.4f}."
                ),
            )
        )

    target = OUTPUT / FROZEN / "record_commands.json"
    write_stamped_artifact({"commands": commands}, target)
    print(f"{len(commands)} commands -> {target}")
    for argv in commands:
        print(argv[3])


if __name__ == "__main__":
    main()

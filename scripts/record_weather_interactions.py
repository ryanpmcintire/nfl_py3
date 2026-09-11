from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"

SCREEN_RESULTS = REPO / "artifacts" / "weather_interactions" / "screen_results.json"
SURFACE_RESULTS = REPO / "artifacts" / "weather_interactions" / "surface_switch_results.json"

CELL_META = {
    "wind_pass_heavy": {
        "family": "weather_interactions_wind_pass_heavy_fade",
        "trait_reliability_key": "pass",
        "description": (
            "Fade the pass-heavier team (prior-season pass-attempt-rate global quartile 4) "
            "in outdoor games with forecast wind>=15mph."
        ),
        "plain_summary": (
            "When it is windy outdoors, the team that likes to throw more than usual tends to "
            "cover less; this checks whether picking against that team pays off."
        ),
    },
    "wind_fg_reliant": {
        "family": "weather_interactions_wind_fg_reliant_fade",
        "trait_reliability_key": "fg",
        "description": (
            "Fade the field-goal-reliant team (prior-season FG-attempts-per-game global "
            "quartile 4, attempt-based proxy) in outdoor games with forecast wind>=15mph."
        ),
        "plain_summary": (
            "When it is windy outdoors, the team that leans on field goals to score tends to "
            "cover less; this checks whether picking against that team pays off."
        ),
    },
    "heat_pace": {
        "family": "weather_interactions_heat_pace_fade",
        "trait_reliability_key": "pace",
        "description": (
            "Fade the faster-pace team (prior-season seconds-per-play global quartile 1, "
            "fastest tempo) in outdoor games with forecast heat>=85F."
        ),
        "plain_summary": (
            "When it is hot outdoors, the team that plays a fast, no-huddle style tends to "
            "cover less; this checks whether picking against that team pays off."
        ),
    },
}

SURFACE_META = {
    "family": "weather_interactions_surface_switch_rush_fade",
    "trait_reliability_key": None,
    "description": (
        "Fade a grass-accustomed away team, run-heavy (prior-season pass-attempt-rate global "
        "quartile 1), switching onto turf this game."
    ),
    "plain_summary": (
        "When a run-first road team that is used to playing on grass has to play on turf, it "
        "tends to cover less; this checks whether picking against that team pays off."
    ),
}


def _run(argv: list[str]) -> None:
    print("RUN:", " ".join(argv))
    completed = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(argv)}")


def _record_signal(
    *,
    name: str,
    description: str,
    source: str,
    gap: dict,
    reliability: float | None,
    plain_summary: str,
    replace: bool,
) -> None:
    lower, upper = gap["ci95_accuracy_points"]
    argv = [
        str(UV),
        "run",
        "--no-sync",
        "nfl-ats",
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        description,
        "--source",
        source,
        "--effect",
        str(gap["point_estimate_accuracy_points"]),
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2009",
        "--season-end",
        "2025",
        "--interval-low",
        str(lower),
        "--interval-high",
        str(upper),
        "--probability-positive",
        str(gap["probability_positive"]),
        "--sample-games",
        str(gap["n_group_a_flag"] + gap["n_group_b_complement"]),
        "--category",
        "environment",
        "--plain-summary",
        plain_summary,
        "--classification-evidence",
        (
            "Interval crosses zero (per AGENTS.md this is the expected shape for a real "
            "small signal, never grounds to close); no positive control run this session."
        ),
    ]
    if reliability is not None:
        argv += ["--reliability", str(reliability)]
    if replace:
        argv.append("--replace")
    _run(argv)


def record_team_style_cells(replace: bool) -> None:
    payload = json.loads(SCREEN_RESULTS.read_text())
    for cell_key, meta in CELL_META.items():
        cell = payload["cells"][cell_key]
        reliability = payload["traits"][meta["trait_reliability_key"]]["reliability"]
        for cutoff_name, by_cutoff in cell["by_cutoff"].items():
            family = meta["family"]
            source = f"docs/weather_interactions.md ({family}, {cutoff_name} cutoff)"

            _record_signal(
                name=f"{family}_{cutoff_name}_vs_population",
                description=(
                    f"{meta['description']} vs. the full REG population, {cutoff_name} cutoff."
                ),
                source=source,
                gap=by_cutoff["gap_vs_population"]["week_blocked_primary"],
                reliability=reliability,
                plain_summary=meta["plain_summary"],
                replace=replace,
            )
            _record_signal(
                name=f"{family}_{cutoff_name}_vs_narrow_control",
                description=(
                    f"{meta['description']} vs. games meeting the same weather threshold "
                    f"but NOT top-quartile on the trait, {cutoff_name} cutoff (isolates the "
                    "interaction from the weather main effect)."
                ),
                source=source,
                gap=by_cutoff["gap_vs_narrow_control"]["week_blocked_primary"],
                reliability=reliability,
                plain_summary=meta["plain_summary"],
                replace=replace,
            )
            odd = by_cutoff["split_half_odd_even_seasons"]["odd"]
            even = by_cutoff["split_half_odd_even_seasons"]["even"]
            _record_signal(
                name=f"{family}_{cutoff_name}_odd_seasons",
                description=(
                    f"{meta['description']} vs. population, odd seasons only, {cutoff_name} cutoff."
                ),
                source=source,
                gap=odd,
                reliability=reliability,
                plain_summary=meta["plain_summary"],
                replace=replace,
            )
            _record_signal(
                name=f"{family}_{cutoff_name}_even_seasons",
                description=(
                    f"{meta['description']} vs. population, even seasons only, "
                    f"{cutoff_name} cutoff."
                ),
                source=source,
                gap=even,
                reliability=reliability,
                plain_summary=meta["plain_summary"],
                replace=replace,
            )


def record_surface_cell(replace: bool) -> None:
    payload = json.loads(SURFACE_RESULTS.read_text())
    family = SURFACE_META["family"]
    reliability = payload["pass_rate_trait_reliability"]
    source = f"docs/weather_interactions.md ({family})"

    _record_signal(
        name=f"{family}_vs_population",
        description=f"{SURFACE_META['description']} vs. the full REG population.",
        source=source,
        gap=payload["gap_vs_population"]["week_blocked_primary"],
        reliability=reliability,
        plain_summary=SURFACE_META["plain_summary"],
        replace=replace,
    )
    _record_signal(
        name=f"{family}_vs_narrow_control",
        description=(
            f"{SURFACE_META['description']} vs. other surface-switch games where the away "
            "team is NOT run-heavy (isolates the interaction from the surface-switch main "
            "effect)."
        ),
        source=source,
        gap=payload["gap_vs_narrow_control"]["week_blocked_primary"],
        reliability=reliability,
        plain_summary=SURFACE_META["plain_summary"],
        replace=replace,
    )
    odd = payload["split_half_odd_even_seasons"]["odd"]
    even = payload["split_half_odd_even_seasons"]["even"]
    _record_signal(
        name=f"{family}_odd_seasons",
        description=f"{SURFACE_META['description']} vs. population, odd seasons only.",
        source=source,
        gap=odd,
        reliability=reliability,
        plain_summary=SURFACE_META["plain_summary"],
        replace=replace,
    )
    _record_signal(
        name=f"{family}_even_seasons",
        description=f"{SURFACE_META['description']} vs. population, even seasons only.",
        source=source,
        gap=even,
        reliability=reliability,
        plain_summary=SURFACE_META["plain_summary"],
        replace=replace,
    )


def main() -> None:
    replace = "--replace" in sys.argv
    record_team_style_cells(replace)
    record_surface_cell(replace)


if __name__ == "__main__":
    main()

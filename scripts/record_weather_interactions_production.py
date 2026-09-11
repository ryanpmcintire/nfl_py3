from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"
OUT_DIR = REPO / "artifacts" / "weather_interactions"

RESULT_FILES = {
    "weather_interactions_wind_pass_heavy_on_production": "production_wind_pass_heavy_results.json",
    "weather_interactions_wind_fg_reliant_on_production": "production_wind_fg_reliant_results.json",
    "weather_interactions_heat_pace_on_production": "production_heat_pace_results.json",
    "weather_interactions_surface_switch_rush_on_production": (
        "production_surface_switch_rush_results.json"
    ),
}


def _run(argv: list[str]) -> None:
    print("RUN:", " ".join(argv))
    completed = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(argv)}")


def main() -> None:
    replace = "--replace" in sys.argv
    for family, filename in RESULT_FILES.items():
        payload = json.loads((OUT_DIR / filename).read_text())
        summary = payload["summary"]
        lower, upper = summary["week_blocked_ci95"]
        argv = [
            str(UV),
            "run",
            "--no-sync",
            "nfl-ats",
            "rotation",
            "record",
            "--name",
            family,
            "--artifact",
            str((OUT_DIR / filename).relative_to(REPO)),
            "--verdict",
            "unresolved",
            "--probability-positive",
            str(summary["week_blocked_probability_positive"]),
            "--effect",
            str(summary["accuracy_points"]),
            "--effect-units",
            "accuracy_points",
            "--interval-low",
            str(lower),
            "--interval-high",
            str(upper),
            "--sample-blocks",
            str(summary["n_weeks"]),
            "--notes",
            (
                f"ENV-05 predeclared fade tilt, opener grade, window "
                f"{payload['window_seasons']}. Picks changed: {payload['picks_changed']}. "
                "Interval crosses zero; per AGENTS.md this is unresolved_below_power, not a "
                "negative. docs/weather_interactions.md has the full predeclaration."
            ),
        ]
        if replace:
            argv.append("--replace")
        _run(argv)


if __name__ == "__main__":
    main()

"""Record the 4 ``forecast_weather_*`` cells from
``scripts/nfl_forecast_weather_screen.py`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the run's ``results.json`` and passed
through as a CLI argument -- no hand-typed numbers.

## Binding closing-grounds taxonomy (verbatim, per AGENTS.md / this task's brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record it with
``nfl-ats weak-signals record``, report ``probability_positive``, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell is recorded regardless of sign.

This script checks each cell's week-blocked interval against that taxonomy
before choosing a classification: ``wrong_sign_resolved`` only if the whole
interval sits on the wrong side of the cell's predicted direction (cells
2-4 predict positive; cell 1, mirroring its unsigned sibling
``weather_battery_high_wind_outdoor``, predicts no direction and can never
be wrong-sign-resolved). No positive control was run in this screen, so
``positive_control_bound`` never applies here. As measured this session, none
of the 4 cells' week-blocked intervals sit entirely on the wrong side --
every cell records ``unresolved_below_power``.

Usage::

    uv run python scripts/record_forecast_weather_screen.py --results <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

SOURCE = (
    "scripts/nfl_forecast_weather_screen.py; artifacts/forecast_weather_screen/"
    "{timestamp}/results.json; docs/forecast_weather_screen.md; "
    "docs/forecast_archive_build.md"
)

CLASSIFICATION_EVIDENCE = (
    "ENV-01 payoff screen: timing re-screen of an already-recorded weather_battery_*/"
    "weather_followup_* mechanism with the Tuesday-noon GFS-MOS forecast substituted for "
    "the game-time actual, predeclared in docs/forecast_weather_screen.md before scoring. "
    "Week-blocked 95% CI crosses zero or is narrowly resolved-shaped -- per AGENTS.md that "
    "is the EXPECTED shape for a real small signal, not grounds to close. Neither admissible "
    "closing ground applies: no resolved wrong sign (interval does not sit entirely on the "
    "wrong side of the predicted direction) and no positive-control bound was run."
)

# Predicted direction per cell, mirroring docs/forecast_weather_screen.md. `None` means the
# sibling cell itself carries no predicted direction (high_wind_outdoor is a plain, unsigned
# subset-vs-complement flag) -- wrong_sign_resolved can never apply to it.
PREDICTED_DIRECTION: dict[str, str | None] = {
    "forecast_weather_high_wind_outdoor": None,
    "forecast_weather_dome_team_outdoors_cold": "positive",
    "forecast_weather_warm_team_cold_late": "positive",
    "forecast_weather_temp_gap_cold_visitor": "positive",
}


def classify(name: str, wb: dict[str, Any]) -> tuple[str, str | None]:
    """Return (classification, closing_ground). Only a RESOLVED wrong sign
    (whole interval on the wrong side of the predicted direction) can produce
    anything other than unresolved_below_power in this screen -- no positive
    control was run.
    """

    direction = PREDICTED_DIRECTION[name]
    lo, hi = wb["ci95_scaled"]
    if direction == "positive" and hi < 0:
        return "refuted_mechanism", "wrong_sign_resolved"
    if direction == "negative" and lo > 0:
        return "refuted_mechanism", "wrong_sign_resolved"
    return "unresolved_below_power", None


def notes_for(cell: dict[str, Any], payload: dict[str, Any]) -> str:
    wb = cell["week_blocked"]
    sb = cell["season_blocked_secondary"]
    ad = cell["actual_sibling_diagnostic"]
    aw = ad["week_blocked_same_population"]
    fa = cell["flag_agreement"]
    playable = cell["playable_fraction_matched_population"]
    playable_str = f"{playable:.4f}" if playable is not None else "n/a"
    return (
        "ENV-01 payoff screen (timing re-screen, not an independent discovery): one of 4 "
        "predeclared cells in scripts/nfl_forecast_weather_screen.py, mirroring an "
        "already-recorded weather_battery_*/weather_followup_* mechanism with the "
        "Tuesday-noon GFS-MOS forecast substituted for the game-time actual on the focal "
        f"game's own temp/wind term only. Scored on {wb['n_total']:,}-row REG "
        f"{payload['season_start']}-{payload['season_end']} population -- forecast archive "
        "coverage, narrower than the 2009-2025 window the actual-weather sibling was "
        f"recorded on ({payload['n_pushes_or_missing_dropped']} pushes/rows with missing "
        f"result or spread_line dropped from {payload['n_reg_games_before_push_drop']:,} REG "
        f"games; forecast matched {payload['n_forecast_matched']:,}, fetch_status=ok "
        f"{payload['n_forecast_ok']:,}). Mined/exploratory; do not sign-test-pool against "
        "battery cells as independent, and do not pool this cell with its actual-weather "
        "sibling as a second independent input -- same mechanism, overlapping population. "
        f"Seed {payload['bootstrap_seed']}, {payload['bootstrap_samples']:,} bootstrap "
        f"draws. n_missing_required_data={cell['n_missing_required_data']} rows had a "
        "required input missing for this cell -- flag forced False, included in complement. "
        f"Season-blocked secondary (block=season, n={sb['n_blocks']} seasons, far fewer "
        "blocks than the 17-season originals): 95% "
        f"[{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
        f"P+={sb['probability_positive']:.4f} (robustness check, not the registry interval). "
        "Diagnostic, same 2020-2025 population, actual-weather construction (NOT recorded "
        f"separately): full_slate_effect={aw.get('full_slate_effect_pts', float('nan')):+.4f}"
        f"pts P+={aw.get('probability_positive', float('nan')):.4f}; playable_fraction "
        f"(forecast/actual full-slate effect, matched population) = {playable_str}. "
        f"Forecast-vs-actual flag agreement rate = {fa.get('agreement_rate')} "
        f"(n_both_defined={fa.get('n_both_defined')}, both_true={fa.get('n_both_true')}, "
        f"both_false={fa.get('n_both_false')}, forecast_only={fa.get('n_forecast_only')}, "
        f"actual_only={fa.get('n_actual_only')})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    timestamp = args.results.parent.name
    source = SOURCE.format(timestamp=timestamp)

    for cell in payload["results"]:
        wb = cell["week_blocked"]
        if wb.get("insufficient_data"):
            print(f"skip {cell['name']}: insufficient_data")
            continue

        classification, closing_ground = classify(cell["name"], wb)

        cmd = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "weak-signals",
            "record",
            "--name",
            cell["name"],
            "--description",
            cell["description"],
            "--source",
            source,
            "--effect",
            f"{wb['full_slate_effect_pts']:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            classification,
            "--league",
            "nfl",
            "--season-start",
            str(payload["season_start"]),
            "--season-end",
            str(payload["season_end"]),
            "--interval-low",
            f"{wb['ci95_scaled'][0]:.10f}",
            "--interval-high",
            f"{wb['ci95_scaled'][1]:.10f}",
            "--probability-positive",
            f"{wb['probability_positive']:.10f}",
            "--sample-games",
            str(wb["n_total"]),
            "--sample-blocks",
            str(wb["n_blocks"]),
            "--classification-evidence",
            CLASSIFICATION_EVIDENCE,
            "--notes",
            notes_for(cell, payload),
        ]
        if closing_ground:
            cmd += ["--closing-ground", closing_ground]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording {cell['name']} (classification={classification}) ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {cell['name']} (exit "
                f"{result.returncode}); per AGENTS.md 'if a record command errors, "
                "the verdict is wrong, not the validator' -- fix the invocation, "
                "do not weaken the classification to force it through."
            )


if __name__ == "__main__":
    main()

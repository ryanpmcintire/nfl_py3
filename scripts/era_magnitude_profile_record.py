"""Record one ``era_trend_<signal>`` entry per signal from
``scripts/era_magnitude_profile.py``'s run into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record`` -- never hand-edited, the CLI's own
validators decide.

``nfl_ats.weak_signals.EFFECT_UNITS`` has no per-season-slope unit
(``ats_points``, ``accuracy_points``, ``brier``, ``log_loss``, ``mae`` --
none is a rate). Per the task's own predeclared fallback
(``docs/era_magnitude_profile.md``, "Registry recording plan"), each entry
therefore records the **2020-2025-era effect** (accuracy_points, the same
scale the rest of this registry already uses) as the recordable ``effect``,
with the full season-trend slope, changepoint, and modulator-regression
findings placed in ``--notes``.

Every entry classifies mechanically via
``nfl_ats.experiment_runner.classify_subset_bias_result`` on the SAME
2020-2025 week-blocked interval already computed by the profiling script --
this script does not re-decide a classification in prose.

Usage::

    uv run --no-sync python scripts/era_magnitude_profile_record.py \
        --results artifacts/era_magnitude_profile/<timestamp>/results.json [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

DOC = "docs/era_magnitude_profile.md"

DESCRIPTIONS = {
    "surface_switch": (
        "Season-trend magnitude drift of the surface_switch construct (grass-modal visitor on "
        "turf, home_cover edge; scripts/nfl_weather_battery_screen.py). Recorded effect is the "
        "2020-2025 era slice; the season-trend slope and free-break changepoint are in notes."
    ),
    "division_revenge_game": (
        "Season-trend magnitude drift of division_revenge_game (2nd meeting this season vs. same "
        "division opponent after losing the 1st; nfl_ats.experiment_runner.FLAG_BUILDERS). "
        "Recorded effect is the 2020-2025 era slice; slope, changepoint, and a mechanistic "
        "modulator regression (league scoring-margin parity) are in notes."
    ),
    "home_underdog": (
        "Season-trend magnitude drift of home_underdog (home team getting points; "
        "nfl_ats.experiment_runner.FLAG_BUILDERS). Recorded effect is the 2020-2025 era slice; "
        "slope, changepoint, and a mechanistic modulator regression (league home-field advantage) "
        "are in notes."
    ),
    "extra_rest_edge": (
        "Season-trend magnitude drift of extra_rest_edge (own rest minus opponent rest >= 4 days; "
        "nfl_ats.experiment_runner.FLAG_BUILDERS). Recorded effect is the 2020-2025 era slice; "
        "slope, changepoint, and a mechanistic modulator regression (league rest-gap mean) are in "
        "notes."
    ),
    "penalty_rate_quartile": (
        "Season-trend magnitude drift of penalty_rate_quartile (prior-season penalty-rate Q1 vs. "
        "Q4; nfl_ats.experiment_runner.FLAG_BUILDERS). Recorded effect is the 2020-2025 era slice; "
        "slope, changepoint, and a mechanistic modulator regression (league mean penalty rate) are "
        "in notes."
    ),
    "hc_year_one_fade": (
        "Season-trend magnitude drift of hc_year_one_fade (first-year HC weeks 1-8 vs. kept-coach "
        "complement; ported from scripts/hc_year_one_fade.py onto the generic subset-vs-complement "
        "pipeline for era-slicing). Recorded effect is the 2020-2025 era slice. This is the "
        "profile's INSTRUMENT CHECK signal (known 2018-2025-concentrated positive control) -- the "
        "free-break changepoint reproduces it independently (point/modal break season 2019, one "
        "year off the known 2018 origin, spread [2013,2021]). Slope, changepoint, and a modulator "
        "regression (count of first-year HCs per season) are in notes."
    ),
    "production_model_opener_proxy_edge": (
        "Season-trend magnitude drift of the frozen production model's own opener-proxy accuracy "
        "edge vs. a 50% coin flip (probability rule; re-sliced from "
        "artifacts/proxy_opener_replication/20260819T194330Z and "
        "artifacts/opener_evaluation/20260819T174244Z, no walk-forward re-run). Recorded effect is "
        "the 2020-2025 true-opener era slice. CAVEAT stated in notes: the grade changes exactly at "
        "the 2019/2020 boundary (SBR-Open proxy through 2019, true Tuesday opener from 2020), a "
        "genuine population-definition seam this signal cannot separate from a real growing edge."
    ),
}

MODULATOR_LABELS = {
    "league_turf_share_pct": "league turf share %",
    "league_mean_abs_scoring_margin": "league mean |scoring margin|",
    "league_home_field_advantage": "league home-field advantage",
    "league_mean_rest_gap": "league mean rest gap",
    "league_mean_penalty_rate_pct": "league mean penalty rate %",
    "league_year_one_hc_count": "count of first-year HCs per season",
    "league_mean_close_books": "mean books at close (2020-2025 only)",
}


def _slope_notes(trend: dict[str, Any]) -> str:
    if trend.get("insufficient_seasons"):
        return "Season-trend slope: insufficient seasons."
    sl = trend["slope"]
    parts = [
        f"SEASON-TREND SLOPE (OLS, accuracy_points/season): point={sl['point_estimate']:+.4f}, "
        f"week-block-bootstrap 95% [{sl['bootstrap_lower']:+.4f}, {sl['bootstrap_upper']:+.4f}], "
        f"P+={sl['probability_positive']:.4f}, n_valid_bootstrap_draws="
        f"{trend.get('n_valid_bootstrap_draws', 'n/a')}."
    ]
    cp = trend.get("changepoint", {})
    if cp.get("insufficient_seasons_for_changepoint"):
        parts.append("Changepoint: insufficient seasons for a stable search.")
    elif "break_season_bootstrap_modal" in cp:
        parts.append(
            f"CHANGEPOINT (free break, min_segment={cp['min_segment_seasons']} seasons): "
            f"point break_season={cp['break_season_point_estimate']}, "
            f"pre={cp['pre_break_mean_point']:+.3f}, post={cp['post_break_mean_point']:+.3f}. "
            f"Bootstrap break-season spread: modal={cp['break_season_bootstrap_modal']} "
            f"(share={cp['break_season_bootstrap_modal_share']:.3f}), "
            f"median={cp['break_season_bootstrap_median']:.1f}, "
            f"[{cp['break_season_bootstrap_p2_5']:.1f}, {cp['break_season_bootstrap_p97_5']:.1f}]. "
            f"Stable (modal share>=0.25): {cp['stable']}."
        )
    elif "break_season_bootstrap_estimate" in cp:
        parts.append(
            f"CHANGEPOINT (free break, min_segment={cp['min_segment_seasons']} seasons): "
            f"point break_season={cp['break_season_point_estimate']}, "
            f"pre={cp['pre_break_mean_point']:+.3f}, post={cp['post_break_mean_point']:+.3f}. "
            f"Bootstrap break-season percentile interval: "
            f"[{cp['break_season_bootstrap_lower']:.1f}, {cp['break_season_bootstrap_upper']:.1f}] "
            f"(percentile of per-draw argmin, not a mode/spread report)."
        )
    mod = trend.get("modulator")
    if mod and "slope_point_estimate" in mod:
        label = MODULATOR_LABELS.get(mod["name"], mod["name"])
        parts.append(
            f"MODULATOR REGRESSION ({label}, weighted OLS, n_seasons={mod['n_seasons']}): "
            f"slope={mod['slope_point_estimate']:+.5f}, "
            f"95% [{mod['bootstrap_lower']:+.5f}, {mod['bootstrap_upper']:+.5f}], "
            f"P+={mod['probability_positive']:.4f}."
        )
        if mod.get("caveat"):
            parts.append(f"Modulator caveat: {mod['caveat']}")
    elif mod and mod.get("insufficient_overlap"):
        parts.append("Modulator regression: insufficient season overlap.")
    return " ".join(parts)


def build_record(name: str, payload: dict[str, Any], sig: dict[str, Any]) -> dict[str, Any] | None:
    era = sig.get("era_results", {}).get("era_2020_2025")
    if era is None or era.get("insufficient_data"):
        print(f"skip {name}: era_2020_2025 insufficient_data")
        return None

    if "week_blocked" in era:  # six subset-vs-complement signals
        wb = era["week_blocked"]
        effect = era["effect"]
        lower, upper = wb["lower"], wb["upper"]
        prob_positive = wb["probability_positive"]
        sample_games = era["n_total"]
        sample_blocks = wb["block_count"]
        classification_block = era["mechanical_classification"]
    else:  # production_model_opener_proxy_edge
        effect = era["estimate"]
        lower, upper = era["lower"], era["upper"]
        prob_positive = era["probability_positive"]
        sample_games = era["n_games"]
        sample_blocks = None
        classification_block = era["mechanical_classification"]

    trend = dict(sig.get("season_trend", {}))
    if "modulator" not in trend and "modulator" in sig:
        # production_model_opener_proxy_edge stores its modulator at the signal
        # level (a different population -- true-opener-only sub-slice -- than
        # the slope/changepoint's full 2011-2025 series), not nested under
        # season_trend like the six subset-vs-complement signals.
        trend["modulator"] = sig["modulator"]
    notes = _slope_notes(trend)
    notes += (
        f" SCALE NOTE: recorded --effect/--interval are the 2020-2025 era slice (fixed-era "
        f"convenience bucket), NOT the season-trend slope itself -- see the SEASON-TREND SLOPE "
        f"figure above for the per-season-magnitude-drift finding this entry exists to record. "
        f"Full profile: {DOC}, artifact "
        f"artifacts/era_magnitude_profile/{payload['_timestamp']}/results.json."
    )
    reliability = sig.get("reliability")

    record: dict[str, Any] = {
        "name": f"era_trend_{name}",
        "description": DESCRIPTIONS[name],
        "source": (
            f"scripts/era_magnitude_profile.py; "
            f"artifacts/era_magnitude_profile/{payload['_timestamp']}/results.json; {DOC}"
        ),
        "effect": f"{effect:.10f}",
        "effect_units": "accuracy_points",
        "classification": classification_block["classification"],
        "league": "nfl",
        "season_start": "2020",
        "season_end": "2025",
        "interval_low": f"{lower:.10f}",
        "interval_high": f"{upper:.10f}",
        "probability_positive": f"{prob_positive:.10f}",
        "sample_games": str(sample_games),
        "classification_evidence": classification_block["note"],
        "notes": notes,
    }
    if sample_blocks is not None:
        record["sample_blocks"] = str(sample_blocks)
    if reliability is not None:
        record["reliability"] = f"{reliability:.6f}"
    if classification_block.get("closing_ground") is not None:
        record["closing_ground"] = classification_block["closing_ground"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    payload["_timestamp"] = args.results.parent.name

    for name, sig in payload["signals"].items():
        record = build_record(name, payload, sig)
        if record is None:
            continue

        cmd = [sys.executable, "-m", "nfl_ats.cli", "weak-signals", "record"]
        for key, value in record.items():
            flag = "--" + key.replace("_", "-")
            cmd += [flag, value]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording era_trend_{name} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for era_trend_{name} (exit {result.returncode}); "
                "per AGENTS.md 'if a record command errors, the verdict is wrong, not the "
                "validator' -- fix the invocation, do not weaken the classification to force it "
                "through."
            )


if __name__ == "__main__":
    main()

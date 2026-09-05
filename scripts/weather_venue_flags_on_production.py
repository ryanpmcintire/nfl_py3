"""Run one predeclared, rotation-assigned opener confirmation for a
weather/venue candidate stacked on PRODUCTION (ROADMAP LEAD-36 open-corner
stadium wind, LEAD-37 rain-on-grass fumble chaos).

Predeclared in ``docs/weather_venue_leads.md`` before either candidate was
scored. Every candidate column is built from the newest local
``schedules.parquet`` snapshot plus the Tuesday-opener market store and,
for ``rain_on_grass`` only, the validated pool-decision forecast archive
(``nfl_ats.weather_venue_flag_features``), merged onto the PRODUCTION
feature table by ``game_id`` at runtime -- no precomputed candidate-specific
parquet is written.

This is a thin wrapper around ``scripts/on_production_opener_confirmation.py``
(imported, never edited): it reuses that module's ``profile_identity``,
``scoped_window_frame``, ``run_arm``, ``paired_frame``, ``summarize``, and
``null_distribution`` verbatim -- the same estimator as the played
``weak_stack`` chain, the same week-blocked bootstrap, the same
within-week permutation null, the same positive-control leak (the
candidate column replaced by the REALIZED margin), and the same refusal to
infer a confirmation window from the command line. ``null`` and
``positive-control`` are instrument checks; only ``screen`` is the single
outcome look.

A sibling script rather than an extension of ``scripts/schedule_flag_on_production.py``'s,
``scripts/pbp_trait_on_production.py``'s, or ``scripts/qb_identity_on_production.py``'s
own ``CANDIDATES`` map, for the same reason each of those gives for not
extending the one before it: concurrent fleet-lane edits, and this
candidate's own inputs (schedule + opener store + forecast archive) do not
match any of their loader signatures. Zero risk of colliding with a
concurrent edit to any of them.

**LEAD-36 disclosure, repeated here:** the ``open_corner_wind`` candidate's
wind/roof inputs are the schedule's OBSERVED game-time actuals, not a
pregame forecast -- a disclosed mechanism/upper-bound screen, matching the
venue-blind ``weather_battery_high_wind_*`` registry precedent. See
``nfl_ats.weather_venue_flag_features``'s module docstring for the full
disclosure. ``rain_on_grass`` uses the validated pregame-safe
``forecast_precip_prob_pct`` proxy instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.weather_venue_flag_features import (  # noqa: E402
    OPEN_CORNER_WIND_DOG_COLUMN,
    RAIN_ON_GRASS_DOG_COLUMN,
    attach_open_corner_wind_dog_features,
    attach_rain_on_grass_dog_features,
    default_forecast_archive,
    default_opener_lines,
    default_schedule,
    open_corner_wind_population_diagnostic,
    rain_on_grass_population_diagnostic,
)

BASELINE_PROFILE = confirmation.BASELINE_PROFILE
DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"


@dataclass(frozen=True)
class WeatherVenueCandidate:
    """Duck-compatible with ``on_production_opener_confirmation.Candidate``:
    carries the same ``family``/``profile``/``column`` attribute names, so
    the template's ``profile_identity``/``scoped_window_frame``/``run_arm``
    accept it unmodified."""

    family: str
    profile: str
    column: str
    predeclaration: str
    artifact_dir: str
    needs_forecast: bool


CANDIDATES: dict[str, WeatherVenueCandidate] = {
    "open_corner_wind": WeatherVenueCandidate(
        family="open_corner_wind_dog_on_production",
        profile="weak_stack_open_corner_wind_dog",
        column=OPEN_CORNER_WIND_DOG_COLUMN,
        predeclaration="docs/weather_venue_leads.md#lead-36-open-corner-stadium-wind",
        artifact_dir="weather_venue_flags_on_production/open_corner_wind",
        needs_forecast=False,
    ),
    "rain_on_grass": WeatherVenueCandidate(
        family="rain_on_grass_dog_on_production",
        profile="weak_stack_rain_on_grass_dog",
        column=RAIN_ON_GRASS_DOG_COLUMN,
        predeclaration="docs/weather_venue_leads.md#lead-37-rain-on-grass-fumble-chaos",
        artifact_dir="weather_venue_flags_on_production/rain_on_grass",
        needs_forecast=True,
    ),
}


def build_candidate_features(
    base_features: pd.DataFrame,
    candidate: WeatherVenueCandidate,
    *,
    schedule: pd.DataFrame,
    opener_lines: pd.DataFrame,
    forecast: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge the one candidate weather/venue column onto the PRODUCTION table."""

    if candidate.family == "open_corner_wind_dog_on_production":
        return attach_open_corner_wind_dog_features(
            base_features, schedule=schedule, opener_lines=opener_lines
        )
    if candidate.family == "rain_on_grass_dog_on_production":
        if forecast is None:
            raise ValueError("rain_on_grass requires the forecast archive")
        return attach_rain_on_grass_dog_features(
            base_features, schedule=schedule, opener_lines=opener_lines, forecast=forecast
        )
    raise ValueError(f"unrecognized candidate family: {candidate.family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--forecast-archive", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=confirmation.DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=confirmation.BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]

    features_path = args.features or DEFAULT_FEATURES
    base_features = pd.read_parquet(features_path)
    schedule = pd.read_parquet(args.schedules) if args.schedules is not None else default_schedule()
    opener_lines = default_opener_lines(schedule, market_root=args.market_root)
    forecast = default_forecast_archive(args.forecast_archive) if candidate.needs_forecast else None
    features = build_candidate_features(
        base_features, candidate, schedule=schedule, opener_lines=opener_lines, forecast=forecast
    )

    identity = confirmation.profile_identity(candidate, features)
    scoped, seasons = confirmation.scoped_window_frame(
        features, load_registry(args.registry), candidate.family
    )
    started = time.time()
    baseline = confirmation.run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=BASELINE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=False,
    )
    treatment = confirmation.run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=candidate.profile,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=args.mode == "positive-control",
    )
    paired = confirmation.paired_frame(baseline, treatment)
    if paired.empty:
        raise RuntimeError("No paired opener-grade games were scored")

    result: dict[str, Any] = {
        "status": "scored",
        "profile_identity": identity,
        "paired_games": len(paired),
        "paired_weeks": int(paired.groupby(["season", "week"]).ngroups),
    }
    if args.mode == "null":
        result["null_production_rule"] = confirmation.null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["null_sign_rule"] = confirmation.null_distribution(
            paired, probability_rule=False, permutations=args.permutations, seed=args.seed
        )
    else:
        for label, reference, treatment_col in (
            ("opener_production_rule", "baseline_correct_open_pr", "candidate_correct_open_pr"),
            ("opener_sign_rule", "baseline_correct_open", "candidate_correct_open"),
            ("close_production_rule", "baseline_correct_close_pr", "candidate_correct_close_pr"),
            ("close_sign_rule", "baseline_correct_close", "candidate_correct_close"),
        ):
            result[label] = confirmation.summarize(
                paired, reference, treatment_col, args.bootstrap_samples, args.seed
            )
        result["permutation_null_production_rule"] = confirmation.null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["baseline_metrics"] = confirmation.opener_evaluation_metrics(baseline)
        result["candidate_metrics"] = confirmation.opener_evaluation_metrics(treatment)
        result["picks_disagreeing_production_rule"] = int(
            (paired.baseline_pick_home_pr != paired.candidate_pick_home_pr).sum()
        )
        if candidate.family == "open_corner_wind_dog_on_production":
            result["open_corner_wind_population_diagnostic"] = (
                open_corner_wind_population_diagnostic(schedule, opener_lines)
            )
        if candidate.family == "rain_on_grass_dog_on_production":
            assert forecast is not None
            result["rain_on_grass_population_diagnostic"] = rain_on_grass_population_diagnostic(
                schedule, opener_lines, forecast
            )

    configuration = {
        "candidate": args.candidate,
        "mode": args.mode,
        "family": candidate.family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": candidate.profile,
        "candidate_column": candidate.column,
        "predeclaration": candidate.predeclaration,
        "features_path": str(features_path),
        "market_root": str(args.market_root),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, features_path, project_root=REPO_ROOT),
    }
    output = (
        REPO_ROOT
        / "artifacts"
        / candidate.artifact_dir
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="weather-venue-flags-on-production",
        metrics={"mode": args.mode, "status": "scored"},
        notes=(
            "Rotation-assigned opener confirmation for a weather/venue "
            "candidate (ROADMAP LEAD-36/LEAD-37); prediction-level paired "
            "output retained; the candidate column is computed at runtime "
            "from the local schedule/opener-line/forecast-archive snapshots, "
            "never read from a precomputed parquet."
        ),
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

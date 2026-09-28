import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    STATION_MAP_RELATIVE_PATH,
    _bulletin_cache_path,
    _fetch_tuesday_noon_forecast_temps,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import games_for_forecast_fetch
from nfl_ats.four_overlay_composition import protection_flags_for_card
from nfl_ats.pick_probability import (
    PickProbabilityModel,
    market_move_toward_home,
    signed_composition_flags,
)

output = Path("artifacts/diagnostics/2026-three-weeks")
grades = pd.read_csv("artifacts/diagnostics/2026-three-weeks/published_grades.csv")
forecasts = []
for w in [2, 3]:
    for path in Path("artifacts/margin_predictions").glob(f"2026-week-{w:02d}-*/metadata.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("feature_profile") == "weak_stack":
            forecasts.append((w, pd.Timestamp(meta["created_at_utc"]), path.parent))
coefficients = [
    (pd.Timestamp(datetime.strptime(p.parent.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)), p)
    for p in Path("artifacts/pick_probability").glob("*/coefficients.json")
]
snapshots = [
    (pd.Timestamp(datetime.strptime(p.parent.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)), p)
    for p in Path("data/raw").glob("20*/schedules.parquet")
]
arrest_snapshots = [
    (pd.Timestamp(datetime.strptime(p.parent.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)), p)
    for p in Path("data/raw/player_arrests").glob("20*/manifest.json")
]
cache_paths = []


def cached_bulletin(station, runtime_utc, *, model):
    path = _bulletin_cache_path(station, model, runtime_utc.strftime("%Y-%m-%dT%H:%MZ"))
    if not path.is_file():
        raise FileNotFoundError(path)
    cache_paths.append(str(path))
    return json.loads(path.read_text(encoding="utf-8"))["data"]


rows = []
failures = []
for (week, instant), published in grades[grades.week.isin([2, 3])].groupby(
    ["week", "published_at_utc"]
):
    as_of = pd.Timestamp(instant)
    _, _, artifact = max(
        [item for item in forecasts if item[0] == week and item[1] <= as_of],
        key=lambda item: item[1],
    )
    _, coef_path = max(
        [item for item in coefficients if item[0] <= as_of], key=lambda item: item[0]
    )
    _, schedule_path = max(
        [item for item in snapshots if item[0] <= as_of], key=lambda item: item[0]
    )
    print("Replaying", week, instant, artifact.name, coef_path.parent.name, flush=True)
    model = PickProbabilityModel.from_dict(
        json.loads(coef_path.read_text(encoding="utf-8")), artifact=str(coef_path.parent)
    )
    predictions = pd.read_csv(artifact / "recommendations.csv")
    schedules = pd.read_parquet(schedule_path)
    try:
        forecasts_tuesday = _fetch_tuesday_noon_forecast_temps(
            games_for_forecast_fetch(predictions, schedules),
            Path("registry") / STATION_MAP_RELATIVE_PATH,
            fetch_bulletin=cached_bulletin,
            delay_seconds=0.0,
        )
        _, arrest_manifest_path = max(
            [item for item in arrest_snapshots if item[0] <= as_of], key=lambda item: item[0]
        )
        arrest_manifest = json.loads(arrest_manifest_path.read_text(encoding="utf-8"))
        assert arrest_manifest["complete"] is True
        assert arrest_manifest["snapshot_id"] == arrest_manifest_path.parent.name
        assert (
            pd.Timedelta(0)
            <= as_of - pd.Timestamp(arrest_manifest["fetched_at_utc"])
            <= pd.Timedelta(hours=36)
        )
        safe_name = arrest_manifest["point_in_time_policy"]["safe_index"]
        assert Path(safe_name).name == safe_name
        arrest_path = arrest_manifest_path.parent / safe_name
        assert (
            hashlib.sha256(arrest_path.read_bytes()).hexdigest()
            == arrest_manifest["files"][safe_name]
        )
        incidents = pd.read_parquet(arrest_path, columns=["record_id", "incident_date", "team"])
        protection = protection_flags_for_card(predictions, Path("data"))
        flags = signed_composition_flags(
            predictions,
            schedules,
            incidents=incidents,
            forecasts_tuesday_noon=forecasts_tuesday,
            protection_back_side=protection,
        )
        movement = market_move_toward_home(
            predictions,
            Path("data"),
            now=as_of.to_pydatetime(),
            feature_version=model.market_move_feature_version,
        )
    except Exception as exc:
        failures.append(
            {
                "week": int(week),
                "as_of": instant,
                "error": str(exc),
                "games": published.game_id.tolist(),
            }
        )
        continue
    merged = (
        predictions.merge(flags, on="game_id").merge(movement, on="game_id").set_index("game_id")
    )
    for published_row in published.to_dict("records"):
        r = merged.loc[published_row["game_id"]]
        assert np.isclose(r.market_spread, published_row["market_spread"])
        raw_p = float(r.home_cover_probability)
        flag_sum = float(r.composition_flag_sum)
        move = float(r.market_move_toward_home)
        available = float(r.market_move_available)
        terms = {
            "raw_model": model.model_logit * math.log(raw_p / (1 - raw_p)),
            "situational_flags": model.flag_sum * flag_sum,
            "market_movement": model.move_toward_home * move,
            "intercept_and_availability": model.intercept + model.move_available * available,
        }
        z = sum(terms.values())
        p = 1 / (1 + math.exp(-z))
        served_home = published_row["pick_team"] == r.home_team
        recorded_home_p = (
            published_row["displayed_score"]
            if served_home
            else 1 - published_row["displayed_score"]
        )
        result = {
            **published_row,
            "raw_home_probability": raw_p,
            "reconstructed_home_probability": p,
            "recorded_home_probability": recorded_home_p,
            "probability_error": p - recorded_home_p,
            "side_matches": (p >= 0.5) == served_home,
            "raw_pick": r.home_team if raw_p >= 0.5 else r.away_team,
            "flag_sum": flag_sum,
            "movement": move,
            "movement_available": available,
            "dominant_term": max(terms, key=lambda k: abs(terms[k])),
            "forecast_artifact": str(artifact),
            "coefficient_artifact": str(coef_path.parent),
            "schedule_source": str(schedule_path),
            "arrest_source": str(arrest_path),
        }
        result.update({k: int(r[k]) for k in flags if k.startswith("flag_")})
        result.update({"term_" + k: v for k, v in terms.items()})
        for name, value in terms.items():
            ablated_p = 1 / (1 + math.exp(-(z - value)))
            result["without_" + name + "_home_probability"] = ablated_p
        rows.append(result)
frame = pd.DataFrame(rows)
frame.to_csv(output / "live_input_reconstruction.csv", index=False)
summary = {
    "replayed_games": len(frame),
    "failures": failures,
    "side_mismatches": frame.loc[
        ~frame.side_matches,
        [
            "game_id",
            "pick_team",
            "raw_pick",
            "reconstructed_home_probability",
            "recorded_home_probability",
        ],
    ].to_dict("records")
    if len(frame)
    else [],
    "probability_mismatches": frame.loc[
        frame.probability_error.abs().gt(1e-7),
        ["game_id", "probability_error", "coefficient_artifact"],
    ].to_dict("records")
    if len(frame)
    else [],
    "weather_cache_paths": sorted(set(cache_paths)),
    "protection_source_limitation": (
        "Production protection lookup has no as-of selector; reconstructed values are checked "
        "against the original published probability, not assumed to be historical snapshots."
    ),
}
(output / "live_input_reconstruction.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(
    "SUMMARY",
    json.dumps({k: v for k, v in summary.items() if k != "weather_cache_paths"}),
    flush=True,
)

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, run_id
from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MODEL_HOME_PROBABILITY_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PickProbabilityModel,
    load_pick_probability_model,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.published_picks import game_deadlines
from nfl_ats.tiebreaker import newest_schedules_path

STUDY_ROOT = Path("prospective")
DEFAULT_STUDY = "independent_validation_v2"
RECIPE_KEYS = (
    "ats_method",
    "regressor",
    "ridge_alpha",
    "calibration_method",
    "feature_profile",
    "probability_method",
    "min_train_games",
)
PINNED_NAMES = {
    "card_view.py",
    "clv.py",
    "constants.py",
    "four_overlay_composition.py",
    "independent_validation.py",
    "mass_preserving_lattice.py",
    "pick_probability.py",
    "pick_refresh.py",
    "prediction_safety.py",
    "snapshots.py",
    "source_policy.py",
}


def _digest(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = [
        path
        for path in root.rglob("*.py")
        if path.name in PINNED_NAMES
        or path.name.endswith(("features.py", "_overlay.py"))
        or "margin" in path.name
        or "model" in path.name
        or path.relative_to(root).as_posix() == "cli_commands/prediction.py"
    ]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)}


def _root(artifacts_root: Path, study: str = DEFAULT_STUDY) -> Path:
    return artifacts_root / STUDY_ROOT / study


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DataContractError(f"Expected an object at {path}")
    return value


def _enrollment(
    artifacts_root: Path, *, check_sources: bool = False, study: str = DEFAULT_STUDY
) -> dict[str, Any]:
    envelope = _read(_root(artifacts_root, study) / "enrollment.json")
    payload = envelope["enrollment"]
    if _digest(payload) != envelope["sha256"]:
        raise DataContractError("Independent-validation enrollment was altered")
    if check_sources and _sources() != payload["source_sha256"]:
        raise DataContractError("Independent-validation source changed; do not mix model versions")
    return envelope


def enroll(
    artifacts_root: Path, data_root: Path, protocol_path: Path, *, study: str = DEFAULT_STUDY
) -> dict[str, Any]:
    destination = _root(artifacts_root, study) / "enrollment.json"
    if destination.exists():
        raise DataContractError(
            "Independent validation is already enrolled; enrollment is immutable"
        )
    protocol = _read(protocol_path)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("primary_metric") != "raw_minus_combined_brier"
    ):
        raise DataContractError("Unsupported independent-validation protocol")
    now = pd.Timestamp(datetime.now(UTC))
    schedule_path = newest_schedules_path(data_root)
    schedule = pd.read_parquet(schedule_path)
    cohort = schedule.loc[
        schedule.season.eq(int(protocol["season"]))
        & schedule.week.between(int(protocol["start_week"]), int(protocol["end_week"]))
        & schedule.game_type.eq(protocol["game_type"])
    ].copy()
    kickoffs = pd.to_datetime(
        cohort.gameday.astype(str) + " " + cohort.gametime.astype(str), errors="raise"
    ).dt.tz_localize("America/New_York")
    cohort = cohort.loc[kickoffs.gt(now)].copy()
    if cohort.empty or cohort.game_id.duplicated().any():
        raise DataContractError("Independent validation needs a nonempty unique future cohort")
    if cohort[["home_score", "away_score"]].notna().any(axis=None):
        raise DataContractError("Independent-validation cohort already has observed outcomes")
    days = pd.to_datetime(cohort.gameday, errors="raise", utc=True)
    if days.isna().any():
        raise DataContractError("Every declared game must be strictly after enrollment")
    cohort["scheduled_gameday"] = days.dt.strftime("%Y-%m-%d")
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise DataContractError("Independent validation requires a synchronized active model")
    forecast = artifacts_root / str(active["weekly_forecast"]["artifact"])
    metadata = _read(forecast / "metadata.json")
    model = load_pick_probability_model(artifacts_root)
    payload = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "enrolled_at_utc": now.isoformat(),
        "cohort": cohort[["game_id", "season", "week", "scheduled_gameday"]].to_dict(
            orient="records"
        ),
        "schedule_sha256": sha256_file(schedule_path),
        "coefficient_artifact": model.artifact,
        "frozen_model": model.to_dict(),
        "raw_recipe": {key: metadata.get(key) for key in RECIPE_KEYS},
        "forecast_columns": list(pd.read_csv(forecast / "recommendations.csv", nrows=0).columns),
        "source_sha256": _sources(),
    }
    envelope = {"enrollment": payload, "sha256": _digest(payload)}
    atomic_json(envelope, destination)
    return status(artifacts_root, study)


def _captures(
    artifacts_root: Path, envelope: dict[str, Any], study: str = DEFAULT_STUDY
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((_root(artifacts_root, study) / "captures").glob("*.json")):
        capture = _read(path)
        payload = capture["capture"]
        if (
            capture["sha256"] != _digest(payload)
            or payload["enrollment_sha256"] != envelope["sha256"]
        ):
            raise DataContractError(f"Independent-validation capture identity changed: {path.name}")
        rows.extend(payload["rows"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    times = ["recorded_at_utc", "kickoff", "deadline_utc", "forecast_created_at_utc"]
    for column in times:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    enrolled = pd.Timestamp(envelope["enrollment"]["enrolled_at_utc"])
    if (
        frame[times].isna().any(axis=None)
        or frame.recorded_at_utc.lt(enrolled).any()
        or frame.recorded_at_utc.ge(frame.kickoff).any()
        or frame.recorded_at_utc.ge(frame.deadline_utc).any()
        or frame.forecast_created_at_utc.gt(frame.recorded_at_utc).any()
    ):
        raise DataContractError("Independent-validation capture violates pregame chronology")
    known = {row["game_id"] for row in envelope["enrollment"]["cohort"]}
    if not frame.game_id.isin(known).all():
        raise DataContractError("Independent-validation capture is outside its declared cohort")
    if frame.groupby("game_id").spread_line.nunique().gt(1).any():
        raise DataContractError("Independent-validation original grading line changed")
    return frame.sort_values("recorded_at_utc").groupby("game_id", as_index=False).tail(1)


def status(artifacts_root: Path, study: str = DEFAULT_STUDY) -> dict[str, Any]:
    if not (_root(artifacts_root, study) / "enrollment.json").exists():
        return {"status": "not_enrolled", "study": study}
    envelope = _enrollment(artifacts_root, study=study)
    enrollment = envelope["enrollment"]
    frame = _captures(artifacts_root, envelope, study)
    return {
        "status": "collecting" if not frame.empty else "awaiting_first_capture",
        "study_id": enrollment["protocol"]["study_id"],
        "enrollment_sha256": envelope["sha256"],
        "enrolled_at_utc": enrollment["enrolled_at_utc"],
        "expected_games": len(enrollment["cohort"]),
        "captured_games": len(frame),
        "source_unchanged": _sources() == enrollment["source_sha256"],
        "results": "withheld_until_declared_cohort_is_final",
    }


def capture(
    artifacts_root: Path, data_root: Path, *, dry: bool = False, study: str = DEFAULT_STUDY
) -> dict[str, Any]:
    from nfl_ats.clv import current_played_card_view

    envelope = _enrollment(artifacts_root, check_sources=True, study=study)
    enrollment = envelope["enrollment"]
    protocol = enrollment["protocol"]
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise DataContractError("No active forecast to capture")
    current = active["weekly_forecast"]
    if int(current["season"]) != int(protocol["season"]) or not (
        int(protocol["start_week"]) <= int(current["week"]) <= int(protocol["end_week"])
    ):
        return {"status": "outside_declared_period", "recorded": 0, "dry": dry}
    now = datetime.now(UTC)
    forecast = artifacts_root / str(current["artifact"])
    metadata = _read(forecast / "metadata.json")
    if {key: metadata.get(key) for key in RECIPE_KEYS} != enrollment["raw_recipe"]:
        raise DataContractError("Independent-validation raw model recipe changed")
    raw = pd.read_csv(forecast / "recommendations.csv")
    if list(raw.columns) != enrollment["forecast_columns"]:
        raise DataContractError("Independent-validation forecast schema changed")
    before = {
        name: sha256_file(forecast / name) for name in ("recommendations.csv", "metadata.json")
    }
    played = current_played_card_view(artifacts_root, data_root=data_root, now=now)
    if played.forecast_artifact != str(current["artifact"]):
        raise DataContractError("Active forecast changed during independent capture")
    frame = played.view.predictions.copy()
    needed = [MODEL_HOME_PROBABILITY_COLUMN, FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN]
    if (
        not set(needed).issubset(frame.columns)
        or not np.isfinite(frame[needed].to_numpy(dtype=float)).all()
    ):
        raise DataContractError("Independent validation is missing fitted-term inputs")
    model = PickProbabilityModel.from_dict(
        enrollment["frozen_model"], artifact=enrollment["coefficient_artifact"]
    )
    if (
        played.view.pick_probability is None
        or played.view.pick_probability.market_move_feature_version
        != model.market_move_feature_version
    ):
        raise DataContractError("Independent-validation movement feature changed")
    raw_lookup = raw.set_index("game_id")
    if not np.allclose(
        frame[MODEL_HOME_PROBABILITY_COLUMN], frame.game_id.map(raw_lookup.home_cover_probability)
    ):
        raise DataContractError(
            "Independent-validation raw probability differs from the served input"
        )
    if (
        model.base_probability_policy is None
        or "base_probability_policy" not in raw
        or not raw.base_probability_policy.eq(model.base_probability_policy).all()
    ):
        raise DataContractError(
            "Independent validation requires the frozen discrete probability policy"
        )
    mass = raw[
        ["home_cover_probability_excluding_push", "push_probability", "home_loss_probability"]
    ].astype(float)
    if (
        not np.isfinite(mass.to_numpy()).all()
        or (mass < 0).any(axis=None)
        or not np.allclose(mass.sum(axis=1), 1)
    ):
        raise DataContractError("Independent-validation discrete mass is invalid")
    if not np.allclose(mass.iloc[:, 0] / (1 - mass.iloc[:, 1]), raw.home_cover_probability):
        raise DataContractError(
            "Independent-validation probability is not conditional on the original line"
        )
    combined = model.home_probability(
        frame[MODEL_HOME_PROBABILITY_COLUMN],
        frame[FLAG_SUM_COLUMN],
        frame[MOVE_COLUMN],
        frame[MOVE_AVAILABLE_COLUMN],
    )
    deadlines = game_deadlines(raw)
    known = {row["game_id"] for row in enrollment["cohort"]}
    existing = _captures(artifacts_root, envelope, study)
    old_lines = {} if existing.empty else existing.set_index("game_id").spread_line.to_dict()
    rows = []
    for index, row in frame.iterrows():
        game_id = str(row.game_id)
        deadline = deadlines.get(game_id)
        if game_id not in known or deadline is None or deadline <= pd.Timestamp(now):
            continue
        original = cast(pd.Series, raw_lookup.loc[game_id])
        if any(pd.notna(original.get(column)) for column in ("home_score", "away_score", "result")):
            raise DataContractError("Independent validation cannot capture an observed outcome")
        spread = float(original.spread_line)
        if not np.isfinite(spread) or (game_id in old_lines and old_lines[game_id] != spread):
            raise DataContractError("Independent-validation original grading line changed")
        raw_p = float(row[MODEL_HOME_PROBABILITY_COLUMN])
        combined_p = float(combined.loc[index])
        push = float(original.push_probability)
        if not 0 < raw_p < 1 or not 0 < combined_p < 1 or not 0 <= push < 1:
            raise DataContractError("Independent-validation probability is invalid")
        created = played.forecast_created_at_utc
        kickoff = pd.Timestamp(row.kickoff)
        if created > pd.Timestamp(now) or kickoff <= pd.Timestamp(now):
            raise DataContractError("Independent-validation capture violates pregame chronology")
        rows.append(
            {
                "game_id": game_id,
                "season": int(row.season),
                "week": int(row.week),
                "kickoff": kickoff.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "recorded_at_utc": now.isoformat(),
                "forecast_created_at_utc": created.isoformat(),
                "forecast_artifact": played.forecast_artifact,
                "model_id": played.model_id,
                "spread_line": spread,
                "raw_home_probability": raw_p,
                "combined_home_probability": combined_p,
                "neutral_home_probability": 0.5,
                "push_probability": push,
                "flag_sum": float(row[FLAG_SUM_COLUMN]),
                "market_move": float(row[MOVE_COLUMN]),
                "market_available": float(row[MOVE_AVAILABLE_COLUMN]),
            }
        )
    if before != {name: sha256_file(forecast / name) for name in before}:
        raise DataContractError("Forecast changed during independent-validation capture")
    if not rows:
        return {"status": "no_eligible_pregame_games", "recorded": 0, "dry": dry}
    payload = {"enrollment_sha256": envelope["sha256"], "source_sha256": before, "rows": rows}
    if not dry:
        path = _root(artifacts_root, study) / "captures" / f"{run_id(now)}.json"
        if path.exists():
            raise DataContractError("Independent-validation capture already exists")
        atomic_json({"capture": payload, "sha256": _digest(payload)}, path)
    return {"status": "dry_run" if dry else "captured", "recorded": len(rows), "dry": dry}


def score(
    artifacts_root: Path, outcomes: pd.DataFrame, study: str = DEFAULT_STUDY
) -> dict[str, Any]:
    from nfl_ats.clv import week_blocked_bootstrap

    envelope = _enrollment(artifacts_root, study=study)
    enrollment = envelope["enrollment"]
    cohort = pd.DataFrame(enrollment["cohort"])
    frame = _captures(artifacts_root, envelope, study)
    required = {"game_id", "home_score", "away_score"}
    if not required.issubset(outcomes.columns):
        raise DataContractError("Independent-validation scoring requires final game scores")
    finals = outcomes.loc[outcomes.game_id.isin(cohort.game_id), sorted(required)].copy()
    if finals.game_id.duplicated().any():
        raise DataContractError("Independent-validation outcomes contain duplicate games")
    finals = finals.dropna(subset=["home_score", "away_score"])
    coverage = status(artifacts_root, study)
    coverage["final_games"] = len(finals)
    if (
        len(finals) != len(cohort)
        or pd.Timestamp(datetime.now(UTC))
        <= pd.to_datetime(cohort.scheduled_gameday, utc=True).max()
    ):
        return {
            **coverage,
            "status": "collecting",
            "results": "withheld_until_declared_cohort_is_final",
        }
    missing = sorted(set(cohort.game_id) - (set() if frame.empty else set(frame.game_id)))
    if missing:
        return {**coverage, "status": "incomplete_capture", "missing_game_ids": missing}
    paired = frame.merge(finals, on="game_id", validate="one_to_one")
    scores = paired[["home_score", "away_score"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(scores.to_numpy()).all() or (scores < 0).any(axis=None):
        raise DataContractError("Independent-validation outcomes contain invalid final scores")
    margin = scores.home_score - scores.away_score - paired.spread_line.astype(float)
    pushes = int(margin.eq(0).sum())
    paired = paired.loc[margin.ne(0)].copy()
    if paired.empty:
        return {**coverage, "status": "no_decisive_games", "pushes": pushes}
    y = margin.loc[paired.index].gt(0).astype(float)
    metrics: dict[str, Any] = {}
    for arm in ("combined", "raw", "neutral"):
        p = paired[f"{arm}_home_probability"].astype(float)
        if not np.isfinite(p.to_numpy()).all() or not p.between(0, 1, inclusive="neither").all():
            raise DataContractError("Independent-validation capture has invalid probabilities")
        hit = p.ge(0.5).eq(y.astype(bool))
        confidence = p.where(p.ge(0.5), 1 - p)
        bins = pd.cut(confidence, [0.499999, 0.55, 0.60, 0.65, 1.000001], right=False)
        reliability = pd.DataFrame({"bin": bins, "p": confidence, "hit": hit.astype(float)})
        table = (
            reliability.groupby("bin", observed=True)
            .agg(games=("hit", "size"), predicted=("p", "mean"), observed=("hit", "mean"))
            .reset_index()
        )
        table["bin"] = table["bin"].astype(str)
        metrics[arm] = {
            "correct": int(hit.sum()),
            "games": len(paired),
            "accuracy": float(hit.mean()),
            "brier": float(((p - y) ** 2).mean()),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
            "reliability": table.to_dict(orient="records"),
        }
    paired["brier_gain"] = (paired.raw_home_probability - y) ** 2 - (
        paired.combined_home_probability - y
    ) ** 2
    paired["accuracy_gain"] = paired.combined_home_probability.ge(0.5).eq(y.astype(bool)).astype(
        float
    ) - paired.raw_home_probability.ge(0.5).eq(y.astype(bool)).astype(float)

    def metric_fn(sample: pd.DataFrame) -> dict[str, float]:
        return {
            "brier_gain": float(sample.brier_gain.mean()),
            "accuracy_gain": float(sample.accuracy_gain.mean()),
        }

    protocol = enrollment["protocol"]
    intervals = week_blocked_bootstrap(
        paired,
        metric_fn,
        samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["bootstrap_seed"]),
        confidence=float(protocol["confidence"]),
        metric_columns=["brier_gain", "accuracy_gain"],
    )
    disagreement = paired.combined_home_probability.ge(0.5).ne(paired.raw_home_probability.ge(0.5))
    result = {
        **coverage,
        "status": "complete",
        "results": metrics,
        "pushes": pushes,
        "disagreements": int(disagreement.sum()),
        "combined_correct_on_disagreements": int(
            paired.loc[disagreement, "accuracy_gain"].gt(0).sum()
        ),
        "paired_intervals": intervals.to_dict(orient="records"),
        "automatic_promotion": False,
    }
    atomic_json(result, _root(artifacts_root) / "report.json")
    return result

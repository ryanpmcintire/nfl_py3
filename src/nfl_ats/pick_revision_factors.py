from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.attribution_waterfall import family_contributions_from_ridge
from nfl_ats.market_decomposition import FAMILY_PHRASES, INTERCEPT_FAMILY
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.pick_refresh import MOVEMENT_POLICY_MODEL_ONLY
from nfl_ats.reporting import read_json
from nfl_ats.spread_explorer import load_feature_table_for_forecast

MATERIAL_FACTOR_POINTS = 0.01

MAX_NAMED_FACTORS = 2

RECONCILIATION_ATOL = 1e-6

_OVERLAY_FLIP_LABELS: tuple[tuple[str, str], ...] = (
    ("coach_fade_flip", "coach fade"),
    ("division_revenge_flip", "division revenge"),
    ("player_arrests_flip", "player arrests"),
    ("spread_gap_zone_flip", "spread-gap zone"),
)


@dataclass(frozen=True)
class FactorMove:
    game_id: str
    toward_team: str
    pick_team: str
    points: float
    phrases: tuple[str, ...]
    read_switched: bool
    flipped: bool
    flip_labels: tuple[str, ...]


def _points_number(points: float) -> str:

    size = abs(float(points))
    return f"{size:.1f}" if size >= 0.1 else f"{size:.2f}"


def _points_text(points: float) -> str:

    rendered = _points_number(points)
    return f"{rendered} point" if rendered in ("1.0", "1") else f"{rendered} points"


def _join_phrases(phrases: tuple[str, ...]) -> str:

    if len(phrases) == 1:
        return phrases[0]
    return " and ".join(phrases)


def factor_move_sentence(move: FactorMove) -> str:

    reason = _join_phrases(move.phrases)
    if move.read_switched:
        lead = (
            f"The computer's own read crossed over to {move.toward_team} -- a "
            f"{_points_number(move.points)}-point move, mostly on {reason}."
        )
    else:
        lead = (
            f"The computer's own read moved {_points_text(move.points)} toward "
            f"{move.toward_team}, mostly on {reason}."
        )
    if not move.flipped:
        return lead
    if not move.flip_labels:
        adjustment = "A situational adjustment plays"
    else:
        named = _join_phrases(move.flip_labels)
        noun = "adjustment plays" if len(move.flip_labels) == 1 else "adjustments play"
        adjustment = f"The {named} {noun}"
    return (
        f"{lead} {adjustment} this game against that read, so the card moved to {move.pick_team}."
    )


def _forecast_directories(artifacts_root: Path, season: int, week: int) -> list[Path]:

    root = artifacts_root / "margin_predictions"
    if not root.is_dir():
        return []
    prefix = f"{season}-week-{week:02d}-"
    return sorted(
        path
        for path in root.glob(f"{prefix}*")
        if (path / "metadata.json").is_file() and (path / "predictions.csv").is_file()
    )


def _metadata(path: Path) -> dict[str, Any]:

    payload = read_json(path / "metadata.json")
    return payload if isinstance(payload, dict) else {}


def _as_timestamp(value: Any) -> Any:

    return pd.to_datetime(value, utc=True, errors="coerce")


def _created_at(metadata: Mapping[str, Any]) -> Any:

    return _as_timestamp(metadata.get("created_at_utc"))


def _feature_table_digest(metadata: Mapping[str, Any]) -> str:

    provenance = metadata.get("provenance")
    table = provenance.get("feature_table") if isinstance(provenance, Mapping) else None
    if isinstance(table, Mapping):
        return str(table.get("sha256") or "")
    return ""


def _directory_for_feature_table(directories: list[Path], digest: str) -> Path | None:

    if not digest:
        return None
    for path in directories:
        if _feature_table_digest(_metadata(path)) == digest:
            return path
    return None


def _directory_before(directories: list[Path], moment: Any) -> Path | None:

    cutoff = _as_timestamp(moment)
    if pd.isna(cutoff):
        return None
    best: Path | None = None
    best_at: Any = None
    for path in directories:
        created = _created_at(_metadata(path))
        if pd.isna(created) or created > cutoff:
            continue
        if best_at is None or created > best_at:
            best, best_at = path, created
    return best


def _week_frame(path: Path, season: int, week: int) -> pd.DataFrame:

    frame = pd.read_csv(path / "predictions.csv")
    mask = (
        frame["method"].astype(str).eq("market_residual")
        & pd.to_numeric(frame["season"], errors="coerce").eq(season)
        & pd.to_numeric(frame["week"], errors="coerce").eq(week)
    )
    return frame.loc[mask].reset_index(drop=True)


def _contributions(model: Any, frame: pd.DataFrame) -> list[dict[str, float]] | None:

    columns = list(model.feature_columns)
    if any(column not in frame.columns for column in columns):
        return None
    return family_contributions_from_ridge(
        model.estimator, frame.loc[:, columns], feature_columns=columns
    )


def _served_adjustment(
    frame: pd.DataFrame, index: int, contributions: Mapping[str, float]
) -> float:

    stored = float(pd.to_numeric(frame.loc[index, "predicted_market_residual"], errors="coerce"))
    return stored - sum(contributions.values())


def _raw_side(probability: float) -> str:

    return "HOME" if probability >= 0.5 else "AWAY"


def _served_side(raw: str, flipped: bool) -> str:

    if not flipped:
        return raw
    return "AWAY" if raw == "HOME" else "HOME"


def _flip_labels(row: Mapping[str, Any]) -> tuple[str, ...]:

    return tuple(label for column, label in _OVERLAY_FLIP_LABELS if bool(row.get(column)))


def _named_phrases(delta: Mapping[str, float], total: float) -> tuple[str, ...]:

    sign = 1.0 if total >= 0.0 else -1.0
    candidates = [
        (family, value)
        for family, value in delta.items()
        if family != INTERCEPT_FAMILY and value * sign >= MATERIAL_FACTOR_POINTS
    ]
    candidates.sort(key=lambda item: abs(item[1]), reverse=True)
    return tuple(
        FAMILY_PHRASES.get(family, family.replace("_", " "))
        for family, _value in candidates[:MAX_NAMED_FACTORS]
    )


def _move_for_row(
    row: Mapping[str, Any],
    *,
    artifacts_root: Path,
    data_root: Path,
    models: dict[tuple[int, int], Any],
) -> FactorMove | None:

    season = int(row["season"])
    week = int(row["week"])
    game_id = str(row["game_id"])
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    if not home or not away:
        return None

    directories = _forecast_directories(artifacts_root, season, week)
    after_dir = _directory_for_feature_table(
        directories, str(row.get("feature_table_sha256") or "")
    )
    before_dir = _directory_before(directories, row.get("original_recorded_at_utc"))
    if after_dir is None or before_dir is None or after_dir == before_dir:
        return None

    key = (season, week)
    if key not in models:
        metadata = _metadata(after_dir)
        features = load_feature_table_for_forecast(metadata, data_root)
        _target, fitted = fit_margin_models_for_week(
            features,
            season=season,
            week=week,
            regressor=str(metadata.get("regressor") or "ridge"),
            min_train_games=int(metadata.get("min_train_games", 500)),
            feature_profile=str(metadata.get("feature_profile") or "base"),  # type: ignore[arg-type]
            ridge_alpha=float(metadata.get("ridge_alpha", 10.0)),
            methods=("market_residual",),
        )
        models[key] = fitted["market_residual"]
    model = models[key]

    before = _week_frame(before_dir, season, week)
    after = _week_frame(after_dir, season, week)
    if before.empty or after.empty:
        return None
    before_index = before.index[before["game_id"].astype(str).eq(game_id)]
    after_index = after.index[after["game_id"].astype(str).eq(game_id)]
    if len(before_index) != 1 or len(after_index) != 1:
        return None

    flipped = bool(row.get("composed_overlay_flip"))
    before_raw = _raw_side(float(before.loc[before_index[0], "home_cover_probability"]))
    after_raw = _raw_side(float(after.loc[after_index[0], "home_cover_probability"]))
    if _served_side(before_raw, flipped) != str(row.get("previous_pick_side") or ""):
        return None
    if _served_side(after_raw, flipped) != str(row.get("new_pick_side") or ""):
        return None

    before_read = _contributions(model, before)
    after_read = _contributions(model, after)
    if before_read is None or after_read is None:
        return None
    before_contributions = before_read[before_index[0]]
    after_contributions = after_read[after_index[0]]
    before_adjustment = _served_adjustment(before, before_index[0], before_contributions)
    after_adjustment = _served_adjustment(after, after_index[0], after_contributions)
    if abs(after_adjustment - before_adjustment) > RECONCILIATION_ATOL:
        return None
    families = set(before_contributions) | set(after_contributions)
    delta = {
        family: float(after_contributions.get(family, 0.0))
        - float(before_contributions.get(family, 0.0))
        for family in families
    }
    total = sum(delta.values())
    if abs(total) < MATERIAL_FACTOR_POINTS:
        return None
    phrases = _named_phrases(delta, total)
    if not phrases:
        return None

    new_side = str(row.get("new_pick_side") or "")
    return FactorMove(
        game_id=game_id,
        toward_team=home if total > 0.0 else away,
        pick_team=home if new_side == "HOME" else away,
        points=abs(total),
        phrases=phrases,
        read_switched=before_raw != after_raw,
        flipped=flipped,
        flip_labels=_flip_labels(row),
    )


def model_only_factor_moves(
    revisions: pd.DataFrame, *, artifacts_root: Path, data_root: Path
) -> dict[str, FactorMove]:

    if revisions.empty or "movement_policy" not in revisions.columns:
        return {}
    selected = revisions.loc[
        revisions["movement_policy"].astype(str).eq(MOVEMENT_POLICY_MODEL_ONLY)
    ]
    if selected.empty:
        return {}
    models: dict[tuple[int, int], Any] = {}
    moves: dict[str, FactorMove] = {}
    for _, row in selected.iterrows():
        try:
            move = _move_for_row(
                dict(row), artifacts_root=artifacts_root, data_root=data_root, models=models
            )
        except Exception:
            move = None
        if move is not None:
            moves[move.game_id] = move
    return moves


__all__ = [
    "FactorMove",
    "factor_move_sentence",
    "model_only_factor_moves",
]

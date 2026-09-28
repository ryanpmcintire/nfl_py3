from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.published_picks import (
    PUBLISHED_PICKS_FILENAME,
    frozen_picks,
    load_published_picks,
)


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise DataContractError("Published probability history requires a timezone-aware now")
    return now.astimezone(UTC)


def _require_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataContractError(
            f"Published probability history is missing columns: {', '.join(missing)}"
        )
    empty = [column for column in columns if frame[column].isna().any()]
    if empty:
        raise DataContractError(
            f"Published probability history has null values in: {', '.join(empty)}"
        )


def _empty_history(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    for column in (
        "pick_side",
        "decision_home_spread",
        "p_served",
        "recorded_home_probability",
        "published_probability_artifact",
    ):
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)


def load_frozen_probability_history(
    artifacts_root: Path,
    *,
    season: int,
    now: datetime | None = None,
) -> pd.DataFrame:
    instant = _utc_now(now)
    published = load_published_picks(artifacts_root)
    if published.empty:
        return _empty_history(published)
    season_number = pd.to_numeric(published["season"], errors="coerce")
    history = published.loc[season_number.eq(int(season))].copy()
    if history.empty:
        return _empty_history(published)
    _require_values(
        history,
        (
            "published_at_utc",
            "season",
            "week",
            "game_id",
            "away_team",
            "home_team",
            "kickoff",
            "pick_deadline_utc",
            "pick_team",
            "market_spread",
            "displayed_score",
            "source",
        ),
    )
    text_columns = ("game_id", "away_team", "home_team", "pick_team", "source")
    blank = [
        column for column in text_columns if history[column].astype(str).str.strip().eq("").any()
    ]
    if blank:
        raise DataContractError(
            f"Published probability history has blank values in: {', '.join(blank)}"
        )
    for column in ("published_at_utc", "kickoff", "pick_deadline_utc"):
        history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    if history[["published_at_utc", "kickoff", "pick_deadline_utc"]].isna().any().any():
        raise DataContractError("Published probability history has invalid timestamps")
    locked_rows: list[pd.Series] = []
    for _, group in history.groupby("game_id", sort=False):
        deadline = group["pick_deadline_utc"].max()
        if deadline > pd.Timestamp(instant):
            continue
        eligible = group.loc[group["published_at_utc"].le(deadline)].sort_values("published_at_utc")
        if eligible.empty:
            game_id = str(group.iloc[-1]["game_id"])
            raise DataContractError(
                f"Frozen published probability has no pre-deadline row for game_id={game_id}"
            )
        locked_rows.append(eligible.iloc[-1])
    if locked_rows:
        locked = pd.DataFrame(locked_rows)
        locked_spread = pd.to_numeric(locked["market_spread"], errors="coerce")
        locked_probability = pd.to_numeric(locked["displayed_score"], errors="coerce")
        invalid_spread = ~np.isfinite(locked_spread.to_numpy())
        if invalid_spread.any():
            bad = locked.loc[invalid_spread, "game_id"].astype(str).tolist()
            raise DataContractError(
                f"Frozen published probability history has an invalid spread: {', '.join(bad)}"
            )
        invalid_probability = (
            ~np.isfinite(locked_probability.to_numpy())
            | ~locked_probability.between(0.0, 1.0).to_numpy()
        )
        if invalid_probability.any():
            bad = locked.loc[invalid_probability, "game_id"].astype(str).tolist()
            raise DataContractError(
                f"Frozen published probability history has an invalid probability: {', '.join(bad)}"
            )
    frozen = frozen_picks(artifacts_root, season=int(season), now=instant)
    if not frozen:
        return _empty_history(published)
    selected: list[pd.Series] = []
    for game_id, pick in frozen.items():
        matches = history.loc[
            history["game_id"].astype(str).eq(game_id)
            & history["published_at_utc"].eq(pick.published_at_utc)
        ]
        if len(matches) != 1:
            raise DataContractError(
                f"Frozen published probability row is ambiguous for game_id={game_id}"
            )
        selected.append(matches.iloc[0])
    result = pd.DataFrame(selected).reset_index(drop=True)
    if result["game_id"].astype(str).duplicated().any():
        raise DataContractError("Frozen published probability history has duplicate game IDs")
    if result["home_team"].astype(str).eq(result["away_team"].astype(str)).any():
        raise DataContractError("Frozen published probability history has identical teams")
    pick_home = result["pick_team"].astype(str).eq(result["home_team"].astype(str))
    pick_away = result["pick_team"].astype(str).eq(result["away_team"].astype(str))
    if (~(pick_home | pick_away)).any():
        bad = result.loc[~(pick_home | pick_away), "game_id"].astype(str).tolist()
        raise DataContractError(f"Frozen pick team does not match game identity: {', '.join(bad)}")
    if (result["published_at_utc"] > result["pick_deadline_utc"]).any():
        raise DataContractError("Frozen published probability was recorded after its deadline")
    if (result["pick_deadline_utc"] > result["kickoff"]).any():
        raise DataContractError("Frozen pick deadline is after kickoff")
    if (result["pick_deadline_utc"] > pd.Timestamp(instant)).any():
        raise DataContractError("Frozen probability history includes an open pick")
    spread = pd.to_numeric(result["market_spread"], errors="coerce").astype(float)
    probability = pd.to_numeric(result["displayed_score"], errors="coerce").astype(float)
    if not np.isfinite(spread.to_numpy()).all():
        raise DataContractError("Frozen published probability history has an invalid spread")
    if not np.isfinite(probability.to_numpy()).all() or not probability.between(0.0, 1.0).all():
        raise DataContractError("Frozen published probability history has an invalid probability")
    result["pick_side"] = np.where(pick_home, "HOME", "AWAY")
    result["decision_home_spread"] = spread
    result["p_served"] = probability
    result["recorded_home_probability"] = probability.where(pick_home, 1.0 - probability)
    result["published_probability_artifact"] = str(
        Path("clv_ledger") / PUBLISHED_PICKS_FILENAME
    ).replace("\\", "/")
    return result.sort_values(["season", "week", "kickoff", "game_id"], kind="stable").reset_index(
        drop=True
    )

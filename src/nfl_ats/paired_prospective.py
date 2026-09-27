from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from .clv import week_blocked_bootstrap
from .data import DataContractError
from .prospective_scoring import DECISION_GRADE

PAIR_KEYS = ("season", "week", "game_id")
PAIR_VALUES = (
    "pick_side",
    "decision_home_spread",
    "recorded_at_utc",
    "kickoff",
    f"correct_at_{DECISION_GRADE}",
)
PAIRED_METRIC = "decision_line_accuracy_delta"


def _strict_utc(series: pd.Series, *, arm: str, field: str) -> pd.Series:
    values = []
    for value in series:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataContractError(f"{arm} paired decisions contain invalid {field}") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise DataContractError(f"{arm} paired decisions need timezone-aware {field}")
        values.append(timestamp.tz_convert("UTC"))
    return pd.Series(values, index=series.index, dtype="datetime64[ns, UTC]")


def paired_decision_rows(challenger: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    columns = [*PAIR_KEYS, *PAIR_VALUES]
    status_column = f"status_at_{DECISION_GRADE}"
    frames = []
    for name, frame in (("challenger", challenger), ("active", active)):
        if frame.empty:
            frames.append(pd.DataFrame(columns=columns))
            continue
        missing = {*columns, status_column}.difference(frame.columns)
        if missing:
            raise DataContractError(f"{name} paired decisions missing columns: {sorted(missing)}")
        if frame[list(PAIR_KEYS)].isna().any().any():
            raise DataContractError(f"{name} paired decisions contain missing identities")
        if frame.duplicated(list(PAIR_KEYS)).any():
            raise DataContractError(f"{name} paired decisions contain duplicate identities")
        selected = frame.loc[frame[status_column] == "settled", columns].copy()
        correct_column = f"correct_at_{DECISION_GRADE}"
        correct = pd.to_numeric(selected[correct_column], errors="coerce")
        if not correct.isin([0.0, 1.0]).all():
            raise DataContractError(f"{name} settled paired decisions need binary correctness")
        selected[correct_column] = correct.astype(float)
        if selected["decision_home_spread"].map(pd.api.types.is_bool).any():
            raise DataContractError(f"{name} paired decisions need finite numeric spreads")
        spreads = pd.to_numeric(selected["decision_home_spread"], errors="coerce")
        if not spreads.map(math.isfinite).all():
            raise DataContractError(f"{name} paired decisions need finite numeric spreads")
        selected["decision_home_spread"] = spreads.astype(float)
        recorded = _strict_utc(selected["recorded_at_utc"], arm=name, field="recorded_at_utc")
        kickoff = _strict_utc(selected["kickoff"], arm=name, field="kickoff")
        if not (recorded < kickoff).all():
            raise DataContractError(
                f"{name} paired decisions require recorded_at_utc before kickoff"
            )
        selected["recorded_at_utc"] = recorded.map(lambda value: value.isoformat())
        selected["kickoff"] = kickoff.map(lambda value: value.isoformat())
        frames.append(selected)
    paired = frames[0].merge(
        frames[1],
        on=list(PAIR_KEYS),
        how="inner",
        suffixes=("_challenger", "_active"),
        validate="one_to_one",
    )
    if not paired.empty and not paired["kickoff_challenger"].equals(paired["kickoff_active"]):
        raise DataContractError("paired decisions contain conflicting kickoff timestamps")
    paired[PAIRED_METRIC] = (
        paired[f"correct_at_{DECISION_GRADE}_challenger"]
        - paired[f"correct_at_{DECISION_GRADE}_active"]
    )
    return paired.sort_values(list(PAIR_KEYS)).reset_index(drop=True)


def paired_decisions_fingerprint(paired: pd.DataFrame) -> str:
    records: list[dict[str, Any]] = []
    for row in paired.sort_values(list(PAIR_KEYS)).to_dict(orient="records"):
        record: dict[str, Any] = {
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_id": str(row["game_id"]),
        }
        for arm in ("challenger", "active"):
            for column in PAIR_VALUES:
                key = f"{column}_{arm}"
                value = row[key]
                record[key] = (
                    float(value).hex()
                    if column in ("decision_home_spread", f"correct_at_{DECISION_GRADE}")
                    else str(value)
                )
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_prospective_report(
    challenger: pd.DataFrame, active: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    paired = paired_decision_rows(challenger, active)
    uncertainty = (
        week_blocked_bootstrap(
            paired[[*PAIR_KEYS, PAIRED_METRIC]],
            metric_fn=lambda frame: {PAIRED_METRIC: float(frame[PAIRED_METRIC].mean())},
            block="week",
            samples=samples,
            seed=seed,
        ).to_dict(orient="records")
        if not paired.empty
        else []
    )
    return {
        "reference_entrant": "active_model",
        "paired_games": len(paired),
        "paired_decisions_sha256": paired_decisions_fingerprint(paired),
        "uncertainty": uncertainty,
    }

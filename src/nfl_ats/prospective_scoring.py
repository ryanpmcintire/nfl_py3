from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    VALID_BET_SIDES,
    VALID_PICK_SIDES,
    pick_correct,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.provenance import sha256_file
from nfl_ats.recorder_override import replace_week_rows
from nfl_ats.settlement import LEDGERS, Arm, LedgerSpec

SETTLEMENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "kickoff",
    "recorded_at_utc",
    "pick_side",
    "decision_home_spread",
)

DECISION_GRADE = "decision_line"
CLOSE_GRADE = "close_line"

TOTAL_CONDITIONED_LATTICE_LEDGER = LedgerSpec(
    key="total_conditioned_lattice",
    relative_path="prospective/total_conditioned_lattice_decisions.parquet",
    arms=(Arm("challenger", "challenger_pick_side", line_column="spread_line"),),
)
PROSPECTIVE_LEDGER_SPECS = (*LEDGERS, TOTAL_CONDITIONED_LATTICE_LEDGER)

DEDICATED_CHALLENGER_SETTLEMENT_ARMS = {
    "crew_tilt_refresh_v1": ("crew_tilt_refresh", "crew_tilt"),
    "half_line_2h_underdog_refresh_v1": ("half_line_refresh", "half_line_2h_underdog"),
    "handle_follow_refresh_off_incumbent": ("handle_follow_refresh", "off_arm"),
    "inactives_refresh_v1": ("inactives_refresh", "inactives"),
    "injury_signal_refresh_tilt": ("injury_signal_refresh", "injury_tilt"),
    "late_week_leader_median_follow_v1": (
        "late_week_move_follow_refresh",
        "movement_follow",
    ),
    "late_week_move_follow_refresh_v1": (
        "late_week_move_follow_refresh",
        "equal_book_off_arm",
    ),
    "model_only_refresh_incumbent": ("pick_revisions", "model_only_off_arm"),
    "nflcom_friday_refresh_out2_starters_v1": ("nflcom_friday_refresh", "nflcom_starters"),
    "specialist_absence_fade_refresh_v1": ("specialist_absence_fade_refresh", "specialist_fade"),
    "total_conditioned_key_number_lattice_v1": ("total_conditioned_lattice", "challenger"),
}

DEDICATED_ARM_IDENTITIES = {
    ("late_week_move_follow_refresh", "equal_book_off_arm"): (
        "challenger_id",
        "late_week_move_follow_refresh_v1",
    ),
    ("late_week_move_follow_refresh", "movement_follow"): (
        "served_challenger_id",
        "late_week_leader_median_follow_v1",
    ),
}

UNSUPPORTED_PROSPECTIVE_CHALLENGERS = frozenset(
    {
        "rookie_crew_underdog_off_incumbent",
        "best_pick_sunday_renomination",
        "tiebreaker_lattice_centre",
        "tiebreaker_low_side_shade",
        "totals_served_method",
    }
)


class InvalidProspectiveArmError(DataContractError):
    def __init__(self, *, source: str, diagnostics: dict[str, Any]) -> None:
        invalid_rows = list(diagnostics["invalid_rows"])
        self.source = source
        self.diagnostics = diagnostics
        self.invalid_row_count = len(invalid_rows)
        self.invalid_rows = invalid_rows
        details = "; ".join(
            f"{row['game_id']} ({', '.join(row['reasons'])})" for row in invalid_rows
        )
        super().__init__(f"{source} has {len(invalid_rows)} invalid selected rows: {details}")


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _settlement_status(margin: pd.Series) -> pd.Series:

    return pd.Series(
        np.where(margin.isna(), "pending", np.where(margin.eq(0.0), "push", "settled")),
        index=margin.index,
        dtype=object,
    )


def assert_recorded_before_kickoff(decisions: pd.DataFrame) -> None:

    if decisions.empty:
        return
    recorded = _utc_series(decisions["recorded_at_utc"])
    kickoff = _utc_series(decisions["kickoff"])
    unknown = recorded.isna() | kickoff.isna()
    if unknown.any():
        examples = ", ".join(decisions.loc[unknown, "game_id"].astype(str).tolist()[:5])
        raise DataContractError(
            f"Prospective decisions cannot prove pre-kickoff timing for: {examples}"
        )
    late = recorded.ge(kickoff)
    if late.any():
        examples = ", ".join(decisions.loc[late, "game_id"].astype(str).tolist()[:5])
        raise DataContractError(
            "Prospective decisions were recorded at or after kickoff and cannot be "
            f"scored: {examples}"
        )


def _strict_utc_series(values: pd.Series, *, field: str) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    invalid: list[object] = []
    for index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError
            parsed.append(timestamp.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError):
            invalid.append(index)
    if invalid:
        raise DataContractError(
            f"Prospective ledger has invalid {field} values at rows: "
            + ", ".join(str(index) for index in invalid[:5])
        )
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def dedicated_challenger_settlement_arm(
    challenger_id: str,
) -> tuple[LedgerSpec, Arm] | None:
    keys = DEDICATED_CHALLENGER_SETTLEMENT_ARMS.get(challenger_id)
    if keys is None:
        return None
    ledger_key, arm_name = keys
    spec = next(spec for spec in PROSPECTIVE_LEDGER_SPECS if spec.key == ledger_key)
    arm = next(arm for arm in spec.arms if arm.label == arm_name)
    return spec, arm


def adapt_settlement_arm_for_prospective_scoring(
    decisions: pd.DataFrame,
    *,
    spec: LedgerSpec,
    arm: Arm,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    source = f"{spec.key}:{arm.label}"
    source_rows = len(decisions)
    identity = DEDICATED_ARM_IDENTITIES.get((spec.key, arm.label))
    if spec.row_keys != ("game_id",) or arm.game_column != "game_id":
        raise DataContractError("Prospective scoring adapters require one game_id row per decision")
    if arm.split_by is not None or arm.filter_column is not None:
        raise DataContractError(
            "Prospective scoring adapters do not support split or filtered settlement arms"
        )
    deadline_column = next(
        (column for column in spec.deadline_columns if column in decisions.columns),
        None,
    )
    if deadline_column is None:
        raise DataContractError(
            "Prospective ledger is missing a supported deadline column: "
            + ", ".join(spec.deadline_columns)
        )
    required = {
        "game_id",
        "season",
        "week",
        spec.order_column,
        "kickoff",
        arm.pick_column,
        arm.line_column,
    }
    if identity is not None:
        required.add(identity[0])
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise DataContractError(f"Prospective ledger is missing columns: {', '.join(missing)}")

    identity_diagnostics: dict[str, Any] = {}
    if identity is not None:
        identity_column, identity_value = identity
        identity_values = decisions[identity_column].astype("string").str.strip()
        identity_matches = identity_values.eq(identity_value).fillna(False)
        excluded = decisions.loc[~identity_matches]
        excluded_games = excluded["game_id"].astype("string").str.strip().dropna().nunique()
        identity_diagnostics = {
            "identity_column": identity_column,
            "identity_value": identity_value,
            "identity_matched_rows": int(identity_matches.sum()),
            "discarded_identity_rows": int((~identity_matches).sum()),
            "discarded_identity_games": int(excluded_games),
        }
        decisions = decisions.loc[identity_matches].copy()
    if decisions.empty:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "source_rows": source_rows,
                    "discarded_late_revisions": 0,
                    "discarded_games_without_predeadline_rows": 0,
                    "selected_rows": 0,
                    "deadline_column": deadline_column,
                    **identity_diagnostics,
                }
            )
        return pd.DataFrame(columns=SETTLEMENT_REQUIRED_COLUMNS)

    game_ids = decisions["game_id"].astype("string").str.strip()
    invalid_identity = game_ids.isna() | game_ids.eq("")
    if invalid_identity.any():
        examples = ", ".join(decisions.loc[invalid_identity, "game_id"].astype(str).tolist()[:5])
        raise DataContractError(f"Prospective ledger has invalid game identity for: {examples}")

    recorded = _strict_utc_series(decisions[spec.order_column], field=spec.order_column)
    kickoff = _strict_utc_series(decisions["kickoff"], field="kickoff")
    deadline = _strict_utc_series(decisions[deadline_column], field=deadline_column)
    eligible = recorded.lt(deadline)
    eligible_rows = decisions.loc[eligible].copy()
    eligible_rows["_prospective_order"] = recorded.loc[eligible]
    selected = (
        eligible_rows.sort_values([*spec.row_keys, "_prospective_order"])
        .drop_duplicates(list(spec.row_keys), keep="last")
        .drop(columns="_prospective_order")
    )
    selected_ids = set(selected["game_id"].astype("string").str.strip())
    missing_games = set(game_ids) - selected_ids

    if selected.empty:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "source_rows": source_rows,
                    "discarded_late_revisions": int((~eligible).sum()),
                    "discarded_games_without_predeadline_rows": len(missing_games),
                    "selected_rows": 0,
                    "deadline_column": deadline_column,
                    **identity_diagnostics,
                }
            )
        return pd.DataFrame(columns=SETTLEMENT_REQUIRED_COLUMNS)

    adapted = pd.DataFrame(
        {
            "game_id": selected["game_id"].astype("string").str.strip(),
            "season": pd.to_numeric(selected["season"], errors="coerce"),
            "week": pd.to_numeric(selected["week"], errors="coerce"),
            "kickoff": kickoff.loc[selected.index],
            "recorded_at_utc": recorded.loc[selected.index],
            "pick_side": selected[arm.pick_column].astype("string").str.strip(),
            "decision_home_spread": pd.to_numeric(selected[arm.line_column], errors="coerce"),
        }
    )
    known_pick_rows = adapted["pick_side"].notna() & adapted["pick_side"].ne("")
    unknown_picks = sorted(
        set(adapted.loc[known_pick_rows, "pick_side"].astype(str)) - VALID_PICK_SIDES
    )
    if unknown_picks:
        raise DataContractError(f"{source} contains invalid pick_side values: {unknown_picks}")
    if adapted["game_id"].duplicated().any():
        duplicates = sorted(
            adapted.loc[adapted["game_id"].duplicated(), "game_id"].astype(str).unique()
        )
        raise DataContractError(f"{source} contains duplicate game_id rows: {duplicates}")
    adapted = adapted.reset_index(drop=True)
    assert_recorded_before_kickoff(adapted)
    finite_season = pd.Series(np.isfinite(adapted["season"]), index=adapted.index)
    finite_week = pd.Series(np.isfinite(adapted["week"]), index=adapted.index)
    finite_spread = pd.Series(np.isfinite(adapted["decision_home_spread"]), index=adapted.index)
    integral_season = finite_season & adapted["season"].mod(1).eq(0)
    integral_week = finite_week & adapted["week"].mod(1).eq(0)
    invalid_rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(adapted.iterrows()):
        reasons: list[str] = []
        if not finite_season.iat[position]:
            reasons.append("non_finite_season")
        elif not integral_season.iat[position]:
            reasons.append("non_integral_season")
        if not finite_week.iat[position]:
            reasons.append("non_finite_week")
        elif not integral_week.iat[position]:
            reasons.append("non_integral_week")
        if not finite_spread.iat[position]:
            reasons.append("non_finite_decision_home_spread")
        if pd.isna(row["pick_side"]) or not str(row["pick_side"]).strip():
            reasons.append("missing_pick_side")
        if reasons:
            invalid_rows.append({"game_id": str(row["game_id"]), "reasons": reasons})
    selection_diagnostics = {
        "source_rows": source_rows,
        "discarded_late_revisions": int((~eligible).sum()),
        "discarded_games_without_predeadline_rows": len(missing_games),
        "selected_rows": len(adapted),
        "deadline_column": deadline_column,
        **identity_diagnostics,
    }
    if invalid_rows:
        selection_diagnostics.update(
            {"invalid_row_count": len(invalid_rows), "invalid_rows": invalid_rows}
        )
    if diagnostics is not None:
        diagnostics.update(selection_diagnostics)
    if invalid_rows:
        raise InvalidProspectiveArmError(source=source, diagnostics=selection_diagnostics)
    adapted["season"] = adapted["season"].astype(int)
    adapted["week"] = adapted["week"].astype(int)
    return adapted


def settle_prospective_picks(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    close_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:

    missing = sorted(set(SETTLEMENT_REQUIRED_COLUMNS).difference(decisions.columns))
    if missing:
        raise DataContractError(f"Prospective decisions are missing columns: {', '.join(missing)}")
    if decisions["game_id"].duplicated().any():
        raise DataContractError("Prospective decisions contain duplicate game rows")
    unknown_picks = sorted(set(decisions["pick_side"].astype(str)) - VALID_PICK_SIDES)
    if unknown_picks:
        raise ValueError(f"Prospective decisions use unsupported pick sides: {unknown_picks}")
    if "bet_side" in decisions.columns:
        unknown_bets = sorted(set(decisions["bet_side"].astype(str)) - VALID_BET_SIDES)
        if unknown_bets:
            raise ValueError(f"Prospective decisions use unsupported bet sides: {unknown_bets}")
    outcome_missing = sorted({"game_id", "result"}.difference(outcomes.columns))
    if outcome_missing:
        raise DataContractError(
            f"Prospective outcomes are missing columns: {', '.join(outcome_missing)}"
        )
    assert_recorded_before_kickoff(decisions)

    scored = decisions.copy()
    scored["game_id"] = scored["game_id"].astype(str)
    results = outcomes.loc[:, ["game_id", "result"]].drop_duplicates("game_id").copy()
    results["game_id"] = results["game_id"].astype(str)
    results["result"] = pd.to_numeric(results["result"], errors="coerce")
    scored = scored.merge(results, on="game_id", how="left")

    scored["close_home_spread"] = np.nan
    scored["close_source"] = pd.NA
    if close_reference is not None and not close_reference.empty:
        columns = [
            column
            for column in ("game_id", "close_home_spread", "close_source")
            if column in close_reference.columns
        ]
        reference = close_reference.loc[:, columns].drop_duplicates("game_id").copy()
        reference["game_id"] = reference["game_id"].astype(str)
        scored = scored.drop(columns=["close_home_spread", "close_source"]).merge(
            reference, on="game_id", how="left"
        )
        for column in ("close_home_spread", "close_source"):
            if column not in scored.columns:
                scored[column] = np.nan if column == "close_home_spread" else pd.NA

    pick_home = scored["pick_side"].astype(str).eq("HOME")
    for grade, line in (
        (DECISION_GRADE, pd.to_numeric(scored["decision_home_spread"], errors="coerce")),
        (CLOSE_GRADE, pd.to_numeric(scored["close_home_spread"], errors="coerce")),
    ):
        margin = scored["result"] - line
        scored[f"ats_margin_at_{grade}"] = margin
        scored[f"correct_at_{grade}"] = pick_correct(pick_home, margin).where(margin.notna())
        scored[f"status_at_{grade}"] = _settlement_status(margin)
    return scored.reset_index(drop=True)


def _column(frame: pd.DataFrame, name: str, *, fill: Any) -> pd.Series:

    if name in frame.columns:
        return frame[name]
    return pd.Series(fill, index=frame.index)


def _grade_summary(settled: pd.DataFrame, grade: str) -> dict[str, Any]:
    correct = pd.to_numeric(
        _column(settled, f"correct_at_{grade}", fill=float("nan")), errors="coerce"
    )
    status = _column(settled, f"status_at_{grade}", fill="pending")
    resolved = correct.dropna()
    games = len(resolved)
    accuracy = float(resolved.mean()) if games else float("nan")
    return {
        "grade": grade,
        "games": games,
        "correct": int(resolved.sum()) if games else 0,
        "accuracy": accuracy,
        "vs_coin_flip": accuracy - 0.5 if games else float("nan"),
        "pushes": int((status == "push").sum()),
        "pending": int((status == "pending").sum()),
    }


def prospective_accuracy(settled: pd.DataFrame) -> dict[str, Any]:

    summary: dict[str, Any] = {
        "decisions": len(settled),
        "forced_picks": {
            DECISION_GRADE: _grade_summary(settled, DECISION_GRADE),
            CLOSE_GRADE: _grade_summary(settled, CLOSE_GRADE),
        },
    }
    if "is_best_pick" in settled.columns:
        best = settled.loc[settled["is_best_pick"].fillna(False).astype(bool)]
        summary["best_pick"] = {
            "weeks_nominated": len(best),
            DECISION_GRADE: _grade_summary(best, DECISION_GRADE),
            CLOSE_GRADE: _grade_summary(best, CLOSE_GRADE),
        }
    if "bet_side" in settled.columns:
        bets = settled.loc[settled["bet_side"].astype(str).ne("PASS")]
        summary["paper_bets"] = {
            "bets": len(bets),
            DECISION_GRADE: _grade_summary(bets, DECISION_GRADE),
            CLOSE_GRADE: _grade_summary(bets, CLOSE_GRADE),
        }
    return summary


def prospective_accuracy_metrics(settled: pd.DataFrame) -> dict[str, float]:

    at_decision = pd.to_numeric(settled[f"correct_at_{DECISION_GRADE}"], errors="coerce").dropna()
    at_close = pd.to_numeric(settled[f"correct_at_{CLOSE_GRADE}"], errors="coerce").dropna()
    both = settled.dropna(subset=[f"correct_at_{DECISION_GRADE}", f"correct_at_{CLOSE_GRADE}"])
    decision_accuracy = float(at_decision.mean()) if len(at_decision) else float("nan")
    return {
        "decision_line_accuracy": decision_accuracy,
        "close_line_accuracy": float(at_close.mean()) if len(at_close) else float("nan"),
        "decision_vs_coin_flip": decision_accuracy - 0.5,
        "decision_minus_close": (
            float(
                both[f"correct_at_{DECISION_GRADE}"].mean()
                - both[f"correct_at_{CLOSE_GRADE}"].mean()
            )
            if len(both)
            else float("nan")
        ),
    }


def prospective_week_summary(settled: pd.DataFrame) -> pd.DataFrame:

    if settled.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "picks",
                "settled",
                "pushes",
                "correct",
                "accuracy",
                "close_correct",
                "close_accuracy",
                "best_pick_game_id",
                "best_pick_correct",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (season, week), group in settled.groupby(["season", "week"], sort=True):
        decision = _grade_summary(group, DECISION_GRADE)
        close = _grade_summary(group, CLOSE_GRADE)
        flags = _column(group, "is_best_pick", fill=False).fillna(False).astype(bool)
        best = group.loc[flags]
        best_correct = pd.to_numeric(
            _column(best, f"correct_at_{DECISION_GRADE}", fill=float("nan")), errors="coerce"
        ).dropna()
        rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "picks": len(group),
                "settled": decision["games"],
                "pushes": decision["pushes"],
                "correct": decision["correct"],
                "accuracy": decision["accuracy"],
                "close_correct": close["correct"],
                "close_accuracy": close["accuracy"],
                "best_pick_game_id": (str(best["game_id"].iloc[0]) if not best.empty else None),
                "best_pick_correct": (
                    float(best_correct.iloc[0]) if len(best_correct) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


CHALLENGER_DECISION_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "challenger_id",
    "config_fingerprint",
    "source_artifact",
    "source_sha256",
    "forecast_created_at_utc",
    "feature_profile",
    "feature_table_sha256",
    "game_id",
    "season",
    "week",
    "kickoff",
    "away_team",
    "home_team",
    "pick_side",
    "bet_side",
    "decision_home_spread",
    "edge",
)

ACTIVE_CHALLENGER_STATUS = "ACTIVE_PROSPECTIVE"

CONFIG_FINGERPRINT_KEYS: tuple[str, ...] = (
    "method",
    "target",
    "regressor",
    "ridge_alpha",
    "calibration_method",
    "feature_profile",
    "min_edge",
    "min_train_games",
    "feature_table",
)


def challenger_registry_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "challengers.json"


def challenger_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "challenger_decisions.parquet"


def _normalise_config(config: Mapping[str, Any]) -> dict[str, Any]:

    normalised: dict[str, Any] = {}
    for key in CONFIG_FINGERPRINT_KEYS:
        value = config.get(key)
        if value is None:
            normalised[key] = None
        elif key == "min_train_games":
            normalised[key] = int(value)
        elif key in ("ridge_alpha", "min_edge"):
            normalised[key] = round(float(value), 10)
        elif key == "feature_table":
            normalised[key] = PurePath(str(value).replace("\\", "/")).name
        else:
            normalised[key] = str(value)
    return normalised


def config_fingerprint(config: Mapping[str, Any]) -> str:

    payload = json.dumps(_normalise_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_challenger_registry(artifacts_root: Path) -> dict[str, Any]:
    path = challenger_registry_path(artifacts_root)
    if not path.is_file():
        raise FileNotFoundError(f"No prospective challenger registry at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("challengers"), list):
        raise DataContractError(f"Malformed prospective challenger registry: {path}")
    return payload


def find_challenger(artifacts_root: Path, challenger_id: str) -> dict[str, Any]:
    registry = load_challenger_registry(artifacts_root)
    entries = [
        entry
        for entry in registry["challengers"]
        if isinstance(entry, dict) and str(entry.get("challenger_id")) == challenger_id
    ]
    if not entries:
        known = ", ".join(
            str(entry.get("challenger_id"))
            for entry in registry["challengers"]
            if isinstance(entry, dict)
        )
        raise ValueError(f"Unknown challenger {challenger_id!r}; registered: {known}")
    if len(entries) > 1:
        raise DataContractError(f"Challenger {challenger_id!r} is registered more than once")
    return entries[0]


def active_challenger_ids(artifacts_root: Path) -> list[str]:

    registry = load_challenger_registry(artifacts_root)
    return [
        str(entry["challenger_id"])
        for entry in registry["challengers"]
        if isinstance(entry, dict) and entry.get("status") == ACTIVE_CHALLENGER_STATUS
    ]


def load_challenger_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = challenger_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(CHALLENGER_DECISION_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(CHALLENGER_DECISION_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"Challenger ledger is missing columns: {', '.join(missing)}")
    if ledger.duplicated(subset=["challenger_id", "game_id"]).any():
        raise DataContractError(f"Challenger ledger contains duplicate rows: {path}")
    return ledger[list(CHALLENGER_DECISION_COLUMNS)]


def artifact_model_config(metadata: Mapping[str, Any]) -> dict[str, Any]:

    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    feature_table = provenance.get("feature_table")
    feature_table = feature_table if isinstance(feature_table, dict) else {}
    method = str(metadata.get("ats_method", "market_residual"))
    return {
        "method": method,
        "target": method,
        "regressor": metadata.get("regressor"),
        "ridge_alpha": metadata.get("ridge_alpha"),
        "calibration_method": metadata.get("calibration_method", "none"),
        "feature_profile": metadata.get("feature_profile"),
        "min_edge": metadata.get("min_edge"),
        "min_train_games": metadata.get("min_train_games"),
        "feature_table": feature_table.get("path"),
        "feature_table_sha256": feature_table.get("sha256"),
    }


def find_challenger_artifact(
    artifacts_root: Path, entry: Mapping[str, Any], *, season: int, week: int
) -> Path | None:

    root = artifacts_root / "margin_predictions"
    if not root.is_dir():
        return None
    expected = config_fingerprint(entry.get("model", {}))
    prefix = f"{season}-week-{week:02d}-"
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)),
        reverse=True,
    )
    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(metadata, dict):
            continue
        if config_fingerprint(artifact_model_config(metadata)) == expected:
            return candidate
    return None


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_challenger_decisions(
    artifacts_root: Path,
    challenger_id: str,
    artifact_directory: Path,
    *,
    now: datetime | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    entry = find_challenger(artifacts_root, challenger_id)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {challenger_id!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )
    metadata_path = artifact_directory / "metadata.json"
    card_path = artifact_directory / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Challenger forecast artifact is incomplete: {artifact_directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared = config_fingerprint(entry.get("model", {}))
    observed = config_fingerprint(artifact_model_config(metadata))
    if declared != observed:
        raise DataContractError(
            f"Challenger {challenger_id!r} is registered with configuration fingerprint "
            f"{declared} but {artifact_directory} was produced with {observed}; refusing "
            "to record picks from a different configuration under this challenger"
        )

    observed_config = artifact_model_config(metadata)
    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
        "bet_side",
        "edge",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Challenger card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Challenger card contains duplicate games")
    unknown_bets = sorted(set(card["bet_side"].astype(str)) - VALID_BET_SIDES)
    if unknown_bets:
        raise DataContractError(f"Challenger card has invalid bet sides: {', '.join(unknown_bets)}")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Challenger card has games without a decision spread")
    kickoffs = _utc_series(card["kickoff"])
    if kickoffs.isna().any():
        raise DataContractError("Challenger card has games without a kickoff timestamp")

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and bool(pre_kickoff.any()):
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=int(card["season"].iloc[0]),
            week=int(card["week"].iloc[0]),
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=challenger_id,
        )
    mine = existing.loc[existing["challenger_id"].astype(str).eq(challenger_id)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    fresh = card.loc[pre_kickoff & ~already]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": challenger_id,
            "config_fingerprint": observed,
            "source_artifact": artifact_directory.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": fresh["bet_side"].astype(str),
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": pd.to_numeric(fresh["edge"], errors="coerce"),
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)
    return {
        "challenger_id": challenger_id,
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": artifact_directory.name,
        "config_fingerprint": observed,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
    }

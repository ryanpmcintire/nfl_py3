"""Season-lagged empirical player-availability probabilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError, require_columns

AVAILABILITY_RATE_VERSION = "v1"
AVAILABILITY_COMBINATION_PRIOR = 20.0
AVAILABILITY_POSITION_PRIOR = 100.0
_ALL = "__all__"

AVAILABILITY_RATE_COLUMNS = (
    "target_season",
    "report_category",
    "practice_category",
    "position_group",
    "unavailability_probability",
    "observations",
    "unavailable",
    "source_start_season",
    "source_end_season",
    "combination_prior",
    "position_prior",
    "rate_version",
)

_OFFENSIVE_LINE = frozenset(("C", "G", "OG", "OL", "OT", "T"))
_SKILL = frozenset(("FB", "HB", "QB", "RB", "TE", "WR"))
_FRONT = frozenset(("DE", "DL", "DT", "EDGE", "ILB", "LB", "NT", "OLB"))
_SECONDARY = frozenset(("CB", "DB", "FS", "S", "SAF", "SS"))


def report_category(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"none", "other"}:
        return normalized
    if normalized in {"out", "doubtful", "questionable", "probable"}:
        return normalized
    if not normalized or normalized in {"nan", "none", "<na>"}:
        return "none"
    return "other"


def practice_category(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"out", "dnp", "limited", "full", "none", "other"}:
        return normalized
    if "out (definitely will not play)" in normalized:
        return "out"
    if "did not participate" in normalized:
        return "dnp"
    if "limited" in normalized:
        return "limited"
    if "full" in normalized:
        return "full"
    if not normalized or normalized in {"nan", "none", "<na>"}:
        return "none"
    return "other"


def position_group(value: object) -> str:
    normalized = str(value).strip().upper()
    if normalized in _OFFENSIVE_LINE:
        return "offensive_line"
    if normalized in _SKILL:
        return "skill"
    if normalized in _FRONT:
        return "front"
    if normalized in _SECONDARY:
        return "secondary"
    return "other"


def fixed_unavailability(report_status: object, practice_status: object) -> float:
    """Return the original hand-authored availability prior."""

    report = report_category(report_status)
    report_mapping = {
        "out": 1.0,
        "doubtful": 0.85,
        "questionable": 0.35,
        "probable": 0.05,
    }
    if report in report_mapping:
        return report_mapping[report]
    practice = practice_category(practice_status)
    return {"out": 1.0, "dnp": 0.25, "limited": 0.10, "full": 0.0}.get(practice, 0.0)


def build_availability_outcomes(
    injuries: pd.DataFrame,
    snaps: pd.DataFrame,
    games: pd.DataFrame,
    *,
    decision_hours_before_kickoff: int = 24,
) -> pd.DataFrame:
    """Create one historical played/unavailable outcome per visible injury row."""

    if decision_hours_before_kickoff < 0:
        raise ValueError("decision_hours_before_kickoff cannot be negative")
    require_columns(
        injuries,
        (
            "season",
            "week",
            "team",
            "gsis_id",
            "position",
            "report_status",
            "practice_status",
            "date_modified",
        ),
        "injuries",
    )
    require_columns(
        snaps,
        (
            "season",
            "week",
            "team",
            "gsis_id",
            "offense_snaps",
            "defense_snaps",
            "st_snaps",
        ),
        "snap_counts_with_gsis_ids",
    )
    require_columns(
        games,
        ("game_id", "season", "week", "home_team", "away_team", "kickoff"),
        "game_features",
    )
    game_rows = games.loc[
        games["game_id"].notna(),
        ["game_id", "season", "week", "home_team", "away_team", "kickoff"],
    ].copy()
    game_rows["kickoff"] = pd.to_datetime(game_rows["kickoff"], errors="coerce", utc=True)
    team_games = pd.concat(
        [
            game_rows.rename(columns={"home_team": "team"})[
                ["game_id", "season", "week", "team", "kickoff"]
            ],
            game_rows.rename(columns={"away_team": "team"})[
                ["game_id", "season", "week", "team", "kickoff"]
            ],
        ],
        ignore_index=True,
    )
    team_games["team"] = team_games["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    if team_games.duplicated(["season", "week", "team"]).any():
        raise DataContractError("game_features contains multiple games for a team in one week")

    visible = injuries.copy()
    visible["date_modified"] = pd.to_datetime(visible["date_modified"], errors="coerce", utc=True)
    visible["team"] = visible["team"].replace(TEAM_ABBREVIATION_ALIASES).astype("string")
    visible = visible.merge(
        team_games,
        on=["season", "week", "team"],
        how="inner",
        validate="many_to_one",
    )
    cutoff = visible["kickoff"] - pd.Timedelta(hours=decision_hours_before_kickoff)
    visible = visible.loc[visible["date_modified"].le(cutoff)].copy()
    visible = visible.sort_values("date_modified").drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="last"
    )

    snap_rows = snaps.copy()
    covered_snap_seasons = set(
        pd.to_numeric(snap_rows["season"], errors="coerce").dropna().astype(int)
    )
    if not covered_snap_seasons:
        raise DataContractError("snap_counts_with_gsis_ids has no covered seasons")
    visible = visible.loc[visible["season"].isin(covered_snap_seasons)].copy()
    snap_rows["total_snaps"] = sum(
        pd.to_numeric(snap_rows[column], errors="coerce").fillna(0.0)
        for column in ("offense_snaps", "defense_snaps", "st_snaps")
    )
    played = (
        snap_rows.loc[snap_rows["gsis_id"].notna()]
        .groupby(["season", "week", "team", "gsis_id"], observed=True)["total_snaps"]
        .max()
        .gt(0)
        .rename("played")
        .reset_index()
    )
    result = visible.merge(
        played,
        on=["season", "week", "team", "gsis_id"],
        how="left",
        validate="one_to_one",
    )
    result["played"] = result["played"].fillna(False).astype(bool)
    result["unavailable"] = (~result["played"]).astype(float)
    result["report_category"] = result["report_status"].map(report_category)
    result["practice_category"] = result["practice_status"].map(practice_category)
    result["position_group"] = result["position"].map(position_group)
    result["fixed_unavailability"] = [
        fixed_unavailability(report, practice)
        for report, practice in zip(result["report_status"], result["practice_status"], strict=True)
    ]
    return (
        result[
            [
                "game_id",
                "season",
                "week",
                "team",
                "gsis_id",
                "position",
                "report_category",
                "practice_category",
                "position_group",
                "played",
                "unavailable",
                "fixed_unavailability",
            ]
        ]
        .sort_values(["season", "week", "game_id", "team", "gsis_id"])
        .reset_index(drop=True)
    )


def canonicalize_availability_rates(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, AVAILABILITY_RATE_COLUMNS, "availability_rates")
    result = frame.loc[:, list(AVAILABILITY_RATE_COLUMNS)].copy()
    for column in (
        "target_season",
        "observations",
        "unavailable",
        "source_start_season",
        "source_end_season",
        "combination_prior",
        "position_prior",
        "unavailability_probability",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.isna().any(axis=None):
        raise DataContractError("availability_rates contains null or invalid values")
    for column in (
        "report_category",
        "practice_category",
        "position_group",
        "rate_version",
    ):
        result[column] = result[column].astype("string")
    for column in (
        "target_season",
        "observations",
        "unavailable",
        "source_start_season",
        "source_end_season",
    ):
        result[column] = result[column].astype(int)
    if result.duplicated(
        ["target_season", "report_category", "practice_category", "position_group"]
    ).any():
        raise DataContractError("availability_rates contains duplicate lookup rows")
    if result["source_end_season"].ge(result["target_season"]).any():
        raise DataContractError("availability rates must use only earlier source seasons")
    if not result["unavailability_probability"].between(0.0, 1.0).all():
        raise DataContractError("availability rates must be bounded probabilities")
    if (
        result["observations"].lt(0).any()
        or result["unavailable"].lt(0).any()
        or result["unavailable"].gt(result["observations"]).any()
        or result["combination_prior"].lt(0).any()
        or result["position_prior"].lt(0).any()
    ):
        raise DataContractError("availability rate counts or priors are invalid")
    return result.sort_values(
        ["target_season", "report_category", "practice_category", "position_group"]
    ).reset_index(drop=True)


def build_season_lagged_availability_rates(
    outcomes: pd.DataFrame,
    *,
    target_seasons: Iterable[int],
    combination_prior: float = AVAILABILITY_COMBINATION_PRIOR,
    position_prior: float = AVAILABILITY_POSITION_PRIOR,
) -> pd.DataFrame:
    """Estimate expanding prior-season rates with declared hierarchical shrinkage."""

    if not np.isfinite(combination_prior) or combination_prior < 0:
        raise ValueError("combination_prior must be finite and nonnegative")
    if not np.isfinite(position_prior) or position_prior < 0:
        raise ValueError("position_prior must be finite and nonnegative")
    require_columns(
        outcomes,
        ("season", "report_category", "practice_category", "position_group", "unavailable"),
        "availability_outcomes",
    )
    targets = sorted({int(value) for value in target_seasons})
    if not targets:
        raise ValueError("At least one target season is required")
    output: list[dict[str, Any]] = []
    for target_season in targets:
        training = outcomes.loc[pd.to_numeric(outcomes["season"]).lt(target_season)].copy()
        if training.empty:
            continue
        source_start = int(training["season"].min())
        source_end = int(training["season"].max())
        total = len(training)
        unavailable = int(training["unavailable"].sum())
        global_rate = unavailable / total
        output.append(
            {
                "target_season": target_season,
                "report_category": _ALL,
                "practice_category": _ALL,
                "position_group": _ALL,
                "unavailability_probability": global_rate,
                "observations": total,
                "unavailable": unavailable,
                "source_start_season": source_start,
                "source_end_season": source_end,
                "combination_prior": combination_prior,
                "position_prior": position_prior,
                "rate_version": AVAILABILITY_RATE_VERSION,
            }
        )
        combo_rates: dict[tuple[str, str], float] = {}
        for (report, practice), group in training.groupby(
            ["report_category", "practice_category"], observed=True, sort=True
        ):
            observations = len(group)
            missing = int(group["unavailable"].sum())
            rate = (missing + combination_prior * global_rate) / (observations + combination_prior)
            combo_rates[(str(report), str(practice))] = rate
            output.append(
                {
                    "target_season": target_season,
                    "report_category": str(report),
                    "practice_category": str(practice),
                    "position_group": _ALL,
                    "unavailability_probability": rate,
                    "observations": observations,
                    "unavailable": missing,
                    "source_start_season": source_start,
                    "source_end_season": source_end,
                    "combination_prior": combination_prior,
                    "position_prior": position_prior,
                    "rate_version": AVAILABILITY_RATE_VERSION,
                }
            )
        for (report, practice, group_name), group in training.groupby(
            ["report_category", "practice_category", "position_group"],
            observed=True,
            sort=True,
        ):
            observations = len(group)
            missing = int(group["unavailable"].sum())
            parent = combo_rates[(str(report), str(practice))]
            rate = (missing + position_prior * parent) / (observations + position_prior)
            output.append(
                {
                    "target_season": target_season,
                    "report_category": str(report),
                    "practice_category": str(practice),
                    "position_group": str(group_name),
                    "unavailability_probability": rate,
                    "observations": observations,
                    "unavailable": missing,
                    "source_start_season": source_start,
                    "source_end_season": source_end,
                    "combination_prior": combination_prior,
                    "position_prior": position_prior,
                    "rate_version": AVAILABILITY_RATE_VERSION,
                }
            )
    if not output:
        raise DataContractError("No season-lagged availability rates could be estimated")
    return canonicalize_availability_rates(pd.DataFrame(output))


def availability_rate_lookup(
    rates: pd.DataFrame,
) -> dict[tuple[int, str, str, str], float]:
    canonical = canonicalize_availability_rates(rates)
    return {
        (
            int(str(row.target_season)),
            str(row.report_category),
            str(row.practice_category),
            str(row.position_group),
        ): float(str(row.unavailability_probability))
        for row in canonical.itertuples(index=False)
    }


def learned_unavailability(
    lookup: dict[tuple[int, str, str, str], float],
    *,
    target_season: int,
    report_status: object,
    practice_status: object,
    position: object,
) -> float | None:
    report = report_category(report_status)
    practice = practice_category(practice_status)
    group_name = position_group(position)
    for key in (
        (target_season, report, practice, group_name),
        (target_season, report, practice, _ALL),
        (target_season, _ALL, _ALL, _ALL),
    ):
        if key in lookup:
            return lookup[key]
    return None


def score_availability_rates(outcomes: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Compare learned and fixed probabilities on their matched target-season rows."""

    lookup = availability_rate_lookup(rates)
    scored = outcomes.copy()
    scored["learned_unavailability"] = [
        learned_unavailability(
            lookup,
            target_season=int(str(season)),
            report_status=report,
            practice_status=practice,
            position=position,
        )
        for season, report, practice, position in zip(
            scored["season"],
            scored["report_category"],
            scored["practice_category"],
            scored["position"],
            strict=True,
        )
    ]
    return scored.loc[scored["learned_unavailability"].notna()].reset_index(drop=True)


def summarize_availability_scores(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual = scored["unavailable"].to_numpy(dtype=float)
    for method, column in (
        ("fixed", "fixed_unavailability"),
        ("learned", "learned_unavailability"),
    ):
        probability = scored[column].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "player_games": len(scored),
                "brier_score": float(np.square(probability - actual).mean()),
                "classification_accuracy": float(((probability >= 0.5) == actual).mean()),
                "mean_probability": float(probability.mean()),
                "observed_unavailability": float(actual.mean()),
            }
        )
    return pd.DataFrame(rows)

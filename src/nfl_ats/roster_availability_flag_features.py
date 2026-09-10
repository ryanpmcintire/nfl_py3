from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS,
    SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.transaction_flag_features import (
    default_schedule,
    default_transactions_index,
    distinct_player_slugs,
    find_player_in_segment,
    latest_pfr_transactions_snapshot,
)
from nfl_ats.transaction_wire_features import canonical_team, match_transaction_teams

REPO_ROOT = Path(__file__).resolve().parents[2]

IR_RETURN_REINFORCEMENT_COLUMN = IR_RETURN_REINFORCEMENT_ON_PRODUCTION_FEATURE_COLUMNS[0]
SPECIALIST_ABSENCE_FADE_COLUMN = SPECIALIST_ABSENCE_FADE_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_SNAP_COUNTS_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/snap_counts.parquet"
DEFAULT_INJURIES_PATH = REPO_ROOT / "data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet"

HIGH_SNAP_SHARE_THRESHOLD = 0.5
IR_RETURN_WEEK_START = 5
IR_RETURN_WEEK_END = 8
SPECIALIST_POSITIONS: tuple[str, ...] = ("LS", "P")
SPECIALIST_INJURY_SEASON_END = 2024

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
}


IR_ACTIVATE_RE = re.compile(
    r"^(?P<prefix>.*?)-activat(?:e|es|ed)-(?P<player>.+?)-(?:from|off)-(?:injured-reserve|ir)(?:-|$)"
)
IR_PLACE_RE = re.compile(
    r"^(?P<prefix>.*?)-(?:to-)?place[sd]?-(?P<player>.+?)-(?:back-)?on-ir(?:-|$)"
)
DESIGNATE_RETURN_RE = re.compile(
    r"^(?P<prefix>.*?)-designat(?:e|es|ed)-(?P<player>.+?)-(?:for|to)-return"
    r"(?P<suffix>-from-[a-z0-9]+)?(?:-|$)"
)
_NON_IR_RETURN_SUFFIX_RE = re.compile(r"^-from-(?:pup|nfi|covid)")


def _normalize_designate_phrasing(slug: str) -> str:

    return slug.replace("-for-ir-return", "-for-return-from-ir").replace(
        "-to-ir-return", "-to-return-from-ir"
    )


def _month_end_timestamp(year: int, month: int) -> pd.Timestamp:

    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _prefix_team(prefix: str) -> str | None:
    teams = match_transaction_teams(prefix)
    if len(teams) != 1:
        return None
    return next(iter(teams))


def pinned_snap_counts(path: Path | None = None) -> pd.DataFrame:

    frame = pd.read_parquet(path or DEFAULT_SNAP_COUNTS_PATH)
    required = {"player", "team", "season", "week", "offense_pct", "defense_pct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"snap_counts is missing columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["snap_share"] = frame[["offense_pct", "defense_pct"]].max(axis=1)
    return frame


def pinned_injuries(path: Path | None = None) -> pd.DataFrame:

    frame = pd.read_parquet(path or DEFAULT_INJURIES_PATH)
    required = {"season", "week", "team", "position", "report_status", "game_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"injuries is missing columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["team"] = frame["team"].astype(str).map(canonical_team)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    return frame


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _reg_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")
    return reg


def _team_week_kickoffs(reg_schedule: pd.DataFrame) -> pd.DataFrame:

    home = reg_schedule[["season", "week", "home_team", "gameday_dt"]].rename(
        columns={"home_team": "team"}
    )
    away = reg_schedule[["season", "week", "away_team", "gameday_dt"]].rename(
        columns={"away_team": "team"}
    )
    return pd.concat([home, away], ignore_index=True)


def _signed_flag_from_qualifying(
    schedule: pd.DataFrame, qualifying: pd.DataFrame, column: str
) -> pd.DataFrame:

    reg = schedule.loc[
        :, ["game_id", "season", "week", "game_type", "home_team", "away_team"]
    ].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)

    if qualifying.empty:
        flag = np.zeros(len(reg), dtype=float)
        return pd.DataFrame({"game_id": reg["game_id"], column: flag})

    qual = qualifying.drop_duplicates(["season", "week", "team"]).assign(qualifies=True)
    merged = reg.merge(
        qual.rename(columns={"team": "home_team"}),
        on=["season", "week", "home_team"],
        how="left",
    ).rename(columns={"qualifies": "home_qualifies"})
    merged = merged.merge(
        qual.rename(columns={"team": "away_team"}),
        on=["season", "week", "away_team"],
        how="left",
    ).rename(columns={"qualifies": "away_qualifies"})
    home_q = merged["home_qualifies"].fillna(False).astype(bool)
    away_q = merged["away_qualifies"].fillna(False).astype(bool)
    flag = np.where(away_q & ~home_q, 1.0, np.where(home_q & ~away_q, -1.0, 0.0))
    return pd.DataFrame({"game_id": merged["game_id"], column: flag})


def _attach(features: pd.DataFrame, derived: pd.DataFrame, column: str) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if column in features.columns:
        raise DataContractError(f"features already carries {column}")
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_roster_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_roster_flag") if c in merged.columns]
    )
    merged.index = features.index
    return merged


def ir_activation_events(
    transactions_index: pd.DataFrame, player_slugs: pd.DataFrame
) -> pd.DataFrame:

    rows = transactions_index.loc[transactions_index["category"] == "ir_activation"]
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        match = IR_ACTIVATE_RE.search(str(row["slug"]))
        if match is None:
            continue
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), player_slugs)
        if player is None:
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "report_year", "report_month", "slug"]
    )


def designate_return_events(
    transactions_index: pd.DataFrame, player_slugs: pd.DataFrame
) -> pd.DataFrame:

    records: list[dict[str, object]] = []
    for _, row in transactions_index.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        normalized = _normalize_designate_phrasing(str(row["slug"]))
        match = DESIGNATE_RETURN_RE.search(normalized)
        if match is None:
            continue
        suffix = match.group("suffix")
        if suffix is not None and _NON_IR_RETURN_SUFFIX_RE.search(suffix):
            continue
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), player_slugs)
        if player is None:
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "report_year", "report_month", "slug"]
    )


def _ir_return_events(transactions_index: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:

    player_slugs = distinct_player_slugs(snap_counts)
    activated = ir_activation_events(transactions_index, player_slugs)
    activated["event_type"] = "activated"
    designated = designate_return_events(transactions_index, player_slugs)
    designated["event_type"] = "designated_return"
    combined = pd.concat([activated, designated], ignore_index=True)
    if combined.empty:
        return combined
    combined["season"] = combined["report_year"].astype(int)
    combined["_month_idx"] = combined["report_year"].astype(int) * 12 + combined[
        "report_month"
    ].astype(int)
    combined = combined.sort_values("_month_idx")
    return combined.drop_duplicates(["player", "team", "season"], keep="first").reset_index(
        drop=True
    )


def describe_ir_return_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:

    player_slugs = distinct_player_slugs(snap_counts)
    activated = ir_activation_events(transactions_index, player_slugs)
    designated = designate_return_events(transactions_index, player_slugs)
    events = _ir_return_events(transactions_index, snap_counts)
    return {
        "n_resolved_activation_events": len(activated),
        "n_resolved_designation_events": len(designated),
        "n_resolved_deduplicated_events": len(events),
        "resolved_slugs": events["slug"].tolist() if not events.empty else [],
    }


def derive_ir_return_reinforcement_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:

    reg = _reg_schedule(schedule)
    kickoffs = _team_week_kickoffs(reg)

    events = _ir_return_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        player, team, season = event["player"], event["team"], event["season"]
        report_end = _month_end_timestamp(int(event["report_year"]), int(event["report_month"]))

        prior_weeks = kickoffs.loc[
            (kickoffs["season"] == season)
            & (kickoffs["team"] == team)
            & (kickoffs["gameday_dt"] < report_end),
            "week",
        ]
        prior_rows = snap_counts.loc[
            (snap_counts["player"] == player)
            & (snap_counts["team"] == team)
            & (snap_counts["season"] == season)
            & (snap_counts["week"].isin(prior_weeks))
        ]
        if prior_rows.empty:
            continue
        trailing_share = float(prior_rows["snap_share"].mean())
        if trailing_share < HIGH_SNAP_SHARE_THRESHOLD:
            continue

        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["week"] >= IR_RETURN_WEEK_START)
            & (reg["week"] <= IR_RETURN_WEEK_END)
            & (reg["gameday_dt"] > report_end)
        ]
        for _, game in team_games.iterrows():
            qualifying_records.append(
                {"season": int(game["season"]), "week": int(game["week"]), "team": team}
            )

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    derived = _signed_flag_from_qualifying(schedule, qualifying, IR_RETURN_REINFORCEMENT_COLUMN)
    derived[IR_RETURN_REINFORCEMENT_COLUMN] = -derived[IR_RETURN_REINFORCEMENT_COLUMN]
    return derived


def attach_ir_return_reinforcement_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_snaps = snap_counts if snap_counts is not None else pinned_snap_counts()
    derived = derive_ir_return_reinforcement_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, IR_RETURN_REINFORCEMENT_COLUMN)


def specialist_player_slugs(injuries: pd.DataFrame) -> pd.DataFrame:

    lsp = injuries.loc[injuries["position"].isin(SPECIALIST_POSITIONS)]
    renamed = lsp.rename(columns={"full_name": "player"})
    return distinct_player_slugs(renamed)


def weekly_specialist_out_qualifying(injuries: pd.DataFrame) -> pd.DataFrame:

    mask = (
        injuries["position"].isin(SPECIALIST_POSITIONS)
        & injuries["report_status"].eq("Out")
        & injuries["game_type"].eq("REG")
        & injuries["week"].notna()
        & (injuries["season"] <= SPECIALIST_INJURY_SEASON_END)
    )
    rows = injuries.loc[mask, ["season", "week", "team"]].copy()
    rows["week"] = rows["week"].astype(int)
    return rows.drop_duplicates().reset_index(drop=True)


def specialist_ir_placement_events(
    transactions_index: pd.DataFrame, lsp_slugs: pd.DataFrame
) -> pd.DataFrame:

    rows = transactions_index.loc[transactions_index["category"] == "ir_placement"]
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        match = IR_PLACE_RE.search(str(row["slug"]))
        if match is None:
            continue
        team = _prefix_team(match.group("prefix"))
        if team is None:
            continue
        player = find_player_in_segment(match.group("player"), lsp_slugs)
        if player is None:
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "report_year", "report_month", "slug"]
    )


def describe_specialist_population(
    injuries: pd.DataFrame, transactions_index: pd.DataFrame
) -> dict[str, object]:

    weekly = weekly_specialist_out_qualifying(injuries)
    lsp_slugs = specialist_player_slugs(injuries)
    placements = specialist_ir_placement_events(transactions_index, lsp_slugs)
    activations = ir_activation_events(transactions_index, lsp_slugs)
    return {
        "n_weekly_out_team_weeks": len(weekly),
        "weekly_out_by_season": {
            str(season): len(group) for season, group in weekly.groupby("season")
        },
        "n_resolved_ir_placement_events": len(placements),
        "n_resolved_ir_activation_events_lsp": len(activations),
        "resolved_placement_slugs": placements["slug"].tolist() if not placements.empty else [],
    }


def _specialist_wire_window_qualifying(
    reg: pd.DataFrame, transactions_index: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:

    lsp_slugs = specialist_player_slugs(injuries)
    placements = specialist_ir_placement_events(transactions_index, lsp_slugs)
    if placements.empty:
        return pd.DataFrame(columns=["season", "week", "team"])
    activations = ir_activation_events(transactions_index, lsp_slugs)

    placements = placements.copy()
    placements["season"] = placements["report_year"].astype(int)
    placements["_month_idx"] = placements["report_year"].astype(int) * 12 + placements[
        "report_month"
    ].astype(int)
    placements = placements.sort_values("_month_idx").drop_duplicates(
        ["player", "team", "season"], keep="first"
    )

    records: list[dict[str, object]] = []
    for _, placement in placements.iterrows():
        player, team, season = placement["player"], placement["team"], placement["season"]
        placement_end = _month_end_timestamp(
            int(placement["report_year"]), int(placement["report_month"])
        )
        closing_end: pd.Timestamp | None = None
        if not activations.empty:
            same = activations.loc[
                (activations["player"] == player)
                & (activations["team"] == team)
                & (activations["report_year"].astype(int) == season)
                & (
                    activations["report_year"].astype(int) * 12
                    + activations["report_month"].astype(int)
                    > int(placement["_month_idx"])
                )
            ]
            if not same.empty:
                earliest = (
                    same.assign(
                        _idx=same["report_year"].astype(int) * 12 + same["report_month"].astype(int)
                    )
                    .sort_values("_idx")
                    .iloc[0]
                )
                closing_end = _month_end_timestamp(
                    int(earliest["report_year"]), int(earliest["report_month"])
                )

        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["gameday_dt"] > placement_end)
        ]
        if closing_end is not None:
            team_games = team_games.loc[team_games["gameday_dt"] <= closing_end]
        for _, game in team_games.iterrows():
            records.append({"season": int(game["season"]), "week": int(game["week"]), "team": team})
    return pd.DataFrame.from_records(records, columns=["season", "week", "team"])


def derive_specialist_absence_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:

    reg = _reg_schedule(schedule)
    weekly = weekly_specialist_out_qualifying(injuries)
    wire = _specialist_wire_window_qualifying(reg, transactions_index, injuries)
    qualifying = pd.concat([weekly, wire], ignore_index=True).drop_duplicates()
    qualifying = qualifying.loc[qualifying["season"] <= SPECIALIST_INJURY_SEASON_END]
    return _signed_flag_from_qualifying(schedule, qualifying, SPECIALIST_ABSENCE_FADE_COLUMN)


def attach_specialist_absence_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    transactions_index: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_transactions = (
        transactions_index if transactions_index is not None else default_transactions_index()
    )
    resolved_injuries = injuries if injuries is not None else pinned_injuries()
    derived = derive_specialist_absence_features(
        resolved_schedule, resolved_transactions, resolved_injuries
    )
    return _attach(features, derived, SPECIALIST_ABSENCE_FADE_COLUMN)


__all__ = [
    "DEFAULT_INJURIES_PATH",
    "DEFAULT_SNAP_COUNTS_PATH",
    "DESIGNATE_RETURN_RE",
    "HIGH_SNAP_SHARE_THRESHOLD",
    "IR_ACTIVATE_RE",
    "IR_PLACE_RE",
    "IR_RETURN_WEEK_END",
    "IR_RETURN_WEEK_START",
    "SPECIALIST_INJURY_SEASON_END",
    "SPECIALIST_POSITIONS",
    "attach_ir_return_reinforcement_features",
    "attach_specialist_absence_features",
    "derive_ir_return_reinforcement_features",
    "derive_specialist_absence_features",
    "describe_ir_return_population",
    "describe_specialist_population",
    "designate_return_events",
    "ir_activation_events",
    "latest_pfr_transactions_snapshot",
    "pinned_injuries",
    "pinned_snap_counts",
    "specialist_ir_placement_events",
    "specialist_player_slugs",
]

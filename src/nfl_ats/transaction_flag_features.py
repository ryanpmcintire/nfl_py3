from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS,
    HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS,
    SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.players import latest_player_snapshot
from nfl_ats.transaction_wire_features import (
    canonical_team,
    classify_transaction_slug,
    match_transaction_teams,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_SLOW_START_COLUMN = HOLDOUT_SLOW_START_ON_PRODUCTION_FEATURE_COLUMNS[0]
DEADLINE_INTEGRATION_DRAG_COLUMN = DEADLINE_INTEGRATION_DRAG_ON_PRODUCTION_FEATURE_COLUMNS[0]
SUSPENSION_RETURN_RUST_COLUMN = SUSPENSION_RETURN_RUST_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_PLAYERS_RAW_ROOT = REPO_ROOT / "data/players/raw"

_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
}

HIGH_SNAP_SHARE_THRESHOLD = 0.5
SUSPENSION_MIN_GAMES = 6

RETROSPECTIVE_SLUG_MARKERS: tuple[str, ...] = (
    "this-date-in-transactions-history",
    "this-date-in-nfl-transactions-history",
)


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def latest_pfr_transactions_snapshot(repo_root: Path | None = None) -> Path:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "pfr_transactions").glob("*/index.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no data/raw/pfr_transactions/*/index.parquet snapshot found under {root}"
        )
    return candidates[-1]


def default_transactions_index(
    repo_root: Path | None = None, *, snapshot: Path | None = None
) -> pd.DataFrame:

    path = snapshot if snapshot is not None else latest_pfr_transactions_snapshot(repo_root)
    frame = pd.read_parquet(path)
    required = {"slug", "transaction_relevant", "url_year", "url_month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"transactions index is missing columns: {', '.join(missing)}")

    frame = frame.loc[frame["transaction_relevant"]].copy()
    retrospective_pattern = "|".join(re.escape(marker) for marker in RETROSPECTIVE_SLUG_MARKERS)
    is_retrospective = frame["slug"].astype(str).str.contains(retrospective_pattern, regex=True)
    frame = frame.loc[~is_retrospective].copy()
    frame["url_year"] = pd.to_numeric(frame["url_year"], errors="coerce")
    frame["url_month"] = pd.to_numeric(frame["url_month"], errors="coerce")
    frame["category"] = frame["slug"].map(classify_transaction_slug)
    return frame.reset_index(drop=True)


def default_snap_counts(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    snapshot = latest_player_snapshot(root / "data" / "players" / "raw")
    frame = pd.read_parquet(snapshot.snaps_path)
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


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def _normalize_name_to_slug(name: str) -> str:
    lowered = name.strip().lower().replace("'", "").replace(".", "")
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def distinct_player_slugs(snap_counts: pd.DataFrame) -> pd.DataFrame:

    names = snap_counts["player"].dropna().astype(str).unique()
    frame = pd.DataFrame({"player": names})
    frame["name_slug"] = frame["player"].map(_normalize_name_to_slug)
    frame = frame.loc[frame["name_slug"].str.len() > 0].copy()
    frame["_len"] = frame["name_slug"].str.len()
    frame = frame.sort_values("_len", ascending=False).drop(columns="_len")
    return frame.drop_duplicates("name_slug").reset_index(drop=True)


def find_player_in_segment(segment: str, player_slugs: pd.DataFrame) -> str | None:

    padded = f"-{segment}-"
    for name_slug, player in zip(player_slugs["name_slug"], player_slugs["player"], strict=True):
        if f"-{name_slug}-" in padded:
            return str(player)
    return None


def _confirm_player_team(player: str, team: str, snap_counts: pd.DataFrame) -> bool:

    return bool(((snap_counts["player"] == player) & (snap_counts["team"] == team)).any())


def _month_end_timestamp(year: int, month: int) -> pd.Timestamp:

    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _implied_season(year: int, month: int) -> int:

    return year - 1 if month <= 2 else year


def _attach_qualifying_sides(
    schedule: pd.DataFrame,
    qualifying: pd.DataFrame,
    column: str,
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


def _attach(
    features: pd.DataFrame,
    derived: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if column in features.columns:
        raise DataContractError(f"features already carries {column}")

    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_txn_flag"),
        validate="one_to_one",
    )
    merged = merged.drop(columns=[c for c in ("key_0", "game_id_txn_flag") if c in merged.columns])
    merged.index = features.index
    return merged


HOLDOUT_END_RE = re.compile(
    r"(?:^|-)(?:ends-holdout|ended-holdout|reports-to-camp|reported-to-camp)(?:-|$)"
)


def holdout_ending_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:

    mask = transactions_index["slug"].astype(str).str.contains(HOLDOUT_END_RE)
    return transactions_index.loc[mask].copy()


def _holdout_events(transactions_index: pd.DataFrame, snap_counts: pd.DataFrame) -> pd.DataFrame:

    rows = holdout_ending_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        teams = match_transaction_teams(str(row["slug"]))
        if len(teams) != 1:
            continue
        team = next(iter(teams))
        player = find_player_in_segment(str(row["slug"]), player_slugs)
        if player is None or not _confirm_player_team(player, team, snap_counts):
            continue
        records.append(
            {
                "player": player,
                "team": team,
                "season": int(row["url_year"]),
                "report_year": int(row["url_year"]),
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["player", "team", "season", "report_year", "report_month", "slug"]
    )


def describe_holdout_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:

    rows = holdout_ending_transactions(transactions_index)
    resolved_team = rows["slug"].astype(str).map(lambda s: len(match_transaction_teams(s)))
    events = _holdout_events(transactions_index, snap_counts)
    return {
        "n_holdout_ending_slugs": len(rows),
        "n_resolved_exactly_one_team": int((resolved_team == 1).sum()),
        "n_resolved_player_and_team": len(events),
        "resolved_slugs": events["slug"].tolist(),
    }


def _player_started_prior_week(
    player: str, team: str, season: int, week: int, snap_counts: pd.DataFrame
) -> bool | None:

    if week <= 1:
        prior = snap_counts.loc[
            (snap_counts["player"] == player)
            & (snap_counts["team"] == team)
            & (snap_counts["season"] == season - 1)
        ]
        if prior.empty:
            return None
        last_week = prior["week"].max()
        share = prior.loc[prior["week"] == last_week, "snap_share"].max()
        return bool(share >= HIGH_SNAP_SHARE_THRESHOLD)

    prior = snap_counts.loc[
        (snap_counts["player"] == player)
        & (snap_counts["team"] == team)
        & (snap_counts["season"] == season)
        & (snap_counts["week"] == week - 1)
    ]
    if prior.empty:
        return None
    return bool(prior["snap_share"].max() >= HIGH_SNAP_SHARE_THRESHOLD)


def derive_holdout_slow_start_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _holdout_events(transactions_index, snap_counts)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        season = event["season"]
        team = event["team"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            (reg["season"] == season) & ((reg["home_team"] == team) | (reg["away_team"] == team))
        ]
        for week in range(1, 5):
            week_games = team_games.loc[team_games["week"] == week]
            if week_games.empty:
                continue
            kickoff = week_games["gameday_dt"].min()
            if not (report_end < kickoff):
                continue
            started = _player_started_prior_week(event["player"], team, season, week, snap_counts)
            if started:
                qualifying_records.append({"season": season, "week": week, "team": team})

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, HOLDOUT_SLOW_START_COLUMN)


def attach_holdout_slow_start_features(
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
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_holdout_slow_start_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, HOLDOUT_SLOW_START_COLUMN)


ACQUISITION_RE = re.compile(r"(?:^|-)(?:to-)?(?:re)?acquir(?:e|es|ed)-")
_ACQUIRE_SPLIT_RE = re.compile(r"^(?P<prefix>.*?)-(?:to-)?(?:re)?acquir(?:e|es|ed)-(?P<rest>.+)$")
DRAFT_PICK_RE = re.compile(r"-no-\d+|-\d+(?:st|nd|rd|th)-pick|-pick-\d+|select-")
SPECULATIVE_ACQUISITION_RE = re.compile(
    r"tried-to-acquir|attempted-to-acquir|wants-to-acquire|hopes-to-acquire|"
    r"hoping-to-acquire|could-acquire|would-acquire|looking-to-acquire|"
    r"interested-in-acquir|eyeing-.*acquir|exploring-.*acquir|in-talks-to-acquire"
)
DEADLINE_WINDOW_MONTHS: tuple[int, ...] = (9, 10, 11, 12)
DEADLINE_INTEGRATION_GAMES = 3


def confirmed_acquisition_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:

    trades = transactions_index.loc[transactions_index["category"] == "trade"].copy()
    slug = trades["slug"].astype(str)
    has_acquire = slug.str.contains(ACQUISITION_RE)
    is_pick = slug.str.contains(DRAFT_PICK_RE)
    is_speculative = slug.str.contains(SPECULATIVE_ACQUISITION_RE)
    in_window = trades["url_month"].isin(DEADLINE_WINDOW_MONTHS)
    return trades.loc[has_acquire & ~is_pick & ~is_speculative & in_window].copy()


def _parse_acquisition(slug: str) -> tuple[str, str] | None:

    match = _ACQUIRE_SPLIT_RE.search(slug)
    if match is None:
        return None
    prefix, rest = match.group("prefix"), match.group("rest")
    acquiring_teams = match_transaction_teams(prefix)
    if len(acquiring_teams) != 1:
        return None
    player_part = rest.split("-from-", 1)[0]
    return next(iter(acquiring_teams)), player_part


def _acquisition_events(
    transactions_index: pd.DataFrame,
    snap_counts: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:

    rows = confirmed_acquisition_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)
    games = schedule if schedule is not None else default_schedule()

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if pd.isna(row["url_year"]) or pd.isna(row["url_month"]):
            continue
        parsed = _parse_acquisition(str(row["slug"]))
        if parsed is None:
            continue
        acquiring_team, player_part = parsed
        player = find_player_in_segment(player_part, player_slugs)
        if player is None:
            continue

        season = int(row["url_year"])
        report_end = _month_end_timestamp(season, int(row["url_month"]))
        decision_games = games.loc[
            games["season"].eq(season)
            & games["game_type"].eq("REG")
            & (games["home_team"].eq(acquiring_team) | games["away_team"].eq(acquiring_team))
            & (pd.to_datetime(games["gameday"]) > report_end)
        ]
        if decision_games.empty:
            continue
        before_week = int(decision_games["week"].min())
        season_rows = snap_counts.loc[
            (snap_counts["player"] == player)
            & (snap_counts["season"] == season)
            & (snap_counts["week"] < before_week)
        ]
        prior_rows = season_rows.loc[season_rows["team"] != acquiring_team]
        if prior_rows.empty:
            continue

        last_prior_week = int(prior_rows["week"].max())
        giving_team = str(prior_rows.loc[prior_rows["week"] == last_prior_week, "team"].iloc[0])
        trailing_rows = prior_rows.loc[prior_rows["team"] == giving_team]
        trailing_share = float(trailing_rows["snap_share"].mean())
        if trailing_share < HIGH_SNAP_SHARE_THRESHOLD:
            continue

        records.append(
            {
                "player": player,
                "acquiring_team": acquiring_team,
                "giving_team": giving_team,
                "season": season,
                "last_prior_week": last_prior_week,
                "trailing_snap_share": trailing_share,
                "report_year": season,
                "report_month": int(row["url_month"]),
                "slug": row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "player",
            "acquiring_team",
            "giving_team",
            "season",
            "last_prior_week",
            "trailing_snap_share",
            "report_year",
            "report_month",
            "slug",
        ],
    )


def describe_deadline_acquisition_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> dict[str, object]:

    rows = confirmed_acquisition_transactions(transactions_index)
    parsed = rows["slug"].astype(str).map(_parse_acquisition)
    n_team_resolved = int(parsed.notna().sum())
    events = _acquisition_events(transactions_index, snap_counts)
    return {
        "n_confirmed_acquisition_slugs": len(rows),
        "n_resolved_acquiring_team": n_team_resolved,
        "n_resolved_player_and_high_snap": len(events),
        "resolved_slugs": events["slug"].tolist(),
    }


def derive_deadline_integration_drag_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _acquisition_events(transactions_index, snap_counts, schedule)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        team = event["acquiring_team"]
        season = event["season"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            (reg["season"] == season)
            & ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["week"] > event["last_prior_week"])
        ].sort_values("week")
        team_games = team_games.loc[team_games["gameday_dt"] > report_end]
        for _, game in team_games.head(DEADLINE_INTEGRATION_GAMES).iterrows():
            qualifying_records.append({"season": season, "week": int(game["week"]), "team": team})

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, DEADLINE_INTEGRATION_DRAG_COLUMN)


def attach_deadline_integration_drag_features(
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
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_deadline_integration_drag_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, DEADLINE_INTEGRATION_DRAG_COLUMN)


REINSTATED_RE = re.compile(r"(?<!suspension-)reinstated")
SUSPENSION_RETURN_GAMES = 2


def suspension_category_transactions(transactions_index: pd.DataFrame) -> pd.DataFrame:
    return transactions_index.loc[transactions_index["category"] == "suspension"].copy()


def _team_for_player_before(
    player: str, snap_counts: pd.DataFrame, season: int, before_week: int
) -> str | None:

    same_season = snap_counts.loc[
        (snap_counts["player"] == player)
        & (snap_counts["season"] == season)
        & (snap_counts["week"] < before_week)
    ]
    if not same_season.empty:
        last_week = same_season["week"].max()
        return str(same_season.loc[same_season["week"] == last_week, "team"].iloc[0])

    prior = snap_counts.loc[
        (snap_counts["player"] == player) & (snap_counts["season"] < season)
    ].sort_values(["season", "week"])
    if prior.empty:
        return None
    return str(prior.iloc[-1]["team"])


def _team_games_between(
    schedule_reg: pd.DataFrame, team: str, start_month_idx: int, end_month_idx: int
) -> int:

    team_games = schedule_reg.loc[
        (schedule_reg["home_team"] == team) | (schedule_reg["away_team"] == team)
    ]
    month_idx = team_games["gameday_dt"].dt.year * 12 + team_games["gameday_dt"].dt.month
    return int(((month_idx >= start_month_idx) & (month_idx < end_month_idx)).sum())


def _suspension_events(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    susp = suspension_category_transactions(transactions_index)
    player_slugs = distinct_player_slugs(snap_counts)
    slug_text = susp["slug"].astype(str)
    is_reinstated = slug_text.str.contains(REINSTATED_RE)

    imposed = susp.loc[~is_reinstated].copy()
    reinstated = susp.loc[is_reinstated].copy()
    imposed["player"] = (
        imposed["slug"].astype(str).map(lambda s: find_player_in_segment(s, player_slugs))
    )
    reinstated["player"] = (
        reinstated["slug"].astype(str).map(lambda s: find_player_in_segment(s, player_slugs))
    )
    imposed = imposed.loc[imposed["player"].notna() & imposed["url_year"].notna()]
    reinstated = reinstated.loc[reinstated["player"].notna() & reinstated["url_year"].notna()]

    records: list[dict[str, object]] = []
    for _, r_row in reinstated.iterrows():
        player = r_row["player"]
        r_idx = int(r_row["url_year"]) * 12 + int(r_row["url_month"])

        candidates = imposed.loc[imposed["player"] == player].copy()
        if candidates.empty:
            continue
        candidates["_idx"] = candidates["url_year"].astype(int) * 12 + candidates[
            "url_month"
        ].astype(int)
        earlier = candidates.loc[candidates["_idx"] < r_idx]
        if earlier.empty:
            continue
        imposed_row = earlier.sort_values("_idx").iloc[0]
        i_idx = int(imposed_row["_idx"])

        season = _implied_season(int(imposed_row["url_year"]), int(imposed_row["url_month"]))
        imposed_start = pd.Timestamp(
            year=int(imposed_row["url_year"]), month=int(imposed_row["url_month"]), day=1
        )
        later_games = reg.loc[reg["season"].eq(season) & reg["gameday_dt"].ge(imposed_start)]
        before_week = int(later_games["week"].min()) if not later_games.empty else 1
        team = _team_for_player_before(player, snap_counts, season, before_week)
        if team is None:
            continue

        n_games = _team_games_between(reg, team, i_idx, r_idx)
        if n_games < SUSPENSION_MIN_GAMES:
            continue

        records.append(
            {
                "player": player,
                "team": team,
                "n_games_measured": n_games,
                "report_year": int(r_row["url_year"]),
                "report_month": int(r_row["url_month"]),
                "slug": r_row["slug"],
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["player", "team", "n_games_measured", "report_year", "report_month", "slug"],
    )


def describe_suspension_return_population(
    transactions_index: pd.DataFrame, snap_counts: pd.DataFrame, schedule: pd.DataFrame
) -> dict[str, object]:

    susp = suspension_category_transactions(transactions_index)
    is_reinstated = susp["slug"].astype(str).str.contains(REINSTATED_RE)
    events = _suspension_events(transactions_index, snap_counts, schedule)
    return {
        "n_suspension_category_slugs": len(susp),
        "n_reinstatement_slugs": int(is_reinstated.sum()),
        "n_resolved_6plus_game_returns": len(events),
        "resolved_slugs": events["slug"].tolist(),
        "measured_game_counts": events["n_games_measured"].tolist(),
    }


def derive_suspension_return_rust_features(
    schedule: pd.DataFrame, transactions_index: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:

    _require_schedule_columns(schedule)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    events = _suspension_events(transactions_index, snap_counts, schedule)
    qualifying_records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        team = event["team"]
        report_end = _month_end_timestamp(event["report_year"], event["report_month"])
        team_games = reg.loc[
            ((reg["home_team"] == team) | (reg["away_team"] == team))
            & (reg["gameday_dt"] > report_end)
        ].sort_values("gameday_dt")
        for _, game in team_games.head(SUSPENSION_RETURN_GAMES).iterrows():
            qualifying_records.append(
                {"season": int(game["season"]), "week": int(game["week"]), "team": team}
            )

    qualifying = pd.DataFrame.from_records(qualifying_records, columns=["season", "week", "team"])
    return _attach_qualifying_sides(schedule, qualifying, SUSPENSION_RETURN_RUST_COLUMN)


def attach_suspension_return_rust_features(
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
    resolved_snaps = snap_counts if snap_counts is not None else default_snap_counts()
    derived = derive_suspension_return_rust_features(
        resolved_schedule, resolved_transactions, resolved_snaps
    )
    return _attach(features, derived, SUSPENSION_RETURN_RUST_COLUMN)


__all__ = [
    "ACQUISITION_RE",
    "DEADLINE_INTEGRATION_DRAG_COLUMN",
    "DEADLINE_INTEGRATION_GAMES",
    "DEADLINE_WINDOW_MONTHS",
    "DEFAULT_PLAYERS_RAW_ROOT",
    "DRAFT_PICK_RE",
    "HIGH_SNAP_SHARE_THRESHOLD",
    "HOLDOUT_END_RE",
    "HOLDOUT_SLOW_START_COLUMN",
    "REINSTATED_RE",
    "RETROSPECTIVE_SLUG_MARKERS",
    "SUSPENSION_MIN_GAMES",
    "SUSPENSION_RETURN_GAMES",
    "SUSPENSION_RETURN_RUST_COLUMN",
    "attach_deadline_integration_drag_features",
    "attach_holdout_slow_start_features",
    "attach_suspension_return_rust_features",
    "confirmed_acquisition_transactions",
    "default_schedule",
    "default_snap_counts",
    "default_transactions_index",
    "derive_deadline_integration_drag_features",
    "derive_holdout_slow_start_features",
    "derive_suspension_return_rust_features",
    "describe_deadline_acquisition_population",
    "describe_holdout_population",
    "describe_suspension_return_population",
    "distinct_player_slugs",
    "find_player_in_segment",
    "holdout_ending_transactions",
    "latest_pfr_transactions_snapshot",
    "suspension_category_transactions",
]

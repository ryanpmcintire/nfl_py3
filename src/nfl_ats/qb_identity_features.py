from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS,
    ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.data import DataContractError
from nfl_ats.players import _stable_crosswalk, canonicalize_rosters, latest_player_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]

ROOKIE_QB_DEBUT_FADE_COLUMN = ROOKIE_QB_DEBUT_FADE_ON_PRODUCTION_FEATURE_COLUMNS[0]
QB_REVENGE_COLUMN = QB_REVENGE_ON_PRODUCTION_FEATURE_COLUMNS[0]

DEFAULT_PLAYERS_RAW_ROOT = REPO_ROOT / "data/players/raw"
DEFAULT_COMBINE_RAW_ROOT = REPO_ROOT / "data/raw/combine"

_ROOKIE_QB_DEBUT_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "game_type",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
}
_QB_REVENGE_REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
}

DRAFT_TEAM_NAME_TO_CODE: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def _require_schedule_columns(schedule: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"schedule is missing columns: {', '.join(missing)}")


def default_schedule(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return pd.read_parquet(candidates[-1])


def default_weekly_rosters(repo_root: Path | None = None) -> pd.DataFrame:

    root = repo_root or REPO_ROOT
    snapshot = latest_player_snapshot(root / "data" / "players" / "raw")
    raw = pd.read_parquet(snapshot.rosters_path)
    return canonicalize_rosters(raw)


def latest_combine_snapshot(repo_root: Path | None = None) -> Path:

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "combine").glob("*/combine.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no data/raw/combine/*/combine.parquet snapshot found under {root}"
        )
    return candidates[-1]


def default_combine(repo_root: Path | None = None) -> pd.DataFrame:

    return pd.read_parquet(latest_combine_snapshot(repo_root))


def _first_reg_start_table(schedule: pd.DataFrame) -> pd.DataFrame:

    _require_schedule_columns(schedule, _ROOKIE_QB_DEBUT_REQUIRED_SCHEDULE_COLUMNS)
    reg = schedule.loc[schedule["game_type"].eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="raise")

    sides = []
    for qb_id_column, is_home in (("home_qb_id", True), ("away_qb_id", False)):
        side = reg.loc[reg[qb_id_column].notna(), ["game_id", "season", "gameday_dt", qb_id_column]]
        side = side.rename(columns={qb_id_column: "qb_id"})
        side["is_home"] = is_home
        sides.append(side)
    long_df = pd.concat(sides, ignore_index=True)
    long_df["qb_id"] = long_df["qb_id"].astype(str)
    long_df["game_id"] = long_df["game_id"].astype(str)
    long_df = long_df.sort_values(["qb_id", "gameday_dt", "game_id"]).reset_index(drop=True)
    long_df["is_first_archived_start"] = long_df.groupby("qb_id", sort=False).cumcount().eq(0)
    return long_df[["game_id", "season", "qb_id", "is_home", "is_first_archived_start"]]


def _season_years_exp(rosters: pd.DataFrame) -> pd.DataFrame:

    required = {"season", "gsis_id", "years_exp"}
    missing = sorted(required.difference(rosters.columns))
    if missing:
        raise DataContractError(f"rosters is missing columns: {', '.join(missing)}")
    cols = rosters.loc[:, ["season", "gsis_id", "years_exp"]].dropna(subset=["gsis_id"]).copy()
    cols["season"] = pd.to_numeric(cols["season"], errors="raise").astype(int)
    cols["gsis_id"] = cols["gsis_id"].astype(str)
    return cols.drop_duplicates(["season", "gsis_id"], keep="first")


def describe_rookie_qb_debut_population(schedule: pd.DataFrame, rosters: pd.DataFrame) -> dict:

    starts = _first_reg_start_table(schedule)
    debut = starts.loc[starts["is_first_archived_start"]].copy()
    years_exp = _season_years_exp(rosters)
    debut = debut.merge(
        years_exp, left_on=["season", "qb_id"], right_on=["season", "gsis_id"], how="left"
    )
    resolved = debut["years_exp"].notna()
    is_rookie = resolved & debut["years_exp"].eq(0.0)
    is_established = resolved & debut["years_exp"].gt(0.0)
    return {
        "n_first_archived_reg_starts": len(debut),
        "n_confirmed_rookie_debuts": int(is_rookie.sum()),
        "n_confirmed_non_rookie_first_starts": int(is_established.sum()),
        "n_unresolved_years_exp": int((~resolved).sum()),
    }


def oracle_derive_rookie_qb_debut_fade_features(
    schedule: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:

    starts = _first_reg_start_table(schedule)
    debut = starts.loc[starts["is_first_archived_start"]].copy()
    years_exp = _season_years_exp(rosters)
    debut = debut.merge(
        years_exp, left_on=["season", "qb_id"], right_on=["season", "gsis_id"], how="left"
    )
    debut["is_debut_rookie"] = debut["years_exp"].eq(0.0)

    home_flags = debut.loc[debut["is_home"], ["game_id", "is_debut_rookie"]].rename(
        columns={"is_debut_rookie": "home_debut_rookie"}
    )
    away_flags = debut.loc[~debut["is_home"], ["game_id", "is_debut_rookie"]].rename(
        columns={"is_debut_rookie": "away_debut_rookie"}
    )

    all_ids = schedule[["game_id"]].astype({"game_id": str})
    result = all_ids.merge(home_flags, on="game_id", how="left")
    result = result.merge(away_flags, on="game_id", how="left")
    result["home_debut_rookie"] = result["home_debut_rookie"].fillna(False)
    result["away_debut_rookie"] = result["away_debut_rookie"].fillna(False)

    flag = np.where(
        result["away_debut_rookie"] & ~result["home_debut_rookie"],
        1.0,
        np.where(result["home_debut_rookie"] & ~result["away_debut_rookie"], -1.0, 0.0),
    )
    return pd.DataFrame(
        {"game_id": result["game_id"], "oracle_" + ROOKIE_QB_DEBUT_FADE_COLUMN: flag}
    )


def attach_rookie_qb_debut_fade_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
    depth_charts: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if ROOKIE_QB_DEBUT_FADE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {ROOKIE_QB_DEBUT_FADE_COLUMN}")

    resolved_schedule = schedule if schedule is not None else default_schedule()
    resolved_rosters = rosters if rosters is not None else default_weekly_rosters()
    derived = derive_rookie_qb_debut_fade_features(
        resolved_schedule, resolved_rosters, depth_charts=depth_charts
    )
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_qb_identity"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_qb_identity") if c in merged.columns]
    )
    merged.index = features.index
    return merged


def _canonical_schedule_team(codes: pd.Series) -> pd.Series:

    return codes.astype(str).replace(TEAM_ABBREVIATION_ALIASES)


def draft_team_by_gsis_id(combine: pd.DataFrame, rosters: pd.DataFrame) -> dict[str, str]:

    required = {"pfr_id", "draft_team", "draft_year"}
    missing = sorted(required.difference(combine.columns))
    if missing:
        raise DataContractError(f"combine is missing columns: {', '.join(missing)}")

    rows = combine.loc[combine["pfr_id"].notna() & combine["draft_team"].notna()].copy()
    unrecognized = sorted(set(rows["draft_team"].unique()) - set(DRAFT_TEAM_NAME_TO_CODE))
    if unrecognized:
        raise DataContractError(f"unrecognized combine draft_team values: {unrecognized}")

    rows["pfr_id"] = rows["pfr_id"].astype(str)
    crosswalk = _stable_crosswalk(rosters)
    rows["gsis_id"] = rows["pfr_id"].map(crosswalk)
    rows = rows.loc[rows["gsis_id"].notna()].copy()
    rows["draft_team_code"] = rows["draft_team"].map(DRAFT_TEAM_NAME_TO_CODE)
    rows["draft_year"] = pd.to_numeric(rows["draft_year"], errors="coerce")
    rows = rows.sort_values(["gsis_id", "draft_year"]).drop_duplicates("gsis_id", keep="first")
    return dict(zip(rows["gsis_id"], rows["draft_team_code"], strict=True))


def qb_revenge_join_diagnostics(schedule: pd.DataFrame, draft_team_lookup: dict[str, str]) -> dict:

    home = schedule["home_qb_id"].dropna().astype(str)
    away = schedule["away_qb_id"].dropna().astype(str)
    all_starts = pd.concat([home, away], ignore_index=True)
    resolved = all_starts.isin(draft_team_lookup)
    return {
        "n_qb_side_starts": len(all_starts),
        "n_resolved_draft_team": int(resolved.sum()),
        "join_rate": float(resolved.mean()) if len(all_starts) else float("nan"),
    }


def oracle_derive_qb_revenge_features(
    schedule: pd.DataFrame, draft_team_lookup: dict[str, str]
) -> pd.DataFrame:

    _require_schedule_columns(schedule, _QB_REVENGE_REQUIRED_SCHEDULE_COLUMNS)
    home_team = _canonical_schedule_team(schedule["home_team"])
    away_team = _canonical_schedule_team(schedule["away_team"])

    home_qb_known = schedule["home_qb_id"].notna()
    away_qb_known = schedule["away_qb_id"].notna()
    home_draft_team = schedule["home_qb_id"].astype(str).map(draft_team_lookup)
    away_draft_team = schedule["away_qb_id"].astype(str).map(draft_team_lookup)

    home_revenge = home_qb_known & home_draft_team.notna() & home_draft_team.eq(away_team)
    away_revenge = away_qb_known & away_draft_team.notna() & away_draft_team.eq(home_team)

    flag = np.where(
        home_revenge & ~away_revenge, 1.0, np.where(away_revenge & ~home_revenge, -1.0, 0.0)
    )
    return pd.DataFrame(
        {"game_id": schedule["game_id"].astype(str), "oracle_" + QB_REVENGE_COLUMN: flag}
    )


def attach_qb_revenge_features(
    features: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
    combine: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
    depth_charts: pd.DataFrame | None = None,
    draft_team_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    if QB_REVENGE_COLUMN in features.columns:
        raise DataContractError(f"features already carries {QB_REVENGE_COLUMN}")

    resolved_schedule = schedule if schedule is not None else default_schedule()
    if draft_team_lookup is not None:
        lookup = draft_team_lookup
    else:
        resolved_combine = combine if combine is not None else default_combine()
        resolved_rosters = rosters if rosters is not None else default_weekly_rosters()
        lookup = draft_team_by_gsis_id(resolved_combine, resolved_rosters)
    derived = derive_qb_revenge_features(resolved_schedule, lookup, depth_charts=depth_charts)
    merged = features.merge(
        derived,
        left_on=features["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_qb_identity"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[c for c in ("key_0", "game_id_qb_identity") if c in merged.columns]
    )
    merged.index = features.index
    return merged


__all__ = [
    "DEFAULT_COMBINE_RAW_ROOT",
    "DEFAULT_PLAYERS_RAW_ROOT",
    "DRAFT_TEAM_NAME_TO_CODE",
    "QB_REVENGE_COLUMN",
    "ROOKIE_QB_DEBUT_FADE_COLUMN",
    "attach_qb_revenge_features",
    "attach_rookie_qb_debut_fade_features",
    "decision_time_qb_schedule",
    "default_combine",
    "default_depth_chart_observations",
    "default_schedule",
    "default_weekly_rosters",
    "derive_qb_revenge_features",
    "derive_rookie_qb_debut_fade_features",
    "describe_rookie_qb_debut_population",
    "draft_team_by_gsis_id",
    "latest_combine_snapshot",
    "oracle_derive_qb_revenge_features",
    "oracle_derive_rookie_qb_debut_fade_features",
    "qb_revenge_join_diagnostics",
]


def decision_time_qb_schedule(
    schedule: pd.DataFrame, depth_charts: pd.DataFrame | None = None
) -> pd.DataFrame:
    from nfl_ats.nfl_week import pool_decision_cutoff
    from nfl_ats.players import _schedule_kickoff_utc

    result = schedule.copy()
    kickoff = (
        pd.to_datetime(result["kickoff"], utc=True)
        if "kickoff" in result
        else _schedule_kickoff_utc(result)
    )
    result["decision_at"] = kickoff.map(
        lambda value: pool_decision_cutoff(value) if pd.notna(value) else pd.NaT
    )
    if depth_charts is None:
        depth_charts = default_depth_chart_observations()
    depth = depth_charts.copy()
    observed = next(
        (name for name in ("depth_observed_at", "observed_at_utc", "dt") if name in depth),
        None,
    )
    for side in ("home", "away"):
        result[f"oracle_{side}_qb_id"] = result[f"{side}_qb_id"]
        result[f"{side}_qb_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
        result[f"{side}_qb_observed_at"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
    if observed is None:
        return result
    depth["_observed"] = pd.to_datetime(depth[observed], utc=True, errors="coerce")
    depth["team"] = _canonical_schedule_team(depth["team"])
    position = "position" if "position" in depth else "pos_abb"
    rank = "depth_rank" if "depth_rank" in depth else "pos_rank"
    depth = depth.loc[depth[position].eq("QB") & depth["_observed"].notna()].copy()
    depth[rank] = pd.to_numeric(depth[rank], errors="coerce")
    groups = dict(iter(depth.groupby("team")))
    for index, game in result.iterrows():
        for side in ("home", "away"):
            team = TEAM_ABBREVIATION_ALIASES.get(
                str(game[f"{side}_team"]), str(game[f"{side}_team"])
            )
            rows = groups.get(team)
            if rows is None or pd.isna(game["decision_at"]):
                continue
            rows = rows.loc[rows["_observed"].lt(game["decision_at"])]
            if rows.empty:
                continue
            latest = rows.loc[rows["_observed"].eq(rows["_observed"].max())]
            qb1 = latest.loc[latest[rank].eq(1) & latest["gsis_id"].notna()]
            if qb1["gsis_id"].nunique() != 1:
                continue
            row = qb1.iloc[0]
            result.at[index, f"{side}_qb_id"] = str(row["gsis_id"])
            result.at[index, f"{side}_qb_observed_at"] = row["_observed"]
    return result


def derive_qb_revenge_features(
    schedule: pd.DataFrame,
    draft_team_lookup: dict[str, str],
    *,
    depth_charts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_schedule_columns(schedule, _QB_REVENGE_REQUIRED_SCHEDULE_COLUMNS)
    projected = decision_time_qb_schedule(schedule, depth_charts)
    result = oracle_derive_qb_revenge_features(projected, draft_team_lookup).rename(
        columns={"oracle_" + QB_REVENGE_COLUMN: QB_REVENGE_COLUMN}
    )
    result.loc[
        projected[["home_qb_id", "away_qb_id"]].isna().any(axis=1).to_numpy(), QB_REVENGE_COLUMN
    ] = np.nan
    return result


def derive_rookie_qb_debut_fade_features(
    schedule: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    depth_charts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from nfl_ats.players import _schedule_kickoff_utc

    _require_schedule_columns(schedule, _ROOKIE_QB_DEBUT_REQUIRED_SCHEDULE_COLUMNS)
    projected = decision_time_qb_schedule(schedule, depth_charts)
    history = schedule.loc[schedule["game_type"].eq("REG")].copy()
    history["kickoff"] = (
        pd.to_datetime(history["kickoff"], utc=True)
        if "kickoff" in history
        else _schedule_kickoff_utc(history)
    )
    starts = (
        pd.concat(
            [
                history[[f"{side}_qb_id", "kickoff"]].rename(columns={f"{side}_qb_id": "qb_id"})
                for side in ("home", "away")
            ]
        )
        .groupby("qb_id")["kickoff"]
        .min()
    )
    years = _season_years_exp(rosters).set_index(["season", "gsis_id"])["years_exp"]
    flags = {}
    for side in ("home", "away"):
        prior = projected[f"{side}_qb_id"].map(starts).dt.normalize()
        rookie = pd.Series(
            [
                years.get((int(row.season), str(row[f"{side}_qb_id"])), np.nan) == 0
                for _, row in projected.iterrows()
            ],
            index=projected.index,
        )
        flags[side] = (
            rookie
            & ~prior.lt(pd.to_datetime(projected["decision_at"], utc=True).dt.normalize())
            & projected["game_type"].eq("REG")
        )
    values = flags["away"].astype(float) - flags["home"].astype(float)
    values.loc[projected[["home_qb_id", "away_qb_id"]].isna().any(axis=1)] = np.nan
    return pd.DataFrame(
        {"game_id": projected["game_id"].astype(str), ROOKIE_QB_DEBUT_FADE_COLUMN: values}
    )


def default_depth_chart_observations() -> pd.DataFrame:
    paths = sorted((DEFAULT_PLAYERS_RAW_ROOT / "depth_charts").glob("*/depth_charts.parquet"))
    paths += sorted((REPO_ROOT / "data/quarterbacks/depth/raw").glob("*/quarterbacks.parquet"))
    pieces = []
    for path in paths:
        rows = pd.read_parquet(path)
        observed = next(
            (name for name in ("depth_observed_at", "observed_at_utc", "dt") if name in rows), None
        )
        if observed is None:
            continue
        position = "position" if "position" in rows else "pos_abb"
        rank = "depth_rank" if "depth_rank" in rows else "pos_rank"
        rows = rows[["team", "gsis_id", position, rank, observed]].rename(
            columns={position: "position", rank: "depth_rank", observed: "depth_observed_at"}
        )
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True).drop_duplicates() if pieces else pd.DataFrame()

"""Recurrence-hazard availability research build (literature_leads section 4 lead 2).

Body-part classifier for NFL.com report text, point-in-time-safe per
player-game recurrence features, player-level validation against DNP/limited
outcomes, split-half reliability of the recurrence-flag signal, and measured
hazard rates compared against published recurrence relative risks (RR 2.7
same-history, RR ~4.8 recent-same-season). Research script; writes only to
``artifacts/recurrence_hazard/`` and prints a JSON summary.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

INJURIES_MAIN_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/injuries.parquet"
ROSTERS_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/weekly_rosters.parquet"
SNAPS_PATH = REPO_ROOT / "data/players/raw/20260817T184901Z/snap_counts.parquet"
SCHEDULES_PATH = REPO_ROOT / "data/raw/20260817T235649Z/schedules.parquet"
NFLCOM_INJURIES_PATH = REPO_ROOT / "data/raw/nflcom_injuries/20260821T222602Z/injuries.parquet"
ARTIFACT_DIR = REPO_ROOT / "artifacts/recurrence_hazard"

INJURY_CLASSES = ("hamstring", "knee", "ankle", "concussion", "shoulder")
OTHER_LABEL = "other"
UNMAPPED_LABEL = "unmapped"

EPISODE_GAP_DAYS = 21
ACTIVE_EPISODE_DAYS = 10
TRAIN_SEASON = 2022
VAL_SEASON = 2023
TEST_SEASON = 2024

_NAMED_CLASS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hamstring", ("hamstring",)),
    ("knee", ("knee", "acl", "mcl", "pcl", "meniscus", "patella")),
    ("ankle", ("ankle",)),
    ("concussion", ("concussion",)),
    (
        "shoulder",
        ("shoulder", "clavicle", "collarbone", "ac joint", "rotator", "scapula"),
    ),
)
_OTHER_CLASS_PATTERNS: tuple[str, ...] = (
    "achilles",
    "groin",
    "calf",
    "foot",
    "back",
    "hip",
    "neck",
    "quad",
    "quadricep",
    "rib",
    "pectoral",
    "chest",
    "abdomen",
    "abdominal",
    "thigh",
    "hand",
    "toe",
    "thumb",
    "wrist",
    "heel",
    "finger",
    "shin",
    "elbow",
    "eye",
    "bicep",
    "tricep",
    "forearm",
    "oblique",
    "fibula",
    "tibia",
    "cramp",
    "head",
    "face",
    "jaw",
    "tooth",
    "dental",
    "nose",
    "throat",
    "kidney",
    "pelvis",
    "pelvic",
    "hernia",
    "appendix",
    "appendicitis",
    "stinger",
    "glute",
    "abductor",
    "adductor",
    "illness",
    "sick",
    "disease",
    "infection",
    "personal",
    "coach",
    "coaching",
    "suspension",
    "dehydration",
    "not injury",
)


def classify_injury_text(value: object) -> str:
    lowered = "" if value is None else str(value).strip().lower()
    if not lowered or lowered in {"nan", "none", "--"}:
        return UNMAPPED_LABEL
    for cls, patterns in _NAMED_CLASS_PATTERNS:
        for pattern in patterns:
            if re.search(rf"\b{re.escape(pattern)}", lowered):
                return cls
    for pattern in _OTHER_CLASS_PATTERNS:
        if re.search(rf"\b{re.escape(pattern)}", lowered):
            return OTHER_LABEL
    return UNMAPPED_LABEL


def normalize_player_name(value: object) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def load_schedules() -> pd.DataFrame:
    schedules = pd.read_parquet(SCHEDULES_PATH)
    frame = schedules.loc[
        schedules["game_type"].eq("REG"),
        ["game_id", "season", "week", "home_team", "away_team", "gameday"],
    ].copy()
    frame["kickoff"] = pd.to_datetime(frame.pop("gameday"), errors="coerce")
    home = frame.rename(columns={"home_team": "team"})[
        ["game_id", "season", "week", "team", "kickoff"]
    ]
    away = frame.rename(columns={"away_team": "team"})[
        ["game_id", "season", "week", "team", "kickoff"]
    ]
    return pd.concat([home, away], ignore_index=True)


def load_availability_games() -> pd.DataFrame:
    schedules = pd.read_parquet(SCHEDULES_PATH)
    frame = schedules.loc[
        schedules["game_type"].eq("REG"),
        ["game_id", "season", "week", "home_team", "away_team", "gameday"],
    ].copy()
    frame["kickoff"] = pd.to_datetime(frame.pop("gameday"), errors="coerce", utc=True)
    return frame.reset_index(drop=True)


def build_roster_bridge(rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    names = rosters.loc[rosters["full_name"].notna() & rosters["gsis_id"].notna()].copy()
    names["normalized_name"] = names["full_name"].map(normalize_player_name)
    counts = (
        names.groupby(["season", "normalized_name", "gsis_id"], observed=True)
        .size()
        .rename("appearances")
        .reset_index()
    )
    seasonal = (
        counts.sort_values(
            ["season", "normalized_name", "appearances", "gsis_id"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["season", "normalized_name"])
        .loc[:, ["season", "normalized_name", "gsis_id"]]
    )
    unique_names = names.groupby("normalized_name")["gsis_id"].agg(["nunique", "first"])
    global_map = {
        str(name): str(gsis)
        for name, gsis in unique_names.loc[unique_names["nunique"].eq(1), "first"]
        .astype(str)
        .items()
    }
    return seasonal, global_map


def load_nflcom_entries(
    team_games: pd.DataFrame,
    roster_bridge: pd.DataFrame,
    roster_global: dict[str, str],
) -> pd.DataFrame:
    raw = pd.read_parquet(NFLCOM_INJURIES_PATH)
    entries = raw.loc[raw["game_status"].notna() | raw["injury"].notna()].copy()
    entries = entries.drop_duplicates(["season", "week", "team", "player"], keep="first")
    entries["normalized_name"] = entries["player"].map(normalize_player_name)
    bridge = roster_bridge.rename(columns={"gsis_id": "gsis_id_seasonal"})
    entries = entries.merge(
        bridge, on=["season", "normalized_name"], how="left", validate="many_to_one"
    )
    entries["gsis_id"] = entries["gsis_id_seasonal"].fillna(
        entries["normalized_name"].map(roster_global)
    )
    entries = entries.drop(columns="gsis_id_seasonal")
    entries["body_part_class"] = entries["injury"].map(classify_injury_text)
    entries["primary_class"] = entries["body_part_class"].where(
        entries["body_part_class"].isin(INJURY_CLASSES), OTHER_LABEL
    )
    entries = entries.merge(
        team_games.rename(columns={"kickoff": "game_date"}),
        on=["season", "week", "team"],
        how="inner",
        validate="many_to_one",
    )
    return entries.sort_values(["gsis_id", "body_part_class", "game_date"]).reset_index(drop=True)


def assign_episodes(entries: pd.DataFrame) -> pd.DataFrame:
    frame = entries.sort_values(["gsis_id", "body_part_class", "game_date"]).copy()
    grouped = frame.groupby(["gsis_id", "body_part_class"], sort=False, observed=True)
    previous = grouped["game_date"].shift(1)
    gap_days = (frame["game_date"] - previous).dt.days
    new_episode = previous.isna() | gap_days.gt(EPISODE_GAP_DAYS)
    frame["episode_seq"] = (
        new_episode.groupby([frame["gsis_id"], frame["body_part_class"]]).cumsum().astype(int)
    )
    frame["episode_id"] = (
        frame["gsis_id"].astype(str)
        + "|"
        + frame["body_part_class"]
        + "|"
        + frame["episode_seq"].astype(str)
    )
    frame["is_episode_start"] = new_episode.astype(bool)
    return frame.reset_index(drop=True)


def build_episode_table(entries_with_episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        entries_with_episodes.groupby("episode_id", observed=True)
        .agg(
            gsis_id=("gsis_id", "first"),
            body_part_class=("body_part_class", "first"),
            start_date=("game_date", "min"),
            end_date=("game_date", "max"),
            start_season=("season", "min"),
            n_reports=("game_date", "size"),
        )
        .reset_index()
        .sort_values(["gsis_id", "body_part_class", "start_date"])
        .reset_index(drop=True)
    )


def build_played_games(snaps_with_dates: pd.DataFrame) -> pd.DataFrame:
    snapped = snaps_with_dates.copy()
    snapped["total_snaps"] = snapped[["offense_snaps", "defense_snaps", "st_snaps"]].sum(axis=1)
    played = snapped.loc[snapped["total_snaps"].gt(0)]
    return (
        played.groupby(["gsis_id", "game_id"], observed=True)["game_date"]
        .first()
        .reset_index()
        .sort_values(["gsis_id", "game_date"])
        .reset_index(drop=True)
    )


EpisodeKey = tuple[pd.Timestamp, pd.Timestamp, int]


def _episode_view(episodes: pd.DataFrame) -> dict[tuple[str, str], list[EpisodeKey]]:
    view: dict[tuple[str, str], list[EpisodeKey]] = {}
    for row in episodes.itertuples(index=False):
        key = (str(row.gsis_id), str(row.body_part_class))
        view.setdefault(key, []).append(
            (pd.Timestamp(row.end_date), pd.Timestamp(row.start_date), int(row.start_season))
        )
    for values in view.values():
        values.sort(key=lambda item: item[0])
    return view


def _played_lookup(played_games: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(gsis_id): pd.to_datetime(group["game_date"]).to_numpy()
        for gsis_id, group in played_games.groupby("gsis_id", observed=True)
    }


def _first_played_between(
    dates: np.ndarray | None, lower: pd.Timestamp, upper: pd.Timestamp
) -> pd.Timestamp | None:
    if dates is None or len(dates) == 0:
        return None
    lower64 = np.datetime64(lower.to_pydatetime())
    upper64 = np.datetime64(upper.to_pydatetime())
    index = int(np.searchsorted(dates, lower64, side="right"))
    if index >= len(dates) or dates[index] >= upper64:
        return None
    return pd.Timestamp(dates[index])


def build_recurrence_features(
    entries_with_episodes: pd.DataFrame,
    episodes: pd.DataFrame,
    played_games: pd.DataFrame,
) -> pd.DataFrame:
    episode_view = _episode_view(episodes)
    played_lookup = _played_lookup(played_games)
    records: list[dict[str, object]] = []
    for row in entries_with_episodes.itertuples(index=False):
        gsis_id = str(row.gsis_id)
        game_date = pd.Timestamp(row.game_date)
        season = int(row.season)
        record: dict[str, object] = {"row_order": len(records)}
        prior_ends: list[pd.Timestamp] = []
        prior_seasons: list[int] = []
        total_prior = 0
        for cls in INJURY_CLASSES:
            history = [item for item in episode_view.get((gsis_id, cls), []) if item[0] < game_date]
            total_prior += len(history)
            record[f"n_prior_episodes_{cls}"] = len(history)
            record[f"ss_prior_episode_{cls}"] = int(any(item[2] == season for item in history))
            if not history:
                for name in (
                    "days_since_last_report",
                    "days_since_rtp",
                    "active_episode",
                    "post_rtp_60d",
                    "post_rtp_120d",
                    "returned_pre_game",
                ):
                    default: float | int = np.nan if name.startswith("days") else 0
                    record[f"{name}_{cls}"] = default
                continue
            latest_end = history[-1][0]
            prior_ends.append(latest_end)
            prior_seasons.append(history[-1][2])
            record[f"days_since_last_report_{cls}"] = float((game_date - latest_end).days)
            record[f"active_episode_{cls}"] = int(
                (game_date - latest_end).days <= ACTIVE_EPISODE_DAYS
            )
            rtp = _first_played_between(played_lookup.get(gsis_id), latest_end, game_date)
            if rtp is None:
                record[f"days_since_rtp_{cls}"] = np.nan
                record[f"post_rtp_60d_{cls}"] = 0
                record[f"post_rtp_120d_{cls}"] = 0
                record[f"returned_pre_game_{cls}"] = 0
            else:
                days = float((game_date - rtp).days)
                record[f"days_since_rtp_{cls}"] = days
                record[f"post_rtp_60d_{cls}"] = int(0 <= days < 60)
                record[f"post_rtp_120d_{cls}"] = int(0 <= days < 120)
                record[f"returned_pre_game_{cls}"] = 1
        record["ever_injured_named"] = int(total_prior > 0)
        record["n_prior_episodes_any"] = total_prior
        record["ss_prior_episode_any"] = int(any(item == season for item in prior_seasons))
        if prior_ends:
            latest_any_end = max(prior_ends)
            record["active_episode_any"] = int(
                (game_date - latest_any_end).days <= ACTIVE_EPISODE_DAYS
            )
            rtp_any = _first_played_between(played_lookup.get(gsis_id), latest_any_end, game_date)
            if rtp_any is None:
                record["days_since_rtp_any"] = np.nan
                record["post_rtp_60d_any"] = 0
                record["post_rtp_120d_any"] = 0
            else:
                days = float((game_date - rtp_any).days)
                record["days_since_rtp_any"] = days
                record["post_rtp_60d_any"] = int(0 <= days < 60)
                record["post_rtp_120d_any"] = int(0 <= days < 120)
        else:
            record["active_episode_any"] = 0
            record["days_since_rtp_any"] = np.nan
            record["post_rtp_60d_any"] = 0
            record["post_rtp_120d_any"] = 0
        records.append(record)
    features = pd.DataFrame.from_records(records)
    return pd.concat(
        [entries_with_episodes.reset_index(drop=True), features.drop(columns="row_order")],
        axis=1,
    )


def build_outcome_labels(entries: pd.DataFrame, snaps_with_dates: pd.DataFrame) -> pd.DataFrame:
    snapped = snaps_with_dates.copy()
    snapped["total_snaps"] = snapped[["offense_snaps", "defense_snaps", "st_snaps"]].sum(axis=1)
    participation = (
        snapped.groupby(["season", "week", "team", "gsis_id"], observed=True)
        .agg(
            total_snaps=("total_snaps", "max"),
            offense_share=("offense_pct", "max"),
            defense_share=("defense_pct", "max"),
        )
        .reset_index()
    )
    frame = entries.merge(participation, on=["season", "week", "team", "gsis_id"], how="left")
    dnp = frame["total_snaps"].isna() | frame["total_snaps"].le(0)
    play_share = frame[["offense_share", "defense_share"]].max(axis=1).fillna(0.0)
    limited = (~dnp) & play_share.lt(0.5)
    frame["dnp_or_limited"] = (dnp | limited).astype(int)
    return frame.drop(columns=["total_snaps", "offense_share", "defense_share"])


MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "ever_injured_named",
    "n_prior_episodes_any",
    "ss_prior_episode_any",
    "active_episode_any",
    "post_rtp_60d_any",
    "post_rtp_120d_any",
    "days_since_rtp_any",
    *tuple(
        name
        for cls in INJURY_CLASSES
        for name in (
            f"post_rtp_120d_{cls}",
            f"n_prior_episodes_{cls}",
            f"ss_prior_episode_{cls}",
        )
    ),
)


def design_matrix(frame: pd.DataFrame, base_probabilities: pd.Series) -> pd.DataFrame:
    clipped = base_probabilities.clip(1e-3, 1 - 1e-3)
    matrix = pd.DataFrame(index=frame.index)
    matrix["logit_p_base"] = np.log(clipped / (1 - clipped))
    for column in MODEL_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if column.startswith("days_since_rtp"):
            matrix[column] = values.fillna(999.0)
        else:
            matrix[column] = values.fillna(0.0)
    return matrix


def fit_and_evaluate(frame: pd.DataFrame) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_probabilities = frame["p_base"]
    matrix = design_matrix(frame, base_probabilities)
    labels = frame["dnp_or_limited"].to_numpy(dtype=int)
    seasons = frame["season"].to_numpy(dtype=int)
    splits = {
        tag: seasons == season
        for tag, season in (("train", TRAIN_SEASON), ("val", VAL_SEASON), ("test", TEST_SEASON))
    }
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    model.fit(matrix[splits["train"]], labels[splits["train"]])
    base_only_columns = ["logit_p_base"]
    base_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    base_model.fit(matrix.loc[:, base_only_columns][splits["train"]], labels[splits["train"]])
    full_by_row = model.predict_proba(matrix)[:, 1]
    refit_base_by_row = base_model.predict_proba(matrix.loc[:, base_only_columns])[:, 1]
    base_by_row = base_probabilities.to_numpy(dtype=float)
    results: dict[str, object] = {}
    for tag in ("train", "val", "test"):
        mask = splits[tag]
        actual = labels[mask]
        brier_base = float(np.square(base_by_row[mask] - actual).mean())
        brier_refit_base = float(np.square(refit_base_by_row[mask] - actual).mean())
        brier_full = float(np.square(full_by_row[mask] - actual).mean())
        results[tag] = {
            "player_games": int(mask.sum()),
            "base_rate": float(actual.mean()),
            "brier_base": brier_base,
            "brier_refit_base": brier_refit_base,
            "brier_full": brier_full,
            "brier_delta_full_minus_refit_base": brier_full - brier_refit_base,
        }
    calibration: dict[str, object] = {}
    for tag in ("val", "test"):
        sub = frame.loc[splits[tag], ["primary_class"]].copy()
        sub["p_base"] = base_by_row[splits[tag]]
        sub["p_refit_base"] = refit_base_by_row[splits[tag]]
        sub["p_full"] = full_by_row[splits[tag]]
        sub["y"] = labels[splits[tag]]
        rows = []
        for cls, group in sub.groupby("primary_class", observed=True):
            rows.append(
                {
                    "primary_class": str(cls),
                    "n": len(group),
                    "observed": float(group["y"].mean()),
                    "mean_p_base": float(group["p_base"].mean()),
                    "mean_p_full": float(group["p_full"].mean()),
                    "brier_base": float(np.square(group["p_base"] - group["y"]).mean()),
                    "brier_refit_base": float(np.square(group["p_refit_base"] - group["y"]).mean()),
                    "brier_full": float(np.square(group["p_full"] - group["y"]).mean()),
                }
            )
        calibration[tag] = rows
    results["calibration_by_class"] = calibration
    coefficients = pd.Series(model.named_steps["logisticregression"].coef_[0], index=matrix.columns)
    results["coefficients"] = {key: round(float(value), 4) for key, value in coefficients.items()}
    return results


def build_baseline_probabilities(entries: pd.DataFrame, snaps_with_ids: pd.DataFrame) -> pd.Series:
    from nfl_ats.availability import (
        availability_rate_lookup,
        build_availability_outcomes,
        build_season_lagged_availability_rates,
        learned_unavailability,
    )

    games = load_availability_games()
    injuries_main = pd.read_parquet(INJURIES_MAIN_PATH)
    outcomes = build_availability_outcomes(injuries_main, snaps_with_ids, games)
    rates = build_season_lagged_availability_rates(
        outcomes, target_seasons=[TRAIN_SEASON, VAL_SEASON, TEST_SEASON]
    )
    lookup = availability_rate_lookup(rates)
    probabilities = []
    for row in entries.itertuples(index=False):
        probability = learned_unavailability(
            lookup,
            target_season=int(row.season),
            report_status=row.game_status,
            practice_status=row.practice_status,
            position=row.position,
        )
        probabilities.append(0.35 if probability is None else float(probability))
    return pd.Series(probabilities, index=entries.index, name="p_base")


def build_hazard_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cls in (*INJURY_CLASSES, OTHER_LABEL):
        subset = episodes.loc[episodes["body_part_class"].eq(cls)].sort_values(
            ["gsis_id", "start_date"]
        )
        if subset.empty:
            continue
        next_start = subset.groupby("gsis_id", observed=True)["start_date"].shift(-1)
        merged = subset.assign(next_start=next_start)
        merged["gap_to_next_days"] = (merged["next_start"] - merged["end_date"]).dt.days
        rows.append(
            {
                "body_part_class": cls,
                "episodes": len(merged),
                "players_with_episode": int(merged["gsis_id"].nunique()),
                "share_with_later_same_class_episode": float(merged["next_start"].notna().mean()),
                "median_days_between_episodes": float(merged["gap_to_next_days"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_incidence_ratios(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cls in INJURY_CLASSES:
        new_case = feature_frame["primary_class"].eq(cls) & feature_frame["is_episode_start"]
        prior = feature_frame[f"n_prior_episodes_{cls}"].gt(0)
        recent = feature_frame[f"post_rtp_120d_{cls}"].eq(1)
        n_prior_exposed = int(prior.sum())
        n_unexposed = int((~prior).sum())
        n_recent = int(recent.sum())
        rate_prior = (
            int((new_case & prior).sum()) / n_prior_exposed * 100 if n_prior_exposed else np.nan
        )
        rate_no = int((new_case & ~prior).sum()) / n_unexposed * 100 if n_unexposed else np.nan
        rate_recent = int((new_case & recent).sum()) / n_recent * 100 if n_recent else np.nan
        rows.append(
            {
                "body_part_class": cls,
                "new_cases": int(new_case.sum()),
                "presences_with_prior_history": n_prior_exposed,
                "rate_per100_prior_history": rate_prior,
                "presences_without_history": n_unexposed,
                "rate_per100_no_history": rate_no,
                "rr_same_history_vs_none": (
                    rate_prior / rate_no if n_prior_exposed and n_unexposed and rate_no else np.nan
                ),
                "presences_within_120d_post_rtp": n_recent,
                "rate_per100_recent_120d": rate_recent,
                "rr_recent_120d_vs_none": (
                    rate_recent / rate_no if n_recent and n_unexposed and rate_no else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def split_half_reliability(feature_frame: pd.DataFrame) -> dict[str, object]:
    frame = feature_frame.sort_values(["gsis_id", "game_date"]).copy()
    signal = frame["post_rtp_120d_any"].astype(float)
    labels = frame["dnp_or_limited"].to_numpy(dtype=float)
    parity = frame.groupby("gsis_id", observed=True).cumcount() % 2
    half_effects: dict[str, object] = {}
    for value in (0, 1):
        mask = (parity == value).to_numpy()
        flagged = mask & signal.eq(1).to_numpy()
        unflagged = mask & signal.eq(0).to_numpy()
        half_effects[str(value)] = {
            "n": int(mask.sum()),
            "risk_flagged": float(labels[flagged].mean()) if flagged.any() else np.nan,
            "risk_unflagged": float(labels[unflagged].mean()) if unflagged.any() else np.nan,
        }
    means_odd = signal.where(parity == 0, np.nan).groupby(frame["gsis_id"]).mean()
    means_even = signal.where(parity == 1, np.nan).groupby(frame["gsis_id"]).mean()
    counts = frame.assign(parity=parity).groupby(["gsis_id", "parity"]).size().unstack(fill_value=0)
    if {0, 1}.issubset(counts.columns):
        eligible = counts.index[counts[0].ge(2) & counts[1].ge(2)]
    else:
        eligible = counts.index[:0]
    paired = (
        pd.concat([means_odd.rename("odd_half"), means_even.rename("even_half")], axis=1)
        .loc[eligible]
        .dropna()
    )
    correlation = float(paired["odd_half"].corr(paired["even_half"])) if len(paired) > 2 else np.nan
    return {
        "signal": "post_rtp_120d_any",
        "population": "NFL.com report entries, 2022-2024",
        "players_eligible": len(paired),
        "pearson_r_odd_even": correlation,
        "half_effects_dnp_or_limited": half_effects,
    }


def main() -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    team_games = load_schedules()
    rosters = pd.read_parquet(ROSTERS_PATH)
    roster_bridge, roster_global = build_roster_bridge(rosters)

    raw_entries = load_nflcom_entries(team_games, roster_bridge, roster_global)
    report_entries = len(raw_entries)
    entries = raw_entries.loc[raw_entries["gsis_id"].notna()].copy()

    class_counts = entries["body_part_class"].value_counts()
    unmapped_mask = entries["body_part_class"].eq(UNMAPPED_LABEL)
    text_mask = entries["injury"].notna()
    classifier_report = {
        "source": str(NFLCOM_INJURIES_PATH.relative_to(REPO_ROOT)),
        "source_rows_total": len(pd.read_parquet(NFLCOM_INJURIES_PATH)),
        "report_entries": report_entries,
        "matched_to_gsis_id": len(entries),
        "match_rate": round(len(entries) / report_entries, 4),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "unmapped_count": int(unmapped_mask.sum()),
        "unmapped_rate_all_entries": round(float(unmapped_mask.mean()), 4),
        "unmapped_rate_given_text": round(
            float(entries.loc[text_mask, "body_part_class"].eq(UNMAPPED_LABEL).mean()), 4
        ),
    }
    pd.DataFrame([classifier_report["class_counts"]]).to_csv(
        ARTIFACT_DIR / "classifier_class_counts.csv", index=False
    )
    unmapped_samples = (
        entries.loc[text_mask & unmapped_mask, ["player", "injury"]]
        .drop_duplicates("injury")
        .head(50)
    )
    unmapped_samples.to_csv(ARTIFACT_DIR / "classifier_unmapped_samples.csv", index=False)

    from nfl_ats.players import attach_snap_player_ids

    snaps = pd.read_parquet(SNAPS_PATH)
    snaps = attach_snap_player_ids(snaps, rosters)
    snaps = snaps.merge(
        team_games[["game_id", "kickoff"]]
        .rename(columns={"kickoff": "game_date"})
        .drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    entries_with_episodes = assign_episodes(entries)
    episodes = build_episode_table(entries_with_episodes)
    featured = build_recurrence_features(entries_with_episodes, episodes, build_played_games(snaps))
    labeled = build_outcome_labels(featured, snaps)
    labeled["p_base"] = build_baseline_probabilities(labeled, snaps)

    eval_frame = labeled.loc[
        labeled["season"].isin((TRAIN_SEASON, VAL_SEASON, TEST_SEASON))
    ].reset_index(drop=True)
    validation = fit_and_evaluate(eval_frame)

    hazard = build_hazard_table(episodes)
    incidence = build_incidence_ratios(labeled)
    reliability = split_half_reliability(labeled)

    hazard.to_csv(ARTIFACT_DIR / "hazard_table.csv", index=False)
    incidence.to_csv(ARTIFACT_DIR / "incidence_ratio_table.csv", index=False)
    keep_columns = [
        "season",
        "week",
        "team",
        "player",
        "position",
        "gsis_id",
        "game_date",
        "injury",
        "body_part_class",
        "primary_class",
        "episode_id",
        "is_episode_start",
        "dnp_or_limited",
        "p_base",
        *MODEL_FEATURE_COLUMNS,
    ]
    labeled[[column for column in keep_columns if column in labeled.columns]].to_parquet(
        ARTIFACT_DIR / "player_game_features.parquet", index=False
    )

    summary = {
        "classifier": classifier_report,
        "validation": validation,
        "hazard_table": hazard.to_dict(orient="records"),
        "incidence_ratios": incidence.to_dict(orient="records"),
        "split_half_reliability": reliability,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


def _synthetic_entries(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def test_classify_core_terms() -> None:
    assert classify_injury_text("Hamstring") == "hamstring"
    assert classify_injury_text("Knee") == "knee"
    assert classify_injury_text("ACL") == "knee"
    assert classify_injury_text("Patella Tendon") == "knee"
    assert classify_injury_text("Ankle") == "ankle"
    assert classify_injury_text("Concussion") == "concussion"
    assert classify_injury_text("gameday concussion protocol evaluation") == "concussion"
    assert classify_injury_text("right Shoulder") == "shoulder"
    assert classify_injury_text("Collarbone") == "shoulder"
    assert classify_injury_text("Groin") == "other"
    assert classify_injury_text("Illness") == "other"
    assert classify_injury_text("coach's decision") == "other"
    assert classify_injury_text("Not injury related - personal matter") == "other"


def test_classifier_unmapped_is_honest() -> None:
    assert classify_injury_text(None) == "unmapped"
    assert classify_injury_text(float("nan")) == "unmapped"
    assert classify_injury_text("--") == "unmapped"
    assert classify_injury_text("xyzzy nonsense words") == "unmapped"


def test_normalize_player_name_strips_suffixes() -> None:
    assert normalize_player_name("Carlton Davis III") == normalize_player_name("Carlton Davis")
    assert normalize_player_name("Deebo Samuel Sr.") == normalize_player_name("Deebo Samuel")


def test_assign_episodes_uses_gap_rule() -> None:
    entries = _synthetic_entries(
        [
            {"gsis_id": "P1", "body_part_class": "hamstring", "game_date": "2022-09-11"},
            {"gsis_id": "P1", "body_part_class": "hamstring", "game_date": "2022-09-18"},
            {"gsis_id": "P1", "body_part_class": "hamstring", "game_date": "2022-10-16"},
        ]
    )
    episodes_frame = assign_episodes(entries)
    assert episodes_frame["episode_seq"].tolist() == [1, 1, 2]
    assert episodes_frame["is_episode_start"].tolist() == [True, False, True]


def test_point_in_time_features_exclude_same_and_future_rows() -> None:
    entries = _synthetic_entries(
        [
            {
                "gsis_id": "P1",
                "body_part_class": "hamstring",
                "game_date": "2022-09-11",
                "season": 2022,
            },
            {
                "gsis_id": "P1",
                "body_part_class": "hamstring",
                "game_date": "2022-10-30",
                "season": 2022,
            },
        ]
    )
    played = pd.DataFrame(
        {
            "gsis_id": ["P1"],
            "game_id": ["g1"],
            "game_date": pd.to_datetime(["2022-09-18"]),
        }
    )
    episodes_table = build_episode_table(assign_episodes(entries))
    featured = build_recurrence_features(entries, episodes_table, played)
    first = featured.iloc[0]
    assert first["n_prior_episodes_hamstring"] == 0
    assert first["ever_injured_named"] == 0
    assert pd.isna(first["days_since_rtp_hamstring"])
    second = featured.iloc[1]
    assert second["n_prior_episodes_hamstring"] == 1
    assert second["ever_injured_named"] == 1
    assert second["returned_pre_game_hamstring"] == 1
    assert second["days_since_rtp_hamstring"] == 42
    assert second["post_rtp_60d_hamstring"] == 1
    assert second["post_rtp_120d_hamstring"] == 1
    assert second["active_episode_hamstring"] == 0


def test_recurrence_flags_respect_day_thresholds() -> None:
    def features_for(gap_days: int) -> pd.Series:
        game_date = pd.Timestamp("2022-10-01")
        rtp_date = game_date - pd.Timedelta(days=gap_days)
        entries = _synthetic_entries(
            [
                {
                    "gsis_id": "P1",
                    "body_part_class": "knee",
                    "game_date": rtp_date - pd.Timedelta(days=7),
                    "season": 2022,
                },
                {
                    "gsis_id": "P1",
                    "body_part_class": "knee",
                    "game_date": game_date,
                    "season": 2022,
                },
            ]
        )
        played = pd.DataFrame(
            {
                "gsis_id": ["P1"],
                "game_id": ["g1"],
                "game_date": pd.to_datetime([rtp_date]),
            }
        )
        episodes_table = build_episode_table(assign_episodes(entries))
        featured = build_recurrence_features(entries, episodes_table, played)
        return featured.iloc[1]

    at_59 = features_for(59)
    assert at_59["post_rtp_60d_knee"] == 1
    assert at_59["post_rtp_120d_knee"] == 1
    at_60 = features_for(60)
    assert at_60["post_rtp_60d_knee"] == 0
    assert at_60["post_rtp_120d_knee"] == 1
    at_120 = features_for(120)
    assert at_120["post_rtp_60d_knee"] == 0
    assert at_120["post_rtp_120d_knee"] == 0


def test_outcome_labels_dnp_and_limited() -> None:
    snaps = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "week": [3, 3, 3],
            "team": ["BUF", "BUF", "BUF"],
            "gsis_id": ["P1", "P2", "P3"],
            "offense_snaps": [0.0, 10.0, 40.0],
            "defense_snaps": [0.0, 0.0, 0.0],
            "st_snaps": [0.0, 0.0, 0.0],
            "offense_pct": [0.0, 0.2, 0.9],
            "defense_pct": [0.0, 0.0, 0.0],
        }
    )
    entries = _synthetic_entries(
        [
            {"season": 2022, "week": 3, "team": "BUF", "gsis_id": "P1"},
            {"season": 2022, "week": 3, "team": "BUF", "gsis_id": "P2"},
            {"season": 2022, "week": 3, "team": "BUF", "gsis_id": "P3"},
            {"season": 2022, "week": 3, "team": "BUF", "gsis_id": "P4"},
        ]
    )
    labeled = build_outcome_labels(entries, snaps)
    assert labeled.set_index("gsis_id")["dnp_or_limited"].to_dict() == {
        "P1": 1,
        "P2": 1,
        "P3": 0,
        "P4": 1,
    }


def test_design_matrix_handles_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "ever_injured_named": [0, 1],
            "n_prior_episodes_any": [0.0, 2.0],
            "ss_prior_episode_any": [0, 1],
            "active_episode_any": [0, 0],
            "post_rtp_60d_any": [0, 1],
            "post_rtp_120d_any": [0, 1],
            "days_since_rtp_any": [np.nan, 30.0],
            **{
                name: [0, 0]
                for cls in INJURY_CLASSES
                for name in (
                    f"post_rtp_120d_{cls}",
                    f"n_prior_episodes_{cls}",
                    f"ss_prior_episode_{cls}",
                )
            },
        }
    )
    base = pd.Series([0.35, 0.35])
    matrix = design_matrix(frame, base)
    assert matrix.loc[0, "days_since_rtp_any"] == 999.0
    assert matrix.loc[1, "logit_p_base"] < 0
    assert not matrix.isna().any().any()


if __name__ == "__main__":
    main()

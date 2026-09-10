from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfl_ats.cfb_features import cfb_competitive_plays
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.evidence_conventions import probability_positive_from_draws

CFB_QB_STATE_SPAN: int = 12
CFB_QB_MIN_DROPBACKS: int = 20
CFB_QB_MIN_GAME_DROPBACKS: int = 5
CFB_PASS_RATE_SPAN: int = 8
CFB_PASS_RATE_MIN_PERIODS: int = 3

CFB_QB_DEPENDENCE_METRICS: tuple[str, ...] = (
    "off_pass_rate",
    "qb_starter_epa_per_dropback",
    "qb_dependence_interaction",
)
CFB_QB_DEPENDENCE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{metric}" for metric in CFB_QB_DEPENDENCE_METRICS for side in ("home", "away", "diff")
)

RELIABILITY_NO_SPLIT_HALF_EXAMPLES: dict[str, float] = {
    "coach_ats_reputation": 0.063,
    "play_epa_dispersion": 0.014,
}
RELIABILITY_CLEARED_EXAMPLES: dict[str, float] = {
    "injury_value_lost_temporal": 0.9325,
    "cfb_role_continuity_dropback": 0.719,
    "cfb_role_continuity_carry": 0.680,
}


def _game_id_key(values: pd.Series) -> pd.Series:

    return pd.to_numeric(values, errors="raise").astype("int64").astype(str)


_QB_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "team_id",
    "passer_player_id",
    "qb_dropbacks",
    "qb_epa_per_dropback",
)


def build_cfb_qb_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:

    require_columns(pbp, ("passer_player_id",), "cfb play_by_play (qb dependence)")
    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[
        plays["competitive_play"] & plays["pass"] & plays["passer_player_id"].notna()
    ].copy()
    if plays.empty:
        return pd.DataFrame(columns=_QB_GAME_COLUMNS)
    plays["team_id"] = pd.to_numeric(plays["pos_team_id"], errors="raise").astype("int64")
    plays["passer_player_id"] = plays["passer_player_id"].astype(str)
    plays["EPA"] = pd.to_numeric(plays["EPA"], errors="coerce")

    grouped = (
        plays.groupby(["game_id", "season", "week", "team_id", "passer_player_id"], sort=False)
        .agg(qb_dropbacks=("EPA", "size"), total_epa=("EPA", "sum"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["qb_dropbacks"].ge(CFB_QB_MIN_GAME_DROPBACKS)].copy()
    grouped["qb_epa_per_dropback"] = grouped["total_epa"] / grouped["qb_dropbacks"]
    return grouped.loc[:, list(_QB_GAME_COLUMNS)].sort_values(
        ["season", "week", "game_id", "team_id"]
    )


def build_cfb_qb_states(qb_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:

    require_columns(qb_games, _QB_GAME_COLUMNS, "cfb qb game metrics")
    require_columns(games, ("game_id", "gameday"), "cfb canonical games")

    dates = games.loc[:, ["game_id", "gameday"]].drop_duplicates("game_id")
    states = qb_games.merge(dates, on="game_id", how="left", validate="many_to_one")
    states = states.loc[states["gameday"].notna()].copy()
    states["gameday"] = pd.to_datetime(states["gameday"], errors="raise")
    states = states.sort_values(["passer_player_id", "gameday", "game_id"])

    alpha = 2.0 / (CFB_QB_STATE_SPAN + 1.0)
    output = pd.Series(np.nan, index=states.index, dtype="float64")
    for _, group in states.groupby("passer_player_id", sort=False):
        current = math.nan
        career_dropbacks = 0.0
        for index, row in group.iterrows():
            value = float(row["qb_epa_per_dropback"])
            if math.isfinite(value):
                current = (
                    value if not math.isfinite(current) else alpha * value + (1.0 - alpha) * current
                )
            career_dropbacks += float(row["qb_dropbacks"])
            if career_dropbacks >= CFB_QB_MIN_DROPBACKS:
                output.at[index] = current
    states["state_qb_epa_per_dropback"] = output
    states["career_dropbacks"] = states.groupby("passer_player_id", sort=False)[
        "qb_dropbacks"
    ].cumsum()
    return states.reset_index(drop=True)


def build_cfb_pass_rate_team_games(pbp: pd.DataFrame) -> pd.DataFrame:

    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[plays["competitive_play"]].copy()
    if plays.empty:
        return pd.DataFrame(columns=("game_id", "season", "week", "team_id", "off_pass_rate"))
    plays["team_id"] = pd.to_numeric(plays["pos_team_id"], errors="raise").astype("int64")
    grouped = (
        plays.groupby(["game_id", "season", "week", "team_id"], sort=False)
        .agg(off_pass_rate=("pass", "mean"))
        .reset_index()
    )
    return grouped.sort_values(["season", "week", "game_id", "team_id"]).reset_index(drop=True)


def build_cfb_pass_rate_states(team_games: pd.DataFrame) -> pd.DataFrame:

    require_columns(
        team_games, ("game_id", "team_id", "gameday", "off_pass_rate"), "pass rate team games"
    )
    states = team_games.copy().sort_values(["team_id", "gameday", "game_id"])
    states["off_pass_rate"] = pd.to_numeric(states["off_pass_rate"], errors="coerce")
    alpha = 2.0 / (CFB_PASS_RATE_SPAN + 1.0)
    output = pd.Series(np.nan, index=states.index, dtype="float64")
    for _, group in states.groupby("team_id", sort=False):
        current = math.nan
        observations = 0
        for index, row in group.iterrows():
            value = float(row["off_pass_rate"])
            if math.isfinite(value):
                current = (
                    value if not math.isfinite(current) else alpha * value + (1.0 - alpha) * current
                )
                observations += 1
            if observations >= CFB_PASS_RATE_MIN_PERIODS:
                output.at[index] = current
    states["state_off_pass_rate"] = output
    return states.loc[:, ["game_id", "team_id", "gameday", "state_off_pass_rate"]].reset_index(
        drop=True
    )


def attach_cfb_qb_dependence(
    games: pd.DataFrame,
    qb_games: pd.DataFrame,
    qb_states: pd.DataFrame,
    pass_rate_states: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(games, ("game_id", "gameday", "home_id", "away_id"), "cfb canonical games")

    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result = result.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    qb_games_keyed = qb_games.copy()
    qb_games_keyed["game_id"] = _game_id_key(qb_games_keyed["game_id"])
    qb_games_keyed["passer_player_id"] = qb_games_keyed["passer_player_id"].astype(str)
    qb_appearances: dict[tuple[str, int], pd.DataFrame] = {
        (str(group["game_id"].iloc[0]), int(group["team_id"].iloc[0])): group.sort_values(
            "qb_dropbacks", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
        for _, group in qb_games_keyed.groupby(["game_id", "team_id"], sort=False)
    }

    qb_states_keyed = qb_states.copy()
    qb_states_keyed["passer_player_id"] = qb_states_keyed["passer_player_id"].astype(str)
    qb_histories: dict[str, pd.DataFrame] = {
        str(player): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for player, group in qb_states_keyed.groupby("passer_player_id", sort=False)
    }

    pass_rate_groups: dict[int, pd.DataFrame] = {
        int(group["team_id"].iloc[0]): group.sort_values(["gameday", "game_id"]).reset_index(
            drop=True
        )
        for _, group in pass_rate_states.groupby("team_id", sort=False)
    }

    for side in ("home", "away"):
        result[f"{side}_off_pass_rate"] = np.nan
        result[f"{side}_qb_starter_epa_per_dropback"] = np.nan

    latest_passer: dict[int, str] = {}

    for index, game in result.iterrows():
        game_id = str(int(game["game_id"]))
        game_date = np.datetime64(pd.Timestamp(game["gameday"]), "ns")
        for side in ("home", "away"):
            team_id = int(game[f"{side}_id"])

            pass_history = pass_rate_groups.get(team_id)
            if pass_history is not None and not pass_history.empty:
                dates = pass_history["gameday"].to_numpy(dtype="datetime64[ns]")
                position = int(np.searchsorted(dates, game_date, side="left")) - 1
                if position >= 0:
                    result.at[index, f"{side}_off_pass_rate"] = pass_history.iloc[position][
                        "state_off_pass_rate"
                    ]

            passer_id = latest_passer.get(team_id)
            if passer_id is not None:
                qb_history = qb_histories.get(passer_id)
                if qb_history is not None and not qb_history.empty:
                    dates = qb_history["gameday"].to_numpy(dtype="datetime64[ns]")
                    position = int(np.searchsorted(dates, game_date, side="left")) - 1
                    if position >= 0:
                        result.at[index, f"{side}_qb_starter_epa_per_dropback"] = qb_history.iloc[
                            position
                        ]["state_qb_epa_per_dropback"]

        for side in ("home", "away"):
            team_id = int(game[f"{side}_id"])
            appearances = qb_appearances.get((game_id, team_id))
            if appearances is not None and not appearances.empty:
                latest_passer[team_id] = str(appearances.iloc[0]["passer_player_id"])

    for side in ("home", "away"):
        result[f"{side}_off_pass_rate"] = pd.to_numeric(
            result[f"{side}_off_pass_rate"], errors="coerce"
        )
        result[f"{side}_qb_starter_epa_per_dropback"] = pd.to_numeric(
            result[f"{side}_qb_starter_epa_per_dropback"], errors="coerce"
        )
        result[f"{side}_qb_dependence_interaction"] = (
            result[f"{side}_qb_starter_epa_per_dropback"] * result[f"{side}_off_pass_rate"]
        )
    for metric in CFB_QB_DEPENDENCE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    return result


def build_and_attach_cfb_qb_dependence(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:

    qb_games = build_cfb_qb_game_metrics(pbp)
    qb_states = build_cfb_qb_states(qb_games, games)
    pass_rate_team_games = build_cfb_pass_rate_team_games(pbp)
    dates = games.loc[:, ["game_id", "gameday"]].drop_duplicates("game_id").copy()
    dates["game_id"] = _game_id_key(dates["game_id"])
    pass_rate_team_games = pass_rate_team_games.copy()
    pass_rate_team_games["game_id"] = _game_id_key(pass_rate_team_games["game_id"])
    pass_rate_team_games = pass_rate_team_games.merge(
        dates, on="game_id", how="left", validate="many_to_one"
    )
    pass_rate_team_games = pass_rate_team_games.loc[pass_rate_team_games["gameday"].notna()].copy()
    pass_rate_states = build_cfb_pass_rate_states(pass_rate_team_games)
    return attach_cfb_qb_dependence(games, qb_games, qb_states, pass_rate_states)


def _reshape_team_game_long(features: pd.DataFrame, metric: str) -> pd.DataFrame:

    pieces: list[pd.DataFrame] = []
    for side in ("home", "away"):
        piece = features.loc[:, ["game_id", "season", "week", f"{side}_id"]].rename(
            columns={f"{side}_id": "team_id"}
        )
        piece[metric] = pd.to_numeric(features[f"{side}_{metric}"], errors="coerce")
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    long["team_id"] = pd.to_numeric(long["team_id"], errors="coerce").astype("Int64")
    long["season"] = pd.to_numeric(long["season"], errors="raise").astype(int)
    long["week"] = pd.to_numeric(long["week"], errors="raise").astype(int)
    return long


def split_half_reliability(
    long: pd.DataFrame, metric: str, *, seed: int, n_boot: int = 4000
) -> dict[str, Any]:

    subset = long.loc[long[metric].notna()].copy()
    subset["half"] = np.where(subset["week"] % 2 == 0, "even", "odd")
    means = subset.groupby(["team_id", "season", "half"])[metric].mean().unstack("half")
    counts = subset.groupby(["team_id", "season", "half"]).size().unstack("half")
    has_both = (counts.get("odd", pd.Series(dtype=float)).fillna(0) >= 2) & (
        counts.get("even", pd.Series(dtype=float)).fillna(0) >= 2
    )
    means = means.dropna()
    has_both = has_both.reindex(means.index).fillna(False)
    means = means.loc[has_both]

    n = len(means)
    if n < 3:
        return {
            "metric": metric,
            "n_team_seasons": n,
            "pearson_r": math.nan,
            "pearson_r_ci95": [math.nan, math.nan],
            "spearman_rho": math.nan,
            "spearman_brown_full_length_reliability": math.nan,
            "probability_positive": math.nan,
        }

    odd_vals = means["odd"].to_numpy(dtype=float)
    even_vals = means["even"].to_numpy(dtype=float)
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    rho = float(spearmanr(odd_vals, even_vals).correlation)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        selected = rng.integers(0, n, size=n)
        boots[draw] = np.corrcoef(odd_vals[selected], even_vals[selected])[0, 1]
    spearman_brown = (2.0 * r) / (1.0 + r) if r > -1.0 else math.nan
    return {
        "metric": metric,
        "n_team_seasons": n,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_rho": rho,
        "spearman_brown_full_length_reliability": spearman_brown,
        "probability_positive": float(probability_positive_from_draws(boots)),
    }


@dataclass(frozen=True)
class ReliabilityAudit:
    interaction: dict[str, Any]
    qb_starter_epa_per_dropback: dict[str, Any]
    off_pass_rate: dict[str, Any]
    no_split_half_examples: dict[str, float]
    cleared_examples: dict[str, float]


def cfb_qb_dependence_reliability(
    features: pd.DataFrame, *, seed_interaction: int = 1, seed_qb: int = 2, seed_pass_rate: int = 3
) -> ReliabilityAudit:

    required = {
        f"{side}_{metric}" for metric in CFB_QB_DEPENDENCE_METRICS for side in ("home", "away")
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB qb-dependence features are missing columns: {', '.join(missing)}"
        )

    interaction_long = _reshape_team_game_long(features, "qb_dependence_interaction")
    qb_long = _reshape_team_game_long(features, "qb_starter_epa_per_dropback")
    pass_rate_long = _reshape_team_game_long(features, "off_pass_rate")

    return ReliabilityAudit(
        interaction=split_half_reliability(
            interaction_long, "qb_dependence_interaction", seed=seed_interaction
        ),
        qb_starter_epa_per_dropback=split_half_reliability(
            qb_long, "qb_starter_epa_per_dropback", seed=seed_qb
        ),
        off_pass_rate=split_half_reliability(pass_rate_long, "off_pass_rate", seed=seed_pass_rate),
        no_split_half_examples=dict(RELIABILITY_NO_SPLIT_HALF_EXAMPLES),
        cleared_examples=dict(RELIABILITY_CLEARED_EXAMPLES),
    )

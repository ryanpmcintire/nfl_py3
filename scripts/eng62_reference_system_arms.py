from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.market_data import POOL_TIMEZONE
from nfl_ats.pick_probability import (
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    active_market_move_feature_version,
    load_pick_probability_model,
)
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _standardisers,
    build_fit_population,
)
from nfl_ats.pool_hook import (
    POOL_OFFSET,
    crossed_push_mass,
    hook_value,
    push_mass_table,
    whole_mask,
)
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eng62_tuesday_reference_oos", REPO / "scripts" / "eng62_tuesday_reference_oos.py"
)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ART = base.ART
OUT = base.OUT
SEASONS = base.SEASONS
KEY = base.KEY
LIVE_SLOT = pd.Timedelta(hours=14, minutes=25)
PREKICK = pd.Timedelta(days=5, hours=12, minutes=45)
DAY = pd.Timedelta(days=1)
HOURLY = "intraday_hourly"
BASELINES = ("late", "slot", "noon")
ANCHOR = {"A": "fit", "B": "live", "C": "nearest", "D": "nearest", "E": "nearest", "A_fold": "fit"}
STAGE = {"A": "A", "B": "B", "C": "A", "D": "D", "E": "E", "A_fold": "A"}
ARMS = tuple(ANCHOR)
CONTRASTS = (
    ("B", "A"),
    ("C", "A"),
    ("D", "A"),
    ("E", "A"),
    ("C", "B"),
    ("D", "B"),
    ("E", "B"),
    ("D", "C"),
    ("E", "C"),
    ("E", "A_fold"),
)
SUBSETS = ("all", "moved", "rest")
UNITS = ("accuracy_points", "log_loss_improvement", "brier_improvement")
UNIT_SCALE = (100.0, 1.0, 1.0)
PRIMARY_POOL = "after"
NULL_REPS = {"D": 1000, "E": 300}
FAMILY = "eng62_reference_system_arms"
CITATIONS = {
    "LIVE_SLOT": "tests/scratch/eng62/baseline_age.py:39, the live Tuesday capture clock",
    "PREKICK": "tests/scratch/eng62/baseline_age.py:118, Sunday 12:45 ET move cutoff",
    "anchor_arms": "scripts/eng62_tuesday_reference_oos.py selections()",
}
PLAIN = {
    "A": "the way they were first set up, one early Tuesday morning and one late Tuesday night",
    "B": "both early on Tuesday afternoon, as the card does today",
    "C": "one at the pool's noon freeze and the other late Tuesday night",
    "D": "both at the pool's noon freeze with the weights left alone",
    "E": "both at the pool's noon freeze with the weights refitted on other seasons",
    "A_fold": "the first setup with its pool-line weight refitted on other seasons",
}
PLAIN_SUBSET = {
    "all": "",
    "moved": ", on games whose number moved on Tuesday afternoon or evening",
    "rest": ", on games whose number stayed put on Tuesday afternoon and evening",
}
LINES: list[str] = []


def emit(text: str) -> None:
    print(text)
    LINES.append(text)


def wall(day: pd.Series, delta: pd.Timedelta) -> pd.Series:
    return (day + delta).dt.tz_localize(POOL_TIMEZONE).dt.tz_convert("UTC")


def leader_moves(noon: pd.Series) -> pd.DataFrame:
    quotes = pd.read_parquet(
        base.QUOTES,
        columns=[
            "nflverse_game_id",
            "bookmaker_key",
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "home_spread_line",
            "commence_time_utc",
            "market",
            "decision_label",
        ],
    )
    quotes = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["decision_label"].eq(HOURLY)
        & quotes["bookmaker_key"].isin(LEADER_BOOKS)
        & quotes["nflverse_game_id"].isin(noon.index)
    ].copy()
    for column in ("observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc"):
        quotes[column] = pd.to_datetime(quotes[column], utc=True, errors="coerce")
    quotes["line"] = pd.to_numeric(quotes["home_spread_line"], errors="coerce")
    quotes["noon"] = pd.to_datetime(quotes["nflverse_game_id"].map(noon), utc=True)
    day = quotes["noon"].dt.tz_convert(POOL_TIMEZONE).dt.tz_localize(None).dt.normalize()
    quotes["tuesday"] = wall(day, pd.Timedelta(0))
    quotes["slot"] = wall(day, LIVE_SLOT)
    quotes["wednesday"] = wall(day, DAY)
    kickoff = quotes.groupby("nflverse_game_id")["commence_time_utc"].transform("min")
    cutoff = pd.concat([kickoff, wall(day, PREKICK)], axis=1).min(axis=1)
    observed = quotes["observed_at_utc"]
    quotes = quotes.loc[
        observed.ge(wall(day, -DAY))
        & observed.lt(cutoff)
        & quotes["bookmaker_last_update_utc"].le(observed)
        & np.isfinite(quotes["line"])
    ]
    quotes = quotes.sort_values([*KEY, "observed_at_utc"]).drop_duplicates(
        [*KEY, "observed_at_utc"]
    )
    early = quotes.loc[quotes["observed_at_utc"].lt(quotes["wednesday"])]
    tuesday = early.loc[early["observed_at_utc"].ge(early["tuesday"])]
    picks = {
        "late": early.groupby(KEY).tail(1),
        "slot": early.loc[early["observed_at_utc"].le(early["slot"])].groupby(KEY).tail(1),
        "noon": tuesday.assign(gap=(tuesday["observed_at_utc"] - tuesday["noon"]).abs())
        .sort_values(["gap", "observed_at_utc"])
        .groupby(KEY)
        .head(1),
    }
    post = (
        quotes.loc[quotes["observed_at_utc"].ge(quotes["wednesday"])]
        .groupby(KEY)
        .tail(1)
        .set_index(KEY)["line"]
    )
    columns = {}
    for name, rows in picks.items():
        indexed = rows.set_index(KEY)
        move = (post - indexed["line"]).dropna()
        lag = (indexed["observed_at_utc"] - indexed["noon"]).dt.total_seconds() / 3600.0
        game = move.index.get_level_values(0)
        columns[f"move_{name}"] = move.groupby(game).median()
        columns[f"books_{name}"] = move.groupby(game).size()
        columns[f"base_{name}"] = indexed["line"].reindex(move.index).groupby(game).median()
        columns[f"lag_{name}"] = lag.reindex(move.index).groupby(game).median()
    return pd.DataFrame(columns)


def fit_stage(train: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    fitted = train.loc[train["margin_vs_open"].ne(0.0)]
    mean, scale = _standardisers(fitted)
    beta = _fit_logit(
        _design(fitted, mean, scale),
        fitted["home_covered"].astype(float).to_numpy(),
        FIT_RIDGE,
    )
    return mean, scale, beta


def folds(train: pd.DataFrame, seasons: tuple[int, ...]) -> dict[int, tuple]:
    return {int(held): fit_stage(train.loc[train["season"] != held]) for held in seasons}


def natural(fit: tuple) -> dict[str, float]:
    return {name: float(fit[2][i + 1] / fit[1][name]) for i, name in enumerate(FIT_FEATURES)}


def inside(fit: tuple, frame: pd.DataFrame) -> np.ndarray:
    return _design(frame, fit[0], fit[1]) @ fit[2]


def outside(fits: dict[int, tuple], frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(frame), np.nan)
    season = frame["season"].to_numpy()
    for held, fit in fits.items():
        test = season == held
        result[test] = inside(fit, frame.loc[test])
    return result


def with_move(slim: pd.DataFrame, delta: np.ndarray, lost: np.ndarray) -> pd.DataFrame:
    frame = slim.copy()
    frame[MOVE_COLUMN] = np.where(lost, 0.0, slim[MOVE_COLUMN].to_numpy(dtype=float) + delta)
    frame[MOVE_AVAILABLE_COLUMN] = np.where(
        lost, 0.0, slim[MOVE_AVAILABLE_COLUMN].to_numpy(dtype=float)
    )
    return frame


def shuffled(
    delta: np.ndarray, eligible: np.ndarray, season: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = delta.copy()
    for year in SEASONS:
        ids = np.flatnonzero(eligible & (season == year))
        result[ids] = delta[rng.permutation(ids)]
    return result


def hooks(anchor: np.ndarray, target: np.ndarray, table: object) -> np.ndarray:
    mass = np.array([crossed_push_mass(a, p, table) for a, p in zip(anchor, target, strict=True)])
    return hook_value(np.sign(target - anchor), mass)


def hook_sets(
    rows: pd.DataFrame, target: np.ndarray, tables: dict[int, object]
) -> dict[str, dict[object, np.ndarray]]:
    season = rows["season"].to_numpy()
    result = {}
    for name in sorted(set(ANCHOR.values())):
        anchor = rows[f"{name}_anchor"].to_numpy(dtype=float)
        full = {held: hooks(anchor, target, tables[held]) for held in (*SEASONS, -1)}
        outer = np.zeros(len(rows))
        for held in SEASONS:
            outer[season == held] = full[held][season == held]
        full["outer"] = outer
        result[name] = full
    return result


def fold_coefficients(
    full: dict[object, np.ndarray],
    offset: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    season: np.ndarray,
) -> dict[int, float]:
    return {
        held: base.fit_coefficient(
            full[held][season != held],
            offset[season != held],
            outcome[season != held],
            weight[season != held],
        )
        for held in SEASONS
    }


def level(probability: np.ndarray, outcome: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    total = float(weight.sum())
    return {
        "games": total,
        "accuracy": float((weight * base.right(probability, outcome)).sum() / total),
        "log_loss": float((weight * base.loss(probability, outcome)).sum() / total),
        "brier": float((weight * (probability - outcome) ** 2).sum() / total),
    }


def point(
    first: np.ndarray, second: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    total = weight.sum()
    accuracy = (weight * (base.right(first, outcome) - base.right(second, outcome))).sum()
    gain = (weight * (base.loss(second, outcome) - base.loss(first, outcome))).sum()
    return float(100.0 * accuracy / total), float(gain / total)


def compare(
    first: np.ndarray,
    second: np.ndarray,
    rows: pd.DataFrame,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    part = rows.loc[mask].reset_index(drop=True)
    one = first[mask]
    two = second[mask]
    outcome = part["y"].to_numpy()
    weight = part["w"].to_numpy()
    season = part["season"].to_numpy()
    won = weight * base.right(one, outcome)
    beaten = weight * base.right(two, outcome)
    stack = np.column_stack(
        [
            won - beaten,
            weight * (base.loss(two, outcome) - base.loss(one, outcome)),
            weight * ((two - outcome) ** 2 - (one - outcome) ** 2),
            weight,
        ]
    )
    differ = (one >= 0.5) != (two >= 0.5)
    blocks = list(part.groupby(["season", "week"]).indices.values())
    sums = np.array([stack[b].sum(axis=0) for b in blocks])
    draws = rng.integers(0, len(blocks), (base.BOOTSTRAP_REPS, len(blocks)))
    totals = sums[draws].sum(axis=1)
    resampled = totals[:, :3] / totals[:, 3:4]
    centre = sums[:, :3].sum(axis=0) / sums[:, 3].sum()
    interval = np.percentile(resampled, [2.5, 97.5], axis=0)
    positive = (resampled > 0).mean(axis=0) + 0.5 * (resampled == 0).mean(axis=0)
    result: dict[str, object] = {
        "games": float(weight.sum()),
        "blocks": len(blocks),
        "decisive_games": float(weight[differ].sum()),
        "decisive_first_right": float(won[differ].sum()),
        "decisive_second_right": float(beaten[differ].sum()),
    }
    for index, unit in enumerate(UNITS):
        scale = UNIT_SCALE[index]
        result[unit] = float(scale * centre[index])
        result[f"{unit}_interval"] = [
            float(scale * interval[0, index]),
            float(scale * interval[1, index]),
        ]
        result[f"{unit}_probability_positive"] = float(positive[index])
    by_season = {}
    for year in SEASONS:
        own = season == year
        if not own.any():
            continue
        total = weight[own].sum()
        by_season[str(year)] = {
            "games": float(total),
            "accuracy_points": float(100.0 * stack[own, 0].sum() / total),
            "log_loss_improvement": float(stack[own, 1].sum() / total),
            "brier_improvement": float(stack[own, 2].sum() / total),
            "decisive_first_right": float(won[own & differ].sum()),
            "decisive_second_right": float(beaten[own & differ].sum()),
        }
    result["by_season"] = by_season
    return result


def percentile(null: np.ndarray, actual: float) -> float:
    return float((null < actual).mean() + 0.5 * (null == actual).mean())


def main() -> None:
    age = json.loads(base.AGE.read_text(encoding="utf-8"))
    fit_lag = (
        float(age["fit_offset_term"]["hook_rows_anchor_hours_after_tue_noon"]["max"])
        + base.AGE_FILE_RESOLUTION
    )
    live_lag = float(age["live_offset_term"]["anchor_age_hours_after_tue_noon"]["median"])
    model = load_pick_probability_model(ART)
    served = float(model.pool_hook)
    version = active_market_move_feature_version(ART)
    population, _ = build_fit_population(
        ART, REPO / "data", market_move_feature_version=version, keep_pushes=True
    )
    population = population.reset_index(drop=True)
    population["margin"] = population["margin_vs_open"] + population["tue_open_home_spread"]
    stamps = pd.concat(
        [pd.read_pickle(base.HARNESS / "u1_arch.pkl"), pd.read_pickle(base.HARNESS / "u1_pre.pkl")]
    )
    stamps = stamps.loc[stamps["season"].isin(SEASONS)]
    noon = stamps.groupby("nflverse_game_id")["noon"].first()
    leaders = leader_moves(noon)
    year = population["season"].to_numpy()
    scored = np.isin(year, SEASONS)
    stated = population[MOVE_COLUMN].to_numpy(dtype=float)
    have = population[MOVE_AVAILABLE_COLUMN].eq(1.0).to_numpy() & scored
    rep = {
        name: population["game_id"].map(leaders[f"move_{name}"]).to_numpy(dtype=float)
        for name in BASELINES
    }
    matched = have & np.isfinite(rep["late"])
    orientation = float(np.sign(np.corrcoef(stated[matched], rep["late"][matched])[0, 1]))
    if not np.isfinite(orientation) or orientation == 0.0:
        raise SystemExit("replicated move has no orientation against the served column")
    gap = np.abs(stated[matched] - orientation * rep["late"][matched])
    fidelity = {
        "games_with_served_move": int(have.sum()),
        "games_replicated": int(matched.sum()),
        "orientation": orientation,
        "share_equal": float((gap < 1e-9).mean()),
        "max_abs_gap": float(gap.max()),
    }
    delta = {}
    lost = {}
    for name in ("slot", "noon"):
        both = matched & np.isfinite(rep[name])
        delta[name] = np.where(both, orientation * (rep[name] - rep["late"]), 0.0)
        lost[name] = matched & ~np.isfinite(rep[name])
    slim = population[[*FIT_FEATURES, "season", "margin_vs_open", "home_covered"]].copy()
    frames = {
        "A": slim,
        "B": with_move(slim, delta["slot"], lost["slot"]),
        "D": with_move(slim, delta["noon"], lost["noon"]),
    }
    every = tuple(int(value) for value in sorted(population["season"].unique()))
    served_folds = folds(slim, every)
    served_whole = fit_stage(slim)
    refit_folds = folds(frames["D"], every)
    refit_whole = fit_stage(frames["D"])
    for arm in ("A", "B", "D"):
        population[f"oos_{arm}"] = outside(served_folds, frames[arm])
        population[f"ins_{arm}"] = inside(served_whole, frames[arm])
    population["oos_E"] = outside(refit_folds, frames["D"])
    population["ins_E"] = inside(refit_whole, frames["D"])
    latest = sorted((ART / "pick_probability").glob("*/per_game.parquet"))[-1]
    artifact = pd.read_parquet(latest)[["game_id", "out_of_season_home_probability"]]
    check = population[["game_id", "oos_A"]].merge(artifact, on="game_id")
    stage_gap = float(
        (base.sigmoid(check["oos_A"].to_numpy()) - check["out_of_season_home_probability"])
        .abs()
        .max()
    )
    tables = {
        int(held): push_mass_table(population.loc[population["season"] != held]) for held in SEASONS
    }
    tables[-1] = push_mass_table(population)
    quotes = base.tuesday_quotes(noon)
    spacing = float(np.median(np.diff(np.sort(quotes["lag"].round(2).unique()))))
    chosen = base.selections(quotes, fit_lag, live_lag)
    population["pid"] = np.arange(len(population))
    games = population.loc[scored].copy()
    arm_summary = {}
    for arm in base.ARMS:
        grouped = (
            chosen[arm]
            .groupby("nflverse_game_id")
            .agg(
                anchor=("home_spread_line", "median"),
                books=("bookmaker_key", "nunique"),
                lag=("lag", "median"),
            )
            .rename(columns=lambda name, arm=arm: f"{arm}_{name}")
            .rename_axis("game_id")
            .reset_index()
        )
        games = games.merge(grouped, on="game_id", how="left")
    source = {
        "after": chosen["after"].loc[
            chosen["after"]["bookmaker_key"].eq(base.SOURCE_BOOK)
            & chosen["after"]["lag"].le(spacing)
        ],
        "before": chosen["before"].loc[
            chosen["before"]["bookmaker_key"].eq(base.SOURCE_BOOK)
            & chosen["before"]["lag"].ge(-spacing)
        ],
    }
    for pool in base.POOLS:
        line = source[pool].set_index("nflverse_game_id")["home_spread_line"]
        games[f"pool_{pool}"] = games["game_id"].map(line)
    lacking = {
        arm: int(games[f"{arm}_books"].fillna(0).lt(base.MINIMUM_BOOKS).sum()) for arm in base.ARMS
    }
    enough = np.logical_and.reduce(
        [games[f"{arm}_books"].ge(base.MINIMUM_BOOKS) for arm in base.ARMS]
    )
    scored_games = len(games)
    games = games.loc[enough].reset_index(drop=True)
    for arm in base.ARMS:
        arm_summary[arm] = {
            "median_hours_after_noon": float(games[f"{arm}_lag"].median()),
            "median_books": float(games[f"{arm}_books"].median()),
        }
    for name in BASELINES:
        games[f"base_{name}"] = games["game_id"].map(leaders[f"base_{name}"])
        games[f"leader_lag_{name}"] = games["game_id"].map(leaders[f"lag_{name}"])
    anchor_moved = games["nearest_anchor"].ne(games["late_anchor"])
    leader_moved = (
        games["base_noon"].notna()
        & games["base_late"].notna()
        & games["base_noon"].ne(games["base_late"])
    )
    games["moved"] = anchor_moved | leader_moved
    pid = games["pid"].to_numpy()
    counts = {
        "scored_population_games": scored_games,
        "games_with_every_arm": len(games),
        "games_short_of_five_books_by_anchor_arm": lacking,
        "served_move_available": int(have[pid].sum()),
        "served_move_not_replicated": int((have & ~matched)[pid].sum()),
        "move_lost_at_live_slot": int(lost["slot"][pid].sum()),
        "move_lost_at_noon": int(lost["noon"][pid].sum()),
        "move_differs_live_slot": int((delta["slot"][pid] != 0).sum()),
        "move_differs_noon": int((delta["noon"][pid] != 0).sum()),
        "blind_spot_games": int(games["moved"].sum()),
        "blind_spot_by_all_book_anchor": int(anchor_moved.sum()),
        "blind_spot_by_leader_baseline": int(leader_moved.sum()),
        "leader_baseline_median_hours_after_noon": {
            name: float(games[f"leader_lag_{name}"].median()) for name in BASELINES
        },
    }
    stage_coefficients = {
        "served_model": {
            MOVE_COLUMN: float(model.move_toward_home),
            MOVE_AVAILABLE_COLUMN: float(model.move_available),
            "model_logit": float(model.model_logit),
            "pool_hook": served,
        },
        "fit_definitions_all_seasons": natural(served_whole),
        "fit_definitions_folds": {str(k): natural(served_folds[k]) for k in SEASONS},
        "noon_inputs_all_seasons": natural(refit_whole),
        "noon_inputs_folds": {str(k): natural(refit_folds[k]) for k in SEASONS},
    }
    rng = np.random.default_rng(base.SEED)
    null_rng = np.random.default_rng([base.SEED, 1])
    looks = len(CONTRASTS) * len(SUBSETS) * len(base.POOLS) * len(UNITS)
    result: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seasons": list(SEASONS),
        "citations": CITATIONS,
        "served_coefficient": served,
        "fit_lag_bound_hours": fit_lag,
        "live_lag_bound_hours": live_lag,
        "snapshot_spacing_hours": spacing,
        "stage_one_gap_to_artifact": stage_gap,
        "stage_one_artifact": str(latest),
        "move_replication": fidelity,
        "counts": counts,
        "anchor_arm_summary": arm_summary,
        "stage_one_natural_coefficients": stage_coefficients,
        "looks": looks,
        "null_looks": 2 * len(NULL_REPS),
        "pools": {},
    }
    emit(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "stage_one_gap_to_artifact",
                    "move_replication",
                    "counts",
                    "anchor_arm_summary",
                    "looks",
                )
            }
        )
    )
    for label, block in stage_coefficients.items():
        if label.endswith("folds"):
            text = " ".join(
                f"{k}: move {v[MOVE_COLUMN]:+.4f} avail {v[MOVE_AVAILABLE_COLUMN]:+.4f}"
                for k, v in block.items()
            )
        else:
            text = f"move {block[MOVE_COLUMN]:+.4f} avail {block[MOVE_AVAILABLE_COLUMN]:+.4f}"
        emit(f"STAGE1 {label}: {text}")
    cells = []
    for pool in base.POOLS:
        frame = games.loc[games[f"pool_{pool}"].notna()].reset_index(drop=True)
        line = frame[f"pool_{pool}"].to_numpy(dtype=float)
        whole = whole_mask(frame[f"pool_{pool}"])
        share = 1.0 / len(base.DIRECTIONS)
        up = frame.assign(
            pool=np.where(whole, line + base.DIRECTIONS[0] * POOL_OFFSET, line),
            w=np.where(whole, share, 1.0),
        )
        down = frame.loc[whole].assign(pool=line[whole] + base.DIRECTIONS[1] * POOL_OFFSET, w=share)
        rows = pd.concat([up, down], ignore_index=True)
        rows = rows.loc[rows["margin"] != rows["pool"]].reset_index(drop=True)
        rows["y"] = (rows["margin"] > rows["pool"]).astype(float)
        outcome = rows["y"].to_numpy()
        weight = rows["w"].to_numpy()
        season = rows["season"].to_numpy()
        target = rows["pool"].to_numpy(dtype=float)
        row_pid = rows["pid"].to_numpy()
        sets = hook_sets(rows, target, tables)
        hook_folds = {}
        hook_whole = {}
        for arm in ("A_fold", "E"):
            full = sets[ANCHOR[arm]]
            stage = STAGE[arm]
            hook_folds[arm] = fold_coefficients(
                full, rows[f"oos_{stage}"].to_numpy(), outcome, weight, season
            )
            hook_whole[arm] = base.fit_coefficient(
                full[-1], rows[f"ins_{stage}"].to_numpy(), outcome, weight
            )
        probabilities = {}
        for arm in ARMS:
            full = sets[ANCHOR[arm]]
            stage = STAGE[arm]
            if arm in hook_folds:
                outer = np.array([hook_folds[arm][int(value)] for value in season])
                whole_coefficient = hook_whole[arm]
            else:
                outer = served
                whole_coefficient = served
            probabilities[arm] = {
                "served": base.sigmoid(rows[f"oos_{stage}"].to_numpy() + outer * full["outer"]),
                "inside": base.sigmoid(
                    rows[f"ins_{stage}"].to_numpy() + whole_coefficient * full[-1]
                ),
            }
        masks = {
            "all": np.ones(len(rows), dtype=bool),
            "moved": rows["moved"].to_numpy(dtype=bool),
            "rest": ~rows["moved"].to_numpy(dtype=bool),
        }
        levels = {
            subset: {
                arm: {
                    mode: level(probabilities[arm][mode][mask], outcome[mask], weight[mask])
                    for mode in ("served", "inside")
                }
                for arm in ARMS
            }
            for subset, mask in masks.items()
        }
        comparisons: dict[str, dict[str, object]] = {}
        for first, second in CONTRASTS:
            name = f"{first}_minus_{second}"
            comparisons[name] = {}
            for subset, mask in masks.items():
                entry = {
                    mode: compare(
                        probabilities[first][mode], probabilities[second][mode], rows, mask, rng
                    )
                    for mode in ("served", "inside")
                }
                entry["accuracy_gap_inside_minus_outside"] = (
                    entry["inside"]["accuracy_points"] - entry["served"]["accuracy_points"]
                )
                comparisons[name][subset] = entry
                if pool != PRIMARY_POOL:
                    continue
                measured = entry["served"]
                for unit in UNITS:
                    interval = measured[f"{unit}_interval"]
                    cells.append(
                        {
                            "name": f"eng62_sysarm_{name}_{subset}_2023_25_{unit}",
                            "description": (
                                "Tuesday reference system arms, 2023-25, pushes kept, stage 1 "
                                "and push table out of season, pool line from DraftKings' first "
                                f"quote after Tuesday noon: arm {first} minus arm {second}, "
                                f"population {subset} (moved = the number changed between "
                                "Tuesday noon and Wednesday 00:00 ET); week-block bootstrap; "
                                "arms defined in docs/lanes/tuesday-reference-bundle-2026-11-14.md"
                            ),
                            "plain_summary": (
                                "Two of the card's adjustments read where the sportsbooks' "
                                "number stood on Tuesday. This compares reading them "
                                f"{PLAIN[first]} against reading them {PLAIN[second]}"
                                f"{PLAIN_SUBSET[subset]}, on past seasons. The two are too "
                                "close to separate."
                            ),
                            "effect": measured[unit],
                            "effect_units": unit,
                            "interval_low": interval[0],
                            "interval_high": interval[1],
                            "probability_positive": measured[f"{unit}_probability_positive"],
                            "season_start": SEASONS[0],
                            "season_end": SEASONS[-1],
                            "sample_games": round(measured["games"]),
                            "sample_blocks": measured["blocks"],
                        }
                    )
        block: dict[str, object] = {
            "games": float(weight.sum()),
            "offset_coefficient_folds": {
                arm: {str(k): v for k, v in hook_folds[arm].items()} for arm in hook_folds
            },
            "offset_coefficient_all_seasons": hook_whole,
            "stage_one": level(base.sigmoid(rows["oos_A"].to_numpy()), outcome, weight),
            "levels": levels,
            "comparisons": comparisons,
        }
        if pool == PRIMARY_POOL:
            eligible = matched & np.isfinite(rep["noon"])
            reference = probabilities["C"]["served"]
            nulls = {"D": [], "E": []}
            for _ in range(NULL_REPS["D"]):
                trial = with_move(
                    slim, shuffled(delta["noon"], eligible, year, null_rng), lost["noon"]
                )
                offset = outside(served_folds, trial)[row_pid]
                guess = base.sigmoid(offset + served * sets["nearest"]["outer"])
                nulls["D"].append(point(guess, reference, outcome, weight))
            for _ in range(NULL_REPS["E"]):
                trial = with_move(
                    slim, shuffled(delta["noon"], eligible, year, null_rng), lost["noon"]
                )
                offset = outside(folds(trial, SEASONS), trial)[row_pid]
                fitted = fold_coefficients(sets["nearest"], offset, outcome, weight, season)
                outer = np.array([fitted[int(value)] for value in season])
                guess = base.sigmoid(offset + outer * sets["nearest"]["outer"])
                nulls["E"].append(point(guess, reference, outcome, weight))
            summary = {}
            for arm, draws in nulls.items():
                table = np.array(draws)
                actual = point(probabilities[arm]["served"], reference, outcome, weight)
                summary[f"{arm}_minus_C"] = {
                    "permutations": len(table),
                    "actual_accuracy_points": actual[0],
                    "null_accuracy_points_mean": float(table[:, 0].mean()),
                    "null_accuracy_points_5_95": [
                        float(value) for value in np.percentile(table[:, 0], [5, 95])
                    ],
                    "accuracy_percentile_in_null": percentile(table[:, 0], actual[0]),
                    "actual_log_loss_improvement": actual[1],
                    "null_log_loss_improvement_mean": float(table[:, 1].mean()),
                    "null_log_loss_improvement_5_95": [
                        float(value) for value in np.percentile(table[:, 1], [5, 95])
                    ],
                    "log_loss_percentile_in_null": percentile(table[:, 1], actual[1]),
                }
            block["matched_null_noon_to_late_change_shuffled_within_season"] = summary
        result["pools"][pool] = block
        emit(
            f"POOL {pool} games {block['games']:g} offset folds "
            + json.dumps(block["offset_coefficient_folds"])
            + " all "
            + json.dumps(hook_whole)
        )
        for subset in SUBSETS:
            if pool != PRIMARY_POOL and subset != "all":
                continue
            text = " ".join(
                f"{arm} {item['served']['accuracy'] * 100:.2f}/{item['served']['log_loss']:.5f}"
                f"/{item['served']['brier']:.5f}"
                for arm, item in levels[subset].items()
            )
            emit(f" SUBSET {subset} games {levels[subset]['A']['served']['games']:g} | {text}")
            for name, by_subset in comparisons.items():
                entry = by_subset[subset]
                m = entry["served"]
                seasons = " ".join(
                    f"{k}:{v['accuracy_points']:+.2f}/{v['log_loss_improvement']:+.5f}"
                    f"({v['decisive_first_right']:g}-{v['decisive_second_right']:g})"
                    for k, v in m["by_season"].items()
                )
                emit(
                    f"  {name:18s} dec {m['decisive_first_right']:g}-"
                    f"{m['decisive_second_right']:g}"
                    f" acc {m['accuracy_points']:+.2f} [{m['accuracy_points_interval'][0]:+.2f},"
                    f"{m['accuracy_points_interval'][1]:+.2f}]"
                    f" P+ {m['accuracy_points_probability_positive']:.2f}"
                    f" ll {m['log_loss_improvement']:+.5f}"
                    f" [{m['log_loss_improvement_interval'][0]:+.5f},"
                    f"{m['log_loss_improvement_interval'][1]:+.5f}]"
                    f" P+ {m['log_loss_improvement_probability_positive']:.2f}"
                    f" br {m['brier_improvement']:+.5f}"
                    f" [{m['brier_improvement_interval'][0]:+.5f},"
                    f"{m['brier_improvement_interval'][1]:+.5f}]"
                    f" P+ {m['brier_improvement_probability_positive']:.2f}"
                    f" gap {entry['accuracy_gap_inside_minus_outside']:+.2f} | {seasons}"
                )
        if pool == PRIMARY_POOL:
            emit(
                " NULL "
                + json.dumps(block["matched_null_noon_to_late_change_shuffled_within_season"])
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference_system_arms.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": FAMILY,
        "source": "artifacts/eng62/reference_system_arms.json",
        "category": "modeling",
        "notes": (
            "Which Tuesday quotes the pool-line offset term and the line-move term read, as one "
            f"system: {len(CONTRASTS)} declared arm contrasts, {len(SUBSETS)} populations, "
            f"{len(base.POOLS)} pool-line builds (the first recorded, the second in the source), "
            f"{len(UNITS)} units; {looks} looks plus {2 * len(NULL_REPS)} matched-null looks."
        ),
        "cells": cells,
    }
    (OUT / "reference_system_arms_batch.json").write_text(
        json.dumps(batch, indent=1), encoding="utf-8"
    )
    (OUT / "reference_system_arms.out").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

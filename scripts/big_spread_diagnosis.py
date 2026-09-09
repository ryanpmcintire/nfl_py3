"""MOD-18 lane L (2026-09-08): WHY the served model is weak on 10.5+ spreads.

Descriptive diagnosis only -- no candidate, no pick flip, no fitted parameter
(AGENTS.md, "No unexplained threshold flips on the played card": a dip located
at a spread threshold is a DIAGNOSIS to publish, never a flip to bolt on).

Inputs, all read-only: the active model's matching opener evaluation
(``nfl_ats.public_board.find_matching_opener_evaluation``; its ``per_game``
table carries the SERVED read after the S3 home-side offset and the raw
twins), the weak-stack feature table (pregame columns only) and the newest
nflverse schedules snapshot (rest days, kickoff, roof, division flag).

Every cell reports its n, the served accuracy and the point error after S3
(actual home margin minus the served point) with a week-blocked bootstrap
interval (20,000 draws, seed 20260817, whole weeks resampled, within-week
correlation zero -- never estimated or padded). Tables land under
``artifacts/research/laneL/`` through the provenance helpers; the write-up is
``docs/big_spread_diagnosis.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.active_model import load_active_ats_model  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.home_side_location import HOME_SIDE_OFFSET_BUCKETS  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact  # noqa: E402
from nfl_ats.public_board import find_matching_opener_evaluation  # noqa: E402
from nfl_ats.spread_regime import BUCKETS, spread_bucket  # noqa: E402

OUT = REPO / "artifacts/research/laneL"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
SEED = 20260817
DRAWS = 20_000
BIG = "10.5+"
MID = "7.5-10"
DIAGNOSED = tuple(HOME_SIDE_OFFSET_BUCKETS)

FEATURE_COLUMNS = (
    "game_id",
    "gameday",
    "weekday",
    "gametime",
    "location",
    "div_game",
    "neutral_site",
    "elo_diff",
    "home_qb_start_probability",
    "away_qb_start_probability",
    "rest_diff",
    "diff_point_diff",
    "diff_ats_residual",
    "diff_off_epa_per_play",
    "diff_def_epa_per_play",
    "diff_qb_expected_epa_per_dropback",
    "diff_injury_skill_epa_value_lost",
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
)
LEAN_FEATURES = (
    "elo_diff",
    "rest_diff",
    "diff_point_diff",
    "diff_ats_residual",
    "diff_off_epa_per_play",
    "diff_def_epa_per_play",
    "diff_qb_expected_epa_per_dropback",
    "diff_injury_skill_epa_value_lost",
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
)
CROSSES: tuple[tuple[str, str], ...] = (
    ("line_band", "pick_location"),
    ("move_band", "pick_location"),
    ("home_side", "pick_location"),
    ("week_band", "pick_location"),
    ("residual_lean", "pick_location"),
)
SCHEDULE_COLUMNS = (
    "game_id",
    "home_rest",
    "away_rest",
    "roof",
    "surface",
    "home_team",
    "away_team",
)

TEAM_ZONE_OFFSET: dict[str, int] = {
    "ARI": -2,
    "ATL": 0,
    "BAL": 0,
    "BUF": 0,
    "CAR": 0,
    "CHI": -1,
    "CIN": 0,
    "CLE": 0,
    "DAL": -1,
    "DEN": -2,
    "DET": 0,
    "GB": -1,
    "HOU": -1,
    "IND": 0,
    "JAX": 0,
    "KC": -1,
    "LA": -3,
    "LAR": -3,
    "LAC": -3,
    "LV": -3,
    "MIA": 0,
    "MIN": -1,
    "NE": 0,
    "NO": -1,
    "NYG": 0,
    "NYJ": 0,
    "OAK": -3,
    "PHI": 0,
    "PIT": 0,
    "SD": -3,
    "SEA": -3,
    "SF": -3,
    "STL": -1,
    "TB": 0,
    "TEN": -1,
    "WAS": 0,
}

CUTS: tuple[tuple[str, str], ...] = (
    ("home_side", "home favourite / home underdog"),
    ("pick_side", "model picked the favourite / the underdog"),
    ("side_by_pick", "home side x pick side"),
    ("pick_location", "model picked the home team / the road team"),
    ("favourite_qb_band", "favourite's projected starting QB expected to start"),
    ("underdog_qb_band", "underdog's projected starting QB expected to start"),
    ("favourite_rest_band", "favourite's rest days"),
    ("favourite_rest_edge", "favourite's rest minus the underdog's"),
    ("favourite_travel", "time zones the favourite crossed (0 when at home)"),
    ("week_band", "week of season"),
    ("late_season", "weeks 15-18 (playoff-race proxy; no race column exists)"),
    ("season", "season"),
    ("move_band", "open-to-close move relative to the model's side"),
    ("move_size", "size of the open-to-close move"),
    ("line_band", "opening line size"),
    ("key_band", "nearest key number to the opening line (10, 14, 17)"),
    ("residual_band", "size of the model's served residual against the opener"),
    ("residual_lean", "the residual leans to the favourite / the underdog"),
    ("confidence_band", "stated confidence of the served pick"),
    ("div_game", "division game"),
    ("primetime", "kickoff at 19:00 ET or later"),
    ("roof_band", "roof"),
)


def latest_schedules() -> Path:
    candidates = sorted((REPO / "data/raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def load_inputs(artifacts_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    active = load_active_ats_model(artifacts_root)
    if not active:
        raise RuntimeError("no synchronized active model")
    matched = find_matching_opener_evaluation(artifacts_root, active)
    if matched is None:
        raise RuntimeError("no opener evaluation matches the active model")
    metadata, directory = matched
    if sha256_file(FEATURES) != active["feature_table_sha256"]:
        raise RuntimeError("feature table digest differs from the active model's")
    per_game = pd.read_parquet(directory / "per_game.parquet")
    if "residual_at_open_served" not in per_game.columns:
        raise RuntimeError("evaluation predates the served home-side offset")
    features = regular_season_rows(pd.read_parquet(FEATURES))[list(FEATURE_COLUMNS)]
    schedules = pd.read_parquet(latest_schedules())[list(SCHEDULE_COLUMNS)]
    provenance = {
        "active_model_id": active.get("model_id"),
        "evaluation": str(directory.relative_to(artifacts_root)).replace("\\", "/"),
        "evaluation_policy": (metadata.get("home_side_offset") or {}).get("policy"),
        "feature_table_sha256": active["feature_table_sha256"],
        "schedules": str(latest_schedules().relative_to(REPO)).replace("\\", "/"),
    }
    return per_game, features, schedules, provenance


def _band(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(values, edges, labels=labels, right=True, include_lowest=True).astype(object)


def build_frame(
    per_game: pd.DataFrame, features: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Join the evaluation to pregame context and derive every cut column.

    Sign conventions (nflverse, verified by ``model_weak_spots``): a POSITIVE
    home spread means the HOME team is favoured; ``margin_vs_open`` =
    ``result - tue_open_home_spread``; ``open_move`` = close minus open, so
    a positive move means the market moved TOWARD the home team.
    """

    frame = per_game.merge(features, on="game_id", how="left", validate="one_to_one")
    frame = frame.merge(schedules, on="game_id", how="left", validate="one_to_one")
    spread = pd.to_numeric(frame["tue_open_home_spread"], errors="coerce")
    close = pd.to_numeric(frame["close_home_spread"], errors="coerce")
    result = pd.to_numeric(frame["result"], errors="coerce")
    margin = pd.to_numeric(frame["margin_vs_open"], errors="coerce")
    move = pd.to_numeric(frame["open_move"], errors="coerce")
    both = spread.notna() & close.notna() & move.notna()
    if both.any() and not np.allclose((close - spread)[both], move[both]):
        raise ValueError("open_move is not close minus open")

    frame["spread_size"] = spread.abs()
    frame["bucket"] = spread_bucket(spread.fillna(0.0)).where(spread.notna())
    frame["home_side"] = np.select(
        [spread.gt(0), spread.lt(0)], ["home favourite", "home underdog"], default="pick'em"
    )
    frame["push"] = margin.eq(0)
    pick_home = frame["pick_home_at_open_probability_rule"].astype(bool)
    frame["pick_home"] = pick_home
    frame["correct"] = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    frame["correct_raw"] = pd.to_numeric(
        frame["correct_at_open_probability_rule_raw"], errors="coerce"
    )
    pick_favourite = pick_home.eq(spread.gt(0)) & spread.ne(0)
    frame["pick_side"] = np.where(pick_favourite, "picked favourite", "picked underdog")
    frame.loc[spread.eq(0), "pick_side"] = "pick'em"
    frame["side_by_pick"] = frame["home_side"] + ", " + frame["pick_side"]
    home_prob = pd.to_numeric(frame["home_cover_probability_at_open"], errors="coerce")
    frame["home_prob"] = home_prob
    frame["confidence"] = home_prob.where(pick_home, 1 - home_prob)
    frame["home_cover"] = margin.gt(0).astype(float).where(margin.notna() & margin.ne(0))
    frame["favourite_cover"] = (
        margin.gt(0).eq(spread.gt(0)).astype(float).where(frame["home_cover"].notna())
    )
    frame["pick_favourite"] = pick_favourite.astype(float).where(spread.ne(0))
    frame["pick_location"] = np.where(pick_home, "picked the home team", "picked the road team")

    served_residual = pd.to_numeric(frame["residual_at_open_served"], errors="coerce")
    raw_residual = pd.to_numeric(frame["residual_at_open"], errors="coerce")
    frame["point_served"] = spread + served_residual
    frame["point_raw"] = spread + raw_residual
    frame["error_served"] = result - frame["point_served"]
    frame["error_raw"] = result - frame["point_raw"]
    frame["error_served_favourite"] = frame["error_served"] * np.sign(spread)
    frame["error_served_pick"] = frame["error_served"] * np.where(pick_home, 1.0, -1.0)
    frame["offset"] = pd.to_numeric(frame["home_side_offset_at_open"], errors="coerce")
    frame["residual_abs"] = served_residual.abs()
    frame["residual_band"] = _band(
        frame["residual_abs"], [0, 1, 2, 3, 99], ["under 1", "1-2", "2-3", "3 or more"]
    )
    lean = np.sign(served_residual) * np.sign(spread)
    frame["residual_lean"] = np.select(
        [lean.gt(0), lean.lt(0)], ["leans favourite", "leans underdog"], default="flat"
    )
    frame["confidence_band"] = _band(
        frame["confidence"],
        [0.0, 0.52, 0.55, 0.60, 1.0],
        ["50-52%", "52-55%", "55-60%", "60% or more"],
    )

    home_rest = pd.to_numeric(frame["home_rest"], errors="coerce")
    away_rest = pd.to_numeric(frame["away_rest"], errors="coerce")
    home_is_favourite = spread.gt(0)
    fav_rest = home_rest.where(home_is_favourite, away_rest)
    dog_rest = away_rest.where(home_is_favourite, home_rest)
    frame["favourite_rest"] = fav_rest
    frame["favourite_rest_band"] = _band(
        fav_rest, [0, 6, 7, 99], ["short (6 or fewer)", "normal (7)", "long (8 or more)"]
    )
    edge = fav_rest - dog_rest
    frame["favourite_rest_edge"] = np.select(
        [edge.lt(0), edge.gt(0)], ["favourite rested less", "favourite rested more"], "equal"
    )
    frame.loc[edge.isna(), "favourite_rest_edge"] = "unknown"
    home_qb = pd.to_numeric(frame["home_qb_start_probability"], errors="coerce")
    away_qb = pd.to_numeric(frame["away_qb_start_probability"], errors="coerce")
    fav_qb = home_qb.where(home_is_favourite, away_qb)
    dog_qb = away_qb.where(home_is_favourite, home_qb)
    for name, series in (("favourite_qb_band", fav_qb), ("underdog_qb_band", dog_qb)):
        frame[name] = np.select(
            [series.ge(0.95), series.notna()],
            ["starter expected (95%+)", "starter in doubt (under 95%)"],
            default="unknown",
        )
    home_zone = frame["home_team"].map(TEAM_ZONE_OFFSET)
    away_zone = frame["away_team"].map(TEAM_ZONE_OFFSET)
    crossed = (home_zone - away_zone).abs()
    neutral_flag = (
        pd.to_numeric(frame["neutral_site"], errors="coerce")
        if "neutral_site" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    neutral = frame["location"].astype(str).str.lower().eq("neutral") | neutral_flag.fillna(
        0
    ).astype(bool)
    fav_travel = pd.Series(0.0, index=frame.index).where(home_is_favourite, crossed)
    frame["favourite_travel"] = np.select(
        [neutral, fav_travel.eq(0), fav_travel.eq(1), fav_travel.ge(2)],
        ["neutral site", "none", "one zone", "two or more zones"],
        default="unknown",
    )

    week = pd.to_numeric(frame["week"], errors="coerce")
    frame["week_band"] = _band(week, [0, 6, 12, 22], ["weeks 1-6", "weeks 7-12", "weeks 13-18"])
    frame["late_season"] = np.where(week.ge(15), "weeks 15-18", "weeks 1-14")

    toward_model = move * np.where(pick_home, 1.0, -1.0)
    frame["move_toward_model"] = toward_model
    frame["move_band"] = np.select(
        [toward_model.lt(0), toward_model.gt(0)],
        ["market moved against the model", "market moved with the model"],
        default="no move",
    )
    frame.loc[move.isna(), "move_band"] = "unknown"
    frame["move_size"] = _band(
        move.abs(), [-1, 0.25, 1, 2, 99], ["no move", "half or one point", "1.5-2", "2.5 or more"]
    )
    frame.loc[move.isna(), "move_size"] = "unknown"
    size = frame["spread_size"]
    frame["line_band"] = np.select(
        [
            size.le(3),
            size.le(6.5),
            size.lt(7.5),
            size.le(9),
            size.le(10),
            size.le(12.5),
            size.le(14),
        ],
        ["0-3", "3.5-6.5", "7", "7.5-9", "9.5-10", "10.5-12.5", "13-14"],
        default="14.5 or more",
    )
    keys = np.array([10.0, 14.0, 17.0])
    distance = np.abs(size.to_numpy()[:, None] - keys[None, :])
    nearest = keys[np.argmin(distance, axis=1)]
    frame["key_distance"] = distance.min(axis=1)
    frame["key_band"] = np.where(
        frame["key_distance"].le(0.5),
        [f"on or beside {int(k)}" for k in nearest],
        [f"between (nearest {int(k)})" for k in nearest],
    )
    frame["div_game"] = np.where(
        pd.to_numeric(frame["div_game"], errors="coerce").fillna(0).gt(0),
        "division game",
        "non-division",
    )
    hour = pd.to_numeric(frame["gametime"].astype(str).str.slice(0, 2), errors="coerce")
    frame["primetime"] = np.where(hour.ge(19), "primetime", "day game")
    roof = frame["roof"].astype(str).str.lower()
    frame["roof_band"] = np.where(roof.isin(["dome", "closed"]), "indoors", "outdoors or open")
    return frame


def week_blocked_mean(
    values: pd.Series, blocks: pd.Series, *, draws: int = DRAWS, seed: int = SEED
) -> dict[str, float]:
    """Mean with a 95% interval from resampling whole (season, week) blocks.

    Returns ``estimate``, ``lower``, ``upper`` and ``probability_positive``
    (the share of draws whose mean exceeds zero). Missing values are dropped
    first; a cell with no rows returns NaNs.
    """

    keep = values.notna()
    v = values[keep].to_numpy(dtype=float)
    if v.size == 0:
        return {
            "estimate": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "probability_positive": np.nan,
        }
    codes = pd.factorize(blocks[keep].astype(str))[0]
    sums = np.bincount(codes, weights=v)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, sums.size, size=(draws, sums.size))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "estimate": float(v.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "probability_positive": float(probability_positive_from_draws(means)),
    }


def week_blocked_slope(
    x: pd.Series, y: pd.Series, blocks: pd.Series, *, draws: int = DRAWS, seed: int = SEED
) -> dict[str, float]:
    """OLS slope of ``y`` on ``x`` with a whole-week block bootstrap interval.

    Slope 1 means the residual is calibrated in magnitude, 0 that it carries
    no information about the outcome, negative that it points the wrong way.
    """

    keep = x.notna() & y.notna()
    xv = x[keep].to_numpy(dtype=float)
    yv = y[keep].to_numpy(dtype=float)
    if xv.size < 3 or xv.std() == 0:
        return {
            "estimate": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "probability_positive": np.nan,
        }
    codes = pd.factorize(blocks[keep].astype(str))[0]
    n = np.bincount(codes).astype(float)
    sx = np.bincount(codes, weights=xv)
    sy = np.bincount(codes, weights=yv)
    sxy = np.bincount(codes, weights=xv * yv)
    sxx = np.bincount(codes, weights=xv * xv)

    def slope(idx: np.ndarray) -> np.ndarray:
        tn, tx, ty = n[idx].sum(axis=-1), sx[idx].sum(axis=-1), sy[idx].sum(axis=-1)
        txy, txx = sxy[idx].sum(axis=-1), sxx[idx].sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (txy - tx * ty / tn) / (txx - tx * tx / tn)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n.size, size=(draws, n.size))
    sampled = slope(picks)
    finite = sampled[np.isfinite(sampled)]
    return {
        "estimate": float(slope(np.arange(n.size))),
        "lower": float(np.quantile(finite, 0.025)) if finite.size else np.nan,
        "upper": float(np.quantile(finite, 0.975)) if finite.size else np.nan,
        "probability_positive": float(probability_positive_from_draws(finite))
        if finite.size
        else np.nan,
    }


def residual_information_table(
    frame: pd.DataFrame, buckets: tuple[str, ...], *, draws: int = DRAWS, seed: int = SEED
) -> pd.DataFrame:
    """Does the model's residual say anything inside each bucket? (diagnosis only)

    The always-home and always-favourite columns are BASE RATES for reading
    the model's discrimination against, never candidate rules (AGENTS.md,
    "No unexplained threshold flips").
    """

    rows = []
    for bucket in buckets:
        inside = frame.loc[frame["bucket"].eq(bucket)]
        decided = inside.loc[~inside["push"] & inside["correct"].isin([0, 1])]
        margin = pd.to_numeric(decided["margin_vs_open"], errors="coerce")
        served = week_blocked_slope(
            pd.to_numeric(decided["residual_at_open_served"], errors="coerce"),
            margin,
            _blocks(decided),
            draws=draws,
            seed=seed,
        )
        raw = week_blocked_slope(
            pd.to_numeric(decided["residual_at_open"], errors="coerce"),
            margin,
            _blocks(decided),
            draws=draws,
            seed=seed,
        )
        versus_home = week_blocked_mean(
            decided["correct"] - decided["home_cover"], _blocks(decided), draws=draws, seed=seed
        )
        rows.append(
            {
                "bucket": bucket,
                "decided": len(decided),
                "model_minus_always_home": versus_home["estimate"],
                "model_minus_always_home_lower": versus_home["lower"],
                "model_minus_always_home_upper": versus_home["upper"],
                "model_minus_always_home_probability_positive": versus_home["probability_positive"],
                "model_accuracy": float(decided["correct"].mean()) if len(decided) else np.nan,
                "model_accuracy_raw": float(decided["correct_raw"].mean())
                if len(decided)
                else np.nan,
                "always_home_cover_rate": float(decided["home_cover"].mean())
                if len(decided)
                else np.nan,
                "always_favourite_cover_rate": float(decided["favourite_cover"].mean())
                if decided["favourite_cover"].notna().any()
                else np.nan,
                "slope_served": served["estimate"],
                "slope_served_lower": served["lower"],
                "slope_served_upper": served["upper"],
                "slope_served_probability_positive": served["probability_positive"],
                "slope_raw": raw["estimate"],
                "slope_raw_lower": raw["lower"],
                "slope_raw_upper": raw["upper"],
                "slope_raw_probability_positive": raw["probability_positive"],
            }
        )
    return pd.DataFrame(rows)


def _blocks(frame: pd.DataFrame) -> pd.Series:
    return frame["season"].astype(str) + "-" + frame["week"].astype(str)


def cell_row(frame: pd.DataFrame, *, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """One cell: served accuracy, stated confidence, cover rates, point error."""

    decided = frame.loc[~frame["push"] & frame["correct"].isin([0, 1])]
    accuracy = week_blocked_mean(decided["correct"], _blocks(decided), draws=draws, seed=seed)
    served_error = week_blocked_mean(frame["error_served"], _blocks(frame), draws=draws, seed=seed)
    raw_error = week_blocked_mean(frame["error_raw"], _blocks(frame), draws=draws, seed=seed)
    sided = decided.loc[decided["pick_favourite"].notna()]

    def mean(series: pd.Series) -> float:
        return float(series.mean()) if series.notna().any() else np.nan

    return {
        "games": len(frame),
        "decided": len(decided),
        "accuracy": accuracy["estimate"],
        "accuracy_lower": accuracy["lower"],
        "accuracy_upper": accuracy["upper"],
        "probability_above_coin_flip": float(np.nan)
        if np.isnan(accuracy["estimate"])
        else _probability_above_half(decided, draws=draws, seed=seed),
        "accuracy_raw": mean(decided["correct_raw"]),
        "stated_confidence": mean(decided["confidence"]),
        "favourite_pick_rate": mean(sided["pick_favourite"]),
        "favourite_cover_rate": mean(sided["favourite_cover"]),
        "home_cover_rate": mean(decided["home_cover"]),
        "home_probability": mean(decided["home_prob"]),
        "error_served": served_error["estimate"],
        "error_served_lower": served_error["lower"],
        "error_served_upper": served_error["upper"],
        "error_served_probability_positive": served_error["probability_positive"],
        "error_raw": raw_error["estimate"],
        "error_raw_lower": raw_error["lower"],
        "error_raw_upper": raw_error["upper"],
        "mean_offset": mean(frame["offset"]),
        "mean_abs_error_served": mean(frame["error_served"].abs()),
        "error_served_favourite": mean(frame["error_served_favourite"]),
        "error_served_pick": mean(decided["error_served_pick"]),
    }


def _probability_above_half(decided: pd.DataFrame, *, draws: int, seed: int) -> float:
    centred = decided["correct"] - 0.5
    return week_blocked_mean(centred, _blocks(decided), draws=draws, seed=seed)[
        "probability_positive"
    ]


def cut_table(
    frame: pd.DataFrame,
    buckets: tuple[str, ...],
    cuts: tuple[tuple[str, str], ...] = CUTS,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Every cut of question (2) inside each bucket, plus the bucket total."""

    rows = []
    for bucket in buckets:
        inside = frame.loc[frame["bucket"].eq(bucket)]
        rows.append(
            {
                "bucket": bucket,
                "cut": "all",
                "level": "all",
                **cell_row(inside, draws=draws, seed=seed),
            }
        )
        for column, description in cuts:
            if progress:
                progress(f"{bucket} / {column}")
            for level, cell in inside.groupby(column, sort=True, dropna=False):
                rows.append(
                    {
                        "bucket": bucket,
                        "cut": column,
                        "cut_description": description,
                        "level": str(level),
                        **cell_row(cell, draws=draws, seed=seed),
                    }
                )
    return pd.DataFrame(rows)


def ladder_table(frame: pd.DataFrame, *, draws: int = DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Question (4): the home point error and cover rate up the spread ladder, by home side."""

    order = ["0-3", "3.5-6.5", "7", "7.5-9", "9.5-10", "10.5-12.5", "13-14", "14.5 or more"]
    rows = []
    for band in order:
        for side in ("all", "home favourite", "home underdog"):
            cell = frame.loc[frame["line_band"].eq(band)]
            if side != "all":
                cell = cell.loc[cell["home_side"].eq(side)]
            if cell.empty:
                continue
            rows.append(
                {"line_band": band, "home_side": side, **cell_row(cell, draws=draws, seed=seed)}
            )
    return pd.DataFrame(rows)


def cross_table(
    frame: pd.DataFrame,
    bucket: str,
    crosses: tuple[tuple[str, str], ...] = CROSSES,
    *,
    draws: int = DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Two cuts crossed inside one bucket, to localise where the losses sit."""

    inside = frame.loc[frame["bucket"].eq(bucket)]
    rows = []
    for first, second in crosses:
        for (a, b), cell in inside.groupby([first, second], sort=True, dropna=False):
            rows.append(
                {
                    "bucket": bucket,
                    "first_cut": first,
                    "first_level": str(a),
                    "second_cut": second,
                    "second_level": str(b),
                    **cell_row(cell, draws=draws, seed=seed),
                }
            )
    return pd.DataFrame(rows)


def lean_table(frame: pd.DataFrame, buckets: tuple[str, ...]) -> pd.DataFrame:
    """Which pregame inputs move with the model's residual, per bucket.

    For each feature: its mean where the served residual leans to the
    favourite versus the underdog, and its Pearson correlation with the served
    residual (home frame). Descriptive; nothing is fitted.
    """

    rows = []
    for bucket in buckets:
        inside = frame.loc[frame["bucket"].eq(bucket)]
        residual = pd.to_numeric(inside["residual_at_open_served"], errors="coerce")
        for feature in LEAN_FEATURES:
            if feature not in inside.columns:
                continue
            values = pd.to_numeric(inside[feature], errors="coerce")
            usable = values.notna() & residual.notna()
            favourite = inside["residual_lean"].eq("leans favourite")
            underdog = inside["residual_lean"].eq("leans underdog")
            rows.append(
                {
                    "bucket": bucket,
                    "feature": feature,
                    "games": int(usable.sum()),
                    "mean_when_leans_favourite": float(values[usable & favourite].mean())
                    if (usable & favourite).any()
                    else np.nan,
                    "mean_when_leans_underdog": float(values[usable & underdog].mean())
                    if (usable & underdog).any()
                    else np.nan,
                    "correlation_with_served_residual": float(
                        np.corrcoef(values[usable], residual[usable])[0, 1]
                    )
                    if usable.sum() > 2 and values[usable].std() > 0
                    else np.nan,
                    "correlation_with_served_error": float(
                        np.corrcoef(values[usable], inside.loc[usable, "error_served"])[0, 1]
                    )
                    if usable.sum() > 2 and values[usable].std() > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame, path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    stamp_sidecar(path, {"rows": len(frame), **(extra or {})}, project_root=REPO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args(argv)

    per_game, features, schedules, provenance = load_inputs(args.artifacts)
    frame = build_frame(per_game, features, schedules)
    print(f"joined {len(frame)} games from {provenance['evaluation']}", flush=True)
    table(frame, args.out / "joined.parquet", provenance)

    cells = cut_table(
        frame, BUCKETS, draws=args.draws, progress=lambda label: print(label, flush=True)
    )
    table(cells, args.out / "cells.parquet", provenance)
    ladder = ladder_table(frame, draws=args.draws)
    table(ladder, args.out / "ladder.parquet", provenance)
    crosses = pd.concat(
        [cross_table(frame, bucket, draws=args.draws) for bucket in (BIG, MID)], ignore_index=True
    )
    table(crosses, args.out / "crosses.parquet", provenance)
    table(lean_table(frame, BUCKETS), args.out / "lean.parquet", provenance)
    information = residual_information_table(frame, BUCKETS, draws=args.draws)
    table(information, args.out / "residual_information.parquet", provenance)

    totals = cells.loc[cells["cut"].eq("all")].set_index("bucket")
    summary = {
        **provenance,
        "draws": args.draws,
        "seed": SEED,
        "within_week_correlation": 0.0,
        "games": len(frame),
        "buckets": {
            bucket: {
                key: (None if pd.isna(value) else float(value))
                for key, value in totals.loc[bucket].items()
                if key not in {"cut", "level", "cut_description"}
            }
            for bucket in DIAGNOSED
        },
        "command": ["python", "scripts/big_spread_diagnosis.py", *(argv or [])],
    }
    write_stamped_artifact(summary, args.out / "summary.json", project_root=REPO)
    print(json.dumps(summary["buckets"][BIG], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

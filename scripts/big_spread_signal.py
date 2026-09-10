"""MOD-18 lane AK: does any input carry out-of-sample signal at 10.5+?"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats import margin as margin_module  # noqa: E402
from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.constants import FEATURE_SETS  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.margin import (  # noqa: E402
    fit_margin_model,
    margin_feature_columns,
    margin_feature_groups,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402

PROFILE = "weak_stack"
BIG = 10.5
SAMPLES = 20_000
SEED = 20260821
HALF_SEED = 20260910
ERAS = (("2009_2013", 2009, 2013), ("2014_2019", 2014, 2019), ("2020_2025", 2020, 2025))
SPEC_FLOOR = 150
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
KICKOFF_MATCH_HOURS = 72.0


def register_family_profiles() -> dict[str, tuple[str, ...]]:
    """One runtime feature profile per declared family of the served profile."""

    columns = margin_feature_columns("market_residual", PROFILE)
    groups = margin_feature_groups("market_residual", PROFILE)
    families: dict[str, list[str]] = {}
    for column, group in zip(columns, groups, strict=True):
        families.setdefault(group, []).append(column)
    registered: dict[str, tuple[str, ...]] = {}
    for family, family_columns in families.items():
        set_name = f"big_spread_signal_{family}"
        profile_name = f"weak_stack_only_{family}"
        FEATURE_SETS[set_name] = tuple(family_columns)
        margin_module._MARGIN_PROFILE_FEATURE_SETS[profile_name] = (set_name, set_name)
        margin_module.MARGIN_FEATURE_PROFILES = (
            *margin_module.MARGIN_FEATURE_PROFILES,
            profile_name,
        )
        registered[family] = tuple(family_columns)
    return registered


def opener_lines(features: pd.DataFrame, market_root: Path) -> pd.DataFrame:
    """Tuesday opener spread per game, paired as opener_pick_evaluation pairs it."""

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "home_spread"]
    ].rename(columns={"home_spread": "opener_spread"})
    paired = tue_open.merge(close[["game_id", "close_home_spread"]], on="game_id", how="inner")
    return paired.drop_duplicates("game_id")


def scored_rows(
    model: Any, frame: pd.DataFrame, *, discrete: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Predicted residual, gaussian_median cover probability, optional discrete read."""

    predicted = model.predict(frame, probability_method="gaussian_median")
    residual = predicted["predicted_market_residual"].to_numpy(dtype=float)
    probability = predicted["home_cover_probability"].to_numpy(dtype=float)
    discrete_probability = None
    if discrete:
        discrete_probability = (
            model.predict(frame, probability_method="discrete_residual")["home_cover_probability"]
            .to_numpy(dtype=float)
            .copy()
        )
    return residual, probability, discrete_probability


def walk_forward(
    features: pd.DataFrame,
    market_root: Path,
    min_train: int,
    families: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """One weekly refit per arm, scored at the table line and at the opener."""

    paired = opener_lines(features, market_root)
    opener_map = dict(zip(paired["game_id"], paired["opener_spread"], strict=True))
    close_map = dict(zip(paired["game_id"], paired["close_home_spread"], strict=True))

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    generator = np.random.default_rng(HALF_SEED)
    records: list[pd.DataFrame] = []
    week_keys = completed[["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    for season, week in week_keys.itertuples(index=False):
        week_rows = completed.loc[completed["season"].eq(season) & completed["week"].eq(week)]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train:
            continue

        at_line = week_rows.copy()
        at_line["surface_line"] = pd.to_numeric(at_line["spread_line"], errors="coerce")
        open_rows = week_rows.loc[week_rows["game_id"].isin(opener_map)].copy()
        if not open_rows.empty:
            open_rows["surface_line"] = open_rows["game_id"].map(opener_map).astype(float)
            open_rows["spread_line"] = open_rows["surface_line"]
            open_rows["close_line"] = open_rows["game_id"].map(close_map).astype(float)
        surfaces = [("close_proxy", at_line)]
        if not open_rows.empty:
            surfaces.append(("opener", open_rows))

        order = generator.permutation(len(training))
        half_a = training.iloc[order[: len(training) // 2]]
        half_b = training.iloc[order[len(training) // 2 :]]

        line_train = pd.to_numeric(training["spread_line"], errors="coerce").abs()
        specialists = {
            "F_spec105": training.loc[line_train.ge(BIG)],
            "F_spec75": training.loc[line_train.ge(7.5)],
        }

        arms: list[tuple[str, pd.DataFrame, str, bool]] = [
            ("F_served", training, PROFILE, True),
        ]
        for name, rows in specialists.items():
            if len(rows) >= SPEC_FLOOR:
                arms.append((name, rows, PROFILE, False))
        for family in families:
            arms.append((f"family_{family}", training, f"weak_stack_only_{family}", False))

        for arm, fit_rows, profile, discrete in arms:
            try:
                model = fit_margin_model(
                    fit_rows,
                    target="market_residual",
                    model_name="ridge",
                    feature_profile=profile,
                    ridge_alpha=10.0,
                )
            except ValueError:
                continue
            half_models = None
            if arm.startswith("family_") or arm == "F_served":
                try:
                    half_models = (
                        fit_margin_model(
                            half_a,
                            target="market_residual",
                            model_name="ridge",
                            feature_profile=profile,
                            ridge_alpha=10.0,
                        ),
                        fit_margin_model(
                            half_b,
                            target="market_residual",
                            model_name="ridge",
                            feature_profile=profile,
                            ridge_alpha=10.0,
                        ),
                    )
                except ValueError:
                    half_models = None
            for surface, sframe in surfaces:
                residual, probability, discrete_probability = scored_rows(
                    model, sframe, discrete=discrete
                )
                block = pd.DataFrame(
                    {
                        "arm": arm,
                        "surface": surface,
                        "game_id": sframe["game_id"].astype(str).to_numpy(),
                        "season": int(season),
                        "week": int(week),
                        "line": sframe["surface_line"].to_numpy(dtype=float),
                        "result": pd.to_numeric(sframe["result"], errors="coerce").to_numpy(
                            dtype=float
                        ),
                        "residual": residual,
                        "home_cover_probability": probability,
                        "training_rows": len(fit_rows),
                    }
                )
                block["discrete_probability"] = (
                    discrete_probability if discrete_probability is not None else np.nan
                )
                if half_models is not None:
                    block["residual_half_a"] = (
                        half_models[0]
                        .predict(sframe, probability_method="gaussian_median")[
                            "predicted_market_residual"
                        ]
                        .to_numpy(dtype=float)
                    )
                    block["residual_half_b"] = (
                        half_models[1]
                        .predict(sframe, probability_method="gaussian_median")[
                            "predicted_market_residual"
                        ]
                        .to_numpy(dtype=float)
                    )
                else:
                    block["residual_half_a"] = np.nan
                    block["residual_half_b"] = np.nan
                records.append(block)
        print(f"scored {season} week {week} ({len(arms)} arms)", flush=True)

    scored = pd.concat(records, ignore_index=True)
    scored["margin_vs_line"] = scored["result"] - scored["line"]
    scored["pick_home"] = scored["residual"].gt(0.0)
    scored["correct"] = pick_correct(scored["pick_home"], scored["margin_vs_line"])
    scored["home_cover"] = np.where(
        scored["margin_vs_line"].gt(0.0),
        1.0,
        np.where(scored["margin_vs_line"].lt(0.0), 0.0, np.nan),
    )
    scored["big"] = scored["line"].abs().ge(BIG)
    scored["era"] = ""
    for label, start, end in ERAS:
        mask = scored["season"].between(start, end)
        scored.loc[mask, "era"] = label
    return scored


def accuracy_cell(subset: pd.DataFrame) -> dict[str, Any]:
    """Accuracy above the coin flip, week-blocked and season-blocked."""

    live = subset.loc[subset["correct"].notna()]
    if live.empty:
        return {}
    delta = (live["correct"].to_numpy(dtype=float) - 0.5)[:, np.newaxis]
    blocks = live[["season", "week"]].reset_index(drop=True)
    out: dict[str, Any] = {
        "n": len(live),
        "accuracy": float(live["correct"].mean()),
        "accuracy_points_above_coin_flip": float((live["correct"].mean() - 0.5) * 100.0),
    }
    for label, block in (("week", "week"), ("season", "season")):
        stats = _blocked(delta, blocks, block)
        out[f"{label}_lower_accuracy_points"] = stats[0]
        out[f"{label}_upper_accuracy_points"] = stats[1]
        out[f"{label}_probability_positive"] = stats[2]
    graded = live.loc[live["home_cover"].notna()]
    if not graded.empty:
        out["brier"] = float(
            np.mean((graded["home_cover_probability"] - graded["home_cover"]) ** 2)
        )
        out["brier_coin_flip"] = float(np.mean((0.5 - graded["home_cover"]) ** 2))
        out["mean_stated_confidence"] = float(
            np.mean(
                np.maximum(graded["home_cover_probability"], 1.0 - graded["home_cover_probability"])
            )
        )
    out["favourite_share_of_picks"] = float(
        np.mean(
            np.where(
                live["line"].to_numpy(dtype=float) > 0,
                live["pick_home"].to_numpy(dtype=bool),
                ~live["pick_home"].to_numpy(dtype=bool),
            )[live["line"].to_numpy(dtype=float) != 0]
        )
        if int((live["line"] != 0).sum()) > 0
        else np.nan
    )
    return out


def _blocked(delta: np.ndarray, blocks: pd.DataFrame, block: str) -> tuple[float, float, float]:
    group_columns = ["season", "week"] if block == "week" else ["season"]
    indices = list(blocks.groupby(group_columns, sort=False, dropna=False).indices.values())
    block_sums = np.vstack([delta[idx].sum(axis=0) for idx in indices])
    block_counts = np.array([len(idx) for idx in indices], dtype=float)
    generator = np.random.default_rng(SEED)
    draws = np.empty((SAMPLES, delta.shape[1]), dtype=float)
    for index in range(SAMPLES):
        selected = generator.integers(0, len(indices), size=len(indices))
        draws[index] = block_sums[selected].sum(axis=0) / block_counts[selected].sum()
    lower = float(np.quantile(draws, 0.025) * 100.0)
    upper = float(np.quantile(draws, 0.975) * 100.0)
    return lower, upper, float(probability_positive_from_draws(draws))


def reliability_cell(subset: pd.DataFrame) -> dict[str, Any]:
    """Split-half reliability of the arm's own signal, week-blocked."""

    live = subset.loc[subset["residual_half_a"].notna() & subset["residual_half_b"].notna()]
    if len(live) < 20:
        return {}
    a = live["residual_half_a"].to_numpy(dtype=float)
    b = live["residual_half_b"].to_numpy(dtype=float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return {"reliability_n": len(live), "split_half_r": 0.0, "spearman_brown": 0.0}
    r = float(np.corrcoef(a, b)[0, 1])
    blocks = live[["season", "week"]].reset_index(drop=True)
    indices = list(blocks.groupby(["season", "week"], sort=False, dropna=False).indices.values())
    generator = np.random.default_rng(SEED)
    draws = np.empty(2_000, dtype=float)
    for index in range(2_000):
        selected = generator.integers(0, len(indices), size=len(indices))
        rows = np.concatenate([indices[choice] for choice in selected])
        sa, sb = a[rows], b[rows]
        draws[index] = 0.0 if np.std(sa) == 0.0 or np.std(sb) == 0.0 else np.corrcoef(sa, sb)[0, 1]
    return {
        "reliability_n": len(live),
        "split_half_r": r,
        "spearman_brown": float(2.0 * r / (1.0 + r)) if r > -1.0 else float("nan"),
        "reliability_lower_r": float(np.quantile(draws, 0.025)),
        "reliability_upper_r": float(np.quantile(draws, 0.975)),
        "reliability_probability_positive": float(probability_positive_from_draws(draws)),
    }


def paired_cell(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    """Paired accuracy-point difference on the games both arms scored."""

    left = candidate.set_index("game_id")[["correct", "season", "week"]].dropna(subset=["correct"])
    right = baseline.set_index("game_id")["correct"].dropna()
    joined = left.join(right.rename("baseline"), how="inner")
    if joined.empty:
        return {}
    delta = (joined["correct"] - joined["baseline"]).to_numpy(dtype=float)[:, np.newaxis]
    blocks = joined[["season", "week"]].reset_index(drop=True)
    out: dict[str, Any] = {
        "n": len(joined),
        "candidate_accuracy": float(joined["correct"].mean()),
        "baseline_accuracy": float(joined["baseline"].mean()),
        "picks_changed": int((joined["correct"] != joined["baseline"]).sum()),
        "accuracy_points": float((joined["correct"].mean() - joined["baseline"].mean()) * 100.0),
    }
    for label in ("week", "season"):
        stats = _blocked(delta, blocks, label)
        out[f"{label}_lower_accuracy_points"] = stats[0]
        out[f"{label}_upper_accuracy_points"] = stats[1]
        out[f"{label}_probability_positive"] = stats[2]
    return out


def margin_facts(features: pd.DataFrame) -> dict[str, Any]:
    """Integer margin mass and push rates at 10.5+, pooled and by era."""

    frame = regular_season_rows(features).copy()
    frame = frame.loc[frame["result"].notna()].copy()
    frame["line"] = pd.to_numeric(frame["spread_line"], errors="coerce")
    frame = frame.loc[frame["line"].notna()]
    frame["margin_vs_line"] = pd.to_numeric(frame["result"], errors="coerce") - frame["line"]
    frame["favourite_ward"] = np.where(
        frame["line"] >= 0, frame["margin_vs_line"], -frame["margin_vs_line"]
    )
    frame["favourite_margin"] = np.where(
        frame["line"] >= 0,
        pd.to_numeric(frame["result"], errors="coerce"),
        -pd.to_numeric(frame["result"], errors="coerce"),
    )
    big = frame.loc[frame["line"].abs().ge(BIG)].copy()
    facts: dict[str, Any] = {"n_big": len(big)}

    margins = big["favourite_margin"].astype(int).value_counts().sort_index()
    facts["favourite_margin_histogram"] = {
        int(k): {"games": int(v), "share": float(v / len(big))} for k, v in margins.items()
    }
    facts["favourite_margin_key_number_share"] = float(
        sum(int(margins.get(k, 0)) for k in (3, 7, 10, 14, 17, -3, -7, -10, -14, -17)) / len(big)
    )
    integer_lines = big.loc[big["line"].abs().mod(1.0).eq(0.0)]
    distance = integer_lines["favourite_ward"].astype(int).value_counts().sort_index()
    facts["integer_line_games"] = len(integer_lines)
    facts["distance_histogram_integer_lines"] = {
        int(k): {"games": int(v), "share": float(v / len(integer_lines))}
        for k, v in distance.items()
    }
    facts["push_rate_big"] = float((big["margin_vs_line"] == 0).mean())
    facts["push_rate_integer_lines"] = float(
        (integer_lines["margin_vs_line"] == 0).mean() if len(integer_lines) else float("nan")
    )
    per_line = []
    for line_value, group in big.groupby(big["line"].abs()):
        if len(group) < 20:
            continue
        per_line.append(
            {
                "abs_line": float(line_value),
                "games": len(group),
                "push_rate": float((group["margin_vs_line"] == 0).mean()),
                "favourite_cover_rate": float(
                    (group["favourite_ward"] > 0).sum()
                    / max(int((group["margin_vs_line"] != 0).sum()), 1)
                ),
            }
        )
    facts["per_line"] = per_line
    for key_line in (11.0, 13.0, 14.0, 17.0):
        group = big.loc[big["line"].abs().eq(key_line)]
        facts[f"push_at_{int(key_line)}"] = {
            "games": len(group),
            "pushes": int((group["margin_vs_line"] == 0).sum()),
            "push_rate": float((group["margin_vs_line"] == 0).mean()) if len(group) else None,
        }
    facts["by_era"] = {}
    for label, start, end in ERAS:
        era_rows = big.loc[big["season"].between(start, end)]
        if era_rows.empty:
            continue
        era_margins = era_rows["favourite_margin"].astype(int).value_counts()
        facts["by_era"][label] = {
            "games": len(era_rows),
            "push_rate": float((era_rows["margin_vs_line"] == 0).mean()),
            "underdog_cover_rate": float(
                (era_rows["favourite_ward"] < 0).sum()
                / max(int((era_rows["margin_vs_line"] != 0).sum()), 1)
            ),
            "favourite_margin_key_number_share": float(
                sum(int(era_margins.get(k, 0)) for k in (3, 7, 10, 14, 17, -3, -7, -10, -14, -17))
                / len(era_rows)
            ),
        }
    return facts


def market_move(archive: Path) -> dict[str, Any]:
    """Does the opener-to-close move predict the cover at 10.5+?"""

    per_game = pd.read_parquet(archive / "per_game.parquet")
    per_game["big"] = pd.to_numeric(per_game["tue_open_home_spread"], errors="coerce").abs().ge(BIG)
    out: dict[str, Any] = {}
    for label, rows in (
        ("2020_2025", per_game),
        ("2023_2025", per_game.loc[per_game["season"].ge(2023)]),
    ):
        for scope, subset in (("all_lines", rows), ("big", rows.loc[rows["big"]])):
            moved = subset.loc[
                subset["open_move"].ne(0.0) & subset["oracle_correct_at_open"].notna()
            ]
            if moved.empty:
                continue
            delta = (moved["oracle_correct_at_open"].to_numpy(dtype=float) - 0.5)[:, np.newaxis]
            blocks = moved[["season", "week"]].reset_index(drop=True)
            week = _blocked(delta, blocks, "week")
            out[f"follow_move_{scope}_{label}"] = {
                "n": len(moved),
                "zero_move_excluded": len(subset.loc[subset["open_move"].eq(0.0)]),
                "accuracy": float(moved["oracle_correct_at_open"].mean()),
                "accuracy_points_above_coin_flip": float(
                    (moved["oracle_correct_at_open"].mean() - 0.5) * 100.0
                ),
                "week_lower_accuracy_points": week[0],
                "week_upper_accuracy_points": week[1],
                "week_probability_positive": week[2],
                "mean_absolute_move": float(moved["open_move"].abs().mean()),
            }
    big_moved = per_game.loc[
        per_game["big"] & per_game["open_move"].ne(0.0) & per_game["oracle_correct_at_open"].notna()
    ]
    out["follow_move_big_by_season"] = {
        int(season): {
            "n": len(group),
            "accuracy": float(group["oracle_correct_at_open"].mean()),
        }
        for season, group in big_moved.groupby("season")
    }
    served = per_game.loc[per_game["big"] & per_game["correct_at_open_probability_rule"].notna()]
    out["served_model_big"] = {
        "n": len(served),
        "accuracy": float(served["correct_at_open_probability_rule"].mean()),
        "accuracy_raw": float(served["correct_at_open_probability_rule_raw"].mean()),
    }
    agree = per_game.loc[
        per_game["big"]
        & per_game["open_move"].ne(0.0)
        & per_game["correct_at_open_probability_rule"].notna()
    ].copy()
    agree["model_home"] = agree["pick_home_at_open_probability_rule"].astype(bool)
    agree["move_home"] = agree["open_move"].gt(0.0)
    out["model_versus_move_big"] = {
        "n": len(agree),
        "agree_share": float((agree["model_home"] == agree["move_home"]).mean()),
        "model_accuracy_when_agree": float(
            agree.loc[
                agree["model_home"].eq(agree["move_home"]), "correct_at_open_probability_rule"
            ].mean()
        ),
        "model_accuracy_when_disagree": float(
            agree.loc[
                agree["model_home"].ne(agree["move_home"]), "correct_at_open_probability_rule"
            ].mean()
        ),
        "n_agree": int((agree["model_home"] == agree["move_home"]).sum()),
        "n_disagree": int((agree["model_home"] != agree["move_home"]).sum()),
    }
    return out


def handle_side(features: pd.DataFrame, archive_path: Path) -> dict[str, Any]:
    """Does the heavy-money side cover at 10.5+?"""

    if not archive_path.is_file():
        return {"available": False, "reason": f"missing {archive_path}"}
    archive = pd.read_parquet(archive_path)
    working = archive.loc[archive["spread_home_money_pct"].notna()].copy()
    if working.empty:
        return {"available": False, "reason": "no money share rows"}
    working["away_team"] = working["away_team"].map(lambda v: TEAM_ALIASES.get(v, v))
    working["home_team"] = working["home_team"].map(lambda v: TEAM_ALIASES.get(v, v))
    working["start_time_utc"] = pd.to_datetime(working["start_time_utc"], utc=True, errors="coerce")
    schedule = regular_season_rows(features).copy()
    schedule = schedule.loc[schedule["result"].notna()].copy()
    schedule["kickoff"] = pd.to_datetime(schedule["gameday"], utc=True, errors="coerce")
    if "gametime" in schedule.columns:
        schedule["kickoff"] = pd.to_datetime(
            schedule["gameday"].astype(str) + " " + schedule["gametime"].astype(str),
            utc=True,
            errors="coerce",
        ).fillna(schedule["kickoff"])
    merged = working.merge(
        schedule[
            [
                "game_id",
                "season",
                "week",
                "away_team",
                "home_team",
                "kickoff",
                "spread_line",
                "result",
            ]
        ],
        on=["away_team", "home_team"],
        how="inner",
        suffixes=("", "_sched"),
    )
    merged["kickoff_delta_hours"] = (
        merged["kickoff"] - merged["start_time_utc"]
    ).dt.total_seconds().abs() / 3600.0
    matched = merged.loc[merged["kickoff_delta_hours"] <= KICKOFF_MATCH_HOURS].copy()
    matched = matched.loc[matched["capture_ts"] < matched["kickoff"]]
    if matched.empty:
        return {"available": False, "reason": "no pregame captures matched"}
    latest = matched.sort_values("capture_ts").groupby("game_id", as_index=False).tail(1)
    latest["margin_vs_line"] = latest["result"] - latest["spread_line"]
    latest["pick_home"] = latest["spread_home_money_pct"].gt(50.0)
    latest["correct"] = pick_correct(latest["pick_home"], latest["margin_vs_line"])
    latest["big"] = latest["spread_line"].abs().ge(BIG)
    out: dict[str, Any] = {
        "available": True,
        "seasons": sorted(int(s) for s in latest["season"].unique()),
        "games_matched": len(latest),
    }
    for scope, subset in (
        ("all_lines", latest),
        ("big", latest.loc[latest["big"]]),
        (
            "big_heavy70",
            latest.loc[
                latest["big"]
                & (latest[["spread_home_money_pct", "spread_away_money_pct"]].max(axis=1).ge(70.0))
            ],
        ),
    ):
        live = subset.loc[subset["correct"].notna()]
        if live.empty:
            out[f"handle_{scope}"] = {"n": 0}
            continue
        delta = (live["correct"].to_numpy(dtype=float) - 0.5)[:, np.newaxis]
        blocks = live[["season", "week"]].reset_index(drop=True)
        week = _blocked(delta, blocks, "week")
        out[f"handle_{scope}"] = {
            "n": len(live),
            "accuracy": float(live["correct"].mean()),
            "accuracy_points_above_coin_flip": float((live["correct"].mean() - 0.5) * 100.0),
            "week_lower_accuracy_points": week[0],
            "week_upper_accuracy_points": week[1],
            "week_probability_positive": week[2],
        }
    return out


def discrete_versus_gaussian(scored: pd.DataFrame) -> dict[str, Any]:
    """Side flips and Brier when the same weekly residuals are read discretely."""

    served = scored.loc[scored["arm"].eq("F_served") & scored["discrete_probability"].notna()]
    out: dict[str, Any] = {}
    for surface in sorted(served["surface"].unique()):
        rows = served.loc[served["surface"].eq(surface)]
        for scope, subset in (("all_lines", rows), ("big", rows.loc[rows["big"]])):
            live = subset.loc[subset["home_cover"].notna()]
            if live.empty:
                continue
            gaussian_home = live["home_cover_probability"].ge(0.5)
            discrete_home = live["discrete_probability"].ge(0.5)
            out[f"{surface}_{scope}"] = {
                "n": len(live),
                "side_flips": int((gaussian_home != discrete_home).sum()),
                "side_flip_share": float((gaussian_home != discrete_home).mean()),
                "brier_gaussian_median": float(
                    np.mean((live["home_cover_probability"] - live["home_cover"]) ** 2)
                ),
                "brier_discrete_residual": float(
                    np.mean((live["discrete_probability"] - live["home_cover"]) ** 2)
                ),
                "accuracy_gaussian_median": float(
                    pick_correct(gaussian_home, live["margin_vs_line"]).mean()
                ),
                "accuracy_discrete_residual": float(
                    pick_correct(discrete_home, live["margin_vs_line"]).mean()
                ),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--public-betting", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--reuse-scored", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    registered = register_family_profiles()
    if args.reuse_scored and (args.out / "scored.parquet").is_file():
        scored = pd.read_parquet(args.out / "scored.parquet")
    else:
        scored = walk_forward(features, args.market_root, args.min_train_games, registered)
        scored.to_parquet(args.out / "scored.parquet", index=False)

    rows: list[dict[str, Any]] = []
    for arm in sorted(scored["arm"].unique()):
        for surface in sorted(scored["surface"].unique()):
            base = scored.loc[scored["arm"].eq(arm) & scored["surface"].eq(surface) & scored["big"]]
            if base.empty:
                continue
            windows: list[tuple[str, pd.DataFrame]] = [("pooled", base)]
            for label, start, end in ERAS:
                era_rows = base.loc[base["season"].between(start, end)]
                if not era_rows.empty:
                    windows.append((label, era_rows))
            for window, subset in windows:
                cell = accuracy_cell(subset)
                if not cell:
                    continue
                cell.update(reliability_cell(subset))
                cell.update({"arm": arm, "surface": surface, "window": window, "scope": "big"})
                rows.append(cell)
            all_lines = scored.loc[scored["arm"].eq(arm) & scored["surface"].eq(surface)]
            cell = accuracy_cell(all_lines)
            if cell:
                cell.update(reliability_cell(all_lines))
                cell.update(
                    {"arm": arm, "surface": surface, "window": "pooled", "scope": "all_lines"}
                )
                rows.append(cell)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "arm_cells.csv", index=False)

    paired: list[dict[str, Any]] = []
    for surface in sorted(scored["surface"].unique()):
        served = scored.loc[
            scored["arm"].eq("F_served") & scored["surface"].eq(surface) & scored["big"]
        ]
        for arm in ("F_spec105", "F_spec75"):
            candidate = scored.loc[
                scored["arm"].eq(arm) & scored["surface"].eq(surface) & scored["big"]
            ]
            if candidate.empty:
                continue
            windows = [("pooled", candidate)]
            for label, start, end in ERAS:
                era_rows = candidate.loc[candidate["season"].between(start, end)]
                if not era_rows.empty:
                    windows.append((label, era_rows))
            for window, subset in windows:
                cell = paired_cell(subset, served)
                if cell:
                    cell.update({"arm": arm, "surface": surface, "window": window})
                    paired.append(cell)
    paired_table = pd.DataFrame(paired)
    paired_table.to_csv(args.out / "paired_cells.csv", index=False)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "features": str(args.features),
        "archive": str(args.archive),
        "min_train_games": args.min_train_games,
        "samples": SAMPLES,
        "seed": SEED,
        "half_seed": HALF_SEED,
        "family_columns": {k: list(v) for k, v in registered.items()},
        "first_scored_season": int(scored["season"].min()),
        "arm_cells": rows,
        "paired_cells": paired,
        "margin_facts": margin_facts(features),
        "market_move": market_move(args.archive),
        "handle_side": handle_side(features, args.public_betting),
        "discrete_versus_gaussian": discrete_versus_gaussian(scored),
    }
    (args.out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("first_scored_season",)}, indent=2))
    headline = [
        column
        for column in (
            "arm",
            "surface",
            "n",
            "accuracy",
            "week_lower_accuracy_points",
            "week_upper_accuracy_points",
            "week_probability_positive",
            "split_half_r",
            "spearman_brown",
        )
        if column in table.columns
    ]
    print(
        table.loc[table["scope"].eq("big") & table["window"].eq("pooled")][headline].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

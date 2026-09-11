from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402

GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
OPENER_ARCHIVE = REPO / "artifacts" / "opener_evaluation" / "20260910T211255Z" / "per_game.parquet"
OPENER_METADATA = REPO / "artifacts" / "opener_evaluation" / "20260910T211255Z" / "metadata.json"
OUT_ROOT = REPO / "artifacts" / "bayesian_team_model_on_production"

HYPERPARAMETER_SEASONS = (2009, 2019)
HYPERPARAMETER_BURN_IN_SEASONS = 2
WEIGHT_SEASONS = (2012, 2019)
EVALUATION_SEASONS = (2020, 2025)
ASSIGNED_WINDOW = (2020, 2021)
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260817
PERMUTATION_DRAWS = 200
PERMUTATION_SEED = 20260911
SWEEP_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Hyperparameters:
    sigma: float
    w_week: float
    w_off: float
    w_hfa: float
    phi_week: float
    phi_off: float
    tau0: float
    hfa_sd0: float

    def as_dict(self) -> dict[str, float]:
        return {
            "observation_sd_points": self.sigma,
            "weekly_innovation_sd_points": self.w_week,
            "offseason_innovation_sd_points": self.w_off,
            "hfa_innovation_sd_points": self.w_hfa,
            "weekly_pooling_factor": self.phi_week,
            "offseason_pooling_factor": self.phi_off,
            "prior_strength_sd_points": self.tau0,
            "prior_hfa_sd_points": self.hfa_sd0,
        }


START_VECTOR = np.array(
    [
        np.log(13.5),
        np.log(0.60),
        np.log(3.0),
        np.log(0.15),
        np.log(0.995 / 0.005),
        np.log(0.70 / 0.30),
        np.log(6.0),
        np.log(3.0),
    ],
    dtype=float,
)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _unpack(vector: np.ndarray) -> Hyperparameters:
    return Hyperparameters(
        sigma=float(np.exp(vector[0])),
        w_week=float(np.exp(vector[1])),
        w_off=float(np.exp(vector[2])),
        w_hfa=float(np.exp(vector[3])),
        phi_week=_sigmoid(float(vector[4])),
        phi_off=_sigmoid(float(vector[5])),
        tau0=float(np.exp(vector[6])),
        hfa_sd0=float(np.exp(vector[7])),
    )


def load_games() -> pd.DataFrame:
    frame = pd.read_parquet(GAME_FEATURES)
    columns = [
        "game_id",
        "season",
        "week",
        "gameday",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "result",
        "spread_line",
        "ats_margin",
        "neutral_site",
    ]
    frame = frame.loc[:, columns].copy()
    frame = frame.loc[frame["result"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["neutral"] = frame["neutral_site"].fillna(0).astype(int)
    frame = frame.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    return frame


def run_filter(
    games: pd.DataFrame,
    params: Hyperparameters,
    *,
    likelihood_seasons: tuple[int, int] | None = None,
    collect: bool = False,
) -> tuple[float, pd.DataFrame | None, pd.DataFrame | None]:
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    index = {team: position for position, team in enumerate(teams)}
    n_teams = len(teams)
    dim = n_teams + 1

    mean = np.zeros(dim, dtype=float)
    cov = np.zeros((dim, dim), dtype=float)
    np.fill_diagonal(cov, params.tau0**2)
    cov[n_teams, n_teams] = params.hfa_sd0**2

    transition_week = np.eye(dim, dtype=float)
    transition_week[:n_teams, :n_teams] *= params.phi_week
    transition_off = np.eye(dim, dtype=float)
    transition_off[:n_teams, :n_teams] *= params.phi_off

    noise_week = np.zeros(dim, dtype=float)
    noise_week[:n_teams] = params.w_week**2
    noise_week[n_teams] = params.w_hfa**2
    noise_off = np.zeros(dim, dtype=float)
    noise_off[:n_teams] = params.w_off**2
    noise_off[n_teams] = params.w_hfa**2

    log_likelihood = 0.0
    game_rows: list[dict[str, float | str | int]] = []
    panel_rows: list[dict[str, float | str | int]] = []

    previous_season: int | None = None
    for (season, week), batch in games.groupby(["season", "week"], sort=True):
        if previous_season is None:
            pass
        elif season != previous_season:
            mean = transition_off @ mean
            cov = transition_off @ cov @ transition_off.T
            cov[np.diag_indices(dim)] += noise_off
        else:
            mean = transition_week @ mean
            cov = transition_week @ cov @ transition_week.T
            cov[np.diag_indices(dim)] += noise_week
        previous_season = int(season)

        count = len(batch)
        design = np.zeros((count, dim), dtype=float)
        for row_position, (_, row) in enumerate(batch.iterrows()):
            design[row_position, index[row["home_team"]]] = 1.0
            design[row_position, index[row["away_team"]]] = -1.0
            design[row_position, n_teams] = 0.0 if int(row["neutral"]) == 1 else 1.0
        observed = batch["result"].to_numpy(dtype=float)

        predicted = design @ mean
        parameter_cov = design @ cov @ design.T
        innovation_cov = parameter_cov + (params.sigma**2) * np.eye(count)
        residual = observed - predicted

        if likelihood_seasons is not None:
            low, high = likelihood_seasons
            if low <= season <= high:
                sign, logdet = np.linalg.slogdet(innovation_cov)
                if sign <= 0:
                    return float("inf"), None, None
                solved = np.linalg.solve(innovation_cov, residual)
                log_likelihood += -0.5 * (
                    logdet + float(residual @ solved) + count * np.log(2.0 * np.pi)
                )

        if collect:
            parameter_sd = np.sqrt(np.clip(np.diag(parameter_cov), 0.0, None))
            for row_position, (_, row) in enumerate(batch.iterrows()):
                game_rows.append(
                    {
                        "game_id": str(row["game_id"]),
                        "season": int(season),
                        "week": int(week),
                        "posterior_margin": float(predicted[row_position]),
                        "posterior_sd": float(parameter_sd[row_position]),
                    }
                )
            strength_sd = np.sqrt(np.clip(np.diag(cov)[:n_teams], 0.0, None))
            for team, position in index.items():
                panel_rows.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "team": team,
                        "strength": float(mean[position]),
                        "strength_sd": float(strength_sd[position]),
                    }
                )

        gain = np.linalg.solve(innovation_cov, design @ cov).T
        mean = mean + gain @ residual
        cov = cov - gain @ design @ cov
        cov = 0.5 * (cov + cov.T)

    game_frame = pd.DataFrame(game_rows) if collect else None
    panel_frame = pd.DataFrame(panel_rows) if collect else None
    return log_likelihood, game_frame, panel_frame


def fit_hyperparameters(games: pd.DataFrame) -> tuple[Hyperparameters, dict[str, float]]:
    low, high = HYPERPARAMETER_SEASONS
    fitting_games = games.loc[games["season"].between(low, high)].copy()
    likelihood_window = (low + HYPERPARAMETER_BURN_IN_SEASONS, high)

    def objective(vector: np.ndarray) -> float:
        params = _unpack(vector)
        if params.sigma < 1.0 or params.sigma > 40.0:
            return 1e12
        value, _, _ = run_filter(
            fitting_games, params, likelihood_seasons=likelihood_window, collect=False
        )
        if not np.isfinite(value):
            return 1e12
        return -value

    result = minimize(
        objective,
        START_VECTOR,
        method="Nelder-Mead",
        options={"maxiter": 4000, "maxfev": 4000, "xatol": 1e-4, "fatol": 1e-3},
    )
    params = _unpack(np.asarray(result.x, dtype=float))
    start_value = objective(START_VECTOR)
    diagnostics = {
        "fit_seasons": list(HYPERPARAMETER_SEASONS),
        "likelihood_seasons": list(likelihood_window),
        "fitting_games": len(fitting_games),
        "negative_log_likelihood_start": float(start_value),
        "negative_log_likelihood_fitted": float(result.fun),
        "optimizer": "Nelder-Mead",
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "converged": bool(result.success),
    }
    return params, diagnostics


def derive_stacking_weight(panel: pd.DataFrame) -> dict[str, float]:
    low, high = WEIGHT_SEASONS
    frame = panel.loc[
        panel["season"].between(low, high)
        & panel["spread_line"].notna()
        & panel["ats_margin"].notna()
    ].copy()
    residual = frame["posterior_margin"].to_numpy(float) - frame["spread_line"].to_numpy(float)
    realised = frame["ats_margin"].to_numpy(float)
    design = np.column_stack([np.ones_like(residual), residual])
    coefficients, *_ = np.linalg.lstsq(design, realised, rcond=None)
    scale = float(np.std(realised, ddof=1))
    correlation = float(np.corrcoef(residual, realised)[0, 1])
    return {
        "seasons": [low, high],
        "games": len(frame),
        "intercept_points": float(coefficients[0]),
        "slope_points_per_point": float(coefficients[1]),
        "realised_residual_sd_points": scale,
        "k": float(coefficients[1] / scale),
        "bayes_residual_sd_points": float(np.std(residual, ddof=1)),
        "correlation": correlation,
    }


def accuracy_metric(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def summarise_pair(frame: pd.DataFrame) -> dict[str, object]:
    point = accuracy_metric(frame)
    week = week_blocked_bootstrap(
        frame, accuracy_metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    season = week_blocked_bootstrap(
        frame, accuracy_metric, block="season", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    week_row = week.loc[week["metric"].eq("accuracy_points")].iloc[0]
    season_row = season.loc[season["metric"].eq("accuracy_points")].iloc[0]
    changed = int((frame["candidate_pick_home"] != frame["baseline_pick_home"]).sum())
    changed_frame = frame.loc[frame["candidate_pick_home"] != frame["baseline_pick_home"]]
    return {
        **point,
        "picks_changed": changed,
        "picks_changed_share": float(changed / len(frame)) if len(frame) else float("nan"),
        "candidate_wins_on_changed": (
            float(changed_frame["candidate_correct"].mean()) if changed else float("nan")
        ),
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95": [float(season_row["lower"]), float(season_row["upper"])],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "games": len(frame),
        "weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "seasons": int(frame["season"].nunique()),
    }


def per_season_deltas(frame: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for season, group in frame.groupby("season", sort=True):
        rows.append(
            {
                "season": int(season),
                "games": len(group),
                "baseline_accuracy": 100.0 * float(group["baseline_correct"].mean()),
                "candidate_accuracy": 100.0 * float(group["candidate_correct"].mean()),
                "accuracy_points": 100.0
                * float((group["candidate_correct"] - group["baseline_correct"]).mean()),
                "picks_changed": int(
                    (group["candidate_pick_home"] != group["baseline_pick_home"]).sum()
                ),
            }
        )
    return rows


def frozen_pick_null(frame: pd.DataFrame, observed: float) -> dict[str, float]:
    generator = np.random.default_rng(PERMUTATION_SEED)
    blocks = [group.index.to_numpy() for _, group in frame.groupby(["season", "week"], sort=True)]
    baseline_pick = frame["baseline_pick_home"].to_numpy(bool)
    candidate_pick = frame["candidate_pick_home"].to_numpy(bool)
    cover = frame["home_cover"].to_numpy(bool)
    positions = {label: position for position, label in enumerate(frame.index)}
    draws = np.empty(PERMUTATION_DRAWS, dtype=float)
    for draw in range(PERMUTATION_DRAWS):
        permuted = cover.copy()
        for block in blocks:
            block_positions = np.array([positions[label] for label in block])
            permuted[block_positions] = generator.permutation(cover[block_positions])
        baseline_correct = baseline_pick == permuted
        candidate_correct = candidate_pick == permuted
        draws[draw] = 100.0 * float(np.mean(candidate_correct) - np.mean(baseline_correct))
    return {
        "draws": PERMUTATION_DRAWS,
        "null_mean": float(np.mean(draws)),
        "null_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "observed_percentile": float(np.mean(draws <= observed)),
    }


def walk_forward_stack(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    ordered = frame.sort_values(["season", "week"]).reset_index(drop=True)
    effective = scale * ordered["z"].to_numpy(float)
    bayes = ordered["bayes_residual"].to_numpy(float)
    realised = ordered["margin_vs_open"].to_numpy(float)
    keys = list(zip(ordered["season"].tolist(), ordered["week"].tolist(), strict=True))

    control_pick = np.full(len(ordered), np.nan)
    candidate_pick = np.full(len(ordered), np.nan)
    unique_keys: list[tuple[int, int]] = []
    for key in keys:
        if not unique_keys or unique_keys[-1] != key:
            unique_keys.append(key)
    key_array = np.array([f"{season:04d}_{week:02d}" for season, week in keys])
    for season, week in unique_keys:
        label = f"{season:04d}_{week:02d}"
        current = key_array == label
        history = key_array < label
        if int(history.sum()) < DEFAULT_MIN_TRAIN_GAMES:
            continue
        target = realised[history]
        control_design = np.column_stack([np.ones(int(history.sum())), effective[history]])
        control_coefficients, *_ = np.linalg.lstsq(control_design, target, rcond=None)
        candidate_design = np.column_stack([control_design, bayes[history]])
        candidate_coefficients, *_ = np.linalg.lstsq(candidate_design, target, rcond=None)
        current_control = np.column_stack([np.ones(int(current.sum())), effective[current]])
        current_candidate = np.column_stack([current_control, bayes[current]])
        control_pick[current] = current_control @ control_coefficients
        candidate_pick[current] = current_candidate @ candidate_coefficients
    ordered["stack_control_score"] = control_pick
    ordered["stack_candidate_score"] = candidate_pick
    return ordered


def leak_control(frame: pd.DataFrame, scale: float, k: float) -> dict[str, object]:
    teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    index = {team: position for position, team in enumerate(teams)}
    design = np.zeros((len(frame), len(teams) + 1), dtype=float)
    for position, (_, row) in enumerate(frame.iterrows()):
        design[position, index[row["home_team"]]] = 1.0
        design[position, index[row["away_team"]]] = -1.0
        design[position, len(teams)] = 1.0
    target = frame["result"].to_numpy(float)
    ridge = design.T @ design + 10.0 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(ridge, design.T @ target)
    leaked_margin = design @ coefficients
    leaked_residual = leaked_margin - frame["tue_open_home_spread"].to_numpy(float)
    scored = frame.copy()
    scored["bayes_residual"] = leaked_residual
    scored["candidate_pick_home"] = (scored["z"].to_numpy(float) + k * leaked_residual) >= 0.0
    scored["candidate_correct"] = (
        scored["candidate_pick_home"].to_numpy(bool) == scored["home_cover"].to_numpy(bool)
    ).astype(float)
    summary = summarise_pair(scored)
    summary["description"] = (
        "leak control: team strengths fitted once over the whole 2020-2025 evaluation sample, "
        "so the columns see the future; same stacking weight k"
    )
    return summary


def tercile_screen(frame: pd.DataFrame, column: str) -> dict[str, object]:
    scored = frame.copy()
    scored["posterior_sd_tercile"] = pd.qcut(scored[column], 3, labels=["low", "mid", "high"])
    rows = []
    for label, group in scored.groupby("posterior_sd_tercile", observed=True, sort=False):

        def metric(inner: pd.DataFrame) -> dict[str, float]:
            return {
                "accuracy": 100.0 * float(inner["baseline_correct"].mean()),
                "brier": float(
                    ((inner["home_cover_probability_at_open"] - inner["home_cover"]) ** 2).mean()
                ),
            }

        point = metric(group)
        boot = week_blocked_bootstrap(
            group, metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )
        accuracy_row = boot.loc[boot["metric"].eq("accuracy")].iloc[0]
        brier_row = boot.loc[boot["metric"].eq("brier")].iloc[0]
        rows.append(
            {
                "tercile": str(label),
                "games": len(group),
                "posterior_sd_min": float(group[column].min()),
                "posterior_sd_max": float(group[column].max()),
                "mean_week": float(group["week"].mean()),
                "accuracy": point["accuracy"],
                "accuracy_ci95": [float(accuracy_row["lower"]), float(accuracy_row["upper"])],
                "brier": point["brier"],
                "brier_ci95": [float(brier_row["lower"]), float(brier_row["upper"])],
                "mean_stated_probability": float(
                    group["home_cover_probability_at_open"]
                    .where(
                        group["baseline_pick_home"], 1.0 - group["home_cover_probability_at_open"]
                    )
                    .mean()
                ),
            }
        )

    contrast_frame = scored.loc[scored["posterior_sd_tercile"].isin(["low", "high"])].copy()

    def contrast(inner: pd.DataFrame) -> dict[str, float]:
        low = inner.loc[inner["posterior_sd_tercile"].eq("low")]
        high = inner.loc[inner["posterior_sd_tercile"].eq("high")]
        if low.empty or high.empty:
            return {"accuracy_points": 0.0, "brier_improvement": 0.0}
        return {
            "accuracy_points": 100.0
            * float(low["baseline_correct"].mean() - high["baseline_correct"].mean()),
            "brier_improvement": float(
                ((high["home_cover_probability_at_open"] - high["home_cover"]) ** 2).mean()
                - ((low["home_cover_probability_at_open"] - low["home_cover"]) ** 2).mean()
            ),
        }

    point = contrast(contrast_frame)
    boot = week_blocked_bootstrap(
        contrast_frame, contrast, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    accuracy_row = boot.loc[boot["metric"].eq("accuracy_points")].iloc[0]
    brier_row = boot.loc[boot["metric"].eq("brier_improvement")].iloc[0]

    correlation = spearmanr(scored[column], scored["week"])
    return {
        "split_column": column,
        "terciles": rows,
        "low_minus_high": {
            "accuracy_points": point["accuracy_points"],
            "week_blocked_ci95": [float(accuracy_row["lower"]), float(accuracy_row["upper"])],
            "probability_positive": float(accuracy_row["probability_positive"]),
            "brier_improvement": point["brier_improvement"],
            "brier_week_blocked_ci95": [float(brier_row["lower"]), float(brier_row["upper"])],
            "brier_probability_positive": float(brier_row["probability_positive"]),
            "games": len(contrast_frame),
            "weeks": int(contrast_frame[["season", "week"]].drop_duplicates().shape[0]),
        },
        "posterior_sd_vs_week_spearman": float(correlation.statistic),
    }


def reliability(panel: pd.DataFrame) -> dict[str, object]:
    evaluation = panel.loc[panel["season"].between(2009, 2025)]
    seasonal = evaluation.groupby(["team", "season"], as_index=False)["strength"].mean()
    seasonal["parity"] = np.where(seasonal["season"] % 2 == 1, "odd", "even")
    split = seasonal.groupby(["team", "parity"], as_index=False)["strength"].mean()
    wide = split.pivot(index="team", columns="parity", values="strength").dropna()
    season_correlation = spearmanr(wide["odd"], wide["even"])

    per_season = []
    for season, group in evaluation.groupby("season", sort=True):
        group = group.copy()
        group["parity"] = np.where(group["week"] % 2 == 1, "odd", "even")
        halves = group.groupby(["team", "parity"], as_index=False)["strength"].mean()
        pivot = halves.pivot(index="team", columns="parity", values="strength").dropna()
        if len(pivot) < 8:
            continue
        result = spearmanr(pivot["odd"], pivot["even"])
        per_season.append({"season": int(season), "spearman": float(result.statistic)})

    within = float(np.mean([row["spearman"] for row in per_season])) if per_season else float("nan")
    brown = 2.0 * within / (1.0 + within) if np.isfinite(within) else float("nan")
    return {
        "odd_vs_even_seasons_spearman": float(season_correlation.statistic),
        "odd_vs_even_seasons_p_value": float(season_correlation.pvalue),
        "odd_vs_even_seasons_teams": len(wide),
        "odd_vs_even_weeks_within_season_spearman_mean": within,
        "odd_vs_even_weeks_spearman_brown": float(brown),
        "per_season_within": per_season,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MOD-06 dynamic arm: build a chronological state-space team-strength posterior and "
            "screen it on top of production's served opener pick"
        )
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    games = load_games()
    params, fit_diagnostics = fit_hyperparameters(games)
    print(json.dumps({"hyperparameters": params.as_dict(), **fit_diagnostics}, indent=2))

    _, posterior, weekly_panel = run_filter(games, params, likelihood_seasons=None, collect=True)
    assert posterior is not None and weekly_panel is not None

    panel = games.merge(posterior, on=["game_id", "season", "week"], how="left")
    weight = derive_stacking_weight(panel)
    print(json.dumps({"stacking_weight": weight}, indent=2))
    k = weight["k"]
    scale = weight["realised_residual_sd_points"]

    archive = pd.read_parquet(OPENER_ARCHIVE)
    archive = archive.merge(
        panel.loc[:, ["game_id", "posterior_margin", "posterior_sd", "home_team", "away_team"]],
        on="game_id",
        how="left",
    )
    missing = int(archive["posterior_margin"].isna().sum())
    archive = archive.loc[archive["posterior_margin"].notna()].copy()

    archive["home_cover"] = np.where(
        archive["margin_vs_open"] > 0, 1.0, np.where(archive["margin_vs_open"] < 0, 0.0, np.nan)
    )
    pushes = int(archive["home_cover"].isna().sum())
    archive = archive.loc[archive["home_cover"].notna()].copy()
    archive["home_cover"] = archive["home_cover"].astype(bool)

    archive["z"] = norm.ppf(
        np.clip(archive["home_cover_probability_at_open"].to_numpy(float), 1e-9, 1.0 - 1e-9)
    )
    archive["baseline_pick_home"] = archive["pick_home_at_open_probability_rule"].astype(bool)
    boundary_mismatch = int(((archive["z"] >= 0.0) != archive["baseline_pick_home"]).sum())
    archive["baseline_correct"] = (
        archive["baseline_pick_home"].to_numpy(bool) == archive["home_cover"].to_numpy(bool)
    ).astype(float)
    saved_mismatch = int(
        (archive["baseline_correct"] != archive["correct_at_open_probability_rule"]).sum()
    )

    archive["bayes_residual"] = archive["posterior_margin"] - archive["tue_open_home_spread"]
    archive["candidate_score"] = archive["z"] + k * archive["bayes_residual"]
    archive["candidate_pick_home"] = archive["candidate_score"] >= 0.0
    archive["candidate_correct"] = (
        archive["candidate_pick_home"].to_numpy(bool) == archive["home_cover"].to_numpy(bool)
    ).astype(float)

    primary = summarise_pair(archive)
    primary["per_season"] = per_season_deltas(archive)
    primary["frozen_pick_null"] = frozen_pick_null(
        archive.reset_index(drop=True), primary["accuracy_points"]
    )

    window_frame = archive.loc[
        archive["season"].between(ASSIGNED_WINDOW[0], ASSIGNED_WINDOW[1])
    ].copy()
    window = summarise_pair(window_frame)

    sweep = []
    for multiplier in SWEEP_MULTIPLIERS:
        scored = archive.copy()
        scored["candidate_pick_home"] = (
            scored["z"] + multiplier * k * scored["bayes_residual"]
        ) >= 0.0
        scored["candidate_correct"] = (
            scored["candidate_pick_home"].to_numpy(bool) == scored["home_cover"].to_numpy(bool)
        ).astype(float)
        sweep.append(
            {
                "multiplier": multiplier,
                "k": multiplier * k,
                **accuracy_metric(scored),
                "picks_changed": int(
                    (scored["candidate_pick_home"] != scored["baseline_pick_home"]).sum()
                ),
            }
        )

    stacked = walk_forward_stack(archive, scale)
    covered = stacked.loc[stacked["stack_candidate_score"].notna()].copy()
    covered["control_pick_home"] = covered["stack_control_score"] >= 0.0
    covered["stack_pick_home"] = covered["stack_candidate_score"] >= 0.0
    covered["control_correct"] = (
        covered["control_pick_home"].to_numpy(bool) == covered["home_cover"].to_numpy(bool)
    ).astype(float)
    covered["stack_correct"] = (
        covered["stack_pick_home"].to_numpy(bool) == covered["home_cover"].to_numpy(bool)
    ).astype(float)

    versus_production = covered.copy()
    versus_production["candidate_pick_home"] = versus_production["stack_pick_home"]
    versus_production["candidate_correct"] = versus_production["stack_correct"]
    stack_vs_production = summarise_pair(versus_production)

    isolation = covered.copy()
    isolation["baseline_pick_home"] = isolation["control_pick_home"]
    isolation["baseline_correct"] = isolation["control_correct"]
    isolation["candidate_pick_home"] = isolation["stack_pick_home"]
    isolation["candidate_correct"] = isolation["stack_correct"]
    stack_isolation = summarise_pair(isolation)

    control = leak_control(archive, scale, k)
    archive["posterior_sd_week_residual"] = archive["posterior_sd"] - archive.groupby("week")[
        "posterior_sd"
    ].transform("mean")
    uncertainty = tercile_screen(archive, "posterior_sd")
    uncertainty_residualised = tercile_screen(archive, "posterior_sd_week_residual")
    reliability_result = reliability(weekly_panel)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration": "docs/bayesian_team_model_on_production.md",
        "incumbent_archive": str(OPENER_ARCHIVE.relative_to(REPO)).replace("\\", "/"),
        "incumbent_model": json.loads(OPENER_METADATA.read_text())["active_model_config"],
        "hyperparameters": params.as_dict(),
        "hyperparameter_fit": fit_diagnostics,
        "stacking_weight": weight,
        "population": {
            "archive_games": 1537,
            "missing_posterior": missing,
            "pushes_excluded": pushes,
            "scored_games": len(archive),
            "seasons": [int(archive["season"].min()), int(archive["season"].max())],
            "boundary_reconstruction_mismatches": boundary_mismatch,
            "saved_correctness_mismatches": saved_mismatch,
        },
        "screen_a_primary_full_archive": primary,
        "screen_a_assigned_window_2020_2021": window,
        "screen_a_weight_sweep_diagnostic": sweep,
        "screen_a_walk_forward_stack_vs_production": stack_vs_production,
        "screen_a_walk_forward_single_variable_isolation": stack_isolation,
        "screen_a_leak_positive_control": control,
        "screen_b_posterior_sd_uncertainty": uncertainty,
        "screen_b_posterior_sd_week_residualised": uncertainty_residualised,
        "reliability": reliability_result,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    archive.to_parquet(out_dir / "per_game.parquet", index=False)
    weekly_panel.to_parquet(out_dir / "weekly_team_panel.parquet", index=False)
    pd.DataFrame(primary["per_season"]).to_csv(out_dir / "per_season.csv", index=False)
    pd.DataFrame(uncertainty["terciles"]).to_csv(out_dir / "posterior_sd_terciles.csv", index=False)
    pd.DataFrame(uncertainty_residualised["terciles"]).to_csv(
        out_dir / "posterior_sd_week_residual_terciles.csv", index=False
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

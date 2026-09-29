from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

SOURCE = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
LANE = Path("docs/lanes/lead78.md")
REPORT = Path("docs/lead78_unit2.md")
OUTPUT = Path("tests/scratch/codex/lead78_unit2")
SEASONS = tuple(range(2020, 2026))
TERM = "common_opponent_news"
METRICS = ("accuracy_points", "log_loss", "brier", "margin_mae")
ARMS = ("fifth_term", "four_term", "model", "market", "elo")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    declaration = LANE.read_text(encoding="utf-8-sig")
    require(
        all(
            value in declaration
            for value in (
                "Unit-two amendment",
                "OPENER",
                "kickoff + 4 hours",
                "look budget is unchanged at 298",
                "seed 78",
            )
        ),
        "Save the unit-two amendment before outcomes",
    )
    frame = pq.read_table(SOURCE, use_threads=False).to_pandas(use_threads=False)
    require(len(frame) == 1503 and frame.game_id.is_unique, "Frozen population changed")
    require(tuple(sorted(frame.season.unique())) == SEASONS, "Wrong holdout seasons")
    require(frame.game_type.eq("REG").all(), "Non-regular-season target")
    require(
        frame.base_probability_policy.eq("discrete_conditional_non_push_v1").all(),
        "Served base must use the discrete conditional probability",
    )
    require(
        frame.margin_vs_open.ne(0).all()
        and frame.home_covered.eq(frame.margin_vs_open.gt(0)).all(),
        "Non-push opener target mismatch",
    )
    fields = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    games = pq.read_table(SCHEDULE, columns=fields, use_threads=False).to_pandas(use_threads=False)
    games = games.loc[games.season.between(2019, 2025) & games.game_type.eq("REG")].copy()
    require(games.game_id.is_unique, "Schedule game IDs must be unique")
    local = pd.to_datetime(
        games.gameday.astype(str) + " " + games.gametime.astype(str), errors="coerce"
    )
    games["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    require(games.kickoff.notna().all(), "A scheduled kickoff is missing")
    games["completion"] = games.kickoff + pd.Timedelta(hours=4)
    games["weekday"] = local.dt.dayofweek
    games["margin"] = games.home_score - games.away_score
    games["home_team"] = games.home_team.replace({"OAK": "LV"})
    games["away_team"] = games.away_team.replace({"OAK": "LV"})
    anchor = local.dt.normalize().groupby([games.season, games.week]).transform("min")
    sunday = anchor + pd.to_timedelta((6 - anchor.dt.dayofweek) % 7, unit="D")
    games["freeze"] = (
        (sunday + pd.Timedelta(days=-5, hours=12))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    games["sunday_deadline"] = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    games["deadline"] = games[["kickoff", "sunday_deadline"]].min(axis=1)
    games = games.loc[np.isfinite(games.margin)].copy()
    require(
        (
            games.loc[games.weekday.isin((3, 5)), "completion"]
            < games.loc[games.weekday.isin((3, 5)), "sunday_deadline"]
        ).all(),
        "Thursday/Saturday completion must precede Sunday deadline",
    )
    frame = frame.merge(
        games[["game_id", "kickoff", "completion", "freeze", "deadline", "margin"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    require(frame.kickoff.notna().all(), "Target absent from schedule")
    require(
        np.allclose(frame.margin_vs_open, frame.margin - frame.tue_open_home_spread),
        "Schedule and frozen opener grades disagree",
    )
    require((frame.freeze < frame.deadline).all(), "Target deadline precedes Tuesday freeze")
    required = [*FIT_FEATURES, "model_probability", "out_of_season_home_probability"]
    require(np.isfinite(frame[required].to_numpy(dtype=float)).all(), "Invalid base inputs")
    require(
        np.allclose(
            frame.model_logit, np.log(frame.model_probability / (1 - frame.model_probability))
        ),
        "Base model logit mismatch",
    )
    return frame.reset_index(drop=True), games


def graph_features(frame: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    teams = sorted(set(games.home_team) | set(games.away_team))
    positions = {team: index for index, team in enumerate(teams)}
    identity = np.eye(len(teams))

    def design(rows: pd.DataFrame) -> np.ndarray:
        x = np.zeros((len(rows), len(teams)))
        x[np.arange(len(rows)), rows.home_team.map(positions).to_numpy()] = 1
        x[np.arange(len(rows)), rows.away_team.map(positions).to_numpy()] = -1
        return x

    frame[TERM] = 0.0
    frame["news_games"] = 0
    frame["latest_news_completion"] = pd.Series(
        pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
    )
    audit = []
    for season in SEASONS:
        previous = games.loc[games.season.eq(season - 1)]
        current = games.loc[games.season.eq(season)]
        require(len(previous) > 0, "Earlier-season rating prior unavailable")
        hfa = float(previous.margin.mean())
        px = design(previous)
        prior = np.linalg.solve(identity + px.T @ px, px.T @ (previous.margin.to_numpy() - hfa))
        for week, targets in frame.loc[frame.season.eq(season)].groupby("week", sort=True):
            freeze = targets.freeze.iloc[0]
            require(targets.freeze.eq(freeze).all(), "Inconsistent Tuesday anchor")
            require((previous.completion < freeze).all(), "Prior contains future results")
            observed = current.loc[current.completion.lt(freeze)]
            require((observed.completion < freeze).all(), "Tuesday state contains future results")
            x = design(observed)
            matrix = identity + x.T @ x
            rhs = prior + x.T @ (observed.margin.to_numpy() - hfa)
            tuesday = np.linalg.solve(matrix, rhs)
            early = current.loc[
                current.week.eq(week)
                & current.weekday.isin((3, 5))
                & current.kickoff.gt(freeze)
                & current.completion.gt(freeze)
            ]
            updates = {}
            for target in targets.itertuples():
                news = early.loc[early.completion.lt(target.deadline)]
                require(
                    target.game_id not in set(observed.game_id), "Target score in Tuesday state"
                )
                require(target.game_id not in set(news.game_id), "Target score in news state")
                require(
                    ((news.completion > freeze) & (news.completion < target.deadline)).all(),
                    "News result misses the target deadline",
                )
                require((news.completion < target.kickoff).all(), "News is post-kickoff")
                key = tuple(news.game_id)
                if key not in updates:
                    nx = design(news)
                    updated = np.linalg.solve(
                        matrix + nx.T @ nx,
                        rhs + nx.T @ (news.margin.to_numpy() - hfa),
                    )
                    updates[key] = updated - tuesday
                delta = updates[key]
                frame.loc[target.Index, TERM] = (
                    delta[positions[target.home_team]] - delta[positions[target.away_team]]
                )
                frame.loc[target.Index, "news_games"] = len(news)
                if len(news):
                    frame.loc[target.Index, "latest_news_completion"] = news.completion.max()
                for source in news.itertuples():
                    audit.append(
                        {
                            "target_game_id": target.game_id,
                            "source_game_id": source.game_id,
                            "source_kickoff": source.kickoff,
                            "source_completion_proxy": source.completion,
                            "tuesday_freeze": freeze,
                            "target_deadline": target.deadline,
                            "target_kickoff": target.kickoff,
                            "tuesday_games": len(observed),
                            "latest_tuesday_completion": observed.completion.max(),
                            "prior_last_completion": previous.completion.max(),
                        }
                    )
    require(np.isfinite(frame[TERM]).all(), "Nonfinite graph update")
    require(frame.loc[frame.news_games.eq(0), TERM].eq(0).all(), "No-news state changed")
    return frame, audit


def elo_features(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    ratings = {}
    gap = {}
    for season, season_games in games.groupby("season", sort=True):
        ratings = {team: 0.75 * value for team, value in ratings.items()}
        pending = []
        kickoffs = dict(iter(season_games.groupby("kickoff")))
        targets = frame.loc[frame.season.eq(season)]
        deadlines = dict(iter(targets.groupby("deadline")))
        for instant in sorted(set(kickoffs) | set(deadlines)):
            completed = [row for row in pending if row[0] < instant]
            for completion, home, away, update in completed:
                require(completion < instant, "Elo result is not available yet")
                ratings[home] = ratings.get(home, 0.0) + update
                ratings[away] = ratings.get(away, 0.0) - update
            pending = [row for row in pending if row[0] >= instant]
            if instant in deadlines:
                for target in deadlines[instant].itertuples():
                    gap[target.game_id] = ratings.get(target.home_team, 0.0) - ratings.get(
                        target.away_team, 0.0
                    )
            if instant in kickoffs:
                for game in kickoffs[instant].itertuples():
                    difference = ratings.get(game.home_team, 0.0) - ratings.get(game.away_team, 0.0)
                    expected = 1 / (1 + 10 ** (-difference / 400))
                    result = float(game.margin > 0) + 0.5 * float(game.margin == 0)
                    pending.append(
                        (game.completion, game.home_team, game.away_team, 20 * (result - expected))
                    )
        for _, home, away, update in pending:
            ratings[home] = ratings.get(home, 0.0) + update
            ratings[away] = ratings.get(away, 0.0) - update
    frame["elo_gap"] = frame.game_id.map(gap)
    require(np.isfinite(frame.elo_gap).all(), "Missing pre-deadline Elo rating")
    return frame


def fit(train: pd.DataFrame, test: pd.DataFrame, terms: list[str]) -> tuple:
    means = train[terms].mean().to_numpy()
    scales = train[terms].std(ddof=0).replace(0, 1).to_numpy()
    tx = np.column_stack((np.ones(len(train)), (train[terms].to_numpy() - means) / scales))
    vx = np.column_stack((np.ones(len(test)), (test[terms].to_numpy() - means) / scales))
    beta = _fit_logit(tx, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    tp = 1 / (1 + np.exp(-np.clip(tx @ beta, -35, 35)))
    vp = 1 / (1 + np.exp(-np.clip(vx @ beta, -35, 35)))
    transform = np.eye(len(terms) + 1)
    transform[0, 1:] = -means / scales
    transform[np.arange(1, len(terms) + 1), np.arange(1, len(terms) + 1)] = 1 / scales
    natural = transform @ beta
    hessian = (tx.T * np.clip(tp * (1 - tp), 1e-6, None)) @ tx + FIT_RIDGE * np.eye(len(beta))
    covariance = transform @ np.linalg.inv(hessian) @ transform.T
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return tp, vp, natural, se


def scores(frame: pd.DataFrame, probability: np.ndarray, margin: np.ndarray) -> np.ndarray:
    y = frame.home_covered.to_numpy(dtype=float)
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.column_stack(
        (
            100 * ((probability >= 0.5) == y),
            -(y * np.log(p) + (1 - y) * np.log(1 - p)),
            (probability - y) ** 2,
            np.abs(margin - frame.margin.to_numpy()),
        )
    )


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def evaluate(frame: pd.DataFrame) -> tuple[dict, list[dict], list[dict], list[list]]:
    result = {
        arm: {
            "p": np.full(len(frame), np.nan),
            "scores": np.zeros((len(frame), 4)),
            "train": [],
            "band": np.zeros(len(frame), dtype=int),
        }
        for arm in ARMS
    }
    coefficients = []
    folds = []
    for season in SEASONS:
        train = frame.loc[frame.season.ne(season)]
        test = frame.loc[frame.season.eq(season)]
        for arm in ARMS:
            terms = {
                "fifth_term": [*FIT_FEATURES, TERM],
                "four_term": list(FIT_FEATURES),
                "elo": ["elo_gap", "tue_open_home_spread"],
            }.get(arm)
            if terms:
                tp, vp, beta, se = fit(train, test, terms)
                for name, value, error in zip(["intercept", *terms], beta, se, strict=True):
                    coefficients.append(
                        {
                            "arm": arm,
                            "held_out": season,
                            "term": name,
                            "coefficient": float(value),
                            "low": float(value - 1.96 * error),
                            "high": float(value + 1.96 * error),
                        }
                    )
            elif arm == "model":
                tp = train.model_probability.to_numpy()
                vp = test.model_probability.to_numpy()
            else:
                tp = np.full(len(train), 0.5)
                vp = np.full(len(test), 0.5)
            if arm == "four_term":
                require(
                    np.max(np.abs(vp - test.out_of_season_home_probability.to_numpy())) < 1e-9,
                    "Four-term LOSO refit failed frozen-probability reproduction",
                )
            margin_tx = np.column_stack((np.ones(len(train)), logit(tp)))
            margin_vx = np.column_stack((np.ones(len(test)), logit(vp)))
            margin_beta = np.linalg.lstsq(margin_tx, train.margin.to_numpy(), rcond=None)[0]
            ts = scores(train, tp, margin_tx @ margin_beta)
            vs = scores(test, vp, margin_vx @ margin_beta)
            result[arm]["p"][test.index] = vp
            result[arm]["scores"][test.index] = vs
            result[arm]["train"].append(ts)
            edges = np.quantile(tp, [0.2, 0.4, 0.6, 0.8])
            result[arm]["band"][test.index] = np.searchsorted(edges, vp, side="right")
            folds.append(
                {
                    "arm": arm,
                    "season": season,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train": ts.mean(0).tolist(),
                    "test": vs.mean(0).tolist(),
                    "gap": (vs.mean(0) - ts.mean(0)).tolist(),
                }
            )
    reliability = []
    for arm in ARMS:
        require(np.isfinite(result[arm]["p"]).all(), "Missing LOSO prediction")
        result[arm]["train"] = np.concatenate(result[arm]["train"]).mean(0)
        for band in range(5):
            mask = result[arm]["band"] == band
            reliability.append(
                [
                    arm,
                    band + 1,
                    int(mask.sum()),
                    f"{result[arm]['p'][mask].mean():.6f}" if mask.any() else "unavailable",
                    f"{frame.loc[mask, 'home_covered'].mean():.6f}"
                    if mask.any()
                    else "unavailable",
                ]
            )
    return result, coefficients, folds, reliability


class Bootstrap:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.index = pd.MultiIndex.from_frame(blocks)
        self.weights = np.zeros((10000, len(blocks)))
        rng = np.random.default_rng(78)
        for season in SEASONS:
            positions = np.flatnonzero(blocks.season.eq(season))
            self.weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), size=10000
            )

    def draws(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        select = np.ones(len(self.frame), dtype=bool) if mask is None else mask
        work = self.frame.loc[select, ["season", "week"]].copy()
        work["value"] = values[select]
        grouped = (
            work.groupby(["season", "week"])
            .value.agg(["sum", "count"])
            .reindex(self.index, fill_value=0)
        )
        denominator = self.weights @ grouped["count"].to_numpy(dtype=float)
        return np.divide(
            self.weights @ grouped["sum"].to_numpy(dtype=float),
            denominator,
            out=np.full(10000, np.nan),
            where=denominator > 0,
        )


def interval(draws: np.ndarray) -> list[float]:
    return np.nanquantile(draws, [0.025, 0.975]).tolist()


def bounds(values: list[float]) -> str:
    return f"[{values[0]:.6f}, {values[1]:.6f}]"


def prepare_records(summary: dict) -> None:
    statements = {
        "accuracy_points": "Using early results to revisit common opponents lost five extra picks.",
        "log_loss": "Using early results made the pick chances slightly less accurate.",
        "brier": "Using early results made the pick chances slightly less accurate.",
        "margin_mae": "Using early results did not improve predicted winning margins.",
    }
    units = {
        "accuracy_points": "accuracy_points",
        "log_loss": "log_loss_improvement",
        "brier": "brier_improvement",
        "margin_mae": "mae_improvement",
    }
    payload = {
        "source": "docs/lead78_unit2.md",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "family": "lead78_within_week_common_opponent",
        "category": "onfield",
        "classification_evidence": (
            "Decisive 4-9; accuracy interval spans favorable and unfavorable effects; "
            "news coefficient positive in four of six folds. Proper-score deterioration "
            "does not by itself refute the common-opponent mechanism; no power control."
        ),
        "notes": (
            "298 predeclared looks; 10000 season-stratified week-block draws, seed 78; "
            "retrospective LOSO, historical opener and kickoff-plus-four-hours proxies. "
            "Primary comparison only; baseline diagnostics are not separate signals."
        ),
        "cells": [],
    }
    for effect in summary["effects"]:
        if effect["baseline"] != "four_term":
            continue
        metric = effect["metric"]
        payload["cells"].append(
            {
                "name": f"lead78_unit2_common_opponent_{metric}",
                "description": f"Common-opponent fifth term versus served four-term: {metric}",
                "effect": effect["effect"],
                "effect_units": units[metric],
                "standard_error": effect["standard_error"],
                "interval_low": effect["interval"][0],
                "interval_high": effect["interval"][1],
                "probability_positive": effect["probability_positive"],
                "sample_games": summary["games"],
                "sample_blocks": summary["blocks"],
                "plain_summary": statements[metric]
                + " Keep the current picks; the idea remains unresolved.",
            }
        )
    (OUTPUT / "registry_batch.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def run() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, games = read_inputs()
    frame, audit = graph_features(frame, games)
    frame = elo_features(frame, games)
    result, coefficients, folds, reliability = evaluate(frame)
    boot = Bootstrap(frame)
    candidate = result["fifth_term"]
    differences = (candidate["p"] >= 0.5) != (result["four_term"]["p"] >= 0.5)
    decisive_wins = int(
        ((candidate["p"] >= 0.5) == frame.home_covered.to_numpy())[differences].sum()
    )
    decisive_n = int(differences.sum())
    effects = []
    metrics_rows = []
    effects_rows = []
    for arm in ARMS:
        for column, metric in enumerate(METRICS):
            values = result[arm]["scores"][:, column]
            draws = boot.draws(values)
            metrics_rows.append(
                [
                    arm,
                    metric,
                    f"{result[arm]['train'][column]:.6f}",
                    f"{values.mean():.6f}",
                    bounds(interval(draws)),
                    f"{values.mean() - result[arm]['train'][column]:.6f}",
                ]
            )
            if arm == "fifth_term":
                continue
            sign = 1 if metric == "accuracy_points" else -1
            delta = sign * (candidate["scores"][:, column] - values)
            sampled = boot.draws(delta)
            estimate = {
                "baseline": arm,
                "metric": metric,
                "effect": float(delta.mean()),
                "interval": interval(sampled),
                "standard_error": float(sampled.std(ddof=1)),
                "probability_positive": float(np.mean(sampled > 0) + 0.5 * np.mean(sampled == 0)),
            }
            effects.append(estimate)
            effects_rows.append(
                [
                    arm,
                    metric,
                    f"{estimate['effect']:.6f}",
                    bounds(estimate["interval"]),
                    f"{estimate['probability_positive']:.6f}",
                ]
            )
    season_rows = []
    for fold in folds:
        for index, metric in enumerate(METRICS):
            arm = fold["arm"]
            mask = frame.season.eq(fold["season"]).to_numpy()
            season_rows.append(
                [
                    arm,
                    fold["season"],
                    f"{fold['train_n']}/{fold['test_n']}",
                    metric,
                    f"{fold['train'][index]:.6f}",
                    f"{fold['test'][index]:.6f}",
                    bounds(interval(boot.draws(result[arm]["scores"][:, index], mask))),
                    f"{fold['gap'][index]:.6f}",
                ]
            )
    primary = [effect for effect in effects if effect["baseline"] == "four_term"]
    coefficient_rows = [
        [
            row["arm"],
            row["held_out"],
            row["term"],
            f"{row['coefficient']:.6f}",
            bounds([row["low"], row["high"]]),
        ]
        for row in coefficients
    ]
    news_coefficients = [row["coefficient"] for row in coefficients if row["term"] == TERM]
    predictions = frame[
        [
            "game_id",
            "season",
            "week",
            "tue_open_home_spread",
            "margin",
            "home_covered",
            "kickoff",
            "freeze",
            "deadline",
            TERM,
            "news_games",
            "latest_news_completion",
        ]
    ].copy()
    for arm in ARMS:
        predictions[f"{arm}_oos_probability"] = result[arm]["p"]
        for index, metric in enumerate(METRICS):
            predictions[f"{arm}_{metric}"] = result[arm]["scores"][:, index]
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(audit).to_csv(OUTPUT / "timing_audit.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUTPUT / "coefficients.csv", index=False)
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, SCHEDULE)
    }
    summary = {
        "games": len(frame),
        "blocks": len(boot.index),
        "look_budget": 298,
        "decisive_wins": decisive_wins,
        "decisive_losses": decisive_n - decisive_wins,
        "decisive_pushes": 0,
        "games_with_news": int(frame.news_games.gt(0).sum()),
        "nonzero_terms": int(frame[TERM].ne(0).sum()),
        "timing_asserted_pairs": len(audit),
        "news_coefficient_positive_folds": int(sum(value > 0 for value in news_coefficients)),
        "base_reproduction_max_error": float(
            np.max(
                np.abs(result["four_term"]["p"] - frame.out_of_season_home_probability.to_numpy())
            )
        ),
        "effects": effects,
        "folds": folds,
        "source_hashes": source_hashes,
        "classification": "unresolved_below_power",
        "registry_status": "pending_orchestrator",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    prepare_records(summary)

    text = [
        "# LEAD-78 unit 2: within-week common-opponent news",
        "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead78_unit2.py`.",
        "## Declaration and timing",
        "**Read:** the unit-two amendment in `docs/lanes/lead78.md` was saved before "
        "outcome computation. Historical openers replace unavailable pool captures; "
        "scheduled kickoff + four hours replaces unavailable published-final timestamps. "
        "The original score-state delta remains one fitted fifth term. B=3, F=6, E=0; the "
        "look budget is unchanged at 298, including baseline/reporting cells. No searched "
        "cutoffs, selected surprise games, or extra challenger specifications.",
        f"**Measured:** {len(frame):,} unique non-push REG games, 2020-2025; "
        f"{len(boot.index)} season-week blocks. {summary['games_with_news']} targets have "
        f"eligible Thursday/Saturday news; {summary['nonzero_terms']} nonzero state deltas. "
        f"All {len(audit)} source-target pairs passed completion-after-Tuesday and "
        f"completion-before-deadline/kickoff assertions. No-news games remain with zero "
        f"delta.",
        "**Read:** completion is a proxy, not an observed publication time. Tuesday freeze "
        "is noon Eastern; each deadline is the earlier of kickoff and Sunday 12:45 Eastern. "
        "Prior-season scores establish a unit-precision ridge rating prior and home "
        "advantage; only scores completed before Tuesday enter the initial current-season "
        "graph. Eligible early results update that graph, and the home-minus-away rating "
        "change is the term. All four existing coefficients and the fifth coefficient are "
        "fitted together; no independent rule flips picks.",
        f"**Measured:** refitted four-term held-out probabilities reproduce the frozen "
        f"served base to maximum error {summary['base_reproduction_max_error']:.3g}.",
        "**Inferred:** six-fold retrospective LOSO trains on the other five seasons, "
        "including later seasons for earlier holdouts; this is not prospective forward "
        "validation. Cached discrete-model and situational inputs inherit their existing "
        "provenance. The news feature itself uses only earlier completed scores; no "
        "full-history pipeline was rebuilt. Final schedule scores cannot verify absence of "
        "later corrections.",
        "## Decisive record and paired gains",
        f"**Measured:** on {decisive_n} games where candidate and four-term sides differ, "
        f"the candidate is {decisive_wins}-{decisive_n - decisive_wins}-0 and the base is "
        f"{decisive_n - decisive_wins}-{decisive_wins}-0. The population excludes opener "
        f"pushes. Historical forced-pick records do not establish a profitable edge.",
        "**Measured:** positive gains favor the candidate: accuracy is candidate minus "
        "baseline; losses and margin MAE are baseline minus candidate. 95% paired intervals "
        "use 10,000 season-stratified week-block draws, seed 78. `probability_positive` "
        "counts half of exact ties. These intervals condition on the fitted LOSO "
        "predictions.",
        table(["Baseline", "Metric", "Gain", "95% interval", "probability_positive"], effects_rows),
        "## In-sample and held-out calibration",
        "**Measured:** training means aggregate each fold's five training seasons; each "
        "game appears five times. Held-out means include each game once; gap is held-out "
        "minus training. Market probability is 0.5 with a deterministic home tie rule. Elo "
        "uses K=20, scale=400 and offseason carry=0.75 with training-fold logistic "
        "calibration of rating gap and opener. Margin MAE uses a reporting-only linear "
        "calibration of each probability's logit to home margin on the training fold; it is "
        "not a served margin distribution.",
        table(
            ["Arm", "Metric", "Training", "Held out", "Held-out 95% interval", "Gap"], metrics_rows
        ),
        "## Season stability",
        "**Measured:** season intervals resample weeks within that season.",
        table(
            [
                "Arm",
                "Held out",
                "n train/test",
                "Metric",
                "Training",
                "Held out",
                "95% interval",
                "Gap",
            ],
            season_rows,
        ),
        "## Fold coefficients",
        f"**Measured:** the news coefficient is positive in "
        f"{summary['news_coefficient_positive_folds']}/6 folds; range "
        f"[{min(news_coefficients):.6f}, {max(news_coefficients):.6f}] per margin point. "
        f"Natural-scale coefficients follow; approximate Hessian intervals describe training "
        f"fits, not six independent studies.",
        table(
            ["Arm", "Held out", "Term", "Coefficient", "Approximate 95% interval"], coefficient_rows
        ),
        "## Reliability",
        "**Measured:** five bands use each fold's training probability quintiles and pool "
        "held-out predictions. Tied market probabilities leave empty bands; no held-out "
        "cutoff selection.",
        table(
            [
                "Arm",
                "Training-quantile band",
                "Held-out n",
                "Mean probability",
                "Observed home-cover rate",
            ],
            reliability,
        ),
        "## Interpretation and saved evidence",
        "**Inferred:** provisional `unresolved_below_power`, pending the orchestrator's "
        "registry write. AGENTS.md lines 67-78 allow closure only with a refuted mechanism "
        "or demonstrated positive-control power; no such closure is established here. "
        "AGENTS.md lines 89-100 require one fitted probability and out-of-season "
        "parameters. The estimates inform research; they do not authorize serving this "
        "term. Correlated reporting cells and the look budget preclude presenting a best "
        "cell as independent evidence.",
        "**Measured:** prediction rows, timing audit, coefficient rows and structured "
        "summary are under `tests/scratch/codex/lead78_unit2/`. Exact bash record commands "
        "are in the lane and were not executed by this worker.",
        table(["Input", "SHA-256"], [[path, digest] for path, digest in source_hashes.items()]),
    ]
    REPORT.write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("effects", "folds", "source_hashes")
            },
            indent=2,
        )
    )
    print(json.dumps({"primary_effects": primary}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()

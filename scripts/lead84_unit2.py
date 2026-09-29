from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import lead84_unit1 as construction
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lead78_unit2 import elo_features
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from sunday_market_probability_eval import sunday_move
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    _arrest_incidents,
    _fit_logit,
    _forecast_temperatures,
    _protection_back_side,
    signed_composition_flags,
)
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

OUTPUT = Path("tests/scratch/codex/lead84_unit2")
LANE = Path("docs/lanes/lead84.md")
REPORT = Path("docs/lead84_unit2.md")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
ARMS = ("candidate", "served_base", "model", "market", "elo")
METRICS = ("rps", "log_loss", "brier", "accuracy_points")
TERMS = {
    "candidate": (*FIT_FEATURES, "jensen_gap"),
    "served_base": FIT_FEATURES,
    "model": ("model_logit",),
    "market": ("market_logit",),
    "elo": ("elo_gap", "tue_open_home_spread"),
}
GRID = construction.GRID
BOOTSTRAPS = 10000
pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def schedule():
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
        "home_coach",
        "away_coach",
        "result",
        "roof",
        "temp",
        "spread_line",
    ]
    games = read(SCHEDULE, fields)
    local = pd.to_datetime(
        games.gameday.astype(str) + " " + games.gametime.astype(str), errors="coerce"
    )
    games["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    games["completion"] = games.kickoff + pd.Timedelta(hours=4)
    games["margin"] = games.home_score - games.away_score
    return games


def recover_moves(games, schedules):
    move_checkpoint = OUTPUT / "move_checkpoint.json"
    if move_checkpoint.exists():
        saved = json.loads(move_checkpoint.read_text(encoding="utf-8"))
        for path, expected in saved["hashes"].items():
            require(digest(path) == expected, f"Movement checkpoint changed: {path}")
        sources = json.loads((OUTPUT / "move_sources.json").read_text(encoding="utf-8"))
        for source in sources:
            require(digest(source["path"]) == source["sha256"], "Movement source changed")
            require(
                digest(Path(source["path"]).with_name("manifest.json"))
                == source["manifest_sha256"],
                "Movement manifest changed",
            )
        return read(OUTPUT / "recovered_moves.parquet"), {
            name: saved[name] for name in ("move_quote_rows", "move_source_files")
        }
    targets = games[["game_id", "season", "week", "cutoff", "kickoff"]].merge(
        schedules[["game_id", "home_team", "away_team"]], on="game_id", validate="one_to_one"
    )
    local = targets.cutoff.dt.tz_convert("America/New_York").dt.tz_localize(None)
    targets["monday"] = (
        (local.dt.normalize() - pd.to_timedelta(local.dt.dayofweek, unit="D"))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    fields = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "line",
        "outcome_side",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
        "home_team",
        "away_team",
    ]
    evidence, sources = [], []
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if (
            manifest.get("provider") != "the-odds-api"
            or manifest.get("capture_kind") != "historical_backfill"
            or request.get("sport") != "americanfootball_nfl"
            or request.get("season") not in range(2020, 2026)
            or "spreads" not in str(request.get("markets", "")).split(",")
        ):
            continue
        selected = targets.loc[
            targets.season.eq(request["season"]) & targets.week.eq(request.get("week"))
        ]
        stamp = pd.to_datetime(manifest.get("snapshot_timestamp_utc"), utc=True)
        if selected.empty or pd.isna(stamp):
            continue
        if not ((selected.monday <= stamp) & (stamp < selected.cutoff)).any():
            continue
        source = path.with_name("quotes.parquet")
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        require(expected and digest(source) == expected, f"Move source hash: {source}")
        quotes = read(source, fields)
        quotes = quotes.loc[quotes.market.eq("spreads") & quotes.bookmaker_key.isin(LEADER_BOOKS)]
        quotes = quotes.merge(
            selected,
            left_on="nflverse_game_id",
            right_on="game_id",
            suffixes=("", "_scheduled"),
            validate="many_to_one",
        )
        for name in (
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ):
            quotes[name] = pd.to_datetime(quotes[name], utc=True, errors="coerce", format="mixed")
        valid = (
            quotes.observed_at_utc.eq(stamp)
            & quotes.observed_at_utc.ge(quotes.monday)
            & quotes.observed_at_utc.lt(quotes.cutoff)
            & quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc)
            & quotes.market_last_update_utc.le(quotes.observed_at_utc)
            & quotes.commence_time_utc.gt(quotes.cutoff)
            & quotes.home_team.eq(quotes.home_team_scheduled)
            & quotes.away_team.eq(quotes.away_team_scheduled)
            & np.isfinite(pd.to_numeric(quotes.home_spread_line, errors="coerce"))
        )
        quotes = quotes.loc[valid].copy()
        signed = np.where(quotes.outcome_side.eq("HOME"), -quotes.line, quotes.line)
        require(
            quotes.outcome_side.isin(("HOME", "AWAY")).all()
            and np.allclose(signed, quotes.home_spread_line),
            "Move source handicap mismatch",
        )
        evidence.append(quotes)
        sources.append({"path": str(source), "sha256": expected, "manifest_sha256": digest(path)})
    require(bool(evidence), "No timestamp-matched movement quotes")
    quotes = pd.concat(evidence, ignore_index=True)
    bounds = targets[["game_id", "kickoff", "monday"]].rename(
        columns={"kickoff": "commence_time_utc", "monday": "week_first_commence_utc"}
    )
    moves = sunday_move(quotes, bounds)
    quotes.to_parquet(OUTPUT / "move_quote_evidence.parquet", index=False)
    moves.to_parquet(OUTPUT / "recovered_moves.parquet", index=False)
    write_json(OUTPUT / "move_sources.json", sources)
    return moves, {"move_quote_rows": len(quotes), "move_source_files": len(sources)}


def inputs():
    unit_one = Path("tests/scratch/codex/lead84_unit1")
    manifest = json.loads((unit_one / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    counts = manifest["inventory"].copy()
    for source in sources:
        require(digest(source["path"]) == source["sha256"], "Book source changed")
        require(
            digest(Path(source["path"]).with_name("manifest.json")) == source["manifest_sha256"],
            "Book manifest changed",
        )
    original = read(unit_one / "book_panels.parquet")
    line, win, cover = original.quote_line, original.quote_win, original.quote_cover
    incompatible = (
        ((line > 0) & (cover > win + 1e-10))
        | ((line < 0) & (cover < win - 1e-10))
        | (line.eq(0) & (abs(cover - win) > 1e-10))
    )
    excluded = original.loc[incompatible].copy()
    require(
        len(excluded) == 17 and excluded.game_id.nunique() == 9,
        "Unit-one incompatible-panel inventory changed",
    )
    excluded.to_parquet(OUTPUT / "excluded_book_panels.parquet", index=False)
    panels = original.loc[~incompatible].copy()
    panels = panels.loc[panels.groupby("game_id").book.transform("size").ge(2)].copy()
    panels.to_parquet(OUTPUT / "book_panels.parquet", index=False)
    games = construction.load_population(panels)
    opener_ids = set(read(construction.OPENER, ["game_id"]).game_id)
    coverage_exclusions = len(set(original.game_id) & opener_ids) - len(games)
    incomplete = games.game_id.eq("2020_11_DET_CAR")
    require(incomplete.sum() == 1, "Declared missing-move game changed")
    games = games.loc[~incomplete].copy()
    panels = panels.loc[panels.game_id.isin(games.game_id)].copy()
    schedules = schedule()
    moves, move_inventory = recover_moves(games, schedules)
    games = games.merge(moves, on="game_id", how="left", validate="one_to_one")
    require(
        games.sunday_move.notna().all(),
        "A target lacks a reconstructed move; do not substitute missing-archive zeros",
    )
    base = read(construction.BASE, ["game_id", *FIT_FEATURES, "model_probability"])
    games = games.merge(base, on="game_id", how="left", validate="one_to_one")
    games[MOVE_COLUMN] = games.sunday_move
    games[MOVE_AVAILABLE_COLUMN] = 1.0
    games = games.merge(
        schedules[["game_id", "game_type", "gameday", "home_team", "away_team"]],
        on="game_id",
        validate="one_to_one",
    )
    opener = read(
        construction.OPENER,
        ["game_id", "home_cover_probability_at_open", "base_probability_policy"],
    )
    games = games.merge(opener, on="game_id", validate="one_to_one")
    require(
        games.base_probability_policy.eq("discrete_conditional_non_push_v1").all(),
        "Upstream input is not the served discrete read",
    )
    pushed = games.result.eq(games.tue_open_home_spread)
    require(games.loc[~pushed, FLAG_SUM_COLUMN].notna().all(), "Missing nonpush base inputs")
    require(
        np.allclose(
            games.loc[~pushed, "model_probability"],
            games.loc[~pushed, "home_cover_probability_at_open"],
        ),
        "Upstream model probabilities disagree",
    )
    games.loc[pushed, "model_probability"] = games.loc[pushed, "home_cover_probability_at_open"]
    games["model_logit"] = logit(games.model_probability.clip(1e-6, 1 - 1e-6))
    if pushed.any():
        push_flags = signed_composition_flags(
            games.loc[pushed],
            schedules,
            incidents=_arrest_incidents(Path("data")),
            forecasts_tuesday_noon=_forecast_temperatures(Path("data")),
            protection_back_side=_protection_back_side(schedules, Path("data")),
        ).set_index("game_id")
        games.loc[pushed, FLAG_SUM_COLUMN] = (
            games.loc[pushed, "game_id"].map(push_flags[FLAG_SUM_COLUMN]).to_numpy()
        )
    games["home_covered"] = games.result.gt(games.tue_open_home_spread).astype(float)
    games["nonpush"] = ~pushed
    games["deadline"] = games.cutoff
    elo_schedule = schedules.loc[
        schedules.season.between(2019, 2025) & schedules.game_type.eq("REG")
    ]
    require(elo_schedule.kickoff.notna().all(), "Missing Elo event clock")
    games = elo_features(games, elo_schedule)
    require(
        np.isfinite(games[[*FIT_FEATURES, "elo_gap"]].to_numpy(float)).all(),
        "Nonfinite combination features",
    )
    inventory = {
        **counts,
        **move_inventory,
        "excluded_panels": len(excluded),
        "affected_games": int(excluded.game_id.nunique()),
        "retained_panels": len(panels),
        "retained_games": len(games),
        "nonpush_games": int(games.nonpush.sum()),
        "pushes": int(pushed.sum()),
        "excluded_games": coverage_exclusions,
        "missing_move_games": 1,
        "coverage": games.groupby("season")
        .agg(games=("game_id", "size"), nonpush=("nonpush", "sum"))
        .reset_index()
        .to_dict("records"),
    }
    write_json(OUTPUT / "panel_sources.json", sources)
    return games, panels, inventory


def fit_logit(frame, terms):
    means = frame[list(terms)].mean().to_numpy(float)
    scales = frame[list(terms)].std(ddof=0).replace(0, 1).to_numpy(float)
    design = np.column_stack(
        (np.ones(len(frame)), (frame[list(terms)].to_numpy(float) - means) / scales)
    )
    beta = _fit_logit(design, frame.home_covered.to_numpy(float), FIT_RIDGE)
    require(np.isfinite(beta).all(), "Invalid fitted coefficient")
    return beta, means, scales


def linear(frame, terms, fit):
    beta, means, scales = fit
    return beta[0] + ((frame[list(terms)].to_numpy(float) - means) / scales) @ beta[1:]


def calibrated_mass(mass, lines, probability):
    home = GRID[None, :] > lines[:, None]
    away = GRID[None, :] < lines[:, None]
    h, a = (mass * home).sum(axis=1), (mass * away).sum(axis=1)
    require((h > 0).all() and (a > 0).all(), "Cannot calibrate an empty nonpush tail")
    result = mass * np.where(
        home,
        ((h + a) * probability / h)[:, None],
        np.where(away, ((h + a) * (1 - probability) / a)[:, None], 1.0),
    )
    require(np.allclose(result.sum(axis=1), 1, atol=1e-12), "Calibration lost PMF mass")
    require(
        np.allclose((result * home).sum(axis=1) / (h + a), probability, atol=1e-12),
        "Calibrated probability does not come from PMF",
    )
    return result


def replay(games, features, mixture, averaged):
    frame = features.merge(
        games.drop(columns=["cutoff", "snapshot", "books", "season", "week"]),
        on="game_id",
        validate="many_to_one",
    )
    frame["market_logit"] = logit(frame.averaged_probability.clip(1e-6, 1 - 1e-6))
    records, coefficients, fold_parameters, pmfs = [], [], [], []
    array_offset = 0
    for outer in construction.FOLDS:
        fold = frame.loc[frame.outer.eq(outer)].copy()
        train = fold.loc[fold.stage.eq("fit_IS") & fold.nonpush]
        tune = fold.loc[fold.stage.eq("tune") & fold.nonpush]
        calibrate = fold.loc[fold.stage.eq("calibrate") & fold.nonpush]
        require(
            train.season.max() <= outer - 3
            and tune.season.eq(outer - 2).all()
            and calibrate.season.eq(outer - 1).all(),
            "Overlapping chronological stages",
        )
        for before, after in (
            (train, tune),
            (tune, calibrate),
            (calibrate, fold.loc[fold.stage.eq("outer_OOS")]),
        ):
            require(
                before.kickoff.max() + pd.Timedelta(days=2) < after.cutoff.min(),
                "Unresolved outcomes cross a fitting boundary",
            )
        fits = {arm: fit_logit(train, TERMS[arm]) for arm in ARMS}
        tuning_logit = linear(tune, TERMS["served_base"], fits["served_base"])
        target = tune.home_covered.to_numpy(float)
        solution = minimize_scalar(
            lambda t, z=tuning_logit, y=target: float(np.mean(np.logaddexp(0, t * z) - y * t * z)),
            bounds=(0, 2),
            method="bounded",
            options={"xatol": 1e-10},
        )
        require(solution.success, "Common multiplier tuning failed")
        multiplier = float(solution.x)
        fold_parameters.append(
            {
                "outer": outer,
                "fit_through": outer - 3,
                "tune": outer - 2,
                "calibrate": outer - 1,
                "fit_nonpush": len(train),
                "tune_nonpush": len(tune),
                "calibrate_nonpush": len(calibrate),
                "common_multiplier": multiplier,
                "ridge": FIT_RIDGE,
            }
        )
        for arm in ARMS:
            terms, fit = TERMS[arm], fits[arm]
            cal_z = multiplier * linear(calibrate, terms, fit)
            cal_y = calibrate.home_covered.to_numpy(float)
            intercept = float(
                brentq(lambda b, z=cal_z, y=cal_y: np.sum(expit(z + b) - y), -100, 100)
            )
            beta, means, scales = fit
            natural = beta[1:] / scales
            entry = {
                "outer": outer,
                "arm": arm,
                "fit_intercept": float(beta[0] - natural @ means),
                "calibration_intercept": intercept,
                "multiplier": multiplier,
                "effective_intercept": float(multiplier * (beta[0] - natural @ means) + intercept),
                "means": dict(zip(terms, means.tolist(), strict=True)),
                "scales": dict(zip(terms, scales.tolist(), strict=True)),
            }
            for name, value in zip(terms, natural, strict=True):
                entry[name] = float(value)
                entry[f"effective_{name}"] = float(multiplier * value)
            coefficients.append(entry)
            for stage, label in (("fit_IS", "IS"), ("outer_OOS", "OOS")):
                test = fold.loc[fold.stage.eq(stage)].copy()
                p = expit(np.clip(multiplier * linear(test, terms, fit) + intercept, -35, 35))
                carrier = mixture if arm == "candidate" else averaged
                mass = calibrated_mass(
                    carrier[test.array_row.to_numpy(int)], test.frozen_opener.to_numpy(float), p
                )
                y = test.home_covered.to_numpy(float)
                scored = test[
                    [
                        "outer",
                        "game_id",
                        "season",
                        "week",
                        "nonpush",
                        "result",
                        "frozen_opener",
                        "home_covered",
                    ]
                ].copy()
                scored["arm"], scored["scheme"], scored["probability"] = arm, label, p
                scored["accuracy_points"] = np.where(test.nonpush, 100 * ((p >= 0.5) == y), np.nan)
                scored["log_loss"] = np.where(
                    test.nonpush, -(y * np.log(p) + (1 - y) * np.log1p(-p)), np.nan
                )
                scored["brier"] = np.where(test.nonpush, (p - y) ** 2, np.nan)
                scored["rps"] = np.square(
                    np.cumsum(mass, axis=1)
                    - (test.result.to_numpy(float)[:, None] <= GRID[None, :])
                ).sum(axis=1)
                scored["pmf_row"] = np.arange(array_offset, array_offset + len(test))
                array_offset += len(test)
                pmfs.append(mass)
                records.append(scored)
        print(f"replay fold {outer} complete", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    np.savez_compressed(OUTPUT / "calibrated_pmfs.npz", grid=GRID, mass=np.concatenate(pmfs))
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    write_json(OUTPUT / "coefficients.json", coefficients)
    write_json(OUTPUT / "fold_parameters.json", fold_parameters)
    return predictions, coefficients, fold_parameters


class Bootstrap:
    def __init__(self, frame, seed):
        self.frame = frame
        self.blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.index = pd.MultiIndex.from_frame(self.blocks)
        self.weights = np.zeros((BOOTSTRAPS, len(self.blocks)))
        rng = np.random.default_rng(seed)
        seasons = sorted(self.blocks.season.unique())
        season_counts = rng.multinomial(
            len(seasons), np.full(len(seasons), 1 / len(seasons)), size=BOOTSTRAPS
        )
        for j, season in enumerate(seasons):
            positions = np.flatnonzero(self.blocks.season.eq(season))
            for multiplicity in range(1, len(seasons) + 1):
                selected = np.flatnonzero(season_counts[:, j] == multiplicity)
                self.weights[np.ix_(selected, positions)] = rng.multinomial(
                    len(positions) * multiplicity,
                    np.full(len(positions), 1 / len(positions)),
                    size=len(selected),
                )

    def draws(self, values):
        grouped = self.frame[["season", "week"]].copy()
        grouped["value"] = values
        sums = grouped.groupby(["season", "week"]).value.agg(["sum", "count"])
        sums = sums.reindex(self.index, fill_value=0)
        denominator = self.weights @ sums["count"].to_numpy(float)
        require((denominator > 0).all(), "Empty bootstrap population")
        return self.weights @ sums["sum"].to_numpy(float) / denominator


def estimate(effect, draws, games, blocks):
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "effect": float(effect),
        "interval_low": float(low),
        "interval_high": float(high),
        "standard_error": float(np.std(draws, ddof=1)),
        "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
        "games": int(games),
        "blocks": int(blocks),
    }


def summarize(predictions):
    cells, decisive, reliability = [], [], []
    for panel in (*construction.FOLDS, "pooled"):
        frame = predictions if panel == "pooled" else predictions.loc[predictions.outer.eq(panel)]
        effects = {}
        for scheme_index, scheme in enumerate(("IS", "OOS")):
            select = frame.loc[frame.scheme.eq(scheme)]
            aligned = {
                arm: select.loc[select.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = aligned["candidate"]
            bootstrap = Bootstrap(reference, 84 + scheme_index)
            for arm in ARMS:
                require(
                    aligned[arm][["outer", "game_id"]].equals(reference[["outer", "game_id"]]),
                    "Unpaired metric population",
                )
            for metric in METRICS:
                draws = {arm: bootstrap.draws(aligned[arm][metric].to_numpy(float)) for arm in ARMS}
                means = {arm: float(aligned[arm][metric].mean()) for arm in ARMS}
                count = int(reference[metric].notna().sum())
                blocks = len(
                    reference.loc[reference[metric].notna(), ["season", "week"]].drop_duplicates()
                )
                for arm in ARMS:
                    cells.append(
                        {
                            "panel": panel,
                            "scheme": scheme,
                            "kind": "absolute",
                            "arm": arm,
                            "metric": metric,
                            **estimate(means[arm], draws[arm], count, blocks),
                        }
                    )
                    effects[scheme, "absolute", arm, metric] = (
                        means[arm],
                        draws[arm],
                        count,
                        blocks,
                    )
                for baseline in ARMS[1:]:
                    direction = 1 if metric == "accuracy_points" else -1
                    effect = direction * (means["candidate"] - means[baseline])
                    delta = direction * (draws["candidate"] - draws[baseline])
                    cells.append(
                        {
                            "panel": panel,
                            "scheme": scheme,
                            "kind": "contrast",
                            "arm": baseline,
                            "metric": metric,
                            **estimate(effect, delta, count, blocks),
                        }
                    )
                    effects[scheme, "contrast", baseline, metric] = effect, delta, count, blocks
            if scheme == "OOS":
                for baseline in ARMS[1:]:
                    other = aligned[baseline]
                    different = reference.nonpush & (
                        (reference.probability >= 0.5) != (other.probability >= 0.5)
                    )
                    wins = int(reference.loc[different, "accuracy_points"].eq(100).sum())
                    decisive.append(
                        {
                            "panel": panel,
                            "baseline": baseline,
                            "games": int(different.sum()),
                            "candidate_wins": wins,
                            "candidate_losses": int(different.sum()) - wins,
                        }
                    )
                if panel == "pooled":
                    for arm in ARMS:
                        rows = aligned[arm].loc[aligned[arm].nonpush].copy()
                        rows["band"] = np.minimum((rows.probability * 5).astype(int), 4)
                        for band in range(5):
                            group = rows.loc[rows.band.eq(band)]
                            reliability.append(
                                {
                                    "arm": arm,
                                    "band": f"{band / 5:.1f}-{(band + 1) / 5:.1f}",
                                    "games": len(group),
                                    "mean_probability": float(group.probability.mean())
                                    if len(group)
                                    else None,
                                    "observed_home_cover": float(group.home_covered.mean())
                                    if len(group)
                                    else None,
                                }
                            )
        for kind, arms in (("absolute", ARMS), ("contrast", ARMS[1:])):
            for arm in arms:
                for metric in METRICS:
                    is_mean, is_draws, _, _ = effects["IS", kind, arm, metric]
                    oos_mean, oos_draws, n, b = effects["OOS", kind, arm, metric]
                    sign = -1 if kind == "absolute" and metric != "accuracy_points" else 1
                    cells.append(
                        {
                            "panel": panel,
                            "scheme": "IS_minus_OOS",
                            "kind": kind,
                            "arm": arm,
                            "metric": metric,
                            **estimate(
                                sign * (is_mean - oos_mean), sign * (is_draws - oos_draws), n, b
                            ),
                        }
                    )
    require(
        len(cells) == 432 and len(decisive) == 16 and len(reliability) == 25,
        "Look accounting mismatch",
    )
    return {
        "cells": cells,
        "decisive": decisive,
        "reliability": reliability,
        "looks": {
            "metric_cells": 432,
            "fit_nuisance_summaries": 32,
            "decisive": 16,
            "reliability": 25,
            "total": 505,
        },
    }


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def interval(cell):
    return f"{cell['effect']:+.6f} [{cell['interval_low']:+.6f}, {cell['interval_high']:+.6f}]"


def render(result, coefficients, parameters, inventory):
    lines = [
        "# LEAD-84 unit 2: combine book distributions before the opener",
        "",
        "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead84_unit2.py`.",
        (
            "Historical OPENER is the frozen Splash-line proxy on the 2020-2025 "
            "source-complete population; pool captures begin in 2026. The base is "
            "the served four-term recipe, refitted within each chronological "
            "fold. No target closing-line inputs, served-card changes or registry writes."
        ),
        "",
        "## Decisive record first",
        "",
        table(
            ["Outer", "Comparator", "Different picks", "Candidate W-L"],
            [
                [
                    d["panel"],
                    d["baseline"],
                    d["games"],
                    f"{d['candidate_wins']}-{d['candidate_losses']}",
                ]
                for d in result["decisive"]
            ],
        ),
        "",
        (
            "**Measured:** records exclude opener pushes. Each side follows its "
            "sole calibrated PMF probability; a tie selects home."
        ),
        "",
        "## Locked design and chronology",
        "",
        (
            "**Read:** the unit-two amendment in `docs/lanes/lead84.md` was saved "
            "before outcomes; its immutable copy and SHA-256 are in scratch. "
            "Outer 2023/2024/2025 fits through Y-3, tunes one shared logit "
            "multiplier on Y-2 base log loss, and calibrates each intercept on "
            "Y-1. Logistic ridge is 0.001 throughout. No candidate was chosen "
            "after scoring."
        ),
        (
            "**Measured:** weekly upstream model training cutoffs precede "
            "predictions by more than two days. Combined cached probabilities are "
            "never read. Base and candidate receive reconstructed "
            "timestamp-matched leader-book moves in every year. Fit, tune, "
            "calibration and outer seasons are disjoint. Elo uses 2019 warm-up, "
            "K=20, 0.75 annual carryover and a four-hour result delay."
        ),
        (
            "**Measured:** the fifth term is mixture logit minus "
            "averaged-constraint logit. Candidate probabilities reweight mixture "
            "PMF nonpush tails; other arms reweight averaged-constraint PMF "
            "tails. Push mass is preserved. RPS sums squared CDF errors on "
            "integer margins -100..100, on all games; binary endpoints omit "
            "pushes."
        ),
        (
            "**Inferred:** this RPS contrast measures both changed PMF shape and "
            "the fitted fifth term, without isolating their contributions. "
            "Baseline RPS uses the declared market lattice, not a reconstruction "
            "of the production model's full served margin distribution. Weekly "
            "upstream model archives are retrospective pregame predictions, not a "
            "new full-pipeline outer refit."
        ),
        "",
        table(
            ["Outer", "Fit n", "Tune n", "Calibrate n", "Common multiplier"],
            [
                [
                    p["outer"],
                    p["fit_nonpush"],
                    p["tune_nonpush"],
                    p["calibrate_nonpush"],
                    f"{p['common_multiplier']:.6f}",
                ]
                for p in parameters
            ],
        ),
        "",
        "## Source repair and population",
        "",
        (
            f"**Measured:** excluded {inventory['excluded_panels']} "
            f"price-incompatible book panels in {inventory['affected_games']} "
            f"games, from both constructions, without outcomes. Retained "
            f"{inventory['retained_panels']} panels and "
            f"{inventory['retained_games']} games ({inventory['nonpush_games']} "
            f"nonpush, {inventory['pushes']} pushes); "
            f"{inventory['excluded_games']} games lost two-book coverage. Both "
            f"priors and all distributions were rebuilt."
        ),
        (
            f"**Measured:** moves use {inventory['move_source_files']} archived "
            f"files and {inventory['move_quote_rows']} admitted quote rows; every "
            f"retained game has a reconstructed move. Hash, target-week, "
            f"observation/book/market-clock and kickoff checks ran before fitting."
        ),
        (
            f"**Measured:** repaired projections: maximum constraint error "
            f"{inventory['max_price_error']:.8f}; maximum stationarity "
            f"{inventory['max_stationarity']:.3g}; maximum PMF normalization error "
            f"{inventory['max_normalization_error']:.3g}."
        ),
        "",
        table(
            ["Season", "Games", "Nonpush"],
            [[r["season"], r["games"], r["nonpush"]] for r in inventory["coverage"]],
        ),
        "",
        "## Effects and uncertainty",
        "",
        (
            "**Measured:** 10,000 paired bootstrap draws, seed 84 (85 for OOS), "
            "resample seasons then whole weeks; single-season panels resample "
            "weeks. Intervals are percentile 95%. Positive contrasts favor the "
            "candidate: baseline minus candidate for losses, candidate minus "
            "baseline for accuracy in percentage points. Historical accuracy is "
            "not a game's probability or evidence of profitability."
        ),
        (
            "**Measured:** optimistic IS scores final fold parameters on fitting "
            "rows; repeated historical rows across folds remain blocked by their "
            "original season/week. Contrast gap is IS benefit minus OOS benefit; "
            "absolute loss gaps are OOS minus IS. Tune and calibration rows are "
            "not outer results."
        ),
    ]
    for kind, title in (("absolute", "Five arms"), ("contrast", "Candidate paired improvements")):
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                table(
                    [
                        "Panel",
                        "Stage",
                        "Arm/comparator",
                        "Metric",
                        "Estimate [95% interval]",
                        "probability_positive",
                    ],
                    [
                        [
                            c["panel"],
                            c["scheme"],
                            c["arm"],
                            c["metric"],
                            interval(c),
                            f"{c['probability_positive']:.4f}" if kind == "contrast" else "-",
                        ]
                        for c in result["cells"]
                        if c["kind"] == kind
                    ],
                ),
            ]
        )
    coefficient_rows = []
    for c in coefficients:
        coefficient_rows.append(
            [
                c["outer"],
                c["arm"],
                f"{c['effective_intercept']:+.6f}",
                *[
                    f"{c[f'effective_{name}']:+.6f}" if f"effective_{name}" in c else "-"
                    for name in (
                        *FIT_FEATURES,
                        "jensen_gap",
                        "market_logit",
                        "elo_gap",
                        "tue_open_home_spread",
                    )
                ],
            ]
        )
    lines.extend(
        [
            "",
            "## Effective coefficients and stability",
            "",
            (
                "**Measured:** effective coefficients incorporate the tuned "
                "multiplier and calibration intercept. Unscaled coefficients, "
                "standardizers and stage sizes remain in scratch. Signs and "
                "variation describe three folds, without selecting one."
            ),
            "",
            table(
                [
                    "Outer",
                    "Arm",
                    "Intercept",
                    "Model logit",
                    "Flags",
                    "Move",
                    "Move available",
                    "Jensen gap",
                    "Market logit",
                    "Elo gap",
                    "Opener",
                ],
                coefficient_rows,
            ),
            "",
            "## Five equal-width reliability bands",
            "",
            "**Measured:** pooled OOS nonpush games; empty cells remain in the look count.",
            "",
            table(
                ["Arm", "Home-probability band", "Games", "Mean p", "Observed home cover"],
                [
                    [
                        r["arm"],
                        r["band"],
                        r["games"],
                        "-" if r["mean_probability"] is None else f"{r['mean_probability']:.6f}",
                        "-"
                        if r["observed_home_cover"] is None
                        else f"{r['observed_home_cover']:.6f}",
                    ]
                    for r in result["reliability"]
                ],
            ),
            "",
            "## Interpretation and handoff",
            "",
            (
                "**Measured:** 505 declared looks: 432 metric cells, 32 "
                "fit/nuisance summaries, 16 decisive records and 25 reliability "
                "bands. Five arms plus shared-prior, prior-variance and "
                "common-multiplier summaries account for B=8. Per-fold prior and "
                "variance parameters are in construction diagnostics; pooled "
                "summaries are descriptive, not new fits."
            ),
            (
                "**Inferred:** provisional `unresolved_below_power`, pending the "
                "orchestrator's serial candidate-versus-served-base records. No "
                "power-matched positive control was run. AGENTS.md research rules "
                "forbid closure from zero crossing; the discrete-serving and "
                "calibrated-probability rules prevent using a side rule or "
                "RPS-only result to change the card. No promotion is claimed."
            ),
            (
                "Prediction rows, normalized calibrated PMFs, exclusions, "
                "provenance, coefficients, all estimates and the protocol "
                "snapshot are under `tests/scratch/codex/lead84_unit2/`. Proposed "
                "bash record commands are in the lane, not executed."
            ),
            "",
        ]
    )
    primary = next(
        c
        for c in result["cells"]
        if (c["panel"], c["scheme"], c["kind"], c["arm"], c["metric"])
        == ("pooled", "OOS", "contrast", "served_base", "rps")
    )
    headline = (
        "**Measured:** primary RPS improvement "
        + interval(primary)
        + f"; probability_positive {primary['probability_positive']:.4f}. "
        + "**Inferred:** the primary mixture question remains unresolved_below_power; "
        + "this does not authorize a served change."
    )
    position = lines.index("## Locked design and chronology")
    lines[position:position] = [headline, ""]
    lines = [
        (
            "**Inferred:** provisional endpoint records await the orchestrator. "
            "Primary RPS and accuracy remain unresolved_below_power. "
            "Log-loss and Brier improvement intervals are wholly negative; the proposed "
            "refuted_mechanism records apply only to this fixed extension on those metrics, "
            "with closing ground wrong_sign_resolved under AGENTS.md research rules. "
            "The RPS mixture mechanism remains open. Intervals are unadjusted across "
            "the declared 505 looks; no family-wide significance or promotion is claimed. "
            "No power-matched positive control was run."
            if paragraph.startswith("**Inferred:** provisional")
            else paragraph
        )
        for paragraph in lines
    ]
    gap = [c["effective_jensen_gap"] for c in coefficients if c["arm"] == "candidate"]
    stability = (
        "**Measured:** effective Jensen coefficients for 2023/2024/2025 are "
        + ", ".join(f"{value:+.6f}" for value in gap)
        + ". The sign changes after the first fold; there is no stable positive weight."
    )
    position = lines.index("## Effective coefficients and stability") + 2
    lines[position:position] = [stability, ""]
    position = lines.index("## Source repair and population") + 2
    lines[position:position] = [
        "**Measured:** the pre-outcome clock amendment excludes 2020_11_DET_CAR "
        "from every arm and prior because only one usable leader-book timestamp exists. "
        "The first two attempts stopped before fitting/scoring; the third completed. "
        "Production push-row flags use historical completed-game spreads solely for "
        "prior-week standings eligibility. No target closing line enters a term or grade.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def record_commands(result):
    units = {
        "rps": "rps_improvement",
        "log_loss": "log_loss_improvement",
        "brier": "brier_improvement",
        "accuracy_points": "accuracy_points",
    }
    commands = []
    for cell in result["cells"]:
        if (cell["panel"], cell["scheme"], cell["kind"], cell["arm"]) != (
            "pooled",
            "OOS",
            "contrast",
            "served_base",
        ):
            continue
        metric = cell["metric"]
        resolved = cell["interval_high"] < 0
        classification = "refuted_mechanism" if resolved else "unresolved_below_power"
        evidence = (
            "Fixed fifth-term extension has a wholly negative improvement interval on "
            + metric
            + "; this does not close the separate primary RPS mixture question"
            if resolved
            else "No admissible closing ground for this endpoint; no power-matched control"
        )
        plain = (
            "Adding this book information made the stated opener chances less accurate. "
            "This concerns this version of the formula; pool picks stay unchanged."
            if resolved
            else "Combining book prices this way has not yet shown a clear improvement. "
            "This question remains open and pool picks stay unchanged."
        )
        args = [
            ".tools/uv.exe",
            "run",
            "--no-sync",
            "nfl-ats",
            "weak-signals",
            "record",
            "--name",
            f"LEAD-84-unit2-{metric}",
            "--description",
            f"LEAD-84 candidate versus served four-term recipe; {metric}",
            "--source",
            "docs/lead84_unit2.md",
            "--league",
            "nfl",
            "--category",
            "market",
            f"--effect={cell['effect']:.12g}",
            "--effect-units",
            units[metric],
            f"--interval-low={cell['interval_low']:.12g}",
            f"--interval-high={cell['interval_high']:.12g}",
            f"--probability-positive={cell['probability_positive']:.12g}",
            "--sample-games",
            str(cell["games"]),
            "--sample-blocks",
            str(cell["blocks"]),
            "--season-start",
            "2023",
            "--season-end",
            "2025",
            "--family",
            "LEAD-84-unit2",
            "--classification",
            classification,
            "--classification-evidence",
            evidence,
            "--plain-summary",
            plain,
            "--notes",
            "505 looks; opener; season/week bootstrap; unadjusted intervals",
        ]
        if cell["standard_error"] > 0:
            args.append(f"--standard-error={cell['standard_error']:.12g}")
        if resolved:
            args.extend(["--closing-ground", "wrong_sign_resolved"])
        commands.append(shlex.join(args))
    return commands


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    declaration = LANE.read_text(encoding="utf-8-sig").split("## Tried")[0]
    require(
        "Unit-two amendment, frozen BEFORE outcomes" in declaration and "505" in declaration,
        "Missing locked protocol",
    )
    protocol_path = OUTPUT / "protocol_declaration.md"
    if protocol_path.exists():
        require(
            protocol_path.read_text(encoding="utf-8") == declaration,
            "Do not revise the protocol after outcomes",
        )
    else:
        protocol_path.write_text(declaration, encoding="utf-8")
    sources = {
        str(p): digest(p)
        for p in (
            construction.OPENER,
            construction.BASE,
            construction.ARCHIVED,
            SCHEDULE,
            Path("scripts/lead84_unit1.py"),
            Path("tests/scratch/codex/lead84_unit1/book_panels.parquet"),
            Path("scripts/lead78_unit2.py"),
            Path("scripts/sunday_market_probability_eval.py"),
        )
    }
    checkpoint = OUTPUT / "construction_checkpoint.json"
    with threadpool_limits(limits=2):
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            require(
                saved["sources"] == sources and saved["protocol_sha256"] == digest(protocol_path),
                "Construction checkpoint is stale",
            )
            for path, expected in saved["artifact_hashes"].items():
                require(digest(path) == expected, f"Construction artifact changed: {path}")
            games = read(OUTPUT / "games.parquet")
            features = read(OUTPUT / "features.parquet")
            arrays = np.load(OUTPUT / "pmfs.npz")
            mixture, averaged = arrays["mixture"], arrays["averaged"]
            inventory = saved["inventory"]
            print("Verified construction checkpoint reused", flush=True)
        else:
            games, panels, inventory = inputs()
            games.to_parquet(OUTPUT / "games.parquet", index=False)
            features, diagnostics, folds, mixture, averaged = construction.construct(games, panels)
            features.to_parquet(OUTPUT / "features.parquet", index=False)
            diagnostics.to_parquet(OUTPUT / "projection_diagnostics.parquet", index=False)
            folds.to_json(OUTPUT / "construction_folds.json", orient="records", indent=2)
            np.savez_compressed(OUTPUT / "pmfs.npz", grid=GRID, mixture=mixture, averaged=averaged)
            inventory.update(
                max_price_error=float(diagnostics.price_error.max()),
                max_stationarity=float(diagnostics.stationarity.max()),
                max_normalization_error=float(
                    max(
                        np.max(abs(mixture.sum(axis=1) - 1)),
                        np.max(abs(averaged.sum(axis=1) - 1)),
                    )
                ),
            )
            files = [
                OUTPUT / name
                for name in (
                    "games.parquet",
                    "features.parquet",
                    "pmfs.npz",
                    "projection_diagnostics.parquet",
                    "construction_folds.json",
                    "panel_sources.json",
                    "move_sources.json",
                )
            ]
            write_json(
                checkpoint,
                {
                    "sources": sources,
                    "protocol_sha256": digest(protocol_path),
                    "artifact_hashes": {str(p): digest(p) for p in files},
                    "inventory": inventory,
                },
            )
        predictions, coefficients, parameters = replay(games, features, mixture, averaged)
        result = summarize(predictions)
    result["inventory"] = inventory
    result["record_commands"] = record_commands(result)
    write_json(OUTPUT / "summary.json", result)
    render(result, coefficients, parameters, inventory)
    for c in result["cells"]:
        if (c["panel"], c["scheme"], c["kind"], c["arm"]) == (
            "pooled",
            "OOS",
            "contrast",
            "served_base",
        ):
            print(
                f"{c['metric']}: {interval(c)} probability_positive={c['probability_positive']:.4f}"
            )
    print(f"Report saved: {REPORT}; looks={result['looks']['total']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

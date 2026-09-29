from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import fit_home_side_offsets
from nfl_ats.margin import fit_margin_model, margin_feature_columns
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    residual_location,
    tilted_atoms,
)
from nfl_ats.pick_probability_fit import FIT_RIDGE, _fit_logit

ROOT = Path("tests/scratch/codex/lead83_unit1")
OUTPUT = Path("tests/scratch/codex/lead83_unit2")
REPORT = Path("docs/lead83_unit2.md")
LANE = Path("docs/lanes/lead83.md")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
FIT = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
OPENER_META = Path("artifacts/opener_evaluation/20260929T192743Z/metadata.json")
SEED = 20260929
BOOTSTRAPS = 10000
GRID = np.arange(-100, 101)
ARMS = ("shrunk_move", "four_term", "model_only", "market", "elo")
TERMS = ("model_logit", "composition_flag_sum", "original_move", "available")
METRICS = ("accuracy_points", "log_loss", "brier", "rps")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def save(frame: pd.DataFrame, name: str) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), OUTPUT / name)


def tuesday_totals(games: pd.DataFrame, source: dict) -> pd.Series:
    collected = []
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "line",
        "outcome_side",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
        "home_team",
        "away_team",
    ]
    for entry in source["source_manifests"]:
        if entry["label"] != "tue_open":
            continue
        manifest_path = Path(entry["manifest"])
        quotes = manifest_path.parent / "quotes.parquet"
        if (
            digest(manifest_path) != entry["manifest_sha256"]
            or digest(quotes) != entry["quotes_sha256"]
        ):
            raise ValueError("Unit-1 Tuesday archive changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        stamp = pd.Timestamp(manifest["snapshot_timestamp_utc"])
        q = read(quotes, columns)
        q = q.loc[q.market.eq("totals")].rename(columns={"nflverse_game_id": "game_id"})
        target = games.loc[games.season.eq(entry["season"]) & games.week.eq(entry["week"])]
        q = q.merge(target, on="game_id", suffixes=("", "_schedule"), validate="many_to_one")
        for name in (
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "market_last_update_utc",
            "commence_time_utc",
        ):
            q[name] = pd.to_datetime(q[name], utc=True, errors="coerce", format="mixed")
        valid = (
            q.observed_at_utc.eq(stamp)
            & q.observed_at_utc.le(q.tuesday_noon)
            & q.observed_at_utc.lt(q.kickoff)
            & q.observed_at_utc.lt(q.commence_time_utc)
            & q.bookmaker_last_update_utc.le(q.observed_at_utc)
            & q.market_last_update_utc.le(q.observed_at_utc)
            & q.home_team.eq(q.home_team_schedule)
            & q.away_team.eq(q.away_team_schedule)
            & q.bookmaker_key.notna()
            & q.bookmaker_key.ne("")
            & np.isfinite(q.line)
        )
        q = q.loc[valid]
        per_book = q.groupby(["game_id", "bookmaker_key"]).line.agg(["median", "nunique", "count"])
        per_book = per_book.loc[per_book["nunique"].eq(1) & per_book["count"].eq(2)]
        collected.append(per_book["median"].groupby("game_id").median())
    totals = pd.concat(collected)
    if totals.index.duplicated().any():
        raise ValueError("Duplicate Tuesday total")
    return totals


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = json.loads((ROOT / "summary.json").read_text(encoding="utf-8-sig"))
    metadata = json.loads(OPENER_META.read_text(encoding="utf-8-sig"))
    if digest(FEATURES) != metadata["feature_table_sha256"]:
        raise ValueError("Upstream feature cache differs from the unit-1 parent")
    games = read(ROOT / "features.parquet")
    folds = read(ROOT / "fold_features.parquet")
    if games.game_id.duplicated().any() or int(games.declared_fit_population.sum()) != 1311:
        raise ValueError("Unit-1 population changed")
    totals = tuesday_totals(games, source)
    needed = list(
        dict.fromkeys(
            [
                "game_id",
                "result",
                "elo_diff",
                *margin_feature_columns("market_residual", "weak_stack"),
            ]
        )
    )
    needed = [name for name in needed if name not in ("spread_line", "total_line")]
    features = read(FEATURES, needed)
    frame = games.merge(features, on="game_id", how="left", validate="one_to_one")
    frame["spread_line"] = frame.tue_open_home_spread
    frame["total_line"] = frame.game_id.map(totals)
    frame["ats_margin"] = frame.result - frame.spread_line
    if not frame.ats_margin.ne(0).equals(frame.declared_fit_population):
        raise ValueError("Declared non-push population disagrees with opener outcomes")
    if frame.result.isna().any() or frame.elo_diff.isna().any() or frame.result.abs().gt(100).any():
        raise ValueError("Missing or out-of-grid outcomes/Elo")
    flags = read(FIT, ["game_id", "composition_flag_sum"])
    frame = frame.merge(flags, on="game_id", how="left", validate="one_to_one")
    if frame.loc[frame.declared_fit_population, "composition_flag_sum"].isna().any():
        raise ValueError("A scored game has no cached signed flags")
    books = read(ROOT / "matched_books.parquet", ["game_id", "home_spread_line_sunday"])
    frame["sunday_line"] = frame.game_id.map(
        books.groupby("game_id").home_spread_line_sunday.median()
    )
    if frame.sunday_line.isna().any():
        raise ValueError("Missing matched-book Sunday reference")
    frame["original_move"] = frame.median_move_toward_home
    frame["available"] = 1.0
    lineage = {
        "inputs": {
            str(path): digest(path)
            for path in (
                FEATURES,
                FIT,
                OPENER_META,
                ROOT / "features.parquet",
                ROOT / "fold_features.parquet",
                ROOT / "matched_books.parquet",
                ROOT / "summary.json",
            )
        },
        "protocol_sha256": digest(LANE),
        "population": len(frame),
        "nonpush": int(frame.declared_fit_population.sum()),
        "missing_tuesday_totals": int(frame.total_line.isna().sum()),
        "upstream_config": metadata["active_model_config"],
    }
    return frame, folds, lineage


def lattice(train: pd.DataFrame, rows: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    pool_lines = train.spread_line.to_numpy(float)
    pool_margins = np.rint(train.result.to_numpy(float))
    output = np.zeros((len(rows), len(GRID)))
    for index, (line, point) in enumerate(zip(rows.spread_line, points, strict=True)):
        band = BAND_HALF_WIDTH
        while True:
            selected = np.abs(pool_lines - line) <= band
            if int(selected.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
                break
            band = min(band + BAND_STEP, MAX_BAND)
        values, counts = np.unique(pool_margins[selected], return_counts=True)
        if not len(values):
            raise ValueError("Empty training lattice")
        mass, _theta = tilted_atoms(values, counts.astype(float), float(line), float(point))
        output[index, values.astype(int) + 100] = mass
    return output


def probability(mass: np.ndarray, lines: np.ndarray) -> np.ndarray:
    cover = (mass * (GRID[None, :] > lines[:, None])).sum(axis=1)
    loss = (mass * (GRID[None, :] < lines[:, None])).sum(axis=1)
    if np.any(cover <= 0) or np.any(loss <= 0):
        raise ValueError("Training lattice lacks one side")
    return np.clip(cover / (cover + loss), 1e-9, 1 - 1e-9)


def fit_probability(
    frame: pd.DataFrame, train_mask: np.ndarray, terms: tuple[str, ...]
) -> tuple[np.ndarray, dict]:
    x = frame.loc[:, list(terms)].to_numpy(float)
    mean = x[train_mask].mean(axis=0)
    scale = x[train_mask].std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(frame)), (x - mean) / scale])
    beta = _fit_logit(
        design[train_mask], frame.loc[train_mask, "home_covered"].to_numpy(float), FIT_RIDGE
    )
    natural = beta[1:] / scale
    coefficients = dict(zip(terms, natural.tolist(), strict=True))
    coefficients["intercept"] = float(beta[0] - mean @ natural)
    return expit(np.clip(design @ beta, -30, 30)), coefficients


def calibrate(p: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))
    tune = frame.role.eq("tune").to_numpy()
    cal = frame.role.eq("calibrate").to_numpy()
    y = frame.home_covered.to_numpy(float)
    optimum = minimize_scalar(
        lambda slope: float(np.mean((expit(slope * z[tune]) - y[tune]) ** 2)),
        bounds=(0.0, 10.0),
        method="bounded",
    )
    if not optimum.success:
        raise ValueError("Temperature tuning failed")
    candidates = [0.0, float(optimum.x), 10.0]
    slope = min(candidates, key=lambda s: float(np.mean((expit(s * z[tune]) - y[tune]) ** 2)))
    intercept = float(
        brentq(lambda b: float(np.mean(expit(slope * z[cal] + b) - y[cal])), -40.0, 40.0)
    )
    return np.clip(expit(slope * z + intercept), 1e-9, 1 - 1e-9), {
        "slope": slope,
        "intercept": intercept,
    }


def metrics(
    frame: pd.DataFrame, p: np.ndarray, raw_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lines = frame.spread_line.to_numpy(float)
    home = GRID[None, :] > lines[:, None]
    away = GRID[None, :] < lines[:, None]
    pushes = ~(home | away)
    cover_mass = (raw_mass * home).sum(axis=1)
    loss_mass = (raw_mass * away).sum(axis=1)
    push_mass = (raw_mass * pushes).sum(axis=1)
    cover_scale = (1 - push_mass) * p / cover_mass
    loss_scale = (1 - push_mass) * (1 - p) / loss_mass
    mass = raw_mass * (home * cover_scale[:, None] + away * loss_scale[:, None] + pushes)
    if not np.allclose(mass.sum(axis=1), 1) or not np.allclose(probability(mass, lines), p):
        raise ValueError("Calibrated lattice failed mass/probability identity")
    y = frame.home_covered.to_numpy(float)
    result = np.column_stack(
        [
            100 * ((p >= 0.5) == y),
            -(y * np.log(p) + (1 - y) * np.log1p(-p)),
            (p - y) ** 2,
            ((np.cumsum(mass, axis=1) - (frame.result.to_numpy()[:, None] <= GRID)) ** 2).sum(
                axis=1
            ),
        ]
    )
    return result, push_mass


def replay(frame: pd.DataFrame, cached: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    details = []
    for outer in (2023, 2024, 2025):
        panel = cached.loc[
            cached.outer_season.eq(outer),
            ["game_id", "role", "tau_squared", "shrink_weight", "shrunk_move_toward_home"],
        ]
        panel = (
            frame.merge(panel, on="game_id", validate="one_to_one")
            .sort_values(["season", "week", "game_id"])
            .reset_index(drop=True)
        )
        train = panel.loc[panel.role.eq("fit")].copy()
        if train.season.max() != outer - 3 or train.season.min() != 2020:
            raise ValueError("Upstream fixed-cutoff mismatch")
        for role, season in (("tune", outer - 2), ("calibrate", outer - 1), ("outer", outer)):
            if not panel.loc[panel.role.eq(role), "season"].eq(season).all():
                raise ValueError("Fold-role mismatch")
        model = fit_margin_model(
            train,
            target="market_residual",
            model_name="ridge",
            feature_profile="weak_stack",
            ridge_alpha=10.0,
        )
        forecast = model.predict(panel, probability_method="gaussian_median")
        point = forecast.predicted_margin.to_numpy() + residual_location(
            model.residuals, "gaussian_median"
        )
        offset_rows = train.copy()
        offset_rows["point_incumbent"] = point[panel.role.eq("fit")]
        offsets = fit_home_side_offsets(offset_rows)
        point += offsets.offset_for(panel.spread_line).to_numpy(float)
        elo_x = np.column_stack([np.ones(len(panel)), panel.elo_diff.to_numpy(float)])
        training = panel.role.eq("fit").to_numpy()
        elo_beta = np.linalg.lstsq(
            elo_x[training], panel.loc[training, "result"].to_numpy(float), rcond=None
        )[0]
        points = {
            "model_only": point,
            "market": panel.sunday_line.to_numpy(float),
            "elo": elo_x @ elo_beta,
        }
        masses = {name: lattice(train, panel, values) for name, values in points.items()}
        raw = {
            name: probability(mass, panel.spread_line.to_numpy(float))
            for name, mass in masses.items()
        }
        panel["model_logit"] = logit(raw["model_only"])
        kept = panel.declared_fit_population.to_numpy()
        upstream = panel[["game_id", "season", "week", "role", "spread_line", "model_logit"]].copy()
        upstream["outer_season"] = outer
        upstream["fit_through"] = outer - 3
        upstream["training_max_gameday"] = str(pd.to_datetime(train.gameday).max().date())
        upstream["point"] = point
        upstream["model_probability"] = raw["model_only"]
        save(upstream, f"upstream_{outer}.parquet")
        panel = panel.loc[kept].reset_index(drop=True)
        panel["home_covered"] = panel.ats_margin.gt(0).astype(float)
        panel["shrunk_move"] = panel.shrunk_move_toward_home
        train_mask = panel.role.eq("fit").to_numpy()
        raw = {name: values[kept] for name, values in raw.items()}
        masses = {name: values[kept] for name, values in masses.items()}
        coefficients = {}
        for arm, terms in (("four_term", TERMS), ("shrunk_move", (*TERMS, "shrunk_move"))):
            raw[arm], coefficients[arm] = fit_probability(panel, train_mask, terms)
            masses[arm] = masses["model_only"]
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "upstream_games_with_pushes": len(train),
            "training_max_gameday": upstream.training_max_gameday.iloc[0],
            "roles": panel.role.value_counts().to_dict(),
            "tau_squared": float(panel.tau_squared.iloc[0]),
            "home_side_offsets": offsets.offsets,
            "elo_margin_coefficients": {
                "intercept": float(elo_beta[0]),
                "elo_diff": float(elo_beta[1]),
            },
            "probability_coefficients": {},
            "calibration": {},
        }
        for arm in ARMS:
            p, calibration = calibrate(raw[arm], panel)
            scores, push = metrics(panel, p, masses[arm])
            detail["calibration"][arm] = calibration
            if arm in coefficients:
                effective = {
                    name: value * calibration["slope"] for name, value in coefficients[arm].items()
                }
                effective["intercept"] += calibration["intercept"]
                detail["probability_coefficients"][arm] = {
                    "fit": coefficients[arm],
                    "final": effective,
                }
            for phase, role in (("IS", "fit"), ("OOS", "outer")):
                selected = panel.role.eq(role).to_numpy()
                result = panel.loc[
                    selected,
                    [
                        "game_id",
                        "season",
                        "week",
                        "result",
                        "spread_line",
                        "home_covered",
                        "matched_books",
                        "shrink_weight",
                    ],
                ].copy()
                result["outer"] = outer
                result["phase"] = phase
                result["arm"] = arm
                result["probability"] = p[selected]
                result["push_probability"] = push[selected]
                for index, metric in enumerate(METRICS):
                    result[metric] = scores[selected, index]
                rows.append(result)
        details.append(detail)
        print(
            f"fold={outer} completed; upstream cutoff={outer - 3}; "
            f"outer games={int(panel.role.eq('outer').sum())}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def blocked(frame: pd.DataFrame, values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = pd.DataFrame(values).assign(season=frame.season.to_numpy(), week=frame.week.to_numpy())
    grouped = blocks.groupby(["season", "week"])
    sums = grouped.sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    years = grouped.size().index.get_level_values(0).to_numpy()
    seasons = np.unique(years)
    indexes = {year: np.flatnonzero(years == year) for year in seasons}
    samples = np.empty((BOOTSTRAPS, values.shape[1]))
    for index in range(BOOTSTRAPS):
        chosen = np.concatenate(
            [
                rng.choice(indexes[year], len(indexes[year]), replace=True)
                for year in rng.choice(seasons, len(seasons), replace=True)
            ]
        )
        samples[index] = sums[chosen].sum(axis=0) / counts[chosen].sum()
    return samples


def estimate(value: float, draws: np.ndarray) -> dict:
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(value),
        "low": float(low),
        "high": float(high),
        "probability_positive": float(
            np.mean(draws > 1e-12) + 0.5 * np.mean(np.abs(draws) <= 1e-12)
        ),
    }


def summarize(predictions: pd.DataFrame) -> dict:
    panels = {}
    labels = [*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:])]
    for outer in (2023, 2024, 2025, "pooled"):
        selected = (
            predictions if outer == "pooled" else predictions.loc[predictions.outer.eq(outer)]
        )
        panel = {"phases": {}, "decisive": {}}
        phase_draws = {}
        phase_means = {}
        for phase in ("IS", "OOS"):
            frame = selected.loc[selected.phase.eq(phase)]
            groups = {
                arm: frame.loc[frame.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = groups["shrunk_move"]
            if any(
                not g[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for g in groups.values()
            ):
                raise ValueError("Unpaired arm populations")
            arrays = [groups[arm].loc[:, list(METRICS)].to_numpy(float) for arm in ARMS]
            direction = np.array([1.0, -1.0, -1.0, -1.0])
            contrasts = [(arrays[0] - array) * direction for array in arrays[1:]]
            all_values = np.column_stack([*arrays, *contrasts])
            samples = blocked(
                reference,
                all_values,
                SEED + (0 if outer == "pooled" else int(outer)) + (0 if phase == "IS" else 100),
            )
            means = all_values.mean(axis=0)
            phase_draws[phase] = samples
            phase_means[phase] = means
            phase_panel = {}
            for index, label in enumerate(labels):
                phase_panel[label] = {
                    metric: estimate(means[index * 4 + m], samples[:, index * 4 + m])
                    for m, metric in enumerate(METRICS)
                }
                phase_panel[label]["games"] = len(reference)
                if label in ARMS:
                    wins = int(groups[label].accuracy_points.eq(100).sum())
                    phase_panel[label]["record"] = f"{wins}-{len(reference) - wins}"
            panel["phases"][phase] = phase_panel
            if phase == "OOS":
                candidate_home = reference.probability.ge(0.5)
                for baseline in ARMS[1:]:
                    decisive = candidate_home.ne(groups[baseline].probability.ge(0.5))
                    wins = int(reference.loc[decisive, "accuracy_points"].eq(100).sum())
                    n = int(decisive.sum())
                    panel["decisive"][baseline] = {
                        "games": n,
                        "candidate_wins": wins,
                        "base_wins": n - wins,
                        "exact_p": float(binomtest(wins, n).pvalue) if n else 1.0,
                    }
        gap_draws = phase_draws["OOS"] - phase_draws["IS"]
        gap_means = phase_means["OOS"] - phase_means["IS"]
        panel["gaps_OOS_minus_IS"] = {
            label: {
                metric: estimate(gap_means[index * 4 + m], gap_draws[:, index * 4 + m])
                for m, metric in enumerate(METRICS)
            }
            for index, label in enumerate(labels)
        }
        panels[str(outer)] = panel
    reliability = []
    oos = predictions.loc[predictions.phase.eq("OOS")]
    for arm in ARMS:
        group = oos.loc[oos.arm.eq(arm)]
        band = np.minimum((group.probability * 5).astype(int), 4)
        for index in range(5):
            cell = group.loc[band.eq(index)]
            reliability.append(
                {
                    "arm": arm,
                    "band": f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    "n": len(cell),
                    "predicted": float(cell.probability.mean()) if len(cell) else None,
                    "observed": float(cell.home_covered.mean()) if len(cell) else None,
                }
            )
    return {"panels": panels, "reliability": reliability}


def number(cell: dict) -> str:
    return f"{cell['estimate']:.6f} [{cell['low']:.6f}, {cell['high']:.6f}]"


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def report(summary: dict) -> None:
    panels = summary["panels"]
    lines = [
        "# LEAD-83 unit 2: shrink noisy movement",
        "",
        "Historical 2020-2025 OPENER is the frozen pool-line proxy; the served four-term",
        "recipe is the base. No pre-2026 Splash captures are used.",
        "",
        "## Decisive games first",
        "",
        "**Measured:** records count only outer games where the candidate and comparator",
        "selected different sides. Exact p is the two-sided fair-coin null on these games;",
        "no population split is claimed.",
        "",
    ]
    rows = []
    for outer, panel in panels.items():
        for base, cell in panel["decisive"].items():
            rows.append(
                [
                    outer,
                    base,
                    cell["games"],
                    f"{cell['candidate_wins']}-{cell['base_wins']}",
                    f"{cell['exact_p']:.6f}",
                ]
            )
    lines += [
        table(["Outer", "Comparator", "Decisive n", "Candidate W-L", "Exact p"], rows),
        "",
        "## Protocol and upstream blocker",
        "",
        "**Read:** the lane's pre-outcome amendment retains 501 looks:",
        "(27*4+7+4)*(3+1)+25. Five arms, four candidate contrasts, four endpoints, three",
        "outer seasons plus pooled panels, IS/OOS/gaps, coefficients, decisive records",
        "and 25 reliability cells are the declared family. No outcome-selected variants.",
        "",
        "**Measured:** upstream weak-stack margin ridge, home-side offsets, line-local",
        "discrete pools, Elo margin fit and four-term/candidate coefficients were refitted",
        "using only 2020 through Y-3. Pushes remain in upstream fits/distributions.",
        "The 1,311 non-push rows retain unit 1's clocks and original/shrunk movements;",
        "outer 2023-2025 has 697 games. Cached weekly-trained model probabilities are",
        "never loaded. The active parent feature hash was verified. Later spread/total",
        "inputs were replaced with opener spreads and archived Tuesday totals.",
        "",
        "**Measured:** a nonnegative logit slope is tuned on Y-2 by Brier, then an",
        "intercept is calibrated on Y-1 by log loss. The final probability reweights",
        "the training discrete margin lattice's cover/loss groups, preserving push",
        "mass and within-side shape; it alone selects sides. Market uses matched-book",
        "Sunday median; Elo uses training-only linear margin on pregame Elo difference.",
        "RPS is the full integer-margin CDF score on -100...100, evaluated on the same",
        "non-push games as the other endpoints.",
        "",
        f"**Measured:** missing Tuesday totals: {summary['lineage']['missing_tuesday_totals']};",
        "missing values stay missing. Thread pools are limited to one. No full-history",
        "feature rebuild or served change was run.",
        "",
        "**Inferred:** this fixed-cutoff refit of the served recipe does not reproduce",
        "its weekly-trained cached scores. Upstream training predictions and downstream",
        "training scores are optimistic; IS is diagnostic. Retrospective source reuse",
        "still requires prospective confirmation.",
        "",
        "## Outer scores and uncertainty",
        "",
        "**Measured:** 95% intervals use 10,000 hierarchical season/whole-week bootstrap",
        "replicates, seed 20260929. Single-season panels resample weeks; pooled panels",
        "resample seasons then weeks. Positive contrasts mean candidate improvement;",
        "probability_positive gives half weight to exact ties. Primary endpoint: Brier.",
        "Historical forced picks are not per-game probabilities or profitability evidence.",
        "",
    ]
    rows = []
    for outer, panel in panels.items():
        for arm in ARMS:
            cell = panel["phases"]["OOS"][arm]
            rows.append([outer, arm, cell["record"], *(number(cell[m]) for m in METRICS)])
    lines += [
        table(["Outer", "Arm", "Record", "Accuracy %", "Log loss", "Brier", "RPS"], rows),
        "",
        "## Candidate improvements",
        "",
        "**Measured:** all four contrasts are reported; registry commands are restricted",
        "to candidate versus four-term. Positive loss contrasts mean lower candidate loss.",
        "",
    ]
    rows = []
    for outer, panel in panels.items():
        for base in ARMS[1:]:
            cell = panel["phases"]["OOS"][f"gain_vs_{base}"]
            for metric in METRICS:
                rows.append(
                    [
                        outer,
                        base,
                        metric,
                        number(cell[metric]),
                        f"{cell[metric]['probability_positive']:.5f}",
                    ]
                )
    lines += [
        table(
            [
                "Outer",
                "Comparator",
                "Endpoint",
                "Improvement [95% interval]",
                "probability_positive",
            ],
            rows,
        ),
        "",
        "## Optimistic IS, OOS and gap",
        "",
        "**Measured:** IS scores each final fold model on its training rows. Pooled IS",
        "repeats earlier games across folds; bootstrap blocks keep repetitions together.",
        "Gap is OOS minus IS, including positively oriented contrasts. Full IS/OOS/gap",
        "intervals for every arm/contrast are also saved in scratch summary.json.",
        "",
    ]
    rows = []
    for outer, panel in panels.items():
        for arm in (*ARMS, *(f"gain_vs_{a}" for a in ARMS[1:])):
            for metric in METRICS:
                inside = panel["phases"]["IS"][arm][metric]
                outside = panel["phases"]["OOS"][arm][metric]
                gap = panel["gaps_OOS_minus_IS"][arm][metric]
                rows.append(
                    [
                        outer,
                        arm,
                        metric,
                        f"{inside['estimate']:.6f}",
                        f"{outside['estimate']:.6f}",
                        number(gap),
                    ]
                )
    lines += [
        table(["Outer", "Arm/contrast", "Endpoint", "IS", "OOS", "Gap [95% interval]"], rows),
        "",
        "## Fold coefficients",
        "",
        "**Measured:** final coefficients include tune slope and calibration intercept.",
        "Availability is one throughout this complete-source population, so its",
        "standardized coefficient is zero. It is not identifiable from the intercept.",
        "Opposed movement coefficients can reflect collinearity; stability is shown below.",
        "",
    ]
    rows = []
    for fold in summary["folds"]:
        for arm, coefficients in fold["probability_coefficients"].items():
            for stage in ("fit", "final"):
                cell = coefficients[stage]
                rows.append(
                    [
                        fold["outer"],
                        arm,
                        stage,
                        *(
                            f"{cell.get(name, 0):.8f}"
                            for name in ("intercept", *TERMS, "shrunk_move")
                        ),
                    ]
                )
    lines += [
        table(
            [
                "Outer",
                "Arm",
                "Stage",
                "Intercept",
                "Model logit",
                "Flags",
                "Original move",
                "Availability",
                "Shrunk move",
            ],
            rows,
        ),
        "",
    ]
    rows = []
    for fold in summary["folds"]:
        for arm, cal in fold["calibration"].items():
            rows.append(
                [
                    fold["outer"],
                    fold["fit_through"],
                    arm,
                    f"{cal['slope']:.8f}",
                    f"{cal['intercept']:.8f}",
                ]
            )
    lines += [
        table(["Outer", "Fit through", "Arm", "Tune slope", "Calibration intercept"], rows),
        "",
    ]
    rows = [
        [
            f["outer"],
            f["upstream_games_with_pushes"],
            f["training_max_gameday"],
            f"{f['tau_squared']:.8f}",
            f"{f['elo_margin_coefficients']['intercept']:.8f}",
            f"{f['elo_margin_coefficients']['elo_diff']:.8f}",
        ]
        for f in summary["folds"]
    ]
    lines += [
        table(
            [
                "Outer",
                "Upstream n incl pushes",
                "Last training day",
                "tau^2",
                "Elo intercept",
                "Elo slope",
            ],
            rows,
        ),
        "",
        "## Reliability",
        "",
        "**Measured:** five equal-width bins per arm; predicted and observed entries",
        "are home-cover probabilities/frequencies conditional on a non-push.",
        "",
    ]
    rows = [
        [
            r["arm"],
            r["band"],
            r["n"],
            "-" if r["predicted"] is None else f"{r['predicted']:.6f}",
            "-" if r["observed"] is None else f"{r['observed']:.6f}",
        ]
        for r in summary["reliability"]
    ]
    lines += [
        table(["Arm", "Band", "Games", "Predicted", "Observed"], rows),
        "",
        "## Interpretation and handoff",
        "",
    ]
    primary = panels["pooled"]
    inside = primary["phases"]["IS"]["shrunk_move"]["accuracy_points"]
    outside = primary["phases"]["OOS"]["shrunk_move"]["accuracy_points"]
    gap = primary["gaps_OOS_minus_IS"]["shrunk_move"]["accuracy_points"]
    rps_2023 = panels["2023"]["phases"]["OOS"]["gain_vs_four_term"]["rps"]
    zero_years = [
        str(f["outer"]) for f in summary["folds"] if f["calibration"]["shrunk_move"]["slope"] == 0
    ]
    lines += [
        f"**Measured:** candidate accuracy IS={inside['estimate']:.6f}%,",
        f"OOS={outside['estimate']:.6f}%; gap={number(gap)} points.",
        f"The tune multiplier is zero in {', '.join(zero_years) or 'no outer season'}.",
        "A zero tune multiplier makes that fold's conditional cover probability",
        "constant after intercept calibration; it is not a split-half reliability test.",
        f"The 2023 secondary RPS improvement is {number(rps_2023)};",
        f"probability_positive={rps_2023['probability_positive']:.5f}.",
        "**Inferred:** this adverse secondary result does not settle the predeclared",
        "pooled Brier mechanism. The opposed original/shrunk coefficients and strong",
        "tuning attenuation do not establish a stable noise-shrinkage gain.",
        "",
    ]

    cell = panels["pooled"]["phases"]["OOS"]["gain_vs_four_term"]
    lines += [
        f"**Measured:** primary Brier improvement: {number(cell['brier'])};",
        f"probability_positive={cell['brier']['probability_positive']:.5f}.",
        f"Opener accuracy improvement: {number(cell['accuracy_points'])} points;",
        f"probability_positive={cell['accuracy_points']['probability_positive']:.5f}.",
        "",
        "**Inferred:** unresolved_below_power; no research closure or serving decision",
        "follows. AGENTS.md:65-85 permits closure only through admissible refutation or",
        "demonstrated positive-control power; AGENTS.md:113-126 separates promotion from",
        "research closure. No positive control was run. The report remains pending the",
        "orchestrator's serial registry commands in the lane.",
        "",
        "**Measured:** executed:",
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead83_unit2.py.",
        "Prediction rows, fixed-cutoff upstream probabilities, lineage, intervals and",
        "fold parameters are under tests/scratch/codex/lead83_unit2/. No prediction-row",
        "dumps are in docs. No registry command was run.",
        "",
    ]
    frozen = (OUTPUT / "protocol.md").read_text(encoding="utf-8-sig")
    frozen_protocol = frozen.split("## Protocol and pre-outcome amendment\n", 1)[1].split(
        "\n## Tried", 1
    )[0]
    lines += ["## Frozen amendment (verbatim)", "", frozen_protocol.strip(), ""]

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("An outcome replay already exists; do not silently rescore")
    (OUTPUT / "protocol.md").write_bytes(LANE.read_bytes())
    with threadpool_limits(limits=1):
        frame, cached, lineage = load()
        predictions, folds = replay(frame, cached)
        save(predictions, "predictions.parquet")
        summary = summarize(predictions)
        summary.update(
            {
                "lineage": lineage,
                "folds": folds,
                "looks": 501,
                "bootstrap_samples": BOOTSTRAPS,
                "seed": SEED,
            }
        )
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        report(summary)
    pooled = summary["panels"]["pooled"]["phases"]["OOS"]
    print(
        json.dumps(
            {
                "status": "complete",
                "games": pooled["shrunk_move"]["games"],
                "record": pooled["shrunk_move"]["record"],
                "candidate_vs_four_term": pooled["gain_vs_four_term"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

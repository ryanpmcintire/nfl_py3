from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from nfl_ats.pick_probability import BASE_PROBABILITY_POLICY
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

ATLAS_POINTER = "active_signal_atlas.json"
LABELS = {
    "overall": "All games",
    "weeks_1_4": "Weeks 1-4",
    "weeks_5_12": "Weeks 5-12",
    "weeks_13_18": "Weeks 13-18",
    "short": "Spread 7 or less",
    "long": "Spread 7.5 or more",
}
SIGNAL_LABELS = {
    "composition_flag_sum": "Combined game situations",
    "market_move_toward_home": "Market move toward the home side",
    "market_move_available": "Whether a market move was available",
}
SPLIT_LABELS = {
    "week_in_season": "Time of season",
    "spread_band": "Spread size",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Signal atlas path escapes its artifact root")
    return path


def _source(artifacts_root: Path) -> tuple[Path, dict[str, Any]]:
    pointer = _read(artifacts_root / "active_pick_probability.json")
    active = _read(artifacts_root / "active_ats_model.json")
    if pointer["active_model_id"] != active["model_id"]:
        raise ValueError("Signal atlas probability source does not match the active model")
    directory = _inside(artifacts_root, pointer["artifact"])
    metadata = _read(directory / "metadata.json")
    for key in ("active_model_id", "base_probability_policy", "market_move_feature_version"):
        if metadata.get(key) != pointer.get(key):
            raise ValueError(f"Signal atlas source mismatch: {key}")
    if metadata["base_probability_policy"] != BASE_PROBABILITY_POLICY:
        raise ValueError("Signal atlas requires discrete conditional opener probabilities")
    if metadata.get("model_probability_source") != "home_cover_probability_at_open":
        raise ValueError("Signal atlas requires opener-graded base probabilities")
    return directory, metadata


def load_signal_atlas(
    artifacts_root: Path, registry_root: Path = Path("registry")
) -> dict[str, Any] | None:
    pointer_path = artifacts_root / ATLAS_POINTER
    if not pointer_path.exists():
        return None
    pointer = _read(pointer_path)
    source, metadata = _source(artifacts_root)
    directory = _inside(artifacts_root, pointer["artifact"])
    if set(pointer["files"]) != {
        "report.json",
        "per_game.parquet",
        "coefficients.json",
        "declaration.json",
        "weak_signals_batch.json",
    }:
        raise ValueError("Signal atlas manifest is incomplete")
    for name, digest in pointer["files"].items():
        if _hash(_inside(directory, name)) != digest:
            raise ValueError(f"Signal atlas artifact changed: {name}")
    report = _read(directory / "report.json")
    for name, field in (
        ("conditional_signal_atlas.json", "declaration_sha256"),
        ("split_library.json", "split_library_sha256"),
    ):
        if _hash(registry_root / name) != report[field]:
            raise ValueError(f"Signal atlas definitions changed: {name}")
    if _hash(directory / "declaration.json") != report["saved_declaration_sha256"]:
        raise ValueError("Signal atlas declaration differs from its report")
    if report["active_model_id"] != metadata["active_model_id"]:
        raise ValueError("Signal atlas is stale for the active model")
    if report["source_artifact"] != source.relative_to(artifacts_root.resolve()).as_posix():
        raise ValueError("Signal atlas is stale for the fitted probability source")
    for name, digest in report["source_hashes"].items():
        if _hash(_inside(source, name)) != digest:
            raise ValueError(f"Signal atlas fitted source changed: {name}")
    return report


def _fit_predict(
    train: pd.DataFrame, target: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, dict[str, float]]:
    means = train[features].mean().to_numpy(dtype=float)
    stds = train[features].std(ddof=0).to_numpy(dtype=float)
    stds = np.where(stds > 0.0, stds, 1.0)
    x = np.column_stack((np.ones(len(train)), (train[features].to_numpy() - means) / stds))
    beta = _fit_logit(x, train.home_covered.to_numpy(dtype=float), ridge=FIT_RIDGE)
    target_x = np.column_stack((np.ones(len(target)), (target[features].to_numpy() - means) / stds))
    probability = 1.0 / (1.0 + np.exp(-np.clip(target_x @ beta, -35.0, 35.0)))
    natural = beta[1:] / stds
    coefficients = {"intercept": float(beta[0] - natural @ means)}
    coefficients.update(dict(zip(features, map(float, natural), strict=True)))
    return probability, coefficients


def _paired_predictions(
    frame: pd.DataFrame, declaration: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pairs = frame[
        [
            "game_id",
            "season",
            "week",
            "home_covered",
            "model_probability",
            "tue_open_home_spread",
        ]
    ].copy()
    coefficients: list[dict[str, Any]] = []
    for evaluation in declaration["evaluations"]:
        for arm in ("full", "reduced"):
            pairs[f"{evaluation}_{arm}"] = np.nan
        seasons = sorted(frame.season.unique()) if evaluation != "in_sample" else [None]
        for season in seasons:
            if evaluation == "in_sample":
                train, target = frame, frame
            elif evaluation == "out_of_season":
                train, target = frame[frame.season != season], frame[frame.season == season]
            else:
                train, target = frame[frame.season < season], frame[frame.season == season]
                if train.season.nunique() < declaration["chronological_minimum_training_seasons"]:
                    continue
                if int(train.season.max()) >= int(target.season.min()):
                    raise ValueError("Signal atlas chronological training crosses its cutoff")
            if train.empty or target.empty:
                raise ValueError("Signal atlas fold has no training or scoring games")
            for arm in ("full", "reduced"):
                probability, weights = _fit_predict(train, target, declaration[f"{arm}_features"])
                pairs.loc[target.index, f"{evaluation}_{arm}"] = probability
                coefficients.append(
                    {
                        "evaluation": evaluation,
                        "arm": arm,
                        "held_out_season": int(season) if season is not None else None,
                        "training_seasons": sorted(map(int, train.season.unique())),
                        "training_games": len(train),
                        "scored_games": len(target),
                        "coefficients": weights,
                    }
                )
        saved = frame[f"{evaluation}_home_probability"].to_numpy(dtype=float)
        reproduced = pairs[f"{evaluation}_full"].to_numpy(dtype=float)
        if not np.allclose(saved, reproduced, atol=1e-10, rtol=1e-10, equal_nan=True):
            raise ValueError(f"Signal atlas did not reproduce saved {evaluation} probabilities")
        if not pairs[f"{evaluation}_full"].isna().equals(pairs[f"{evaluation}_reduced"].isna()):
            raise ValueError("Signal atlas full and reduced populations differ")
    return pairs, coefficients


def _scores(probability: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.column_stack(
        (
            ((p >= 0.5) == outcomes).astype(float),
            (p - outcomes) ** 2,
            -(outcomes * np.log(p) + (1.0 - outcomes) * np.log1p(-p)),
        )
    )


def _metrics(probability: np.ndarray, outcomes: np.ndarray, edges: list[float]) -> dict[str, Any]:
    scores = _scores(probability, outcomes)
    reliability = []
    for lower, upper in pairwise(edges):
        mask = (probability >= lower) & (
            (probability < upper) if upper < 1.0 else (probability <= upper)
        )
        if mask.any():
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "games": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()),
                    "observed_frequency": float(outcomes[mask].mean()),
                }
            )
    return dict(
        zip(("accuracy", "brier", "log_loss"), map(float, scores.mean(axis=0)), strict=True)
    ) | {"reliability": reliability}


def _bootstrap(frame: pd.DataFrame, difference: np.ndarray, draws: int, seed: int) -> np.ndarray:
    blocks = []
    for season in sorted(frame.season.unique()):
        weeks = []
        for week in sorted(frame.loc[frame.season == season, "week"].unique()):
            mask = ((frame.season == season) & (frame.week == week)).to_numpy()
            weeks.append(np.append(difference[mask].sum(axis=0), mask.sum()))
        blocks.append(np.asarray(weeks, dtype=float))
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((draws, 3))
    for draw in range(draws):
        total = np.zeros(4)
        for index in rng.integers(0, len(blocks), size=len(blocks)):
            block = blocks[index]
            total += block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        bootstrap[draw] = total[:3] / total[3]
    return bootstrap


def _cell(
    frame: pd.DataFrame, evaluation: str, cell: str, declaration: dict[str, Any]
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"Signal atlas declared cell has no games: {evaluation}/{cell}")
    y = frame.home_covered.to_numpy(dtype=float)
    full = frame[f"{evaluation}_full"].to_numpy(dtype=float)
    reduced = frame[f"{evaluation}_reduced"].to_numpy(dtype=float)
    difference = (_scores(full, y) - _scores(reduced, y)) * np.array([100.0, -1.0, -1.0])
    bootstrap = _bootstrap(frame, difference, declaration["bootstrap_draws"], declaration["seed"])
    tail = (1.0 - declaration["interval_level"]) / 2.0
    interval = np.quantile(bootstrap, [tail, 1.0 - tail], axis=0)
    positive = (bootstrap > 0).mean(axis=0) + 0.5 * (bootstrap == 0).mean(axis=0)
    decisive = (full >= 0.5) != (reduced >= 0.5)
    full_wins = int(((full >= 0.5) == y)[decisive].sum())
    reduced_wins = int(decisive.sum()) - full_wins
    probabilities = {
        "full": full,
        "reduced": reduced,
        "model": frame.model_probability.to_numpy(dtype=float),
        "market": np.full(len(frame), 0.5),
    }
    seasons = []
    for season in sorted(frame.season.unique()):
        mask = (frame.season == season).to_numpy()
        seasons.append(
            {
                "season": int(season),
                "games": int(mask.sum()),
                "accuracy_delta_points": float(difference[mask, 0].mean()),
                "brier_improvement": float(difference[mask, 1].mean()),
            }
        )
    return {
        "cell": cell,
        "label": LABELS[cell],
        "games": len(frame),
        "blocks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "decisive_games": int(decisive.sum()),
        "full_decisive_wins": full_wins,
        "reduced_decisive_wins": reduced_wins,
        "accuracy_delta_points": float(difference[:, 0].mean()),
        "accuracy_interval": interval[:, 0].tolist(),
        "probability_positive": float(positive[0]),
        "brier_improvement": float(difference[:, 1].mean()),
        "brier_interval": interval[:, 1].tolist(),
        "brier_probability_positive": float(positive[1]),
        "log_loss_improvement": float(difference[:, 2].mean()),
        "log_loss_interval": interval[:, 2].tolist(),
        "log_loss_probability_positive": float(positive[2]),
        "standard_errors": bootstrap.std(axis=0, ddof=1).tolist(),
        "exact_null_p": float(binomtest(full_wins, full_wins + reduced_wins).pvalue)
        if decisive.any()
        else 1.0,
        "seasons": seasons,
        "metrics": {
            arm: _metrics(probability, y, declaration["reliability_edges"])
            for arm, probability in probabilities.items()
        },
    }


def _split_masks(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    if split == "week_in_season":
        return {
            "overall": np.ones(len(frame), dtype=bool),
            "weeks_1_4": (frame.week <= 4).to_numpy(),
            "weeks_5_12": frame.week.between(5, 12).to_numpy(),
            "weeks_13_18": (frame.week >= 13).to_numpy(),
        }
    if split == "spread_band":
        spread = frame.tue_open_home_spread.abs().to_numpy(dtype=float)
        return {
            "overall": np.ones(len(frame), dtype=bool),
            "short": spread <= 7.0,
            "long": spread >= 7.5,
        }
    raise ValueError(f"Signal atlas split is not registered: {split}")


def _family_registry_batch(family: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    cells = []
    metrics = (
        ("accuracy_delta_points", "accuracy_points", "accuracy_interval", "probability_positive"),
        ("brier_improvement", "brier_improvement", "brier_interval", "brier_probability_positive"),
        (
            "log_loss_improvement",
            "log_loss_improvement",
            "log_loss_interval",
            "log_loss_probability_positive",
        ),
    )
    for view, rows in family["evaluations"].items():
        for row in rows:
            seasons = [item["season"] for item in row["seasons"]]
            for index, (metric, units, interval, positive) in enumerate(metrics):
                cells.append(
                    {
                        "name": f"{family['family']}_{view}_{row['cell']}_{units}",
                        "family": family["family"],
                        "description": "; ".join(
                            (
                                f"Paired fitted term {family['signal']}",
                                f"split {family['split']}",
                                view,
                                row["label"],
                                f"{units}.",
                            )
                        ),
                        "effect": row[metric],
                        "effect_units": units,
                        "interval_low": row[interval][0],
                        "interval_high": row[interval][1],
                        "probability_positive": row[positive],
                        "standard_error": row["standard_errors"][index],
                        "sample_games": row["games"],
                        "sample_blocks": row["blocks"],
                        "season_start": min(seasons),
                        "season_end": max(seasons),
                    }
                )
    return cells


def _registry_batch(report: dict[str, Any], directory: Path) -> dict[str, Any]:
    cells = []
    families = []
    for family in report["families"]:
        cells.extend(_family_registry_batch(family, directory))
        families.append(family["family"])
    return {
        "source": (directory / "report.json").as_posix(),
        "classification": "unresolved_below_power",
        "league": "nfl",
        "families": families,
        "category": "schedule",
        "classification_evidence": (
            "Retrospective diagnostic; no untouched selection test or certified historical "
            "feature availability. In-sample rows are descriptive only. No research-closing "
            "ground is established."
        ),
        "notes": " ".join(report["limitations"]),
        "plain_summary": (
            "How the served model's own fitted terms change a probability when added to the "
            "same model and market inputs, across several ways of splitting the games."
        ),
        "cells": cells,
    }


def build_signal_atlas(artifacts_root: Path, registry_root: Path) -> Path:
    declaration_path = registry_root / "conditional_signal_atlas.json"
    declaration = _read(declaration_path)
    split_path = registry_root / "split_library.json"
    split = _read(split_path)
    if declaration["full_features"] != list(FIT_FEATURES):
        raise ValueError("Signal atlas full features differ from the active fitter")
    for family in declaration["families"]:
        if family["signal"] not in FIT_FEATURES:
            raise ValueError(f"Signal atlas family declares an unfitted signal: {family['signal']}")
        if family["cells"][0] != "overall":
            raise ValueError("Signal atlas declared cells must start with overall")
        if family["cells"][1:] != split["splits"][family["split"]]["cells"]:
            raise ValueError("Signal atlas declared cells differ from the split library")
    source, metadata = _source(artifacts_root)
    frame = pd.read_parquet(source / "per_game.parquet").reset_index(drop=True)
    required = [
        "model_probability",
        "home_covered",
        *FIT_FEATURES,
        "season",
        "week",
        "margin_vs_open",
        "tue_open_home_spread",
    ]
    if frame.empty or frame.game_id.isna().any() or frame.game_id.duplicated().any():
        raise ValueError("Signal atlas source must have one row per game")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Signal atlas source contains missing or nonfinite features")
    if not frame.home_covered.isin([0, 1]).all() or (frame.margin_vs_open == 0).any():
        raise ValueError("Signal atlas source includes ungraded games or pushes")
    if not (frame.home_covered.astype(bool) == (frame.margin_vs_open > 0)).all():
        raise ValueError("Signal atlas labels disagree with the opener grade")
    if (
        not frame.week.between(1, 18).all()
        or not frame.model_probability.between(0, 1, inclusive="neither").all()
    ):
        raise ValueError("Signal atlas source has invalid weeks or base probabilities")
    if len(frame) != metadata["graded_games"]:
        raise ValueError("Signal atlas source population differs from its metadata")
    signals = sorted({family["signal"] for family in declaration["families"]})
    pairs_by_signal: dict[str, pd.DataFrame] = {}
    coefficients: list[dict[str, Any]] = []
    for signal in signals:
        signal_declaration = {
            "evaluations": declaration["evaluations"],
            "full_features": declaration["full_features"],
            "reduced_features": [name for name in FIT_FEATURES if name != signal],
            "chronological_minimum_training_seasons": declaration[
                "chronological_minimum_training_seasons"
            ],
        }
        signal_pairs, signal_coefficients = _paired_predictions(frame, signal_declaration)
        for entry in signal_coefficients:
            entry["signal"] = signal
        coefficients.extend(signal_coefficients)
        pairs_by_signal[signal] = signal_pairs
    families_report: list[dict[str, Any]] = []
    for family in declaration["families"]:
        signal = family["signal"]
        pairs = pairs_by_signal[signal]
        evaluations: dict[str, list[dict[str, Any]]] = {}
        for evaluation in declaration["evaluations"]:
            eligible = pairs.dropna(subset=[f"{evaluation}_full", f"{evaluation}_reduced"])
            masks = _split_masks(eligible, family["split"])
            evaluations[evaluation] = [
                _cell(eligible.loc[masks[cell]], evaluation, cell, declaration)
                for cell in family["cells"]
            ]
        look_inventory = {
            "arm_cell_evaluation_combinations": len(declaration["arms"])
            * len(family["cells"])
            * len(declaration["evaluations"]),
            "paired_metric_comparisons": len(family["cells"]) * len(declaration["evaluations"]) * 3,
            "fitted_models": len([c for c in coefficients if c["signal"] == signal]),
            "reliability_bins_declared_per_arm": len(declaration["reliability_edges"]) - 1,
            "year_breakdowns": sum(
                len(cell["seasons"]) for cells in evaluations.values() for cell in cells
            ),
            "interpretation": (
                "Overlapping diagnostic looks, not independent confirmations; no selected best "
                "cell or multiplicity-adjusted claim."
            ),
        }
        families_report.append(
            {
                "family": family["family"],
                "signal": signal,
                "split": family["split"],
                "signal_label": SIGNAL_LABELS.get(signal, signal),
                "split_label": SPLIT_LABELS.get(family["split"], family["split"]),
                "cells": family["cells"],
                "reduced_features": [name for name in FIT_FEATURES if name != signal],
                "evaluations": evaluations,
                "look_count": look_inventory["arm_cell_evaluation_combinations"],
                "look_inventory": look_inventory,
                "in_sample_gap": {
                    metric: evaluations["in_sample"][0][metric]
                    - evaluations["out_of_season"][0][metric]
                    for metric in (
                        "accuracy_delta_points",
                        "brier_improvement",
                        "log_loss_improvement",
                    )
                },
            }
        )
    base_columns = [
        "game_id",
        "season",
        "week",
        "home_covered",
        "model_probability",
        "tue_open_home_spread",
    ]
    combined = pairs_by_signal[signals[0]][base_columns].copy()
    for evaluation in declaration["evaluations"]:
        combined[f"{evaluation}_full"] = pairs_by_signal[signals[0]][f"{evaluation}_full"]
    for signal in signals:
        for evaluation in declaration["evaluations"]:
            combined[f"{evaluation}_reduced__{signal}"] = pairs_by_signal[signal][
                f"{evaluation}_reduced"
            ]
    directory = artifacts_root / "signal_atlas" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    report = {
        "schema_version": 2,
        "active_model_id": metadata["active_model_id"],
        "source_artifact": source.relative_to(artifacts_root.resolve()).as_posix(),
        "source_hashes": {
            name: _hash(source / name)
            for name in ("per_game.parquet", "metadata.json", "coefficients.json")
        },
        "declaration_sha256": _hash(declaration_path),
        "split_library_sha256": _hash(split_path),
        "bootstrap_draws": declaration["bootstrap_draws"],
        "seed": declaration["seed"],
        "interval_level": declaration["interval_level"],
        "look_count": sum(entry["look_count"] for entry in families_report),
        "families": families_report,
        "excluded": {
            "pushes": metadata["pushes_dropped"],
            "ungraded": metadata["ungraded_dropped"],
            "chronological_warmup_games": int(
                pairs_by_signal[signals[0]].chronological_full.isna().sum()
            ),
        },
        "availability_certified": False,
        "serving": False,
        "limitations": [
            (
                "This is a retrospective comparison. The saved inputs lack per-game evidence "
                "proving when every feature became available; historical pregame timing is not "
                "certified."
            ),
            (
                "The signals were chosen using these years. Neither view is an untouched test "
                "of a new idea."
            ),
            (
                "Leaving one year out can train on later years. The earlier-years-only view "
                "trains on at least two completed years and scores the next year."
            ),
            (
                "Both fits use the same games, model probability and available market movement; "
                "one also includes the pooled game situations. Grades use the opening spread, "
                "even when an input arrived later in the week."
            ),
            (
                "Intervals resample years and weeks together for both fits. They hold fitted "
                "predictions fixed, so they exclude feature-selection and refitting uncertainty."
            ),
            (
                "Chance of improvement is the share of resamples above zero, counting ties as "
                "half. It is not a probability that a betting edge exists."
            ),
            (
                "The exact decisive-game check assumes independent games; shared weeks and "
                "training data limit that assumption."
            ),
            (
                "The even-market reference assigns each side 50%. Its accuracy uses a home-side "
                "tie break; its probability scores are the useful comparison."
            ),
            (
                "All declared week groups are shown. These overlapping comparisons do not select "
                "a winning group or change the picks."
            ),
        ],
    }
    combined.to_parquet(directory / "per_game.parquet", index=False)
    _write(directory / "coefficients.json", coefficients)
    _write(directory / "declaration.json", declaration)
    report["saved_declaration_sha256"] = _hash(directory / "declaration.json")
    _write(directory / "report.json", report)
    _write(directory / "weak_signals_batch.json", _registry_batch(report, directory))
    _write(
        artifacts_root / ATLAS_POINTER,
        {
            "artifact": directory.relative_to(artifacts_root).as_posix(),
            "files": {
                name: _hash(directory / name)
                for name in (
                    "report.json",
                    "per_game.parquet",
                    "coefficients.json",
                    "declaration.json",
                    "weak_signals_batch.json",
                )
            },
        },
    )
    return directory

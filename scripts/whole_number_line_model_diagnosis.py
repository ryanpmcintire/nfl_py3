from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlay_union_line_shape as q1  # noqa: E402

from nfl_ats.clv import CLOSE_LABEL_PRIORITY, cached_pairing_table  # noqa: E402
from nfl_ats.discrete_margin_mapping import (  # noqa: E402
    ACCURACY_POINT_CEILING,
    floor_degenerate_cell,
)
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.key_line_pick_read import KEY_LINE_ATOMS  # noqa: E402
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact  # noqa: E402
from nfl_ats.public_board import find_matching_opener_evaluation  # noqa: E402

OUT = REPO / "artifacts/research/laneQ2"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
FAMILY = "whole_number_line_model_diagnosis_v1"
PREFIX = "wn"
SEED = 20260817
DRAWS = 20_000
PERMUTATION_SEED = 20260911
PERMUTATIONS = 2_000
TOLERANCE = 1e-9
SERVED_PICK = "pick_home_at_open_probability_rule"
RAW_PICK = "pick_home_at_open_probability_rule_raw"
SIGN_PICK = "pick_home_at_open"
PREDECLARATION = REPO / "docs/whole_number_line_model_diagnosis.md"

DISCOUNT = (
    "Mechanism diagnostics for a weakness lane P1 noticed POST-HOC and lane Q1 measured on the "
    "same mined 1,537-game Tuesday-opener archive lanes K, T, H, S, V, C2, P1 and Q1 were "
    "selected on. The mechanism tests are predeclared in "
    "docs/whole_number_line_model_diagnosis.md, frozen before measurement; the observation they "
    "explain was not, and this lane cannot un-see it. Descriptive reuse of a mined era, not "
    "independent confirmation; no rotation window spent."
)

DIAGNOSTIC_NOTE = (
    " CONDITIONS ON THE REALISED RESULT or on information that does not exist before kickoff, "
    "and is a diagnostic only: this row can never become a rule."
)


def fixed(value: float) -> str:
    return f"{float(value):.12f}"


def table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    stamp_sidecar(path)


def correctness(frame: pd.DataFrame, pick: str, truth_column: str = "margin_vs_open") -> np.ndarray:
    scored = (
        q1.graded(frame) if truth_column == "margin_vs_open" else graded_on(frame, truth_column)
    )
    truth = scored[truth_column].gt(0).to_numpy()
    return (scored[pick].to_numpy(dtype=bool) == truth).astype(float)


def graded_on(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    keep = frame[column].notna() & frame[column].ne(0)
    return frame.loc[keep].reset_index(drop=True)


def shape_gap(
    frame: pd.DataFrame, left: np.ndarray, right: np.ndarray, pick: str = SERVED_PICK
) -> dict:
    return q1.gap_effect(frame, left, right, values=correctness(frame, pick))


def accuracy_row(frame: pd.DataFrame, mask: np.ndarray, pick: str = SERVED_PICK) -> dict:
    subset = frame.loc[np.asarray(mask, dtype=bool)]
    scored = q1.graded(subset)
    if scored.empty:
        return {"n": 0, "accuracy": float("nan")}
    truth = scored.margin_vs_open.gt(0).to_numpy()
    picks = scored[pick].to_numpy(dtype=bool)
    return {
        "n": len(scored),
        "games": len(subset),
        "pushes": int(len(subset) - len(scored)),
        "accuracy": float((picks == truth).mean()),
        "home_pick_rate": float(picks.mean()),
        "home_cover_rate": float(truth.mean()),
    }


def standardised_gap(
    frame: pd.DataFrame,
    strata: np.ndarray,
    correct: np.ndarray,
    minimum: int,
    draws: int = 2_000,
) -> dict:
    scored = q1.graded(frame)
    keep = (frame.margin_vs_open.notna() & frame.margin_vs_open.ne(0)).to_numpy()
    stratum = np.asarray(strata, dtype=object)[keep]
    shape = scored.shape_label.to_numpy()
    delta = np.asarray(correct, dtype=float)
    names = sorted({str(value) for value in stratum})
    eligible = [
        name
        for name in names
        if ((stratum == name) & (shape == "half")).sum() >= minimum
        and ((stratum == name) & (shape == "whole")).sum() >= minimum
    ]
    if not eligible:
        return {"eligible_strata": [], "delta": float("nan")}
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    width = len(eligible)
    sums = np.zeros((len(groups), width, 2), dtype=float)
    counts = np.zeros((len(groups), width, 2), dtype=float)
    for block, index in enumerate(groups):
        for column, name in enumerate(eligible):
            inside = stratum[index] == name
            for side, label in enumerate(("half", "whole")):
                pick = inside & (shape[index] == label)
                sums[block, column, side] = delta[index][pick].sum()
                counts[block, column, side] = pick.sum()
    weights = counts.sum(axis=0).sum(axis=1)
    total_sums, total_counts = sums.sum(axis=0), counts.sum(axis=0)
    per_stratum = 100 * (
        total_sums[:, 1] / total_counts[:, 1] - total_sums[:, 0] / total_counts[:, 0]
    )
    point = float(np.average(per_stratum, weights=weights))
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (draws, len(groups)))
    values = np.empty(draws, dtype=float)
    flat_sums = sums.reshape(len(groups), -1)
    flat_counts = counts.reshape(len(groups), -1)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        block = selected[start:stop]
        drawn_sums = flat_sums[block].sum(axis=1).reshape(-1, width, 2)
        drawn_counts = flat_counts[block].sum(axis=1).reshape(-1, width, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            gaps = 100 * (
                drawn_sums[:, :, 1] / drawn_counts[:, :, 1]
                - drawn_sums[:, :, 0] / drawn_counts[:, :, 0]
            )
        usable = np.isfinite(gaps)
        weighted = np.where(usable, gaps * weights, 0.0).sum(axis=1)
        mass = np.where(usable, weights, 0.0).sum(axis=1)
        values[start:stop] = np.where(mass > 0, weighted / np.where(mass > 0, mass, 1.0), np.nan)
    values = values[np.isfinite(values)]
    return {
        "delta": point,
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "probability_positive": float(probability_positive_from_draws(values)),
        "standard_error": float(values.std(ddof=1)),
        "n": int(counts.sum()),
        "weeks": len(groups),
        "eligible_strata": eligible,
        "per_stratum_gap": {name: float(per_stratum[index]) for index, name in enumerate(eligible)},
        "stratum_weight": {name: float(weights[index]) for index, name in enumerate(eligible)},
    }


def side_matched_excess(frame: pd.DataFrame, pick: str = SERVED_PICK) -> dict:
    scored = q1.graded(frame)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    picks = scored[pick].to_numpy(dtype=bool)
    shape = scored.shape_label.to_numpy()
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    blocks = np.zeros((len(groups), 2, 4), dtype=float)
    for block, index in enumerate(groups):
        for side, label in enumerate(("half", "whole")):
            inside = shape[index] == label
            rows = index[inside]
            blocks[block, side, 0] = (picks[rows] == truth[rows]).sum()
            blocks[block, side, 1] = len(rows)
            blocks[block, side, 2] = picks[rows].sum()
            blocks[block, side, 3] = truth[rows].sum()

    def excess(matrix: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            count = matrix[..., 1]
            accuracy = matrix[..., 0] / count
            home_rate = matrix[..., 2] / count
            cover = matrix[..., 3] / count
            null = home_rate * cover + (1.0 - home_rate) * (1.0 - cover)
        return 100.0 * (accuracy - null)

    observed = excess(blocks.sum(axis=0))
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (DRAWS, len(groups)))
    drawn = excess(blocks[selected].sum(axis=1))
    draws = drawn[:, 1] - drawn[:, 0]
    draws = draws[np.isfinite(draws)]
    totals = blocks.sum(axis=0)
    permutation: dict[str, float] = {}
    generator = np.random.default_rng(PERMUTATION_SEED)
    for label in ("half", "whole"):
        inside = shape == label
        local_truth = truth[inside]
        wanted = int(picks[inside].sum())
        size = int(inside.sum())
        scores = np.empty(PERMUTATIONS, dtype=float)
        for draw in range(PERMUTATIONS):
            assignment = np.zeros(size, dtype=bool)
            assignment[generator.choice(size, size=wanted, replace=False)] = True
            scores[draw] = (assignment == local_truth).mean()
        permutation[f"{label}_null_mean"] = float(100.0 * scores.mean())
        permutation[f"{label}_null_standard_deviation"] = float(100.0 * scores.std(ddof=1))
        permutation[f"{label}_observed"] = float(100.0 * (picks[inside] == local_truth).mean())
    return {
        "delta": float(observed[1] - observed[0]),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "standard_error": float(draws.std(ddof=1)),
        "n": int(totals[:, 1].sum()),
        "weeks": len(groups),
        "half_excess": float(observed[0]),
        "whole_excess": float(observed[1]),
        "analytic_null": {
            "half": float(
                100.0
                * (
                    (totals[0, 2] / totals[0, 1]) * (totals[0, 3] / totals[0, 1])
                    + (1 - totals[0, 2] / totals[0, 1]) * (1 - totals[0, 3] / totals[0, 1])
                )
            ),
            "whole": float(
                100.0
                * (
                    (totals[1, 2] / totals[1, 1]) * (totals[1, 3] / totals[1, 1])
                    + (1 - totals[1, 2] / totals[1, 1]) * (1 - totals[1, 3] / totals[1, 1])
                )
            ),
        },
        "permutation_null": permutation,
        "permutations": PERMUTATIONS,
    }


def reliability(frame: pd.DataFrame, edges: np.ndarray) -> dict:
    scored = q1.graded(frame)
    probability = scored.home_cover_probability_at_open.to_numpy(dtype=float)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    picks = scored[SERVED_PICK].to_numpy(dtype=bool)
    index = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for position in range(len(edges) - 1):
        inside = index == position
        if not inside.any():
            continue
        bins.append(
            {
                "bin": f"{edges[position]:.3f}-{edges[position + 1]:.3f}",
                "n": int(inside.sum()),
                "mean_stated_probability": float(probability[inside].mean()),
                "realised_home_cover_rate": float(truth[inside].mean()),
                "accuracy": float((picks[inside] == truth[inside]).mean()),
            }
        )
    confidence = np.maximum(probability, 1.0 - probability)
    return {
        "n": len(scored),
        "brier": float(((probability - truth.astype(float)) ** 2).mean()),
        "mean_stated_confidence": float(confidence.mean()),
        "accuracy": float((picks == truth).mean()),
        "confidence_minus_accuracy": float(confidence.mean() - (picks == truth).mean()),
        "bins": bins,
    }


def spearman_brown(value: float) -> float:
    if not np.isfinite(value) or value <= -1.0:
        return float("nan")
    return float(2.0 * value / (1.0 + value))


def block_split_half(frame: pd.DataFrame) -> dict:
    scored = q1.graded(frame).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (scored[SERVED_PICK].to_numpy(dtype=bool) == truth).astype(float)
    shape = scored.shape_label.to_numpy()
    pairs: list[tuple[float, float]] = []
    for _, index in scored.groupby(["season", "week"], sort=True).indices.items():
        order = np.arange(len(index))
        halves: list[float] = []
        usable = True
        for parity in (0, 1):
            side = index[order % 2 == parity]
            whole = delta[side][shape[side] == "whole"]
            half = delta[side][shape[side] == "half"]
            if len(whole) == 0 or len(half) == 0:
                usable = False
                break
            halves.append(100.0 * (whole.mean() - half.mean()))
        if usable:
            pairs.append((halves[0], halves[1]))
    if len(pairs) < 3:
        return {"blocks": len(pairs), "pearson": float("nan")}
    left = np.array([pair[0] for pair in pairs])
    right = np.array([pair[1] for pair in pairs])
    pearson = float(stats.pearsonr(left, right).statistic)
    spearman = float(stats.spearmanr(left, right).statistic)
    rng = np.random.default_rng(SEED)
    boot = np.array(
        [
            stats.pearsonr(left[pick], right[pick]).statistic
            for pick in rng.integers(0, len(pairs), (1000, len(pairs)))
        ]
    )
    boot = boot[np.isfinite(boot)]
    return {
        "blocks": len(pairs),
        "pearson": pearson,
        "pearson_spearman_brown": spearman_brown(pearson),
        "spearman": spearman,
        "pearson_lower": float(np.quantile(boot, 0.025)),
        "pearson_upper": float(np.quantile(boot, 0.975)),
    }


def season_split_half(frame: pd.DataFrame) -> dict:
    scored = q1.graded(frame)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (scored[SERVED_PICK].to_numpy(dtype=bool) == truth).astype(float)
    shape = scored.shape_label.to_numpy()
    season = scored.season.to_numpy()
    week = scored.week.to_numpy()
    pairs: list[tuple[int, float, float]] = []
    for value in sorted(set(season.tolist())):
        halves: list[float] = []
        usable = True
        for parity in (1, 0):
            side = (season == value) & (week % 2 == parity)
            whole = delta[side & (shape == "whole")]
            half = delta[side & (shape == "half")]
            if len(whole) == 0 or len(half) == 0:
                usable = False
                break
            halves.append(100.0 * (whole.mean() - half.mean()))
        if usable:
            pairs.append((int(value), halves[0], halves[1]))
    if len(pairs) < 3:
        return {"seasons": len(pairs)}
    left = np.array([pair[1] for pair in pairs])
    right = np.array([pair[2] for pair in pairs])
    return {
        "seasons": len(pairs),
        "per_season": [
            {"season": pair[0], "odd_week_gap": pair[1], "even_week_gap": pair[2]} for pair in pairs
        ],
        "spearman": float(stats.spearmanr(left, right).statistic),
        "pearson": float(stats.pearsonr(left, right).statistic),
        "pearson_spearman_brown": spearman_brown(float(stats.pearsonr(left, right).statistic)),
    }


def dispersion_join(frame: pd.DataFrame) -> pd.DataFrame:
    schedule = frame[["game_id", "season", "week"]].drop_duplicates("game_id")
    pairing = cached_pairing_table(
        REPO / "data/market/raw", labels=("tue_open", *CLOSE_LABEL_PRIORITY), schedule=schedule
    )
    opener = pairing.loc[pairing.decision_label.eq("tue_open")][
        ["game_id", "home_spread", "spread_books", "spread_min", "spread_max", "spread_std"]
    ]
    merged = frame.merge(opener, on="game_id", how="left", validate="one_to_one")
    if merged.home_spread.isna().any():
        raise ValueError("STOP: archive games without a Tuesday opener consensus row")
    if float((merged.home_spread - merged.tue_open_home_spread).abs().max()) > TOLERANCE:
        raise ValueError("STOP: the joined consensus line disagrees with the archive decision line")
    if (
        int((merged.spread_books.astype(float) - merged.opener_books.astype(float)).abs().max())
        != 0
    ):
        raise ValueError("STOP: the joined book count disagrees with the archive book count")
    return merged


def feature_line(frame: pd.DataFrame, active: dict) -> pd.DataFrame:
    digest = sha256_file(FEATURES)
    if digest != active["feature_table_sha256"]:
        raise ValueError("STOP: the feature table digest does not match the active model")
    features = pd.read_parquet(FEATURES, columns=["game_id", "spread_line"])
    merged = frame.merge(
        features.rename(columns={"spread_line": "feature_table_spread_line"}),
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    if merged.feature_table_spread_line.isna().any():
        raise ValueError("STOP: archive games missing from the feature table")
    return merged


def magnitude_buckets(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    integer = np.minimum(np.floor(size), 14.0)
    fine = np.array([f"{int(value):02d}" for value in integer], dtype=object)
    edges = [0.0, 2.0, 4.0, 7.0, 10.0, np.inf]
    names = ["0-2", "2-4", "4-7", "7-10", "10+"]
    coarse = np.full(len(size), "", dtype=object)
    for index, name in enumerate(names):
        coarse[(size >= edges[index]) & (size < edges[index + 1])] = name
    return fine, coarse


def bucket_labels(values: np.ndarray, edges: list[float], names: list[str]) -> np.ndarray:
    labels = np.full(len(values), names[-1], dtype=object)
    for index in range(len(names) - 1):
        labels[(values >= edges[index]) & (values < edges[index + 1])] = names[index]
    return labels


def run() -> None:
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(REPO / "artifacts", active)
    if match is None:
        raise ValueError("STOP: no opener evaluation matches the active model")
    archive_path = match[1]
    metadata = json.loads((archive_path / "metadata.json").read_text())
    if metadata["active_model_id"] != active["model_id"]:
        raise ValueError("STOP: the matched evaluation belongs to a different model")
    if metadata["probability_method"] != "gaussian_median":
        raise ValueError("STOP: the archive was not graded on the served smooth read")
    frame = pd.read_parquet(archive_path / "per_game.parquet")
    if not frame[SERVED_PICK].eq(frame.home_cover_probability_at_open.ge(0.5)).all():
        raise ValueError("STOP: the served pick is not the served probability rule")

    frame = dispersion_join(frame)
    frame = feature_line(frame, active)
    frame["shape_label"] = q1.shape_labels(frame.tue_open_home_spread)
    frame["line_size"] = frame.tue_open_home_spread.abs()
    size = frame.line_size.to_numpy(dtype=float)
    fine, coarse = magnitude_buckets(size)
    frame["magnitude_fine"] = fine
    frame["magnitude_coarse"] = coarse
    frame["book_range"] = (frame.spread_max - frame.spread_min).astype(float)
    frame["unanimous"] = frame.book_range.abs().lt(TOLERANCE)
    frame["line_shift"] = frame.tue_open_home_spread - frame.feature_table_spread_line
    frame["edge"] = frame.residual_at_open.abs()
    frame["point_error"] = frame.margin_vs_open - frame.residual_at_open_served
    frame["dispersion_stratum"] = np.where(
        frame.unanimous.to_numpy(),
        "unanimous",
        np.where(frame.book_range.to_numpy() <= 0.5, "range_to_half", "range_over_half"),
    )
    frame["shift_stratum"] = bucket_labels(
        frame.line_shift.abs().to_numpy(dtype=float),
        [0.0, 0.25, 1.0, 2.0, np.inf],
        ["shift_none", "shift_small", "shift_medium", "shift_large"],
    )
    frame["edge_stratum"] = bucket_labels(
        frame.edge.to_numpy(dtype=float),
        [0.0, 0.5, 1.0, 2.0, np.inf],
        ["edge_under_half", "edge_half_to_one", "edge_one_to_two", "edge_over_two"],
    )

    shapes = {name: (frame.shape_label == name).to_numpy() for name in ("half", "whole", "quarter")}
    shapes["off_pool"] = ~shapes["half"]
    shapes["every_line"] = np.ones(len(frame), dtype=bool)
    half, whole = shapes["half"], shapes["whole"]

    cells: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    diagnostics["D0_shape_counts"] = {
        name: accuracy_row(frame, mask) for name, mask in shapes.items()
    }
    for label, left in (("whole", whole), ("quarter", shapes["quarter"]), ("off_pool", ~half)):
        cells[f"{PREFIX}_d0_gap_{label}_minus_half"] = {
            **shape_gap(frame, left, half),
            "hypothesis": "D0",
            "domain": f"{label}_minus_half",
            "units": "accuracy_points",
        }
    raw_gap = cells[f"{PREFIX}_d0_gap_whole_minus_half"]["delta"]

    diagnostics["D1a_dispersion_by_shape"] = {
        name: {
            "mean_books": float(frame.loc[mask].opener_books.astype(float).mean()),
            "median_books": float(frame.loc[mask].opener_books.astype(float).median()),
            "mean_spread_std": float(frame.loc[mask].spread_std.mean()),
            "median_spread_std": float(frame.loc[mask].spread_std.median()),
            "mean_book_range": float(frame.loc[mask].book_range.mean()),
            "unanimous_share": float(frame.loc[mask].unanimous.mean()),
            "unanimous_games": int(frame.loc[mask].unanimous.sum()),
        }
        for name, mask in shapes.items()
    }
    for label, restrict in (
        ("unanimous", frame.unanimous.to_numpy()),
        ("books_disagreed", ~frame.unanimous.to_numpy()),
    ):
        subset = frame.loc[restrict]
        cells[f"{PREFIX}_d1b_gap_{label}"] = {
            **shape_gap(
                subset,
                (subset.shape_label == "whole").to_numpy(),
                (subset.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D1b",
            "domain": f"whole_minus_half_{label}",
            "units": "accuracy_points",
        }
        diagnostics[f"D1b_{label}_accuracy"] = {
            name: accuracy_row(subset, (subset.shape_label == name).to_numpy())
            for name in ("half", "whole")
        }
    diagnostics["D1c_standardised_gap_dispersion"] = standardised_gap(
        frame, frame.dispersion_stratum.to_numpy(), correctness(frame, SERVED_PICK), 20
    )
    cells[f"{PREFIX}_d1c_gap_standardised_by_dispersion"] = {
        **{
            key: value
            for key, value in diagnostics["D1c_standardised_gap_dispersion"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D1c",
        "domain": "whole_minus_half_standardised_by_book_disagreement",
        "units": "accuracy_points",
    }
    agreed = frame.unanimous.to_numpy()
    for name, mask in (("half", half), ("whole", whole), ("every_line", shapes["every_line"])):
        cells[f"{PREFIX}_d1c_{name}_books_disagreed_minus_unanimous"] = {
            **shape_gap(frame, mask & ~agreed, mask & agreed),
            "hypothesis": "D1c",
            "domain": f"{name}_books_disagreed_minus_unanimous",
            "units": "accuracy_points",
        }

    atoms = np.asarray(KEY_LINE_ATOMS, dtype=float)
    on_key = np.any(np.abs(size[:, None] - atoms[None, :]) < TOLERANCE, axis=1)
    wide_atoms = np.asarray([3.0, 7.0, 10.0, 14.0], dtype=float)
    on_wide_key = np.any(np.abs(size[:, None] - wide_atoms[None, :]) < TOLERANCE, axis=1)
    hook_atoms = np.asarray([2.5, 3.5, 6.5, 7.5], dtype=float)
    on_hook = np.any(np.abs(size[:, None] - hook_atoms[None, :]) < TOLERANCE, axis=1)
    cells[f"{PREFIX}_d2a_whole_on_3_or_7_minus_other_whole"] = {
        **shape_gap(frame, whole & on_key, whole & ~on_key),
        "hypothesis": "D2a",
        "domain": "whole_on_3_or_7_minus_other_whole",
        "units": "accuracy_points",
    }
    cells[f"{PREFIX}_d2a_whole_on_a_key_number_minus_other_whole"] = {
        **shape_gap(frame, whole & on_wide_key, whole & ~on_wide_key),
        "hypothesis": "D2a",
        "domain": "whole_on_key_number_minus_other_whole",
        "units": "accuracy_points",
    }
    cells[f"{PREFIX}_d2a_half_beside_3_or_7_minus_other_half"] = {
        **shape_gap(frame, half & on_hook, half & ~on_hook),
        "hypothesis": "D2a",
        "domain": "half_beside_3_or_7_minus_other_half",
        "units": "accuracy_points",
    }
    cells[f"{PREFIX}_d2a_gap_away_from_the_key_numbers"] = {
        **shape_gap(frame, whole & ~on_wide_key, half & ~on_hook),
        "hypothesis": "D2a",
        "domain": "whole_minus_half_away_from_the_key_numbers",
        "units": "accuracy_points",
    }
    diagnostics["D2a_accuracy_by_line_size"] = {
        name: {
            str(value): accuracy_row(frame, mask & (np.abs(size - value) < TOLERANCE))
            for value in sorted({float(v) for v in size[mask]})
            if int((mask & (np.abs(size - value) < TOLERANCE)).sum()) >= 10
        }
        for name, mask in (("half", half), ("whole", whole))
    }

    diagnostics["D2b_push_counts"] = {
        name: {
            "games": int(mask.sum()),
            "pushes": int((mask & frame.margin_vs_open.eq(0).to_numpy()).sum()),
        }
        for name, mask in shapes.items()
    }
    if diagnostics["D2b_push_counts"]["half"]["pushes"] != 0:
        raise ValueError("STOP: a half-point line recorded a push")
    common = frame.loc[frame.margin_vs_open.abs().ge(1.0).to_numpy()]
    cells[f"{PREFIX}_d2b_gap_decided_by_a_point"] = {
        **shape_gap(
            common,
            (common.shape_label == "whole").to_numpy(),
            (common.shape_label == "half").to_numpy(),
        ),
        "hypothesis": "D2b",
        "domain": "whole_minus_half_margin_at_least_one_point",
        "units": "accuracy_points",
    }
    for direction, shift in (("home_favourable", 0.5), ("away_favourable", -0.5)):
        shifted = frame.copy()
        adjust = shifted.shape_label.eq("whole").to_numpy()
        shifted.loc[adjust, "margin_vs_open"] = shifted.loc[adjust, "margin_vs_open"] + shift
        cells[f"{PREFIX}_d2b_gap_pushes_reinstated_{direction}"] = {
            **shape_gap(
                shifted,
                (shifted.shape_label == "whole").to_numpy(),
                (shifted.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D2b",
            "domain": f"whole_minus_half_pushes_reinstated_{direction}",
            "units": "accuracy_points",
        }

    probability = q1.graded(frame).home_cover_probability_at_open.to_numpy(dtype=float)
    edges = np.unique(np.quantile(probability, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    edges[0], edges[-1] = 0.0, 1.0
    diagnostics["D2c_reliability_by_shape"] = {
        name: reliability(frame.loc[mask], edges) for name, mask in shapes.items()
    }

    diagnostics["D2d_served_atom_applicability"] = {
        "games_on_a_served_atom": int(on_key.sum()),
        "share_of_whole_domain": float((on_key & whole).sum() / max(int(whole.sum()), 1)),
        "share_of_half_domain": float((on_key & half).sum()),
        "accuracy_on_a_served_atom": accuracy_row(frame, on_key),
    }

    diagnostics["D2e_magnitude_by_shape"] = {
        name: {
            "mean_line": float(frame.loc[mask].line_size.mean()),
            "median_line": float(frame.loc[mask].line_size.median()),
            "coarse": {
                str(key): int(count)
                for key, count in frame.loc[mask]
                .magnitude_coarse.value_counts()
                .sort_index()
                .items()
            },
        }
        for name, mask in shapes.items()
    }
    for label, strata, minimum in (
        ("fine", frame.magnitude_fine.to_numpy(), 20),
        ("coarse", frame.magnitude_coarse.to_numpy(), 20),
    ):
        diagnostics[f"D2e_standardised_gap_magnitude_{label}"] = standardised_gap(
            frame, strata, correctness(frame, SERVED_PICK), minimum
        )
    cells[f"{PREFIX}_d2e_gap_standardised_by_line_size"] = {
        **{
            key: value
            for key, value in diagnostics["D2e_standardised_gap_magnitude_fine"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D2e",
        "domain": "whole_minus_half_standardised_by_line_size",
        "units": "accuracy_points",
    }
    for bucket, group in frame.groupby("magnitude_coarse"):
        for name in ("half", "whole"):
            subset = group.loc[group.shape_label == name]
            if len(q1.graded(subset)) < 20:
                continue
            diagnostics.setdefault("D2e_accuracy_by_bucket", {})[f"{name}_{bucket}"] = accuracy_row(
                subset, np.ones(len(subset), dtype=bool)
            )

    scored = q1.graded(frame)
    diagnostics["D3a_point_error_by_shape"] = {
        name: {
            "n": int((scored.shape_label == name).sum()),
            "mean_error": float(scored.loc[scored.shape_label == name].point_error.mean()),
            "median_error": float(scored.loc[scored.shape_label == name].point_error.median()),
            "standard_deviation": float(
                scored.loc[scored.shape_label == name].point_error.std(ddof=1)
            ),
            "mean_absolute_error": float(
                scored.loc[scored.shape_label == name].point_error.abs().mean()
            ),
        }
        for name in ("half", "whole", "quarter")
    }
    diagnostics["D3a_error_split_into_line_and_model"] = {
        name: {
            "mean_realised_margin_vs_opener": float(
                scored.loc[scored.shape_label == name].margin_vs_open.mean()
            ),
            "mean_model_residual_served": float(
                scored.loc[scored.shape_label == name].residual_at_open_served.mean()
            ),
            "mean_model_residual_unoffset": float(
                scored.loc[scored.shape_label == name].residual_at_open.mean()
            ),
        }
        for name in ("half", "whole", "quarter")
    }
    diagnostics["D3a_point_error_by_shape_and_bucket"] = {
        f"{name}_{bucket}": {
            "n": len(group),
            "mean_error": float(group.point_error.mean()),
            "standard_deviation": float(group.point_error.std(ddof=1)),
        }
        for name in ("half", "whole")
        for bucket, group in scored.loc[scored.shape_label == name].groupby("magnitude_coarse")
        if len(group) >= 20
    }

    diagnostics["D3b_line_shift_by_shape"] = {
        name: {
            "mean_shift": float(frame.loc[mask].line_shift.mean()),
            "mean_absolute_shift": float(frame.loc[mask].line_shift.abs().mean()),
            "standard_deviation": float(frame.loc[mask].line_shift.std(ddof=1)),
            "share_identical": float(frame.loc[mask].line_shift.abs().lt(TOLERANCE).mean()),
            "strata": {
                str(key): int(count)
                for key, count in frame.loc[mask].shift_stratum.value_counts().sort_index().items()
            },
        }
        for name, mask in shapes.items()
    }
    diagnostics["D3b_what_the_training_line_is"] = {
        "share_equal_to_the_tuesday_opener": float(frame.line_shift.abs().lt(TOLERANCE).mean()),
        "share_equal_to_the_close": float(
            (frame.feature_table_spread_line - frame.close_home_spread).abs().lt(TOLERANCE).mean()
        ),
        "mean_absolute_distance_to_the_close": float(
            (frame.feature_table_spread_line - frame.close_home_spread).abs().mean()
        ),
        "mean_absolute_distance_to_the_opener": float(frame.line_shift.abs().mean()),
        "training_line_shape": {
            str(key): int(count)
            for key, count in pd.Series(q1.shape_labels(frame.feature_table_spread_line))
            .value_counts()
            .items()
        },
    }
    diagnostics["D3b_standardised_gap_line_shift"] = standardised_gap(
        frame, frame.shift_stratum.to_numpy(), correctness(frame, SERVED_PICK), 20
    )
    cells[f"{PREFIX}_d3b_gap_standardised_by_line_shift"] = {
        **{
            key: value
            for key, value in diagnostics["D3b_standardised_gap_line_shift"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D3b",
        "domain": "whole_minus_half_standardised_by_line_substitution",
        "units": "accuracy_points",
    }
    identical = frame.line_shift.abs().lt(TOLERANCE).to_numpy()
    for label, restrict in (("same_line", identical), ("moved_line", ~identical)):
        subset = frame.loc[restrict]
        if len(q1.graded(subset.loc[subset.shape_label == "whole"])) < 20:
            continue
        cells[f"{PREFIX}_d3b_gap_{label}"] = {
            **shape_gap(
                subset,
                (subset.shape_label == "whole").to_numpy(),
                (subset.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D3b",
            "domain": f"whole_minus_half_{label}",
            "units": "accuracy_points",
        }

    diagnostics["D3c_edge_by_shape"] = {
        name: {
            "mean_edge": float(frame.loc[mask].edge.mean()),
            "median_edge": float(frame.loc[mask].edge.median()),
            "share_under_half_a_point": float(frame.loc[mask].edge.lt(0.5).mean()),
            "share_under_a_point": float(frame.loc[mask].edge.lt(1.0).mean()),
            "share_under_two_points": float(frame.loc[mask].edge.lt(2.0).mean()),
        }
        for name, mask in shapes.items()
    }
    diagnostics["D3c_accuracy_by_edge_and_shape"] = {
        f"{name}_{stratum}": accuracy_row(frame, mask & (frame.edge_stratum == stratum).to_numpy())
        for name, mask in (("half", half), ("whole", whole))
        for stratum in ("edge_under_half", "edge_half_to_one", "edge_one_to_two", "edge_over_two")
    }
    diagnostics["D3c_standardised_gap_edge"] = standardised_gap(
        frame, frame.edge_stratum.to_numpy(), correctness(frame, SERVED_PICK), 20
    )
    cells[f"{PREFIX}_d3c_gap_standardised_by_model_edge"] = {
        **{
            key: value
            for key, value in diagnostics["D3c_standardised_gap_edge"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D3c",
        "domain": "whole_minus_half_standardised_by_model_edge",
        "units": "accuracy_points",
    }
    for stratum in ("edge_under_half", "edge_half_to_one", "edge_one_to_two", "edge_over_two"):
        subset = frame.loc[(frame.edge_stratum == stratum).to_numpy()]
        if len(q1.graded(subset.loc[subset.shape_label == "whole"])) < 20:
            continue
        cells[f"{PREFIX}_d3c_gap_{stratum}"] = {
            **shape_gap(
                subset,
                (subset.shape_label == "whole").to_numpy(),
                (subset.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D3c",
            "domain": f"whole_minus_half_{stratum}",
            "units": "accuracy_points",
        }

    moved = frame.open_move.ne(0)
    diagnostics["D3d_close_agreement_by_shape"] = {
        name: {
            "mean_absolute_open_move": float(frame.loc[mask].open_move.abs().mean()),
            "share_line_moved": float(frame.loc[mask].open_move.ne(0).mean()),
            "residual_agrees_with_market_move": float(
                (
                    frame.loc[mask & moved.to_numpy()].residual_at_open.gt(0)
                    == frame.loc[mask & moved.to_numpy()].open_move.gt(0)
                ).mean()
            ),
            "accuracy_at_close": accuracy_row(frame, mask, "pick_home_at_close_probability_rule"),
        }
        for name, mask in shapes.items()
    }

    diagnostics["D4_shape_share_by_season"] = {
        str(int(season)): {
            "games": len(group),
            "mean_books": float(group.opener_books.astype(float).mean()),
            **{
                name: float((group.shape_label == name).mean())
                for name in ("half", "whole", "quarter")
            },
        }
        for season, group in frame.groupby("season")
    }
    for season, group in frame.groupby("season"):
        cells[f"{PREFIX}_d4_gap_season_{int(season)}"] = {
            **shape_gap(
                group,
                (group.shape_label == "whole").to_numpy(),
                (group.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D4",
            "domain": f"whole_minus_half_season_{int(season)}",
            "season": int(season),
            "units": "accuracy_points",
        }
    diagnostics["D4_standardised_gap_season"] = standardised_gap(
        frame,
        np.array([str(int(value)) for value in frame.season], dtype=object),
        correctness(frame, SERVED_PICK),
        20,
    )
    cells[f"{PREFIX}_d4_gap_standardised_by_season"] = {
        **{
            key: value
            for key, value in diagnostics["D4_standardised_gap_season"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D4",
        "domain": "whole_minus_half_standardised_by_season",
        "units": "accuracy_points",
    }

    for label, seasons in (("odd", (2021, 2023, 2025)), ("even", (2020, 2022, 2024))):
        subset = frame.loc[frame.season.isin(seasons)]
        cells[f"{PREFIX}_d5a_gap_{label}_seasons"] = {
            **shape_gap(
                subset,
                (subset.shape_label == "whole").to_numpy(),
                (subset.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "D5a",
            "domain": f"whole_minus_half_{label}_seasons",
            "units": "accuracy_points",
        }
    diagnostics["D5b_season_split_half"] = season_split_half(frame)
    diagnostics["D5c_block_split_half"] = block_split_half(frame)

    diagnostics["D6_offset_by_shape"] = {
        name: {
            "games_with_an_offset": int(frame.loc[mask].home_side_offset_at_open.ne(0).sum()),
            "share_with_an_offset": float(frame.loc[mask].home_side_offset_at_open.ne(0).mean()),
            "mean_absolute_offset": float(frame.loc[mask].home_side_offset_at_open.abs().mean()),
            "picks_changed_by_the_offset": int(
                (frame.loc[mask][SERVED_PICK] != frame.loc[mask][RAW_PICK]).sum()
            ),
        }
        for name, mask in shapes.items()
    }
    for label, pick in (("raw_probability_pick", RAW_PICK), ("residual_sign_pick", SIGN_PICK)):
        cells[f"{PREFIX}_d6_gap_{label}"] = {
            **shape_gap(frame, whole, half, pick),
            "hypothesis": "D6",
            "domain": f"whole_minus_half_{label}",
            "units": "accuracy_points",
        }
    no_offset = frame.home_side_offset_at_open.eq(0).to_numpy()
    subset = frame.loc[no_offset]
    cells[f"{PREFIX}_d6_gap_offset_never_applied"] = {
        **shape_gap(
            subset,
            (subset.shape_label == "whole").to_numpy(),
            (subset.shape_label == "half").to_numpy(),
        ),
        "hypothesis": "D6",
        "domain": "whole_minus_half_games_the_offset_left_alone",
        "units": "accuracy_points",
    }

    excess = side_matched_excess(frame)
    diagnostics["D7_side_matched_null"] = excess
    cells[f"{PREFIX}_d7_gap_over_a_side_matched_null"] = {
        **{
            key: value
            for key, value in excess.items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "D7",
        "domain": "whole_minus_half_excess_over_a_side_matched_null",
        "units": "accuracy_points",
    }

    oracle = frame.loc[frame.oracle_correct_at_open.notna()].copy()
    oracle["oracle_pick_home"] = oracle.open_move.gt(0)
    cells[f"{PREFIX}_d8_gap_market_movement_reference"] = {
        **shape_gap(
            oracle,
            (oracle.shape_label == "whole").to_numpy(),
            (oracle.shape_label == "half").to_numpy(),
            "oracle_pick_home",
        ),
        "hypothesis": "D8",
        "domain": "whole_minus_half_market_movement_reference",
        "units": "accuracy_points",
        "category": "control",
        "diagnostic": True,
    }
    closing = frame.loc[frame.margin_vs_close.notna()].copy()
    closing["close_shape"] = q1.shape_labels(closing.close_home_spread)
    closing["margin_vs_open"] = closing.margin_vs_close
    cells[f"{PREFIX}_d8_gap_same_model_graded_at_the_close"] = {
        **shape_gap(
            closing,
            (closing.shape_label == "whole").to_numpy(),
            (closing.shape_label == "half").to_numpy(),
            "pick_home_at_close_probability_rule",
        ),
        "hypothesis": "D8",
        "domain": "whole_minus_half_by_opener_shape_graded_at_the_close",
        "units": "accuracy_points",
        "category": "control",
    }
    diagnostics["D8_reference_accuracy"] = {
        "movement_reference": {
            name: {
                "n": int((oracle.shape_label == name).sum()),
                "accuracy": float(
                    oracle.loc[oracle.shape_label == name].oracle_correct_at_open.mean()
                ),
            }
            for name in ("half", "whole", "quarter")
        },
        "model_at_close_by_opener_shape": {
            name: accuracy_row(
                closing,
                (closing.shape_label == name).to_numpy(),
                "pick_home_at_close_probability_rule",
            )
            for name in ("half", "whole", "quarter")
        },
        "close_line_shape_counts": {
            str(key): int(count) for key, count in closing.close_shape.value_counts().items()
        },
    }

    diagnostics["explanation_bands"] = {
        "raw_gap_whole_minus_half": raw_gap,
        "unanimous_books_only": cells[f"{PREFIX}_d1b_gap_unanimous"]["delta"],
        "books_disagreed_only": cells[f"{PREFIX}_d1b_gap_books_disagreed"]["delta"],
        "standardised_by_dispersion": cells[f"{PREFIX}_d1c_gap_standardised_by_dispersion"][
            "delta"
        ],
        "away_from_the_key_numbers": cells[f"{PREFIX}_d2a_gap_away_from_the_key_numbers"]["delta"],
        "decided_by_a_point": cells[f"{PREFIX}_d2b_gap_decided_by_a_point"]["delta"],
        "standardised_by_line_size": cells[f"{PREFIX}_d2e_gap_standardised_by_line_size"]["delta"],
        "standardised_by_line_shift": cells[f"{PREFIX}_d3b_gap_standardised_by_line_shift"][
            "delta"
        ],
        "standardised_by_model_edge": cells[f"{PREFIX}_d3c_gap_standardised_by_model_edge"][
            "delta"
        ],
        "standardised_by_season": cells[f"{PREFIX}_d4_gap_standardised_by_season"]["delta"],
        "raw_probability_pick": cells[f"{PREFIX}_d6_gap_raw_probability_pick"]["delta"],
        "residual_sign_pick": cells[f"{PREFIX}_d6_gap_residual_sign_pick"]["delta"],
        "excess_over_a_side_matched_null": cells[f"{PREFIX}_d7_gap_over_a_side_matched_null"][
            "delta"
        ],
        "market_movement_reference": cells[f"{PREFIX}_d8_gap_market_movement_reference"]["delta"],
        "same_model_at_the_close": cells[f"{PREFIX}_d8_gap_same_model_graded_at_the_close"][
            "delta"
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predeclaration.md").write_bytes(PREDECLARATION.read_bytes())
    table(
        frame[
            [
                "game_id",
                "season",
                "week",
                "tue_open_home_spread",
                "close_home_spread",
                "feature_table_spread_line",
                "line_size",
                "shape_label",
                "opener_books",
                "spread_min",
                "spread_max",
                "spread_std",
                "book_range",
                "unanimous",
                "dispersion_stratum",
                "magnitude_fine",
                "magnitude_coarse",
                "line_shift",
                "shift_stratum",
                "edge",
                "edge_stratum",
                "point_error",
                "open_move",
                "margin_vs_open",
                "margin_vs_close",
                "home_cover_probability_at_open",
                "home_side_offset_at_open",
                SERVED_PICK,
                RAW_PICK,
                SIGN_PICK,
                "oracle_correct_at_open",
            ]
        ],
        OUT / "shapes.parquet",
    )
    write_stamped_artifact(cells, OUT / "cells.json")
    write_stamped_artifact(diagnostics, OUT / "diagnostics.json")
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(FEATURES),
            "predeclaration_sha256": sha256_file(PREDECLARATION),
            "games": len(frame),
            "weeks": int(frame.groupby(["season", "week"]).ngroups),
            "served_pick": SERVED_PICK,
            "gates": (
                "active model matched by find_matching_opener_evaluation; feature digest equal to "
                "the active model's; consensus join equal to the archive decision line and book "
                "count on every row; served pick equal to the served probability rule"
            ),
        },
        OUT / "reproduction.json",
    )
    print(json.dumps(diagnostics["explanation_bands"], indent=2), flush=True)
    print(f"computed {len(cells)} cells", flush=True)


PLAIN_HEAD = (
    "This row is about the model's own pick, not about the weekly adjustments. It is how many "
    "more games in a hundred the model gets right when the opening spread was a whole number "
    "than when it ended in a half point, which is the only kind the pool ever posts. A negative "
    "number means the model is weaker on whole numbers."
)

PLAIN_TAIL = {
    "D0": "",
    "D1b": (
        " Measured separately for games where every sportsbook posted the same number and for "
        "games where they did not agree."
    ),
    "D1c": (
        " The two kinds of spread are first matched on how far apart the sportsbooks were, so a "
        "difference in how much the books disagreed cannot produce the answer."
    ),
    "D2a": (
        " Split by whether the spread sat on one of the numbers football scores pile up on, or "
        "right beside it."
    ),
    "D2b": (
        " Games that landed exactly on a whole-number spread are either taken out of both groups "
        "or put back in and counted as a result, instead of being thrown away from one group only."
    ),
    "D2e": (
        " The two kinds of spread are first matched on how big the spread was, so the key numbers "
        "sitting on whole numbers cannot produce the answer."
    ),
    "D3b": (
        " The model is trained against one week's market line and asked about the Tuesday opening "
        "line. This row matches the two kinds of spread on how far those two lines were apart."
    ),
    "D3c": (
        " The two kinds of spread are first matched on how far the model thought the line was "
        "wrong, so a difference in how much the model had to say cannot produce the answer."
    ),
    "D4": " Measured one season at a time, or with the seasons matched.",
    "D5a": " Measured on half the seasons, to see whether the same answer comes back twice.",
    "D6": (
        " Measured on the model's pick before the home-side correction is applied, or on games "
        "that correction leaves alone."
    ),
    "D7": (
        " Both groups are first compared against picking the same number of home sides at random, "
        "so a difference in how often the home side covered cannot produce the answer."
    ),
    "D8": (
        " This row is a reference rather than the model: it scores the side the market itself "
        "moved toward, or the same model graded against the final spread instead of the opener."
    ),
}


def plain_summary(values: dict) -> str:
    hypothesis = str(values["hypothesis"])
    domain = str(values["domain"])
    if hypothesis == "D2a" and "minus_other" in domain:
        head = (
            "This row is about the model's own pick. It is how many more games in a hundred the "
            "model gets right on spreads that sat on one of the numbers football scores pile up "
            "on than on the other spreads of the same kind."
        )
    elif hypothesis == "D8":
        head = (
            "This row is a reference rather than the model's pick. It is how many more games in "
            "a hundred the reference gets right when the opening spread was a whole number than "
            "when it ended in a half point, which is the only kind the pool ever posts."
        )
    elif hypothesis == "D7":
        head = (
            "This row is about the model's own pick. It is how much more the model beats a "
            "coin-flip stand-in on whole-number opening spreads than it beats the same stand-in "
            "on half-point ones, which is the only kind the pool ever posts."
        )
    else:
        head = PLAIN_HEAD
    return head + PLAIN_TAIL.get(hypothesis, "")


def record(execute: bool) -> None:
    from nfl_ats.cli import main as cli_main

    cells = json.loads((OUT / "cells.json").read_text())
    reference = [
        (float(values["standard_error"]), int(values["n"]))
        for values in cells.values()
        if isinstance(values, dict)
        and values.get("units") == "accuracy_points"
        and float(values.get("standard_error", 0.0)) > 0.0
        and int(values.get("n", 0)) > 0
    ]
    lines = ["$ErrorActionPreference = 'Stop'"]
    names: list[str] = []
    failures: list[str] = []
    for name, values in cells.items():
        if not isinstance(values, dict) or "delta" not in values:
            continue
        metrics, floored = floor_degenerate_cell(values, reference, ceiling=ACCURACY_POINT_CEILING)
        if metrics["standard_error"] <= 0.0:
            raise ValueError(f"{name} has a zero-width band and no floor could be fitted")
        notes = DISCOUNT if floored is None else f"{DISCOUNT} {floored}"
        if values.get("diagnostic"):
            notes = f"{notes} This row{DIAGNOSTIC_NOTE}"
        season = values.get("season")
        argv = [
            "weak-signals",
            "record",
            "--name",
            f"{FAMILY}_{name}_{int(season) if season else 2020}_{int(season) if season else 2025}",
            "--family",
            FAMILY,
            "--description",
            (
                f"Whole-number opener diagnosis {values['hypothesis']} cell {values['domain']}, "
                "positive favours the whole-number domain"
            ),
            "--source",
            str(OUT / "cells.json"),
            "--classification",
            "unresolved_below_power",
            "--classification-evidence",
            "No resolved wrong sign, reliability or positive-control closing ground established.",
            "--league",
            "nfl",
            "--season-start",
            str(int(season) if season else 2020),
            "--season-end",
            str(int(season) if season else 2025),
            "--sample-games",
            str(int(metrics["n"])),
            "--sample-blocks",
            str(int(metrics["weeks"])),
            "--category",
            str(values.get("category", "modeling")),
            "--effect",
            fixed(metrics["delta"]),
            "--effect-units",
            "accuracy_points",
            "--interval-low",
            fixed(metrics["lower"]),
            "--interval-high",
            fixed(metrics["upper"]),
            "--probability-positive",
            fixed(metrics["probability_positive"]),
            "--standard-error",
            fixed(metrics["standard_error"]),
            "--notes",
            notes,
            "--plain-summary",
            plain_summary(values),
            "--replace",
        ]
        names.append(argv[argv.index("--name") + 1])
        lines.append(
            ".\\.tools\\uv.exe run --no-sync nfl-ats "
            + " ".join(powershell_quote(token) for token in argv)
        )
        if execute:
            code = cli_main(argv)
            if code:
                failures.append(names[-1])
    (OUT / "record_commands.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamped_artifact(
        {
            "names": sorted(names),
            "count": len(names),
            "failures": failures,
            "execution": "run serially in process" if execute else "emitted, not run",
            "registry_dir": os.environ.get("NFL_ATS_REGISTRY_DIR", "default"),
        },
        OUT / "registry_names.json",
    )
    if failures:
        raise RuntimeError(f"Recorder refused {len(failures)} rows: {failures}")
    print(f"{'recorded' if execute else 'emitted'} {len(names)} rows", flush=True)


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def report() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    diagnostics = json.loads((OUT / "diagnostics.json").read_text())
    order = ("D0", "D1b", "D1c", "D2a", "D2b", "D2e", "D3b", "D3c", "D4", "D5a", "D6", "D7", "D8")
    for hypothesis in order:
        print(f"\n=== {hypothesis} ===", flush=True)
        for name, values in cells.items():
            if not isinstance(values, dict) or values.get("hypothesis") != hypothesis:
                continue
            print(
                f"{name[3:]:56s} n={values['n']:5d} delta={values['delta']:+8.3f} "
                f"[{values['lower']:+8.3f},{values['upper']:+8.3f}] "
                f"P+={values['probability_positive']:.4f}",
                flush=True,
            )
    for key, values in diagnostics.items():
        print(f"\n=== {key} ===", flush=True)
        print(json.dumps(values, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Why the model is weaker on whole-number opening lines"
    )
    parser.add_argument("--stage", choices=("run", "record", "report"), required=True)
    parser.add_argument("--execute-records", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    if not args.execute_records:
        os.environ.setdefault("NFL_ATS_REGISTRY_DIR", str(OUT / "registry"))
    with threadpool_limits(limits=1):
        if args.stage == "run":
            run()
        elif args.stage == "record":
            record(args.execute_records)
        else:
            report()


if __name__ == "__main__":
    main()

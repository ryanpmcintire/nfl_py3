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

import mod18_pool_shaped_read_eval as p1  # noqa: E402

from nfl_ats.discrete_margin_mapping import (  # noqa: E402
    ACCURACY_POINT_CEILING,
    floor_degenerate_cell,
)
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact  # noqa: E402

OUT = REPO / "artifacts/research/laneQ1"
P1OUT = REPO / "artifacts/research/laneP1"
FAMILY = "overlay_union_line_shape_mechanism_v1"
PREFIX = "ls"
SEED = 20260817
PERMUTATION_SEED = 20260911
DRAWS = 20_000
STRATUM_DRAWS = 2_000
PERMUTATIONS = 2_000
TOLERANCE = 1e-9
PREDECLARATION = REPO / "docs/overlay_union_line_shape.md"

MEMBERS = ("coach_fade", "division_revenge", "player_arrests")

DISCOUNT = (
    "Mechanism diagnostics for a split lane P1 noticed POST-HOC on the same mined 1,537-game "
    "Tuesday-opener archive lanes K, T, H, S, V, C2 and P1 were selected on. The mechanism "
    "tests are predeclared in docs/overlay_union_line_shape.md, frozen before measurement; the "
    "observation they explain was not, and this lane cannot un-see it. Descriptive reuse of a "
    "mined era, not independent confirmation; no rotation window spent."
)

DIAGNOSTIC_NOTE = (
    " CONDITIONS ON THE REALISED RESULT and is a diagnostic only: the final margin is not "
    "known when a card is submitted, so this row can never become a rule."
)

SHAPE_WORDS = {
    "half": (
        "measured only on games whose opening spread ended in a half point, which is the only "
        "kind the pool ever posts"
    ),
    "whole": (
        "measured only on games whose opening spread was a whole number, which the pool never posts"
    ),
    "quarter": (
        "measured only on games whose opening spread ended in a quarter point, which no "
        "sportsbook posts and which only appears because several books were averaged together"
    ),
    "off_pool": (
        "measured only on games whose opening spread was not a half point, which the pool never "
        "posts"
    ),
    "every_line": "measured on every game in the archive, whatever the opening spread looked like",
}


def fixed(value: float) -> str:
    return f"{float(value):.12f}"


def table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    stamp_sidecar(path)


def shape_labels(lines: pd.Series) -> np.ndarray:
    size = np.abs(pd.to_numeric(lines, errors="raise").to_numpy(dtype=float))
    fraction = size - np.floor(size)
    labels = np.full(len(size), "other", dtype=object)
    labels[np.abs(fraction) < TOLERANCE] = "whole"
    labels[np.abs(fraction - 0.5) < TOLERANCE] = "half"
    labels[np.abs(fraction - 0.25) < TOLERANCE] = "quarter"
    labels[np.abs(fraction - 0.75) < TOLERANCE] = "quarter"
    if (labels == "other").any():
        raise ValueError("STOP: opener lines off the quarter-point grid")
    return labels


def graded(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame.margin_vs_open.notna() & frame.margin_vs_open.ne(0)
    return frame.loc[keep].reset_index(drop=True)


def union_effect(group: pd.DataFrame) -> dict:
    return p1.comparison(
        group,
        group.card_S3.to_numpy(dtype=bool),
        group.p_S3.ge(0.5).to_numpy(dtype=bool),
    )


def flip_effect(group: pd.DataFrame, flipped: np.ndarray) -> dict:
    base = group.p_S3.ge(0.5).to_numpy(dtype=bool)
    return p1.comparison(group, np.asarray(base) ^ np.asarray(flipped, dtype=bool), base)


def gap_effect(
    frame: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray | None = None,
) -> dict:
    scored = graded(frame)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (
        (scored.card_S3.to_numpy(dtype=bool) == truth).astype(float)
        - (scored.p_S3.ge(0.5).to_numpy() == truth).astype(float)
        if values is None
        else np.asarray(values, dtype=float)
    )
    keep = (frame.margin_vs_open.notna() & frame.margin_vs_open.ne(0)).to_numpy()
    mask_left = np.asarray(left, dtype=bool)[keep]
    mask_right = np.asarray(right, dtype=bool)[keep]
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    left_sum = np.array([delta[g][mask_left[g]].sum() for g in groups], dtype=float)
    left_count = np.array([mask_left[g].sum() for g in groups], dtype=float)
    right_sum = np.array([delta[g][mask_right[g]].sum() for g in groups], dtype=float)
    right_count = np.array([mask_right[g].sum() for g in groups], dtype=float)
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (DRAWS, len(groups)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = 100 * (
            left_sum[selected].sum(axis=1) / left_count[selected].sum(axis=1)
            - right_sum[selected].sum(axis=1) / right_count[selected].sum(axis=1)
        )
    draws = draws[np.isfinite(draws)]
    point = 100 * (
        delta[mask_left].mean() - delta[mask_right].mean()
        if mask_left.any() and mask_right.any()
        else np.nan
    )
    return {
        "delta": float(point),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "standard_error": float(draws.std(ddof=1)),
        "n": int(mask_left.sum() + mask_right.sum()),
        "weeks": len(groups),
        "n_left": int(mask_left.sum()),
        "n_right": int(mask_right.sum()),
    }


def standardised_gap(
    frame: pd.DataFrame, strata: np.ndarray, minimum: int, draws: int = STRATUM_DRAWS
) -> dict:
    scored = graded(frame)
    keep = (frame.margin_vs_open.notna() & frame.margin_vs_open.ne(0)).to_numpy()
    stratum = np.asarray(strata, dtype=object)[keep]
    shape = scored.shape_label.to_numpy()
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (scored.card_S3.to_numpy(dtype=bool) == truth).astype(float) - (
        scored.p_S3.ge(0.5).to_numpy() == truth
    ).astype(float)
    names = sorted({str(value) for value in stratum})
    eligible = []
    for name in names:
        inside = stratum == name
        if (inside & (shape == "half")).sum() >= minimum and (
            inside & (shape == "whole")
        ).sum() >= minimum:
            eligible.append(name)
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
        "per_stratum_gap": {name: float(per_stratum[i]) for i, name in enumerate(eligible)},
        "stratum_weight": {name: float(weights[i]) for i, name in enumerate(eligible)},
    }


def decomposition(group: pd.DataFrame) -> dict:
    scored = graded(group)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    base = scored.p_S3.ge(0.5).to_numpy()
    card = scored.card_S3.to_numpy(dtype=bool)
    flipped = card != base
    correct_base = (base == truth).astype(float)
    weight = 1.0 - 2.0 * correct_base
    total = len(scored)
    flips = int(flipped.sum())
    hit = float((card[flipped] == truth[flipped]).mean()) if flips else float("nan")
    identity = 100.0 * (flips / total) * (2.0 * hit - 1.0) if flips else 0.0
    measured = 100.0 * float((weight[flipped]).sum()) / total if total else float("nan")
    return {
        "n": total,
        "flips": flips,
        "flip_rate": flips / total if total else float("nan"),
        "hit_rate": hit,
        "identity_accuracy_points": float(identity),
        "measured_accuracy_points": float(measured),
        "identity_gap": float(abs(identity - measured)),
    }


def direction_null(group: pd.DataFrame) -> dict:
    scored = graded(group)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    base = scored.p_S3.ge(0.5).to_numpy()
    card = scored.card_S3.to_numpy(dtype=bool)
    flipped = card != base
    weight = 1.0 - 2.0 * (base == truth).astype(float)
    total = len(scored)
    to_home = np.flatnonzero(~base)
    to_away = np.flatnonzero(base)
    want_home = int((flipped & ~base).sum())
    want_away = int((flipped & base).sum())
    observed = 100.0 * float(weight[flipped].sum()) / total
    rng = np.random.default_rng(PERMUTATION_SEED)
    values = np.empty(PERMUTATIONS, dtype=float)
    for draw in range(PERMUTATIONS):
        picked_home = rng.choice(to_home, size=min(want_home, len(to_home)), replace=False)
        picked_away = rng.choice(to_away, size=min(want_away, len(to_away)), replace=False)
        values[draw] = 100.0 * float(weight[picked_home].sum() + weight[picked_away].sum()) / total
    return {
        "n": total,
        "observed_accuracy_points": observed,
        "flips_to_home": want_home,
        "flips_to_away": want_away,
        "home_cover_rate": float(truth.mean()),
        "raw_pick_home_rate": float(base.mean()),
        "null_mean": float(values.mean()),
        "null_standard_deviation": float(values.std(ddof=1)),
        "null_share_below_observed": float(
            (values < observed).mean() + 0.5 * (values == observed).mean()
        ),
        "permutations": PERMUTATIONS,
    }


def spearman_brown(value: float) -> float:
    if not np.isfinite(value) or value <= -1.0:
        return float("nan")
    return float(2.0 * value / (1.0 + value))


def block_split_half(frame: pd.DataFrame) -> dict:
    scored = graded(frame).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (scored.card_S3.to_numpy(dtype=bool) == truth).astype(float) - (
        scored.p_S3.ge(0.5).to_numpy() == truth
    ).astype(float)
    shape = scored.shape_label.to_numpy()
    pairs: list[tuple[float, float]] = []
    for _, index in scored.groupby(["season", "week"], sort=True).indices.items():
        order = np.arange(len(index))
        halves = []
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
        return {"blocks": len(pairs), "pearson": float("nan"), "spearman": float("nan")}
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
        "spearman_spearman_brown": spearman_brown(spearman),
        "pearson_lower": float(np.quantile(boot, 0.025)),
        "pearson_upper": float(np.quantile(boot, 0.975)),
    }


def season_split_half(frame: pd.DataFrame) -> dict:
    scored = graded(frame)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    delta = (scored.card_S3.to_numpy(dtype=bool) == truth).astype(float) - (
        scored.p_S3.ge(0.5).to_numpy() == truth
    ).astype(float)
    shape = scored.shape_label.to_numpy()
    season = scored.season.to_numpy()
    week = scored.week.to_numpy()
    pairs = []
    for value in sorted(set(season.tolist())):
        halves = []
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
        return {"seasons": len(pairs), "spearman": float("nan")}
    left = np.array([pair[1] for pair in pairs])
    right = np.array([pair[2] for pair in pairs])
    spearman = float(stats.spearmanr(left, right).statistic)
    pearson = float(stats.pearsonr(left, right).statistic)
    return {
        "seasons": len(pairs),
        "per_season": [
            {"season": pair[0], "odd_week_gap": pair[1], "even_week_gap": pair[2]} for pair in pairs
        ],
        "spearman": spearman,
        "spearman_spearman_brown": spearman_brown(spearman),
        "pearson": pearson,
        "pearson_spearman_brown": spearman_brown(pearson),
    }


def member_flip_sets() -> dict[str, set[str]]:
    per_game_path = P1OUT / "opener_evaluation/S3/per_game.parquet"
    per_game, schedules, player, _, _ = p1.load_inputs(per_game_path, REPO / "data")
    predictions = p1.build_predictions_frame(per_game, schedules)
    overlays = p1.run_overlays(predictions, schedules, player)
    arrests, _ = p1.reconstruct_arrest_flip_set(
        per_game, REPO / p1.DEFAULT_FEATURES, REPO / p1.DEFAULT_INCIDENTS
    )
    return {
        "coach_fade": {flip.game_id for flip in overlays["coach_fade_overlay"].flips},
        "division_revenge": {
            flip.game_id for flip in overlays["division_revenge_tilt_overlay"].flips
        },
        "player_arrests": {str(value) for value in arrests},
    }


def magnitude_buckets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    size = frame.tue_open_home_spread.abs().to_numpy(dtype=float)
    integer = np.minimum(np.floor(size), 14.0)
    fine = np.array([f"{int(value):02d}" for value in integer], dtype=object)
    edges = [0.0, 2.0, 4.0, 7.0, 10.0, np.inf]
    names = ["0-2", "2-4", "4-7", "7-10", "10+"]
    coarse = np.full(len(size), "", dtype=object)
    for index, name in enumerate(names):
        coarse[(size >= edges[index]) & (size < edges[index + 1])] = name
    return fine, coarse


def run() -> None:
    frame = pd.read_parquet(P1OUT / "scored.parquet")
    frame["shape_label"] = shape_labels(frame.tue_open_home_spread)
    frame["line_size"] = frame.tue_open_home_spread.abs()
    fine, coarse = magnitude_buckets(frame)
    frame["magnitude_fine"] = fine
    frame["magnitude_coarse"] = coarse
    books = frame.opener_books.to_numpy(dtype=float)
    low, high = np.quantile(books, [1 / 3, 2 / 3])
    frame["book_stratum"] = np.where(
        books <= low, "books_low", np.where(books <= high, "books_mid", "books_high")
    )

    flips = member_flip_sets()
    rebuilt = frame.p_S3.ge(0.5).to_numpy(dtype=bool)
    union = set().union(*flips.values())
    rebuilt = rebuilt ^ frame.game_id.isin(union).to_numpy()
    if not np.array_equal(rebuilt, frame.card_S3.to_numpy(dtype=bool)):
        raise ValueError("STOP: the three rebuilt members do not reproduce the played card")
    for name, ids in flips.items():
        frame[f"flip_{name}"] = frame.game_id.isin(ids).to_numpy()

    shapes = {name: (frame.shape_label == name).to_numpy() for name in ("half", "whole", "quarter")}
    shapes["off_pool"] = ~shapes["half"]

    cells: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    shapes["every_line"] = np.ones(len(frame), dtype=bool)

    for name, mask in shapes.items():
        cells[f"{PREFIX}_union_{name}"] = {
            **union_effect(frame.loc[mask]),
            "hypothesis": "H0",
            "shape": name,
            "domain": name,
            "units": "accuracy_points",
        }

    diagnostics["H0_shape_counts"] = {
        name: {
            "games": int(mask.sum()),
            "graded": int((mask & frame.margin_vs_open.ne(0).to_numpy()).sum()),
            "pushes": int((mask & frame.margin_vs_open.eq(0).to_numpy()).sum()),
        }
        for name, mask in shapes.items()
    }

    cells[f"{PREFIX}_gap_raw_whole_minus_half"] = {
        **gap_effect(frame, shapes["whole"], shapes["half"]),
        "hypothesis": "H0",
        "shape": "gap",
        "domain": "whole_minus_half",
        "units": "accuracy_points",
    }
    cells[f"{PREFIX}_gap_raw_offpool_minus_half"] = {
        **gap_effect(frame, shapes["off_pool"], shapes["half"]),
        "hypothesis": "H0",
        "shape": "gap",
        "domain": "off_pool_minus_half",
        "units": "accuracy_points",
    }
    raw_gap = cells[f"{PREFIX}_gap_raw_whole_minus_half"]["delta"]

    for season, group in frame.groupby("season"):
        for name in ("half", "whole"):
            subset = group.loc[group.shape_label == name]
            if graded(subset).empty:
                continue
            cells[f"{PREFIX}_h1a_union_{name}_{int(season)}"] = {
                **union_effect(subset),
                "hypothesis": "H1a",
                "shape": name,
                "domain": f"{name}_season_{int(season)}",
                "units": "accuracy_points",
            }
    diagnostics["H1a_shape_share_by_season"] = {
        str(int(season)): {
            "games": len(group),
            **{
                name: float((group.shape_label == name).mean())
                for name in ("half", "whole", "quarter")
            },
        }
        for season, group in frame.groupby("season")
    }

    for stratum, group in frame.groupby("book_stratum"):
        for name in ("half", "whole"):
            subset = group.loc[group.shape_label == name]
            if graded(subset).empty:
                continue
            cells[f"{PREFIX}_h1b_union_{name}_{stratum}"] = {
                **union_effect(subset),
                "hypothesis": "H1b",
                "shape": name,
                "domain": f"{name}_{stratum}",
                "units": "accuracy_points",
            }
    diagnostics["H1b_books_by_shape"] = {
        name: {
            "mean_books": float(frame.loc[mask].opener_books.mean()),
            "median_books": float(frame.loc[mask].opener_books.median()),
            "counts": {
                str(int(value)): int(count)
                for value, count in frame.loc[mask].opener_books.value_counts().sort_index().items()
            },
        }
        for name, mask in shapes.items()
    }

    season_books = np.array(
        [f"{int(s)}_{b}" for s, b in zip(frame.season, frame.book_stratum, strict=True)],
        dtype=object,
    )
    diagnostics["H1c_standardised_gap"] = standardised_gap(frame, season_books, 15)
    cells[f"{PREFIX}_h1c_gap_standardised_season_books"] = {
        **{
            key: value
            for key, value in diagnostics["H1c_standardised_gap"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "H1c",
        "shape": "gap",
        "domain": "whole_minus_half_standardised_by_season_and_books",
        "units": "accuracy_points",
    }

    for name in ("half", "whole"):
        subset = frame.loc[shapes[name] & frame.margin_vs_open.abs().ge(1.0).to_numpy()]
        cells[f"{PREFIX}_h2b_union_{name}_decided_by_a_point"] = {
            **union_effect(subset),
            "hypothesis": "H2b",
            "shape": name,
            "domain": f"{name}_margin_at_least_one_point",
            "units": "accuracy_points",
        }
    common = frame.loc[frame.margin_vs_open.abs().ge(1.0).to_numpy()]
    cells[f"{PREFIX}_h2b_gap_decided_by_a_point"] = {
        **gap_effect(
            common,
            (common.shape_label == "whole").to_numpy(),
            (common.shape_label == "half").to_numpy(),
        ),
        "hypothesis": "H2b",
        "shape": "gap",
        "domain": "whole_minus_half_margin_at_least_one_point",
        "units": "accuracy_points",
    }

    for direction, shift in (("home_favourable", 0.5), ("away_favourable", -0.5)):
        subset = frame.loc[shapes["whole"]].copy()
        subset["margin_vs_open"] = subset.margin_vs_open + shift
        cells[f"{PREFIX}_h2c_union_whole_pushes_{direction}"] = {
            **union_effect(subset),
            "hypothesis": "H2c",
            "shape": "whole",
            "domain": f"whole_pushes_reinstated_{direction}",
            "units": "accuracy_points",
        }

    hook = frame.margin_vs_open.abs().eq(0.5).to_numpy()
    for label, mask in (("on_the_hook", hook), ("clear_of_the_hook", ~hook)):
        subset = frame.loc[shapes["half"] & mask]
        if graded(subset).empty:
            continue
        cells[f"{PREFIX}_h2d_union_half_{label}"] = {
            **union_effect(subset),
            "hypothesis": "H2d",
            "shape": "half",
            "domain": f"half_{label}",
            "units": "accuracy_points",
            "diagnostic": True,
        }

    for member in MEMBERS:
        for name in ("half", "whole", "every_line"):
            subset = frame.loc[shapes[name]]
            flipped = subset[f"flip_{member}"].to_numpy(dtype=bool)
            if not flipped.any():
                continue
            cells[f"{PREFIX}_h3_{member}_{name}"] = {
                **flip_effect(subset, flipped),
                "hypothesis": "H3",
                "shape": name,
                "domain": f"{member}_{name}",
                "member": member,
                "units": "accuracy_points",
            }
    diagnostics["H3_member_flip_counts"] = {
        member: {
            name: int((frame[f"flip_{member}"].to_numpy() & mask).sum())
            for name, mask in shapes.items()
        }
        for member in MEMBERS
    }

    diagnostics["H4a_magnitude_by_shape"] = {
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
    for bucket, group in frame.groupby("magnitude_coarse"):
        for name in ("half", "whole"):
            subset = group.loc[group.shape_label == name]
            if len(graded(subset)) < 20:
                continue
            cells[
                f"{PREFIX}_h4b_union_{name}_line_{bucket.replace('-', '_').replace('+', 'plus')}"
            ] = {
                **union_effect(subset),
                "hypothesis": "H4b",
                "shape": name,
                "domain": f"{name}_line_{bucket}",
                "units": "accuracy_points",
            }
    diagnostics["H4c_standardised_gap_fine"] = standardised_gap(
        frame, frame.magnitude_fine.to_numpy(), 20
    )
    diagnostics["H4c_standardised_gap_coarse"] = standardised_gap(
        frame, frame.magnitude_coarse.to_numpy(), 20
    )
    cells[f"{PREFIX}_h4c_gap_standardised_magnitude"] = {
        **{
            key: value
            for key, value in diagnostics["H4c_standardised_gap_fine"].items()
            if key
            in ("delta", "lower", "upper", "probability_positive", "standard_error", "n", "weeks")
        },
        "hypothesis": "H4c",
        "shape": "gap",
        "domain": "whole_minus_half_standardised_by_line_size",
        "units": "accuracy_points",
    }

    diagnostics["H5a_block_split_half"] = block_split_half(frame)
    diagnostics["H5b_season_split_half"] = season_split_half(frame)
    for label, seasons in (("odd", (2021, 2023, 2025)), ("even", (2020, 2022, 2024))):
        subset = frame.loc[frame.season.isin(seasons)]
        cells[f"{PREFIX}_h5c_gap_{label}_seasons"] = {
            **gap_effect(
                subset,
                (subset.shape_label == "whole").to_numpy(),
                (subset.shape_label == "half").to_numpy(),
            ),
            "hypothesis": "H5c",
            "shape": "gap",
            "domain": f"whole_minus_half_{label}_seasons",
            "units": "accuracy_points",
        }
        for name in ("half", "whole"):
            cells[f"{PREFIX}_h5c_union_{name}_{label}_seasons"] = {
                **union_effect(subset.loc[subset.shape_label == name]),
                "hypothesis": "H5c",
                "shape": name,
                "domain": f"{name}_{label}_seasons",
                "units": "accuracy_points",
            }

    diagnostics["H6_decomposition"] = {
        name: decomposition(frame.loc[mask]) for name, mask in shapes.items()
    }
    worst = max(
        float(values["identity_gap"]) for values in diagnostics["H6_decomposition"].values()
    )
    if worst > 1e-9:
        raise ValueError(f"STOP: the flip-rate identity does not hold, worst gap {worst}")

    half_terms = diagnostics["H6_decomposition"]["half"]
    whole_terms = diagnostics["H6_decomposition"]["whole"]

    def points(rate: float, hit: float) -> float:
        return 100.0 * rate * (2.0 * hit - 1.0)

    flip_only = points(whole_terms["flip_rate"], half_terms["hit_rate"])
    hit_only = points(half_terms["flip_rate"], whole_terms["hit_rate"])
    observed_half = half_terms["measured_accuracy_points"]
    observed_whole = whole_terms["measured_accuracy_points"]
    diagnostics["H6_gap_decomposition"] = {
        "gap": observed_whole - observed_half,
        "flip_rate_term": flip_only - observed_half,
        "hit_rate_term": hit_only - observed_half,
        "interaction_term": (observed_whole - observed_half)
        - (flip_only - observed_half)
        - (hit_only - observed_half),
        "half_flip_rate": half_terms["flip_rate"],
        "whole_flip_rate": whole_terms["flip_rate"],
        "half_hit_rate": half_terms["hit_rate"],
        "whole_hit_rate": whole_terms["hit_rate"],
    }

    diagnostics["H7_direction_null"] = {
        name: direction_null(frame.loc[mask]) for name, mask in shapes.items()
    }
    excess = {
        name: values["observed_accuracy_points"] - values["null_mean"]
        for name, values in diagnostics["H7_direction_null"].items()
    }
    diagnostics["H7_excess_over_direction_matched_null"] = {
        **excess,
        "gap_in_excess": excess["whole"] - excess["half"],
        "gap_accounted_for_by_direction_and_base_rate": (observed_whole - observed_half)
        - (excess["whole"] - excess["half"]),
    }

    scored_all = graded(frame)
    baseline_correct = (
        scored_all.p_S3.ge(0.5).to_numpy() == scored_all.margin_vs_open.gt(0).to_numpy()
    ).astype(float)
    cells[f"{PREFIX}_h7_model_own_pick_whole_minus_half"] = {
        **gap_effect(frame, shapes["whole"], shapes["half"], values=baseline_correct),
        "hypothesis": "H7",
        "shape": "gap",
        "domain": "model_own_pick_accuracy_whole_minus_half",
        "units": "accuracy_points",
    }

    diagnostics["explanation_bands"] = {
        "raw_gap_whole_minus_half": raw_gap,
        "standardised_by_season_and_books": diagnostics["H1c_standardised_gap"].get("delta"),
        "standardised_by_line_size_fine": diagnostics["H4c_standardised_gap_fine"].get("delta"),
        "standardised_by_line_size_coarse": diagnostics["H4c_standardised_gap_coarse"].get("delta"),
        "gap_margin_at_least_one_point": cells[f"{PREFIX}_h2b_gap_decided_by_a_point"]["delta"],
        "gap_after_direction_and_base_rate": diagnostics["H7_excess_over_direction_matched_null"][
            "gap_in_excess"
        ],
        "gap_carried_by_division_revenge": (
            cells[f"{PREFIX}_h3_division_revenge_whole"]["delta"]
            - cells[f"{PREFIX}_h3_division_revenge_half"]["delta"]
        ),
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
                "line_size",
                "shape_label",
                "opener_books",
                "book_stratum",
                "magnitude_fine",
                "magnitude_coarse",
                "margin_vs_open",
                "p_S3",
                "card_S3",
                *[f"flip_{member}" for member in MEMBERS],
            ]
        ],
        OUT / "shapes.parquet",
    )
    write_stamped_artifact(cells, OUT / "cells.json")
    write_stamped_artifact(diagnostics, OUT / "diagnostics.json")
    write_stamped_artifact(
        {
            "frame": str(P1OUT / "scored.parquet"),
            "frame_sha256": sha256_file(P1OUT / "scored.parquet"),
            "predeclaration_sha256": sha256_file(PREDECLARATION),
            "games": len(frame),
            "weeks": int(frame.groupby(["season", "week"]).ngroups),
            "card_rebuild": "the three members re-unioned reproduce card_S3 exactly",
        },
        OUT / "reproduction.json",
    )
    print(json.dumps(diagnostics["explanation_bands"], indent=2), flush=True)
    print(f"computed {len(cells)} cells", flush=True)


def plain_summary(values: dict) -> str:
    shape = str(values["shape"])
    hypothesis = str(values["hypothesis"])
    member = str(values.get("member", ""))
    who = {
        "coach_fade": "the coach-fade rule on its own",
        "division_revenge": "the division-rematch rule on its own",
        "player_arrests": "the player-arrest rule on its own",
    }.get(member, "the three rules about coaches, division rematches and player arrests")
    if hypothesis == "H7":
        head = (
            "This row is not about the weekly adjustments at all. It is how much more often the "
            "model's own pick is right on games whose opening spread was a whole number than on "
            "games whose opening spread ended in a half point, which is the only kind the pool "
            "posts. A negative number means the model is already harder to improve on the "
            "spreads the pool actually posts."
        )
    elif shape == "gap":
        head = (
            "The card is adjusted every week by three rules about coaches, division rematches "
            "and player arrests. This row is the difference between how much those adjustments "
            "help on games whose opening spread was a whole number and how much they help on "
            "games whose opening spread ended in a half point, which is the only kind the pool "
            "posts. A positive number means the rules do better on spreads the pool never posts."
        )
    else:
        where = SHAPE_WORDS.get(shape, "measured on a slice of the archive")
        head = (
            f"The card is adjusted every week by {who}. This row counts how many more games in "
            f"a hundred that adjustment gets right than the model's own pick, {where}."
        )
    tail = {
        "H1a": " Measured one season at a time.",
        "H1b": " Measured separately for games where few and many sportsbooks were captured.",
        "H1c": (
            " The two kinds of spread are first matched on season and on how many sportsbooks "
            "were captured, so a difference in the mix of games cannot produce the answer."
        ),
        "H2b": (
            " Measured only on games decided by at least a full point, so games that landed "
            "exactly on the spread are out of both groups rather than out of one."
        ),
        "H2c": (
            " Games that landed exactly on the whole-number spread are put back in and counted "
            "as a result rather than thrown away, which is how a half-point spread already "
            "treats them."
        ),
        "H2d": (
            " Split by whether the game was decided by the half point itself." + DIAGNOSTIC_NOTE
        ),
        "H3": " Measured for this one rule rather than all three together.",
        "H4b": " Measured one range of spread sizes at a time.",
        "H4c": (
            " The two kinds of spread are first matched on how big the spread was, so the key "
            "numbers sitting on whole numbers cannot produce the answer."
        ),
        "H5c": " Measured on half the seasons, to see whether the same answer comes back twice.",
        "H7": "",
    }.get(hypothesis, "")
    return head + tail


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
        argv = [
            "weak-signals",
            "record",
            "--name",
            f"{FAMILY}_{name}_2020_2025",
            "--family",
            FAMILY,
            "--description",
            (
                f"Line-shape mechanism {values['hypothesis']} cell {values['domain']}, "
                "positive favours the overlay adjustment"
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
            "2020",
            "--season-end",
            "2025",
            "--sample-games",
            str(int(metrics["n"])),
            "--sample-blocks",
            str(int(metrics["weeks"])),
            "--category",
            "modeling",
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
            + " ".join(p1.powershell_quote(token) for token in argv)
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


def report() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    diagnostics = json.loads((OUT / "diagnostics.json").read_text())
    order = ("H0", "H1a", "H1b", "H1c", "H2b", "H2c", "H2d", "H3", "H4b", "H4c", "H5c", "H7")
    for hypothesis in order:
        print(f"\n=== {hypothesis} ===", flush=True)
        for name, values in cells.items():
            if not isinstance(values, dict) or values.get("hypothesis") != hypothesis:
                continue
            print(
                f"{name[3:]:58s} n={values['n']:5d} flips={values.get('flips', -1):4d} "
                f"delta={values['delta']:+8.3f} "
                f"[{values['lower']:+8.3f},{values['upper']:+8.3f}] "
                f"P+={values['probability_positive']:.4f}",
                flush=True,
            )
    for key in (
        "H0_shape_counts",
        "H1a_shape_share_by_season",
        "H1b_books_by_shape",
        "H1c_standardised_gap",
        "H3_member_flip_counts",
        "H4a_magnitude_by_shape",
        "H4c_standardised_gap_fine",
        "H4c_standardised_gap_coarse",
        "H5a_block_split_half",
        "H5b_season_split_half",
        "H6_decomposition",
        "H6_gap_decomposition",
        "H7_direction_null",
        "H7_excess_over_direction_matched_null",
        "explanation_bands",
    ):
        print(f"\n=== {key} ===", flush=True)
        print(json.dumps(diagnostics[key], indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanism tests for the overlay union's split by opening-line shape"
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

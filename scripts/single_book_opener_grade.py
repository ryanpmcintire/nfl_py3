from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nfl_ats.conditional_margin import BANDWIDTH
from nfl_ats.discrete_margin_mapping import (
    ACCURACY_POINT_CEILING,
    SERVED_ATOMS,
    apply_arm,
    floor_degenerate_cell,
    walk_forward_side_reads,
)
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.mass_preserving_lattice import prior_pool
from nfl_ats.modeling import regular_season_rows
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    build_predictions_frame,
    load_inputs,
    reconstruct_arrest_flip_set,
    run_overlays,
)
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.single_book_opener import is_half_point

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
OUT = REPO / "artifacts/research/laneP2"
FAMILY = "single_book_opener_grade_v1"
PREFIX = "sb"
SEED = 20260817
DRAWS = 20_000
TOLERANCE = 1e-9
PREDECLARATION = REPO / "docs/single_book_opener_grade.md"

SERIES = {
    "book": REPO / "artifacts/research/laneP2/line_series",
    "halfpoint_median": REPO / "artifacts/research/laneP2/line_series_half",
}

DISCOUNT = (
    "Same 2020-2025 seasons every other lane has read, and the atom set {3, 7} is lane T's, "
    "chosen after seeing lane K's bucket-7 gain on these same games. What is new is the LINE: a "
    "per-book posted opener that no prior lane has graded at, so the graded outcome differs from "
    "the consensus archive wherever the book and the consensus disagree. Descriptive reuse of a "
    "mined era, not independent confirmation; no rotation window spent."
)

RULE_SUMMARIES = {
    "probability_rule": (
        "the pick the card actually plays, the model's cover chance with the home-side adjustment"
    ),
    "raw": "the model's cover chance before the home-side adjustment",
    "residual": "the pick taken from the sign of the model's points edge",
}

DOMAIN_SUMMARIES = {
    "all": "Measured on every game the book posted an opening spread for.",
    "half": (
        "Measured only on games whose opening spread at that book was a half point, which is the "
        "only kind of spread the pool ever posts."
    ),
    "near": "Measured only on half-point spreads of 2.5, 3.5, 6.5 or 7.5.",
    "near_3": "Measured only on half-point spreads of 2.5 or 3.5.",
    "near_7": "Measured only on half-point spreads of 6.5 or 7.5.",
    "away": "Measured only on half-point spreads well away from 3 and 7.",
    "whole": (
        "Measured only on games whose opening spread at that book was a whole number, which the "
        "pool never posts; reported for contrast."
    ),
}

ARM_SUMMARIES = {
    "G2": (
        "Football scores pile up on 3 and 7. When the spread is on one of those numbers, or a "
        "half point either side of it, this version reads the chance of covering off how games "
        "at a similar spread actually finished; every other game is left alone."
    ),
    "KL1b": (
        "The rule the card plays today: when the spread is set exactly on 3 or 7, the chance of "
        "covering is read off how games at a similar spread actually finished."
    ),
}

KIND_SUMMARIES = {
    "standalone": (
        "This row counts how many more games in a hundred it gets right than the smooth reading "
        "in use."
    ),
    "card": (
        "This row counts the same thing after the usual weekly adjustments are applied to both."
    ),
}


def fixed(value: float) -> str:
    return f"{float(value):.12f}"


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    stamp_sidecar(path)


def blocked(frame: pd.DataFrame, difference: np.ndarray, samples: int = DRAWS) -> dict:
    groups = list(frame.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([difference[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (samples, len(groups)))
    draws = 100 * sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "delta": float(100 * difference.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "standard_error": float(draws.std(ddof=1)),
        "n": len(frame),
        "weeks": len(groups),
    }


def pick_comparison(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, samples: int = DRAWS
) -> dict:
    valid = (frame.margin_vs_open.ne(0) & frame.margin_vs_open.notna()).to_numpy()
    scored = frame.loc[valid].reset_index(drop=True)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    cp, bp = np.asarray(candidate)[valid], np.asarray(baseline)[valid]
    difference = (cp == truth).astype(float) - (bp == truth).astype(float)
    cells = blocked(scored, difference, samples)
    cells.update(
        candidate_accuracy=float((cp == truth).mean()),
        baseline_accuracy=float((bp == truth).mean()),
        flips=int((cp != bp).sum()),
    )
    return cells


def latest_series_summary(name: str) -> dict:
    return json.loads((SERIES[name] / "summary.json").read_text(encoding="utf-8"))


def line_source_directory(label: str) -> Path:
    root = REPO / "artifacts/opener_evaluation_line_source"
    matches = [
        directory
        for directory in sorted(root.iterdir(), reverse=True)
        if (directory / "metadata.json").is_file()
        and json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        .get("opener_line_source", {})
        .get("label")
        == label
    ]
    if not matches:
        raise ValueError(f"No line-source opener evaluation for {label!r}")
    return matches[0]


def headline(active: dict) -> None:
    consensus_meta, consensus_dir = find_matching_opener_evaluation(REPO / "artifacts", active)
    consensus = pd.read_parquet(consensus_dir / "per_game.parquet")
    payload: dict[str, object] = {
        "consensus_artifact": str(consensus_dir),
        "consensus_model_id": consensus_meta.get("active_model_id"),
        "active_model_id": active["model_id"],
        "predeclaration_sha256": sha256_file(PREDECLARATION),
        "feature_table_sha256": sha256_file(FEATURES),
        "series": {},
    }
    if sha256_file(FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    cells: dict[str, dict] = {}
    for name, label in (("book", None), ("halfpoint_median", "halfpoint_median")):
        summary = latest_series_summary(name)
        series_label = label or f"book:{summary['book']}"
        directory = line_source_directory(series_label)
        candidate = pd.read_parquet(directory / "per_game.parquet")
        merged = candidate.merge(
            consensus[
                [
                    "game_id",
                    "tue_open_home_spread",
                    "margin_vs_open",
                    "correct_at_open",
                    "correct_at_open_probability_rule",
                    "correct_at_open_probability_rule_raw",
                ]
            ].rename(columns=lambda c: c if c == "game_id" else f"consensus_{c}"),
            on="game_id",
            how="inner",
        )
        rules = {
            "probability_rule": (
                "correct_at_open_probability_rule",
                "consensus_correct_at_open_probability_rule",
            ),
            "raw": (
                "correct_at_open_probability_rule_raw",
                "consensus_correct_at_open_probability_rule_raw",
            ),
            "residual": ("correct_at_open", "consensus_correct_at_open"),
        }
        entry: dict[str, object] = {
            "series": name,
            "label": series_label,
            "line_source_artifact": str(directory),
            "series_summary": summary,
            "paired_games": len(merged),
            "lines_equal": int(
                (merged.tue_open_home_spread - merged.consensus_tue_open_home_spread)
                .abs()
                .lt(TOLERANCE)
                .sum()
            ),
            "half_point_games": int(is_half_point(merged.tue_open_home_spread).sum()),
        }
        for rule, (candidate_column, baseline_column) in rules.items():
            graded = merged.loc[
                merged[candidate_column].notna() & merged[baseline_column].notna()
            ].reset_index(drop=True)
            difference = graded[candidate_column].to_numpy(dtype=float) - graded[
                baseline_column
            ].to_numpy(dtype=float)
            cell = blocked(graded, difference)
            cell.update(
                candidate_accuracy=float(graded[candidate_column].mean()),
                baseline_accuracy=float(graded[baseline_column].mean()),
                series=name,
                rule=rule,
                domain="paired",
            )
            cells[f"{PREFIX}_{name}_{rule}_paired"] = cell
            entry[rule] = cell
            for season, group in graded.groupby("season", sort=True):
                season_difference = group[candidate_column].to_numpy(dtype=float) - group[
                    baseline_column
                ].to_numpy(dtype=float)
                season_cell = blocked(group.reset_index(drop=True), season_difference)
                season_cell.update(
                    candidate_accuracy=float(group[candidate_column].mean()),
                    baseline_accuracy=float(group[baseline_column].mean()),
                    series=name,
                    rule=rule,
                    domain=f"paired_season_{int(season)}",
                )
                cells[f"{PREFIX}_{name}_{rule}_season_{int(season)}"] = season_cell
        half = merged.loc[is_half_point(merged.tue_open_home_spread)]
        graded_half = half.loc[
            half["correct_at_open_probability_rule"].notna()
            & half["consensus_correct_at_open_probability_rule"].notna()
        ].reset_index(drop=True)
        half_difference = graded_half["correct_at_open_probability_rule"].to_numpy(
            dtype=float
        ) - graded_half["consensus_correct_at_open_probability_rule"].to_numpy(dtype=float)
        half_cell = blocked(graded_half, half_difference)
        half_cell.update(
            candidate_accuracy=float(graded_half["correct_at_open_probability_rule"].mean()),
            baseline_accuracy=float(
                graded_half["consensus_correct_at_open_probability_rule"].mean()
            ),
            series=name,
            rule="probability_rule",
            domain="paired_half_point",
        )
        cells[f"{PREFIX}_{name}_probability_rule_half_point"] = half_cell
        entry["half_point"] = half_cell
        payload["series"][name] = entry  # type: ignore[index]
    payload["cells"] = cells
    write_stamped_artifact(payload, OUT / "headline.json")
    (OUT / "predeclaration.md").write_bytes(PREDECLARATION.read_bytes())
    print(json.dumps({k: v for k, v in cells.items() if "season" not in k}, indent=2), flush=True)


def recover_week_shape(group: pd.DataFrame) -> tuple[float, float]:
    quantile = stats.norm.ppf(group.home_cover_probability_at_open.to_numpy(dtype=float))
    served = group.residual_at_open_served.to_numpy(dtype=float)
    design = np.column_stack([quantile, -np.ones(len(group))])
    solution, *_ = np.linalg.lstsq(design, served, rcond=None)
    return float(solution[0]), float(solution[1])


def exact_atom_mask(lines: pd.Series, atoms: tuple[float, ...] = SERVED_ATOMS) -> np.ndarray:
    size = np.abs(pd.to_numeric(lines, errors="raise").to_numpy(dtype=float))
    return np.asarray(
        np.any(np.abs(size[:, None] - np.asarray(atoms, dtype=float)) < TOLERANCE, axis=1),
        dtype=bool,
    )


def map_stage(active: dict) -> None:
    summary = latest_series_summary("book")
    label = f"book:{summary['book']}"
    directory = line_source_directory(label)
    frame = pd.read_parquet(directory / "per_game.parquet")
    if not frame.probability_method.eq("gaussian_median").all():
        raise ValueError("The run was not graded on the served smooth read")
    frame["residual_std"] = np.nan
    frame["median_S3"] = np.nan
    for _, group in frame.groupby(["season", "week"], sort=True):
        std, median = recover_week_shape(group)
        frame.loc[group.index, "residual_std"] = std
        frame.loc[group.index, "median_S3"] = median
    frame["p_S3"] = stats.norm.sf(
        -(frame.residual_at_open_served.to_numpy(dtype=float) + frame.median_S3.to_numpy()),
        loc=0.0,
        scale=frame.residual_std.to_numpy(),
    )
    gap = float((frame.p_S3 - frame.home_cover_probability_at_open).abs().max())
    if not np.isfinite(gap) or gap > 1e-9:
        raise ValueError(f"STOP: S3 replay mismatch: {gap}")
    frame["push_S3"] = np.nan
    frame["offset_S3"] = frame.home_side_offset_at_open
    frame["center_S3"] = (
        frame.tue_open_home_spread + frame.residual_at_open_served + frame.median_S3
    )
    frame["line_size"] = frame.tue_open_home_spread.abs()
    frame["half_point"] = is_half_point(frame.tue_open_home_spread)

    series = pd.read_parquet(SERIES["book"] / "book.parquet")
    pool = prior_pool(
        regular_season_rows(pd.read_parquet(FEATURES)),
        series.set_index("game_id").home_spread,
    )
    table(pool, OUT / "pool.parquet")
    targets = frame[["game_id", "season", "week"]].copy()
    targets["gameday"] = pool.set_index("game_id").gameday.reindex(frame.game_id).to_numpy()
    targets["line"] = frame.tue_open_home_spread.to_numpy()
    targets["point"] = frame.center_S3.to_numpy()
    if targets.gameday.isna().any():
        raise ValueError("Archive rows without a gameday in the feature table")
    mapped = walk_forward_side_reads(pool, targets, float(BANDWIDTH)).set_index("game_id")
    mapped = mapped.reindex(frame.game_id)
    frame["p_lattice"] = mapped.home_cover_probability.to_numpy()
    frame["push_lattice"] = mapped.push.to_numpy()
    frame["band_lattice"] = mapped.band.to_numpy()

    served_touched = exact_atom_mask(frame.tue_open_home_spread)
    frame["touched_KL1b"] = served_touched
    frame["p_KL1b"] = np.where(served_touched, frame.p_lattice, frame.p_S3)
    applied = apply_arm(
        frame.tue_open_home_spread,
        frame.p_S3.to_numpy(),
        frame.push_S3.to_numpy(),
        frame.p_lattice.to_numpy(),
        frame.push_lattice.to_numpy(),
        "G2",
    )
    frame["touched_G2"] = applied["touched"]
    frame["p_G2"] = applied["probability"]
    untouched = ~applied["touched"]
    if not np.array_equal(frame.loc[untouched, "p_G2"], frame.loc[untouched, "p_S3"]):
        raise ValueError("G2 moved an untouched probability")

    half = frame.half_point.to_numpy(dtype=bool)
    near = near_masks(frame)["near"]
    structural = {
        "kl1b_touches_nothing_on_half_points": bool((served_touched & half).sum() == 0),
        "kl1b_identical_to_s3_on_half_points": bool(
            np.array_equal(frame.p_KL1b.to_numpy()[half], frame.p_S3.to_numpy()[half])
        ),
        "g2_touch_set_equals_near_on_half_points": bool(
            np.array_equal(frame.touched_G2.to_numpy(dtype=bool)[half], near[half])
        ),
    }
    if not all(structural.values()):
        raise ValueError(f"STOP: predeclared structural invariant failed: {structural}")
    table(frame, OUT / "replay.parquet")
    write_stamped_artifact(
        {
            "line_source_artifact": str(directory),
            "label": label,
            "active_model_id": active["model_id"],
            "max_probability_gap": gap,
            "bandwidth": float(BANDWIDTH),
            "games": len(frame),
            "half_point_games": int(half.sum()),
            "near_games": int((half & near).sum()),
            "kl1b_touched": int(served_touched.sum()),
            "g2_touched": int(frame.touched_G2.sum()),
            "g2_touched_on_half_points": int((frame.touched_G2.to_numpy(dtype=bool) & half).sum()),
            "structural_invariants": structural,
            "line_size_counts": {
                str(size): int(count)
                for size, count in frame.line_size.value_counts().sort_index().items()
            },
        },
        OUT / "mapping.json",
    )
    print(json.dumps({"max_probability_gap": gap, "structural": structural}, indent=2), flush=True)


def near_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    size = frame.line_size.to_numpy(dtype=float)
    near_3 = (np.abs(size - 2.5) < TOLERANCE) | (np.abs(size - 3.5) < TOLERANCE)
    near_7 = (np.abs(size - 6.5) < TOLERANCE) | (np.abs(size - 7.5) < TOLERANCE)
    return {"near_3": near_3, "near_7": near_7, "near": near_3 | near_7}


def domain_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    half = frame.half_point.to_numpy(dtype=bool)
    near = near_masks(frame)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "half": half,
        "near": half & near["near"],
        "near_3": half & near["near_3"],
        "near_7": half & near["near_7"],
        "away": half & ~near["near"],
        "whole": ~half,
    }


def composed_picks(per_game: pd.DataFrame, path: Path, probability: np.ndarray) -> np.ndarray:
    _, schedules, player, _, _ = load_inputs(path, REPO / "data")
    frame = per_game.copy()
    frame["home_cover_probability_at_open"] = probability
    predictions = build_predictions_frame(frame, schedules)
    overlays = run_overlays(predictions, schedules, player)
    arrests, _ = reconstruct_arrest_flip_set(frame, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    union = set(arrests)
    for member in ("coach_fade_overlay", "division_revenge_tilt_overlay"):
        union.update(flip.game_id for flip in overlays[member].flips)
    base = np.asarray(probability) >= 0.5
    return np.asarray(base ^ frame.game_id.isin(union).to_numpy(), dtype=bool)


def score() -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    summary = latest_series_summary("book")
    directory = line_source_directory(f"book:{summary['book']}")
    per_game_path = directory / "per_game.parquet"
    picks = {
        "S3": frame.p_S3.ge(0.5).to_numpy(),
        "G2": frame.p_G2.ge(0.5).to_numpy(),
        "KL1b": frame.p_KL1b.ge(0.5).to_numpy(),
    }
    cards = {
        name: composed_picks(frame, per_game_path, frame[f"p_{name}"].to_numpy())
        for name in ("S3", "G2", "KL1b")
    }
    masks = domain_masks(frame)
    cells: dict[str, dict] = {}
    for arm in ("G2", "KL1b"):
        for baseline in ("S3",):
            for domain, mask in masks.items():
                group = frame.loc[mask].reset_index(drop=True)
                if group.empty:
                    continue
                index = np.flatnonzero(mask)
                for kind, source in (("standalone", picks), ("card", cards)):
                    cell = pick_comparison(group, source[arm][index], source[baseline][index])
                    if cell["flips"] == 0:
                        cell["degenerate_no_flip"] = True
                    cell.update(arm=arm, baseline=baseline, domain=domain, kind=kind)
                    cells[f"{PREFIX}_{arm.lower()}_vs_{baseline.lower()}_{domain}_{kind}"] = cell
    half = masks["half"]
    for season, group in frame.loc[half].groupby("season", sort=True):
        index = group.index.to_numpy()
        for arm in ("G2",):
            for kind, source in (("standalone", picks), ("card", cards)):
                cell = pick_comparison(
                    group.reset_index(drop=True), source[arm][index], source["S3"][index]
                )
                if cell["flips"] == 0:
                    cell["degenerate_no_flip"] = True
                cell.update(arm=arm, baseline="S3", domain=f"half_season_{int(season)}", kind=kind)
                cells[f"{PREFIX}_{arm.lower()}_vs_s3_half_season_{int(season)}_{kind}"] = cell
    write_stamped_artifact(cells, OUT / "cells.json")
    print(
        json.dumps(
            {
                k: {m: round(float(v[m]), 4) for m in ("delta", "probability_positive", "n")}
                for k, v in cells.items()
            },
            indent=2,
        ),
        flush=True,
    )


def band_reference(cells: dict) -> list[tuple[float, int]]:
    return [
        (float(values["standard_error"]), int(values["n"]))
        for values in cells.values()
        if isinstance(values, dict)
        and float(values.get("standard_error", 0.0)) > 0.0
        and int(values.get("n", 0)) > 0
    ]


def headline_plain_summary(values: dict) -> str:
    series = str(values["series"])
    where = (
        "one book's posted opening spread"
        if series == "book"
        else "the middle of the half-point prices the books posted"
    )
    when = str(values["domain"])
    scope = (
        "on every paired game"
        if when == "paired"
        else (
            "on the games whose spread was a half point"
            if when == "paired_half_point"
            else f"on the {when.rsplit('_', 1)[-1]} season"
        )
    )
    return (
        f"The pool always posts a half-point spread at one sportsbook. This row grades the same "
        f"model against {where} instead of the average of every book's number, {scope}, and "
        f"counts how many more games in a hundred it gets right that way, using "
        f"{RULE_SUMMARIES[str(values['rule'])]}."
    )


def arm_plain_summary(values: dict) -> str:
    return (
        f"{ARM_SUMMARIES[str(values['arm'])]} Every game is graded against one book's posted "
        f"opening spread, not the average across books. "
        f"{DOMAIN_SUMMARIES[str(values['domain']).split('_season_')[0]]} "
        f"{KIND_SUMMARIES[str(values['kind'])]}"
    )


def record_argv(
    name: str,
    values: dict,
    description: str,
    plain: str,
    start: int,
    end: int,
    reference: Sequence[tuple[float, int]] = (),
) -> list[str]:
    metrics, floored = floor_degenerate_cell(values, reference, ceiling=ACCURACY_POINT_CEILING)
    if metrics["standard_error"] <= 0.0:
        raise ValueError(f"{name} has a zero-width band with no admissible floor")
    notes = DISCOUNT if floored is None else f"{DISCOUNT} {floored}"
    return [
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        description,
        "--source",
        str(values["source"]),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        "No resolved wrong sign, reliability or positive-control closing ground established.",
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
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
        plain,
        "--replace",
    ]


def record(execute: bool) -> None:
    from nfl_ats.cli import main as cli_main

    headline_cells = json.loads((OUT / "headline.json").read_text(encoding="utf-8"))["cells"]
    arm_cells = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    reference = band_reference({**headline_cells, **arm_cells})
    lines = ["$ErrorActionPreference = 'Stop'"]
    names: list[str] = []
    failures: list[str] = []
    skipped: list[str] = []
    for name, values in headline_cells.items():
        if not isinstance(values, dict) or "delta" not in values:
            continue
        entry = {**values, "source": str(OUT / "headline.json")}
        domain = str(values["domain"])
        season = int(domain.rsplit("_", 1)[-1]) if domain.startswith("paired_season_") else None
        argv = record_argv(
            name,
            entry,
            f"Opener grade at {values['series']} line vs the cross-book consensus line, "
            f"{values['rule']} pick, positive favours the single-book grade",
            headline_plain_summary(values),
            season or 2020,
            season or 2025,
            reference,
        )
        lines.append(
            ".\\.tools\\uv.exe run --no-sync nfl-ats "
            + " ".join(powershell_quote(token) for token in argv)
        )
        names.append(argv[argv.index("--name") + 1])
        if execute and cli_main(argv):
            failures.append(names[-1])
    for name, values in arm_cells.items():
        if not isinstance(values, dict) or "delta" not in values:
            continue
        if values.get("degenerate_no_flip"):
            skipped.append(name)
            continue
        entry = {**values, "source": str(OUT / "cells.json")}
        domain = str(values["domain"])
        season = int(domain.rsplit("_", 1)[-1]) if "_season_" in domain else None
        argv = record_argv(
            name,
            entry,
            f"Discrete key-number read {values['arm']} against the served smooth read at one "
            f"book's posted opener, positive favours the candidate",
            arm_plain_summary(values),
            season or 2020,
            season or 2025,
            reference,
        )
        lines.append(
            ".\\.tools\\uv.exe run --no-sync nfl-ats "
            + " ".join(powershell_quote(token) for token in argv)
        )
        names.append(argv[argv.index("--name") + 1])
        if execute and cli_main(argv):
            failures.append(names[-1])
    path = OUT / "record_commands.ps1"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamped_artifact(
        {
            "names": sorted(names),
            "count": len(names),
            "skipped_no_flip": sorted(skipped),
            "commands": str(path),
            "failures": failures,
            "execution": "run serially in process" if execute else "emitted, not run",
            "registry_dir": os.environ.get("NFL_ATS_REGISTRY_DIR", "default"),
        },
        OUT / "registry_names.json",
    )
    if failures:
        raise RuntimeError(f"Recorder refused {len(failures)} rows: {failures}")
    print(f"{'recorded' if execute else 'emitted'} {len(names)} rows", flush=True)


POSTHOC_FAMILY = "single_book_opener_grade_posthoc_v1"
POSTHOC_DISCOUNT = (
    "POST-HOC and NOT predeclared: this decomposition was cut after the headline signs were "
    "visible, so it is recorded in its own family rather than pooled with the predeclared cells. "
    "Descriptive reuse of the same 2020-2025 archive at two different opener lines; no rotation "
    "window spent."
)
POSTHOC_SUMMARIES = {
    "same_line": (
        "On the games where the book's opening spread was the same number as the average across "
        "books, this row counts how many more games in a hundred the model gets right when it is "
        "refitted and graded against the book's number instead of the average."
    ),
    "different_line": (
        "On the games where the book's opening spread was a different number from the average "
        "across books, this row counts how many more games in a hundred the model gets right "
        "against the book's number."
    ),
    "offset_value_book": (
        "The card nudges every pick toward the home or away side by an amount learned from past "
        "weeks. This row counts how many more games in a hundred that nudge wins when every game "
        "is graded against one book's posted opening spread."
    ),
    "offset_value_consensus": (
        "The same nudge, counted against the average opening spread across books, which is the "
        "number the published history is graded on."
    ),
}


def posthoc(execute: bool) -> None:
    from nfl_ats.cli import main as cli_main

    summary = latest_series_summary("book")
    directory = line_source_directory(f"book:{summary['book']}")
    consensus_directory = REPO / "artifacts/opener_evaluation/20260911T161354Z"
    candidate = pd.read_parquet(directory / "per_game.parquet")
    consensus = pd.read_parquet(consensus_directory / "per_game.parquet")
    merged = candidate.merge(
        consensus[
            [
                "game_id",
                "tue_open_home_spread",
                "correct_at_open_probability_rule",
                "correct_at_open_probability_rule_raw",
                "pick_home_at_open_probability_rule",
                "home_side_offset_at_open",
            ]
        ].rename(columns=lambda c: c if c == "game_id" else f"consensus_{c}"),
        on="game_id",
        how="inner",
    )
    same = (merged.tue_open_home_spread - merged.consensus_tue_open_home_spread).abs().lt(TOLERANCE)
    cells: dict[str, dict] = {}
    for label, mask in (("same_line", same), ("different_line", ~same)):
        group = merged.loc[
            mask
            & merged.correct_at_open_probability_rule.notna()
            & merged.consensus_correct_at_open_probability_rule.notna()
        ].reset_index(drop=True)
        difference = group.correct_at_open_probability_rule.to_numpy(
            dtype=float
        ) - group.consensus_correct_at_open_probability_rule.to_numpy(dtype=float)
        cell = blocked(group, difference)
        cell.update(
            candidate_accuracy=float(group.correct_at_open_probability_rule.mean()),
            baseline_accuracy=float(group.consensus_correct_at_open_probability_rule.mean()),
            domain=label,
        )
        cells[f"{PREFIX}_posthoc_{label}"] = cell
    for label, rule_column, raw_column in (
        (
            "offset_value_book",
            "correct_at_open_probability_rule",
            "correct_at_open_probability_rule_raw",
        ),
        (
            "offset_value_consensus",
            "consensus_correct_at_open_probability_rule",
            "consensus_correct_at_open_probability_rule_raw",
        ),
    ):
        group = merged.loc[merged[rule_column].notna() & merged[raw_column].notna()].reset_index(
            drop=True
        )
        difference = group[rule_column].to_numpy(dtype=float) - group[raw_column].to_numpy(
            dtype=float
        )
        cell = blocked(group, difference)
        cell.update(
            candidate_accuracy=float(group[rule_column].mean()),
            baseline_accuracy=float(group[raw_column].mean()),
            domain=label,
        )
        cells[f"{PREFIX}_posthoc_{label}"] = cell
    cells_payload = {
        **cells,
        "_context": {
            "paired_games": len(merged),
            "same_line_games": int(same.sum()),
            "different_line_games": int((~same).sum()),
            "pick_side_differs": int(
                (
                    merged.pick_home_at_open_probability_rule
                    != merged.consensus_pick_home_at_open_probability_rule
                ).sum()
            ),
            "consensus_artifact": str(consensus_directory),
            "line_source_artifact": str(directory),
        },
    }
    write_stamped_artifact(cells_payload, OUT / "posthoc.json")
    reference = band_reference(cells)
    failures: list[str] = []
    names: list[str] = []
    for name, values in cells.items():
        entry = {**values, "source": str(OUT / "posthoc.json")}
        argv = record_argv(
            name,
            entry,
            f"Post-hoc decomposition of the single-book opener grade, {values['domain']}",
            POSTHOC_SUMMARIES[str(values["domain"])],
            2020,
            2025,
            reference,
        )
        argv[argv.index("--family") + 1] = POSTHOC_FAMILY
        argv[argv.index("--name") + 1] = f"{POSTHOC_FAMILY}_{name}_2020_2025"
        argv[argv.index("--notes") + 1] = POSTHOC_DISCOUNT
        names.append(argv[argv.index("--name") + 1])
        if execute and cli_main(argv):
            failures.append(names[-1])
    if failures:
        raise RuntimeError(f"Recorder refused {len(failures)} rows: {failures}")
    print(
        json.dumps(
            {
                k: {
                    m: round(float(v[m]), 4)
                    for m in ("delta", "lower", "upper", "probability_positive", "n")
                }
                for k, v in cells.items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"{'recorded' if execute else 'computed'} {len(names)} post-hoc rows", flush=True)


def report() -> None:
    headline_cells = json.loads((OUT / "headline.json").read_text(encoding="utf-8"))["cells"]
    arm_cells = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    print("HEADLINE, single-book line vs consensus line, same model, paired games")
    for name, values in headline_cells.items():
        print(
            f"  {name:52s} n={int(values['n']):5d} "
            f"cand={100 * values['candidate_accuracy']:.2f}% "
            f"base={100 * values['baseline_accuracy']:.2f}% "
            f"delta={values['delta']:+.3f} "
            f"[{values['lower']:+.3f},{values['upper']:+.3f}] "
            f"P+={values['probability_positive']:.4f}"
        )
    print()
    print("MOD-18 arms at the posted single-book line")
    for name, values in arm_cells.items():
        if not isinstance(values, dict) or "delta" not in values:
            continue
        print(
            f"  {name:52s} n={int(values['n']):5d} flips={int(values['flips']):4d} "
            f"cand={100 * values['candidate_accuracy']:.2f}% "
            f"base={100 * values['baseline_accuracy']:.2f}% "
            f"delta={values['delta']:+.3f} "
            f"[{values['lower']:+.3f},{values['upper']:+.3f}] "
            f"P+={values['probability_positive']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane P2: grade at a single book's posted opener")
    parser.add_argument(
        "--stage",
        choices=("headline", "map", "score", "record", "posthoc", "report"),
        required=True,
    )
    parser.add_argument("--execute-records", action="store_true")
    parser.add_argument("--isolated-registry", action="store_true")
    args = parser.parse_args()
    if args.isolated_registry:
        os.environ["NFL_ATS_REGISTRY_DIR"] = str(OUT / "registry")
    OUT.mkdir(parents=True, exist_ok=True)
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text(encoding="utf-8"))
    if args.stage == "headline":
        headline(active)
    elif args.stage == "map":
        map_stage(active)
    elif args.stage == "score":
        score()
    elif args.stage == "record":
        record(args.execute_records)
    elif args.stage == "posthoc":
        posthoc(args.execute_records)
    else:
        report()


if __name__ == "__main__":
    main()

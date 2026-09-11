from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from nfl_ats.conditional_margin import BANDWIDTH
from nfl_ats.discrete_margin_mapping import (
    ACCURACY_POINT_CEILING,
    ARMS,
    SERVED_ATOMS,
    apply_arm,
    distance_to_nearest_atom,
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
from nfl_ats.spread_regime import spread_bucket

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
OUT = REPO / "artifacts/research/laneP1"
FAMILY = "mod18_pool_shaped_read_v1"
PREFIX = "ps"
SEED = 20260817
DRAWS = 20_000
TOLERANCE = 1e-9
BASELINES = ("S3", "KL1b")
PREDECLARATION = REPO / "docs/mod18_pool_shaped_read.md"

DISCOUNT = (
    "Same mined 1,537-game Tuesday-opener archive lanes K, T, H, S, V and C2 were selected on; "
    "POOL is a subset of it. S3 is a post-hoc restriction fitted on that archive and the atom "
    "set {3, 7} is lane T's, chosen after seeing lane K's bucket-7 gain on these same games. "
    "The half-point domain restriction is new and was named by docs/mod18_discrete_margin_mapping"
    ".md rather than chosen from a result, but the arms and the atoms were not. Descriptive "
    "reuse of a mined era, not independent confirmation; no rotation window spent."
)

ARM_SUMMARIES = {
    "G1": (
        "Football scores pile up on 3, 7, 10 and 14. This version reads every game's chance of "
        "covering off how games at a similar spread actually finished, instead of a smooth curve."
    ),
    "G2": (
        "Football scores pile up on 3 and 7. When the spread is on one of those numbers, or a "
        "half point either side of it, this version reads the chance of covering off how games "
        "at a similar spread actually finished; every other game is left alone."
    ),
    "G3": (
        "Football scores pile up on 3, 7, 10 and 14. When the spread is on one of those numbers, "
        "or a half point either side of it, this version reads the chance of covering off how "
        "games at a similar spread actually finished; every other game is left alone."
    ),
    "G4": (
        "The pool always posts a half-point spread, so a spread of 3.5 or 2.5 is its way of "
        "pricing 3, and every game that lands on 3 falls on one side of it rather than tying. On "
        "those half-point spreads either side of 3 and 7 this version reads the chance of "
        "covering off how games at a similar spread actually finished; every other game is left "
        "alone."
    ),
    "PCFULL": (
        "A deliberate check that the scoring harness can see a real effect: the result is fed "
        "back in on the games the half-point key-number rule touches."
    ),
    "PCSMALL": (
        "A deliberate check of how small an effect the scoring harness can see: the result is "
        "fed back in on a handful of games, sized to move the score by about one game in a "
        "hundred."
    ),
}

DOMAIN_SUMMARIES = {
    "pool": (
        "Measured only on games whose spread was a half point, which is the only kind of spread "
        "the pool ever posts."
    ),
    "pool_near_3": "Measured only on half-point spreads of 2.5 or 3.5.",
    "pool_near_7": "Measured only on half-point spreads of 6.5 or 7.5.",
    "pool_near": "Measured only on half-point spreads of 2.5, 3.5, 6.5 or 7.5.",
    "pool_away": "Measured only on half-point spreads well away from 3 and 7.",
    "off_pool": (
        "Measured only on games whose spread was not a half point, which the pool never posts; "
        "reported for contrast."
    ),
    "touched": "Measured only on the games this version actually changes.",
}

KIND_SUMMARIES = {
    "standalone": (
        "This row counts how many more games in a hundred it gets right than the version in use."
    ),
    "card": (
        "This row counts the same thing after the usual weekly adjustments are applied to both."
    ),
    "brier": (
        "This row scores how well the stated chances matched what happened, not how many picks "
        "were right."
    ),
    "log_loss": (
        "This row scores how well the stated chances matched what happened, not how many picks "
        "were right."
    ),
}

BASELINE_SUMMARIES = {
    "S3": "The comparison is against the smooth reading used before this change.",
    "KL1b": "The comparison is against the card as it is played today.",
}


def table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    stamp_sidecar(path)


def fixed(value: float) -> str:
    return f"{float(value):.12f}"


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def half_point_mask(lines: pd.Series) -> np.ndarray:
    size = np.abs(pd.to_numeric(lines, errors="raise").to_numpy(dtype=float))
    return np.asarray(np.abs((size % 1.0) - 0.5) < TOLERANCE, dtype=bool)


def exact_atom_mask(lines: pd.Series, atoms: tuple[float, ...] = SERVED_ATOMS) -> np.ndarray:
    size = np.abs(pd.to_numeric(lines, errors="raise").to_numpy(dtype=float))
    return np.asarray(
        np.any(np.abs(size[:, None] - np.asarray(atoms, dtype=float)) < TOLERANCE, axis=1),
        dtype=bool,
    )


def recover_week_shape(group: pd.DataFrame) -> tuple[float, float]:
    quantile = stats.norm.ppf(group.home_cover_probability_at_open.to_numpy(dtype=float))
    served = group.residual_at_open_served.to_numpy(dtype=float)
    design = np.column_stack([quantile, -np.ones(len(group))])
    solution, *_ = np.linalg.lstsq(design, served, rcond=None)
    return float(solution[0]), float(solution[1])


def comparison(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, samples: int = DRAWS
) -> dict:
    valid = (frame.margin_vs_open.ne(0) & frame.margin_vs_open.notna()).to_numpy()
    scored = frame.loc[valid].reset_index(drop=True)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    cp, bp = np.asarray(candidate)[valid], np.asarray(baseline)[valid]
    diff = (cp == truth).astype(float) - (bp == truth).astype(float)
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([diff[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (samples, len(groups)))
    draws = 100 * sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "delta": float(100 * diff.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "standard_error": float(draws.std(ddof=1)),
        "n": len(scored),
        "weeks": len(groups),
        "candidate_accuracy": float((cp == truth).mean()),
        "baseline_accuracy": float((bp == truth).mean()),
        "flips": int((cp != bp).sum()),
    }


def metric(frame: pd.DataFrame, differences: np.ndarray) -> dict:
    groups = list(frame.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([differences[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    draw = np.random.default_rng(SEED).integers(0, len(groups), (DRAWS, len(groups)))
    values = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {
        "delta": float(differences.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "probability_positive": float(probability_positive_from_draws(values)),
        "standard_error": float(values.std(ddof=1)),
        "n": len(frame),
        "weeks": len(groups),
    }


def composed_picks(per_game: pd.DataFrame, path: Path) -> np.ndarray:
    _, schedules, player, _, _ = load_inputs(path, REPO / "data")
    predictions = build_predictions_frame(per_game, schedules)
    overlays = run_overlays(predictions, schedules, player)
    arrests, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    union = set(arrests)
    for member in ("coach_fade_overlay", "division_revenge_tilt_overlay"):
        union.update(flip.game_id for flip in overlays[member].flips)
    base = per_game.pick_home_at_open_probability_rule.to_numpy(dtype=bool)
    return np.asarray(base ^ per_game.game_id.isin(union).to_numpy(), dtype=bool)


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    if sha256_file(FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    if not archive.probability_method.eq("gaussian_median").all():
        raise ValueError("The archive was not graded on the served smooth read")
    frame = archive.copy()
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
    frame["half_point"] = half_point_mask(frame.tue_open_home_spread)
    frame["atom_distance"] = distance_to_nearest_atom(frame.tue_open_home_spread, SERVED_ATOMS)
    table(frame, OUT / "replay.parquet")
    (OUT / "predeclaration.md").write_bytes(PREDECLARATION.read_bytes())
    domain = {
        "games": len(frame),
        "half_point_games": int(frame.half_point.sum()),
        "half_point_share": float(frame.half_point.mean()),
        "exact_atom_games": int(exact_atom_mask(frame.tue_open_home_spread).sum()),
        "half_point_per_season": {
            str(int(season)): {
                "games": len(group),
                "half_point": int(group.half_point.sum()),
                "share": float(group.half_point.mean()),
            }
            for season, group in frame.groupby("season")
        },
        "line_size_counts": {
            str(size): int(count)
            for size, count in frame.line_size.value_counts().sort_index().items()
        },
    }
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(FEATURES),
            "predeclaration_sha256": sha256_file(PREDECLARATION),
            "max_probability_gap": gap,
            "weeks": int(frame.groupby(["season", "week"]).ngroups),
            "replay_method": (
                "Algebraic inversion of the served gaussian_median read, per (season, week); "
                "gated at 1e-9 against the archive's own served probability."
            ),
            "domain": domain,
        },
        OUT / "reproduction.json",
    )
    print(json.dumps({"max_probability_gap": gap, **domain}, indent=2)[:2000], flush=True)


def map_replay(archive: pd.DataFrame) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    pool = prior_pool(
        regular_season_rows(pd.read_parquet(FEATURES)),
        archive.set_index("game_id").tue_open_home_spread,
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
    frame["theta_lattice"] = mapped.theta.to_numpy()
    served_touched = exact_atom_mask(frame.tue_open_home_spread)
    frame["touched_KL1b"] = served_touched
    frame["p_KL1b"] = np.where(served_touched, frame.p_lattice, frame.p_S3)
    frame["push_KL1b"] = np.where(served_touched, frame.push_lattice, frame.push_S3)
    half = frame.half_point.to_numpy(dtype=bool)
    counts: dict[str, dict] = {}
    for name in ARMS:
        applied = apply_arm(
            frame.tue_open_home_spread,
            frame.p_S3.to_numpy(),
            frame.push_S3.to_numpy(),
            frame.p_lattice.to_numpy(),
            frame.push_lattice.to_numpy(),
            name,
        )
        frame[f"touched_{name}"] = applied["touched"]
        frame[f"p_{name}"] = applied["probability"]
        frame[f"push_{name}"] = applied["push"]
        untouched = ~applied["touched"]
        if not np.array_equal(frame.loc[untouched, f"p_{name}"], frame.loc[untouched, "p_S3"]):
            raise ValueError(f"{name} moved an untouched probability")
        counts[name] = {
            "description": ARMS[name].description,
            "games_touched": int(applied["touched"].sum()),
            "games_touched_in_pool": int((applied["touched"] & half).sum()),
            "games_total": len(frame),
            "picks_changed_vs_S3": int(
                (frame[f"p_{name}"].ge(0.5) != frame.p_S3.ge(0.5)).to_numpy().sum()
            ),
            "picks_changed_vs_S3_in_pool": int(
                ((frame[f"p_{name}"].ge(0.5) != frame.p_S3.ge(0.5)).to_numpy() & half).sum()
            ),
        }
    counts["KL1b"] = {
        "description": "served exact-atom override (lane T), rebuilt as the played baseline",
        "games_touched": int(served_touched.sum()),
        "games_touched_in_pool": int((served_touched & half).sum()),
        "games_total": len(frame),
        "picks_changed_vs_S3": int((frame.p_KL1b.ge(0.5) != frame.p_S3.ge(0.5)).to_numpy().sum()),
        "picks_changed_vs_S3_in_pool": int(
            ((frame.p_KL1b.ge(0.5) != frame.p_S3.ge(0.5)).to_numpy() & half).sum()
        ),
    }
    structural = {
        "kl1b_touches_nothing_in_pool": bool((served_touched & half).sum() == 0),
        "kl1b_identical_to_s3_in_pool": bool(
            np.array_equal(frame.p_KL1b.to_numpy()[half], frame.p_S3.to_numpy()[half])
        ),
        "g2_equals_g4_in_pool": bool(
            np.array_equal(
                frame.touched_G2.to_numpy(dtype=bool)[half],
                frame.touched_G4.to_numpy(dtype=bool)[half],
            )
        ),
    }
    if not all(structural.values()):
        raise ValueError(f"STOP: predeclared structural invariant failed: {structural}")
    table(frame, OUT / "replay.parquet")
    write_stamped_artifact(
        {"bandwidth": float(BANDWIDTH), "arms": counts, "structural_invariants": structural},
        OUT / "mapping.json",
    )
    print(json.dumps({"arms": counts, "structural": structural}, indent=2), flush=True)


def research_per_game(frame: pd.DataFrame, column: str, arm: str, archive_path: Path) -> Path:
    candidate = frame.copy()
    candidate["home_cover_probability_at_open"] = candidate[column]
    candidate["home_side_offset_at_open"] = candidate.offset_S3
    for suffix, pick in [
        ("", candidate.residual_at_open.gt(0)),
        ("_probability_rule", candidate.home_cover_probability_at_open.ge(0.5)),
    ]:
        candidate[f"pick_home_at_open{suffix}"] = pick
        candidate[f"correct_at_open{suffix}"] = (
            pick.eq(candidate.margin_vs_open.gt(0))
            .astype(float)
            .where(candidate.margin_vs_open.ne(0))
        )
    directory = OUT / "opener_evaluation" / arm
    table(candidate, directory / "per_game.parquet")
    metadata = json.loads((archive_path / "metadata.json").read_text())
    metadata.update(
        research_arm=arm,
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id=f"research_laneP1_{arm}",
        research_scope="Opener only; close columns retain the incumbent, not candidate evidence.",
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": f"research_laneP1_{arm}",
    }
    write_stamped_artifact(metadata, directory / "metadata.json")
    return directory


def domain_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    size = frame.line_size.to_numpy(dtype=float)
    half = frame.half_point.to_numpy(dtype=bool)
    near_3 = (np.abs(size - 2.5) < TOLERANCE) | (np.abs(size - 3.5) < TOLERANCE)
    near_7 = (np.abs(size - 6.5) < TOLERANCE) | (np.abs(size - 7.5) < TOLERANCE)
    near = near_3 | near_7
    return {
        "pool": half,
        "pool_near": half & near,
        "pool_near_3": half & near_3,
        "pool_near_7": half & near_7,
        "pool_away": half & ~near,
        "off_pool": ~half,
    }


def arm_groups(frame: pd.DataFrame, arm: str) -> list[tuple[str, pd.DataFrame]]:
    masks = domain_masks(frame)
    groups: list[tuple[str, pd.DataFrame]] = [
        (label, frame.loc[mask]) for label, mask in masks.items()
    ]
    pool = frame.loc[masks["pool"]]
    groups += [(f"pool_season_{int(season)}", g) for season, g in pool.groupby("season")]
    if f"touched_{arm}" in frame:
        groups.append(
            (
                "pool_touched",
                frame.loc[masks["pool"] & frame[f"touched_{arm}"].to_numpy(dtype=bool)],
            )
        )
    return [(label, group) for label, group in groups if not group.empty]


def positive_control(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    truth = frame.margin_vs_open.gt(0).to_numpy()
    graded = frame.margin_vs_open.ne(0).to_numpy() & frame.margin_vs_open.notna().to_numpy()
    pool = frame.half_point.to_numpy(dtype=bool)
    touched = frame.touched_G4.to_numpy(dtype=bool) & pool
    leaked = np.where(truth, 1.0, 0.0)
    full = np.where(touched, leaked, frame.p_S3.to_numpy())
    wrong = graded & touched & (frame.p_S3.ge(0.5).to_numpy() != truth)
    target_flips = round(0.01 * int((graded & pool).sum()))
    candidates = np.flatnonzero(wrong)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(candidates, size=min(target_flips, len(candidates)), replace=False)
    small = frame.p_S3.to_numpy().copy()
    small[chosen] = leaked[chosen]
    return {
        "PCFULL": full,
        "PCSMALL": small,
        "_target_flips": np.asarray([target_flips, len(chosen)], dtype=int),
    }


def loss_cells(group: pd.DataFrame, column: str, baseline_column: str) -> dict[str, dict]:
    valid = group.loc[group.margin_vs_open.notna() & group.margin_vs_open.ne(0)].reset_index(
        drop=True
    )
    if valid.empty:
        return {}
    truth = valid.margin_vs_open.gt(0).to_numpy(dtype=float)
    candidate = np.clip(valid[column].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    baseline = np.clip(valid[baseline_column].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    cells = {}
    for kind, cl, bl in [
        ("brier", (candidate - truth) ** 2, (baseline - truth) ** 2),
        (
            "log_loss",
            -(truth * np.log(candidate) + (1 - truth) * np.log1p(-candidate)),
            -(truth * np.log(baseline) + (1 - truth) * np.log1p(-baseline)),
        ),
    ]:
        cells[kind] = {
            **metric(valid, bl - cl),
            "candidate": float(cl.mean()),
            "baseline": float(bl.mean()),
        }
    return cells


def score(archive_path: Path) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    control = positive_control(frame)
    frame["p_PCFULL"] = control["PCFULL"]
    frame["p_PCSMALL"] = control["PCSMALL"]
    frame["touched_PCFULL"] = frame.touched_G4 & frame.half_point
    frame["touched_PCSMALL"] = frame.p_PCSMALL != frame.p_S3
    columns = {name: f"p_{name}" for name in (*BASELINES, *ARMS, "PCFULL", "PCSMALL")}
    for arm, column in columns.items():
        directory = research_per_game(frame, column, arm, archive_path)
        candidate = pd.read_parquet(directory / "per_game.parquet")
        frame[f"card_{arm}"] = composed_picks(candidate, directory / "per_game.parquet")
        print(f"composed {arm}", flush=True)
    frame["bucket"] = spread_bucket(frame.tue_open_home_spread)
    masks = domain_masks(frame)
    invariants: dict[str, dict] = {}
    for name in ARMS:
        untouched = frame.loc[~frame[f"touched_{name}"]]
        invariants[name] = {
            "untouched_games": len(untouched),
            "untouched_max_probability_gap": (
                0.0
                if untouched.empty
                else float(np.abs(untouched[f"p_{name}"] - untouched.p_S3).max())
            ),
            "untouched_card_disagreements": int(
                (untouched[f"card_{name}"] != untouched.card_S3).sum()
            ),
        }
        if invariants[name]["untouched_max_probability_gap"] > 0.0:
            raise ValueError(f"{name} changed an untouched probability")
    pool_mask = masks["pool"]
    invariants["pool_domain"] = {
        "pool_games": int(pool_mask.sum()),
        "kl1b_card_equals_s3_card_in_pool": bool(
            np.array_equal(
                frame.card_KL1b.to_numpy(dtype=bool)[pool_mask],
                frame.card_S3.to_numpy(dtype=bool)[pool_mask],
            )
        ),
    }
    invariants["positive_control"] = {
        "target_flips": int(control["_target_flips"][0]),
        "applied_flips": int(control["_target_flips"][1]),
    }
    write_stamped_artifact(invariants, OUT / "invariants.json")
    cells: dict[str, dict] = {}
    skipped: list[str] = []
    for arm in (*ARMS, "PCFULL", "PCSMALL"):
        for baseline in BASELINES:
            groups = arm_groups(frame, arm)
            if arm.startswith("PC"):
                groups = [(label, g) for label, g in groups if label in ("pool", "pool_touched")]
            for label, group in groups:
                for kind in ("standalone", "card"):
                    if kind == "standalone":
                        cp = group[f"p_{arm}"].ge(0.5).to_numpy()
                        bp = group[f"p_{baseline}"].ge(0.5).to_numpy()
                    else:
                        cp = group[f"card_{arm}"].to_numpy()
                        bp = group[f"card_{baseline}"].to_numpy()
                    graded = (group.margin_vs_open.ne(0) & group.margin_vs_open.notna()).to_numpy()
                    if not bool((cp != bp)[graded].any()):
                        skipped.append(f"{arm}_vs_{baseline}_{label}_{kind}")
                        continue
                    cells[f"{PREFIX}_{arm.lower()}_vs_{baseline.lower()}_{label}_{kind}"] = {
                        **comparison(group, cp, bp),
                        "arm": arm,
                        "baseline": baseline,
                        "domain": label,
                        "kind": kind,
                        "units": "accuracy_points",
                    }
                if baseline == "S3" and label in ("pool", "pool_near"):
                    for kind, values in loss_cells(group, f"p_{arm}", "p_S3").items():
                        cells[f"{PREFIX}_{arm.lower()}_vs_s3_{label}_{kind}"] = {
                            **values,
                            "arm": arm,
                            "baseline": baseline,
                            "domain": label,
                            "kind": kind,
                            "units": f"{kind}_improvement",
                        }
    rows = []
    for label, mask in masks.items():
        group = frame.loc[mask]
        valid = group.loc[group.margin_vs_open.ne(0) & group.margin_vs_open.notna()]
        if valid.empty:
            continue
        for arm in (*BASELINES, *ARMS):
            rows.append(
                {
                    "domain": label,
                    "arm": arm,
                    "n": len(valid),
                    "standalone_accuracy": float(
                        valid[f"p_{arm}"].ge(0.5).eq(valid.margin_vs_open.gt(0)).mean()
                    ),
                    "card_accuracy": float(
                        (
                            valid[f"card_{arm}"].to_numpy() == valid.margin_vs_open.gt(0).to_numpy()
                        ).mean()
                    ),
                    "mean_probability": float(valid[f"p_{arm}"].mean()),
                    "realised_home_cover_rate": float(valid.margin_vs_open.gt(0).mean()),
                }
            )
    table(pd.DataFrame(rows), OUT / "domains.parquet")
    table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    write_stamped_artifact({"skipped_degenerate": skipped}, OUT / "skipped.json")
    headline = {
        name: {
            key: values[key]
            for key in ("delta", "lower", "upper", "probability_positive", "n", "flips")
            if key in values
        }
        for name, values in cells.items()
        if values.get("domain") in ("pool", "pool_near")
        and values.get("kind") in ("standalone", "card")
    }
    print(json.dumps(headline, indent=2), flush=True)


def week1(active: dict) -> None:
    from nfl_ats.four_overlay_composition import apply_four_overlay_composition_for_publication
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    forecast = REPO / "artifacts" / active["weekly_forecast"]["artifact"]
    sidecar = json.loads((forecast / "key_line_pick_read.json").read_text())
    if not sidecar.get("served"):
        raise ValueError("The active forecast carries no served key_line_pick_read sidecar")
    scoring = pd.read_csv(forecast / "predictions.csv")
    scoring = scoring.loc[scoring.method.eq("market_residual")].reset_index(drop=True)
    reads = pd.DataFrame(sidecar["games"]).set_index("game_id")
    reads = reads.loc[scoring.game_id.astype(str)]
    rows = scoring[["game_id", "home_team", "away_team", "spread_line"]].copy()
    rows["p_S3"] = reads.home_cover_probability_smooth.to_numpy(dtype=float)
    rows["push_S3"] = reads.push.to_numpy(dtype=float)
    rows["p_lattice"] = reads.home_cover_probability_discrete.to_numpy(dtype=float)
    rows["push_lattice"] = reads.push.to_numpy(dtype=float)
    gap = float(
        np.abs(
            rows.p_S3.to_numpy()
            - pd.to_numeric(scoring.home_cover_probability, errors="coerce").to_numpy()
        ).max()
    )
    if not np.isfinite(gap) or gap > 1e-9:
        raise ValueError(f"STOP: Week 1 sidecar replay mismatch: {gap}")
    rows["half_point"] = half_point_mask(rows.spread_line)
    served_touched = exact_atom_mask(rows.spread_line)
    rows["touched_KL1b"] = served_touched
    rows["p_KL1b"] = np.where(served_touched, rows.p_lattice, rows.p_S3)
    changes: dict[str, dict] = {}
    touched_counts = {"KL1b": int(served_touched.sum())}
    for name in ARMS:
        applied = apply_arm(
            rows.spread_line,
            rows.p_S3.to_numpy(),
            rows.push_S3.to_numpy(),
            rows.p_lattice.to_numpy(),
            rows.push_lattice.to_numpy(),
            name,
        )
        rows[f"touched_{name}"] = applied["touched"]
        rows[f"p_{name}"] = applied["probability"]
        touched_counts[name] = int(applied["touched"].sum())
        changes[name] = {
            "vs_S3": rows.loc[rows[f"p_{name}"].ge(0.5).ne(rows.p_S3.ge(0.5))].to_dict(
                orient="records"
            )
        }
    schedules, _ = load_snapshot(latest_snapshot(REPO / "data/raw"))
    card_changes: dict[str, object] = {}
    card_note = None
    for arm in (*BASELINES, *ARMS):
        candidate = scoring.copy()
        candidate["home_cover_probability"] = rows[f"p_{arm}"].to_numpy()
        try:
            composed = apply_four_overlay_composition_for_publication(
                candidate, schedules, REPO / "data"
            ).overlaid_predictions
        except Exception as error:
            card_note = f"{type(error).__name__}: {error}"
            card_changes = {"unavailable": card_note}
            break
        rows[f"card_home_{arm}"] = composed.home_cover_probability.ge(0.5).to_numpy()
        if arm not in BASELINES:
            card_changes[arm] = rows.loc[rows[f"card_home_{arm}"].ne(rows.card_home_S3)].to_dict(
                orient="records"
            )
    table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_smooth_vs_card_gap": gap,
            "games": len(rows),
            "half_point_games": int(rows.half_point.sum()),
            "games_touched": touched_counts,
            "changes": changes,
            "card_changes": card_changes,
            "card_composition_note": card_note,
        },
        OUT / "week1.json",
    )
    print(
        json.dumps(
            {
                "touched": touched_counts,
                "half_point_games": int(rows.half_point.sum()),
                "gap": gap,
            },
            indent=2,
        ),
        flush=True,
    )


OVERLAY_FAMILY = "mod18_overlay_union_by_line_shape_v1"
OVERLAY_DISCOUNT = (
    "POST-HOC and NOT predeclared: this split was noticed while reading lane P1's domain table, "
    "after the signs were visible, so it is recorded in its own family rather than pooled with "
    "the predeclared discrete-read arms. It is a descriptive reading of the same mined "
    "1,537-game Tuesday-opener archive and is not independent confirmation. It needs a "
    "predeclared lane of its own before it may govern the played card; no rotation window spent."
)
OVERLAY_DOMAIN_SUMMARY = {
    "pool": (
        "Measured only on games whose spread was a half point, which is the only kind of spread "
        "the pool ever posts."
    ),
    "off_pool": (
        "Measured only on games whose spread was not a half point, which the pool never posts."
    ),
    "pool_near_3": "Measured only on half-point spreads of 2.5 or 3.5.",
    "pool_near_7": "Measured only on half-point spreads of 6.5 or 7.5.",
}


def overlay_domain(execute: bool) -> None:
    from nfl_ats.cli import main as cli_main

    frame = pd.read_parquet(OUT / "scored.parquet")
    masks = domain_masks(frame)
    cells: dict[str, dict] = {}
    for label in ("pool", "off_pool", "pool_near_3", "pool_near_7"):
        group = frame.loc[masks[label]]
        cells[f"overlay_union_{label}"] = {
            **comparison(group, group.card_S3.to_numpy(), group.p_S3.ge(0.5).to_numpy()),
            "domain": label,
        }
    write_stamped_artifact(cells, OUT / "overlay_domain.json")
    names: list[str] = []
    failures: list[str] = []
    for name, values in cells.items():
        argv = [
            "weak-signals",
            "record",
            "--name",
            f"{OVERLAY_FAMILY}_{name}_2020_2025",
            "--family",
            OVERLAY_FAMILY,
            "--description",
            (
                f"Played three-member overlay union against the raw read on {values['domain']}, "
                "positive favours the overlays"
            ),
            "--source",
            str(OUT / "overlay_domain.json"),
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
            str(values["n"]),
            "--sample-blocks",
            str(values["weeks"]),
            "--category",
            "modeling",
            "--effect",
            fixed(values["delta"]),
            "--effect-units",
            "accuracy_points",
            "--interval-low",
            fixed(values["lower"]),
            "--interval-high",
            fixed(values["upper"]),
            "--probability-positive",
            fixed(values["probability_positive"]),
            "--standard-error",
            fixed(values["standard_error"]),
            "--notes",
            OVERLAY_DISCOUNT,
            "--plain-summary",
            (
                "The card is adjusted every week by three rules about coaches, division rematches "
                f"and player arrests. {OVERLAY_DOMAIN_SUMMARY[str(values['domain'])]} This row "
                "counts how many more games in a hundred those adjustments get right than leaving "
                "the model's own pick alone."
            ),
            "--replace",
        ]
        names.append(argv[argv.index("--name") + 1])
        if execute:
            code = cli_main(argv)
            if code:
                failures.append(names[-1])
    if failures:
        raise RuntimeError(f"Recorder refused {len(failures)} rows: {failures}")
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ("delta", "lower", "upper", "probability_positive", "n")}
                for k, v in cells.items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"{'recorded' if execute else 'computed'} {len(names)} overlay rows", flush=True)


def plain_summary(arm: str, baseline: str, domain: str, kind: str) -> str:
    if domain.startswith("pool_season_"):
        where = f"Measured on the {domain.rsplit('_', 1)[-1]} season, half-point spreads only."
    elif domain == "pool_touched":
        where = DOMAIN_SUMMARIES["touched"]
    else:
        where = DOMAIN_SUMMARIES[domain]
    return f"{ARM_SUMMARIES[arm]} {where} {BASELINE_SUMMARIES[baseline]} {KIND_SUMMARIES[kind]}"


def band_reference(cells: dict) -> list[tuple[float, int]]:
    return [
        (float(values["standard_error"]), int(values["n"]))
        for values in cells.values()
        if isinstance(values, dict)
        and values.get("units") == "accuracy_points"
        and float(values.get("standard_error", 0.0)) > 0.0
        and int(values.get("n", 0)) > 0
    ]


def record_argv(
    name: str, values: dict, start: int, end: int, reference: Sequence[tuple[float, int]] = ()
) -> list[str]:
    arm = str(values["arm"])
    baseline = str(values["baseline"])
    domain = str(values["domain"])
    kind = str(values["kind"])
    units = str(values["units"])
    category = "control" if arm.startswith("PC") else "modeling"
    evidence = (
        "Positive control, not a candidate: it measures the harness's resolution, not a rule."
        if arm.startswith("PC")
        else "No resolved wrong sign, reliability or positive-control closing ground established."
    )
    ceiling = ACCURACY_POINT_CEILING if units == "accuracy_points" else None
    metrics, floored = floor_degenerate_cell(values, reference, ceiling=ceiling)
    if metrics["standard_error"] <= 0.0:
        raise ValueError(
            f"{name} has a zero-width band and this family could not fit a floor for it; "
            "recording it would break the registry contract"
        )
    notes = DISCOUNT if floored is None else f"{DISCOUNT} {floored}"
    return [
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        f"Pool-shaped discrete read {name}, positive favours the candidate",
        "--source",
        str(OUT / "cells.json"),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        evidence,
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
        "--sample-games",
        str(metrics["n"]),
        "--sample-blocks",
        str(metrics["weeks"]),
        "--category",
        category,
        "--effect",
        fixed(metrics["delta"]),
        "--effect-units",
        units,
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
        plain_summary(arm, baseline, domain, kind),
        "--replace",
    ]


def record(execute: bool) -> None:
    from nfl_ats.cli import main as cli_main

    cells = json.loads((OUT / "cells.json").read_text())
    lines = ["$ErrorActionPreference = 'Stop'"]
    reference = band_reference(cells)
    names: list[str] = []
    failures: list[str] = []
    for name, values in cells.items():
        if not isinstance(values, dict) or "delta" not in values:
            continue
        domain = str(values.get("domain", ""))
        season = int(domain.rsplit("_", 1)[-1]) if domain.startswith("pool_season_") else None
        argv = record_argv(name, values, season or 2020, season or 2025, reference)
        quoted = " ".join(powershell_quote(token) for token in argv)
        lines.append(f".\\.tools\\uv.exe run --no-sync nfl-ats {quoted}")
        names.append(argv[argv.index("--name") + 1])
        if execute:
            code = cli_main(argv)
            if code:
                failures.append(names[-1])
    path = OUT / "record_commands.ps1"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamped_artifact(
        {
            "names": sorted(names),
            "count": len(names),
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


def report() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    domains = pd.read_parquet(OUT / "domains.parquet")
    wanted: Iterable[str] = ("pool", "pool_near", "pool_near_3", "pool_near_7", "pool_away")
    lines = []
    for domain in wanted:
        for arm in ("G1", "G2", "G3", "G4"):
            for baseline in ("s3", "kl1b"):
                for kind in ("standalone", "card"):
                    key = f"{PREFIX}_{arm.lower()}_vs_{baseline}_{domain}_{kind}"
                    values = cells.get(key)
                    if values is None:
                        continue
                    lines.append(
                        f"{domain:14s} {arm} vs {baseline.upper():5s} {kind:10s} "
                        f"n={values['n']:4d} flips={values['flips']:3d} "
                        f"delta={values['delta']:+.3f} "
                        f"[{values['lower']:+.3f},{values['upper']:+.3f}] "
                        f"P+={values['probability_positive']:.4f}"
                    )
    print("\n".join(lines), flush=True)
    print(flush=True)
    print(domains.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOD-18 pool-shaped grade of the discrete mapping family"
    )
    parser.add_argument(
        "--stage",
        choices=("replay", "map", "score", "week1", "record", "report", "overlay"),
        required=True,
    )
    parser.add_argument("--execute-records", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    if not args.execute_records:
        os.environ.setdefault("NFL_ATS_REGISTRY_DIR", str(OUT / "registry"))
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(REPO / "artifacts", active)
    if match is None:
        raise ValueError("No matching opener evaluation")
    with threadpool_limits(limits=1):
        if args.stage == "replay":
            replay(pd.read_parquet(match[1] / "per_game.parquet"), active, match[1])
        elif args.stage == "map":
            map_replay(pd.read_parquet(match[1] / "per_game.parquet"))
        elif args.stage == "score":
            score(match[1])
        elif args.stage == "week1":
            week1(active)
        elif args.stage == "report":
            report()
        elif args.stage == "overlay":
            overlay_domain(args.execute_records)
        else:
            record(args.execute_records)


if __name__ == "__main__":
    main()

"""Frozen MOD-18 C2 research: the discrete side read at every line and near the key numbers.

Predeclaration: ``docs/mod18_discrete_margin_mapping.md`` (frozen before any
candidate number was computed; its digest is stamped into ``reproduction.json``).

Lane K's construction is imported, never reimplemented: every candidate
probability comes out of ``nfl_ats.mass_preserving_lattice.band_read``, the
same core the served push read and the line sweep already call.

Stages: ``replay`` (rebuild and verify the served S3 baseline), ``map`` (the
walk-forward discrete reads and the four frozen arms), ``score`` (paired
opener grade, standalone and through the played three-member card, plus the
positive control), ``week1`` (read-only 2026 Week 1 side changes), ``record``
(the weak-signal rows).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import home_side_location_opener_eval as lane_s
import mass_preserving_lattice_opener_eval as lane_k
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
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
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

OUT = common.REPO / "artifacts/research/laneC2"
FAMILY = "mod18_discrete_side_read_v1"
PREFIX = "ds"
#: The two declared baselines: the served smooth read, and the played card's
#: exact-atom override (lane T's KL1b) rebuilt on this archive.
BASELINES = ("S3", "KL1b")
DISCOUNT = (
    "Mined archive lanes K, T, H, S and V were selected on; S3 is itself a post-hoc restriction "
    "fitted on it; lane T's atom set was chosen after seeing lane K's bucket-7 gain on it; the "
    "2.5/3.5 asymmetry motivating this lane was measured on these same games before the arms "
    "were declared; four nested correlated arms. Descriptive reuse, not independent "
    "confirmation; no rotation window spent."
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
        "A spread of 3.5 or 2.5 is the pool's way of pricing 3, and every game that lands on 3 "
        "falls on one side of it rather than tying. On those half-point spreads either side of 3 "
        "and 7 this version reads the chance of covering off how games at a similar spread "
        "actually finished; every other game is left alone."
    ),
    "PCFULL": (
        "A deliberate check that the scoring harness can see a real effect: the result is fed "
        "back in on the games the key-number rule touches."
    ),
    "PCSMALL": (
        "A deliberate check of how small an effect the scoring harness can see: the result is "
        "fed back in on a handful of games, sized to move the score by about one game in a "
        "hundred."
    ),
}
KIND_SUMMARIES = {
    "standalone": "This row counts how many more games in a hundred it gets right than the "
    "version in use.",
    "card": "This row counts the same thing after the usual weekly adjustments are applied to "
    "both.",
    "brier": "This row scores how well the stated chances matched what happened, not how many "
    "picks were right.",
    "log_loss": "This row scores how well the stated chances matched what happened, not how many "
    "picks were right.",
}


# ---------------------------------------------------------------------------
# Stage 1: the served baseline, rebuilt from the archive and verified
# ---------------------------------------------------------------------------


def recover_week_shape(group: pd.DataFrame) -> tuple[float, float]:
    """The week's fitted residual median and standard deviation, exactly.

    The served read is ``P = Phi((center + offset + median - line) / std)``
    with ``center + offset - line`` equal to the archive's own
    ``residual_at_open_served``. Two unknowns per week and many games, in an
    exact linear relation, so least squares recovers them to floating-point
    precision -- and :func:`replay` refuses to continue unless reproducing
    the served probability from them lands inside 1e-9. This replaces ~110
    ridge refits with an algebraic inversion of the artifact the refits
    produced; the reproduction gate is what makes the two equivalent.
    """

    quantile = stats.norm.ppf(group.home_cover_probability_at_open.to_numpy(dtype=float))
    served = group.residual_at_open_served.to_numpy(dtype=float)
    design = np.column_stack([quantile, -np.ones(len(group))])
    solution, *_ = np.linalg.lstsq(design, served, rcond=None)
    return float(solution[0]), float(solution[1])


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    if not archive.probability_method.eq("gaussian_median").all():
        raise ValueError("The archive was not graded on the served smooth read")
    frame = archive.copy()
    frame["residual_std"] = np.nan
    frame["median_S3"] = np.nan
    for (_, _), group in frame.groupby(["season", "week"], sort=True):
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
    # The push is deliberately NOT reconstructed for the smooth baseline: the
    # served push already comes from this same lattice (lane S,
    # docs/discrete_push_read.md), so there is no smooth push in production to
    # grade against and inventing one here would be a number with no source.
    # This lane grades the SIDE; the push column exists only to carry the
    # lattice's own value through the arm application.
    frame["push_S3"] = np.nan
    frame["offset_S3"] = frame.home_side_offset_at_open
    frame["center_S3"] = (
        frame.tue_open_home_spread + frame.residual_at_open_served + frame.median_S3
    )
    common.table(frame, OUT / "replay.parquet")
    predeclaration = common.REPO / "docs/mod18_discrete_margin_mapping.md"
    (OUT / "predeclaration.md").write_bytes(predeclaration.read_bytes())
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(common.FEATURES),
            "predeclaration_sha256": sha256_file(predeclaration),
            "laneK_construction_sha256": sha256_file(
                common.REPO / "scripts/mass_preserving_lattice_opener_eval.py"
            ),
            "max_probability_gap": gap,
            "games": len(frame),
            "weeks": int(frame.groupby(["season", "week"]).ngroups),
            "residual_std_range": [
                float(frame.residual_std.min()),
                float(frame.residual_std.max()),
            ],
            "median_range": [float(frame.median_S3.min()), float(frame.median_S3.max())],
            "replay_method": (
                "Algebraic inversion of the served gaussian_median read, per (season, week); "
                "gated at 1e-9 against the archive's own served probability."
            ),
        },
        OUT / "reproduction.json",
    )
    print(json.dumps({"games": len(frame), "max_probability_gap": gap}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Stage 2: the discrete reads and the frozen arms
# ---------------------------------------------------------------------------


def exact_atom_mask(line: pd.Series, atoms: tuple[float, ...] = SERVED_ATOMS) -> np.ndarray:
    """Lane T's served predicate: the line sits EXACTLY on a declared atom."""

    size = pd.to_numeric(line, errors="raise").abs().to_numpy(dtype=float)
    return np.any(np.abs(size[:, None] - np.asarray(atoms, dtype=float)) < 1e-9, axis=1)


def map_replay(archive: pd.DataFrame) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    pool = lane_k.build_pool(archive)
    common.table(pool, OUT / "pool.parquet")
    targets = frame[["game_id", "season", "week"]].copy()
    targets["gameday"] = pool.set_index("game_id").gameday.reindex(frame.game_id).to_numpy()
    targets["line"] = frame.tue_open_home_spread.to_numpy()
    targets["point"] = frame.center_S3.to_numpy()
    if targets.gameday.isna().any():
        raise ValueError("Archive rows without a gameday in the feature table")
    mapped = walk_forward_side_reads(pool, targets, float(BANDWIDTH)).set_index("game_id")
    mapped = mapped.reindex(frame.game_id)
    for column in mapped:
        frame[f"{column}_lattice"] = mapped[column].to_numpy()
    frame["p_lattice"] = mapped.home_cover_probability.to_numpy()
    frame["push_lattice"] = mapped.push.to_numpy()
    # Baseline 2: lane T's served exact-atom override, rebuilt on this archive.
    served_touched = exact_atom_mask(frame.tue_open_home_spread)
    frame["touched_KL1b"] = served_touched
    frame["p_KL1b"] = np.where(served_touched, frame.p_lattice, frame.p_S3)
    frame["push_KL1b"] = np.where(served_touched, frame.push_lattice, frame.push_S3)
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
            "games_total": len(frame),
            "picks_changed_vs_S3": int(
                (frame[f"p_{name}"].ge(0.5) != frame.p_S3.ge(0.5)).to_numpy().sum()
            ),
            "picks_changed_vs_KL1b": int(
                (frame[f"p_{name}"].ge(0.5) != frame.p_KL1b.ge(0.5)).to_numpy().sum()
            ),
        }
    counts["KL1b"] = {
        "description": "served exact-atom override (lane T), rebuilt as the played baseline",
        "games_touched": int(served_touched.sum()),
        "games_total": len(frame),
        "picks_changed_vs_S3": int((frame.p_KL1b.ge(0.5) != frame.p_S3.ge(0.5)).to_numpy().sum()),
        "picks_changed_vs_KL1b": 0,
    }
    frame["line_size"] = frame.tue_open_home_spread.abs()
    frame["atom_distance"] = distance_to_nearest_atom(frame.tue_open_home_spread, SERVED_ATOMS)
    common.table(frame, OUT / "replay.parquet")
    write_stamped_artifact(
        {
            "bandwidth": float(BANDWIDTH),
            "arms": counts,
            "line_size_counts": {
                str(size): int(count)
                for size, count in frame.line_size.value_counts().sort_index().items()
            },
            "half_point_share": float(np.mean(~np.isclose(frame.line_size % 1.0, 0.0))),
        },
        OUT / "mapping.json",
    )
    print(json.dumps(counts, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Stage 3: the paired opener grade
# ---------------------------------------------------------------------------


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
    common.table(candidate, directory / "per_game.parquet")
    metadata = json.loads((archive_path / "metadata.json").read_text())
    metadata.update(
        research_arm=arm,
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id=f"research_laneC2_{arm}",
        research_scope="Opener only; close columns retain the incumbent, not candidate evidence.",
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": f"research_laneC2_{arm}",
    }
    write_stamped_artifact(metadata, directory / "metadata.json")
    return directory


def line_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Grading question 2: does the gain sit where the mechanism predicts?"""

    size = frame.line_size
    blocks = {
        "atom_3": size.sub(3.0).abs().lt(1e-9),
        "adjacent_3": size.sub(2.5).abs().lt(1e-9) | size.sub(3.5).abs().lt(1e-9),
        "atom_7": size.sub(7.0).abs().lt(1e-9),
        "adjacent_7": size.sub(6.5).abs().lt(1e-9) | size.sub(7.5).abs().lt(1e-9),
        "away_from_atoms": frame.atom_distance.gt(0.5 + 1e-9),
    }
    return [(f"line_{label}", frame.loc[mask]) for label, mask in blocks.items() if mask.any()]


def arm_groups(frame: pd.DataFrame, arm: str) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups += [(f"season_{int(season)}", g) for season, g in frame.groupby("season")]
    groups += line_slices(frame)
    if f"touched_{arm}" in frame:
        groups.append(("touched", frame.loc[frame[f"touched_{arm}"]]))
    return [(label, group) for label, group in groups if not group.empty]


def positive_control(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Leaked realised truth on G2's touched games: full, and at ~+1 point.

    Unit slope, zero noise -- the same leak shape ``nfl_ats.totals_wave2``'s
    control and MOD-17's control use. The small control is sized before any
    candidate number is read: enough leaked games that the injected effect is
    about one accuracy point on the whole archive.
    """

    truth = frame.margin_vs_open.gt(0).to_numpy()
    graded = frame.margin_vs_open.ne(0).to_numpy() & frame.margin_vs_open.notna().to_numpy()
    touched = frame.touched_G2.to_numpy(dtype=bool)
    leaked = np.where(truth, 1.0, 0.0)
    full = np.where(touched, leaked, frame.p_S3.to_numpy())
    wrong = graded & touched & (frame.p_S3.ge(0.5).to_numpy() != truth)
    target_flips = round(0.01 * int(graded.sum()))
    candidates = np.flatnonzero(wrong)
    rng = np.random.default_rng(common.SEED)
    chosen = rng.choice(candidates, size=min(target_flips, len(candidates)), replace=False)
    small = frame.p_S3.to_numpy().copy()
    small[chosen] = leaked[chosen]
    return {
        "PCFULL": full,
        "PCSMALL": small,
        "_target_flips": np.asarray([target_flips, len(chosen)], dtype=int),
    }


def score(archive_path: Path) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    control = positive_control(frame)
    frame["p_PCFULL"] = control["PCFULL"]
    frame["p_PCSMALL"] = control["PCSMALL"]
    frame["touched_PCFULL"] = frame.touched_G2
    frame["touched_PCSMALL"] = frame.p_PCSMALL != frame.p_S3
    columns = {name: f"p_{name}" for name in (*BASELINES, *ARMS, "PCFULL", "PCSMALL")}
    for arm, column in columns.items():
        directory = research_per_game(frame, column, arm, archive_path)
        candidate = pd.read_parquet(directory / "per_game.parquet")
        _, card = common.composed_picks(candidate, directory / "per_game.parquet")
        frame[f"card_{arm}"] = card
        print(f"composed {arm}", flush=True)
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
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
                # The controls answer one question -- what can this harness
                # resolve -- so they are recorded overall and on the games they
                # touch, never sliced into a second family of rows.
                groups = [(label, g) for label, g in groups if label in ("overall", "touched")]
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
                        # No graded pick moved: the paired difference is
                        # identically zero, a fact for the write-up and never a
                        # registry row (a bootstrap of zeros is a dead heat).
                        skipped.append(f"{arm}_vs_{baseline}_{label}_{kind}")
                        continue
                    cells[f"{PREFIX}_{arm.lower()}_vs_{baseline.lower()}_{label}_{kind}"] = (
                        common.comparison(group, cp, bp)
                    )
                if baseline == "S3" and label == "overall":
                    for kind, metrics in loss_cells(group, f"p_{arm}", "p_S3").items():
                        cells[f"{PREFIX}_{arm.lower()}_vs_s3_{label}_{kind}"] = metrics
    bucket_rows = []
    for arm in (*BASELINES, *ARMS):
        for bucket, group in frame.groupby("bucket", observed=True):
            valid = group.loc[group.margin_vs_open.ne(0)]
            if valid.empty:
                continue
            bucket_rows.append(
                {
                    "arm": arm,
                    "bucket": str(bucket),
                    "n": len(valid),
                    "standalone_accuracy": float(
                        valid[f"p_{arm}"].ge(0.5).eq(valid.margin_vs_open.gt(0)).mean()
                    ),
                    "card_accuracy": float(
                        (
                            valid[f"card_{arm}"].to_numpy() == valid.margin_vs_open.gt(0).to_numpy()
                        ).mean()
                    ),
                }
            )
    line_rows = []
    for label, group in line_slices(frame):
        valid = group.loc[group.margin_vs_open.ne(0)]
        if valid.empty:
            continue
        for arm in (*BASELINES, *ARMS):
            line_rows.append(
                {
                    "slice": label,
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
                }
            )
    common.table(pd.DataFrame(bucket_rows), OUT / "buckets.parquet")
    common.table(pd.DataFrame(line_rows), OUT / "line_slices.parquet")
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    write_stamped_artifact({"skipped_degenerate": skipped}, OUT / "skipped.json")
    headline = {
        k: v for k, v in cells.items() if "_overall_" in k and k.endswith(("standalone", "card"))
    }
    print(json.dumps(headline, indent=2), flush=True)


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
            **lane_s.metric(valid, bl - cl),
            "candidate": float(cl.mean()),
            "baseline": float(bl.mean()),
        }
    return cells


# ---------------------------------------------------------------------------
# Stage 4: Week 1 2026, read only
# ---------------------------------------------------------------------------


def week1(active: dict) -> None:
    from nfl_ats.four_overlay_composition import apply_four_overlay_composition_for_publication
    from nfl_ats.margin import fit_margin_model
    from nfl_ats.modeling import regular_season_rows
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    forecast = common.REPO / "artifacts" / active["weekly_forecast"]["artifact"]
    sidecar = json.loads((forecast / "home_side_offset.json").read_text())
    scoring = pd.read_csv(forecast / "predictions.csv")
    scoring = scoring.loc[scoring.method.eq("market_residual")].reset_index(drop=True)
    recorded = pd.DataFrame(sidecar["games"]).set_index("game_id").loc[scoring.game_id]
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    training = features.loc[
        features.result.notna()
        & pd.to_datetime(features.gameday).lt(pd.to_datetime(scoring.gameday).min())
    ]
    model = fit_margin_model(
        training,
        target="market_residual",
        model_name=active["regressor"],
        feature_profile=active["feature_profile"],
        ridge_alpha=active["ridge_alpha"],
    )
    base = model.predict(
        scoring,
        probability_method="gaussian_median",
        center_offset=recorded.home_side_offset.to_numpy(),
    )
    gap = float(
        np.abs(
            base.home_cover_probability.to_numpy() - recorded.home_cover_probability.to_numpy()
        ).max()
    )
    if gap > 1e-9:
        raise ValueError(f"Week 1 sidecar replay mismatch: {gap}")
    rows = scoring[["game_id", "home_team", "away_team", "spread_line"]].copy()
    rows["p_S3"] = recorded.home_cover_probability.to_numpy()
    rows["push_S3"] = base.push_probability.to_numpy()
    pool = pd.read_parquet(OUT / "pool.parquet")
    targets = scoring[["game_id", "season", "week"]].copy()
    targets["gameday"] = pd.to_datetime(scoring.gameday)
    targets["line"] = scoring.spread_line.to_numpy()
    targets["point"] = base.predicted_margin.to_numpy() + float(np.median(model.residuals))
    mapped = walk_forward_side_reads(pool, targets, float(BANDWIDTH)).set_index("game_id")
    mapped = mapped.reindex(rows.game_id)
    rows["p_lattice"] = mapped.home_cover_probability.to_numpy()
    rows["push_lattice"] = mapped.push.to_numpy()
    served_touched = exact_atom_mask(rows.spread_line)
    rows["touched_KL1b"] = served_touched
    rows["p_KL1b"] = np.where(served_touched, rows.p_lattice, rows.p_S3)
    changes, touched_counts = {}, {"KL1b": int(served_touched.sum())}
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
            ),
            "vs_KL1b": rows.loc[rows[f"p_{name}"].ge(0.5).ne(rows.p_KL1b.ge(0.5))].to_dict(
                orient="records"
            ),
        }
    schedules, _ = load_snapshot(latest_snapshot(common.REPO / "data/raw"))
    card_changes: dict[str, object] = {}
    card_note = None
    for arm in (*BASELINES, *ARMS):
        candidate = scoring.copy()
        candidate["home_cover_probability"] = rows[f"p_{arm}"].to_numpy()
        try:
            composed = apply_four_overlay_composition_for_publication(
                candidate, schedules, common.REPO / "data"
            ).overlaid_predictions
        except Exception as error:  # reported below, never swallowed
            # A concurrent capture can leave the newest arrest snapshot without
            # a manifest; the raw side changes above are the primary Week 1
            # output and the reason is recorded rather than hidden.
            card_note = f"{type(error).__name__}: {error}"
            card_changes = {"unavailable": card_note}
            break
        rows[f"card_home_{arm}"] = composed.home_cover_probability.ge(0.5).to_numpy()
        if arm not in BASELINES:
            card_changes[arm] = {
                "vs_S3": rows.loc[rows[f"card_home_{arm}"].ne(rows.card_home_S3)].to_dict(
                    orient="records"
                ),
                "vs_KL1b": rows.loc[rows[f"card_home_{arm}"].ne(rows.card_home_KL1b)].to_dict(
                    orient="records"
                ),
            }
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_replay_gap": gap,
            "games": len(rows),
            "games_touched": touched_counts,
            "changes": changes,
            "card_changes": card_changes,
            "card_composition_note": card_note,
        },
        OUT / "week1.json",
    )
    print(json.dumps({"touched": touched_counts}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Stage 5: the registry rows
# ---------------------------------------------------------------------------


def split_cell(name: str) -> tuple[str, str, str, str]:
    """``ds_g2_vs_s3_season_2023_card`` -> arm G2, baseline S3, slice, kind."""

    body = name[len(PREFIX) + 1 :]
    arm, _, body = body.partition("_vs_")
    baseline, _, body = body.partition("_")
    for kind in ("log_loss", "standalone", "card", "brier"):
        if body.endswith(f"_{kind}"):
            return arm.upper(), baseline.upper(), body[: -len(kind) - 1], kind
    raise ValueError(f"Unrecognised cell name: {name}")


def slice_summary(label: str) -> str:
    if label == "overall":
        return "Measured on every game from 2020 through 2025."
    if label.startswith("season_"):
        return f"Measured on the {label.split('_')[-1]} season only."
    if label == "line_atom_3":
        return "Measured only on games whose spread was exactly 3 points."
    if label == "line_atom_7":
        return "Measured only on games whose spread was exactly 7 points."
    if label == "line_adjacent_3":
        return "Measured only on games whose spread was 2.5 or 3.5 points."
    if label == "line_adjacent_7":
        return "Measured only on games whose spread was 6.5 or 7.5 points."
    if label == "line_away_from_atoms":
        return "Measured only on games whose spread was well away from 3 and 7."
    return "Measured only on the games this version actually changes."


def baseline_summary(baseline: str) -> str:
    if baseline == "S3":
        return "The comparison is against the smooth reading used before this change."
    return "The comparison is against the card as it is played today."


def plain_summary(arm: str, baseline: str, label: str, kind: str) -> str:
    return (
        f"{ARM_SUMMARIES[arm]} {slice_summary(label)} {baseline_summary(baseline)} "
        f"{KIND_SUMMARIES[kind]}"
    )


def units_for(name: str) -> str:
    if name.endswith("brier"):
        return "brier_improvement"
    if name.endswith("log_loss"):
        return "log_loss_improvement"
    return "accuracy_points"


def band_reference(cells: dict) -> list[tuple[float, int]]:
    """This family's own (standard error, sample games) pairs, accuracy cells only.

    Brier and log-loss cells are a different scale and would corrupt the
    ``SE^2 * n`` curve, so they are excluded; commensurability is the one
    discipline AGENTS.md keeps around pooling.
    """

    return [
        (float(metrics["standard_error"]), int(metrics["n"]))
        for name, metrics in cells.items()
        if isinstance(metrics, dict)
        and "delta" in metrics
        and units_for(name) == "accuracy_points"
        and float(metrics.get("standard_error", 0.0)) > 0.0
        and int(metrics.get("n", 0)) > 0
    ]


def record_argv(
    name: str, metrics: dict, start: int, end: int, reference: Sequence[tuple[float, int]] = ()
) -> list[str]:
    arm, baseline, label, kind = split_cell(name)
    category = "control" if arm.startswith("PC") else "modeling"
    evidence = (
        "Positive control, not a candidate: it measures the harness's resolution, not a rule."
        if arm.startswith("PC")
        else "No resolved wrong sign, reliability or positive-control closing ground established."
    )
    ceiling = ACCURACY_POINT_CEILING if units_for(name) == "accuracy_points" else None
    metrics, floored = floor_degenerate_cell(metrics, reference, ceiling=ceiling)
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
        f"Discrete conditional side read {name}, positive favours the candidate",
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
        lane_s.fixed(metrics["delta"]),
        "--effect-units",
        units_for(name),
        "--interval-low",
        lane_s.fixed(metrics["lower"]),
        "--interval-high",
        lane_s.fixed(metrics["upper"]),
        "--probability-positive",
        lane_s.fixed(metrics["probability_positive"]),
        "--standard-error",
        lane_s.fixed(metrics["standard_error"]),
        "--notes",
        notes,
        "--plain-summary",
        plain_summary(arm, baseline, label, kind),
        "--replace",
    ]


def record(execute: bool) -> None:
    """Emit the recorder argv, and run them SERIALLY when asked.

    The registry CLI holds ``nfl_ats.io.file_lock`` around load-record-save,
    so a serial run from one lane is safe; the emitted script is kept either
    way so the run is reproducible.
    """

    from nfl_ats.cli import main as cli_main

    cells = json.loads((OUT / "cells.json").read_text())
    lines = [
        "# MOD-18 lane C2 (docs/mod18_discrete_margin_mapping.md) weak-signal records.",
        "# Every row is unresolved_below_power with no closing ground and a",
        "# plain-English summary; --replace makes a re-run idempotent.",
        "$ErrorActionPreference = 'Stop'",
    ]
    reference = band_reference(cells)
    names, failures = [], []
    for name, metrics in cells.items():
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        season = int(name.split("season_")[1].split("_")[0]) if "season_" in name else None
        argv = record_argv(name, metrics, season or 2020, season or 2025, reference)
        quoted = " ".join(lane_k.powershell_quote(token) for token in argv)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("replay", "map", "score", "week1", "record"), required=True
    )
    parser.add_argument(
        "--execute-records",
        action="store_true",
        help="run the recorder argv serially against the live registry",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    if not args.execute_records:
        os.environ["NFL_ATS_REGISTRY_DIR"] = str(OUT / "registry")
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
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
        else:
            record(args.execute_records)


if __name__ == "__main__":
    main()

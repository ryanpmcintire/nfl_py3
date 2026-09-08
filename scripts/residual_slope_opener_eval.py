"""Frozen MOD-18 lane R residual-slope research; see docs/residual_slope.md.

R1 rescales the model's raw out-of-time residual by a walk-forward, shrunken
per-bucket slope before the served point is formed; R1b shares one slope
across the three big buckets. S3 is R1 with every slope equal to one, so the
incumbent is reproduced exactly rather than approximated.

Every write lands under ``artifacts/research/laneR``; the live registry is
never touched -- the ``record`` stage WRITES the record commands to a
PowerShell file for the coordinator to run serially instead of running them.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import home_side_location_opener_eval as lane_s
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.home_side_location import (
    PRIOR_WEIGHT_GAMES,
    archive_prior_stream,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.spread_regime import BUCKETS, spread_bucket

OUT = common.REPO / "artifacts/research/laneR"
PREDECLARATION = common.REPO / "docs/residual_slope.md"
FAMILY = "mod18_home_side_location_v1"
ARMS = ("R1", "R1b")
#: The three buckets R1b pools into one slope (the same buckets S3 serves).
BIG_BUCKETS = ("7", "7.5-10", "10.5+")
#: The incumbent weight the slope is shrunk toward: keep the whole residual.
INCUMBENT_SLOPE = 1.0
DISCOUNT = (
    "Post-hoc mechanism found on the mined archive S2/S3 were selected on, and the 10.5+ slope "
    "that motivated it was estimated on these same games; walk-forward fitting removes "
    "look-ahead within a game but not the archive-level selection; two correlated arms; "
    "descriptive reuse, not independent confirmation."
)
BUCKET_WORDS = {
    "0-3": "the opening line was 3 points or less",
    "3.5-6.5": "the opening line was between 3.5 and 6.5 points",
    "7": "the opening line was exactly 7 points",
    "7.5-10": "the opening line was between 7.5 and 10 points",
    "10.5+": "the opening line was 10.5 points or more",
}
ARM_WORDS = {
    "R1": (
        "On the biggest lines the model's own disagreement with the betting number has been "
        "pointing the wrong way, so this version keeps only the share of that disagreement "
        "earlier seasons say is worth keeping at each size of line"
    ),
    "R1b": (
        "The same idea, except one shared amount of trust is used for every line of seven "
        "points or more instead of a separate amount for each size"
    ),
}
KIND_WORDS = {
    "standalone": "how often the model on its own picked the right side of the opening line",
    "card": "how often the picks actually played picked the right side of the opening line",
    "brier": (
        "how close the stated chances were to what happened, where a positive number means "
        "the new way was closer"
    ),
    "log_loss": (
        "how well the stated chances scored, where a positive number means the new way "
        "scored better"
    ),
}
KIND_UNITS = {
    "standalone": "accuracy_points",
    "card": "accuracy_points",
    "brier": "brier_improvement",
    "log_loss": "log_loss_improvement",
}


# ---------------------------------------------------------------------------
# The predeclared estimator
# ---------------------------------------------------------------------------


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y`` on ``x`` WITH an intercept.

    Returns the incumbent weight 1.0 when there is nothing to estimate from
    (fewer than three rows, or no variation in ``x``), so an empty or
    degenerate bucket serves the incumbent rather than a fabricated slope.
    """

    n = int(x.size)
    if n < 3:
        return INCUMBENT_SLOPE
    sxx = float((x * x).sum() - x.sum() ** 2 / n)
    if not sxx > 0.0:
        return INCUMBENT_SLOPE
    sxy = float((x * y).sum() - x.sum() * y.sum() / n)
    return sxy / sxx


def shrink(slope: float, games: int) -> float:
    """Shrink toward the incumbent 1.0 with the declared 100-game prior."""

    return float(
        (games * slope + PRIOR_WEIGHT_GAMES * INCUMBENT_SLOPE) / (games + PRIOR_WEIGHT_GAMES)
    )


@dataclass(frozen=True)
class SlopeFit:
    """Per-bucket walk-forward slopes fitted on prior completed games only."""

    betas: dict[str, float]
    raw: dict[str, float]
    games: dict[str, int]


def slope_inputs(prior: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """``x`` = raw out-of-time residual, ``y`` = home margin minus the opener."""

    completed = prior.loc[prior["result"].notna() & prior["point_incumbent"].notna()]
    x = (completed["point_incumbent"] - completed["spread_line"]).to_numpy(dtype=float)
    y = (completed["result"] - completed["spread_line"]).to_numpy(dtype=float)
    return x, y, spread_bucket(completed["spread_line"])


def fit_slopes(prior: pd.DataFrame, *, pooled: bool) -> SlopeFit:
    """R1 (``pooled=False``) or R1b (``pooled=True``) slopes for one target week."""

    x, y, buckets = slope_inputs(prior)
    betas: dict[str, float] = {}
    raw: dict[str, float] = {}
    games: dict[str, int] = {}
    for bucket in BUCKETS:
        mask = buckets.eq(bucket).to_numpy()
        n = int(mask.sum())
        estimate = ols_slope(x[mask], y[mask])
        raw[bucket] = estimate
        games[bucket] = n
        betas[bucket] = shrink(estimate, n)
    if pooled:
        mask = buckets.isin(BIG_BUCKETS).to_numpy()
        n = int(mask.sum())
        estimate = ols_slope(x[mask], y[mask])
        value = shrink(estimate, n)
        for bucket in BIG_BUCKETS:
            raw[bucket] = estimate
            games[bucket] = n
            betas[bucket] = value
    return SlopeFit(betas=betas, raw=raw, games=games)


def center_shift(
    betas: Mapping[str, float],
    lines: pd.Series,
    residual: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """``(beta_b - 1) * raw residual + S3 offset`` -- see the predeclaration.

    ``MarginModel.predict`` forms ``line + raw residual`` and adds this shift,
    so the served point becomes ``line + beta_b * raw residual + offset_b``.
    A row whose line has no bucket keeps ``beta = 1`` (the incumbent).
    """

    beta = spread_bucket(lines).map(dict(betas)).astype(float).fillna(INCUMBENT_SLOPE).to_numpy()
    return (beta - INCUMBENT_SLOPE) * np.asarray(residual, dtype=float) + np.asarray(
        offset, dtype=float
    )


def week_blocked_slope(
    x: np.ndarray, y: np.ndarray, blocks: Sequence[str], *, draws: int = 20_000
) -> dict[str, float]:
    """The slope with a whole-week block bootstrap; within-week correlation zero.

    Draws whose resampled weeks carry no variation in ``x`` have no slope and
    are left out of the quantiles, never padded.
    """

    if x.size < 3:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
            "n": int(x.size),
            "weeks": 0,
        }
    codes = pd.factorize(pd.Series(list(blocks)).astype(str))[0]
    n = np.bincount(codes).astype(float)
    sx = np.bincount(codes, weights=x)
    sy = np.bincount(codes, weights=y)
    sxy = np.bincount(codes, weights=x * y)
    sxx = np.bincount(codes, weights=x * x)

    def slope(index: np.ndarray) -> np.ndarray:
        tn, tx, ty = n[index].sum(axis=-1), sx[index].sum(axis=-1), sy[index].sum(axis=-1)
        txy, txx = sxy[index].sum(axis=-1), sxx[index].sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (txy - tx * ty / tn) / (txx - tx * tx / tn)

    rng = np.random.default_rng(common.SEED)
    sampled = slope(rng.integers(0, n.size, size=(draws, n.size)))
    finite = sampled[np.isfinite(sampled)]
    return {
        "estimate": float(slope(np.arange(n.size))),
        "lower": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "upper": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "probability_positive": float(probability_positive_from_draws(finite))
        if finite.size
        else float("nan"),
        "n": int(x.size),
        "weeks": int(n.size),
    }


def slope_intervals(prior: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Bootstrap intervals for every bucket slope, plus the pooled 7+ slope."""

    x, y, buckets = slope_inputs(prior)
    completed = prior.loc[prior["result"].notna() & prior["point_incumbent"].notna()]
    weeks = (
        completed["season"].astype(int).astype(str)
        + "-"
        + completed["week"].astype(int).astype(str)
    ).tolist()
    out: dict[str, dict[str, float]] = {}
    for bucket in (*BUCKETS, "7plus_pooled"):
        mask = (
            buckets.isin(BIG_BUCKETS).to_numpy()
            if bucket == "7plus_pooled"
            else buckets.eq(bucket).to_numpy()
        )
        interval = week_blocked_slope(
            x[mask], y[mask], [w for w, m in zip(weeks, mask, strict=True) if m]
        )
        interval["shrunk"] = shrink(
            interval["estimate"] if np.isfinite(interval["estimate"]) else INCUMBENT_SLOPE,
            interval["n"],
        )
        out[bucket] = interval
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    """Reproduce S3 exactly, then score R1/R1b on the same weekly refits."""

    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    stream = archive_prior_stream(archive)
    batches: list[pd.DataFrame] = []
    slopes: list[dict[str, object]] = []
    for (season, week), group in archive.groupby(["season", "week"], sort=True):
        scoring = features.set_index("game_id").loc[group.game_id].reset_index()
        scoring["spread_line"] = group.tue_open_home_spread.to_numpy()
        training = completed.loc[completed.gameday.lt(scoring.gameday.min())]
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=active["regressor"],
            feature_profile=active["feature_profile"],
            ridge_alpha=active["ridge_alpha"],
        )
        prior = prior_rows_before(stream, int(season), int(week))
        offset = fit_home_side_offsets(prior).offset_for(scoring.spread_line).to_numpy()
        residual_raw = model.predict(
            scoring, probability_method="gaussian_median"
        ).predicted_market_residual.to_numpy()
        fits = {arm: fit_slopes(prior, pooled=arm == "R1b") for arm in ARMS}
        shifts = {"S3": offset}
        for arm in ARMS:
            shifts[arm] = center_shift(fits[arm].betas, scoring.spread_line, residual_raw, offset)
        row = group[["game_id"]].copy()
        row["residual_raw"] = residual_raw
        for arm, shift in shifts.items():
            prediction = model.predict(
                scoring, probability_method="gaussian_median", center_offset=shift
            )
            row[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
            row[f"shift_{arm}"] = shift
            row[f"residual_{arm}"] = prediction.predicted_market_residual.to_numpy()
        for arm in ARMS:
            row[f"beta_{arm}"] = (
                spread_bucket(scoring.spread_line)
                .map(fits[arm].betas)
                .astype(float)
                .fillna(INCUMBENT_SLOPE)
                .to_numpy()
            )
            slopes.extend(
                {
                    "season": int(season),
                    "week": int(week),
                    "arm": arm,
                    "bucket": bucket,
                    "prior_games": fits[arm].games[bucket],
                    "slope_raw": fits[arm].raw[bucket],
                    "beta": fits[arm].betas[bucket],
                }
                for bucket in BUCKETS
            )
        gap = float(
            np.max(np.abs(row.p_S3.to_numpy() - group.home_cover_probability_at_open.to_numpy()))
        )
        if gap > 1e-9:
            raise ValueError(f"STOP: S3 replay mismatch {season}/{week}: {gap}")
        batches.append(row)
        if int(week) == 1:
            print(f"S3 replay verified {season}", flush=True)
    result = archive.merge(pd.concat(batches), on="game_id", validate="one_to_one")
    common.table(result, OUT / "replay.parquet")
    common.table(pd.DataFrame(slopes), OUT / "slopes.parquet")
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(common.FEATURES),
            "predeclaration_sha256": sha256_file(PREDECLARATION),
            "max_probability_gap": float(
                (result.p_S3 - result.home_cover_probability_at_open).abs().max()
            ),
            "max_offset_gap": float(
                (result.shift_S3 - result.home_side_offset_at_open).abs().max()
            ),
            "max_raw_residual_gap": float(
                (result.residual_raw - result.residual_at_open).abs().max()
            ),
            "games": len(result),
        },
        OUT / "reproduction.json",
    )


def bucket_tag(bucket: str) -> str:
    return str(bucket).replace(".", "p").replace("+", "plus")


def cell_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Overall, per season, per bucket, per incumbent pick side, and crossed."""

    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups += [(f"season_{int(season)}", g) for season, g in frame.groupby("season")]
    groups += [
        (f"bucket_{bucket_tag(bucket)}", g)
        for bucket, g in frame.groupby("bucket", observed=True, sort=True)
    ]
    groups += [(f"pick_{side}", g) for side, g in frame.groupby("pick_side", sort=True)]
    groups += [
        (f"bucket_{bucket_tag(bucket)}_pick_{side}", g)
        for (bucket, side), g in frame.groupby(["bucket", "pick_side"], observed=True, sort=True)
    ]
    return [(label, group) for label, group in groups if len(group)]


def score(archive_path: Path) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    _, base_card = common.composed_picks(frame, archive_path / "per_game.parquet")
    frame["card_S3"] = base_card
    for arm in ARMS:
        candidate = frame.copy()
        candidate["home_cover_probability_at_open"] = candidate[f"p_{arm}"]
        candidate["residual_at_open_served"] = candidate[f"residual_{arm}"]
        candidate["home_side_offset_at_open"] = candidate[f"shift_{arm}"]
        for suffix, pick in [
            ("", candidate[f"residual_{arm}"].gt(0)),
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
            active_model_id=f"research_laneR_{arm}",
            research_scope="Opener only; close columns retain S3, not candidate evidence.",
        )
        metadata["active_model_config"] = {
            **metadata["active_model_config"],
            "model_id": f"research_laneR_{arm}",
        }
        write_stamped_artifact(metadata, directory / "metadata.json")
        lane_s.original_cli(
            OUT,
            "overlay-composition",
            "--per-game-artifact",
            str(directory / "per_game.parquet"),
            "--bootstrap-samples",
            "20000",
            "--bootstrap-seed",
            str(common.SEED),
        )
        _, card = common.composed_picks(candidate, directory / "per_game.parquet")
        frame[f"card_{arm}"] = card
    frame["bucket"] = spread_bucket(frame.tue_open_home_spread)
    frame["pick_side"] = np.where(frame.p_S3.ge(0.5), "home", "road")
    cells: dict[str, dict] = {}
    for arm in ARMS:
        for label, group in cell_groups(frame):
            decided = group.margin_vs_open.notna() & group.margin_vs_open.ne(0)
            if not decided.any():
                continue
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                cells[f"r1_{arm.lower()}_{label}_{kind}"] = common.comparison(
                    group, np.asarray(cp), np.asarray(bp)
                )
            valid = group.loc[decided].reset_index(drop=True)
            y = valid.margin_vs_open.gt(0).to_numpy(float)
            p, b = [np.clip(valid[f"p_{a}"].to_numpy(), 1e-9, 1 - 1e-9) for a in (arm, "S3")]
            for kind, candidate_loss, baseline_loss in [
                ("brier", (p - y) ** 2, (b - y) ** 2),
                (
                    "log_loss",
                    -(y * np.log(p) + (1 - y) * np.log1p(-p)),
                    -(y * np.log(b) + (1 - y) * np.log1p(-b)),
                ),
            ]:
                cells[f"r1_{arm.lower()}_{label}_{kind}"] = {
                    **lane_s.metric(valid, baseline_loss - candidate_loss),
                    "candidate": float(candidate_loss.mean()),
                    "baseline": float(baseline_loss.mean()),
                }
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    print(
        json.dumps({key: value for key, value in cells.items() if "overall" in key}, indent=2),
        flush=True,
    )


def week1(active: dict, archive_path: Path) -> None:
    """Week 1 2026 sides under each arm; the linked card is only READ."""

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
    offset = recorded.home_side_offset.to_numpy(dtype=float)
    base = model.predict(scoring, probability_method="gaussian_median", center_offset=offset)
    gap = float(
        np.abs(
            base.home_cover_probability.to_numpy() - recorded.home_cover_probability.to_numpy()
        ).max()
    )
    if gap > 1e-9:
        raise ValueError(f"Week 1 sidecar replay mismatch: {gap}")
    residual_raw = model.predict(
        scoring, probability_method="gaussian_median"
    ).predicted_market_residual.to_numpy()
    prior = prior_rows_before(
        archive_prior_stream(pd.read_parquet(archive_path / "per_game.parquet")), 2026, 1
    )
    rows = scoring[["game_id", "home_team", "away_team", "spread_line"]].copy()
    rows["bucket"] = spread_bucket(scoring.spread_line)
    rows["residual_raw"] = residual_raw
    rows["offset_S3"] = offset
    rows["p_S3"] = recorded.home_cover_probability.to_numpy()
    fitted: dict[str, dict[str, object]] = {}
    changes: dict[str, list[dict]] = {}
    for arm in ARMS:
        fit = fit_slopes(prior, pooled=arm == "R1b")
        shift = center_shift(fit.betas, scoring.spread_line, residual_raw, offset)
        prediction = model.predict(
            scoring, probability_method="gaussian_median", center_offset=shift
        )
        rows[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
        rows[f"point_{arm}"] = prediction.predicted_margin.to_numpy()
        rows[f"shift_{arm}"] = shift
        rows[f"beta_{arm}"] = (
            spread_bucket(scoring.spread_line)
            .map(fit.betas)
            .astype(float)
            .fillna(INCUMBENT_SLOPE)
            .to_numpy()
        )
        fitted[arm] = {"betas": fit.betas, "slope_raw": fit.raw, "prior_games": fit.games}
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        changes[arm] = rows.loc[changed].to_dict(orient="records")
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_replay_gap": gap,
            "games": len(rows),
            "prior_rows": len(prior),
            "fitted": fitted,
            "positive_control": slope_intervals(prior),
            "changes": changes,
        },
        OUT / "week1.json",
    )
    print(json.dumps(fitted, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Record commands (written, never run: the shared registry is serialised by
# the coordinator this session)
# ---------------------------------------------------------------------------


def scope_words(label: str) -> str:
    if label == "overall":
        return "all games from 2020 to 2025"
    parts: list[str] = []
    for token in label.split("_pick_"):
        if token.startswith("season_"):
            parts.append(f"the {token.removeprefix('season_')} season")
        elif token.startswith("bucket_"):
            tag = token.removeprefix("bucket_")
            match = next(b for b in BUCKETS if bucket_tag(b) == tag)
            parts.append(f"games where {BUCKET_WORDS[match]}")
        elif token in {"home", "road"}:
            parts.append(f"games where the model picked the {token} team")
    return " and ".join(parts) if parts else label


def plain_summary(arm: str, label: str, kind: str) -> str:
    return f"{ARM_WORDS[arm]}. This row reports {KIND_WORDS[kind]}, on {scope_words(label)}."


def quoted(value: str) -> str:
    text = " ".join(str(value).split())
    for banned in ('"', "`", "$"):
        text = text.replace(banned, "")
    return f'"{text}"'


def record_commands() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    source = "artifacts/research/laneR/cells.json"
    lines = [
        "# MOD-18 lane R weak-signal records, written by "
        "scripts/residual_slope_opener_eval.py --stage record.",
        "# The lane never runs these: the shared registry is serialised by the coordinator.",
        "# Run from the repository root, serially, in this order. A command that ERRORS means",
        "# the verdict is wrong, not the validator (AGENTS.md closing-grounds taxonomy).",
    ]
    for name, metrics in sorted(cells.items()):
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        arm = "R1b" if name.startswith("r1_r1b_") else "R1"
        rest = name.removeprefix(f"r1_{arm.lower()}_")
        kind = next(k for k in KIND_WORDS if rest.endswith(k))
        label = rest.removesuffix(f"_{kind}").rstrip("_")
        season = int(label.removeprefix("season_")) if label.startswith("season_") else None
        start, end = (season, season) if season else (2020, 2025)
        arguments = [
            ".\\.tools\\uv.exe run --no-sync nfl-ats weak-signals record",
            f"--name {FAMILY}_{name}_{start}_{end}",
            f"--family {FAMILY}",
            "--description "
            + quoted(
                f"Residual-slope arm {arm} versus the served S3, {label} {kind}; "
                "positive favours the candidate"
            ),
            f"--source {source}",
            "--classification unresolved_below_power",
            "--classification-evidence "
            + quoted(
                "No resolved wrong sign, split-half reliability or positive-control closing "
                "ground established; retained unresolved."
            ),
            "--league nfl",
            f"--season-start {start}",
            f"--season-end {end}",
            f"--sample-games {metrics['n']}",
            f"--sample-blocks {metrics['weeks']}",
            "--category modeling",
            f"--effect {lane_s.fixed(metrics['delta'])}",
            f"--effect-units {KIND_UNITS[kind]}",
            f"--interval-low {lane_s.fixed(metrics['lower'])}",
            f"--interval-high {lane_s.fixed(metrics['upper'])}",
            f"--probability-positive {lane_s.fixed(metrics['probability_positive'])}",
            "--plain-summary " + quoted(plain_summary(arm, label, kind)),
            "--notes " + quoted(DISCOUNT),
        ]
        lines.append(" ".join(arguments))
    path = OUT / "record_commands.ps1"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    stamp_sidecar(path, {"commands": len(lines) - 4})
    print(f"wrote {len(lines) - 4} record commands to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replay", "score", "week1", "record"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    # Never the live registry: this lane writes its record commands to a file.
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(OUT / "registry")
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
    if match is None:
        raise ValueError("No matching opener evaluation")
    with threadpool_limits(limits=1):
        if args.stage == "replay":
            replay(pd.read_parquet(match[1] / "per_game.parquet"), active, match[1])
        elif args.stage == "score":
            score(match[1])
        elif args.stage == "week1":
            week1(active, match[1])
        else:
            record_commands()


if __name__ == "__main__":
    main()

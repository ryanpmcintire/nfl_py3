"""Frozen MOD-18 lane U residual-slope refit; see docs/residual_slope_shrunk.md.

Lane R rescaled the model's raw residual by a walk-forward per-bucket slope
shrunk toward the incumbent weight 1.0 with the served offset's 100-game count
prior, and lost through the played card. Lane U keeps the mechanism and
derives the shrinkage instead:

* ``R2`` shrinks each bucket's slope toward 1.0 by ``tau^2 / (tau^2 + se^2)``
  -- empirical Bayes from the slope's OWN standard error, with ``tau^2`` the
  between-bucket variance of the estimates about the incumbent -- so a noisy
  bucket stays near 1 and only a precisely-estimated departure moves.
* ``R2b`` keeps lane R's 100-game shrinkage but serves it ONLY in ``10.5+``,
  the bucket lane L diagnosed. Disclosed as post-hoc bucket restriction.

``tau^2 = 0`` gives ``beta = 1`` in every bucket, which is exactly S3, so the
incumbent is reproduced rather than approximated. Everything else -- the
walk-forward exclusions, the unchanged S3 offset, the ``gaussian_median``
mapping through ``MarginModel.predict(center_offset=...)``, the centre-shift
algebra, the bootstrap and the positive control -- is imported from
``scripts/residual_slope_opener_eval.py`` rather than re-implemented.

Every write lands under ``artifacts/research/laneU``; the live registry is
never touched -- the ``record`` stage WRITES the record commands to a
PowerShell file for the coordinator to run serially instead of running them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import home_side_location_opener_eval as lane_s
import numpy as np
import pandas as pd
import residual_slope_opener_eval as lane_r
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import (
    archive_prior_stream,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.spread_regime import BUCKETS, spread_bucket

OUT = common.REPO / "artifacts/research/laneU"
PREDECLARATION = common.REPO / "docs/residual_slope_shrunk.md"
FAMILY = lane_r.FAMILY
ARMS = ("R2", "R2b")
INCUMBENT_SLOPE = lane_r.INCUMBENT_SLOPE
RESTRICTED_BUCKET = "10.5+"
MIN_SLOPE_ROWS = 4
DISCOUNT = (
    "Post-hoc refit of a post-hoc mechanism on the mined archive S2/S3 and lane R were all "
    "scored on; the shrinkage defect and the 10.5+ bucket R2b restricts to were both diagnosed "
    "on these same games; strongly correlated with lane R's R1/R1b by construction; "
    "walk-forward fitting removes look-ahead within a game but not the archive-level "
    "selection; descriptive reuse, not independent confirmation."
)
ARM_WORDS = {
    "R2": (
        "On the biggest lines the model's own disagreement with the betting number has been "
        "pointing the wrong way. This version works out from earlier seasons how much of that "
        "disagreement to keep at each size of line AND how firmly the record pins that amount "
        "down, so a size of line where the record is murky is left alone and only a clear "
        "pattern is allowed to change anything"
    ),
    "R2b": (
        "The same idea, except the change is made only on lines of 10.5 points or more, where "
        "the problem was found, and every other size of line is left exactly as it is played "
        "today"
    ),
}


def slope_and_standard_error(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y`` on ``x`` and its classical standard error.

    Within-week correlation is zero by mandate, so there is no clustering term:
    the homoskedastic ``sqrt(RSS / (n - 2) / Sxx)`` is the declared estimator.
    A bucket that cannot support one -- fewer than four rows, no variation in
    ``x``, or a degenerate (zero or non-finite) sampling variance -- returns
    the incumbent slope and ``nan``, which marks it not estimable: it serves
    ``beta = 1`` and contributes nothing to ``tau^2``.
    """

    n = int(x.size)
    if n < MIN_SLOPE_ROWS:
        return lane_r.ols_slope(x, y), float("nan")
    sxx = float((x * x).sum() - x.sum() ** 2 / n)
    if not sxx > 0.0:
        return INCUMBENT_SLOPE, float("nan")
    slope = float((x * y).sum() - x.sum() * y.sum() / n) / sxx
    intercept = float(y.mean() - slope * x.mean())
    residual_sum_of_squares = float(((y - intercept - slope * x) ** 2).sum())
    variance = residual_sum_of_squares / (n - 2) / sxx
    if not math.isfinite(variance) or variance <= 0.0:
        return slope, float("nan")
    return slope, math.sqrt(variance)


def tau_squared(estimates: Mapping[str, float], errors: Mapping[str, float]) -> float:
    """Between-bucket variance of the slopes about the INCUMBENT 1.0.

    DerSimonian-Laird's moment identity with the centre fixed at the shrinkage
    target instead of estimated: under ``beta_hat_b ~ N(1 + u_b, se_b^2)`` with
    ``u_b ~ N(0, tau^2)``, ``E[Q] = K + tau^2 * sum_b w_b``, so
    ``tau^2 = max(0, (Q - K) / sum_b w_b)``. Standard DL centres ``Q`` on the
    precision-weighted mean, which estimates dispersion about THAT mean and is
    not the prior variance a shrinkage toward the incumbent needs.
    """

    estimable = [bucket for bucket in estimates if math.isfinite(errors.get(bucket, float("nan")))]
    if not estimable:
        return 0.0
    weights = {bucket: 1.0 / errors[bucket] ** 2 for bucket in estimable}
    q = sum(weights[b] * (estimates[b] - INCUMBENT_SLOPE) ** 2 for b in estimable)
    return max(0.0, (q - len(estimable)) / sum(weights.values()))


def shrinkage_fraction(standard_error: float, tau2: float) -> float:
    """``tau^2 / (tau^2 + se^2)``: the share of the departure from 1.0 served."""

    if not math.isfinite(standard_error) or tau2 <= 0.0:
        return 0.0
    return float(tau2 / (tau2 + standard_error**2))


def empirical_bayes(slope: float, standard_error: float, tau2: float) -> float:
    """``beta* = 1 + (beta_hat - 1) * tau^2 / (tau^2 + se^2)``."""

    return INCUMBENT_SLOPE + (slope - INCUMBENT_SLOPE) * shrinkage_fraction(standard_error, tau2)


@dataclass(frozen=True)
class ShrunkFit:
    """One target week's served weights, with everything needed to audit them."""

    betas: dict[str, float]
    raw: dict[str, float]
    standard_error: dict[str, float]
    games: dict[str, int]
    fraction: dict[str, float]
    tau_squared: float


def fit_shrunk(prior: pd.DataFrame, *, restricted: bool) -> ShrunkFit:
    """R2 (``restricted=False``) or R2b (``restricted=True``) for one week."""

    x, y, buckets = lane_r.slope_inputs(prior)
    raw: dict[str, float] = {}
    errors: dict[str, float] = {}
    games: dict[str, int] = {}
    for bucket in BUCKETS:
        mask = buckets.eq(bucket).to_numpy()
        games[bucket] = int(mask.sum())
        raw[bucket], errors[bucket] = slope_and_standard_error(x[mask], y[mask])
    if restricted:
        betas = dict.fromkeys(BUCKETS, INCUMBENT_SLOPE)
        fraction = dict.fromkeys(BUCKETS, 0.0)
        count = games[RESTRICTED_BUCKET]
        betas[RESTRICTED_BUCKET] = lane_r.shrink(raw[RESTRICTED_BUCKET], count)
        fraction[RESTRICTED_BUCKET] = count / (count + 100.0)
        return ShrunkFit(betas, raw, errors, games, fraction, float("nan"))
    tau2 = tau_squared(raw, errors)
    fraction = {b: shrinkage_fraction(errors[b], tau2) for b in BUCKETS}
    betas = {b: empirical_bayes(raw[b], errors[b], tau2) for b in BUCKETS}
    return ShrunkFit(betas, raw, errors, games, fraction, tau2)


def finite_json(payload: object) -> object:
    """``nan`` is not JSON; a bucket with no estimable slope writes ``null``."""

    if isinstance(payload, Mapping):
        return {key: finite_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [finite_json(value) for value in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def beta_column(betas: Mapping[str, float], lines: pd.Series) -> np.ndarray:
    return (
        spread_bucket(lines)
        .map(dict(betas))
        .astype(float)
        .fillna(INCUMBENT_SLOPE)
        .to_numpy(dtype=float)
    )


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    """Reproduce S3 exactly, then score R2/R2b on the same weekly refits."""

    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    stream = archive_prior_stream(archive)
    batches: list[pd.DataFrame] = []
    weights: list[dict[str, object]] = []
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
        fits = {arm: fit_shrunk(prior, restricted=arm == "R2b") for arm in ARMS}
        shifts = {"S3": offset}
        for arm in ARMS:
            shifts[arm] = lane_r.center_shift(
                fits[arm].betas, scoring.spread_line, residual_raw, offset
            )
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
            row[f"beta_{arm}"] = beta_column(fits[arm].betas, scoring.spread_line)
            weights.extend(
                {
                    "season": int(season),
                    "week": int(week),
                    "arm": arm,
                    "bucket": bucket,
                    "prior_games": fits[arm].games[bucket],
                    "slope_raw": fits[arm].raw[bucket],
                    "standard_error": fits[arm].standard_error[bucket],
                    "shrinkage_fraction": fits[arm].fraction[bucket],
                    "tau_squared": fits[arm].tau_squared,
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
    common.table(pd.DataFrame(weights), OUT / "slopes.parquet")
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(common.FEATURES),
            "predeclaration_sha256": sha256_file(PREDECLARATION),
            "lane_r_script_sha256": sha256_file(
                common.REPO / "scripts/residual_slope_opener_eval.py"
            ),
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
            active_model_id=f"research_laneU_{arm}",
            research_scope="Opener only; close columns retain S3, not candidate evidence.",
        )
        metadata["active_model_config"] = {
            **metadata["active_model_config"],
            "model_id": f"research_laneU_{arm}",
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
        for label, group in lane_r.cell_groups(frame):
            decided = group.margin_vs_open.notna() & group.margin_vs_open.ne(0)
            if not decided.any():
                continue
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                cells[f"r2_{arm.lower()}_{label}_{kind}"] = common.comparison(
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
                cells[f"r2_{arm.lower()}_{label}_{kind}"] = {
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
    """Week 1 2026 weights and sides under each arm; the card is only READ."""

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
        fit = fit_shrunk(prior, restricted=arm == "R2b")
        shift = lane_r.center_shift(fit.betas, scoring.spread_line, residual_raw, offset)
        prediction = model.predict(
            scoring, probability_method="gaussian_median", center_offset=shift
        )
        rows[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
        rows[f"point_{arm}"] = prediction.predicted_margin.to_numpy()
        rows[f"shift_{arm}"] = shift
        rows[f"beta_{arm}"] = beta_column(fit.betas, scoring.spread_line)
        fitted[arm] = {
            "betas": fit.betas,
            "slope_raw": fit.raw,
            "standard_error": fit.standard_error,
            "shrinkage_fraction": fit.fraction,
            "prior_games": fit.games,
            "tau_squared": fit.tau_squared,
        }
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        changes[arm] = rows.loc[changed].to_dict(orient="records")
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        finite_json(
            {
                "forecast": str(forecast),
                "sidecar_replay_gap": gap,
                "games": len(rows),
                "prior_rows": len(prior),
                "fitted": fitted,
                "positive_control": lane_r.slope_intervals(prior),
                "changes": changes,
            }
        ),
        OUT / "week1.json",
    )
    print(json.dumps(finite_json(fitted), indent=2), flush=True)


def plain_summary(arm: str, label: str, kind: str) -> str:
    return (
        f"{ARM_WORDS[arm]}. This row reports {lane_r.KIND_WORDS[kind]}, "
        f"on {lane_r.scope_words(label)}."
    )


def record_commands() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    source = "artifacts/research/laneU/cells.json"
    lines = [
        "# MOD-18 lane U weak-signal records, written by "
        "scripts/residual_slope_shrunk_opener_eval.py --stage record.",
        "# The lane never runs these: the shared registry is serialised by the coordinator.",
        "# Run from the repository root, serially, in this order. A command that ERRORS means",
        "# the verdict is wrong, not the validator (AGENTS.md closing-grounds taxonomy).",
    ]
    for name, metrics in sorted(cells.items()):
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        arm = "R2b" if name.startswith("r2_r2b_") else "R2"
        rest = name.removeprefix(f"r2_{arm.lower()}_")
        kind = next(k for k in lane_r.KIND_WORDS if rest.endswith(k))
        label = rest.removesuffix(f"_{kind}").rstrip("_")
        season = int(label.removeprefix("season_")) if label.startswith("season_") else None
        start, end = (season, season) if season else (2020, 2025)
        arguments = [
            ".\\.tools\\uv.exe run --no-sync nfl-ats weak-signals record",
            f"--name {FAMILY}_{name}_{start}_{end}",
            f"--family {FAMILY}",
            "--description "
            + lane_r.quoted(
                f"Derived-shrinkage residual-slope arm {arm} versus the served S3, "
                f"{label} {kind}; positive favours the candidate"
            ),
            f"--source {source}",
            "--classification unresolved_below_power",
            "--classification-evidence "
            + lane_r.quoted(
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
            f"--effect-units {lane_r.KIND_UNITS[kind]}",
            f"--interval-low {lane_s.fixed(metrics['lower'])}",
            f"--interval-high {lane_s.fixed(metrics['upper'])}",
            f"--probability-positive {lane_s.fixed(metrics['probability_positive'])}",
            "--plain-summary " + lane_r.quoted(plain_summary(arm, label, kind)),
            "--notes " + lane_r.quoted(DISCOUNT),
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

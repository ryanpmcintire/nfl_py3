"""Frozen Lane K MP1/MP1b research; docs/mass_preserving_lattice.md.

The atoms of the conditional margin distribution stay at their ABSOLUTE
integer positions and the model's point enters as an exponential reweighting
of those atoms, so the key-number mass at 3/7/10/14 is never translated away.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import home_side_location_opener_eval as lane_s
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from scipy import optimize
from threadpoolctl import threadpool_limits

from nfl_ats.conditional_margin import BANDWIDTH
from nfl_ats.home_side_location import (
    COMPLETION_ALLOWANCE_DAYS,
    TRAILING_SEASONS,
    archive_prior_stream,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

OUT = common.REPO / "artifacts/research/laneK"
#: MP1 reuses the frozen lane-K bandwidth; MP1b is the one predeclared sibling.
ARMS = {"MP1": float(BANDWIDTH), "MP1b": 2.0 * float(BANDWIDTH)}
#: Declared floor (docs/mass_preserving_lattice.md): 200 prior games put ~20 on
#: the ~10% push atom of an integer key line, a ~22% relative standard error.
MIN_BAND_GAMES = 200
MAX_BAND = 20.0
BAND_STEP = 0.5
#: Far wider than attainable need: one point of mean shift costs theta ~0.005.
THETA_BRACKET = 1.0
FAMILY = "mod18_conditional_margin_v1"
DISCOUNT = (
    "Mass-preserving conditional read on the mined archive lanes K and S were selected on; "
    "S3 is itself a post-hoc restriction fitted on it; two correlated band widths, correlated "
    "with lane H's M1; descriptive reuse, not independent confirmation."
)
SUMMARY = (
    "Football margins pile up on 3, 7, 10 and 14, so the chance a team covers is read off how "
    "often past games at a similar spread actually finished on each of those numbers, with the "
    "model's own call tilting the weight toward its side instead of sliding the numbers around."
)


def tilted_atoms(
    margins: np.ndarray, counts: np.ndarray, line: float, target: float
) -> tuple[np.ndarray, float]:
    """Reweight fixed integer atoms to a declared mean; positions never move.

    ``p(m) ~ q(m) * exp(theta * (m - line))`` with ``theta`` solved so the
    tilted mean equals ``target``. The tilted mean is strictly increasing in
    ``theta``, so the root is unique inside the declared bracket; outside it
    the tilt is clamped to the nearer bracket end.
    """

    base = counts / counts.sum()
    if margins.size == 1:
        return base, 0.0
    offsets = margins - line

    def weights(theta: float) -> np.ndarray:
        logits = theta * offsets
        mass = base * np.exp(logits - logits.max())
        return mass / mass.sum()

    def mean_at(theta: float) -> float:
        return float((margins * weights(theta)).sum())

    low, high = -THETA_BRACKET, THETA_BRACKET
    if target <= mean_at(low):
        theta = low
    elif target >= mean_at(high):
        theta = high
    else:
        theta = float(optimize.brentq(lambda t: mean_at(t) - target, low, high, xtol=1e-13))
    return weights(theta), theta


def band_read(
    pool_line: np.ndarray, pool_margin: np.ndarray, line: float, point: float, half_width: float
) -> dict[str, float]:
    """The mass-preserving read for one game at one declared band width."""

    band = half_width
    while True:
        selected = np.abs(pool_line - line) <= band
        if int(selected.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    margins = pool_margin[selected]
    if margins.size == 0:
        raise ValueError(f"No prior games within {band} points of line {line}")
    values, counts = np.unique(margins, return_counts=True)
    mass, theta = tilted_atoms(values, counts.astype(float), line, point)
    is_push = np.abs(values - line) < 1e-9
    cover = float(mass[values > line + 1e-9].sum())
    push = float(mass[is_push].sum())
    loss = float(mass[values < line - 1e-9].sum())
    return {
        "cover": cover,
        "push": push,
        "loss": loss,
        "home_cover_probability": cover + 0.5 * push,
        "conditional_cover_probability": cover / (cover + loss) if cover + loss > 0 else 0.5,
        "theta": theta,
        "band": band,
        "band_games": int(selected.sum()),
        "atoms": int(values.size),
        "key_mass_3": float(mass[np.abs(np.abs(values) - 3) < 1e-9].sum()),
    }


def build_pool(archive: pd.DataFrame) -> pd.DataFrame:
    """Completed games with a line: the archived opener where known, else nflverse."""

    features = regular_season_rows(pd.read_parquet(common.FEATURES)).copy()
    features["gameday"] = pd.to_datetime(features.gameday)
    pool = features.loc[
        features.result.notna(), ["game_id", "season", "week", "gameday", "spread_line", "result"]
    ].copy()
    opener = archive.set_index("game_id").tue_open_home_spread
    pool["line"] = pool.game_id.map(opener).fillna(pool.spread_line)
    pool = pool.loc[pool.line.notna()].reset_index(drop=True)
    pool["result"] = np.rint(pool.result.to_numpy(dtype=float))
    return pool


def mass_preserving(pool: pd.DataFrame, targets: pd.DataFrame, half_width: float) -> pd.DataFrame:
    """Walk-forward read for every target row; prior games only, five seasons."""

    rows = []
    for (season, week), batch in targets.groupby(["season", "week"], sort=True):
        cutoff = batch.gameday.min()
        eligible = pool.loc[
            (pool.gameday + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)).lt(cutoff)
            & pool.season.ge(int(season) - TRAILING_SEASONS)
            & ~(pool.season.eq(season) & pool.week.eq(week))
            & ~pool.game_id.isin(set(batch.game_id))
        ]
        lines = eligible.line.to_numpy(dtype=float)
        margins = eligible.result.to_numpy(dtype=float)
        for _, row in batch.iterrows():
            read = band_read(lines, margins, float(row.line), float(row.point), half_width)
            rows.append({"game_id": row.game_id, "prior_rows": len(eligible), **read})
    return pd.DataFrame(rows)


def verify_replay(actual: np.ndarray, served: np.ndarray) -> float:
    """Fail closed before candidate construction, including missing probabilities."""

    if not np.isfinite(actual).all() or not np.isfinite(served).all():
        raise ValueError("STOP: S3 replay contains missing probabilities")
    gap = float(np.max(np.abs(actual - served)))
    if gap > 1e-9:
        raise ValueError(f"STOP: S3 replay mismatch: {gap}")
    return gap


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    stream = archive_prior_stream(archive)
    batches = []
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
        prediction = model.predict(
            scoring, probability_method="gaussian_median", center_offset=offset
        )
        row = group[["game_id"]].copy()
        row["p_S3"] = prediction.home_cover_probability.to_numpy()
        row["push_S3"] = prediction.push_probability.to_numpy()
        row["offset_S3"] = offset
        row["residual_S3"] = prediction.predicted_market_residual.to_numpy()
        row["median_S3"] = float(np.median(model.residuals))
        row["center_S3"] = prediction.predicted_margin.to_numpy() + float(
            np.median(model.residuals)
        )
        verify_replay(row.p_S3.to_numpy(), group.home_cover_probability_at_open.to_numpy())
        batches.append(row)
        if int(week) == 1:
            print(f"S3 replay verified {season}", flush=True)
    result = archive.merge(pd.concat(batches), on="game_id", validate="one_to_one")
    common.table(result, OUT / "replay.parquet")
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(common.FEATURES),
            "predeclaration_sha256": sha256_file(common.REPO / "docs/mass_preserving_lattice.md"),
            "max_probability_gap": float(
                (result.p_S3 - result.home_cover_probability_at_open).abs().max()
            ),
            "max_offset_gap": float(
                (result.offset_S3 - result.home_side_offset_at_open).abs().max()
            ),
            "median_residual_range": [float(result.median_S3.min()), float(result.median_S3.max())],
        },
        OUT / "reproduction.json",
    )


def map_replay(archive: pd.DataFrame) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    verify_replay(frame.p_S3.to_numpy(), frame.home_cover_probability_at_open.to_numpy())
    pool = build_pool(archive)
    common.table(pool, OUT / "pool.parquet")
    targets = frame[["game_id", "season", "week"]].copy()
    targets["gameday"] = pool.set_index("game_id").gameday.reindex(frame.game_id).to_numpy()
    targets["line"] = frame.tue_open_home_spread.to_numpy()
    targets["point"] = frame.center_S3.to_numpy()
    if targets.gameday.isna().any():
        raise ValueError("Archive rows without a gameday in the feature table")
    for arm, half_width in ARMS.items():
        mapped = mass_preserving(pool, targets, half_width).set_index("game_id")
        mapped = mapped.reindex(frame.game_id)
        for column in mapped:
            frame[f"{column}_{arm}"] = mapped[column].to_numpy()
        frame[f"p_{arm}"] = mapped.home_cover_probability.to_numpy()
        frame[f"push_{arm}"] = mapped.push.to_numpy()
        frame[f"offset_{arm}"] = frame.offset_S3
        frame[f"residual_{arm}"] = frame.residual_S3
        print(f"mapped {arm}", flush=True)
    common.table(frame, OUT / "replay.parquet")


def research_per_game(frame: pd.DataFrame, arm: str, archive_path: Path) -> Path:
    candidate = frame.copy()
    candidate["home_cover_probability_at_open"] = candidate[f"p_{arm}"]
    candidate["home_side_offset_at_open"] = candidate[f"offset_{arm}"]
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
        active_model_id=f"research_laneK_{arm}",
        research_scope="Opener only; close columns retain S3, not candidate evidence.",
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": f"research_laneK_{arm}",
    }
    write_stamped_artifact(metadata, directory / "metadata.json")
    return directory


def loss_cells(group: pd.DataFrame, column: str) -> dict[str, dict]:
    valid = group.loc[group.margin_vs_open.notna() & group.margin_vs_open.ne(0)].reset_index(
        drop=True
    )
    truth = valid.margin_vs_open.gt(0).to_numpy(dtype=float)
    candidate = np.clip(valid[column].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    baseline = np.clip(valid.p_S3.to_numpy(dtype=float), 1e-9, 1 - 1e-9)
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


def score(archive_path: Path) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    _, base_card = common.composed_picks(frame, archive_path / "per_game.parquet")
    frame["card_S3"] = base_card
    for arm in ARMS:
        directory = research_per_game(frame, arm, archive_path)
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
        candidate = pd.read_parquet(directory / "per_game.parquet")
        _, card = common.composed_picks(candidate, directory / "per_game.parquet")
        frame[f"card_{arm}"] = card
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    groups = [("overall", frame)]
    groups += [(f"season_{season}", g) for season, g in frame.groupby("season")]
    groups += [
        (lane_s.cell_name("cell", "bucket", str(bucket), "all"), g)
        for bucket, g in frame.groupby("bucket", observed=True)
    ]
    cells: dict[str, dict] = {}
    for arm in ARMS:
        for label, group in groups:
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                cells[f"mp1_{arm.lower()}_{label}_{kind}"] = common.comparison(
                    group, cp.to_numpy(), bp.to_numpy()
                )
            for kind, metrics in loss_cells(group, f"p_{arm}").items():
                cells[f"mp1_{arm.lower()}_{label}_{kind}"] = metrics
        # Declared diagnostic: the push-renormalised read, overall only.
        diagnostic = loss_cells(frame, f"conditional_cover_probability_{arm}")
        for kind, metrics in diagnostic.items():
            cells[f"mp1_{arm.lower()}_overall_conditional_{kind}"] = metrics
    push_rows = []
    for size in (3.0, 7.0):
        block = frame.loc[frame.tue_open_home_spread.abs().eq(size)].reset_index(drop=True)
        realized = block.margin_vs_open.eq(0).to_numpy(dtype=float)
        for arm in ("S3", *ARMS):
            predicted = block[f"push_{arm}"].to_numpy(dtype=float)
            push_rows.append(
                {
                    "line_size": size,
                    "arm": arm,
                    "n": len(block),
                    "predicted": float(predicted.mean()),
                    "realized": float(realized.mean()),
                }
            )
            if arm != "S3":
                base_loss = (block.push_S3.to_numpy(dtype=float) - realized) ** 2
                candidate_loss = (predicted - realized) ** 2
                cells[f"mp1_{arm.lower()}_push_at_{int(size)}_brier"] = {
                    **lane_s.metric(block, base_loss - candidate_loss),
                    "candidate": float(candidate_loss.mean()),
                    "baseline": float(base_loss.mean()),
                }
    write_stamped_artifact({"rows": push_rows}, OUT / "push_calibration.json")
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    print(json.dumps({k: v for k, v in cells.items() if "overall" in k}, indent=2), flush=True)


def week1(active: dict) -> None:
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
    pool = pd.read_parquet(OUT / "pool.parquet")
    targets = scoring[["game_id", "season", "week"]].copy()
    targets["gameday"] = pd.to_datetime(scoring.gameday)
    targets["line"] = scoring.spread_line.to_numpy()
    targets["point"] = base.predicted_margin.to_numpy() + float(np.median(model.residuals))
    changes = {}
    for arm, half_width in ARMS.items():
        mapped = mass_preserving(pool, targets, half_width).set_index("game_id")
        mapped = mapped.reindex(rows.game_id)
        rows[f"p_{arm}"] = mapped.home_cover_probability.to_numpy()
        rows[f"push_{arm}"] = mapped.push.to_numpy()
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        changes[arm] = rows.loc[changed].to_dict(orient="records")
    from nfl_ats.four_overlay_composition import apply_four_overlay_composition_for_publication
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    schedules, _ = load_snapshot(latest_snapshot(common.REPO / "data/raw"))
    card_changes = {}
    for arm in ("S3", *ARMS):
        candidate = scoring.copy()
        candidate["home_cover_probability"] = rows[f"p_{arm}"].to_numpy()
        composed = apply_four_overlay_composition_for_publication(
            candidate, schedules, common.REPO / "data"
        ).overlaid_predictions
        rows[f"card_home_{arm}"] = composed.home_cover_probability.ge(0.5).to_numpy()
        if arm != "S3":
            changed = rows[f"card_home_{arm}"].ne(rows.card_home_S3)
            card_changes[arm] = rows.loc[changed].to_dict(orient="records")
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_replay_gap": gap,
            "games": len(rows),
            "changes": changes,
            "card_changes": card_changes,
        },
        OUT / "week1.json",
    )


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def record_argv(name: str, metrics: dict, units: str, start: int, end: int) -> list[str]:
    """The exact recorder argv for one comparison cell."""

    return [
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        f"Mass-preserving conditional margin arm {name}, positive favours the candidate",
        "--source",
        str(OUT / "cells.json"),
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
        str(metrics["n"]),
        "--sample-blocks",
        str(metrics["weeks"]),
        "--category",
        "modeling",
        "--effect",
        lane_s.fixed(metrics["delta"]),
        "--effect-units",
        units,
        "--interval-low",
        lane_s.fixed(metrics["lower"]),
        "--interval-high",
        lane_s.fixed(metrics["upper"]),
        "--probability-positive",
        lane_s.fixed(metrics["probability_positive"]),
        "--standard-error",
        lane_s.fixed(metrics["standard_error"]),
        "--notes",
        DISCOUNT,
        "--plain-summary",
        SUMMARY,
        "--replace",
    ]


def record() -> None:
    """Emit the recorder commands; the coordinator runs them serially.

    Coordinator instruction, 2026-09-08: ``registry/weak_signals.json`` was
    corrupted by two lanes recording at once (``nfl_ats.io.atomic_json`` writes
    every payload through the same ``<destination>.tmp`` sibling, so concurrent
    writers interleave), so research lanes emit their argv instead of running it.
    """

    cells = json.loads((OUT / "cells.json").read_text())
    lines = [
        "# MOD-18 lane K (docs/mass_preserving_lattice.md) weak-signal records.",
        "# Run SERIALLY, from the repository root, with no other lane recording.",
        "# --replace is set so a re-run is idempotent; 15 of these rows were",
        "# already written before the coordinator's serialisation instruction.",
        "$ErrorActionPreference = 'Stop'",
    ]
    names = []
    for name, metrics in cells.items():
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        units = (
            "brier_improvement"
            if name.endswith("brier")
            else "log_loss_improvement"
            if name.endswith("log_loss")
            else "accuracy_points"
        )
        season = int(name.split("season_")[1].split("_")[0]) if "season_" in name else None
        argv = record_argv(name, metrics, units, season or 2020, season or 2025)
        quoted = " ".join(powershell_quote(token) for token in argv)
        lines.append(f".\\.tools\\uv.exe run --no-sync nfl-ats {quoted}")
        names.append(argv[argv.index("--name") + 1])
    path = OUT / "record_commands.ps1"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stamped_artifact(
        {
            "names": sorted(names),
            "count": len(names),
            "commands": str(path),
            "execution": "emitted, not run: the coordinator runs them serially",
        },
        OUT / "registry_names.json",
    )
    print(f"emitted {len(names)} recorder commands to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("replay", "map", "score", "week1", "record"), required=True
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    # Never point a research stage at the live registry: the recorder argv is
    # emitted for serial execution by the coordinator, never run from here.
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
            record()


if __name__ == "__main__":
    main()

"""Frozen Lane T KL1/KL1b research; docs/key_line_lattice.md.

Lane K's mass-preserving read (MP1) is served ONLY where the opener line is
quoted exactly on a key atom -- 3, 7, 10 or 14, either sign -- and every other
game keeps the served S3 probability bit-for-bit. The construction itself is
imported from ``mass_preserving_lattice_opener_eval``; nothing about it is
re-tuned or reimplemented here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import home_side_location_opener_eval as lane_s
import mass_preserving_lattice_opener_eval as lane_k
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.conditional_margin import BANDWIDTH
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

OUT = common.REPO / "artifacts/research/laneT"
#: The atoms MOD-05 named and AGENTS.md makes binding, in absolute points.
KEY_NUMBERS = (3.0, 7.0, 10.0, 14.0)
#: KL1 is the primary form; KL1b is the one predeclared sibling (3 and 7 only).
ARMS: dict[str, tuple[float, ...]] = {"KL1": KEY_NUMBERS, "KL1b": (3.0, 7.0)}
FAMILY = "mod18_conditional_margin_v1"
PREFIX = "kl1"
DISCOUNT = (
    "Post-hoc restriction of lane K's mass-preserving read to key-number lines, chosen after "
    "seeing MP1's bucket-7 gain on these same games; mined archive lanes K and S were selected "
    "on; S3 is itself a post-hoc restriction fitted on it; two nested correlated arms, "
    "correlated with lane K's MP1 and lane H's M1; descriptive reuse, not independent "
    "confirmation."
)
BASE_SUMMARIES = {
    "KL1": (
        "Football scores bunch up on 3, 7, 10 and 14. When the spread is set exactly on one of "
        "those four numbers, this version reads the chance of covering from how often past games "
        "at a similar spread really finished on each score; every other game is left alone."
    ),
    "KL1b": (
        "Football scores bunch up on 3 and 7 more than anywhere else. When the spread is set "
        "exactly on one of those two numbers, this version reads the chance of covering from how "
        "often past games at a similar spread really finished on each score; every other game is "
        "left alone."
    ),
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
    "push_brier": (
        "This row checks how often it expected a game to land exactly on the spread against how "
        "often that really happened."
    ),
}


def key_line_mask(line: pd.Series, keys: tuple[float, ...]) -> np.ndarray:
    """True where the quoted line sits exactly ON one of the declared atoms."""

    size = pd.to_numeric(line, errors="raise").abs().to_numpy(dtype=float)
    return np.any(np.abs(size[:, None] - np.asarray(keys, dtype=float)) < 1e-9, axis=1)


def restrict(
    line: pd.Series,
    served_probability: np.ndarray,
    served_push: np.ndarray,
    mapped_probability: np.ndarray,
    mapped_push: np.ndarray,
    keys: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """The whole arm: the mapped read on key lines, the served read elsewhere."""

    touched = key_line_mask(line, keys)
    return {
        "touched": touched,
        "probability": np.where(touched, mapped_probability, served_probability),
        "push": np.where(touched, mapped_push, served_push),
    }


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    """Lane K's frozen S3 replay, redirected here; fails closed above 1e-9."""

    # Lane K's construction is imported, not reimplemented; its output root is
    # redirected so this lane can never rewrite a lane-K artifact.
    lane_k.OUT = OUT
    lane_k.replay(archive, active, archive_path)
    # Freeze the predeclaration as written, before any measured section is
    # appended to the living document, so the stamped digest stays checkable.
    # Bytes, not text: newline translation would break the digest it certifies.
    predeclaration = common.REPO / "docs/key_line_lattice.md"
    (OUT / "predeclaration.md").write_bytes(predeclaration.read_bytes())
    payload = json.loads((OUT / "reproduction.json").read_text())
    payload.pop("_provenance_stamp", None)
    payload["laneT_predeclaration_sha256"] = sha256_file(predeclaration)
    payload["laneK_construction_sha256"] = sha256_file(
        common.REPO / "scripts/mass_preserving_lattice_opener_eval.py"
    )
    payload["key_numbers"] = list(KEY_NUMBERS)
    payload["arms"] = {arm: list(keys) for arm, keys in ARMS.items()}
    write_stamped_artifact(payload, OUT / "reproduction.json")


def map_replay(archive: pd.DataFrame) -> None:
    """MP1 on every archive row, then the declared key-line restriction."""

    frame = pd.read_parquet(OUT / "replay.parquet")
    lane_k.verify_replay(frame.p_S3.to_numpy(), frame.home_cover_probability_at_open.to_numpy())
    pool = lane_k.build_pool(archive)
    common.table(pool, OUT / "pool.parquet")
    targets = frame[["game_id", "season", "week"]].copy()
    targets["gameday"] = pool.set_index("game_id").gameday.reindex(frame.game_id).to_numpy()
    targets["line"] = frame.tue_open_home_spread.to_numpy()
    targets["point"] = frame.center_S3.to_numpy()
    if targets.gameday.isna().any():
        raise ValueError("Archive rows without a gameday in the feature table")
    mapped = lane_k.mass_preserving(pool, targets, float(BANDWIDTH)).set_index("game_id")
    mapped = mapped.reindex(frame.game_id)
    for column in mapped:
        frame[f"{column}_MP1"] = mapped[column].to_numpy()
    frame["p_MP1"] = mapped.home_cover_probability.to_numpy()
    frame["push_MP1"] = mapped.push.to_numpy()
    reference = common.REPO / "artifacts/research/laneK/scored.parquet"
    cross_lane = None
    if reference.exists():
        lane_k_scored = pd.read_parquet(reference).set_index("game_id").reindex(frame.game_id)
        cross_lane = float(np.abs(lane_k_scored.p_MP1.to_numpy() - frame.p_MP1.to_numpy()).max())
    counts = {}
    for arm, keys in ARMS.items():
        applied = restrict(
            frame.tue_open_home_spread,
            frame.p_S3.to_numpy(),
            frame.push_S3.to_numpy(),
            frame.p_MP1.to_numpy(),
            frame.push_MP1.to_numpy(),
            keys,
        )
        frame[f"touched_{arm}"] = applied["touched"]
        frame[f"p_{arm}"] = applied["probability"]
        frame[f"push_{arm}"] = applied["push"]
        frame[f"offset_{arm}"] = frame.offset_S3
        frame[f"residual_{arm}"] = frame.residual_S3
        untouched = ~applied["touched"]
        if not np.array_equal(frame.loc[untouched, f"p_{arm}"], frame.loc[untouched, "p_S3"]):
            raise ValueError(f"{arm} moved a non-key-line probability")
        counts[arm] = {
            "keys": list(keys),
            "games_touched": int(applied["touched"].sum()),
            "games_total": len(frame),
            "picks_changed": int(
                (frame[f"p_{arm}"].ge(0.5) != frame.p_S3.ge(0.5)).to_numpy().sum()
            ),
            "per_key": {
                str(int(key)): int(np.abs(frame.tue_open_home_spread.abs() - key).lt(1e-9).sum())
                for key in keys
            },
        }
    common.table(frame, OUT / "replay.parquet")
    write_stamped_artifact(
        {
            "bandwidth": float(BANDWIDTH),
            "cross_lane_max_probability_gap_vs_laneK_MP1": cross_lane,
            "arms": counts,
        },
        OUT / "mapping.json",
    )
    print(json.dumps(counts, indent=2), flush=True)


def research_per_game(frame: pd.DataFrame, arm: str, archive_path: Path) -> Path:
    """A research opener evaluation for one arm; never the active identity."""

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
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id=f"research_laneT_{arm}",
        research_scope="Opener only; close columns retain S3, not candidate evidence.",
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": f"research_laneT_{arm}",
    }
    write_stamped_artifact(metadata, directory / "metadata.json")
    return directory


def arm_groups(frame: pd.DataFrame, arm: str) -> list[tuple[str, pd.DataFrame]]:
    """Overall, per season, per key line, and the games the arm actually moves."""

    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups += [(f"season_{int(season)}", g) for season, g in frame.groupby("season")]
    for key in KEY_NUMBERS:
        block = frame.loc[frame.tue_open_home_spread.abs().sub(key).abs().lt(1e-9)]
        if not block.empty:
            groups.append((f"key_line_{int(key)}", block))
    groups.append(("touched", frame.loc[frame[f"touched_{arm}"]]))
    return [(label, group) for label, group in groups if not group.empty]


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
    invariants = {}
    for arm in ARMS:
        untouched = frame.loc[~frame[f"touched_{arm}"]]
        invariants[arm] = {
            "untouched_games": len(untouched),
            "untouched_max_probability_gap": float(
                np.abs(untouched[f"p_{arm}"] - untouched.p_S3).max()
            ),
            "untouched_card_disagreements": int(
                (untouched[f"card_{arm}"] != untouched.card_S3).sum()
            ),
        }
        if invariants[arm]["untouched_max_probability_gap"] > 0.0:
            raise ValueError(f"{arm} changed a non-key-line probability")
    write_stamped_artifact(invariants, OUT / "invariants.json")
    cells: dict[str, dict] = {}
    skipped = []
    for arm in ARMS:
        for label, group in arm_groups(frame, arm):
            if not bool((group[f"p_{arm}"] != group.p_S3).any()):
                skipped.append(f"{arm}_{label}_all")
                continue
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                graded = (group.margin_vs_open.ne(0) & group.margin_vs_open.notna()).to_numpy()
                if not bool((cp.to_numpy() != bp.to_numpy())[graded].any()):
                    # No pick moved, so the paired difference is identically
                    # zero: a fact for the write-up, never a registry row (a
                    # bootstrap of zeros reports probability_positive 0.0).
                    skipped.append(f"{arm}_{label}_{kind}")
                    continue
                cells[f"{PREFIX}_{arm.lower()}_{label}_{kind}"] = common.comparison(
                    group, cp.to_numpy(), bp.to_numpy()
                )
            for kind, metrics in lane_k.loss_cells(group, f"p_{arm}").items():
                cells[f"{PREFIX}_{arm.lower()}_{label}_{kind}"] = metrics
    push_rows = []
    for size in KEY_NUMBERS:
        block = frame.loc[frame.tue_open_home_spread.abs().sub(size).abs().lt(1e-9)].reset_index(
            drop=True
        )
        if block.empty:
            continue
        realized = block.margin_vs_open.eq(0).to_numpy(dtype=float)
        for arm in ("S3", *ARMS):
            predicted = block[f"push_{arm}"].to_numpy(dtype=float)
            push_rows.append(
                {
                    "line_size": float(size),
                    "arm": arm,
                    "n": len(block),
                    "predicted": float(predicted.mean()),
                    "realized": float(realized.mean()),
                }
            )
            if arm == "S3" or not bool((block[f"push_{arm}"] != block.push_S3).any()):
                continue
            base_loss = (block.push_S3.to_numpy(dtype=float) - realized) ** 2
            candidate_loss = (predicted - realized) ** 2
            cells[f"{PREFIX}_{arm.lower()}_push_at_{int(size)}_push_brier"] = {
                **lane_s.metric(block, base_loss - candidate_loss),
                "candidate": float(candidate_loss.mean()),
                "baseline": float(base_loss.mean()),
            }
    write_stamped_artifact({"rows": push_rows, "skipped_degenerate": skipped}, OUT / "push.json")
    bucket_rows = []
    for arm in ("S3", *ARMS):
        for bucket, group in frame.groupby("bucket", observed=True):
            valid = group.loc[group.margin_vs_open.ne(0)]
            pick = valid[f"p_{arm}"].ge(0.5)
            bucket_rows.append(
                {
                    "arm": arm,
                    "bucket": str(bucket),
                    "n": len(valid),
                    "standalone_accuracy": float(pick.eq(valid.margin_vs_open.gt(0)).mean()),
                    "card_accuracy": float(
                        (
                            valid[f"card_{arm}"].to_numpy() == valid.margin_vs_open.gt(0).to_numpy()
                        ).mean()
                    ),
                }
            )
    common.table(pd.DataFrame(bucket_rows), OUT / "buckets.parquet")
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    print(json.dumps({k: v for k, v in cells.items() if "overall" in k}, indent=2), flush=True)


def week1(active: dict) -> None:
    """Week 1 2026 sides that would change; no forecast is regenerated."""

    from nfl_ats.four_overlay_composition import apply_four_overlay_composition_for_publication
    from nfl_ats.margin import fit_margin_model
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
    mapped = lane_k.mass_preserving(pool, targets, float(BANDWIDTH)).set_index("game_id")
    mapped = mapped.reindex(rows.game_id)
    rows["p_MP1"] = mapped.home_cover_probability.to_numpy()
    rows["push_MP1"] = mapped.push.to_numpy()
    changes, touched_counts = {}, {}
    for arm, keys in ARMS.items():
        applied = restrict(
            rows.spread_line,
            rows.p_S3.to_numpy(),
            rows.push_S3.to_numpy(),
            rows.p_MP1.to_numpy(),
            rows.push_MP1.to_numpy(),
            keys,
        )
        rows[f"touched_{arm}"] = applied["touched"]
        rows[f"p_{arm}"] = applied["probability"]
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        touched_counts[arm] = int(applied["touched"].sum())
        changes[arm] = rows.loc[changed].to_dict(orient="records")
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
            "games_touched": touched_counts,
            "changes": changes,
            "card_changes": card_changes,
        },
        OUT / "week1.json",
    )
    print(json.dumps({"touched": touched_counts}, indent=2), flush=True)


def slice_summary(label: str) -> str:
    if label == "overall":
        return "Measured on every game from 2020 through 2025."
    if label.startswith("season_"):
        return f"Measured on the {label.split('_')[-1]} season only."
    if label.startswith("push_at_"):
        return f"Measured only on games whose spread was exactly {label.split('_')[-1]} points."
    if label.startswith("key_line_"):
        return f"Measured only on games whose spread was exactly {label.split('_')[-1]} points."
    return "Measured only on the games this version actually changes."


def plain_summary(arm: str, label: str, kind: str) -> str:
    return f"{BASE_SUMMARIES[arm]} {slice_summary(label)} {KIND_SUMMARIES[kind]}"


def split_cell(name: str) -> tuple[str, str, str]:
    """``kl1_kl1b_season_2023_card`` -> arm KL1b, slice season_2023, kind card."""

    body = name[len(PREFIX) + 1 :]
    arm = "KL1b" if body.startswith("kl1b_") else "KL1"
    body = body[len(arm) + 1 :]
    for kind in ("push_brier", "log_loss", "standalone", "card", "brier"):
        if body.endswith(f"_{kind}"):
            return arm, body[: -len(kind) - 1], kind
    raise ValueError(f"Unrecognised cell name: {name}")


def record_argv(name: str, metrics: dict, units: str, start: int, end: int) -> list[str]:
    """The exact recorder argv for one comparison cell."""

    arm, label, kind = split_cell(name)
    return [
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        f"Key-line restricted mass-preserving read {name}, positive favours the candidate",
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
        plain_summary(arm, label, kind),
        "--replace",
    ]


def units_for(name: str) -> str:
    if name.endswith("brier"):
        return "brier_improvement"
    if name.endswith("log_loss"):
        return "log_loss_improvement"
    return "accuracy_points"


def record() -> None:
    """Emit the recorder commands; the coordinator runs them serially.

    Coordinator instruction, 2026-09-08: ``registry/weak_signals.json`` was
    corrupted by two lanes recording at once, so research lanes emit their argv
    instead of running it. Nothing here touches the live registry.
    """

    cells = json.loads((OUT / "cells.json").read_text())
    lines = [
        "# MOD-18 lane T (docs/key_line_lattice.md) weak-signal records.",
        "# Run SERIALLY, from the repository root, with no other lane recording.",
        "# Every row is unresolved_below_power with no closing ground and a",
        "# plain-English summary; --replace makes a re-run idempotent.",
        "$ErrorActionPreference = 'Stop'",
    ]
    names = []
    for name, metrics in cells.items():
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        season = int(name.split("season_")[1].split("_")[0]) if "season_" in name else None
        argv = record_argv(name, metrics, units_for(name), season or 2020, season or 2025)
        quoted = " ".join(lane_k.powershell_quote(token) for token in argv)
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

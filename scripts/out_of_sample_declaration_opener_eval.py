"""Frozen MOD-18 lane V out-of-sample declaration; docs/out_of_sample_declaration.md.

Lane T's KL1b chose its atom set after seeing the archive it was graded on, and
lane U's R2b chose its bucket after lane R's per-bucket table. Both live
candidates therefore carry a selection discount neither lane could quantify.

This lane takes the discount off the only way it can be taken off without new
football: each rule's STRUCTURE is declared mechanically on an earlier block of
seasons, and the arm it produces is graded only on seasons the declaration
never saw.

* Rule A (atoms): every key atom in {3, 7, 10, 14} whose standalone paired
  delta on the declaration seasons is strictly positive.
* Rule B (buckets): every spread bucket whose walk-forward slope interval on
  the declaration seasons lies wholly below +0.5.
* Combination: the key-line read takes precedence on the games it names.

Nothing here is re-implemented. The S3 replay gate comes from lane K, the
mass-preserving read from lane K, the key-line restriction from lane T, the
slope fit and its week-blocked interval from lane R, the empirical-Bayes
serialisation helper from lane U, the composition and the accuracy bootstrap
from ``spread_regime_opener_eval``, and the loss bootstrap from lane S.

Every write lands under ``artifacts/research/laneV``; the live registry is never
touched -- the ``record`` stage WRITES the record commands to a PowerShell file
for the coordinator to run serially.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import home_side_location_opener_eval as lane_s
import key_line_lattice_opener_eval as lane_t
import mass_preserving_lattice_opener_eval as lane_k
import numpy as np
import pandas as pd
import residual_slope_opener_eval as lane_r
import residual_slope_shrunk_opener_eval as lane_u
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import archive_prior_stream
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.spread_regime import BUCKETS

OUT = common.REPO / "artifacts/research/laneV"
PREDECLARATION = common.REPO / "docs/out_of_sample_declaration.md"
LANE_T_SCORED = common.REPO / "artifacts/research/laneT/scored.parquet"
LANE_R_REPLAY = common.REPO / "artifacts/research/laneR/replay.parquet"
LANE_U_SCORED = common.REPO / "artifacts/research/laneU/scored.parquet"
LANE_REPRODUCTIONS = ("laneT", "laneR", "laneU")

KEY_ATOMS: tuple[float, ...] = lane_t.KEY_NUMBERS
SLOPE_CEILING = 0.5
POSTHOC_ATOMS: tuple[float, ...] = (3.0, 7.0)
POSTHOC_BUCKETS: tuple[str, ...] = ("10.5+",)

WINDOWS: dict[str, dict[str, tuple[int, int]]] = {
    "W1": {"declare": (2020, 2023), "holdout": (2024, 2025)},
    "W2": {"declare": (2020, 2022), "holdout": (2023, 2025)},
}
ARMS = ("KL", "RS", "BOTH")
ARM_FAMILY = {
    "KL": "mod18_conditional_margin_v1",
    "RS": "mod18_home_side_location_v1",
    "BOTH": "mod18_conditional_margin_v1",
}
ACTIVE_MODEL_ID = "research_laneV_oos1"
PREFIX = "oos1"
KINDS = ("log_loss", "standalone", "card", "brier")

DISCOUNT = (
    "Out-of-sample declaration of two post-hoc MOD-18 candidates: the atom set and the bucket set "
    "are chosen by a mechanical rule fixed in advance on the declaration seasons and graded only "
    "on later seasons the choice never saw. The declaration removes the discount on THAT CHOICE "
    "ONLY: the archive was mined by lanes H, K, L, R, S, T and U, the S3 baseline is itself a "
    "restriction fitted on it, the two cuts overlap in their declaration seasons and in 2023, and "
    "the three arms within a cut are nested and correlated. The combination arm is correlated "
    "with both mod18_conditional_margin_v1 and mod18_home_side_location_v1."
)

BUCKET_WORDS = {
    "0-3": "the opening line is 3 points or less",
    "3.5-6.5": "the opening line is between 3.5 and 6.5 points",
    "7": "the opening line is exactly 7 points",
    "7.5-10": "the opening line is between 7.5 and 10 points",
    "10.5+": "the opening line is 10.5 points or more",
}
KIND_WORDS = {
    "standalone": (
        "This row counts how many more games in a hundred the model on its own got right than the "
        "version in use."
    ),
    "card": (
        "This row counts the same thing after the usual weekly adjustments are applied to both."
    ),
    "brier": (
        "This row scores how close the stated chances were to what happened, not how many picks "
        "were right, and a positive number means the new way was closer."
    ),
    "log_loss": (
        "This row scores how well the stated chances held up, not how many picks were right, and "
        "a positive number means the new way scored better."
    ),
}
KIND_UNITS = {
    "standalone": "accuracy_points",
    "card": "accuracy_points",
    "brier": "brier_improvement",
    "log_loss": "log_loss_improvement",
}


def apply_declaration(
    frame: pd.DataFrame, atoms: tuple[float, ...], buckets: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Serve the declared reads; every other game keeps S3 bit-for-bit.

    Precedence is the predeclared one: a game whose line sits exactly on a
    declared atom takes lane K's mass-preserving read, a game whose bucket is
    declared takes lane R's re-scaled read, and everything else is untouched.

    Both mechanisms are local by construction -- the atom read looks only at
    the game's own line, and the re-scale multiplies the game's own residual by
    its own bucket's weight -- so the served probability for a game never
    depends on which OTHER atoms or buckets are in the declared sets. That is
    why the frozen lane columns can be recombined instead of refitted, and the
    two reconstruction gates prove it rather than assume it.
    """

    on_atom = lane_t.key_line_mask(frame.tue_open_home_spread, atoms)
    in_bucket = frame.bucket.isin(buckets).to_numpy() & ~on_atom
    return {
        "on_atom": on_atom,
        "in_bucket": in_bucket,
        "touched": on_atom | in_bucket,
        "probability": np.where(
            on_atom, frame.p_MP1.to_numpy(), np.where(in_bucket, frame.p_R1, frame.p_S3)
        ),
        "push": np.where(on_atom, frame.push_MP1.to_numpy(), frame.push_S3.to_numpy()),
        "offset": np.where(in_bucket, frame.shift_R1.to_numpy(), frame.offset_S3.to_numpy()),
        "residual": np.where(in_bucket, frame.residual_R1.to_numpy(), frame.residual_S3.to_numpy()),
    }


def arm_sets(
    arm: str, atoms: tuple[float, ...], buckets: tuple[str, ...]
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Which declared sets each arm serves."""

    return (atoms if arm in {"KL", "BOTH"} else ()), (buckets if arm in {"RS", "BOTH"} else ())


def declare_atoms(frame: pd.DataFrame, seasons: tuple[int, int]) -> list[dict[str, object]]:
    """Rule A: every atom whose standalone paired delta is strictly positive."""

    block = frame.loc[frame.season.between(*seasons)].reset_index(drop=True)
    if block.empty:
        raise ValueError(f"Declaration block {seasons} holds no games")
    if int(block.season.max()) > seasons[1] or int(block.season.min()) < seasons[0]:
        raise ValueError("Declaration block reached outside its declared seasons")
    rows: list[dict[str, object]] = []
    for atom in KEY_ATOMS:
        applied = apply_declaration(block, (atom,), ())
        metrics = common.comparison(
            block, applied["probability"] >= 0.5, block.p_S3.ge(0.5).to_numpy()
        )
        rows.append(
            {
                "atom": float(atom),
                "games_on_atom": int(applied["on_atom"].sum()),
                "delta": metrics["delta"],
                "lower": metrics["lower"],
                "upper": metrics["upper"],
                "probability_positive": metrics["probability_positive"],
                "flips": metrics["flips"],
                "n": metrics["n"],
                "weeks": metrics["weeks"],
                "selected": bool(metrics["delta"] > 0.0),
            }
        )
    return rows


def declare_buckets(stream: pd.DataFrame, seasons: tuple[int, int]) -> list[dict[str, object]]:
    """Rule B: every bucket whose slope interval lies wholly below +0.5."""

    rows_in_window = stream.loc[stream.season.between(*seasons) & stream.result.notna()]
    if rows_in_window.empty:
        raise ValueError(f"Declaration stream {seasons} holds no completed games")
    if int(rows_in_window.season.max()) > seasons[1]:
        raise ValueError("Declaration stream reached outside its declared seasons")
    intervals = lane_r.slope_intervals(rows_in_window)
    rows: list[dict[str, object]] = []
    for bucket in BUCKETS:
        interval = intervals[bucket]
        upper = float(interval["upper"])
        rows.append(
            {
                "bucket": bucket,
                "n": interval["n"],
                "weeks": interval["weeks"],
                "slope": interval["estimate"],
                "lower": interval["lower"],
                "upper": upper,
                "shrunk": interval["shrunk"],
                "selected": bool(math.isfinite(upper) and upper < SLOPE_CEILING),
            }
        )
    return rows


def selected_atoms(rows: list[dict[str, object]]) -> tuple[float, ...]:
    return tuple(float(row["atom"]) for row in rows if row["selected"])  # type: ignore[arg-type]


def selected_buckets(rows: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(row["bucket"]) for row in rows if row["selected"])


def load_frames() -> tuple[pd.DataFrame, Path, dict]:
    """One frame carrying S3, lane K's read and lane R's re-scaled read."""

    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
    if match is None:
        raise ValueError("No matching opener evaluation")
    archive_path = match[1]
    for lane in LANE_REPRODUCTIONS:
        recorded = json.loads(
            (common.REPO / "artifacts/research" / lane / "reproduction.json").read_text()
        )
        if Path(recorded["archive"]) != archive_path:
            raise ValueError(f"{lane} was scored on a different archive: {recorded['archive']}")
    archive = pd.read_parquet(archive_path / "per_game.parquet").set_index("game_id")
    frame = pd.read_parquet(LANE_T_SCORED)
    served = archive.home_cover_probability_at_open.reindex(frame.game_id).to_numpy()
    lane_k.verify_replay(frame.p_S3.to_numpy(), served)
    slope = pd.read_parquet(LANE_R_REPLAY).set_index("game_id").reindex(frame.game_id)
    lane_k.verify_replay(slope.p_S3.to_numpy(), served)
    if not np.array_equal(frame.p_S3.to_numpy(), slope.p_S3.to_numpy()):
        raise ValueError("Lane T and lane R disagree on the served S3 read")
    if not np.array_equal(
        frame.pick_home_at_open_probability_rule.to_numpy(dtype=bool),
        frame.p_S3.ge(0.5).to_numpy(),
    ):
        raise ValueError("The archive's served pick is not the served S3 probability rule")
    for column in ("p_R1", "shift_R1", "residual_R1", "beta_R1"):
        frame[column] = slope[column].to_numpy()
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    reconstruction_gates(frame)
    return frame, archive_path, active


def reconstruction_gates(frame: pd.DataFrame) -> dict[str, float]:
    """The recombination must reproduce lane T's KL1b and lane U's R2b exactly."""

    key_line = apply_declaration(frame, POSTHOC_ATOMS, ())["probability"]
    key_gap = float(np.abs(key_line - frame.p_KL1b.to_numpy()).max())
    shrunk = pd.read_parquet(LANE_U_SCORED).set_index("game_id").reindex(frame.game_id)
    rescale = apply_declaration(frame, (), POSTHOC_BUCKETS)["probability"]
    slope_gap = float(np.abs(rescale - shrunk.p_R2b.to_numpy()).max())
    if key_gap != 0.0 or slope_gap != 0.0:
        raise ValueError(f"Reconstruction gate failed: atoms {key_gap}, buckets {slope_gap}")
    return {"key_line_gap_vs_laneT_KL1b": key_gap, "rescale_gap_vs_laneU_R2b": slope_gap}


def declare() -> None:
    """Run both declaration rules on each declaration window and freeze them."""

    frame, archive_path, active = load_frames()
    stream = archive_prior_stream(pd.read_parquet(archive_path / "per_game.parquet"))
    payload: dict[str, object] = {
        "archive": str(archive_path),
        "active_model_id": active["model_id"],
        "feature_sha256": sha256_file(common.FEATURES),
        "predeclaration_sha256": sha256_file(PREDECLARATION),
        "slope_ceiling": SLOPE_CEILING,
        "key_atoms_searched": list(KEY_ATOMS),
        "buckets_searched": list(BUCKETS),
        "posthoc_atoms": list(POSTHOC_ATOMS),
        "posthoc_buckets": list(POSTHOC_BUCKETS),
        "reconstruction": reconstruction_gates(frame),
        "windows": {},
    }
    windows: dict[str, object] = {}
    for window, spec in WINDOWS.items():
        atom_rows = declare_atoms(frame, spec["declare"])
        bucket_rows = declare_buckets(stream, spec["declare"])
        atoms = selected_atoms(atom_rows)
        buckets = selected_buckets(bucket_rows)
        held = frame.loc[frame.season.between(*spec["holdout"])]
        windows[window] = {
            "declare": list(spec["declare"]),
            "holdout": list(spec["holdout"]),
            "declaration_games": int(frame.season.between(*spec["declare"]).sum()),
            "holdout_games": len(held),
            "holdout_graded_games": int(held.margin_vs_open.ne(0).sum()),
            "atom_table": atom_rows,
            "bucket_table": bucket_rows,
            "atoms": list(atoms),
            "buckets": list(buckets),
            "atoms_match_posthoc": tuple(atoms) == POSTHOC_ATOMS,
            "buckets_match_posthoc": tuple(buckets) == POSTHOC_BUCKETS,
        }
    payload["windows"] = windows
    (OUT / "predeclaration.md").write_bytes(PREDECLARATION.read_bytes())
    write_stamped_artifact(lane_u.finite_json(payload), OUT / "declarations.json")
    print(
        json.dumps(
            {w: {"atoms": v["atoms"], "buckets": v["buckets"]} for w, v in windows.items()},  # type: ignore[index]
            indent=2,
        ),
        flush=True,
    )


def candidate_frame(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Lane T's ``research_per_game`` candidate columns, formed in memory."""

    candidate = frame.copy()
    candidate["home_cover_probability_at_open"] = candidate[f"p_{tag}"]
    candidate["home_side_offset_at_open"] = candidate[f"offset_{tag}"]
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
    return candidate


def write_research_artifact(candidate: pd.DataFrame, tag: str, archive_path: Path) -> Path:
    """A research opener evaluation for one arm; never the active identity."""

    directory = OUT / "opener_evaluation" / tag
    common.table(candidate, directory / "per_game.parquet")
    metadata = json.loads((archive_path / "metadata.json").read_text())
    metadata.update(
        research_arm=tag,
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id=ACTIVE_MODEL_ID,
        research_scope=(
            "Held-out seasons only; opener columns only; close columns retain the incumbent read."
        ),
    )
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "model_id": ACTIVE_MODEL_ID,
    }
    write_stamped_artifact(metadata, directory / "metadata.json")
    return directory


def held_out_groups(held: pd.DataFrame, tag: str) -> list[tuple[str, pd.DataFrame]]:
    """Overall, each held-out season, and the games the arm actually moves."""

    groups: list[tuple[str, pd.DataFrame]] = [("overall", held)]
    groups += [(f"season_{int(season)}", g) for season, g in held.groupby("season")]
    groups.append(("touched", held.loc[held[f"touched_{tag}"]]))
    return [(label, group) for label, group in groups if not group.empty]


def score(archive_path: Path | None = None) -> None:
    """Grade every declared arm on its held-out seasons only."""

    declarations = json.loads((OUT / "declarations.json").read_text())
    frame, resolved_archive, _ = load_frames()
    archive_path = archive_path or resolved_archive
    if str(resolved_archive) != declarations["archive"]:
        raise ValueError("The declaration was frozen against a different archive")
    frame["card_S3"] = common.composed_picks(frame, archive_path / "per_game.parquet")[1]
    cells: dict[str, dict] = {}
    invariants: dict[str, dict] = {}
    skipped: list[str] = []
    for window, spec in declarations["windows"].items():
        atoms = tuple(float(atom) for atom in spec["atoms"])
        buckets = tuple(str(bucket) for bucket in spec["buckets"])
        for arm in ARMS:
            tag = f"{window}_{arm}"
            applied = apply_declaration(frame, *arm_sets(arm, atoms, buckets))
            frame[f"p_{tag}"] = applied["probability"]
            frame[f"offset_{tag}"] = applied["offset"]
            frame[f"touched_{tag}"] = applied["touched"]
            untouched = ~applied["touched"]
            gap = float(
                np.abs(frame.loc[untouched, f"p_{tag}"] - frame.loc[untouched, "p_S3"]).max()
                if untouched.any()
                else 0.0
            )
            if gap != 0.0:
                raise ValueError(f"{tag} moved an untouched game: {gap}")
            frame[f"card_{tag}"] = common.composed_picks(
                candidate_frame(frame, tag), archive_path / "per_game.parquet"
            )[1]
        held = frame.loc[frame.season.between(*spec["holdout"])].reset_index(drop=True)
        base_card = common.composed_picks(held, archive_path / "per_game.parquet")[1]
        if not np.array_equal(base_card, held.card_S3.to_numpy()):
            raise ValueError(f"{window}: restricting the archive changed the incumbent card")
        for arm in ARMS:
            tag = f"{window}_{arm}"
            candidate = candidate_frame(held, tag)
            directory = write_research_artifact(candidate, tag, archive_path)
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
            restricted = common.composed_picks(candidate, directory / "per_game.parquet")[1]
            invariants[tag] = {
                "holdout_games": len(held),
                "games_touched": int(held[f"touched_{tag}"].sum()),
                "picks_changed": int(
                    (held[f"p_{tag}"].ge(0.5) != held.p_S3.ge(0.5)).to_numpy().sum()
                ),
                "card_picks_changed": int((restricted != held.card_S3.to_numpy()).sum()),
                "composition_restriction_disagreements": int(
                    (restricted != held[f"card_{tag}"].to_numpy()).sum()
                ),
                "untouched_max_probability_gap": 0.0,
            }
            if invariants[tag]["composition_restriction_disagreements"]:
                raise ValueError(f"{tag}: restricting the artifact changed the composed card")
            held[f"card_{tag}"] = restricted
            for label, group in held_out_groups(held, tag):
                for kind in ("standalone", "card"):
                    cp = group[f"p_{tag}"].ge(0.5) if kind == "standalone" else group[f"card_{tag}"]
                    bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                    graded = (group.margin_vs_open.ne(0) & group.margin_vs_open.notna()).to_numpy()
                    if not bool((cp.to_numpy() != bp.to_numpy())[graded].any()):
                        skipped.append(f"{tag}_{label}_{kind}")
                        continue
                    cells[cell_name(window, arm, label, kind)] = common.comparison(
                        group, cp.to_numpy(), bp.to_numpy()
                    )
            for kind, metrics in lane_k.loss_cells(held, f"p_{tag}").items():
                cells[cell_name(window, arm, "overall", kind)] = metrics
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(
        {"invariants": invariants, "skipped_degenerate": skipped}, OUT / "invariants.json"
    )
    write_stamped_artifact(cells, OUT / "cells.json")
    print(
        json.dumps({k: v for k, v in cells.items() if "_overall_" in k}, indent=2),
        flush=True,
    )


def cell_name(window: str, arm: str, label: str, kind: str) -> str:
    return f"{PREFIX}_{window.lower()}_{arm.lower()}_{label}_{kind}"


def split_cell(name: str) -> tuple[str, str, str, str]:
    """``oos1_w1_kl_season_2024_card`` -> window W1, arm KL, season_2024, card."""

    body = name[len(PREFIX) + 1 :]
    window, _, body = body.partition("_")
    if window.upper() not in WINDOWS:
        raise ValueError(f"Unrecognised cell name: {name}")
    arm, _, body = body.partition("_")
    if arm.upper() not in ARMS:
        raise ValueError(f"Unrecognised cell name: {name}")
    for kind in KINDS:
        if body.endswith(f"_{kind}"):
            return window.upper(), arm.upper(), body[: -len(kind) - 1], kind
    raise ValueError(f"Unrecognised cell name: {name}")


def seasons_for(window: str, label: str) -> tuple[int, int]:
    if label.startswith("season_"):
        season = int(label.removeprefix("season_"))
        return season, season
    start, end = WINDOWS[window]["holdout"]
    return int(start), int(end)


def atom_words(atoms: tuple[float, ...]) -> str:
    named = [f"{atom:g}" for atom in atoms]
    if len(named) == 1:
        return f"exactly {named[0]} points"
    return "exactly " + ", ".join(named[:-1]) + f" or {named[-1]} points"


def arm_words(arm: str, window: str, atoms: tuple[float, ...], buckets: tuple[str, ...]) -> str:
    declare_start, declare_end = WINDOWS[window]["declare"]
    chosen = (
        "The choice was made by a fixed rule written down in advance, using only the "
        f"{declare_start} to {declare_end} seasons."
    )
    parts = []
    if atoms:
        parts.append(
            "Football scores bunch up on a few numbers, so when the spread is set on "
            f"{atom_words(atoms)}, this version reads the chance of covering from how often past "
            "games at a similar spread really finished on each score, and leaves every other game "
            "alone"
        )
    if buckets:
        parts.append(
            "Where "
            + " or ".join(BUCKET_WORDS[bucket] for bucket in buckets)
            + ", the model's own disagreement with the betting number has been worth less than "
            "face value, so this version keeps only the share of it earlier seasons say is worth "
            "keeping, and leaves every other size of line exactly as it is played today"
        )
    if not parts:
        return (
            "The rule picked nothing at all here, so this version makes exactly the same picks as "
            f"the one in use. {chosen}"
        )
    lead = "Both changes at once. " if len(parts) > 1 else ""
    return f"{lead}{'. '.join(parts)}. {chosen}"


def scope_words(window: str, label: str) -> str:
    start, end = WINDOWS[window]["holdout"]
    if label == "overall":
        return f"Measured on the {start} to {end} seasons, which the choice never saw."
    if label.startswith("season_"):
        return (
            f"Measured on the {label.removeprefix('season_')} season only, which the choice never "
            "saw."
        )
    return (
        "Measured only on the games this version actually changes, in seasons the choice never saw."
    )


def plain_summary(
    window: str, arm: str, label: str, kind: str, atoms: tuple[float, ...], buckets: tuple[str, ...]
) -> str:
    return " ".join(
        [
            arm_words(arm, window, atoms, buckets),
            scope_words(window, label),
            KIND_WORDS[kind],
        ]
    )


def record_argv(
    name: str, metrics: dict, atoms: tuple[float, ...], buckets: tuple[str, ...]
) -> list[str]:
    """The exact recorder argv for one held-out cell."""

    window, arm, label, kind = split_cell(name)
    start, end = seasons_for(window, label)
    served_atoms, served_buckets = arm_sets(arm, atoms, buckets)
    return [
        "weak-signals",
        "record",
        "--name",
        f"{ARM_FAMILY[arm]}_{name}_{start}_{end}",
        "--family",
        ARM_FAMILY[arm],
        "--description",
        (
            f"Out-of-sample declared arm {arm} on window {window} versus the served S3, "
            f"{label} {kind}; positive favours the candidate"
        ),
        "--source",
        str(OUT / "cells.json"),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        "No resolved wrong sign, split-half reliability or positive-control closing ground "
        "established; retained unresolved.",
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
        KIND_UNITS[kind],
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
        plain_summary(window, arm, label, kind, served_atoms, served_buckets),
        "--replace",
    ]


def record() -> None:
    """Emit the recorder commands; the coordinator runs them serially."""

    declarations = json.loads((OUT / "declarations.json").read_text())
    cells = json.loads((OUT / "cells.json").read_text())
    lines = [
        "# MOD-18 lane V (docs/out_of_sample_declaration.md) weak-signal records.",
        "# Run SERIALLY, from the repository root, with no other lane recording.",
        "# Every row is unresolved_below_power with no closing ground and a",
        "# plain-English summary; --replace makes a re-run idempotent.",
        "$ErrorActionPreference = 'Stop'",
    ]
    names = []
    for name, metrics in cells.items():
        if not isinstance(metrics, dict) or "delta" not in metrics:
            continue
        window, _, _, _ = split_cell(name)
        spec = declarations["windows"][window]
        argv = record_argv(
            name,
            metrics,
            tuple(float(atom) for atom in spec["atoms"]),
            tuple(str(bucket) for bucket in spec["buckets"]),
        )
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
    parser.add_argument("--stage", choices=("declare", "score", "record"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(OUT / "registry")
    with threadpool_limits(limits=1):
        if args.stage == "declare":
            declare()
        elif args.stage == "score":
            score()
        else:
            record()


if __name__ == "__main__":
    main()

"""Re-measure paired accuracy cells whose ``probability_positive`` was summarised
under the strict ``draws > 0`` convention, from the per-game arms still on disk.

Background (``docs/weak_signal_pooling.md``, defect D2): until 2026-09-08 every
screen computed ``probability_positive`` as ``mean(draws > 0)``. A paired
accuracy delta has a large atom of probability at exactly zero -- the resamples
in which the two arms happen to make the same picks -- and the strict ``>``
charged that whole atom against the candidate. The shared helper
``nfl_ats.evidence_conventions.probability_positive_from_draws`` now splits it
(``P(>0) + 0.5*P(==0)``), but rows recorded before the fix still carry the old
summary, and the bootstrap draws behind them were never stored.

For the MOD-18 research lanes the draws do not need to be stored: the lanes kept
their scored per-game frame, so the identical bootstrap can simply be run again.
This script rebuilds each lane's cell grid from its ``scored.parquet`` using the
lane's own grouping code and the shared ``comparison()`` helper -- same
construction, same 20,000-draw week-blocked bootstrap, same seed -- and then
requires that **every field except ``probability_positive`` reproduces the
recorded cell exactly**. That equality is what makes this a re-measurement of
the same measurement rather than an edit: the effect, the interval, the standard
error, the sample size, the block count and the flip count all have to come back
bit-for-bit before the corrected ``probability_positive`` is allowed to stand.

Nothing here closes, reclassifies, or reopens any line of work. A corrected
summary is not a verdict: every row keeps its ``unresolved_below_power``
classification, its effect and its interval. Re-measurement moves the number
that says how sure we are, not the number that says what happened.

Usage::

    # report only; touches nothing
    python scripts/weak_signal_zero_atom_remeasure.py --report out.json

    # re-record the verified rows through the weak-signals CLI
    python scripts/weak_signal_zero_atom_remeasure.py --report out.json --apply
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import spread_regime_opener_eval as common  # noqa: E402

from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

MEASUREMENT_FIELDS = (
    "delta",
    "lower",
    "upper",
    "standard_error",
    "n",
    "weeks",
    "flips",
    "candidate_accuracy",
    "baseline_accuracy",
)

TOLERANCE = 1e-9

Cells = dict[str, dict[str, Any]]


def _accuracy_cells(
    frame: pd.DataFrame,
    prefix: str,
    arms: Iterator[str] | tuple[str, ...] | dict[str, Any],
    groups_for: Callable[[pd.DataFrame, str], list[tuple[str, pd.DataFrame]]],
) -> Cells:
    """The ``standalone``/``card`` accuracy grid every MOD-18 lane builds.

    Each lane differs only in its prefix, its arm names and how it cuts the
    frame into cells; the paired comparison itself is one shared helper. Cells a
    lane skipped because no graded pick moved are skipped here too, by the same
    test the lanes use, so this grid cannot invent a row the lane never recorded.
    """

    cells: Cells = {}
    for arm in arms:
        for label, group in groups_for(frame, arm):
            for kind in ("standalone", "card"):
                candidate = (
                    group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                )
                baseline = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                graded = (group.margin_vs_open.ne(0) & group.margin_vs_open.notna()).to_numpy()
                if not bool((candidate.to_numpy() != baseline.to_numpy())[graded].any()):
                    continue
                cells[f"{prefix}_{arm.lower()}_{label}_{kind}"] = common.comparison(
                    group, candidate.to_numpy(), baseline.to_numpy()
                )
    return cells


def _lane_g() -> Cells:
    import home_side_location_opener_eval as lane_s
    import home_side_side_aware_opener_eval as lane_g

    frame = pd.read_parquet(REPO / "artifacts/research/laneG/scored.parquet")

    def groups_for(scored: pd.DataFrame, _arm: str) -> list[tuple[str, pd.DataFrame]]:
        groups = [("overall", scored)]
        groups += [(f"season_{season}", g) for season, g in scored.groupby("season")]
        groups += [
            (lane_s.cell_name("cell", "bucket", str(bucket), str(side)), g)
            for (bucket, side), g in scored.groupby(["bucket", "home_side"], observed=True)
        ]
        return groups

    return _accuracy_cells(frame, "s4", lane_g.ARMS, groups_for)


def _lane_i() -> Cells:
    import home_side_location_opener_eval as lane_s
    import home_side_prior_opener_eval as lane_i

    frame = pd.read_parquet(REPO / "artifacts/research/laneI/scored.parquet")

    def groups_for(scored: pd.DataFrame, _arm: str) -> list[tuple[str, pd.DataFrame]]:
        groups = [("overall", scored)]
        groups += [(f"season_{season}", g) for season, g in scored.groupby("season")]
        groups += [
            (lane_s.cell_name("cell", "bucket", str(bucket), "all"), g)
            for bucket, g in scored.groupby("bucket", observed=True)
        ]
        return groups

    return _accuracy_cells(frame, "s5", lane_i.ARMS, groups_for)


def _lane_r() -> Cells:
    import residual_slope_opener_eval as lane_r

    frame = pd.read_parquet(REPO / "artifacts/research/laneR/scored.parquet")
    return _accuracy_cells(frame, "r1", lane_r.ARMS, lambda f, _a: lane_r.cell_groups(f))


def _lane_u() -> Cells:
    import residual_slope_opener_eval as lane_r
    import residual_slope_shrunk_opener_eval as lane_u

    frame = pd.read_parquet(REPO / "artifacts/research/laneU/scored.parquet")
    return _accuracy_cells(frame, "r2", lane_u.ARMS, lambda f, _a: lane_r.cell_groups(f))


def _lane_t() -> Cells:
    import key_line_lattice_opener_eval as lane_t

    frame = pd.read_parquet(REPO / "artifacts/research/laneT/scored.parquet")
    return _accuracy_cells(frame, lane_t.PREFIX, lane_t.ARMS, lane_t.arm_groups)


def _lane_v() -> Cells:
    """Lane V grades each declared arm on its own held-out seasons only."""

    import out_of_sample_declaration_opener_eval as lane_v

    frame = pd.read_parquet(REPO / "artifacts/research/laneV/scored.parquet")
    declarations = json.loads((REPO / "artifacts/research/laneV/declarations.json").read_text())
    cells: Cells = {}
    for window, spec in declarations["windows"].items():
        held = frame.loc[frame.season.between(*spec["holdout"])].reset_index(drop=True)
        for arm in lane_v.ARMS:
            tag = f"{window}_{arm}"
            grid = _accuracy_cells(
                held,
                lane_v.PREFIX,
                (tag,),
                lambda scored, current=tag: lane_v.held_out_groups(scored, current),
            )
            for key, metrics in grid.items():
                label_kind = key[len(f"{lane_v.PREFIX}_{tag.lower()}_") :]
                label, _, kind = label_kind.rpartition("_")
                cells[lane_v.cell_name(window, arm, label, kind)] = metrics
    return cells


SCRATCHPAD = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\dcbb74c0-77a9-470f-8e6e-713bc3331924\scratchpad"
)


def _season_and_bucket_cells(lane: Path, arms: tuple[str, ...]) -> Cells:
    """The per-season and per-bucket accuracy grid lanes S and Q both build.

    Keys carry the season window (``@start_end``) because both lanes file every
    season under the same cell name (``S1_season``) and separate them only by
    the season range in the registry name.
    """

    import home_side_location_opener_eval as lane_s

    frame = pd.read_parquet(lane / "paired.parquet")
    cells: Cells = {}
    for season, group in frame.loc[frame.margin_vs_open.ne(0)].groupby("season"):
        for arm in arms:
            cells[f"{arm}_season@{int(season)}_{int(season)}"] = common.comparison(
                group.reset_index(drop=True),
                group[f"p_{arm}"].ge(0.5).to_numpy(),
                group.pick_home_at_open_probability_rule.to_numpy(dtype=bool),
            )
    valid = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    valid = lane_s.sides(valid, valid.tue_open_home_spread)
    base_pick = valid.home_cover_probability_at_open.ge(0.5)
    base_correct = base_pick.eq(valid.margin_vs_open.gt(0)).astype(float)
    for arm in arms:
        scored = valid.copy()
        pick = scored[f"p_{arm}"].ge(0.5)
        scored["correct"] = pick.eq(scored.margin_vs_open.gt(0)).astype(float)
        scored["base_correct"] = base_correct
        for bucket, block in scored.groupby("bucket", observed=True):
            selections = {"all": block}
            selections.update({str(side): g for side, g in block.groupby("home_side")})
            for side, group in selections.items():
                if side not in lane_s.BUCKET_SIDES:
                    continue
                group = group.reset_index(drop=True)
                cells[lane_s.cell_name(arm, "bucket", str(bucket), side)] = lane_s.metric(
                    group, 100 * (group.correct - group.base_correct).to_numpy()
                )
    return cells


@cache
def _lane_s() -> Cells:
    return _season_and_bucket_cells(SCRATCHPAD / "laneS", ("S1", "S2"))


@cache
def _lane_q() -> Cells:
    return _season_and_bucket_cells(SCRATCHPAD / "laneQ", ("Q1", "Q2"))


@cache
def _lane_t_mapping() -> Cells:
    """Lane T's home-side mapping split, graded per spread bucket and side."""

    import home_side_mapping_opener_eval as lane_map

    from nfl_ats.home_side_mapping import home_side

    frame = pd.read_parquet(SCRATCHPAD / "laneT/paired.parquet")
    frame = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    frame["home_side"] = home_side(frame.tue_open_home_spread.to_numpy(dtype=float))
    cells: Cells = {}
    for bucket, block in frame.groupby("bucket", observed=True):
        for side in ("all", *sorted(block.home_side.unique())):
            group = block if side == "all" else block.loc[block.home_side.eq(side)]
            group = group.reset_index(drop=True)
            for arm in lane_map.ARMS:
                pick = group[f"p_{arm}"].ge(0.5)
                name = f"{arm}_home_{bucket}_{side}".replace(".", "p").replace("+", "plus")
                cells[f"{name}_accuracy"] = common.comparison(
                    group,
                    pick.to_numpy(),
                    group.pick_home_at_open_probability_rule.to_numpy(),
                )
    return cells


LANES: dict[str, tuple[Path, Callable[[], Cells]]] = {
    "laneG": (REPO / "artifacts/research/laneG/cells.json", _lane_g),
    "laneI": (REPO / "artifacts/research/laneI/cells.json", _lane_i),
    "laneR": (REPO / "artifacts/research/laneR/cells.json", _lane_r),
    "laneU": (REPO / "artifacts/research/laneU/cells.json", _lane_u),
    "laneT": (REPO / "artifacts/research/laneT/cells.json", _lane_t),
    "laneV": (REPO / "artifacts/research/laneV/cells.json", _lane_v),
    "laneS_seasons": (SCRATCHPAD / "laneS/seasons.json", _lane_s),
    "laneS_buckets": (SCRATCHPAD / "laneS/bucket_cells.json", _lane_s),
    "laneQ_seasons": (SCRATCHPAD / "laneQ/seasons.json", _lane_q),
    "laneQ_buckets": (SCRATCHPAD / "laneQ/bucket_cells.json", _lane_q),
    "laneT_mapping": (SCRATCHPAD / "laneT/home_split.parquet", _lane_t_mapping),
}


def _normalise(source: str) -> str:
    """Compare sources by repository-relative path.

    Lanes recorded their ``source`` inconsistently -- some absolute
    (``F:\\Repos\\nfl_py3\\artifacts\\...``), some relative
    (``artifacts/research/laneR/cells.json``) -- so matching on the raw string
    silently found nothing for half the lanes.
    """

    text = source.replace("\\", "/").casefold()
    root = str(REPO).replace("\\", "/").casefold() + "/"
    return text[len(root) :] if text.startswith(root) else text


def _cell_key(name: str, family: str | None, seasons: list[int] | tuple[int, int]) -> str | None:
    """``mod18_..._v1_s4_s4_overall_card_2020_2025`` -> ``s4_s4_overall_card``."""

    suffix = f"_{int(seasons[0])}_{int(seasons[1])}"
    if not family or not name.startswith(f"{family}_") or not name.endswith(suffix):
        return None
    return name[len(family) + 1 : len(name) - len(suffix)]


def _registry_rows(registry: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    wanted = _normalise(str(source))
    return {
        name: payload
        for name, payload in registry["signals"].items()
        if _normalise(str(payload.get("source", ""))) == wanted
    }


def _row_is_replaceable(payload: dict[str, Any]) -> str | None:
    """Why this row may not be re-recorded through ``weak-signals record``, if so.

    ``record --replace`` rebuilds the entry from CLI arguments, and a handful of
    fields have no argument: rewriting such a row would silently drop them. Those
    rows are reported and left alone rather than quietly flattened.
    """

    if payload.get("status") not in (None, "active"):
        return f"status={payload.get('status')!r}"
    if payload.get("corrections"):
        return "carries a corrections trail"
    if payload.get("superseded_by"):
        return "carries superseded_by"
    return None


def build_report() -> dict[str, Any]:
    registry_path = REPO / "registry/weak_signals.json"
    registry = json.loads(registry_path.read_text())
    lanes: dict[str, Any] = {}
    corrections: list[dict[str, Any]] = []
    for lane, (source, rebuild) in LANES.items():
        cells = rebuild()
        rows = _registry_rows(registry, source)
        matched = mismatched = unchanged = 0
        problems: list[dict[str, Any]] = []
        for name, payload in sorted(rows.items()):
            if payload.get("effect_units") != "accuracy_points":
                continue
            key = _cell_key(name, payload.get("family"), payload["seasons"])
            seasons = payload["seasons"]
            qualified = f"{key}@{int(seasons[0])}_{int(seasons[1])}" if key else None
            fresh = (cells.get(qualified) or cells.get(key)) if key else None
            if fresh is None:
                no_op = payload["effect"] == 0.0 and payload.get("interval") == [0.0, 0.0]
                problems.append(
                    {
                        "name": name,
                        "problem": "no-op cell, already corrected to 0.5"
                        if no_op
                        else "no matching re-measured cell",
                    }
                )
                continue
            recorded = {
                "delta": payload["effect"],
                "lower": payload["interval"][0],
                "upper": payload["interval"][1],
                "n": payload.get("sample_games"),
                "weeks": payload.get("sample_blocks"),
            }
            drifted = [
                field
                for field, value in recorded.items()
                if value is not None and abs(float(value) - float(fresh[field])) > TOLERANCE
            ]
            if drifted:
                mismatched += 1
                problems.append({"name": name, "problem": f"drifted fields: {drifted}"})
                continue
            matched += 1
            old = payload.get("probability_positive")
            new = float(fresh["probability_positive"])
            if old is not None and abs(float(old) - new) <= TOLERANCE:
                unchanged += 1
                continue
            blocked = _row_is_replaceable(payload)
            corrections.append(
                {
                    "name": name,
                    "lane": lane,
                    "cell": key,
                    "probability_positive_recorded": old,
                    "probability_positive_remeasured": new,
                    "blocked": blocked,
                }
            )
        lanes[lane] = {
            "cells_rebuilt": len(cells),
            "registry_rows": len(rows),
            "verified": matched,
            "drifted": mismatched,
            "already_correct": unchanged,
            "problems": problems,
        }
    return {
        "registry": str(registry_path),
        "measurement_fields_required_to_reproduce": list(MEASUREMENT_FIELDS[:7]),
        "lanes": lanes,
        "corrections": corrections,
        "nothing_is_closed_by_this": (
            "A re-measured summary is not a verdict. Every row keeps its "
            "classification, its effect and its interval; only "
            "probability_positive moves, and only where the measurement "
            "reproduced exactly."
        ),
    }


def _record_argv(payload: dict[str, Any], name: str, probability_positive: float) -> list[str]:
    argv = [
        str(REPO / ".tools/uv.exe"),
        "run",
        "--no-sync",
        "nfl-ats",
        "weak-signals",
        "record",
        "--replace",
        "--name",
        name,
        "--description",
        payload["description"],
        "--source",
        payload["source"],
        "--effect",
        repr(float(payload["effect"])),
        "--effect-units",
        payload["effect_units"],
        "--classification",
        payload["classification"],
        "--league",
        payload["league"],
        "--season-start",
        str(int(payload["seasons"][0])),
        "--season-end",
        str(int(payload["seasons"][1])),
        "--probability-positive",
        repr(float(probability_positive)),
        "--recorded-at",
        payload["recorded_at"],
    ]
    if payload.get("interval") is not None:
        argv += [
            "--interval-low",
            repr(float(payload["interval"][0])),
            "--interval-high",
            repr(float(payload["interval"][1])),
        ]
    for flag, key in (
        ("--standard-error", "standard_error"),
        ("--sample-games", "sample_games"),
        ("--sample-blocks", "sample_blocks"),
        ("--reliability", "reliability"),
        ("--family", "family"),
        ("--classification-evidence", "classification_evidence"),
        ("--closing-ground", "closing_ground"),
        ("--plain-summary", "plain_summary"),
        ("--category", "category"),
        ("--notes", "notes"),
    ):
        value = payload.get(key)
        if value is not None:
            argv += [flag, str(value)]
    return argv


def _registry_loads() -> str | None:
    """``None`` when the shared registry parses, else the validator's complaint.

    The registry is written by several lanes at once. One row anywhere in the
    file that the loader rejects makes EVERY ``weak-signals`` write fail, so a
    long apply pass has to be able to say *whose* row stopped it rather than
    reporting its own rows as failures.
    """

    result = subprocess.run(
        [str(REPO / ".tools/uv.exe"), "run", "--no-sync", "nfl-ats", "weak-signals", "status"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return None if result.returncode == 0 else result.stderr.strip() or result.stdout.strip()


def apply_corrections(report: dict[str, Any]) -> dict[str, Any]:
    """Re-record each verified row, skipping any already at the re-measured value.

    Idempotent on purpose: a pass interrupted by another lane's bad row can be
    re-run without repeating the writes it already made.
    """

    blocked_by = _registry_loads()
    if blocked_by is not None:
        return {"applied": [], "skipped": [], "already_correct": [], "blocked_by": blocked_by}
    registry = json.loads((REPO / "registry/weak_signals.json").read_text())
    applied: list[str] = []
    skipped: list[dict[str, Any]] = []
    already: list[str] = []
    failures: list[dict[str, str]] = []
    for correction in report["corrections"]:
        if correction["blocked"]:
            skipped.append(correction)
            continue
        name = correction["name"]
        payload = registry["signals"][name]
        target = float(correction["probability_positive_remeasured"])
        stored = payload.get("probability_positive")
        if stored is not None and abs(float(stored) - target) <= TOLERANCE:
            already.append(name)
            continue
        result = subprocess.run(
            _record_argv(payload, name, target), cwd=REPO, capture_output=True, text=True
        )
        if result.returncode != 0:
            failures.append({"name": name, "error": result.stderr.strip()})
            break
        applied.append(name)
    return {
        "applied": applied,
        "skipped": skipped,
        "already_correct": already,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="where to write the JSON report")
    parser.add_argument(
        "--from-report",
        type=Path,
        help="reuse an earlier report's verified corrections instead of re-running the "
        "bootstraps; the apply pass is idempotent, so this is how an interrupted "
        "pass is resumed",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="re-record the verified rows through `nfl-ats weak-signals record --replace`",
    )
    args = parser.parse_args()
    report = json.loads(args.from_report.read_text()) if args.from_report else build_report()
    if args.apply:
        report["application"] = apply_corrections(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(report, args.report)
    summary = {
        lane: {k: v for k, v in stats.items() if k != "problems"}
        for lane, stats in report["lanes"].items()
    }
    outcome: dict[str, Any] = {"lanes": summary, "corrections": len(report["corrections"])}
    application = report.get("application")
    if application is not None:
        outcome["application"] = {
            key: len(value) if isinstance(value, list) else value
            for key, value in application.items()
        }
        outcome["failures"] = application.get("failures")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()

"""Tank-zone fade tilt overlay, STACKED ON PRODUCTION, at the opener grade.

The question this answers is not "does the tank-zone fade beat a bare market
baseline" -- it is the only question that decides anything: **does it add
anything on top of the picks that are actually PLAYED?** (AGENTS.md, "composition
is not the signal": an overlay positive alone can go negative stacked on the
played chain.)

Baseline archive: ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``
-- the tracked run of ``nfl-ats opener-evaluation`` for the active
``weak_stack``/ridge-alpha-10 model, 1,537 REG games 2020-2025, graded at the
Tuesday OPENER under the PRODUCTION probability rule
(``correct_at_open_probability_rule``), baseline 53.36% on 1,503 scored games.
Same frozen artifact every other overlay back-test in this repo uses, loaded
through ``scripts/overlay_stack_backtest.py``'s own ``load_inputs`` /
``build_predictions_frame`` / ``build_eval_frame`` / ``run_both_blockings``.

**The incumbent is the production chain, not the bare model.** Production's
played card applies ``nfl_ats.four_overlay_composition``'s frozen union policy
``overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`` -- four
members (coach fade, division-revenge tilt, player-arrests back-side policy,
spread-gap zone fade), each evaluated against the raw card, the union
complemented exactly once. This script reconstructs that union on the archive
(the three schedule/card-derived members from ``run_overlays``; the arrests
member from ``scripts/overlay_subset_composition.py``'s own
``reconstruct_arrest_flip_set``, the same reconstruction the published
production-chain figures came from), calls that the INCUMBENT, then adds the
tank-zone fade's flips on top with the identical OR/complement-once semantics
the production policy itself uses.

Flip rule under test (frozen before this script was run; see
``src/nfl_ats/tank_zone_fade_tilt_overlay.py`` and
``docs/tank_zone_fade_tilt_overlay.md``): REG, weeks 14-18 only, exactly one
side in the league's bottom two records entering the week, and the pick is on
that side -> flip it off. Parameter-free.

**Verdict handling, stated before the numbers exist.** This is a MINED-SEASONS
read on an archive several already-registered overlays were themselves screened
against; it is CONTEXT, not a gate, and it spends no rotation-registry window.
Per the binding taxonomy: an interval containing zero is NEVER grounds to
reject, fail or close an experiment; only a RESOLVED wrong sign (the whole
interval on the wrong side of zero), zero split-half reliability, or a positive
control proven able to detect an effect that size closes a line of work.
Everything else is ``unresolved_below_power``, reported with
``probability_positive``, never the binary "contains zero". A thin flagged
population (weeks 14-18 are ~5% of the slate) is a POWER statement, not a
defect -- ``n_flipped`` is reported explicitly for exactly that reason.

Usage (from the repo root, per AGENTS.md environment conventions)::

    .\\.tools\\uv.exe run --no-sync python scripts/tank_zone_fade_stacked_backtest.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from overlay_stack_backtest import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_eval_frame,
    build_predictions_frame,
    load_inputs,
    run_both_blockings,
    run_overlays,
    verify_no_direction_conflicts,
)
from overlay_subset_composition import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    reconstruct_arrest_flip_set,
)

from nfl_ats.four_overlay_composition import (  # noqa: E402
    COACH_FADE,
    COMPOSITION_ORDER,
    DIVISION_REVENGE_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    POLICY_ID,
    SPREAD_GAP_ZONE_FADE,
)
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.tank_zone_fade_tilt_overlay import (  # noqa: E402
    CHALLENGER_ID,
    OVERLAY_WEEK_MAX,
    OVERLAY_WEEK_MIN,
    apply_tank_zone_fade_tilt_overlay,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "tank_zone_fade_stacked"
DEFAULT_SAMPLES = 20_000

#: Recorded in the artifact and quoted in the report. Chosen as this session's
#: UTC date before any result was seen, mirroring every sibling script's
#: fixed-seed convention.
DEFAULT_SEED = 20260901

#: The four production members, mapped onto the overlay names
#: ``overlay_stack_backtest.run_overlays`` returns. The arrests member has no
#: ``apply_*`` overlay on the historical archive and is reconstructed instead.
PRODUCTION_MEMBER_TO_OVERLAY: dict[str, str] = {
    COACH_FADE: "coach_fade_overlay",
    DIVISION_REVENGE_TILT: "division_revenge_tilt_overlay",
    SPREAD_GAP_ZONE_FADE: "spread_gap_zone_fade_overlay",
}


def production_chain_flip_sets(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    player_features: pd.DataFrame,
    per_game: pd.DataFrame,
    features_path: Path,
    incidents_path: Path,
) -> dict[str, set[str]]:
    """The four production members' flip sets on the frozen opener archive."""

    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)

    arrest_ids, _scored = reconstruct_arrest_flip_set(per_game, features_path, incidents_path)

    members: dict[str, set[str]] = {
        member: flip_sets[overlay_name]
        for member, overlay_name in PRODUCTION_MEMBER_TO_OVERLAY.items()
    }
    members[PLAYER_ARRESTS_BACK_SIDE_POLICY] = arrest_ids
    missing = [member for member in COMPOSITION_ORDER if member not in members]
    if missing:
        raise ValueError(f"production chain members not reconstructed: {missing}")
    return members


def stacked_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_baseline"])
    if valid.empty:
        return {
            "baseline_accuracy": float("nan"),
            "production_accuracy": float("nan"),
            "candidate_accuracy": float("nan"),
            "candidate_minus_production": float("nan"),
            "production_minus_baseline": float("nan"),
        }
    baseline = float(valid["correct_baseline"].mean())
    production = float(valid["correct_production"].mean())
    candidate = float(valid["correct_candidate"].mean())
    return {
        "baseline_accuracy": baseline,
        "production_accuracy": production,
        "candidate_accuracy": candidate,
        "candidate_minus_production": candidate - production,
        "production_minus_baseline": production - baseline,
    }


def solo_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["correct_baseline"])
    if valid.empty:
        return {"baseline_accuracy": float("nan"), "solo_minus_baseline": float("nan")}
    baseline = float(valid["correct_baseline"].mean())
    solo = float(valid["correct_solo_tank_zone"].mean())
    return {
        "baseline_accuracy": baseline,
        "solo_accuracy": solo,
        "solo_minus_baseline": solo - baseline,
    }


def run_backtest(
    per_game_path: Path,
    data_root: Path,
    features_path: Path,
    incidents_path: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        per_game_path, data_root
    )
    predictions = build_predictions_frame(per_game, schedules)

    members = production_chain_flip_sets(
        predictions, schedules, player_features, per_game, features_path, incidents_path
    )
    production_ids: set[str] = set().union(*members.values())

    tilt = apply_tank_zone_fade_tilt_overlay(predictions, schedules)
    tank_ids = {flip.game_id for flip in tilt.flips}

    # build_eval_frame gives us the frozen baseline correctness column with the
    # archive's own push handling; the six-overlay stack columns it also
    # produces are unused here (empty flip sets), so correct_combined is
    # identical to correct_baseline by construction.
    empty_flip_sets: dict[str, set[str]] = {name: set() for name in OVERLAY_NAMES}
    eval_frame = build_eval_frame(predictions, per_game, empty_flip_sets)
    eval_frame = eval_frame[["game_id", "season", "week", "correct_baseline"]].copy()

    eval_frame["in_production"] = eval_frame["game_id"].isin(production_ids)
    eval_frame["in_tank_zone"] = eval_frame["game_id"].isin(tank_ids)
    candidate_ids = production_ids | tank_ids
    eval_frame["in_candidate"] = eval_frame["game_id"].isin(candidate_ids)

    base = eval_frame["correct_baseline"]
    eval_frame["correct_production"] = np.where(eval_frame["in_production"], 1.0 - base, base)
    eval_frame["correct_candidate"] = np.where(eval_frame["in_candidate"], 1.0 - base, base)
    eval_frame["correct_solo_tank_zone"] = np.where(eval_frame["in_tank_zone"], 1.0 - base, base)

    scored = eval_frame.dropna(subset=["correct_baseline"])
    incremental_ids = sorted(tank_ids - production_ids)
    scored_tank_ids = sorted(set(scored.loc[scored["in_tank_zone"], "game_id"]))
    scored_incremental = sorted(
        set(scored.loc[scored["in_tank_zone"] & ~scored["in_production"], "game_id"])
    )

    stacked = run_both_blockings(eval_frame, stacked_metric, samples=samples, seed=seed)
    solo = run_both_blockings(eval_frame, solo_metric, samples=samples, seed=seed)

    per_season = (
        eval_frame.loc[eval_frame["in_tank_zone"]].groupby("season").size().astype(int).to_dict()
    )

    return {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "challenger_id": CHALLENGER_ID,
        "rule": (
            f"REG, weeks {OVERLAY_WEEK_MIN}-{OVERLAY_WEEK_MAX} only. Exactly one side of the "
            "game is in the league's bottom two records entering the week (standings from "
            "strictly prior weeks of the same season), and the model's own forced pick IS "
            "that side -> flip the pick to the other side. Both-flagged games are never "
            "touched. Parameter-free; no threshold, nothing fitted to outcomes."
        ),
        "read_kind": (
            "MINED-SEASONS read on an already-looked-at archive; CONTEXT, not a gate. "
            "Several members of the incumbent production chain were themselves screened "
            "on windows this 2020-2025 archive re-touches. No rotation-registry window is "
            "spent by this run."
        ),
        "closing_grounds_taxonomy": (
            "An interval containing zero is NEVER grounds to reject, fail or close an "
            "experiment. Only (1) a refuted mechanism -- a RESOLVED wrong sign (whole "
            "interval on the wrong side of zero) or zero split-half reliability -- or "
            "(2) bounded by a positive control proven able to detect an effect that size "
            "closes a line of work. Everything else is unresolved_below_power: report "
            "probability_positive, never the binary 'contains zero'."
        ),
        "source_artifact": str(per_game_path),
        "source_artifact_sha256": sha256_file(per_game_path),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "arrest_features": str(features_path),
        "arrest_incidents": str(incidents_path),
        "incumbent_policy_id": POLICY_ID,
        "incumbent_members": list(COMPOSITION_ORDER),
        "incumbent_member_flip_counts": {member: len(ids) for member, ids in members.items()},
        "grading_rule": (
            "production probability rule (home_cover_probability >= 0.5) at the "
            "Tuesday-opener decision line, matching pool.py/backtest.py"
        ),
        "seasons": [int(eval_frame["season"].min()), int(eval_frame["season"].max())],
        "n_games": len(eval_frame),
        "n_pushes": int(eval_frame["correct_baseline"].isna().sum()),
        "n_scored_games": len(scored),
        "week_block_count": int(eval_frame[["season", "week"]].drop_duplicates().shape[0]),
        "season_block_count": int(eval_frame["season"].nunique()),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "n_flipped": len(tank_ids),
        "n_flipped_scored": len(scored_tank_ids),
        "n_flipped_incremental_over_production": len(incremental_ids),
        "n_flipped_incremental_over_production_scored": len(scored_incremental),
        "flipped_game_ids": sorted(tank_ids),
        "incremental_game_ids": incremental_ids,
        "both_tank_zone_game_ids": list(tilt.both_tank_zone_games),
        "flips_per_season": {str(k): int(v) for k, v in per_season.items()},
        "production_union_flip_count": len(production_ids),
        "candidate_union_flip_count": len(candidate_ids),
        "stacked_on_production": stacked,
        "solo_vs_bare_baseline": solo,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    result = run_backtest(
        args.per_game_artifact,
        args.data_root,
        args.features,
        args.incidents,
        samples=args.samples,
        seed=args.seed,
    )

    configuration = {
        "command": "tank-zone-fade-stacked-backtest",
        "per_game_artifact": str(args.per_game_artifact),
        "data_root": str(args.data_root),
        "arrest_features": str(args.features),
        "arrest_incidents": str(args.incidents),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "challenger_id": CHALLENGER_ID,
        "overlay_week_window": [OVERLAY_WEEK_MIN, OVERLAY_WEEK_MAX],
        "incumbent_policy_id": POLICY_ID,
        "predeclaration": "docs/tank_zone_fade_tilt_overlay.md",
    }
    payload = {
        **result,
        "provenance": artifact_provenance(
            configuration, args.per_game_artifact, project_root=REPO_ROOT
        ),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_path = output_dir / "results.json"
    stacked = payload["stacked_on_production"]["week_blocked"]["candidate_minus_production"]
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="tank-zone-fade-stacked-backtest",
        metrics={
            "candidate_minus_production_accuracy_points": stacked["estimate"] * 100.0,
            "probability_positive": stacked["probability_positive"],
            "n_flipped": payload["n_flipped"],
            "n_flipped_incremental_over_production": payload[
                "n_flipped_incremental_over_production"
            ],
            "n_scored_games": payload["n_scored_games"],
        },
        notes=(
            "Tank-zone fade tilt (registry cell motivation_ladder_tank_zone_wk14_18) stacked "
            "on the played four-overlay production chain at the Tuesday opener grade. A "
            "MINED-SEASONS read on an already-looked-at 2020-2025 archive: context, not a "
            "gate, and it spends no rotation-registry window. See "
            "docs/tank_zone_fade_tilt_overlay.md."
        ),
        project_root=REPO_ROOT,
    )

    print(f"Wrote {output_path}")
    print(
        f"n_flipped={payload['n_flipped']} "
        f"(incremental over production: {payload['n_flipped_incremental_over_production']}) "
        f"scored games={payload['n_scored_games']}"
    )
    for block in ("week_blocked", "season_blocked"):
        row = payload["stacked_on_production"][block]["candidate_minus_production"]
        print(
            f"candidate - production ({block}): {row['estimate'] * 100:+.4f} pts "
            f"95% [{row['lower'] * 100:+.4f}, {row['upper'] * 100:+.4f}] "
            f"P+ {row['probability_positive']:.4f}"
        )
    accuracy = payload["stacked_on_production"]["week_blocked"]
    print(
        "accuracies: baseline "
        f"{accuracy['baseline_accuracy']['estimate'] * 100:.4f}%, production "
        f"{accuracy['production_accuracy']['estimate'] * 100:.4f}%, candidate "
        f"{accuracy['candidate_accuracy']['estimate'] * 100:.4f}%"
    )
    solo = payload["solo_vs_bare_baseline"]["week_blocked"]["solo_minus_baseline"]
    print(
        f"context -- solo vs bare baseline (week-blocked): {solo['estimate'] * 100:+.4f} pts "
        f"95% [{solo['lower'] * 100:+.4f}, {solo['upper'] * 100:+.4f}] "
        f"P+ {solo['probability_positive']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

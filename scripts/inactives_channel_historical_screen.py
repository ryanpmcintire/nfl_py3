"""Run the frozen Section 5 historical inactives proxy screen.

The protocol is frozen in ``docs/inactives_channel.md``.  This script is
measure-only: it writes an artifact and per-game rows, but registry recording
is deliberately performed by the two explicit CLI commands after the result
has been inspected.  Snap counts are used only as a post-game label; they are
never supplied to model training or to the pregame feature table.

The candidate changes only the seven existing injury-unavailability columns,
using the production ``players._injury_features`` aggregation.  The two
value-lost columns are intentionally unchanged, matching the live overlay's
declared scope boundary.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.availability import fixed_unavailability  # noqa: E402
from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.io import atomic_parquet, run_id  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock  # noqa: E402
from nfl_ats.players import (  # noqa: E402
    PLAYER_INJURY_STATE_METRICS,
    _injury_features,
    _position_group,
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    load_player_snapshot,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

FAMILY = "inactives_channel_historical_proxy_v1"
SEASONS = (2020, 2021)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
NULL_DRAWS = 200
NULL_SEED = 20260902
MIN_TRAIN_GAMES = DEFAULT_MIN_TRAIN_GAMES
PLAYER_SNAPSHOT_ID = "20260817T184901Z"


def _identity_roles(
    snaps: pd.DataFrame, season: int, week: int
) -> dict[str, dict[str, float | str]]:
    """Return production-shaped roles from strictly earlier same-season snaps."""

    earlier = snaps.loc[(snaps["season"] == season) & (snaps["week"] < week)].copy()
    if earlier.empty:
        return {}
    earlier = earlier.sort_values("week").drop_duplicates("gsis_id", keep="last")
    roles: dict[str, dict[str, float | str]] = {}
    for row in earlier.itertuples(index=False):
        player = str(row.gsis_id)
        roles[player] = {
            "offense_pct": float(pd.to_numeric(row.offense_pct, errors="coerce"))
            if pd.notna(row.offense_pct)
            else 0.0,
            "defense_pct": float(pd.to_numeric(row.defense_pct, errors="coerce"))
            if pd.notna(row.defense_pct)
            else 0.0,
            "st_pct": float(pd.to_numeric(row.st_pct, errors="coerce"))
            if pd.notna(row.st_pct)
            else 0.0,
            "position_group": _position_group(row.position),
        }
    return roles


def build_historical_increments(
    features: pd.DataFrame,
    injuries: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame,
    *,
    seasons: tuple[int, int],
) -> tuple[dict[str, dict[str, dict[str, float]]], pd.DataFrame]:
    """Label surprise absences and fold them through production aggregation.

    The report cutoff is the game's kickoff minus 24 hours, the existing
    late-week Saturday convention.  A zero-snap player qualifies when their
    newest report row visible at that cutoff is absent or is not ``Out``.
    """

    frame = regular_season_rows(features).copy()
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    injuries = canonicalize_injuries(injuries)
    rosters = canonicalize_rosters(rosters)
    snaps = attach_snap_player_ids(canonicalize_snaps(snaps), rosters)
    snaps["total_snaps"] = (
        snaps[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    )
    target = frame.loc[frame["season"].between(*seasons)].copy()
    injury_groups = {
        key: group.sort_values("date_modified")
        for key, group in injuries.groupby(["season", "week", "team"], sort=False)
    }
    # nflverse snap_counts omits almost every player with zero snaps.  The
    # historical proxy therefore starts with the weekly active roster and
    # subtracts players with a positive snap row; this is the same
    # roster-minus-play construction used by the project's availability
    # outcome builder and preserves the predeclared zero-snap label.
    played = snaps.loc[snaps["total_snaps"].gt(0), ["season", "week", "game_id", "team", "gsis_id"]]
    played = played.drop_duplicates()
    roster_groups = dict(
        rosters.loc[rosters["status"].isin(("ACT", "INA"))].groupby(
            ["season", "week", "team"], sort=False
        )
    )
    played_groups = {
        key: set(group["gsis_id"].astype(str))
        for key, group in played.groupby(["season", "week", "team"], sort=False)
    }
    increments: dict[str, dict[str, dict[str, float]]] = {}
    label_rows: list[dict[str, Any]] = []
    for game in target.itertuples(index=False):
        game_id = str(game.game_id)
        kickoff = pd.Timestamp(game.kickoff)
        cutoff = kickoff - pd.Timedelta(hours=24)
        for side in ("home", "away"):
            team = TEAM_ABBREVIATION_ALIASES.get(
                str(getattr(game, f"{side}_team")), str(getattr(game, f"{side}_team"))
            )
            group = roster_groups.get((int(game.season), int(game.week), team))
            if group is None or group.empty:
                continue
            played_ids = played_groups.get((int(game.season), int(game.week), team), set())
            zero = group.loc[
                group["gsis_id"].notna() & ~group["gsis_id"].astype(str).isin(played_ids)
            ].copy()
            if zero.empty:
                continue
            report = injury_groups.get((int(game.season), int(game.week), team))
            visible: dict[str, pd.Series] = {}
            if report is not None:
                report = report.loc[report["date_modified"].le(cutoff)]
                for player, report_rows in report.groupby("gsis_id", sort=False):
                    visible[str(player)] = report_rows.iloc[-1]
            roles = _identity_roles(snaps, int(game.season), int(game.week))
            player_rows: list[dict[str, Any]] = []
            for row in zero.itertuples(index=False):
                player = str(row.gsis_id)
                latest = visible.get(player)
                credited = (
                    0.0
                    if latest is None
                    else fixed_unavailability(latest.report_status, latest.practice_status)
                )
                if credited >= 1.0:
                    continue
                player_rows.append(
                    {"gsis_id": player, "position": row.position, "_unavailability": 1.0 - credited}
                )
                label_rows.append(
                    {
                        "game_id": game_id,
                        "season": int(game.season),
                        "week": int(game.week),
                        "team": team,
                        "gsis_id": player,
                        "report_visible": latest is not None,
                        "report_status": None if latest is None else str(latest.report_status),
                        "cutoff": cutoff,
                    }
                )
            if player_rows:
                side_increments = _injury_features(pd.DataFrame(player_rows), roles)
                increments.setdefault(game_id, {})[side] = side_increments
    labels = pd.DataFrame(label_rows)
    return increments, labels


def apply_increments(
    features: pd.DataFrame, increments: dict[str, dict[str, dict[str, float]]]
) -> pd.DataFrame:
    """Copy the feature table and modify only target-game unavailability fields."""

    adjusted = features.copy()
    game_ids = adjusted["game_id"].astype(str)
    for game_id, sides in increments.items():
        positions = adjusted.index[game_ids.eq(game_id)]
        for metric in PLAYER_INJURY_STATE_METRICS:
            home = f"home_{metric}"
            away = f"away_{metric}"
            diff = f"diff_{metric}"
            if home not in adjusted or away not in adjusted:
                continue
            adjusted.loc[positions, home] += float(sides.get("home", {}).get(metric, 0.0))
            adjusted.loc[positions, away] += float(sides.get("away", {}).get(metric, 0.0))
            if diff in adjusted:
                adjusted.loc[positions, diff] = (
                    adjusted.loc[positions, home] - adjusted.loc[positions, away]
                )
    return adjusted


def _paired(baseline: pd.DataFrame, candidate: pd.DataFrame, playable: pd.Series) -> pd.DataFrame:
    cols = [
        "game_id",
        "season",
        "week",
        "correct_at_open_probability_rule",
        "pick_home_at_open_probability_rule",
    ]
    left = baseline.loc[playable, cols].rename(
        columns={cols[3]: "baseline_correct", cols[4]: "baseline_pick_home"}
    )
    right = candidate.loc[playable, cols].rename(
        columns={cols[3]: "candidate_correct", cols[4]: "candidate_pick_home"}
    )
    result = left.merge(right, on=["game_id", "season", "week"], how="inner", validate="one_to_one")
    result = result.dropna(subset=["candidate_correct", "baseline_correct"]).copy()
    result["candidate_correct"] = result["candidate_correct"].astype(float)
    result["baseline_correct"] = result["baseline_correct"].astype(float)
    result["delta"] = result["candidate_correct"] - result["baseline_correct"]
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _metric(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "candidate_minus_tuesday": float(
            (rows["candidate_correct"] - rows["baseline_correct"]).mean()
        )
    }


def _bootstrap(paired: pd.DataFrame, block: str) -> dict[str, float]:
    row = week_blocked_bootstrap(
        paired, _metric, block=block, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    ).iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "samples": BOOTSTRAP_SAMPLES,
        "block": block,
    }


def permutation_null(paired: pd.DataFrame) -> dict[str, Any]:
    """Swap candidate/baseline labels within week, retaining slate composition."""

    rng = np.random.default_rng(NULL_SEED)
    deltas: list[float] = []
    for _ in range(NULL_DRAWS):
        values = paired["delta"].to_numpy(dtype=float).copy()
        for positions in paired.groupby(["season", "week"], sort=False).indices.values():
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(positions))
            values[positions] *= signs
        deltas.append(float(values.mean()))
    values = np.asarray(deltas)
    observed = float(paired["delta"].mean())
    return {
        "observed": observed,
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "null_lower": float(np.quantile(values, 0.025)),
        "null_upper": float(np.quantile(values, 0.975)),
        "share_at_or_above_observed": float(np.mean(values >= observed)),
        "draws": NULL_DRAWS,
        "seed": NULL_SEED,
        "note": "within-week paired-label swap; not zero-centred by assumption",
    }


def _experiment_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the registry row compact while retaining the screen headlines."""

    return {
        "status": result["status"],
        "paired_games": result["population"]["paired_games"],
        "paired_weeks": result["population"]["paired_weeks"],
        "candidate_week_estimate": result["candidate_vs_tuesday"]["week"]["estimate"],
        "candidate_week_probability_positive": result["candidate_vs_tuesday"]["week"][
            "probability_positive"
        ],
        "positive_control_week_estimate": result["positive_control_oracle"]["week"]["estimate"],
    }


def main() -> int:
    features_path = REPO / "data/processed/game_features_weak_stack.parquet"
    features = pd.read_parquet(features_path)
    snapshot = latest_player_snapshot(REPO / "data/players/raw")
    injuries, rosters, snaps = load_player_snapshot(snapshot)
    increments, labels = build_historical_increments(
        features, injuries, rosters, snaps, seasons=SEASONS
    )
    adjusted = apply_increments(features, increments)
    config = {
        "family": FAMILY,
        "seasons": list(SEASONS),
        "grade": "opener",
        "feature_profile": "weak_stack",
        "method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "min_train_games": MIN_TRAIN_GAMES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "null_draws": NULL_DRAWS,
        "null_seed": NULL_SEED,
        "player_snapshot": snapshot.snapshot_id,
        "scope_boundary": "seven injury-unavailability columns only; value-lost columns unchanged",
        "predeclaration": "docs/inactives_channel.md Section 5 execution protocol",
    }
    baseline = opener_pick_evaluation(
        REPO / "data/market/raw",
        features,
        active_model_config={
            "feature_profile": "weak_stack",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
        },
        min_train_games=MIN_TRAIN_GAMES,
    )
    candidate = opener_pick_evaluation(
        REPO / "data/market/raw",
        adjusted,
        active_model_config={
            "feature_profile": "weak_stack",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
        },
        min_train_games=MIN_TRAIN_GAMES,
    )
    target = features[["game_id", "season", "week", "kickoff"]].drop_duplicates("game_id")
    target["kickoff"] = pd.to_datetime(target["kickoff"], utc=True)
    target = target.loc[target["season"].between(*SEASONS)].copy()
    # Structural playability uses the unchanged Sunday lock rule: exclude only SNF/MNF.
    target["deadline"] = target["kickoff"].map(lambda _: pd.NaT)
    lock = sunday_pick_lock(target["kickoff"])
    target["deadline"] = target["kickoff"].map(lambda value: pick_deadline(value, lock))
    target["playable"] = target["kickoff"] - pd.Timedelta(minutes=90) < target["deadline"]
    playable = (
        candidate["game_id"]
        .astype(str)
        .map(target.set_index("game_id")["playable"].to_dict())
        .fillna(False)
    )
    paired = _paired(baseline, candidate, candidate["season"].between(*SEASONS) & playable)
    control = paired.copy()
    control["candidate_correct"] = 1.0
    result = {
        "status": "scored",
        "configuration": config,
        "population": {
            "target_games": len(target),
            "playable_games": int(target["playable"].sum()),
            "proxy_label_rows": len(labels),
            "adjusted_games": len(increments),
            "paired_games": len(paired),
            "paired_weeks": int(paired.groupby(["season", "week"]).ngroups),
        },
        "positive_control_oracle": {
            "week": _bootstrap(control, "week"),
            "season": _bootstrap(control, "season"),
        },
        "permutation_null": permutation_null(paired),
        "candidate_vs_tuesday": {
            "week": _bootstrap(paired, "week"),
            "season": _bootstrap(paired, "season"),
        },
    }
    output = REPO / "artifacts" / "inactives_channel_historical_proxy" / run_id()
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **result,
        "provenance": artifact_provenance(config, features_path, project_root=REPO),
    }
    write_experiment_artifact(
        output,
        "results.json",
        metadata,
        command="inactives-channel-historical-screen",
        metrics=_experiment_metrics(result),
        notes=(
            "Frozen Section 5 historical inactives proxy screen; prediction-level pairs, "
            "positive control, and within-week paired-label null retained in the artifact."
        ),
        source="scripts/inactives_channel_historical_screen.py",
        rotation_family=FAMILY,
        project_root=REPO,
    )
    atomic_parquet(paired, output / "paired_predictions.parquet")
    atomic_parquet(labels, output / "historical_proxy_labels.parquet")
    print(json.dumps({"artifact": str(output), **result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

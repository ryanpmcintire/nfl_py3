r"""Evaluate the frozen broad player-arrest back-side policy at the opener.

The predeclaration is ``docs/player_arrests_policy_eval.md``. This script must
not reinterpret that policy: it flips the archived production probability-rule
pick to the sole 1-14-day incident-affected side only when production opposes
that side, grades both arms at the same Tuesday opener, and preserves pushes.

Usage::

    .\.tools\uv.exe run python scripts/player_arrests_policy_eval.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import pick_correct
from nfl_ats.estimation_variance import guard_block_count
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OPENER = REPO / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_FEATURES = REPO / "data/processed/game_features_pbp.parquet"
DEFAULT_INCIDENTS = (
    REPO / "data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet"
)
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/player_arrests_policy_eval"
DEFAULT_REGISTRY_ROOT = REPO / "registry"

WINDOW_DAYS = 14
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260820
SIGNAL_NAME = "player_arrests_recent_14d_back_side_policy_opener"

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
    "IN": "IND",
}


class PolicyEvaluationError(ValueError):
    """Frozen input or policy contract was violated."""


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PolicyEvaluationError(f"{label} is missing columns: {', '.join(missing)}")


def broad_incident_game_flags(
    games: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    window_days: int = WINDOW_DAYS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one broad, point-in-time incident flag for each game side."""

    if window_days != WINDOW_DAYS:
        raise PolicyEvaluationError(
            f"Frozen player-arrest policy requires window_days={WINDOW_DAYS}"
        )
    _require_columns(
        games,
        {"game_id", "gameday", "home_team", "away_team"},
        "game identity table",
    )
    _require_columns(
        incidents,
        {"record_id", "incident_date", "team"},
        "safe incident index",
    )
    if games["game_id"].duplicated().any():
        raise PolicyEvaluationError("game identity table contains duplicate game_id rows")
    if incidents["record_id"].duplicated().any():
        raise PolicyEvaluationError("safe incident index contains duplicate record_id rows")

    safe = incidents[["record_id", "incident_date", "team"]].copy()
    safe["incident_date"] = pd.to_datetime(safe["incident_date"], errors="coerce")
    if safe["incident_date"].isna().any():
        raise PolicyEvaluationError("safe incident index contains invalid incident dates")
    safe["team"] = safe["team"].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)

    identity = games[["game_id", "gameday", "home_team", "away_team"]].copy()
    for team_column in ("home_team", "away_team"):
        identity[team_column] = (
            identity[team_column].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)
        )
    identity["gameday"] = pd.to_datetime(identity["gameday"], errors="coerce")
    if identity["gameday"].isna().any():
        raise PolicyEvaluationError("game identity table contains invalid gameday values")
    days_since_tuesday = (identity["gameday"].dt.weekday - 1) % 7
    identity["decision_date"] = (
        identity["gameday"] - pd.to_timedelta(days_since_tuesday, unit="D")
    ).dt.normalize()

    schedule_teams = set(identity["home_team"]) | set(identity["away_team"])
    mapped = safe.loc[safe["team"].isin(schedule_teams)].copy()
    mapped = mapped.sort_values(["incident_date", "team", "record_id"])

    flags = identity[["game_id"]].copy()
    for side, team_column in (("home", "home_team"), ("away", "away_team")):
        team_games = identity[["game_id", "decision_date", team_column]].rename(
            columns={team_column: "team"}
        )
        team_games = team_games.sort_values(["decision_date", "team", "game_id"])
        joined = pd.merge_asof(
            team_games,
            mapped,
            by="team",
            left_on="decision_date",
            right_on="incident_date",
            direction="backward",
            allow_exact_matches=False,
        )
        age = (joined["decision_date"] - joined["incident_date"]).dt.days
        joined[f"{side}_incident_flag"] = age.between(1, window_days, inclusive="both")
        flags = flags.merge(
            joined[["game_id", f"{side}_incident_flag"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    return flags, {
        "source_incidents": len(safe),
        "schedule_mapped_incidents": len(mapped),
    }


def apply_frozen_policy(opener: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Apply the predeclared flip rule and grade both arms at the opener."""

    _require_columns(
        opener,
        {
            "game_id",
            "season",
            "week",
            "margin_vs_open",
            "pick_home_at_open_probability_rule",
            "correct_at_open_probability_rule",
        },
        "frozen opener evaluation",
    )
    _require_columns(flags, {"game_id", "home_incident_flag", "away_incident_flag"}, "flags")
    if opener["game_id"].duplicated().any() or flags["game_id"].duplicated().any():
        raise PolicyEvaluationError("opener and flag tables must each have unique game_id rows")

    scored = opener.merge(flags, on="game_id", how="left", validate="one_to_one")
    if scored[["home_incident_flag", "away_incident_flag"]].isna().any().any():
        raise PolicyEvaluationError("every frozen opener game must receive both incident flags")
    scored["home_incident_flag"] = scored["home_incident_flag"].astype(bool)
    scored["away_incident_flag"] = scored["away_incident_flag"].astype(bool)

    production_pick = scored["pick_home_at_open_probability_rule"].astype(bool)
    exactly_one = scored["home_incident_flag"] ^ scored["away_incident_flag"]
    production_opposes = production_pick.ne(scored["home_incident_flag"])
    scored["policy_flip"] = exactly_one & production_opposes
    scored["candidate_pick_home"] = production_pick.where(
        ~scored["policy_flip"], scored["home_incident_flag"]
    )

    margin = pd.to_numeric(scored["margin_vs_open"], errors="coerce")
    recomputed_baseline = pick_correct(production_pick, margin).where(margin.notna())
    archived_baseline = pd.to_numeric(scored["correct_at_open_probability_rule"], errors="coerce")
    if not np.allclose(recomputed_baseline, archived_baseline, equal_nan=True):
        raise PolicyEvaluationError(
            "archived production correctness does not match its pick and opener margin"
        )
    scored["candidate_correct_at_open"] = pick_correct(scored["candidate_pick_home"], margin).where(
        margin.notna()
    )
    if not scored.loc[margin.eq(0.0), "candidate_correct_at_open"].isna().all():
        raise PolicyEvaluationError("candidate policy did not preserve opener pushes")
    return scored


def paired_policy_bootstrap(
    scored: pd.DataFrame,
    *,
    block: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Whole-block bootstrap of candidate-minus-production accuracy points."""

    if block not in {"week", "season"}:
        raise ValueError("block must be 'week' or 'season'")
    if samples < 10:
        raise ValueError("samples must be at least 10")
    required = {"season", "correct_at_open_probability_rule", "candidate_correct_at_open"}
    if block == "week":
        required.add("week")
    _require_columns(scored, required, "scored policy frame")

    paired = scored.dropna(
        subset=["correct_at_open_probability_rule", "candidate_correct_at_open"]
    ).copy()
    if paired.empty:
        raise PolicyEvaluationError("no paired graded games are available")
    group_columns = ["season", "week"] if block == "week" else ["season"]
    grouped_indices = list(paired.groupby(group_columns, sort=False, dropna=False).indices.values())
    verdict = guard_block_count(
        len(grouped_indices),
        on_degenerate="warn",
        context=f"player_arrests_policy_eval(block={block})",
    )
    row_delta = (
        100.0
        * (
            paired["candidate_correct_at_open"].astype(float)
            - paired["correct_at_open_probability_rule"].astype(float)
        ).to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        positions = np.concatenate([grouped_indices[position] for position in selected])
        draws[index] = float(row_delta[positions].mean())
    return {
        "block": block,
        "blocks": verdict.block_count,
        "degenerate_blocks": verdict.degenerate,
        "block_warning": verdict.message,
        "samples": samples,
        "seed": seed,
        "estimate": float(row_delta.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "standard_error": float(draws.std(ddof=1)),
        "probability_positive": float(np.mean(draws > 0.0)),
        "paired_games": len(paired),
    }


def summarize(scored: pd.DataFrame) -> dict[str, Any]:
    paired = scored.dropna(subset=["correct_at_open_probability_rule", "candidate_correct_at_open"])
    primary = paired_policy_bootstrap(scored, block="week")
    secondary = paired_policy_bootstrap(scored, block="season")
    return {
        "population_rows": len(scored),
        "graded_games": len(paired),
        "pushes": int(scored["correct_at_open_probability_rule"].isna().sum()),
        "exactly_one_flagged_games": int(
            (scored["home_incident_flag"] ^ scored["away_incident_flag"]).sum()
        ),
        "both_flagged_games": int(
            (scored["home_incident_flag"] & scored["away_incident_flag"]).sum()
        ),
        "policy_flips": int(scored["policy_flip"].sum()),
        "production_accuracy": float(paired["correct_at_open_probability_rule"].mean()),
        "candidate_accuracy": float(paired["candidate_correct_at_open"].mean()),
        "primary_week_blocked": primary,
        "secondary_season_blocked": secondary,
    }


def season_summary(scored: pd.DataFrame) -> pd.DataFrame:
    paired = scored.dropna(
        subset=["correct_at_open_probability_rule", "candidate_correct_at_open"]
    ).copy()
    rows: list[dict[str, Any]] = []
    for season, group in paired.groupby("season", sort=True):
        rows.append(
            {
                "season": int(season),
                "graded_games": len(group),
                "policy_flips": int(group["policy_flip"].sum()),
                "production_accuracy": float(group["correct_at_open_probability_rule"].mean()),
                "candidate_accuracy": float(group["candidate_correct_at_open"].mean()),
                "delta_accuracy_points": float(
                    100.0
                    * (
                        group["candidate_correct_at_open"]
                        - group["correct_at_open_probability_rule"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _record_weak_signal(metadata: dict[str, Any], artifact_dir: Path) -> None:
    primary = metadata["metrics"]["primary_week_blocked"]
    description = (
        "Post-result direct policy evaluation: flip the frozen production probability-rule "
        "opener pick to the sole team with a broad USA Today incident 1-14 days before the "
        "Tuesday decision date, only when production opposes that team."
    )
    evidence = (
        f"Frozen opener population 2020-2025; {metadata['metrics']['policy_flips']} policy "
        f"flips among {metadata['metrics']['graded_games']} graded games. Week-blocked "
        f"candidate-minus-production estimate={primary['estimate']:+.4f} accuracy points, "
        f"interval=[{primary['lower']:+.4f}, {primary['upper']:+.4f}], "
        f"probability_positive={primary['probability_positive']:.4f}. This is a visibly "
        "post-result direction on reused history; no refuted-mechanism or positive-control "
        "closing ground was established, so it remains unresolved_below_power."
    )
    cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "weak-signals",
        "record",
        "--name",
        SIGNAL_NAME,
        "--description",
        description,
        "--source",
        f"{artifact_dir.as_posix()}/metadata.json; docs/player_arrests_policy_eval.md",
        "--effect",
        f"{primary['estimate']:.10f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2020",
        "--season-end",
        "2025",
        "--standard-error",
        f"{primary['standard_error']:.10f}",
        "--interval-low",
        f"{primary['lower']:.10f}",
        "--interval-high",
        f"{primary['upper']:.10f}",
        "--probability-positive",
        f"{primary['probability_positive']:.10f}",
        "--sample-games",
        str(primary["paired_games"]),
        "--sample-blocks",
        str(primary["blocks"]),
        "--classification-evidence",
        evidence,
        "--notes",
        (
            f"seed={primary['seed']}; samples={primary['samples']}; broad 14-day flag; "
            "season-blocked result is secondary; no split-half reliability applies to a "
            f"per-game policy exposure; artifact={artifact_dir.as_posix()}"
        ),
        "--recorded-at",
        metadata["created_at_utc"],
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"weak-signals record failed for {SIGNAL_NAME}: {result.stdout}\n{result.stderr}"
        )
    print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opener", type=Path, default=DEFAULT_OPENER)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    opener = pd.read_parquet(args.opener)
    identity = pd.read_parquet(
        args.features,
        columns=["game_id", "season", "week", "gameday", "home_team", "away_team"],
    )
    identity = identity.loc[identity["game_id"].isin(opener["game_id"])].copy()
    if len(identity) != len(opener):
        raise PolicyEvaluationError(
            f"identity join covers {len(identity)} of {len(opener)} frozen opener rows"
        )
    incidents = pd.read_parquet(args.incidents, columns=["record_id", "incident_date", "team"])
    flags, incident_coverage = broad_incident_game_flags(identity, incidents)
    scored = apply_frozen_policy(opener, flags)
    metrics = summarize(scored)
    by_season = season_summary(scored)

    output = args.output or (DEFAULT_OUTPUT_ROOT / run_id())
    output.mkdir(parents=True, exist_ok=False)
    atomic_parquet(scored, output / "per_game.parquet")
    atomic_csv(by_season, output / "season_summary.csv")

    configuration = {
        "policy": SIGNAL_NAME,
        "predeclaration": "docs/player_arrests_policy_eval.md",
        "opener": str(args.opener),
        "features": str(args.features),
        "incidents": str(args.incidents),
        "window_days": WINDOW_DAYS,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "primary_block": "week",
        "secondary_block": "season",
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "input_hashes": {
            "opener": sha256_file(args.opener),
            "features": sha256_file(args.features),
            "incidents": sha256_file(args.incidents),
            "predeclaration": sha256_file(REPO / "docs/player_arrests_policy_eval.md"),
        },
        "incident_coverage": incident_coverage,
        "metrics": metrics,
        "season_summary": by_season.to_dict(orient="records"),
        "provenance": artifact_provenance(configuration, args.opener, project_root=REPO),
    }
    experiment_record = write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="player-arrests-policy-eval",
        metrics=metrics,
        schema_version=1,
        notes=(
            "Frozen direct policy evaluation; prediction-level rows retained in per_game.parquet."
        ),
        source="scripts/player_arrests_policy_eval.py; docs/player_arrests_policy_eval.md",
        weak_signal_name=SIGNAL_NAME,
        project_root=REPO,
        registry_root=DEFAULT_REGISTRY_ROOT,
    )
    _record_weak_signal(metadata, output)
    print(
        {
            "artifact_directory": str(output),
            "experiment_id": experiment_record["experiment_id"],
            "weak_signal_name": SIGNAL_NAME,
            "metrics": metrics,
        }
    )


if __name__ == "__main__":
    main()

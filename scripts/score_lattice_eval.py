"""Walk-forward evaluation of the correct-score lattice (MOD-05, WP23).

Predeclaration: ``docs/score_lattice.md`` §1, frozen before this script was
ever run against an outcome.

Two modes, both chronological and both scoring one prediction per lined game:

``--mode screen``
    The real comparison. Candidate = the residual-recentred score lattice
    (:mod:`nfl_ats.score_lattice`); comparator = the shipped kernel-weighted
    mode list (:func:`nfl_ats.tiebreaker.weighted_score_counts`); declared
    secondary arm = the same lattice without recentring, which isolates
    "smoothing" from "recentring".

``--mode positive-control``
    Everything ``screen`` does plus an ``oracle_total`` arm that conditions
    the lattice on the REALISED total. It is a deliberate outcome peek whose
    only job is to prove the instrument can move exact-score hit rate at all,
    so that a null on the real candidate can be classified honestly instead of
    by assumption.

Both arms are centred on the same ``(spread_line, total_line)``: the
walk-forward has no weekly forecast and no totals-model view, so the blend
weights cannot confound the contrast. The only thing that differs between
candidate and comparator is how the identical weighted neighborhood is turned
into an exact-score answer.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from _common import REPO, latest_schedules

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.key_numbers import LINE_BUCKET_LABELS, line_bucket
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.score_lattice import (
    build_lattice,
    feasible_team_scores,
    mode_list_probability,
    ranked_modes,
)
from nfl_ats.tiebreaker import (
    _neighborhood,
    lined_finals,
    weighted_median,
    weighted_score_counts,
)

#: Chronological sort key. Playoff rounds carry week numbers above the regular
#: season's, so ``(season, week)`` orders the whole history correctly and a
#: strict prefix of the sorted frame is exactly "everything before this week".
_ORDER = ["season", "week", "gameday", "game_id"]


def _hit(top: tuple[tuple[int, int, float], ...], home: int, away: int, depth: int) -> int:
    return int(any((h, a) == (home, away) for h, a, _ in top[:depth]))


def _log_loss(probability: float) -> float:
    return -math.log(probability)


def walk_forward(
    finals: pd.DataFrame,
    season_start: int,
    season_end: int,
    *,
    with_oracle: bool,
) -> pd.DataFrame:
    """One row per scored game, with every arm's answer on that game."""

    ordered = finals.sort_values(_ORDER, kind="stable").reset_index(drop=True)
    week_keys = list(
        ordered.loc[ordered["season"].between(season_start, season_end), ["season", "week"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    season_column = ordered["season"].to_numpy()
    week_column = ordered["week"].to_numpy()

    rows: list[dict[str, Any]] = []
    for season, week in week_keys:
        before = (season_column < season) | ((season_column == season) & (week_column < week))
        training = ordered.loc[before]
        if training.empty:
            continue
        support = feasible_team_scores(training)
        support_set = {int(value) for value in support}
        targets = ordered.loc[(season_column == season) & (week_column == week)]
        for game in targets.itertuples(index=False):
            spread = float(game.spread_line)
            total_line = float(game.total_line)
            home = int(game.home_score)
            away = int(game.away_score)
            actual_total = home + away
            actual_margin = home - away

            neighborhood = _neighborhood(training, spread, total_line)
            frame, weights = neighborhood.frame, neighborhood.weights

            counts = weighted_score_counts(frame, weights)
            modes = ranked_modes(counts, 3)
            modes_median_total = weighted_median(
                (frame["home_score"] + frame["away_score"]).to_numpy(dtype=float), weights
            )

            lattice = build_lattice(
                frame,
                weights,
                spread,
                total_line,
                support,
                recentre=True,
                effective_size=neighborhood.effective_size,
                label=neighborhood.label,
            )
            raw = build_lattice(
                frame,
                weights,
                spread,
                total_line,
                support,
                recentre=False,
                effective_size=neighborhood.effective_size,
                label=neighborhood.label,
            )
            lattice_top = lattice.top_scores(3)
            raw_top = raw.top_scores(3)

            row: dict[str, Any] = {
                "game_id": game.game_id,
                "season": int(season),
                "week": int(week),
                "gameday": str(game.gameday),
                "game_type": str(game.game_type),
                "home_team": str(game.home_team),
                "away_team": str(game.away_team),
                "spread_line": spread,
                "total_line": total_line,
                "home_score": home,
                "away_score": away,
                "actual_total": actual_total,
                "actual_margin": actual_margin,
                "training_games": len(training),
                "neighborhood_ess": float(neighborhood.effective_size),
                "neighborhood_label": neighborhood.label,
                "neighborhood_all_history": int(neighborhood.label == "all history"),
                "support_scores": int(support.size),
                "realised_in_support": int(home in support_set and away in support_set),
                "comparator_nonzero": int(counts.get((home, away), 0.0) > 0.0),
                "modes_top1_home": modes[0][0],
                "modes_top1_away": modes[0][1],
                "modes_top1_hit": _hit(modes, home, away, 1),
                "modes_top3_hit": _hit(modes, home, away, 3),
                "modes_log_loss": _log_loss(mode_list_probability(counts, support, home, away)),
                "modes_total_guess": float(round(modes_median_total)),
                "lattice_top1_home": lattice_top[0][0],
                "lattice_top1_away": lattice_top[0][1],
                "lattice_top1_probability": lattice_top[0][2],
                "lattice_top1_hit": _hit(lattice_top, home, away, 1),
                "lattice_top3_hit": _hit(lattice_top, home, away, 3),
                "lattice_log_loss": _log_loss(lattice.smoothed_probability(home, away)),
                "lattice_modal_total": float(lattice.modal_total()),
                "lattice_median_total": float(round(lattice.median_total())),
                "lattice_push_probability": lattice.push_probability(spread),
                "realised_push": int(actual_margin == spread),
                "raw_top1_hit": _hit(raw_top, home, away, 1),
                "raw_top3_hit": _hit(raw_top, home, away, 3),
                "raw_log_loss": _log_loss(raw.smoothed_probability(home, away)),
                "raw_push_probability": raw.push_probability(spread),
            }
            if with_oracle:
                oracle = lattice.condition_on_total(actual_total)
                oracle_top = oracle.top_scores(3)
                row["oracle_top1_hit"] = _hit(oracle_top, home, away, 1)
                row["oracle_top3_hit"] = _hit(oracle_top, home, away, 3)
                row["oracle_log_loss"] = _log_loss(oracle.smoothed_probability(home, away))
            rows.append(row)
    scored = pd.DataFrame(rows)
    scored["line_bucket"] = line_bucket(scored["spread_line"])
    return scored


def _paired_metrics(frame: pd.DataFrame, *, with_oracle: bool) -> dict[str, float]:
    """Every predeclared paired delta, candidate minus comparator.

    Hit-rate deltas are in accuracy points (percentage points); log-loss and
    closest-total deltas are improvements (comparator minus candidate), so
    positive always favours the candidate.
    """

    def points(column: str) -> float:
        return 100.0 * float(frame[column].mean())

    alive = frame.loc[frame["comparator_nonzero"] == 1]
    metrics = {
        "top1_exact_points": points("lattice_top1_hit") - points("modes_top1_hit"),
        "top3_exact_points": points("lattice_top3_hit") - points("modes_top3_hit"),
        "log_loss_improvement": float(
            frame["modes_log_loss"].mean() - frame["lattice_log_loss"].mean()
        ),
        "log_loss_improvement_where_comparator_alive": float(
            alive["modes_log_loss"].mean() - alive["lattice_log_loss"].mean()
        )
        if len(alive)
        else 0.0,
        "closest_total_mae_improvement_median": float(
            (frame["modes_total_guess"] - frame["actual_total"]).abs().mean()
            - (frame["lattice_median_total"] - frame["actual_total"]).abs().mean()
        ),
        "closest_total_mae_improvement_modal": float(
            (frame["modes_total_guess"] - frame["actual_total"]).abs().mean()
            - (frame["lattice_modal_total"] - frame["actual_total"]).abs().mean()
        ),
        "raw_arm_top1_exact_points": points("raw_top1_hit") - points("modes_top1_hit"),
        "raw_arm_top3_exact_points": points("raw_top3_hit") - points("modes_top3_hit"),
        "raw_arm_log_loss_improvement": float(
            frame["modes_log_loss"].mean() - frame["raw_log_loss"].mean()
        ),
    }
    if with_oracle:
        metrics["oracle_top1_exact_points"] = points("oracle_top1_hit") - points("modes_top1_hit")
        metrics["oracle_top3_exact_points"] = points("oracle_top3_hit") - points("modes_top3_hit")
        metrics["oracle_log_loss_improvement"] = float(
            frame["modes_log_loss"].mean() - frame["oracle_log_loss"].mean()
        )
    return metrics


def _season_table(frame: pd.DataFrame, *, with_oracle: bool) -> list[dict[str, Any]]:
    columns = ["modes_top1_hit", "lattice_top1_hit", "raw_top1_hit"]
    columns += ["modes_top3_hit", "lattice_top3_hit", "raw_top3_hit"]
    if with_oracle:
        columns += ["oracle_top1_hit", "oracle_top3_hit"]
    rows: list[dict[str, Any]] = []
    for season, group in frame.groupby("season", sort=True):
        row: dict[str, Any] = {"season": int(season), "games": len(group)}
        for column in columns:
            row[column.replace("_hit", "_rate_pct")] = 100.0 * float(group[column].mean())
        row["all_history_fallback_pct"] = 100.0 * float(group["neighborhood_all_history"].mean())
        row["log_loss_improvement"] = float(
            group["modes_log_loss"].mean() - group["lattice_log_loss"].mean()
        )
        rows.append(row)
    return rows


def _push_calibration(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Predeclared metric (d), descriptive: mean predicted ``P(push)`` against
    the realised push rate per line bucket.

    Both lattice arms are shown. The declared secondary (un-recentred) arm is
    the one that preserves MOD-05's raw key-number spikes, so a gap between
    the two columns at ``|line| = 3`` and ``|line| = 7`` is the smearing cost
    of recentring, priced in push probability instead of in exact scores.
    """

    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("line_bucket", sort=False):
        realised = float(group["realised_push"].mean())
        lattice = float(group["lattice_push_probability"].mean())
        raw = float(group["raw_push_probability"].mean())
        rows.append(
            {
                "line_bucket": str(bucket),
                "label": LINE_BUCKET_LABELS.get(str(bucket), str(bucket)),
                "games": len(group),
                "mean_predicted_push_probability": lattice,
                "raw_arm_mean_predicted_push_probability": raw,
                "realised_push_rate": realised,
                "calibration_gap": lattice - realised,
                "raw_arm_calibration_gap": raw - realised,
            }
        )
    return sorted(rows, key=lambda row: -row["games"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("positive-control", "screen"), required=True)
    parser.add_argument("--season-start", type=int, default=2012)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--warmup-start", type=int, default=2009)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out-root", type=Path, default=REPO / "artifacts" / "score_lattice")
    args = parser.parse_args()

    schedules_path = latest_schedules()
    schedules = pd.read_parquet(schedules_path)
    finals = lined_finals(schedules)
    finals = finals.loc[finals["season"] >= args.warmup_start]
    with_oracle = args.mode == "positive-control"

    scored = walk_forward(finals, args.season_start, args.season_end, with_oracle=with_oracle)
    bootstrap = week_blocked_bootstrap(
        scored,
        lambda frame: _paired_metrics(frame, with_oracle=with_oracle),
        block="week",
        samples=args.bootstrap_samples,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "predictions.csv", index=False)

    headline = {
        "games": len(scored),
        "week_blocks": int(scored.groupby(["season", "week"]).ngroups),
        "modes_top1_rate_pct": 100.0 * float(scored["modes_top1_hit"].mean()),
        "lattice_top1_rate_pct": 100.0 * float(scored["lattice_top1_hit"].mean()),
        "raw_top1_rate_pct": 100.0 * float(scored["raw_top1_hit"].mean()),
        "modes_top3_rate_pct": 100.0 * float(scored["modes_top3_hit"].mean()),
        "lattice_top3_rate_pct": 100.0 * float(scored["lattice_top3_hit"].mean()),
        "raw_top3_rate_pct": 100.0 * float(scored["raw_top3_hit"].mean()),
        "modes_log_loss": float(scored["modes_log_loss"].mean()),
        "lattice_log_loss": float(scored["lattice_log_loss"].mean()),
        "raw_log_loss": float(scored["raw_log_loss"].mean()),
        "comparator_alive_pct": 100.0 * float(scored["comparator_nonzero"].mean()),
        "realised_in_support_pct": 100.0 * float(scored["realised_in_support"].mean()),
        "all_history_fallback_pct": 100.0 * float(scored["neighborhood_all_history"].mean()),
        "modes_total_mae": float(
            (scored["modes_total_guess"] - scored["actual_total"]).abs().mean()
        ),
        "lattice_median_total_mae": float(
            (scored["lattice_median_total"] - scored["actual_total"]).abs().mean()
        ),
        "lattice_modal_total_mae": float(
            (scored["lattice_modal_total"] - scored["actual_total"]).abs().mean()
        ),
    }
    if with_oracle:
        headline["oracle_top1_rate_pct"] = 100.0 * float(scored["oracle_top1_hit"].mean())
        headline["oracle_top3_rate_pct"] = 100.0 * float(scored["oracle_top3_hit"].mean())
        headline["oracle_log_loss"] = float(scored["oracle_log_loss"].mean())

    configuration = {
        "predeclaration": "docs/score_lattice.md §1",
        "mode": args.mode,
        "warmup_seasons": [int(args.warmup_start), int(args.season_start) - 1],
        "evaluation_seasons": [int(args.season_start), int(args.season_end)],
        "bootstrap_samples": int(args.bootstrap_samples),
        "block": "week",
        "comparator": "nfl_ats.tiebreaker weighted_score_counts mode list",
        "candidate": "nfl_ats.score_lattice residual-recentred lattice",
    }
    summary = {
        "mode": args.mode,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "headline": headline,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "by_season": _season_table(scored, with_oracle=with_oracle),
        "push_calibration": _push_calibration(scored),
        "manifest": {**configuration, "schedules": str(schedules_path), "generated_at": stamp},
        # ``artifact_provenance`` hashes the input table it is handed; the
        # schedules snapshot IS this evaluation's only input table, so it is
        # what gets pinned (there is no engineered feature parquet here).
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        out,
        "summary.json",
        summary,
        command="score-lattice-eval",
        metrics={
            "mode": args.mode,
            "games": headline["games"],
            "top1_exact_points": float(
                bootstrap.loc[bootstrap["metric"] == "top1_exact_points", "estimate"].iloc[0]
            ),
            "top1_probability_positive": float(
                bootstrap.loc[
                    bootstrap["metric"] == "top1_exact_points", "probability_positive"
                ].iloc[0]
            ),
        },
        notes=(
            "MOD-05 correct-score lattice vs the shipped tiebreaker mode list; "
            "predeclared in docs/score_lattice.md §1 before any outcome number."
        ),
        project_root=REPO,
    )

    print(f"artifact: {out}")
    print(json.dumps(headline, indent=2))
    print(bootstrap.to_string(index=False))
    print(pd.DataFrame(summary["by_season"]).to_string(index=False))
    print(pd.DataFrame(summary["push_calibration"]).to_string(index=False))


if __name__ == "__main__":
    main()

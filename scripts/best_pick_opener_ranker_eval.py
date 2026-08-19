"""MEASURE-ONLY: does a ranking signal choose a better weekly Best Pick than
the status-quo chooser, on the paired opener population the ridge_alpha
promotion evaluation already produced?

Predeclared in ``scratchpad/bestpick_opener/predeclaration.md`` (session
scratchpad, not this repo) BEFORE any accuracy or lift number below was
computed. This script reads only the two stored arms from
``artifacts/ridge_alpha_promotion/20260818T221459Z/`` (baseline ridge
alpha=10, candidate alpha=2000, both ``weak_stack``/``market_residual``, both
already scored at the opener on the same 1,537-game/107-week paired archive
``docs/opener_evaluation.md`` cites). No model is refit, no registry file is
written, no docs are edited.

Five predeclared choosers nominate one game per week; each is scored against
that SAME week's own average pick accuracy (the "Best-Pick lift"), never
against a global pooled mean -- see the predeclaration for why that is a
deliberate departure from ``scripts/best_pick_ranker.py``'s pooled top1-minus
-all metric. A sixth, explicitly-labeled DIAGNOSTIC (not one of the five)
gives chooser 4 a same-arm blind floor, because the alphabetical chooser (2)
is graded on the baseline arm and chooser 4 runs on the candidate arm.

Addendum (2026-08-18 evening, predeclaration addendum in the same scratchpad
file): three more choosers (6-8) test whether cross-book opener spread
dispersion (``spread_std``, read from
``artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet``
-- high dispersion predicts a bigger eventual market miss) helps choosers 4/1
avoid noisier games, either as a below-median eligibility FILTER (6, 7) or as
a tie-break (8). The filter never changes the lift's baseline (always the
full week's own average, unfiltered) -- only which games are eligible to be
nominated -- so a filtered chooser's lift is directly comparable to its
unfiltered parent's on the same yardstick. A week falls back to its full,
unfiltered game set if any game that week is missing ``spread_std``, or if a
strict below-median filter would leave zero eligible games (ties at the
week's minimum, common since 302/1,537 games league-wide have
``spread_std == 0.0``); both triggers are counted separately.

MULTIPLICITY: this is exploratory scoring on the same 107 opener weeks the
ridge_alpha promotion look AND the odds-microstructure battery already used
-- every number below carries that look-reuse discount. BINDING: an interval
containing zero is never treated as a rejection; ``probability_positive`` is
reported as continuous evidence throughout, never collapsed to
significant/not-significant. The relevant EV bar for a PLAY decision is
chooser 1 (status quo), which is the currently signal-free incumbent, not
zero -- per AGENTS.md, a promotion bar gates claims/registry entries, not
plays.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/best_pick_opener_ranker_eval.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "artifacts" / "ridge_alpha_promotion" / "20260818T221459Z"
DEFAULT_MICROSTRUCTURE_SOURCE = REPO / "artifacts" / "odds_microstructure" / "20260818T225430Z"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818


class ChooserSpec(NamedTuple):
    name: str
    primary: str | None
    secondary: str | None
    correct_col: str
    alphabetical_only: bool
    pool_column: str | None = None


PRIMARY_CHOOSERS: tuple[ChooserSpec, ...] = (
    ChooserSpec("status_quo_residual", "abs_residual", None, "baseline_correct_open", False),
    ChooserSpec("alphabetical_floor", None, None, "baseline_correct_open", True),
    ChooserSpec("baseline_prob_distance", "baseline_dist", None, "baseline_correct_open", False),
    ChooserSpec("candidate_prob_distance", "candidate_dist", None, "candidate_correct_open", False),
    ChooserSpec(
        "status_quo_tiebreak",
        "abs_residual",
        "candidate_dist",
        "baseline_correct_open",
        False,
    ),
)
# Diagnostic only -- not one of the five predeclared choosers. Exists solely so
# chooser 4 (candidate arm) has a same-arm blind floor, since chooser 2 is
# graded on the baseline arm.
DIAGNOSTIC_CHOOSER = ChooserSpec(
    "alphabetical_floor_candidate_arm", None, None, "candidate_correct_open", True
)

# Addendum choosers 6-8 (dispersion-filtered / tie-broken). 6 and 7 share the
# same eligibility pool ("dispersion_pool_pass", built by
# ``build_dispersion_pool`` -- a property of the game/market, not of either
# model arm); 8 is a tie-break enhancement over the full week, no filter.
DISPERSION_CHOOSERS: tuple[ChooserSpec, ...] = (
    ChooserSpec(
        "dispersion_filtered_candidate",
        "candidate_dist",
        None,
        "candidate_correct_open",
        False,
        pool_column="dispersion_pool_pass",
    ),
    ChooserSpec(
        "dispersion_filtered_status_quo",
        "abs_residual",
        None,
        "baseline_correct_open",
        False,
        pool_column="dispersion_pool_pass",
    ),
    ChooserSpec(
        "dispersion_tiebreak",
        "candidate_dist",
        "neg_spread_std",
        "candidate_correct_open",
        False,
    ),
)


# ---------------------------------------------------------------------------
# 1. Load + build the working frame
# ---------------------------------------------------------------------------


def load_working_frame(source: Path) -> pd.DataFrame:
    paired = pd.read_parquet(source / "opener_paired.parquet")
    baseline = pd.read_parquet(source / "opener_baseline.parquet")

    required_paired = {
        "game_id",
        "season",
        "week",
        "baseline_correct_open",
        "baseline_prob_open",
        "candidate_correct_open",
        "candidate_prob_open",
    }
    missing_paired = sorted(required_paired.difference(paired.columns))
    if missing_paired:
        raise ValueError(f"opener_paired.parquet is missing columns: {', '.join(missing_paired)}")
    if "residual_at_open" not in baseline.columns:
        raise ValueError("opener_baseline.parquet is missing column: residual_at_open")

    work = paired.merge(
        baseline[["game_id", "residual_at_open"]], on="game_id", how="inner", validate="one_to_one"
    )
    if len(work) != len(paired):
        raise ValueError(
            f"residual merge dropped rows: paired={len(paired)} baseline-merged={len(work)}"
        )

    # Structural check from the predeclaration: pushes (correct_at_open NaN)
    # must be the SAME games in both arms, because margin_vs_open does not
    # depend on the model. Fail loudly if that ever stops being true.
    baseline_push = set(work.loc[work["baseline_correct_open"].isna(), "game_id"])
    candidate_push = set(work.loc[work["candidate_correct_open"].isna(), "game_id"])
    if baseline_push != candidate_push:
        only_baseline = baseline_push - candidate_push
        only_candidate = candidate_push - baseline_push
        raise ValueError(
            "push-game sets disagree between arms (predeclaration assumed they cannot): "
            f"baseline-only={sorted(only_baseline)} candidate-only={sorted(only_candidate)}"
        )

    work["abs_residual"] = work["residual_at_open"].abs()
    work["baseline_dist"] = (work["baseline_prob_open"] - 0.5).abs()
    work["candidate_dist"] = (work["candidate_prob_open"] - 0.5).abs()
    return work


def build_dispersion_pool(
    work: pd.DataFrame, microstructure_source: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge Tuesday-opener cross-book ``spread_std`` and build the
    below-median-dispersion eligibility pool (choosers 6/7) plus
    ``neg_spread_std`` for the ascending tie-break (chooser 8).

    Fallback rule (predeclared): a week falls back to its FULL, unfiltered
    game set -- every game passes the pool -- if any game that week is
    missing ``spread_std``, or if a strict ``spread_std < week_median`` filter
    would leave zero eligible games. Both triggers are counted separately and
    returned in the summary dict for reporting.
    """

    spread = pd.read_parquet(microstructure_source / "spread_novig_tue_open.parquet")
    required = {"game_id", "spread_std"}
    missing_cols = sorted(required.difference(spread.columns))
    if missing_cols:
        raise ValueError(
            f"spread_novig_tue_open.parquet is missing columns: {', '.join(missing_cols)}"
        )

    working_games = set(work["game_id"])
    microstructure_games = set(spread["game_id"])
    overlap_report = {
        "working_frame_games": len(working_games),
        "microstructure_games": len(microstructure_games),
        "overlap_games": len(working_games & microstructure_games),
        "working_frame_only": sorted(working_games - microstructure_games),
        "microstructure_only": sorted(microstructure_games - working_games),
    }

    work = work.merge(
        spread[["game_id", "spread_std"]], on="game_id", how="left", validate="one_to_one"
    )
    work["neg_spread_std"] = -work["spread_std"]

    pool_pass = pd.Series(False, index=work.index)
    fallback_rows: list[dict[str, Any]] = []
    for (season, week), group in work.groupby(["season", "week"], sort=True):
        idx = group.index
        n_missing = int(group["spread_std"].isna().sum())
        if n_missing > 0:
            reason = "missing_data"
            pool_pass.loc[idx] = True
        else:
            median = group["spread_std"].median()
            below = group["spread_std"] < median
            if not below.any():
                reason = "empty_filter"
                pool_pass.loc[idx] = True
            else:
                reason = "none"
                pool_pass.loc[idx] = below
        fallback_rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "n_games": len(group),
                "n_missing": n_missing,
                "fallback": reason != "none",
                "fallback_reason": reason,
                "n_pool_pass": int(pool_pass.loc[idx].sum()),
            }
        )
    work["dispersion_pool_pass"] = pool_pass
    fallback_frame = (
        pd.DataFrame(fallback_rows).sort_values(["season", "week"]).reset_index(drop=True)
    )

    summary = {
        "overlap": overlap_report,
        "games_missing_spread_std": int(work["spread_std"].isna().sum()),
        "weeks_total": len(fallback_frame),
        "weeks_fallback_total": int(fallback_frame["fallback"].sum()),
        "weeks_fallback_missing_data": int(
            (fallback_frame["fallback_reason"] == "missing_data").sum()
        ),
        "weeks_fallback_empty_filter": int(
            (fallback_frame["fallback_reason"] == "empty_filter").sum()
        ),
        "fallback_weeks_detail": fallback_frame.loc[fallback_frame["fallback"]].to_dict(
            orient="records"
        ),
    }
    return work, summary


# ---------------------------------------------------------------------------
# 2. Per-week nomination
# ---------------------------------------------------------------------------


def _week_average(group: pd.DataFrame, correct_col: str) -> float:
    valid = group[correct_col].dropna()
    return float(valid.mean()) if len(valid) else float("nan")


def nominate(
    work: pd.DataFrame,
    *,
    primary: str | None,
    secondary: str | None,
    correct_col: str,
    alphabetical_only: bool,
    pool_column: str | None = None,
) -> pd.DataFrame:
    """One row per (season, week): the chooser's nominee, its correctness, the
    week's own average correctness (over that arm, pushes excluded), the lift,
    and how many candidates tied for the top primary-signal value.

    Tie-break, matching ``best_pick.py::select_best_pick``: sort by
    ``(primary desc, secondary desc, game_id asc)`` and take the first row.
    ``alphabetical_only`` ignores both signal columns and sorts purely by
    ascending ``game_id`` -- the tie-luck floor.

    ``pool_column``, when given, restricts which games are ELIGIBLE to be
    nominated/ranked (``group.loc[group[pool_column]]``) -- e.g. the
    dispersion below-median-eligibility flag. The week-average baseline
    (``week_avg``) is always computed over the FULL, unfiltered group so a
    filtered chooser's lift stays on the same yardstick as its unfiltered
    parent. The fallback logic that builds ``pool_column`` already guarantees
    every game passes on a fallback week, so the filtered group is never
    empty here.
    """

    rows: list[dict[str, Any]] = []
    for (season, week), group in work.groupby(["season", "week"], sort=True):
        candidates = group.loc[group[pool_column]] if pool_column is not None else group
        if candidates.empty:
            raise ValueError(
                f"pool_column={pool_column!r} left zero candidates for ({season}, {week}); "
                "the fallback rule was supposed to make this impossible"
            )
        if alphabetical_only:
            ordered = candidates.sort_values("game_id", ascending=True)
            n_tied_at_max = 1  # no ranking signal, no ties to speak of
        else:
            sort_cols = [primary]
            ascending = [False]
            if secondary is not None:
                sort_cols.append(secondary)
                ascending.append(False)
            sort_cols.append("game_id")
            ascending.append(True)
            ordered = candidates.sort_values(sort_cols, ascending=ascending)
            top_value = candidates[primary].max()
            n_tied_at_max = int((candidates[primary] == top_value).sum())
        nominee = ordered.iloc[0]
        # Baseline always uses the FULL week, not the filtered candidate set --
        # see the docstring: this keeps filtered choosers on the same scale as
        # their unfiltered parents.
        week_avg = _week_average(group, correct_col)
        nominee_correct = nominee[correct_col]
        nominee_correct = float(nominee_correct) if pd.notna(nominee_correct) else float("nan")
        lift = (
            nominee_correct - week_avg
            if pd.notna(nominee_correct) and pd.notna(week_avg)
            else float("nan")
        )
        # Tie-agnostic recompute: average correctness over ALL candidates tied
        # at the max primary signal (only meaningful when n_tied_at_max > 1;
        # otherwise identical to the recorded nominee).
        if not alphabetical_only and n_tied_at_max > 1:
            tied = candidates.loc[candidates[primary] == top_value]
            tied_correct = tied[correct_col].dropna()
            tie_agnostic_correct = float(tied_correct.mean()) if len(tied_correct) else float("nan")
        else:
            tie_agnostic_correct = nominee_correct
        tie_agnostic_lift = (
            tie_agnostic_correct - week_avg
            if pd.notna(tie_agnostic_correct) and pd.notna(week_avg)
            else float("nan")
        )
        rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "n_candidates": len(candidates),
                "n_tied_at_max": n_tied_at_max,
                "nominee_game_id": str(nominee["game_id"]),
                "nominee_correct": nominee_correct,
                "week_avg_correct": week_avg,
                "lift": lift,
                "tie_agnostic_correct": tie_agnostic_correct,
                "tie_agnostic_lift": tie_agnostic_lift,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_series(
    frame: pd.DataFrame, value_col: str, *, samples: int, seed: int
) -> dict[str, float]:
    """Week-blocked bootstrap of the mean of ``value_col`` over a one
    -row-per-week frame. Reduces to resampling weeks with replacement because
    each row already IS one week (``clv.week_blocked_bootstrap``'s block="week"
    grouping is then trivially one row per group).
    """

    valid = frame.dropna(subset=[value_col])
    if valid.empty:
        return {
            "n_weeks": 0,
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "probability_positive": float("nan"),
        }
    result = week_blocked_bootstrap(
        valid,
        lambda f: {"mean": float(f[value_col].mean())},
        block="week",
        samples=samples,
        seed=seed,
    )
    row = result.iloc[0]
    return {
        "n_weeks": len(valid),
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def _paired_delta(
    a: pd.DataFrame, b: pd.DataFrame, value_col: str, *, samples: int, seed: int
) -> dict[str, Any]:
    """Bootstrap of the paired per-week difference ``a[value_col] -
    b[value_col]``, inner-joined on (season, week) so only weeks where BOTH
    choosers have a valid (non-push) nominee contribute.
    """

    merged = a[["season", "week", value_col]].merge(
        b[["season", "week", value_col]], on=["season", "week"], suffixes=("_a", "_b")
    )
    merged["delta"] = merged[f"{value_col}_a"] - merged[f"{value_col}_b"]
    stats = _bootstrap_series(merged, "delta", samples=samples, seed=seed)
    stats["n_weeks_paired"] = stats.pop("n_weeks")
    return stats


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------


def evaluate_chooser(
    spec: ChooserSpec, work: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    weekly = nominate(
        work,
        primary=spec.primary,
        secondary=spec.secondary,
        correct_col=spec.correct_col,
        alphabetical_only=spec.alphabetical_only,
        pool_column=spec.pool_column,
    )
    lift_stats = _bootstrap_series(weekly, "lift", samples=samples, seed=seed)
    tie_agnostic_stats = _bootstrap_series(weekly, "tie_agnostic_lift", samples=samples, seed=seed)
    valid_nominee = weekly["nominee_correct"].dropna()
    n_tie_weeks = int((weekly["n_tied_at_max"] > 1).sum())
    return {
        "chooser": spec.name,
        "arm_correct_column": spec.correct_col,
        "pool_column": spec.pool_column,
        "weeks_total": len(weekly),
        "weeks_scored": lift_stats["n_weeks"],
        "top1_accuracy": float(valid_nominee.mean()) if len(valid_nominee) else float("nan"),
        "n_tie_weeks": n_tie_weeks,
        "tie_break_rule": (
            "none (pure ascending game_id)"
            if spec.alphabetical_only
            else "ascending game_id after primary"
            + (" then secondary signal" if spec.secondary else "")
        ),
        "mean_weekly_lift": lift_stats,
        "tie_agnostic_mean_weekly_lift": tie_agnostic_stats,
        "weekly_frame": weekly,
    }


def run(
    work: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    dispersion_pool_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    all_specs = list(PRIMARY_CHOOSERS)
    if dispersion_pool_summary is not None:
        all_specs += list(DISPERSION_CHOOSERS)
    for spec in all_specs:
        results[spec.name] = evaluate_chooser(spec, work, samples=samples, seed=seed)
    diagnostic_name = DIAGNOSTIC_CHOOSER.name
    results[diagnostic_name] = evaluate_chooser(
        DIAGNOSTIC_CHOOSER, work, samples=samples, seed=seed
    )

    floor = results["alphabetical_floor"]["weekly_frame"]
    status_quo = results["status_quo_residual"]["weekly_frame"]
    candidate_floor = results[diagnostic_name]["weekly_frame"]

    comparisons: dict[str, Any] = {}
    for spec in PRIMARY_CHOOSERS:
        name = spec.name
        if name == "alphabetical_floor":
            continue
        weekly = results[name]["weekly_frame"]
        comparisons[f"{name}_vs_alphabetical_floor"] = _paired_delta(
            weekly, floor, "lift", samples=samples, seed=seed
        )
        if name != "status_quo_residual":
            comparisons[f"{name}_vs_status_quo_residual"] = _paired_delta(
                weekly, status_quo, "lift", samples=samples, seed=seed
            )
    # Same-arm diagnostic comparison for chooser 4.
    comparisons["candidate_prob_distance_vs_alphabetical_floor_candidate_arm"] = _paired_delta(
        results["candidate_prob_distance"]["weekly_frame"],
        candidate_floor,
        "lift",
        samples=samples,
        seed=seed,
    )

    if dispersion_pool_summary is not None:
        candidate_weekly = results["candidate_prob_distance"]["weekly_frame"]
        status_quo_weekly = results["status_quo_residual"]["weekly_frame"]
        filtered_candidate = results["dispersion_filtered_candidate"]["weekly_frame"]
        filtered_status_quo = results["dispersion_filtered_status_quo"]["weekly_frame"]
        tiebreak = results["dispersion_tiebreak"]["weekly_frame"]

        # The incremental question: does the filter/tie-break ADD anything on
        # top of the SAME unfiltered signal, on the SAME yardstick.
        comparisons["dispersion_filtered_candidate_vs_candidate_prob_distance"] = _paired_delta(
            filtered_candidate, candidate_weekly, "lift", samples=samples, seed=seed
        )
        comparisons["dispersion_filtered_status_quo_vs_status_quo_residual"] = _paired_delta(
            filtered_status_quo, status_quo_weekly, "lift", samples=samples, seed=seed
        )
        comparisons["dispersion_tiebreak_vs_candidate_prob_distance"] = _paired_delta(
            tiebreak, candidate_weekly, "lift", samples=samples, seed=seed
        )
        # Same primary/secondary comparisons the original five choosers got.
        comparisons["dispersion_filtered_candidate_vs_alphabetical_floor_candidate_arm"] = (
            _paired_delta(filtered_candidate, candidate_floor, "lift", samples=samples, seed=seed)
        )
        comparisons["dispersion_filtered_status_quo_vs_alphabetical_floor"] = _paired_delta(
            filtered_status_quo, floor, "lift", samples=samples, seed=seed
        )
        comparisons["dispersion_tiebreak_vs_alphabetical_floor_candidate_arm"] = _paired_delta(
            tiebreak, candidate_floor, "lift", samples=samples, seed=seed
        )
        comparisons["dispersion_filtered_candidate_vs_status_quo_residual"] = _paired_delta(
            filtered_candidate, status_quo_weekly, "lift", samples=samples, seed=seed
        )
        comparisons["dispersion_tiebreak_vs_status_quo_residual"] = _paired_delta(
            tiebreak, status_quo_weekly, "lift", samples=samples, seed=seed
        )

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "population": {
            "games": len(work),
            "weeks": int(work.groupby(["season", "week"]).ngroups),
            "seasons": sorted(int(s) for s in work["season"].unique()),
        },
        "dispersion_pool": dispersion_pool_summary,
        "choosers": {
            name: {k: v for k, v in payload.items() if k != "weekly_frame"}
            for name, payload in results.items()
        },
        "comparisons": comparisons,
        "multiplicity_note": (
            "Exploratory scoring on the same 107 opener weeks the ridge_alpha "
            "promotion evaluation and the odds-microstructure battery already "
            "used; every number here carries that look-reuse discount. Not a "
            "fresh rotation window; no registry write."
        ),
        "binding_note": (
            "An interval containing zero is never treated as a rejection here; "
            "probability_positive is the continuous evidence to read, not a "
            "significant/not-significant collapse."
        ),
    }
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--microstructure-source", type=Path, default=DEFAULT_MICROSTRUCTURE_SOURCE)
    parser.add_argument(
        "--skip-dispersion",
        action="store_true",
        help="run only the original five choosers, skipping choosers 6-8",
    )
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    work = load_working_frame(args.source)
    dispersion_pool_summary = None
    if not args.skip_dispersion:
        work, dispersion_pool_summary = build_dispersion_pool(work, args.microstructure_source)
    summary, results = run(
        work, samples=args.samples, seed=args.seed, dispersion_pool_summary=dispersion_pool_summary
    )
    print(json.dumps(summary, indent=2, default=str))

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "artifacts" / "best_pick_opener_ranker" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    for name, payload in results.items():
        payload["weekly_frame"].to_parquet(out_dir / f"{name}.weekly.parquet")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()

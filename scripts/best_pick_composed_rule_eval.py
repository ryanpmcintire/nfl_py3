"""SCORE-ONLY: the production Best Pick nomination rule (v2), AS PLAYED, on
the paired opener archive -- predeclared in ``docs/best_pick_composed_rule.md``
(family ``pol09_best_pick_composed_v1``) before any number below was computed.

What is new versus the 2026-08-19 v3 audit
(``scripts/best_pick_nomination_v3_audit.py``): that audit reproduced the
composed rule but scored its nominee on the CANDIDATE arm's pick side
(``candidate_correct_open``, the alpha=2000 model's own pick). Production
never changes sides -- the nominee's Best Pick settles on the ACTIVE model's
pick (``baseline_correct_open``) -- and the two arms disagree on the side in
255 of 1,537 archive games, so "as played" is a different number. This
script scores the rule the way the pool grades it, against the two rules it
replaced (v1 ``sweep_robustness`` and the raw top-|residual| pick), and
scores the side-ledger big-spread eligibility discount the same way.

Reproduction, not re-implementation: every nominee comes from the
production modules (``nfl_ats.best_pick_nomination.select_nominee`` /
``select_nominee_v3`` / ``dispersion_pool_from_frame``,
``nfl_ats.best_pick_big_spread_challenger.apply_big_spread_eligibility``,
``nfl_ats.best_pick.select_best_pick``). The archive frame, the predeclared
eval script's own dispersion pool (cross-checked row-for-row against the
production pool), chooser 1, and the bootstrap helpers are reused from
``scripts/best_pick_opener_ranker_eval.py`` by file-path import, unmodified.
v1 needs a line sweep the archive does not carry, so the active recipe is
refit walk-forward per archive week (the archive's own loop, mirrored) and
swept at the archived opener; a reproduction check reports the max absolute
residual difference against the archive.

BINDING (owner mandates, restated because this script's output feeds a
registry write): an interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment -- "contains zero" is the EXPECTED
outcome for a real small signal at this evaluator's ~2-point resolution.
Only two closing grounds exist: (1) refuted mechanism -- a RESOLVED wrong
sign (the WHOLE interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is ``unresolved_below_power``, reported
with ``probability_positive``, never collapsed to "contains zero". Decide on
EV: the pool is forced picks, a nomination happens every week regardless, so
``probability_positive`` above 0.5 favours the first-named rule; a promotion
bar governs what may be CLAIMED, never what gets PLAYED. Grade at the
OPENER. Within-week correlation is zero (owner-mandated, hardcoded).

Fourth reuse of the ~107-week opener population for this family; no
rotation window is assigned or spent (a scoring of an already-played rule).

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_composed_rule_eval.py
    ... best_pick_composed_rule_eval.py --record --artifact <summary.json>
    ... best_pick_composed_rule_eval.py --week1-check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.best_pick import best_pick_tie_count, select_best_pick  # noqa: E402
from nfl_ats.best_pick_big_spread_challenger import (  # noqa: E402
    BIG_SPREAD_THRESHOLD,
    apply_big_spread_eligibility,
)
from nfl_ats.best_pick_nomination import (  # noqa: E402
    NominationV2Result,
    dispersion_pool_from_frame,
    select_nominee,
    select_nominee_v3,
)
from nfl_ats.margin import DEFAULT_LINE_SWEEP_OFFSETS, fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402

FAMILY = "pol09_best_pick_composed_v1"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817

V1_FEATURE_PROFILE = "weak_stack"
V1_REGRESSOR = "ridge"
V1_RIDGE_ALPHA = 10.0
V1_PROBABILITY_METHOD = "gaussian_median"
V1_MIN_TRAIN_GAMES = 500


def active_feature_table(artifacts_root: Path = REPO / "artifacts") -> Path:
    """The feature table the ACTIVE forecast was built from, resolved exactly
    the way production's v2 nomination resolves it
    (``nfl_ats.card_view.v2_nomination_inputs`` on the linked forecast's
    metadata) -- so the v1 sweep refit and the live v2 fit read one table."""

    from nfl_ats.active_model import load_active_ats_model
    from nfl_ats.card_view import v2_nomination_inputs
    from nfl_ats.public_board import active_artifact_path

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise SystemExit("No synchronized active ATS model; pass --features explicitly")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise SystemExit("Active model has no linked weekly forecast; pass --features")
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    inputs = v2_nomination_inputs(metadata, REPO / "data")
    if inputs is None:
        raise SystemExit("Forecast metadata carries no feature table; pass --features")
    return Path(inputs.feature_table)


def _load_script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ranker = _load_script("best_pick_opener_ranker_eval")


def build_archive_frame(
    source: Path, microstructure_source: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The 2026-08-18 working frame with BOTH the predeclared eval script's
    ``dispersion_pool_pass`` and production's ``pool_pass``
    (``dispersion_pool_from_frame`` per week). They must agree on every game
    or the run aborts -- the production code path is what is scored."""

    work = _ranker.load_working_frame(source)
    work, eval_pool_summary = _ranker.build_dispersion_pool(work, microstructure_source)
    work = attach_production_pool(work)
    mismatch = work.loc[work["pool_pass"] != work["dispersion_pool_pass"]]
    if not mismatch.empty:
        raise SystemExit(
            "Production dispersion_pool_from_frame disagrees with the predeclared eval "
            f"pool on {len(mismatch)} games; refusing to score. First rows: "
            f"{mismatch[['game_id', 'spread_std']].head().to_dict(orient='records')}"
        )
    return work, eval_pool_summary


def attach_production_pool(work: pd.DataFrame) -> pd.DataFrame:
    """Add ``pool_pass``/``pool_fallback``/``pool_fallback_reason`` per week
    from the production pool rule."""

    work = work.copy()
    work["pool_pass"] = False
    work["pool_fallback"] = False
    work["pool_fallback_reason"] = ""
    for _, group in work.groupby(["season", "week"], sort=True):
        pool = dispersion_pool_from_frame(group[["game_id", "spread_std"]])
        passes = pool.frame.set_index("game_id")["pool_pass"]
        work.loc[group.index, "pool_pass"] = (
            group["game_id"].astype(str).map(passes).astype(bool).to_numpy()
        )
        work.loc[group.index, "pool_fallback"] = pool.fallback
        work.loc[group.index, "pool_fallback_reason"] = pool.fallback_reason or ""
    return work


def _correct(group: pd.DataFrame, game_id: str, column: str) -> float:
    value = group.loc[group["game_id"].astype(str).eq(game_id), column]
    if value.empty:
        raise ValueError(f"nominee {game_id} is not in its own week")
    scalar = value.iloc[0]
    return float(scalar) if pd.notna(scalar) else float("nan")


def nominate_composed(work: pd.DataFrame) -> pd.DataFrame:
    """Per week: the v2 nominee AS PLAYED (production ``select_nominee`` on
    the production pool), the alphabetical fallback (``select_nominee_v3``,
    chooser 6), the unfiltered dispersion tie-break (chooser 8), the
    unfiltered chooser 4, and the big-spread discount
    (``apply_big_spread_eligibility``). Every nominee is scored on the
    ACTIVE model's pick (``baseline_correct_open``); the v2 nominee is also
    scored on the candidate arm for reconciliation with the v3 audit.

    Requires ``pool_pass`` (see :func:`attach_production_pool`). Each week
    is nominated from its own rows only, so a later week's rows can never
    change an earlier week's nominee (pinned by the leakage test).
    """

    required = {
        "game_id",
        "season",
        "week",
        "candidate_dist",
        "spread_std",
        "pool_pass",
        "tue_open_home_spread",
        "baseline_correct_open",
        "candidate_correct_open",
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"nominate_composed is missing columns: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for (season, week), group in work.groupby(["season", "week"], sort=True):
        table = group[["game_id", "candidate_dist", "spread_std", "pool_pass"]].copy()
        table["game_id"] = table["game_id"].astype(str)
        eligible = table.loc[table["pool_pass"]]
        if eligible.empty:
            raise ValueError(f"({season}, {week}) has no eligible games; fallback rule broken")

        v2_id, v2_tied, v2_tie_break = select_nominee(eligible)
        v3_id, v3_tied, v3_tie_break = select_nominee_v3(eligible)
        ch8_id, _, _ = select_nominee(table)
        ch4_id, _, _ = select_nominee_v3(table)

        base = NominationV2Result(
            game_id=v2_id,
            n_tied_at_max=v2_tied,
            tie_break=v2_tie_break,
            probability_table=table.sort_values("game_id").reset_index(drop=True),
            dispersion=dispersion_pool_from_frame(table[["game_id", "spread_std"]]),
        )
        predictions = pd.DataFrame(
            {
                "game_id": group["game_id"].astype(str).to_numpy(),
                "spread_line": group["tue_open_home_spread"].astype(float).to_numpy(),
            }
        )
        discount = apply_big_spread_eligibility(predictions, base)
        if discount.base_v2_game_id != v2_id:
            raise ValueError("big-spread discount lost track of its v2 base nominee")

        week_avg = group["baseline_correct_open"].dropna()
        rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "n_games": len(group),
                "n_pool": int(table["pool_pass"].sum()),
                "pool_fallback": bool(group["pool_fallback"].iloc[0])
                if "pool_fallback" in group.columns
                else False,
                "v2_game_id": v2_id,
                "v2_n_tied": v2_tied,
                "v2_tie_break": v2_tie_break,
                "v2_correct": _correct(group, v2_id, "baseline_correct_open"),
                "v2_correct_candidate_arm": _correct(group, v2_id, "candidate_correct_open"),
                "v3_game_id": v3_id,
                "v3_n_tied": v3_tied,
                "v3_tie_break": v3_tie_break,
                "v3_correct": _correct(group, v3_id, "baseline_correct_open"),
                "ch8_game_id": ch8_id,
                "ch8_correct": _correct(group, ch8_id, "baseline_correct_open"),
                "ch4_game_id": ch4_id,
                "ch4_correct": _correct(group, ch4_id, "baseline_correct_open"),
                "u2_game_id": discount.game_id,
                "u2_n_tied": discount.n_tied_at_max,
                "u2_tie_break": discount.tie_break,
                "u2_n_excluded": len(discount.excluded_game_ids),
                "u2_fallback_to_v2": discount.fallback_to_v2,
                "u2_correct": _correct(group, discount.game_id, "baseline_correct_open"),
                "week_avg_correct": float(week_avg.mean()) if len(week_avg) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def opener_sweeps(
    features: pd.DataFrame,
    work: pd.DataFrame,
    *,
    feature_profile: str = V1_FEATURE_PROFILE,
    regressor: str = V1_REGRESSOR,
    ridge_alpha: float = V1_RIDGE_ALPHA,
    min_train_games: int = V1_MIN_TRAIN_GAMES,
    probability_method: str = V1_PROBABILITY_METHOD,
) -> pd.DataFrame:
    """Line sweeps anchored at the archived Tuesday opener from the weekly
    walk-forward fit -- the archive's own loop
    (``scripts/ridge_alpha_promotion_eval.py::evaluate_arm``) mirrored: train
    on completed regular-season games strictly before the week's first
    kickoff, swap ONLY ``spread_line`` to the opener, sweep. Carries the
    fit's opener residual for the reproduction check."""

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()
    rows: list[pd.DataFrame] = []
    for (season, week), group in work.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"].astype(str)))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=regressor,
            feature_profile=feature_profile,  # type: ignore[arg-type]
            ridge_alpha=ridge_alpha,
        )
        at_open = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        sweep = model.line_sweep(
            at_open,
            offsets=DEFAULT_LINE_SWEEP_OFFSETS,
            probability_method=probability_method,  # type: ignore[arg-type]
        )
        sweep["season"] = int(str(season))
        sweep["week"] = int(str(week))
        point = model.predict(at_open)
        point["game_id"] = at_open["game_id"].to_numpy()
        rows.append(
            sweep.merge(point[["game_id", "predicted_market_residual"]], on="game_id", how="left")
        )
    if not rows:
        raise ValueError("No archive week had enough prior training games for the v1 sweep")
    return pd.concat(rows, ignore_index=True)


def refit_divergence(check: pd.DataFrame) -> dict[str, Any]:
    """How far the v1 sweep refit sits from the archived fit: the archive was
    built 2026-08-18 from a ``weak_stack`` table that no longer exists on
    disk, so an exact reproduction is not available and the gap is reported
    rather than assumed away. The v1 nominee's correctness is always scored
    on the ARCHIVE's played side, never the refit's."""

    diff = (check["residual_at_open"] - check["predicted_market_residual"]).abs()
    refit_home = check["predicted_market_residual"].gt(0.0)
    side_agree = check["baseline_pick_home"].astype(bool).eq(refit_home)
    per_season = {
        str(season): {
            "mean_abs_residual_diff": float(diff.loc[group.index].mean()),
            "pick_side_agreement": float(side_agree.loc[group.index].mean()),
        }
        for season, group in check.groupby("season", sort=True)
    }
    rank_agreements: list[float] = []
    for _, group in check.groupby(["season", "week"], sort=True):
        if len(group) < 3:
            continue
        rho = (
            group["residual_at_open"]
            .abs()
            .rank()
            .corr(group["predicted_market_residual"].abs().rank())
        )
        if pd.notna(rho):
            rank_agreements.append(float(rho))
    return {
        "max_abs_residual_diff": float(diff.max()),
        "mean_abs_residual_diff": float(diff.mean()),
        "median_abs_residual_diff": float(diff.median()),
        "p95_abs_residual_diff": float(diff.quantile(0.95)),
        "residual_correlation": float(
            check["residual_at_open"].corr(check["predicted_market_residual"])
        ),
        "pick_side_agreement": float(side_agree.mean()),
        "median_within_week_abs_residual_rank_correlation": (
            float(pd.Series(rank_agreements).median()) if rank_agreements else float("nan")
        ),
        "per_season": per_season,
    }


def nominate_v1(work: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    """Per week: ``nfl_ats.best_pick.select_best_pick`` (frozen v1) on the
    archive's own pick sides (``baseline_prob_open``) and the week's sweep."""

    rows: list[dict[str, Any]] = []
    for (season, week), group in work.groupby(["season", "week"], sort=True):
        predictions = pd.DataFrame(
            {
                "game_id": group["game_id"].astype(str).to_numpy(),
                "home_cover_probability": group["baseline_prob_open"].astype(float).to_numpy(),
            }
        )
        week_sweep = sweep.loc[
            sweep["game_id"].astype(str).isin(set(predictions["game_id"])),
            ["game_id", "line_offset", "home_cover_probability"],
        ]
        v1_id = select_best_pick(predictions, week_sweep)
        rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "v1_game_id": v1_id,
                "v1_n_tied": best_pick_tie_count(predictions, week_sweep),
                "v1_correct": _correct(group, v1_id, "baseline_correct_open")
                if v1_id is not None
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_delta(
    weekly: pd.DataFrame, a_col: str, b_col: str, *, samples: int, seed: int
) -> dict[str, Any]:
    """Week-blocked bootstrap of the mean paired difference ``a - b`` over
    weeks where BOTH nominees resolved (no push). Reuses the predeclared eval
    script's ``_bootstrap_series`` (``clv.week_blocked_bootstrap``)."""

    valid = weekly.dropna(subset=[a_col, b_col]).copy()
    valid["delta"] = valid[a_col] - valid[b_col]
    stats = _ranker._bootstrap_series(valid, "delta", samples=samples, seed=seed)
    stats["n_weeks_paired"] = stats.pop("n_weeks")
    a_id, b_id = a_col.replace("_correct", "_game_id"), b_col.replace("_correct", "_game_id")
    if a_id in valid.columns and b_id in valid.columns:
        differs = valid.loc[valid[a_id] != valid[b_id]]
        stats["n_weeks_nominee_differs"] = len(differs)
        stats["n_weeks_outcome_differs"] = int((differs["delta"] != 0).sum())
    return stats


def hit_rate(weekly: pd.DataFrame, col: str) -> dict[str, Any]:
    valid = weekly[col].dropna()
    return {
        "weeks_scored": len(valid),
        "hits": int(valid.sum()),
        "hit_rate": float(valid.mean()) if len(valid) else float("nan"),
    }


def tie_break_audit(weekly: pd.DataFrame) -> dict[str, Any]:
    tied = weekly.loc[weekly["v2_n_tied"] > 1]
    decided = tied.loc[tied["v2_tie_break"].eq("dispersion")]
    differs = decided.loc[decided["v2_game_id"] != decided["v3_game_id"]]
    return {
        "weeks_total": len(weekly),
        "weeks_tied_at_top_of_pool": len(tied),
        "weeks_dispersion_decided": len(decided),
        "weeks_fell_through_to_game_id": int(len(tied) - len(decided)),
        "weeks_dispersion_nominee_differs_from_alphabetical": len(differs),
        "dispersion_nominee_in_tied_weeks": hit_rate(tied, "v2_correct"),
        "alphabetical_nominee_in_tied_weeks": hit_rate(tied, "v3_correct"),
        "dispersion_vs_alphabetical_in_tied_weeks": (
            paired_delta(tied, "v2_correct", "v3_correct", samples=2_000, seed=BOOTSTRAP_SEED)
            if len(tied.dropna(subset=["v2_correct", "v3_correct"])) >= 2
            else None
        ),
        "tied_weeks_detail": tied[
            [
                "season",
                "week",
                "v2_n_tied",
                "v2_tie_break",
                "v2_game_id",
                "v2_correct",
                "v3_game_id",
                "v3_correct",
            ]
        ].to_dict(orient="records"),
    }


def score(
    weekly: pd.DataFrame, *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    cells = {
        "v2_as_played_vs_v1_sweep": ("v2_correct", "v1_correct"),
        "v2_as_played_vs_top_residual": ("v2_correct", "sq_correct"),
        "v2_plus_big_spread_discount_vs_v1_sweep": ("u2_correct", "v1_correct"),
        "v2_plus_big_spread_discount_vs_top_residual": ("u2_correct", "sq_correct"),
        "v2_plus_big_spread_discount_vs_v2_as_played": ("u2_correct", "v2_correct"),
    }
    return {
        "hit_rates": {
            "v2_as_played": hit_rate(weekly, "v2_correct"),
            "v2_candidate_arm_reconciliation": hit_rate(weekly, "v2_correct_candidate_arm"),
            "v2_plus_big_spread_discount": hit_rate(weekly, "u2_correct"),
            "v1_sweep_robustness": hit_rate(weekly, "v1_correct"),
            "top_residual": hit_rate(weekly, "sq_correct"),
            "chooser6_filter_game_id_tiebreak": hit_rate(weekly, "v3_correct"),
            "chooser8_unfiltered_dispersion_tiebreak": hit_rate(weekly, "ch8_correct"),
            "chooser4_unfiltered": hit_rate(weekly, "ch4_correct"),
            "all_pick_week_average": {
                "mean_of_weekly_averages": float(weekly["week_avg_correct"].mean())
            },
        },
        "composition_mattered": {
            "weeks_v2_differs_from_chooser6_filter_alone": int(
                (weekly["v2_game_id"] != weekly["v3_game_id"]).sum()
            ),
            "weeks_v2_differs_from_chooser8_tiebreak_alone": int(
                (weekly["v2_game_id"] != weekly["ch8_game_id"]).sum()
            ),
            "weeks_v2_differs_from_unfiltered_chooser4": int(
                (weekly["v2_game_id"] != weekly["ch4_game_id"]).sum()
            ),
            "weeks_v2_differs_from_v1": int((weekly["v2_game_id"] != weekly["v1_game_id"]).sum()),
            "weeks_v2_differs_from_top_residual": int(
                (weekly["v2_game_id"] != weekly["sq_game_id"]).sum()
            ),
            "weeks_discount_changed_the_nominee": int(
                (weekly["u2_game_id"] != weekly["v2_game_id"]).sum()
            ),
            "weeks_discount_excluded_something": int((weekly["u2_n_excluded"] > 0).sum()),
            "weeks_discount_fell_back_to_v2": int(weekly["u2_fallback_to_v2"].sum()),
            "weeks_v1_tied": int((weekly["v1_n_tied"] > 1).sum()),
        },
        "cells": {
            name: paired_delta(weekly, a, b, samples=samples, seed=seed)
            for name, (a, b) in cells.items()
        },
        "tie_break_audit": tie_break_audit(weekly),
    }


_CELL_TEXT: dict[str, tuple[str, str]] = {
    "v2_as_played_vs_v1_sweep": (
        "Production Best Pick nomination (v2 as played: alpha=2000 probability distance, "
        "below-median cross-book opener dispersion pool, dispersion then game_id tie-break; "
        "nfl_ats.best_pick_nomination.select_nominee) vs the v1 sweep_robustness rule it "
        "replaced (nfl_ats.best_pick.select_best_pick on a gaussian_median opener sweep), "
        "both settled on the ACTIVE model's pick side, top-1 hit rate per week",
        "Which game we flag as the week's Best Pick. The rule we use now (the calmer model's "
        "surest game, among games the sportsbooks agree on) against the old rule "
        "(the game whose pick survives the biggest line move). {delta:+.1f} points a week "
        "for the current rule, {pp:.0%} likely the better of the two; they name a different "
        "game in {differs} of {paired} weeks.",
    ),
    "v2_as_played_vs_top_residual": (
        "Production Best Pick nomination (v2 as played) vs the raw model's top-|residual| "
        "pick (scripts/best_pick_opener_ranker_eval.py chooser 1), both settled on the "
        "ACTIVE model's pick side, top-1 hit rate per week",
        "Which game we flag as the week's Best Pick. The rule we use now against simply "
        "taking the game where the model disagrees most with the line. {delta:+.1f} points "
        "a week for the current rule, {pp:.0%} likely the better of the two; they name a "
        "different game in {differs} of {paired} weeks.",
    ),
    "v2_plus_big_spread_discount_vs_v1_sweep": (
        "Production Best Pick nomination with the side-ledger 10+ opener-spread eligibility "
        "exclusion (nfl_ats.best_pick_big_spread_challenger.apply_big_spread_eligibility, "
        "fallback to the unmodified v2 pool) vs the v1 sweep_robustness rule, both settled "
        "on the ACTIVE model's pick side, top-1 hit rate per week",
        "Which game we flag as the week's Best Pick, if we also refuse to flag any game with "
        "a spread of 10 or more, against the old rule. {delta:+.1f} points a week for the "
        "no-blowouts version, {pp:.0%} likely the better of the two.",
    ),
    "v2_plus_big_spread_discount_vs_top_residual": (
        "Production Best Pick nomination with the 10+ opener-spread eligibility exclusion vs "
        "the raw model's top-|residual| pick, both settled on the ACTIVE model's pick side, "
        "top-1 hit rate per week",
        "Which game we flag as the week's Best Pick, if we also refuse to flag any game with "
        "a spread of 10 or more, against simply taking the game where the model disagrees "
        "most with the line. {delta:+.1f} points a week for the no-blowouts version, "
        "{pp:.0%} likely the better of the two.",
    ),
    "v2_plus_big_spread_discount_vs_v2_as_played": (
        "The 10+ opener-spread eligibility exclusion's marginal on top of the production "
        "Best Pick nomination (v2 as played), both settled on the ACTIVE model's pick side, "
        "top-1 hit rate per week",
        "Does refusing to flag a 10-point-or-more spread as the Best Pick help? Same rule "
        "otherwise. {delta:+.1f} points a week, {pp:.0%} likely to help; it changes the "
        "flagged game in only {differs} of {paired} weeks.",
    ),
}

_TIEBREAK_TEXT = (
    "Tie-break audit: in the weeks where the production Best Pick pool ties at the top, "
    "the dispersion tie-break's nominee vs the alphabetical game_id fallback "
    "(nfl_ats.best_pick_nomination.select_nominee_v3) on the same pool, both settled on "
    "the ACTIVE model's pick side, top-1 hit rate; descriptive, tie weeks only",
    "When two games tie for the week's Best Pick, we now break the tie toward the game the "
    "sportsbooks agree on most instead of alphabetical order. That happened in only "
    "{paired} weeks, and the two ways picked a different game in {differs} of them; the "
    "new tie-break came out {delta:+.0f} points a week against the alphabetical choice "
    "there. One week's difference, far too few to call either way.",
)


def record_cells(artifact_path: Path, *, replace: bool, recorded_at: str | None) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    pop = artifact["population"]
    timestamp = artifact_path.parent.name
    source = (
        f"scripts/best_pick_composed_rule_eval.py; artifacts/best_pick_composed_rule/"
        f"{timestamp}/summary.json; docs/best_pick_composed_rule.md"
    )
    common_notes = (
        f"Fourth reuse of the ~107-week opener population for the Best Pick family "
        f"(ridge_alpha promotion look, odds-microstructure battery, 2026-08-18 ranker "
        f"screen, 2026-08-19 v3 audit) -- compounding look-reuse discount. No rotation "
        f"window assigned or spent: a scoring of an already-played rule. Population: "
        f"{pop['games']:,} games, {pop['weeks']} weeks, seasons {pop['seasons'][0]}-"
        f"{pop['seasons'][-1]}. Nominees settle on the ACTIVE model's pick "
        f"(baseline_correct_open), not the candidate arm's; the 2026-08-19 v3 audit's "
        f"54.37% was the candidate-arm read. Bootstrap: seed {artifact['bootstrap_seed']}, "
        f"{artifact['bootstrap_samples']:,} draws, week-blocked, within-week correlation "
        f"zero. v1 sweep refit reproduction max |residual diff| vs archive: "
        f"{artifact['v1_sweep']['reproduction_max_abs_residual_diff']:.3g} "
        f"(probability_method {artifact['v1_sweep']['probability_method']})."
    )
    evidence = (
        "Predeclared cell (docs/best_pick_composed_rule.md). Neither admissible closing "
        "ground applies: the interval is not resolved on the wrong side of zero and no "
        "positive control was run for this family, so unresolved_below_power is the only "
        "admissible classification. An interval containing zero is the EXPECTED shape for "
        "a real-but-small signal at this evaluator's resolution, never a rejection."
    )

    jobs: list[tuple[str, dict[str, Any], str, str]] = []
    for key, (description, plain) in _CELL_TEXT.items():
        jobs.append((f"{FAMILY}_{key}", artifact["cells"][key], description, plain))
    audit = artifact["tie_break_audit"]
    if audit.get("dispersion_vs_alphabetical_in_tied_weeks"):
        jobs.append(
            (
                f"{FAMILY}_tiebreak_dispersion_vs_alphabetical",
                audit["dispersion_vs_alphabetical_in_tied_weeks"],
                _TIEBREAK_TEXT[0],
                _TIEBREAK_TEXT[1],
            )
        )

    for name, entry, description, plain_template in jobs:
        plain = plain_template.format(
            delta=entry["estimate"] * 100,
            pp=entry["probability_positive"],
            differs=entry.get("n_weeks_nominee_differs", 0),
            paired=entry["n_weeks_paired"],
        )
        cmd = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "weak-signals",
            "record",
            "--name",
            name,
            "--description",
            description,
            "--source",
            source,
            "--effect",
            f"{entry['estimate'] * 100:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            "unresolved_below_power",
            "--league",
            "nfl",
            "--season-start",
            str(pop["seasons"][0]),
            "--season-end",
            str(pop["seasons"][-1]),
            "--interval-low",
            f"{entry['lower'] * 100:.10f}",
            "--interval-high",
            f"{entry['upper'] * 100:.10f}",
            "--probability-positive",
            f"{entry['probability_positive']:.10f}",
            "--sample-games",
            str(pop["games"]),
            "--sample-blocks",
            str(entry["n_weeks_paired"]),
            "--family",
            FAMILY,
            "--category",
            "modeling",
            "--classification-evidence",
            evidence,
            "--plain-summary",
            plain,
            "--notes",
            common_notes + f" Nominees differ in {entry.get('n_weeks_nominee_differs', 'n/a')} of "
            f"{entry['n_weeks_paired']} paired weeks; the outcome differs in "
            f"{entry.get('n_weeks_outcome_differs', 'n/a')}.",
        ]
        if recorded_at:
            cmd += ["--recorded-at", recorded_at]
        if replace:
            cmd.append("--replace")
        print(f"=== recording {name} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {name} (exit {result.returncode}); per "
                "AGENTS.md 'if a record command errors, the verdict is wrong, not the "
                "validator' -- fix the invocation, never weaken the classification."
            )


def week1_check(artifacts_root: Path, data_root: Path) -> dict[str, Any]:
    """Reproduce the live Week 1 nominee from the active forecast through the
    production modules. Reads only; writes nothing."""

    from nfl_ats.active_model import load_active_ats_model
    from nfl_ats.best_pick_nomination import nominate_v2, nomination_v2_disclosure_note
    from nfl_ats.card_view import v2_nomination_inputs
    from nfl_ats.public_board import active_artifact_path

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise SystemExit("No synchronized active ATS model")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise SystemExit("Active model has no linked weekly forecast")
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    card = pd.read_csv(forecast / "recommendations.csv")
    inputs = v2_nomination_inputs(metadata, data_root)
    if inputs is None:
        raise SystemExit("Forecast metadata does not carry v2's inputs")
    features = pd.read_parquet(inputs.feature_table)
    result = nominate_v2(
        card,
        features,
        market_root=Path(inputs.market_root),
        season=inputs.season,
        week=inputs.week,
        regressor=inputs.regressor,
        feature_profile=inputs.feature_profile,
        min_train_games=inputs.min_train_games,
    )
    if result is None:
        raise SystemExit("nominate_v2 returned None for the active card")
    discount = apply_big_spread_eligibility(card, result)
    row = card.loc[card["game_id"].astype(str).eq(result.game_id)].iloc[0]
    prob = float(row["home_cover_probability"])
    side = str(row["home_team"]) if prob >= 0.5 else str(row["away_team"])
    spread = float(row["spread_line"])
    side_spread = -spread if prob >= 0.5 else spread
    return {
        "forecast": forecast.name,
        "active_model_id": active.get("model_id"),
        "season": inputs.season,
        "week": inputs.week,
        "v2_nominee": result.game_id,
        "v2_n_tied": result.n_tied_at_max,
        "v2_tie_break": result.tie_break,
        "v2_disclosure": nomination_v2_disclosure_note(result),
        "pool_fallback": result.dispersion.fallback,
        "pool_n_pass": result.dispersion.n_pool_pass,
        "nominee_side": side,
        "nominee_side_spread": side_spread,
        "big_spread_nominee": discount.game_id,
        "big_spread_excluded": list(discount.excluded_game_ids),
        "big_spread_fallback_to_v2": discount.fallback_to_v2,
    }


def run(
    *,
    source: Path,
    microstructure_source: Path,
    features_path: Path,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work, eval_pool_summary = build_archive_frame(source, microstructure_source)
    composed = nominate_composed(work)

    status_quo = _ranker.nominate(
        work,
        primary="abs_residual",
        secondary=None,
        correct_col="baseline_correct_open",
        alphabetical_only=False,
    )[["season", "week", "nominee_game_id", "nominee_correct"]].rename(
        columns={"nominee_game_id": "sq_game_id", "nominee_correct": "sq_correct"}
    )

    features = pd.read_parquet(features_path)
    sweep = opener_sweeps(features, work)
    check = work[["game_id", "season", "week", "residual_at_open", "baseline_pick_home"]].merge(
        sweep[["game_id", "predicted_market_residual"]].drop_duplicates("game_id"),
        on="game_id",
        how="inner",
    )
    divergence = refit_divergence(check)
    v1 = nominate_v1(work, sweep)

    weekly = composed.merge(status_quo, on=["season", "week"], how="inner", validate="one_to_one")
    weekly = weekly.merge(v1, on=["season", "week"], how="inner", validate="one_to_one")
    if len(weekly) != len(composed):
        raise ValueError("week join dropped rows between the three nominators")

    summary: dict[str, Any] = {
        "family": FAMILY,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "population": {
            "games": len(work),
            "weeks": len(weekly),
            "seasons": sorted(int(s) for s in work["season"].unique()),
            "pushes": int(work["baseline_correct_open"].isna().sum()),
            "arm_side_disagreements": int(
                (work["baseline_pick_home"] != work["candidate_pick_home"]).sum()
            ),
            "source": str(source.relative_to(REPO)),
            "microstructure_source": str(microstructure_source.relative_to(REPO)),
            "features": str(features_path),
        },
        "dispersion_pool": {
            "eval_summary": eval_pool_summary,
            "production_pool_matches_eval_pool_on_every_game": True,
        },
        "v1_sweep": {
            "feature_profile": V1_FEATURE_PROFILE,
            "regressor": V1_REGRESSOR,
            "ridge_alpha": V1_RIDGE_ALPHA,
            "probability_method": V1_PROBABILITY_METHOD,
            "min_train_games": V1_MIN_TRAIN_GAMES,
            "games_checked": len(check),
            "reproduction_max_abs_residual_diff": divergence["max_abs_residual_diff"],
            "refit_vs_archive": divergence,
            "big_spread_threshold": BIG_SPREAD_THRESHOLD,
        },
        **score(weekly, samples=samples, seed=seed),
        "multiplicity_note": (
            "Fourth reuse of the ~107-week opener population for the Best Pick family; "
            "compounding look-reuse discount. No rotation window assigned or spent."
        ),
        "binding_note": (
            "An interval containing zero is never a rejection; probability_positive is "
            "the continuous evidence. Closing grounds are limited to refuted_mechanism "
            "(resolved wrong sign or zero split-half reliability) and bounded_by_control; "
            "nothing here claims either."
        ),
    }
    return summary, weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_ranker.DEFAULT_SOURCE)
    parser.add_argument(
        "--microstructure-source", type=Path, default=_ranker.DEFAULT_MICROSTRUCTURE_SOURCE
    )
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--record", action="store_true", help="record cells from --artifact")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    parser.add_argument("--week1-check", action="store_true")
    args = parser.parse_args()

    if args.week1_check:
        print(json.dumps(week1_check(REPO / "artifacts", REPO / "data"), indent=2, default=str))
        return
    if args.record:
        if args.artifact is None:
            raise SystemExit("--record needs --artifact <summary.json>")
        record_cells(args.artifact, replace=args.replace, recorded_at=args.recorded_at)
        return

    summary, weekly = run(
        source=args.source,
        microstructure_source=args.microstructure_source,
        features_path=args.features or active_feature_table(),
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=str))
    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "artifacts" / "best_pick_composed_rule" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(summary, out_dir / "summary.json")
    weekly_path = out_dir / "weekly.parquet"
    weekly.to_parquet(weekly_path)
    stamp_sidecar(weekly_path)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()

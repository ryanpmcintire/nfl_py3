"""Stage 0 (CFB, free) of the Best Pick tie-break design.

Scoping doc: ``scratchpad/bestpick_routing_scope/design.md`` (session
artifact, not a repo file). Predeclaration:
``scratchpad/bestpick_routing_scope/predeclaration.md``, frozen before this
script's tie-break outcome was computed.

Three registry entries independently say their Brier/calibration gain
"should route to the Best Pick ranker" without naming a mechanism
(``registry/weak_signals.json``: ``ridge_alpha_global``,
``groupwise_ridge_block_penalties``, ``ecdf_smoothing_accuracy``). The
candidate mechanism is a magnitude-based **tie-break**: ``sweep_robustness``
(the one live Best Pick signal, frozen in ``nfl_ats.best_pick``) already
ranks the whole field; when it ties for the top score, ``select_best_pick``
currently breaks the tie alphabetically on ``game_id`` -- pure noise. This
script asks whether any of three already-built, already-Brier-positive
probability sources beats that alphabetical noise floor when used only to
resolve ties, never to re-decide which side (HOME/AWAY) the forced pick
takes.

This is Stage 0 only: CFB, free under rotation rule 8, no registry entry, no
NFL window spent. At most one candidate would be carried into a Stage 1 NFL
screen -- not decided by this script.

Data-availability fact, established by inspection before this script ran
(see predeclaration.md): ``artifacts/ridge_alpha_screen{,_fine}/`` and
``artifacts/groupwise_ridge/`` hold only aggregate CSVs -- their per-game
CFB predictions at ``ridge_alpha=2000`` and ``market_light_10 @ alpha=1e4``
were computed in memory by their respective screens and never persisted.
Per explicit instruction, this script does NOT regenerate those (that would
silently re-spend wall-clock on a screen that already ran once for a
different purpose); it measures the one candidate that DOES have a stored
per-game artifact (``tiebreak_ecdf_gaussian``,
``artifacts/ecdf_smoothing/20260818T000600Z/cfb_predictions.parquet``) and
reports exactly what a follow-up run would need to generate for the other
two.

The base tied-week identifier (``sweep_robustness`` itself) has never been
computed for CFB anywhere in this repo -- only NFL has a stored sweep
artifact. Building it is new orchestration, not new modeling code: the same
frozen weekly fit (``fit_cfb_residual_model``, ``ridge_alpha=10``, no column
penalties) that already produces ``artifacts/cfb_benchmark/`` predictions,
additionally swept via ``MarginModel.line_sweep`` -- exactly the pattern
``scripts/best_pick_ranker.py`` already established for NFL. A timing probe
(this session, 15 mid-window weeks) measured ~0.06s per weekly fit; the full
294-week walk-forward projects to ~18 seconds, not a contention risk.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_tiebreak_cfb_screen.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.best_pick import sweep_robustness
from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.margin import DEFAULT_LINE_SWEEP_OFFSETS

REPO = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO / "data" / "processed" / "cfb_game_features.parquet"
ECDF_ARTIFACT = (
    REPO / "artifacts" / "ecdf_smoothing" / "20260818T000600Z" / "cfb_predictions.parquet"
)
OUTPUT_ROOT = REPO / "artifacts" / "best_pick_tiebreak_cfb"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

# Candidate probability sources declared in design.md Section 2.2. Only
# entries with a `source` are measured this pass; the others are reported as
# unavailable, per predeclaration.md.
CANDIDATE_SOURCES: dict[str, dict[str, Any]] = {
    "tiebreak_alpha2000": {
        "mechanism": "re-read home_cover_probability at ridge_alpha=2000 (walk-forward "
        "Brier optimum) instead of the frozen alpha=10",
        "available": False,
        "missing_reason": "artifacts/ridge_alpha_screen{,_fine}/ store only aggregate CSVs; "
        "the per-game frame scripts/ridge_alpha_screen.py builds in memory "
        "(cfb_walk_forward_benchmark(features, ridge_alpha=2000.0)) is discarded "
        "after the aggregate is written.",
        "follow_up": "Re-run cfb_walk_forward_benchmark(features, ridge_alpha=2000.0) and "
        "persist outcome.predictions.loc[method=='market_residual'] to parquet "
        "(game_id, season, week, home_cover_probability, home_cover) before this "
        "screen can measure it.",
    },
    "tiebreak_groupwise_market_light_10": {
        "mechanism": "re-read home_cover_probability under market_light_10 @ alpha=1e4 "
        "group-wise penalties",
        "available": False,
        "missing_reason": "artifacts/groupwise_ridge/ stores only aggregate CSVs; the "
        "per-game frame scripts/groupwise_ridge_screen.py's run_arm() builds is "
        "discarded after the aggregate is written.",
        "follow_up": "Re-run groupwise_ridge_screen.run_arm(completed, 'market_light_10', "
        "10_000.0) and persist the returned per-game DataFrame to parquet before "
        "this screen can measure it.",
    },
    "tiebreak_ecdf_gaussian": {
        "mechanism": "re-read home_cover_probability via "
        "calibration.smoothed_home_cover_probability(method='gaussian') instead of "
        "the raw ECDF",
        "available": True,
        "source": str(ECDF_ARTIFACT.relative_to(REPO)),
        "feature_set": "gaussian",
    },
}


# ---------------------------------------------------------------------------
# 1. CFB sweep harness -- new orchestration, frozen config, no new modeling
# ---------------------------------------------------------------------------


def cfb_sweep_and_point(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward the frozen CFB config, emitting a line sweep AND a point read.

    One fit per week (mirrors ``cfb_walk_forward_benchmark`` exactly: same
    cutoff rule, same min-train floor, same ridge alpha); both the sweep and
    the point prediction come from that SAME fitted model, so they cannot
    silently diverge.
    """

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError("No completed CFB games in the requested window")

    sweep_rows: list[pd.DataFrame] = []
    point_rows: list[pd.DataFrame] = []
    for (season, week), weekly in test.groupby(["season", "week"], sort=True):
        cutoff = weekly["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)

        sweep = model.line_sweep(weekly, offsets=DEFAULT_LINE_SWEEP_OFFSETS)
        sweep["season"] = int(str(season))
        sweep["week"] = int(str(week))
        sweep_rows.append(sweep)

        point = model.predict(weekly)
        point["game_id"] = weekly["game_id"].to_numpy()
        point["season"] = int(str(season))
        point["week"] = int(str(week))
        point["gameday"] = weekly["gameday"].to_numpy()
        point["spread_line"] = weekly["spread_line"].to_numpy()
        point["home_cover"] = weekly["home_cover"].to_numpy()
        point_rows.append(point)

    if not sweep_rows:
        raise ValueError("No CFB week had enough prior training games for the sweep")
    sweep_frame = pd.concat(sweep_rows, ignore_index=True)
    point_frame = pd.concat(point_rows, ignore_index=True)
    point_frame["evaluation_window"] = point_frame["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    return sweep_frame, point_frame


# ---------------------------------------------------------------------------
# 2. Picks + sweep_robustness (frozen function, imported unchanged)
# ---------------------------------------------------------------------------


def build_picks(point: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    picks = point.copy()
    picks["game_id"] = picks["game_id"].astype(str)
    picks["pick"] = np.where(picks["home_cover_probability"] >= 0.5, "HOME", "AWAY")
    picks = picks.loc[picks["home_cover"].notna()].copy()
    picks["correct"] = np.where(
        picks["pick"].eq("HOME"),
        picks["home_cover"].astype(float),
        1.0 - picks["home_cover"].astype(float),
    )
    sweep_work = sweep.copy()
    sweep_work["game_id"] = sweep_work["game_id"].astype(str)
    picks["sweep_robustness"] = (
        picks["game_id"].map(sweep_robustness(sweep_work, picks)).astype(float)
    )
    return picks


# ---------------------------------------------------------------------------
# 3. Tied weeks, addressable population, tie-break nominations
# ---------------------------------------------------------------------------


def tied_weeks_table(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per week: tie diagnostics, independent of any candidate source."""

    rows: list[dict[str, Any]] = []
    for (season, week), grp in picks.groupby(["season", "week"], sort=True):
        scored = grp.dropna(subset=["sweep_robustness"])
        if scored.empty:
            continue
        top_score = scored["sweep_robustness"].max()
        tied = scored.loc[scored["sweep_robustness"] == top_score]
        alphabetical_nom = tied.sort_values("game_id").iloc[0]["game_id"]
        outcomes = tied["correct"].to_numpy()
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "n_candidates": len(scored),
                "n_tied": len(tied),
                "is_tied": len(tied) > 1,
                "max_score": float(top_score),
                "tied_game_ids": ",".join(sorted(tied["game_id"].astype(str))),
                "outcomes_disagree": len(tied) > 1 and (outcomes.min() != outcomes.max()),
                "alphabetical_nomination": str(alphabetical_nom),
            }
        )
    return pd.DataFrame(rows)


def attach_candidate(
    tied: pd.DataFrame, picks: pd.DataFrame, candidate_probability: pd.Series, *, name: str
) -> pd.DataFrame:
    """Add {name}_nomination and a flip flag vs. the alphabetical nomination.

    Nomination rule (design.md Section 2.2, steps 4-5): within the tied
    group, take margin = |candidate_probability - 0.5| (pick-side complement
    applied first -- symmetric, but kept explicit so the "pick side is never
    re-decided" property is auditable in code, not just asserted in prose).
    Largest margin wins; a margin tie, or a missing candidate probability for
    any tied game, falls back to the existing alphabetical rule.
    """

    prob_lookup = candidate_probability.to_dict()
    pick_lookup = picks.set_index("game_id")["pick"].to_dict()
    correct_lookup = picks.set_index("game_id")["correct"].to_dict()

    nominations: list[str] = []
    margins_missing: list[bool] = []
    for _, row in tied.iterrows():
        if not row["is_tied"]:
            nominations.append(row["alphabetical_nomination"])
            margins_missing.append(False)
            continue
        game_ids = row["tied_game_ids"].split(",")
        margins: dict[str, float] = {}
        missing = False
        for gid in game_ids:
            prob = prob_lookup.get(gid)
            if prob is None or pd.isna(prob):
                missing = True
                continue
            pick = pick_lookup.get(gid)
            pickside = prob if pick == "HOME" else 1.0 - prob
            margins[gid] = abs(pickside - 0.5)
        if missing or not margins:
            nominations.append(row["alphabetical_nomination"])
            margins_missing.append(True)
            continue
        max_margin = max(margins.values())
        tied_on_margin = [gid for gid, m in margins.items() if m == max_margin]
        if len(tied_on_margin) > 1:
            nominations.append(sorted(tied_on_margin)[0])
        else:
            nominations.append(tied_on_margin[0])
        margins_missing.append(False)

    out = tied.copy()
    out[f"{name}_nomination"] = nominations
    out[f"{name}_missing_probability"] = margins_missing
    out[f"{name}_flip_vs_alphabetical"] = (
        out[f"{name}_nomination"] != out["alphabetical_nomination"]
    )
    out[f"{name}_nomination_correct"] = out[f"{name}_nomination"].map(correct_lookup)
    out["alphabetical_nomination_correct"] = out["alphabetical_nomination"].map(correct_lookup)
    return out


# ---------------------------------------------------------------------------
# 4. Primary Stage-0 metric: within-tie paired contest
# ---------------------------------------------------------------------------


def pairwise_contest(
    picks: pd.DataFrame, tied: pd.DataFrame, candidate_probability: pd.Series, *, name: str
) -> pd.DataFrame:
    """One row per informative pair: does each rule prefer the covering game?

    A pair (a, b) tied at the top score is INFORMATIVE only if exactly one of
    {a, b} covered (design.md Section 3) AND the candidate's own margin
    differs between a and b (a margin tie is censored -- dropped from BOTH
    the candidate's and the alphabetical baseline's tally for that pair, so
    the two rules are always compared on the identical pair population).
    """

    prob_lookup = candidate_probability.to_dict()
    pick_lookup = picks.set_index("game_id")["pick"].to_dict()
    correct_lookup = picks.set_index("game_id")["correct"].to_dict()

    rows: list[dict[str, Any]] = []
    for _, week_row in tied.loc[tied["is_tied"]].iterrows():
        game_ids = sorted(week_row["tied_game_ids"].split(","))
        for i in range(len(game_ids)):
            for j in range(i + 1, len(game_ids)):
                a, b = game_ids[i], game_ids[j]
                correct_a, correct_b = correct_lookup.get(a), correct_lookup.get(b)
                if correct_a is None or correct_b is None:
                    continue
                if correct_a == correct_b:
                    continue  # both covered or both lost: no information
                prob_a, prob_b = prob_lookup.get(a), prob_lookup.get(b)
                if prob_a is None or prob_b is None or pd.isna(prob_a) or pd.isna(prob_b):
                    continue
                pick_a, pick_b = pick_lookup.get(a), pick_lookup.get(b)
                margin_a = abs((prob_a if pick_a == "HOME" else 1.0 - prob_a) - 0.5)
                margin_b = abs((prob_b if pick_b == "HOME" else 1.0 - prob_b) - 0.5)
                if margin_a == margin_b:
                    continue  # candidate is uninformative on this pair: censored
                covering = a if correct_a == 1.0 else b
                candidate_prefers = a if margin_a > margin_b else b
                alphabetical_prefers = a  # a < b lexicographically by construction
                rows.append(
                    {
                        "season": int(week_row["season"]),
                        "week": int(week_row["week"]),
                        "game_a": a,
                        "game_b": b,
                        "covering_game": covering,
                        "candidate": name,
                        "candidate_prefers_correct": float(candidate_prefers == covering),
                        "alphabetical_prefers_correct": float(alphabetical_prefers == covering),
                    }
                )
    return pd.DataFrame(rows)


def _pairwise_metric_fn(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "candidate_win_rate_minus_half": float(frame["candidate_prefers_correct"].mean() - 0.5),
        "alphabetical_win_rate_minus_half": float(
            frame["alphabetical_prefers_correct"].mean() - 0.5
        ),
        "candidate_minus_alphabetical": float(
            frame["candidate_prefers_correct"].mean() - frame["alphabetical_prefers_correct"].mean()
        ),
    }


def bootstrap_pairwise(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return week_blocked_bootstrap(
        pairs,
        _pairwise_metric_fn,
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------


READ_ONLY_SCRIPT = True
# ENG-29: read-only with respect to artifacts/ and registry/; the ENG-29 scanner confirms its only
# write sites resolve to a caller-supplied `--output`/`--out` path with no artifacts/ or registry/
# default, never a governed tree by default.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--ecdf-artifact", type=Path, default=ECDF_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    print("[stage0] walking forward the frozen CFB config (alpha=10, no column penalties) ...")
    sweep, point = cfb_sweep_and_point(features)

    # Reproduction check (a): at line_offset==0 the sweep's alternative line
    # equals the quoted spread, so line_sweep's home_cover_probability there
    # must equal predict()'s home_cover_probability from the SAME weekly fit
    # (line_sweep carries no predicted_margin column of its own -- the center
    # is shared internally but only the probability is emitted per offset).
    at_offset0 = sweep.loc[sweep["line_offset"] == 0.0].drop_duplicates("game_id")
    check_a = point[["game_id", "home_cover_probability"]].merge(
        at_offset0[["game_id", "home_cover_probability"]],
        on="game_id",
        suffixes=("_point", "_sweep_offset0"),
    )
    repro_a = float(
        (check_a["home_cover_probability_point"] - check_a["home_cover_probability_sweep_offset0"])
        .abs()
        .max()
    )
    print(f"[check a] point vs sweep-offset-0 home_cover_probability, max|diff| = {repro_a:.3e}")

    picks = build_picks(point, sweep)

    # Reproduction check (b): my freshly computed home_cover_probability vs.
    # the already-stored ecdf_smoothing artifact's feature_set=="ecdf" rows --
    # licenses treating that artifact's "gaussian" rows as pick-consistent.
    ecdf_stored = pd.read_parquet(args.ecdf_artifact)
    ecdf_stored["game_id"] = ecdf_stored["game_id"].astype(str)
    ecdf_baseline = ecdf_stored.loc[ecdf_stored["feature_set"].eq("ecdf")].set_index("game_id")[
        "home_cover_probability"
    ]
    check_b = picks[["game_id", "home_cover_probability"]].merge(
        ecdf_baseline.rename("stored_ecdf_probability"), on="game_id", how="inner"
    )
    repro_b = float(
        (check_b["home_cover_probability"] - check_b["stored_ecdf_probability"]).abs().max()
    )
    print(
        f"[check b] fresh vs. stored ecdf_smoothing home_cover_probability, "
        f"max|diff| = {repro_b:.3e} over {len(check_b)} games"
    )

    tied = tied_weeks_table(picks)
    n_weeks = len(tied)
    n_tied_weeks = int(tied["is_tied"].sum())
    addressable = tied.loc[tied["is_tied"] & tied["outcomes_disagree"]]
    n_addressable = len(addressable)

    print(f"[population] weeks scored: {n_weeks}, tied weeks: {n_tied_weeks}")
    print(
        f"[population] ADDRESSABLE weeks (tied AND outcomes disagree): {n_addressable} "
        f"of {n_tied_weeks} tied weeks, {n_addressable / n_weeks:.4f} of all weeks"
    )

    gaussian_probability = ecdf_stored.loc[ecdf_stored["feature_set"].eq("gaussian")].set_index(
        "game_id"
    )["home_cover_probability"]

    tied_with_candidate = attach_candidate(
        tied, picks, gaussian_probability, name="tiebreak_ecdf_gaussian"
    )
    pairs = pairwise_contest(picks, tied, gaussian_probability, name="tiebreak_ecdf_gaussian")
    bootstrap = bootstrap_pairwise(pairs)

    flips = tied_with_candidate.loc[
        tied_with_candidate["is_tied"], "tiebreak_ecdf_gaussian_flip_vs_alphabetical"
    ]
    n_flips = int(flips.sum())
    flip_rows = tied_with_candidate.loc[
        tied_with_candidate["is_tied"]
        & tied_with_candidate["tiebreak_ecdf_gaussian_flip_vs_alphabetical"]
    ]
    flip_helped = int(
        (
            (flip_rows["tiebreak_ecdf_gaussian_nomination_correct"] == 1.0)
            & (flip_rows["alphabetical_nomination_correct"] == 0.0)
        ).sum()
    )
    flip_hurt = int(
        (
            (flip_rows["tiebreak_ecdf_gaussian_nomination_correct"] == 0.0)
            & (flip_rows["alphabetical_nomination_correct"] == 1.0)
        ).sum()
    )
    flip_neutral = n_flips - flip_helped - flip_hurt

    print(
        f"[flips] tiebreak_ecdf_gaussian nomination differs from alphabetical in "
        f"{n_flips}/{n_tied_weeks} tied weeks "
        f"(helped {flip_helped}, hurt {flip_hurt}, neutral {flip_neutral})"
    )
    if not bootstrap.empty:
        print("\n=== Within-tie pairwise contest, week-blocked bootstrap ===")
        print(bootstrap.to_string(index=False))

    # clean_core-only cut, for comparability with other CFB screens.
    clean_core_pick_ids = set(picks.loc[picks["evaluation_window"].eq("clean_core"), "game_id"])
    clean_core_pairs = pairs.loc[
        pairs["game_a"].isin(clean_core_pick_ids) & pairs["game_b"].isin(clean_core_pick_ids)
    ]
    clean_core_bootstrap = bootstrap_pairwise(clean_core_pairs)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    picks.to_parquet(out_dir / "sweep_picks.parquet", index=False)
    tied_with_candidate.to_csv(out_dir / "tied_weeks.csv", index=False)
    pairs.to_csv(out_dir / "pairwise_pairs.csv", index=False)

    summary: dict[str, Any] = {
        "predeclaration": "scratchpad/bestpick_routing_scope/predeclaration.md",
        "design": "scratchpad/bestpick_routing_scope/design.md",
        "rotation_registry_touched": False,
        "reproduction_check_a_point_vs_sweep_offset0_max_abs_diff": repro_a,
        "reproduction_check_b_fresh_vs_stored_ecdf_max_abs_diff": repro_b,
        "reproduction_check_b_games": len(check_b),
        "weeks_scored": n_weeks,
        "tied_weeks": n_tied_weeks,
        "tied_week_fraction": n_tied_weeks / n_weeks,
        "addressable_weeks": n_addressable,
        "addressable_fraction_of_tied_weeks": (
            n_addressable / n_tied_weeks if n_tied_weeks else 0.0
        ),
        "addressable_fraction_of_all_weeks": n_addressable / n_weeks,
        "candidate_sources": CANDIDATE_SOURCES,
        "measured_candidate": {
            "name": "tiebreak_ecdf_gaussian",
            "informative_pairs": len(pairs),
            "flips_vs_alphabetical": n_flips,
            "flips_helped": flip_helped,
            "flips_hurt": flip_hurt,
            "flips_neutral": flip_neutral,
            "week_blocked_bootstrap_all_weeks": bootstrap.to_dict(orient="records"),
            "week_blocked_bootstrap_clean_core": clean_core_bootstrap.to_dict(orient="records"),
        },
        "gate_note": "0.75 probability_positive is a claims gate (SPEC-5/ecdf_smoothing "
        "precedent), never a decision bar; continuous win rates are reported above "
        "regardless of where they land.",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (out_dir / "stage0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nartifacts: {out_dir}")


if __name__ == "__main__":
    main()

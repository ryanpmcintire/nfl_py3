"""MEASURE-ONLY: weak-stack v2 -- the ACTIVE weak_stack profile plus the next
tier of positive-leaning, buildable NFL signals, paired against the active
model at the opener grade.

Predeclared in
``.../scratchpad/stack_v2/predeclaration.md`` (read that file first -- it
scopes every candidate ingredient, states which are buildable vs excluded and
why, and states the double-dip discount before any number below existed).

Two ingredients are stacked on top of the active ``weak_stack`` profile:

1. **Narrowed injury severity** (``docs/injury_value_lost.md`` sec 4): the
   freshly-rebuilt ``game_features_player.parquet`` (2026-08-18) is, column
   for column, ``FEATURE_SETS["full_weak_stack"]`` with FIXED-prior injury
   severity instead of the active table's LEARNED severity -- verified this
   session by direct column-set comparison, not assumed. No new table build.
2. **``penalty_discipline``** (ROADMAP PBP-07 / registry, P+ 0.68): one new
   diff-only stacker column, ``diff_penalty_rate_prior`` = a team's prior
   *fully completed* season's penalty rate (home minus away), reusing the
   exact rate reconstruction ``scripts/penalty_discipline_interval.py``
   already verified reproduces the recorded mean/sd/reliability.

Both are wired in WITHOUT touching ``src/nfl_ats`` -- ``nfl_ats.margin``'s
``MARGIN_FEATURE_PROFILES`` tuple and ``_MARGIN_PROFILE_FEATURE_SETS`` dict,
and ``nfl_ats.constants.FEATURE_SETS`` (the same dict object ``margin.py``
imports by reference), are ordinary runtime data, not enforced by the
``Literal`` type hint. This script adds one profile, ``weak_stack_v2``, to
those registries in-process before calling the existing, UNMODIFIED
``evaluate_arm`` machinery from ``scripts/ridge_alpha_promotion_eval.py``
(imported, not rewritten -- that script reproduced the published
52.83%/51.566% baselines exactly).

Four arms, all at ridge_alpha=10.0 (frozen, unchanged) so any delta is
attributable to features alone:

  * Baseline  -- weak_stack / game_features_weak_stack.parquet (learned severity)
  * Arm A     -- weak_stack / game_features_player.parquet (fixed severity)
  * Arm B     -- weak_stack_v2 / game_features_player.parquet + penalty (fixed + penalty)
  * Arm C     -- weak_stack_v2 / game_features_weak_stack.parquet + penalty (learned + penalty)

Run:  ./.tools/uv.exe run --no-sync python scripts/weak_stack_v2_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from ridge_alpha_promotion_eval import (  # noqa: E402
    evaluate_arm,
    line_flip_buckets,
    paired_frame,
)

import nfl_ats.constants as constants_mod  # noqa: E402
import nfl_ats.margin as margin_mod  # noqa: E402
import nfl_ats.outcomes as outcomes_mod  # noqa: E402
from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.outcomes import walk_forward_outcomes  # noqa: E402
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # noqa: E402

NFLVERSE_SAMPLES = 2_000
NFLVERSE_SEED = 20260818

OUT_ROOT = REPO / "artifacts" / "weak_stack_v2"
RESULTS_MD = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\c8c5fbdd-027f-438d-b992-979e83a91c2e\scratchpad\stack_v2\results.md"
)

RIDGE_ALPHA = 10.0
OPENER_SAMPLES = 20_000
OPENER_SEED = 20260817  # this project's standing opener-bootstrap seed
NFLVERSE_START_SEASON = 2018

BASELINE_FEATURES_PATH = REPO / "data/processed/game_features_weak_stack.parquet"
NARROWED_FEATURES_PATH = REPO / "data/processed/game_features_player.parquet"
MARKET_ROOT = REPO / "data/market/raw"
PBP_RAW_ROOT = REPO / "data/pbp/raw"


# ---------------------------------------------------------------------------
# Step 1.2: penalty_discipline as a new diff-only stacker column
# ---------------------------------------------------------------------------


def _canonical(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_season_penalty_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Identical construction to ``scripts/penalty_discipline_interval.py``:

    mean(penalty) over every raw regular-season play where posteam == team.
    Reused verbatim (not re-derived) because it is the one definition already
    verified to reproduce the registry's recorded mean/sd/reliability.
    """

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["penalty"] = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        plays=("penalty", "size"), penalties=("penalty", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays"]
    return grouped.reset_index()


def add_penalty_discipline_feature(features: pd.DataFrame, rate: pd.DataFrame) -> pd.DataFrame:
    """Attach ``diff_penalty_rate_prior`` = home_prior_rate - away_prior_rate.

    Leak safety, argued and checked: ``prior_rate`` for a team-season is that
    team's rate in ``season - 1`` ONLY, joined by an explicit
    ``prev_season = season - 1`` merge key -- no team-season can match its own
    season's plays or any later season's. A team with no locally-observed
    prior season (2009, or any franchise's first year in this snapshot) gets
    NaN, which ``fit_margin_model``'s imputer already handles for every other
    frozen feature the same way. This mirrors ``diff_`` = home - away, the
    convention every other player-composite stacker column in this project
    uses (``constants.py`` ``_difference_features``).
    """

    lag = rate.copy()
    lag["prev_season"] = lag["season"] + 1
    lag = lag.rename(columns={"season": "rate_season", "rate": "prior_rate"})[
        ["team", "rate_season", "prev_season", "prior_rate"]
    ]

    result = features.copy()
    for side in ("home", "away"):
        # Every matched row's join key is season == prev_season, and
        # prev_season = rate_season + 1 by construction above, so a match can
        # only ever pull a STRICTLY EARLIER season's plays. No self-merge, no
        # same-season leakage, checked separately in _leak_safety_selfcheck.
        joined = result[["game_id", "season", f"{side}_team"]].merge(
            lag[["team", "prev_season", "prior_rate"]],
            left_on=["season", f"{side}_team"],
            right_on=["prev_season", "team"],
            how="left",
        )
        result[f"{side}_penalty_rate_prior"] = joined["prior_rate"].to_numpy()
    result["diff_penalty_rate_prior"] = (
        result["home_penalty_rate_prior"] - result["away_penalty_rate_prior"]
    )
    return result


def _leak_safety_selfcheck(rate: pd.DataFrame) -> None:
    """Assert the lag join can never pair a team-season with its own or a
    future season's plays (in-script stand-in for a pytest leakage test,
    since this column is not landed in ``src/``; see predeclaration sec 1.2)."""

    lag = rate.copy()
    lag["prev_season"] = lag["season"] + 1
    for _, row in lag.iterrows():
        # The season this rate is attached FORWARD to (prev_season) must be
        # strictly greater than the season the plays were drawn from.
        assert row["prev_season"] > row["season"], "penalty rate must lag strictly backward"
    print(f"  leak-safety self-check passed on {len(lag)} team-season rate rows")


# ---------------------------------------------------------------------------
# Step 1.2 (continued): wire weak_stack_v2 into margin.py's runtime registries
# ---------------------------------------------------------------------------


def register_weak_stack_v2_profile() -> None:
    penalty_family = ("diff_penalty_rate_prior",)
    constants_mod.FEATURE_FAMILIES["penalty_discipline"] = penalty_family
    constants_mod.FEATURE_SETS["football_weak_stack_v2"] = (
        constants_mod.FEATURE_SETS["football_weak_stack"] + penalty_family
    )
    constants_mod.FEATURE_SETS["full_weak_stack_v2"] = (
        constants_mod.FEATURE_SETS["full_weak_stack"] + penalty_family
    )
    if "weak_stack_v2" not in margin_mod.MARGIN_FEATURE_PROFILES:
        margin_mod.MARGIN_FEATURE_PROFILES = (
            *margin_mod.MARGIN_FEATURE_PROFILES,
            "weak_stack_v2",
        )
    margin_mod._MARGIN_PROFILE_FEATURE_SETS["weak_stack_v2"] = (
        "football_weak_stack_v2",
        "full_weak_stack_v2",
    )
    # nfl_ats.outcomes did `from nfl_ats.margin import MARGIN_FEATURE_PROFILES`,
    # which bound its OWN name to the original tuple object at import time --
    # rebinding margin_mod's attribute above does not reach it. Patch the
    # outcomes-module name directly so walk_forward_outcomes (used only for
    # the nflverse-grade completeness read) accepts the new profile too.
    outcomes_mod.MARGIN_FEATURE_PROFILES = margin_mod.MARGIN_FEATURE_PROFILES
    print(
        "  registered runtime profile weak_stack_v2 -> "
        f"{len(constants_mod.FEATURE_SETS['full_weak_stack_v2'])} columns "
        f"({len(constants_mod.FEATURE_SETS['full_weak_stack'])} + "
        f"{len(penalty_family)} penalty)"
    )


# ---------------------------------------------------------------------------
# Paired scoring, one arm pair at a time (reuses evaluate_arm unmodified)
# ---------------------------------------------------------------------------


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": "ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def score_pair(
    *,
    label: str,
    baseline: pd.DataFrame,
    baseline_profile: str,
    candidate_features: pd.DataFrame,
    candidate_profile: str,
    min_train_games: int,
) -> dict[str, Any]:
    t1 = time.perf_counter()
    candidate = evaluate_arm(
        MARKET_ROOT,
        candidate_features,
        config=_config(candidate_profile),
        min_train_games=min_train_games,
    )
    t2 = time.perf_counter()

    paired = paired_frame(baseline, candidate)

    def accuracy_metric_fn(open_col: str, base_col: str):
        def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
            return {
                "candidate_minus_baseline": float(frame[open_col].mean() - frame[base_col].mean())
            }

        return metric_fn

    open_week = week_blocked_bootstrap(
        paired,
        accuracy_metric_fn("candidate_correct_open", "baseline_correct_open"),
        block="week",
        samples=OPENER_SAMPLES,
        seed=OPENER_SEED,
    )
    open_season = week_blocked_bootstrap(
        paired,
        accuracy_metric_fn("candidate_correct_open", "baseline_correct_open"),
        block="season",
        samples=OPENER_SAMPLES,
        seed=OPENER_SEED,
    )
    close_week = week_blocked_bootstrap(
        paired,
        accuracy_metric_fn("candidate_correct_close", "baseline_correct_close"),
        block="week",
        samples=OPENER_SAMPLES,
        seed=OPENER_SEED,
    )

    def brier_metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        valid = frame.dropna(subset=["actual_home_cover_open"])
        actual = valid["actual_home_cover_open"].to_numpy(dtype=float)
        cand = valid["candidate_prob_open"].to_numpy(dtype=float)
        base = valid["baseline_prob_open"].to_numpy(dtype=float)
        clip = lambda p: np.clip(p, 1e-15, 1.0 - 1e-15)  # noqa: E731
        cand_brier = np.mean(np.square(cand - actual))
        base_brier = np.mean(np.square(base - actual))
        cand_ll = np.mean(-(actual * np.log(clip(cand)) + (1 - actual) * np.log(1.0 - clip(cand))))
        base_ll = np.mean(-(actual * np.log(clip(base)) + (1 - actual) * np.log(1.0 - clip(base))))
        return {
            "brier_improvement": float(base_brier - cand_brier),
            "log_loss_improvement": float(base_ll - cand_ll),
        }

    brier_week = week_blocked_bootstrap(
        paired, brier_metric_fn, block="week", samples=OPENER_SAMPLES, seed=OPENER_SEED
    )

    disagreements = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])]
    flip_buckets = line_flip_buckets(paired)

    row = open_week.iloc[0]
    summary = {
        "label": label,
        "baseline_profile": baseline_profile,
        "candidate_profile": candidate_profile,
        "timing_seconds": {"candidate": t2 - t1},
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "seasons": sorted(int(s) for s in paired["season"].unique()),
        "baseline_accuracy_open": float(paired["baseline_correct_open"].mean()),
        "candidate_accuracy_open": float(paired["candidate_correct_open"].mean()),
        "open_delta_week_blocked": row.to_dict(),
        "open_delta_season_blocked": open_season.iloc[0].to_dict(),
        "close_delta_week_blocked": close_week.iloc[0].to_dict(),
        "brier_logloss_week_blocked": brier_week.to_dict(orient="records"),
        "picks_disagreeing": len(disagreements),
        "disagreement_baseline_correct_open": (
            float(disagreements["baseline_correct_open"].mean()) if len(disagreements) else None
        ),
        "disagreement_candidate_correct_open": (
            float(disagreements["candidate_correct_open"].mean()) if len(disagreements) else None
        ),
        "flip_buckets": flip_buckets.to_dict(orient="records"),
    }
    return {"summary": summary, "paired": paired, "baseline": baseline, "candidate": candidate}


def nflverse_predictions(
    features: pd.DataFrame, profile: str, min_train_games: int
) -> pd.DataFrame:
    result = walk_forward_outcomes(
        features,
        start_season=NFLVERSE_START_SEASON,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=min_train_games,
        feature_profile=profile,
        methods=("market_residual",),
        ridge_alpha=RIDGE_ALPHA,
    )
    return result.predictions


def score_nflverse_pair(
    *, label: str, baseline_pred: pd.DataFrame, candidate_pred: pd.DataFrame
) -> dict[str, Any]:
    base_cover = baseline_pred.loc[
        baseline_pred["home_cover"].notna() & baseline_pred["home_cover_probability"].notna(),
        ["game_id", "season", "week", "home_cover", "home_cover_probability"],
    ].rename(columns={"home_cover_probability": "baseline_prob"})
    cand_cover = candidate_pred.loc[
        candidate_pred["home_cover"].notna() & candidate_pred["home_cover_probability"].notna(),
        ["game_id", "home_cover_probability"],
    ].rename(columns={"home_cover_probability": "candidate_prob"})
    merged = base_cover.merge(cand_cover, on="game_id", how="inner")
    merged["actual"] = merged["home_cover"].astype(float)
    merged["baseline_correct"] = (
        (merged["baseline_prob"] >= 0.5).astype(int) == merged["actual"].astype(int)
    ).astype(float)
    merged["candidate_correct"] = (
        (merged["candidate_prob"] >= 0.5).astype(int) == merged["actual"].astype(int)
    ).astype(float)

    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        actual = frame["actual"].to_numpy(dtype=float)
        cand = frame["candidate_prob"].to_numpy(dtype=float)
        base = frame["baseline_prob"].to_numpy(dtype=float)
        clip = lambda p: np.clip(p, 1e-15, 1.0 - 1e-15)  # noqa: E731
        return {
            "accuracy_delta": float(
                frame["candidate_correct"].mean() - frame["baseline_correct"].mean()
            ),
            "brier_improvement": float(
                np.mean(np.square(base - actual)) - np.mean(np.square(cand - actual))
            ),
            "log_loss_improvement": float(
                np.mean(-(actual * np.log(clip(base)) + (1 - actual) * np.log(1.0 - clip(base))))
                - np.mean(-(actual * np.log(clip(cand)) + (1 - actual) * np.log(1.0 - clip(cand))))
            ),
        }

    week_bootstrap = week_blocked_bootstrap(
        merged, metric_fn, block="week", samples=NFLVERSE_SAMPLES, seed=NFLVERSE_SEED
    )
    return {
        "label": label,
        "paired_games": len(merged),
        "baseline_accuracy": float(merged["baseline_correct"].mean()),
        "candidate_accuracy": float(merged["candidate_correct"].mean()),
        "week_blocked": week_bootstrap.to_dict(orient="records"),
    }


def main() -> None:
    started = time.time()
    print("=== Step 1: build penalty_discipline (season-lagged team rate) ===")
    snapshot = latest_pbp_snapshot(PBP_RAW_ROOT)
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    print(f"pbp snapshot={snapshot.snapshot_id} rows={len(pbp)}")
    rate = team_season_penalty_rate(pbp)
    _leak_safety_selfcheck(rate)

    print("\n=== Step 2: load feature tables, verify column-set claim, attach penalty ===")
    baseline_features = pd.read_parquet(BASELINE_FEATURES_PATH)
    narrowed_features = pd.read_parquet(NARROWED_FEATURES_PATH)
    baseline_rows, baseline_cols = len(baseline_features), len(baseline_features.columns)
    narrowed_rows, narrowed_cols = len(narrowed_features), len(narrowed_features.columns)
    print(f"baseline (learned severity) rows={baseline_rows} cols={baseline_cols}")
    print(f"narrowed (fixed severity) rows={narrowed_rows} cols={narrowed_cols}")

    register_weak_stack_v2_profile()

    baseline_plus_penalty = add_penalty_discipline_feature(baseline_features, rate)
    narrowed_plus_penalty = add_penalty_discipline_feature(narrowed_features, rate)
    coverage = float(baseline_plus_penalty["diff_penalty_rate_prior"].notna().mean())
    print(f"diff_penalty_rate_prior non-null coverage: {coverage:.3%}")

    out_dir = OUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"penalty_coverage": coverage, "arms": {}}

    print("\n=== Baseline (computed once, reused for every arm): active weak_stack ===")
    t_base0 = time.perf_counter()
    baseline = evaluate_arm(
        MARKET_ROOT,
        baseline_features,
        config=_config("weak_stack"),
        min_train_games=500,
    )
    t_base1 = time.perf_counter()
    print(f"baseline computed in {t_base1 - t_base0:.1f}s, {len(baseline)} scored games")

    print("\n=== Arm A: narrowed injury severity only (weak_stack, fixed vs learned) ===")
    arm_a = score_pair(
        label="A_narrowed_only",
        baseline=baseline,
        baseline_profile="weak_stack",
        candidate_features=narrowed_features,
        candidate_profile="weak_stack",
        min_train_games=500,
    )
    print(json.dumps(arm_a["summary"], indent=2, default=str))
    results["arms"]["A_narrowed_only"] = arm_a["summary"]
    arm_a["paired"].to_parquet(out_dir / "arm_A_paired.parquet")
    stamp_sidecar(out_dir / "arm_A_paired.parquet")  # ENG-38
    print(f"Elapsed so far: {time.time() - started:.1f}s")

    print("\n=== Arm B: v2 stack (narrowed + penalty) vs active baseline ===")
    arm_b = score_pair(
        label="B_v2_stack",
        baseline=baseline,
        baseline_profile="weak_stack",
        candidate_features=narrowed_plus_penalty,
        candidate_profile="weak_stack_v2",
        min_train_games=500,
    )
    print(json.dumps(arm_b["summary"], indent=2, default=str))
    results["arms"]["B_v2_stack"] = arm_b["summary"]
    arm_b["paired"].to_parquet(out_dir / "arm_B_paired.parquet")
    stamp_sidecar(out_dir / "arm_B_paired.parquet")  # ENG-38
    print(f"Elapsed so far: {time.time() - started:.1f}s")

    print("\n=== Arm C: penalty only (learned severity unchanged) vs active baseline ===")
    arm_c = score_pair(
        label="C_penalty_only",
        baseline=baseline,
        baseline_profile="weak_stack",
        candidate_features=baseline_plus_penalty,
        candidate_profile="weak_stack_v2",
        min_train_games=500,
    )
    print(json.dumps(arm_c["summary"], indent=2, default=str))
    results["arms"]["C_penalty_only"] = arm_c["summary"]
    arm_c["paired"].to_parquet(out_dir / "arm_C_paired.parquet")
    stamp_sidecar(out_dir / "arm_C_paired.parquet")  # ENG-38

    print("\n=== nflverse_spread grade (completeness): full 2018+ history ===")
    t_nf0 = time.perf_counter()
    nf_baseline_pred = nflverse_predictions(baseline_features, "weak_stack", 500)
    nf_a_pred = nflverse_predictions(narrowed_features, "weak_stack", 500)
    nf_b_pred = nflverse_predictions(narrowed_plus_penalty, "weak_stack_v2", 500)
    nf_c_pred = nflverse_predictions(baseline_plus_penalty, "weak_stack_v2", 500)
    print(f"nflverse predictions computed in {time.perf_counter() - t_nf0:.1f}s")

    nflverse_results = {
        "A_narrowed_only": score_nflverse_pair(
            label="A_narrowed_only", baseline_pred=nf_baseline_pred, candidate_pred=nf_a_pred
        ),
        "B_v2_stack": score_nflverse_pair(
            label="B_v2_stack", baseline_pred=nf_baseline_pred, candidate_pred=nf_b_pred
        ),
        "C_penalty_only": score_nflverse_pair(
            label="C_penalty_only", baseline_pred=nf_baseline_pred, candidate_pred=nf_c_pred
        ),
    }
    print(json.dumps(nflverse_results, indent=2, default=str))
    results["nflverse_grade"] = nflverse_results

    results["elapsed_seconds"] = time.time() - started
    write_stamped_artifact(results, out_dir / "summary.json")  # ENG-38
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()

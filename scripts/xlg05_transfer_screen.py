"""XLG-05: does a CFB-anchored PRIOR on the NFL residual model's coefficients help?

Predeclared in ``docs/xlg05_transfer_prior.md`` BEFORE this script was pointed
at an NFL outcome column. Read that document first -- it declares the shared
feature subset S, the four arms, the rotation family/window, the null, the
positive control, and the recording rules.

Four arms, one evaluator, one rotation-assigned window, close-graded. Every arm
is fit on the SAME 14-column design (``XLG05_FEATURE_COLUMNS``), at the SAME
frozen ``ridge_alpha = 10.0``, against the SAME ``market_residual`` target. No
arm sees a column another does not see. The only thing that varies is what the
ridge shrinks its coefficients toward, so this measures a MODEL change (a prior
on the estimator) and cannot be a team-quality-measurement change:

* ``nfl_only``        -- (a) plain ridge on NFL rows. THE BASELINE. Not full
  production: the point is the estimator, and comparing a transfer arm against
  the ~90-column production chain would confound "does transfer help" with "do
  we already have richer features".
* ``naive_pooled``    -- (b) one ridge on NFL + CFB rows stacked with a league
  indicator, scored at the NFL indicator value. The control.
* ``cfb_prior``       -- (c) ridge whose prior MEAN is the CFB-only fit
  ``theta_cfb`` (fit on ``y - X theta_cfb``, then add it back), alpha unchanged.
* ``partial_pooled``  -- (d) as (c) but toward ``kappa * theta_cfb`` with the
  prior strength ``kappa`` chosen by leave-one-season-out on TRAINING seasons
  only, over the frozen grid {0, .25, .5, .75, 1}. ``kappa=0`` is exactly (a)
  and ``kappa=1`` is exactly (c).

Reported beside them, never as a paired comparison arm:

* ``production``         -- full ``weak_stack`` on the same games, so the reader
  sees what the pool actually plays next to a 14-column research space.
* ``prior_market_only``  -- arm (d) with the prior forbidden from touching any
  team-quality coefficient. The explicit check of the owner's standing "team
  quality is already priced" bound. A correlated decomposition of the same
  window, so it is reported and NOT recorded to the signal registry.

Instrument checks that run BEFORE the window is spent (``--mode``):

* ``null``             -- settle margins shuffled within each week.
* ``positive-control`` -- the BASELINE arm's ``diff_off_epa_per_play`` is
  replaced by the realized ``ats_margin``, a deliberate leak in a team-quality
  column. The effect's sign is negative BY CONSTRUCTION (the leak is in the
  reference arm and the metric is candidate-minus-baseline); what it proves is
  magnitude-detection, not direction.
* ``screen``           -- the real look. Spends the family's assigned rotation
  window (``xlg05_transfer_prior``); run once, after both checks above.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands on
its own): an interval containing zero is NEVER grounds to reject, fail or close
an experiment. Only a RESOLVED wrong sign (whole interval on the wrong side of
zero), zero split-half reliability, or a positive control proven able to detect
an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``, never
the binary "contains zero". This run is CLOSE-graded, so per the binding "grade
the decision at the opener" rule it settles no play/no-play decision regardless
of sign -- every arm here is recorded ``unresolved_below_power``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from graph_input_screen import BOOTSTRAP_SAMPLES, SEED  # noqa: E402

from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES  # noqa: E402
from nfl_ats.cross_league_transfer import (  # noqa: E402
    fit_joint_league_model,
    fit_pooled_preprocessor,
    fit_prior_mean_ridge_model,
    fit_target_only_model,
)
from nfl_ats.margin import fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402
from nfl_ats.xlg05_transfer import (  # noqa: E402
    XLG05_FEATURE_COLUMNS,
    XLG05_PRIOR_STRENGTH_GRID,
    XLG05_TEAM_QUALITY_COLUMNS,
    fit_partially_pooled_model,
    prior_vector_stability,
    team_quality_mask,
)

ROTATION_FAMILY = "xlg05_transfer_prior"
RIDGE_ALPHA = 10.0
REGRESSOR = "ridge"
PRODUCTION_PROFILE = "weak_stack"
LEAK_COLUMN = "diff_off_epa_per_play"

DEFAULT_NFL_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
DEFAULT_CFB_FEATURES = REPO_ROOT / "data/processed/cfb_game_features.parquet"

BASELINE_ARM = "nfl_only"
#: Recorded to the weak-signal registry, one entry each, all vs the baseline.
CANDIDATE_ARMS: tuple[str, ...] = ("naive_pooled", "cfb_prior", "partial_pooled")
#: Reported in the write-up, deliberately NOT recorded (correlated decomposition
#: of the same window / a level rather than a paired comparison).
REPORTED_ARMS: tuple[str, ...] = ("prior_market_only", "production")
ALL_ARMS: tuple[str, ...] = (BASELINE_ARM, *CANDIDATE_ARMS, *REPORTED_ARMS)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _completed(frame: pd.DataFrame, *, regular_only: bool) -> pd.DataFrame:
    working = regular_season_rows(frame).copy() if regular_only else frame.copy()
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    keep = (
        pd.to_numeric(working["result"], errors="coerce").notna()
        & pd.to_numeric(working["ats_margin"], errors="coerce").notna()
    )
    return working.loc[keep].sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _leaked(frame: pd.DataFrame) -> pd.DataFrame:
    """The baseline arm's deliberate leak: one team-quality column becomes the outcome."""

    leaked = frame.copy()
    leaked[LEAK_COLUMN] = pd.to_numeric(leaked["ats_margin"], errors="coerce")
    return leaked


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def arm_names(frame: pd.DataFrame) -> tuple[str, ...]:
    """The arms actually present in a scored frame, read off its columns.

    Read rather than assumed so the grader and the bootstrap cannot silently
    disagree with what ``run_window`` was asked to fit.
    """

    suffix = "_probability"
    return tuple(
        str(column)[: -len(suffix)] for column in frame.columns if str(column).endswith(suffix)
    )


def run_window(
    nfl: pd.DataFrame,
    cfb: pd.DataFrame,
    seasons: tuple[int, ...],
    *,
    leak_treatment: bool = False,
    arms: tuple[str, ...] = ALL_ARMS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Close-graded per-game rows: the realized settle margin plus one home-cover
    probability per arm.

    Forward-chaining only -- every arm's week ``w`` model is trained strictly on
    games (in BOTH leagues) that kicked off before week ``w``'s earliest
    kickoff. Probabilities rather than correctness are returned because the null
    check re-grades the SAME fitted models against many permuted outcomes: model
    fitting never sees the grading outcome, so a permutation changes only the
    grade.

    The pooled preprocessor (one median imputer, one scaler, fit on the UNION of
    both leagues' rows) is fit ONCE, on rows strictly before the window's first
    game, so no scaling moment is computed from a scored week.
    """

    nfl_completed = _completed(nfl, regular_only=True)
    cfb_completed = _completed(cfb, regular_only=False)
    window = nfl_completed.loc[nfl_completed["season"].astype(int).isin(seasons)]
    if window.empty:
        raise ValueError(f"No completed NFL games in seasons {seasons}")

    baseline_source = _leaked(nfl_completed) if leak_treatment else nfl_completed
    cut = window["gameday"].min()
    nfl_pre = nfl_completed.loc[nfl_completed["gameday"].lt(cut)]
    cfb_pre = cfb_completed.loc[cfb_completed["gameday"].lt(cut)]
    baseline_pre = baseline_source.loc[baseline_source["gameday"].lt(cut)]
    if len(nfl_pre) < MIN_FITTABLE_TRAIN_GAMES or len(cfb_pre) < MIN_FITTABLE_TRAIN_GAMES:
        raise ValueError("Not enough pre-window history in both leagues to fit a preprocessor")

    shared_imputer, shared_scaler = fit_pooled_preprocessor(nfl_pre, cfb_pre, XLG05_FEATURE_COLUMNS)
    # Identical to the shared pair in every mode except positive-control, where
    # the baseline arm's design carries the leaked column and needs its own
    # honest standardisation.
    baseline_imputer, baseline_scaler = fit_pooled_preprocessor(
        baseline_pre, cfb_pre, XLG05_FEATURE_COLUMNS
    )

    quality_mask = team_quality_mask(XLG05_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    kappa_log: list[dict[str, Any]] = []
    coefficient_shift: list[dict[str, float]] = []
    n_weeks = 0
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        nfl_training = nfl_completed.loc[nfl_completed["gameday"].lt(cutoff)]
        cfb_training = cfb_completed.loc[cfb_completed["gameday"].lt(cutoff)]
        baseline_training = baseline_source.loc[baseline_source["gameday"].lt(cutoff)]
        if (
            len(nfl_training) < MIN_FITTABLE_TRAIN_GAMES
            or len(cfb_training) < MIN_FITTABLE_TRAIN_GAMES
        ):
            continue
        n_weeks += 1

        baseline_model = fit_target_only_model(
            baseline_training,
            baseline_imputer,
            baseline_scaler,
            XLG05_FEATURE_COLUMNS,
            RIDGE_ALPHA,
        )
        partial_model, selection = fit_partially_pooled_model(
            nfl_training,
            cfb_training,
            shared_imputer,
            shared_scaler,
            XLG05_FEATURE_COLUMNS,
            RIDGE_ALPHA,
            grid=XLG05_PRIOR_STRENGTH_GRID,
        )
        restricted_model, restricted_selection = fit_partially_pooled_model(
            nfl_training,
            cfb_training,
            shared_imputer,
            shared_scaler,
            XLG05_FEATURE_COLUMNS,
            RIDGE_ALPHA,
            grid=XLG05_PRIOR_STRENGTH_GRID,
            prior_mask=quality_mask,
            model_name="prior_market_only",
        )
        models: dict[str, Any] = {}
        for arm in arms:
            if arm == BASELINE_ARM:
                models[arm] = baseline_model
            elif arm == "partial_pooled":
                models[arm] = partial_model
            elif arm == "prior_market_only":
                models[arm] = restricted_model
            elif arm == "naive_pooled":
                models[arm] = fit_joint_league_model(
                    nfl_training,
                    cfb_training,
                    shared_imputer,
                    shared_scaler,
                    XLG05_FEATURE_COLUMNS,
                    RIDGE_ALPHA,
                )
            elif arm == "cfb_prior":
                models[arm] = fit_prior_mean_ridge_model(
                    nfl_training,
                    cfb_training,
                    shared_imputer,
                    shared_scaler,
                    XLG05_FEATURE_COLUMNS,
                    RIDGE_ALPHA,
                )
            elif arm == "production":
                models[arm] = fit_margin_model(
                    nfl_training,
                    target="market_residual",
                    model_name=REGRESSOR,
                    ridge_alpha=RIDGE_ALPHA,
                    feature_profile=PRODUCTION_PROFILE,
                )
            else:
                raise ValueError(f"Unknown arm {arm!r}; choose from {ALL_ARMS}")
        kappa_log.append(
            {
                "season": int(str(_season)),
                "week": int(str(_week)),
                "kappa": selection.kappa,
                "restricted_kappa": restricted_selection.kappa,
                "loso_fold_seasons": len(selection.fold_seasons),
                "used_fallback": selection.used_fallback,
                "train_games": len(nfl_training),
                "auxiliary_games": len(cfb_training),
            }
        )
        coefficient_shift.append(_coefficient_shift(baseline_model, partial_model, quality_mask))

        baseline_scoring = (
            _leaked(group.copy()) if leak_treatment else group  # leak only the baseline arm
        )
        probabilities = {
            arm: model.predict(baseline_scoring if arm == BASELINE_ARM else group)[
                "home_cover_probability"
            ].to_numpy(dtype=float)
            for arm, model in models.items()
        }
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        for position, (game_id, season_value, week_value, margin) in enumerate(
            zip(
                group["game_id"],
                group["season"],
                group["week"],
                settle_margin,
                strict=True,
            )
        ):
            row: dict[str, Any] = {
                "game_id": game_id,
                "season": int(str(season_value)),
                "week": int(str(week_value)),
                "settle_margin": margin,
            }
            for arm in arms:
                row[f"{arm}_probability"] = probabilities[arm][position]
            rows.append(row)

    print(f"  {n_weeks} weeks fitted over seasons {min(seasons)}-{max(seasons)}")
    diagnostics: dict[str, Any] = {
        "weeks_fitted": n_weeks,
        "prior_strength_by_week": kappa_log,
        "coefficient_shift_by_week": coefficient_shift,
    }
    if not leak_treatment:
        stability = prior_vector_stability(
            cfb_pre, shared_imputer, shared_scaler, XLG05_FEATURE_COLUMNS, RIDGE_ALPHA
        )
        diagnostics["prior_vector_stability"] = {
            "pearson_correlation": stability.pearson_correlation,
            "spearman_brown": stability.spearman_brown,
            "cosine_similarity": stability.cosine_similarity,
            "odd_season_rows": stability.odd_rows,
            "even_season_rows": stability.even_rows,
        }
    return pd.DataFrame(rows), diagnostics


def _coefficient_shift(baseline: Any, candidate: Any, quality_mask: np.ndarray) -> dict[str, float]:
    """L2 norm of ``theta_candidate - theta_baseline``, split by feature block.

    Answers "where did the estimator change actually move the model" rather than
    leaving it to inference (predeclaration section 6).
    """

    theta_a = np.asarray(baseline.estimator.coefficients, dtype=float)
    theta_d = np.asarray(candidate.estimator.coefficients, dtype=float)
    shift = theta_d - theta_a
    return {
        "shift_l2_total": float(np.linalg.norm(shift)),
        "shift_l2_team_quality": float(np.linalg.norm(shift[quality_mask])),
        "shift_l2_other": float(np.linalg.norm(shift[~quality_mask])),
    }


# ---------------------------------------------------------------------------
# Grading, null, bootstrap
# ---------------------------------------------------------------------------


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    """Attach ``<arm>_correct`` for every arm, graded against ``margins``."""

    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm in arm_names(frame):
        graded[f"{arm}_correct"] = pick_correct(graded[f"{arm}_probability"].ge(0.5), settle)
    return graded


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each week, computed once and reused by every permutation."""

    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    """Settle margins shuffled WITHIN each week.

    This null is deliberately NOT centred on zero: it preserves each week's
    realized home-cover rate, and the arms may carry different home-pick rates.
    """

    values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


# Two arms that make the SAME picks produce a paired delta of exactly zero in
# every resample, and ``week_blocked_bootstrap``'s ``probability_positive`` is
# ``mean(draws > 0)``, which reports that dead heat as 0.000 -- indistinguishable
# from a certain loss, and materially misleading for an EV decision where a tie
# is neither better nor worse. ``delta_is_tie`` exists so the same tool's own
# ``mean(draws > 0)`` machinery also returns the fraction of resamples that are
# exact ties, from which a tie-aware ``P+ = P(better) + 0.5 * P(tie)`` follows.
# The smallest non-zero |delta| this metric can take is 1/n (>= 1.3e-3 here), so
# the exact-zero test needs no tolerance argument.
_TIE_TOLERANCE = 1e-12


def _paired_metric(reference: str, candidate: str) -> Any:
    def metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=[reference, candidate])
        if valid.empty:
            return {
                "delta_accuracy": float("nan"),
                "delta_is_tie": float("nan"),
                "candidate_accuracy": float("nan"),
                "reference_accuracy": float("nan"),
            }
        delta = float((valid[candidate] - valid[reference]).mean())
        return {
            "delta_accuracy": delta,
            "delta_is_tie": 1.0 if abs(delta) < _TIE_TOLERANCE else -1.0,
            "candidate_accuracy": float(valid[candidate].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return metric


def null_distribution(
    frame: pd.DataFrame,
    arms: tuple[str, ...],
    *,
    permutations: int,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Every candidate arm's paired delta under many within-week permutations.

    One pass: each permutation re-grades all arms at once, so adding arms costs
    nothing beyond the grading itself and no extra model fit ever happens.
    """

    rng = np.random.default_rng(seed)
    groups = week_positions(frame)
    metrics = {arm: _paired_metric(f"{BASELINE_ARM}_correct", f"{arm}_correct") for arm in arms}
    draws: dict[str, list[float]] = {arm: [] for arm in arms}
    for _ in range(permutations):
        graded = grade(frame, permuted_margins(frame, rng, groups))
        for arm in arms:
            draws[arm].append(metrics[arm](graded)["delta_accuracy"])
    observed_frame = grade(frame)
    report: dict[str, dict[str, Any]] = {}
    for arm in arms:
        values = np.asarray(draws[arm], dtype=float)
        finite = values[np.isfinite(values)]
        observed = metrics[arm](observed_frame)["delta_accuracy"]
        report[arm] = {
            "permutations": len(finite),
            "null_mean_delta": float(finite.mean()),
            "null_sd_delta": float(finite.std(ddof=1)),
            "null_q025": float(np.quantile(finite, 0.025)),
            "null_q975": float(np.quantile(finite, 0.975)),
            "observed_delta": float(observed),
            "fraction_of_null_below_observed": float((finite < observed).mean()),
            # A percentile is meaningless against a point mass at zero (two arms
            # that make the same picks tie under every permutation too), so the
            # tie share is reported beside it rather than left to be inferred
            # from a 0.0 percentile that looks like an extreme tail.
            "fraction_of_null_tied_with_observed": float(
                (np.abs(finite - observed) < _TIE_TOLERANCE).mean()
            ),
        }
    return report


def summarize_pair(
    paired: pd.DataFrame, arm: str, samples: int, seed: int
) -> dict[str, Any] | None:
    """Point estimate plus week- and season-blocked bootstrap for one arm vs the baseline.

    Within-week game correlation is zero by owner mandate, so the week block is
    the honest primary and the season block is reported beside it, never
    averaged with it.
    """

    reference, candidate = f"{BASELINE_ARM}_correct", f"{arm}_correct"
    usable = paired.dropna(subset=[reference, candidate])
    if usable.empty:
        return None
    metric = _paired_metric(reference, candidate)
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)

    def _rows(table: pd.DataFrame) -> tuple[float, float, float, float]:
        delta = table.loc[table["metric"].eq("delta_accuracy")].iloc[0]
        tie = table.loc[table["metric"].eq("delta_is_tie")].iloc[0]
        better = float(delta["probability_positive"])
        tied = float(tie["probability_positive"])
        return float(delta["lower"]), float(delta["upper"]), better, tied

    week_low, week_high, week_better, week_tied = _rows(week)
    season_low, season_high, season_better, season_tied = _rows(season)

    # Deterministic sign-test counts, no resampling: how often the two arms
    # actually disagreed about a game. For near-identical arms these are the
    # honest read and the bootstrap interval is nearly a point mass.
    differing_picks = int(
        (
            usable[f"{arm}_probability"].ge(0.5) != usable[f"{BASELINE_ARM}_probability"].ge(0.5)
        ).sum()
    )
    candidate_better = int((usable[candidate] > usable[reference]).sum())
    baseline_better = int((usable[candidate] < usable[reference]).sum())
    return {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [week_low, week_high],
        "week_blocked_probability_positive": week_better,
        "week_blocked_probability_tie": week_tied,
        # A dead heat is neither better nor worse, so it splits: an exactly tied
        # arm reads 0.500 (EV-neutral) rather than 0.000 (a certain loss).
        "week_blocked_probability_positive_tie_aware": week_better + 0.5 * week_tied,
        "season_blocked_ci95": [season_low, season_high],
        "season_blocked_probability_positive": season_better,
        "season_blocked_probability_tie": season_tied,
        "season_blocked_probability_positive_tie_aware": season_better + 0.5 * season_tied,
        "n_games": len(usable),
        "n_picks_differing_from_baseline": differing_picks,
        "n_games_candidate_better": candidate_better,
        "n_games_baseline_better": baseline_better,
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
    }


# ---------------------------------------------------------------------------
# Window resolution -- read from the ledger, never hand-picked
# ---------------------------------------------------------------------------


def assigned_seasons(family: str = ROTATION_FAMILY) -> tuple[int, ...]:
    """The rotation-ASSIGNED window for ``family``, read from the ledger.

    Read rather than hardcoded so a hand-picked window is not expressible from
    this script. Falls back to a family's most recent spent window when the
    assignment has already been recorded, so reruns after ``rotation record``
    still reproduce the same block.
    """

    declared = load_registry().families.get(family)
    if declared is None:
        raise SystemExit(
            f"Rotation family {family!r} is not declared; run "
            f"'nfl-ats rotation declare --name {family} ...' then 'rotation assign' first."
        )
    window = declared.assigned_window or (declared.windows[-1] if declared.windows else None)
    if window is None:
        raise SystemExit(f"Rotation family {family!r} has no assigned window yet.")
    return tuple(window.covered_seasons)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("null", "positive-control", "screen"),
        required=True,
        help="null and positive-control are instrument checks; screen is the real look",
    )
    parser.add_argument(
        "--seasons",
        default="",
        help="inclusive season range override; defaults to the rotation-ASSIGNED window "
        f"for {ROTATION_FAMILY}, read from registry/rotation_registry.json",
    )
    parser.add_argument("--nfl-features", type=Path, default=DEFAULT_NFL_FEATURES)
    parser.add_argument("--cfb-features", type=Path, default=DEFAULT_CFB_FEATURES)
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="within-week permutations for the null (a single draw is not a test)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    if args.seasons:
        start, _, end = args.seasons.partition("-")
        seasons = tuple(range(int(start), int(end or start) + 1))
        window_source = "command line override"
    else:
        seasons = assigned_seasons()
        window_source = f"rotation assign ({ROTATION_FAMILY})"

    nfl = pd.read_parquet(args.nfl_features)
    cfb = pd.read_parquet(args.cfb_features)
    print(
        f"NFL table: {len(nfl)} rows, seasons {int(nfl['season'].min())}-"
        f"{int(nfl['season'].max())}; CFB table: {len(cfb)} rows, seasons "
        f"{int(cfb['season'].min())}-{int(cfb['season'].max())}"
    )
    print(f"window {min(seasons)}-{max(seasons)} from {window_source}")

    started = time.time()
    fitted, diagnostics = run_window(
        nfl, cfb, seasons, leak_treatment=args.mode == "positive-control"
    )

    result: dict[str, Any] = {"status": "no_scored_games"}
    present = arm_names(fitted) if not fitted.empty else ()
    scored_arms = tuple(arm for arm in (*CANDIDATE_ARMS, "prior_market_only") if arm in present)
    if not fitted.empty:
        if args.mode == "null":
            result = {
                "status": "scored",
                "null": null_distribution(
                    fitted, scored_arms, permutations=args.permutations, seed=args.seed
                ),
            }
        else:
            graded = grade(fitted)
            result = {
                "status": "scored",
                "home_pick_rate": {
                    arm: float(graded[f"{arm}_probability"].ge(0.5).mean()) for arm in present
                },
                "accuracy_level": {
                    arm: float(graded[f"{arm}_correct"].mean(skipna=True)) for arm in present
                },
                "permutation_null": null_distribution(
                    fitted, scored_arms, permutations=args.permutations, seed=args.seed
                ),
                "vs_baseline": {
                    arm: summarize_pair(graded, arm, args.bootstrap_samples, args.seed)
                    for arm in scored_arms
                },
            }

    configuration = {
        "mode": args.mode,
        "seasons": list(seasons),
        "window_source": window_source,
        "grade": "close",
        "rotation_family": ROTATION_FAMILY,
        "baseline_arm": BASELINE_ARM,
        "candidate_arms": list(CANDIDATE_ARMS),
        "reported_arms": list(REPORTED_ARMS),
        "feature_columns": list(XLG05_FEATURE_COLUMNS),
        "team_quality_columns": list(XLG05_TEAM_QUALITY_COLUMNS),
        "prior_strength_grid": list(XLG05_PRIOR_STRENGTH_GRID),
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "min_train_games": MIN_FITTABLE_TRAIN_GAMES,
        "production_profile": PRODUCTION_PROFILE,
        "leak_column": LEAK_COLUMN,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "permutations": args.permutations,
        "predeclaration": "docs/xlg05_transfer_prior.md",
        "nfl_features_path": str(args.nfl_features),
        "cfb_features_path": str(args.cfb_features),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "diagnostics": diagnostics,
        "result": result,
        "provenance": artifact_provenance(configuration, args.nfl_features, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO_ROOT / "artifacts" / (args.out or ROTATION_FAMILY) / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="xlg05-transfer-screen",
        metrics={"mode": args.mode, "status": result.get("status", "unknown")},
        rotation_family=ROTATION_FAMILY,
        notes=(
            "XLG-05: CFB-anchored prior on the NFL residual model's coefficients, four "
            "arms on one shared 14-column feature space; see docs/xlg05_transfer_prior.md."
        ),
    )
    print("wrote " + str(output_dir / "results.json"))
    _report(args, result, diagnostics)
    return 0


def _report(args: argparse.Namespace, result: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    if result.get("status") != "scored":
        return
    print()
    if args.mode == "null":
        print(
            f"NULL CHECK ({args.permutations} within-week permutations): the distribution "
            "must be centred near its own closed-form expectation, not necessarily zero."
        )
        for arm, null in result["null"].items():
            print(
                f"  {arm:>18s}: null mean {null['null_mean_delta'] * 100:+.3f} pts, sd "
                f"{null['null_sd_delta'] * 100:.3f}, 95% [{null['null_q025'] * 100:+.3f}, "
                f"{null['null_q975'] * 100:+.3f}], observed "
                f"{null['observed_delta'] * 100:+.3f}"
            )
        return

    print(f"accuracy levels ({args.mode}):")
    for arm, value in result["accuracy_level"].items():
        print(f"  {arm:>18s}: {value * 100:.2f}%   home-pick {result['home_pick_rate'][arm]:.3f}")
    print()
    print(f"paired vs {BASELINE_ARM}:")
    for arm, pair in result["vs_baseline"].items():
        if pair is None:
            continue
        low, high = pair["week_blocked_ci95"]
        null = result["permutation_null"][arm]
        print(
            f"  {arm:>18s}: {pair['delta_accuracy'] * 100:+.3f} pts  P+ "
            f"{pair['week_blocked_probability_positive']:.3f} (tie-aware "
            f"{pair['week_blocked_probability_positive_tie_aware']:.3f}, tie share "
            f"{pair['week_blocked_probability_tie']:.3f})  week 95% CI "
            f"[{low * 100:+.3f}, {high * 100:+.3f}]  null pct "
            f"{null['fraction_of_null_below_observed'] * 100:.1f}  n={pair['n_games']}, "
            f"picks differing {pair['n_picks_differing_from_baseline']} "
            f"({pair['n_games_candidate_better']} better / "
            f"{pair['n_games_baseline_better']} worse)"
        )
    kappa = [row["kappa"] for row in diagnostics["prior_strength_by_week"]]
    if kappa:
        counts = {value: kappa.count(value) for value in sorted(set(kappa))}
        fallbacks = sum(row["used_fallback"] for row in diagnostics["prior_strength_by_week"])
        print()
        print(f"selected prior strength kappa by week: {counts} (LOSO fallbacks: {fallbacks})")
    stability = diagnostics.get("prior_vector_stability")
    if stability:
        print(
            "prior-vector stability (odd vs even CFB seasons, NOT trait reliability): "
            f"r={stability['pearson_correlation']:.4f}, Spearman-Brown "
            f"{stability['spearman_brown']:.4f}, cosine {stability['cosine_similarity']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

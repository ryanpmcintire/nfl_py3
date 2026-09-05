"""LEAD-51: Splash-style confidence-point allocation, simulation only.

Predeclaration (frozen in ``docs/confidence_allocation_sim.md`` before this
script was ever run against the full archive): allocating confidence points
in proportion to edge, under calibration honesty, BEATS a flat/random
allocation on expected weekly points and on P(finish first) in a simulated
confidence pool.

This is a paper exercise on the project's own already-published opener
predictions -- it touches no rotation window, no weak-signal registry, and no
real-world outcome beyond ``artifacts/opener_evaluation/<stamp>/per_game.parquet``
(the frozen opener-evaluation archive; see ``nfl_ats.clv.opener_pick_evaluation``
for its schema). Every strategy below shares the SAME forced pick per game --
the model's probability-rule side (``pick_home_at_open_probability_rule``,
already the production forced-pick rule) -- and differs ONLY in how the
week's confidence points (1..n, distinct, highest to the most confident pick)
are allocated across that fixed set of picks. That is exactly the "Best Pick
lever" framing in ``docs/pool_edge_plan.md``: the side is already decided,
the open question is the ORDERING.

Five entrant strategies (docs/confidence_allocation_sim.md):
  - flat               random distinct ranks each week (control)
  - edge_proportional  rank by |home_cover_probability_at_open - 0.5|
  - calibrated_edge    same, after walk-forward isotonic calibration fitted
                        on strictly earlier weeks only (falls back to raw
                        edge_proportional before 200 prior graded games exist)
  - market_only        rank by |opener spread| (a known public heuristic)
  - oracle             rank by realized cover (positive control: must
                        dominate every other strategy every week, by
                        construction -- it is the score-maximizing
                        permutation for that week's fixed pick pattern)

Field model: N i.i.d. opponents (N in {20, 100, 500}) who each independently
pick the game's "public" side (the spread favorite) with probability
``p_favorite`` (measured, archive-wide, as the fraction of graded games where
the model's own forced pick already lands on the favorite) and rank their own
picks with an independent uniformly random permutation. Pick'em games
(opener spread exactly 0, no favorite) give every opponent a 50/50 coin flip
regardless of ``p_favorite``.

Per-week per-strategy scores are computed by EXACT enumeration wherever the
week's fixed pick pattern makes that possible (the 4 non-FLAT strategies'
weekly score is a single deterministic number; the field's per-entrant score
distribution and FLAT's own-score distribution are estimated via Monte Carlo
over random permutations/coin flips, MC_DRAWS per week, ONE fixed seed
consumed sequentially in chronological (season, week) order for full
determinism) -- then combined into P(finish first against N i.i.d. entrants),
including fractional credit for ties, via the closed-form binomial-tie sum
(the same "exact, not an approximation" philosophy `nfl_ats.pool` already
uses for its own field draws). See the module functions' docstrings for the
exact formula. This is NOT literal (samples x entrants x games) Monte Carlo
across the field (infeasible at N=500 without it); i.i.d. entrants make the
distribution of the field's maximum a pure function of one entrant's score
distribution and N.

Writes ``artifacts/confidence_allocation_sim/<run_id>/`` (``per_week.csv``,
``strategy_comparison.csv``, ``metadata.json`` via ``write_experiment_artifact``,
which also stamps ``registry/experiments/``). NOTHING is recorded to the
weak-signal registry (``nfl-ats weak-signals record``): its metrics
(expected points, P(finish first)) are not commensurable with any
``EFFECT_UNITS`` entry in ``nfl_ats.weak_signals`` (not ``accuracy_points``,
not a probability improvement on the same scale) -- AGENTS.md's pooling
discipline ("pooled inputs must be commensurable -- same units, same scale,
same population") applies just as much to a single recorded entry as to a
pool of several. The artifact and ``docs/confidence_allocation_sim.md`` are
the record.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_CALIBRATION_GAMES  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/confidence_allocation_sim"

#: A week must have at least this many *graded* (non-push) games to enter
#: the sample -- ROADMAP LEAD-51's frozen design threshold.
MIN_GRADED_GAMES = 8

#: Field sizes named in the frozen design (a Splash-style pool's plausible
#: field range; POL-05's own sweep goes wider but this experiment only needs
#: to show the allocation lever's sign is stable across scale).
FIELD_SIZES: tuple[int, ...] = (20, 100, 500)

#: Monte Carlo draws per week for the field's one-entrant score distribution
#: and FLAT's own-score distribution (both estimated; every other strategy's
#: weekly score is exact). i.i.d. entrants mean N never multiplies this cost.
MC_DRAWS = 20_000

#: One fixed seed, consumed sequentially over weeks in chronological order --
#: the whole run is reproducible from this single number (tests pin it).
SIM_SEED = 20260905

#: Outer week-blocked bootstrap over the per-week strategy-vs-FLAT deltas.
BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_CONFIDENCE = 0.95

CANDIDATE_STRATEGIES: tuple[str, ...] = (
    "edge_proportional",
    "calibrated_edge",
    "market_only",
    "oracle",
)
ALL_STRATEGIES: tuple[str, ...] = ("flat", *CANDIDATE_STRATEGIES)

REQUIRED_ARCHIVE_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "margin_vs_open",
        "home_cover_probability_at_open",
        "correct_at_open_probability_rule",
        "pick_home_at_open_probability_rule",
    }
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_archive(path: Path, *, min_graded_games: int = MIN_GRADED_GAMES) -> pd.DataFrame:
    """Load the frozen opener archive, drop pushes, keep only qualifying weeks.

    A push (``margin_vs_open == 0``) has no correct/incorrect pick to grade
    (``correct_at_open_probability_rule`` is NaN there -- see
    ``nfl_ats.clv.pick_correct``), so it is excluded from the week's confidence
    roster entirely rather than assigned a rank that can never score. Weeks
    with fewer than ``min_graded_games`` remaining games are dropped (none are,
    measured: min is 10 of 107 weeks in the frozen 20260819T174244Z archive).
    """

    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_ARCHIVE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Opener archive is missing columns: {', '.join(missing)}")

    graded = frame.dropna(subset=["correct_at_open_probability_rule"]).copy()
    graded["home_covered"] = graded["margin_vs_open"].gt(0.0)
    graded["favorite_side"] = np.select(
        [graded["tue_open_home_spread"].lt(0.0), graded["tue_open_home_spread"].gt(0.0)],
        ["HOME", "AWAY"],
        default="NONE",
    )
    graded = graded.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    counts = graded.groupby(["season", "week"])["game_id"].transform("size")
    graded = graded.loc[counts >= min_graded_games].reset_index(drop=True)
    if graded.empty:
        raise ValueError("No weeks meet the minimum graded-game threshold")
    return graded


def measure_favorite_share(graded: pd.DataFrame) -> float:
    """Archive-wide share of games where the model's own forced pick is the
    spread favorite -- the field model's ``p_favorite`` input (an assumption
    fed to the simulator, exactly like ``nfl_ats.pool.FieldModel.public_lean``'s
    default; not a leak, since it never touches which side WE pick)."""

    has_favorite = graded["favorite_side"].ne("NONE")
    if not bool(has_favorite.any()):
        raise ValueError("No games with a nonzero opener spread; cannot measure favorite share")
    picked_home = graded.loc[has_favorite, "pick_home_at_open_probability_rule"].to_numpy(
        dtype=bool
    )
    model_side = np.where(picked_home, "HOME", "AWAY")
    favorite = graded.loc[has_favorite, "favorite_side"].to_numpy()
    return float(np.mean(model_side == favorite))


# ---------------------------------------------------------------------------
# Walk-forward calibration (calibration honesty: strictly earlier weeks only)
# ---------------------------------------------------------------------------


def walk_forward_calibration(
    graded: pd.DataFrame, *, min_train: int = DEFAULT_MIN_CALIBRATION_GAMES
) -> pd.DataFrame:
    """Isotonic-calibrate ``home_cover_probability_at_open`` week by week.

    Week (season, week)'s calibrator is fit ONLY on graded games from
    strictly earlier (season, week) pairs (an expanding walk-forward window,
    the same discipline ``nfl_ats.calibration.calibrate_cover_prediction_stream``
    uses for the production probability stream). Weeks without at least
    ``min_train`` prior graded games, or whose prior history is single-class
    (isotonic regression needs both outcomes), keep the RAW probability and
    are flagged ``calibration_applied=False`` -- CALIBRATED_EDGE then reduces
    to EDGE_PROPORTIONAL for those weeks, which is disclosed, not silently
    smoothed over.
    """

    frame = graded.sort_values(["season", "week"]).reset_index(drop=True)
    season = frame["season"].to_numpy()
    week = frame["week"].to_numpy()
    raw_probability = frame["home_cover_probability_at_open"].to_numpy(dtype=float)
    outcome = frame["home_covered"].to_numpy(dtype=float)

    calibrated = raw_probability.copy()
    applied = np.zeros(len(frame), dtype=bool)

    week_keys = frame[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    for target_season, target_week in week_keys:
        train_mask = (season < target_season) | ((season == target_season) & (week < target_week))
        target_mask = (season == target_season) & (week == target_week)
        n_train = int(train_mask.sum())
        if n_train < min_train:
            continue
        train_outcome = outcome[train_mask]
        if len(np.unique(train_outcome)) < 2:
            continue
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(raw_probability[train_mask], train_outcome)
        predicted = model.predict(raw_probability[target_mask])
        calibrated[target_mask] = np.clip(predicted, 1e-6, 1.0 - 1e-6)
        applied[target_mask] = True

    frame["calibrated_probability_at_open"] = calibrated
    frame["calibration_applied"] = applied
    return frame


# ---------------------------------------------------------------------------
# Confidence-point assignment
# ---------------------------------------------------------------------------


def assign_points(confidence: np.ndarray, game_ids: np.ndarray) -> np.ndarray:
    """Distinct point values ``1..n``, with ``n`` (the most points) going to
    the highest-confidence game -- Splash-style confidence-pool scoring
    ("points for a correct pick equal its rank", rank read here as a point
    value, not `nfl_ats.pool.build_ats_pool_card`'s inverted display
    convention where rank 1 is the *most* confident pick).

    Ties broken by ``game_id`` ascending, the same deterministic tie-break
    `nfl_ats.pool.build_ats_pool_card` already uses.
    """

    n = len(confidence)
    if n == 0:
        return np.array([], dtype=int)
    order_frame = pd.DataFrame(
        {"confidence": np.asarray(confidence, dtype=float), "game_id": np.asarray(game_ids)}
    )
    order_frame["_position"] = np.arange(n)
    order_frame = order_frame.sort_values(["confidence", "game_id"], ascending=[False, True])
    points = np.empty(n, dtype=int)
    points[order_frame["_position"].to_numpy()] = np.arange(n, 0, -1)
    return points


# ---------------------------------------------------------------------------
# Per-week simulation
# ---------------------------------------------------------------------------


def _field_cdf_pmf_at(
    score: float, field_values: np.ndarray, field_pmf: np.ndarray, field_cdf: np.ndarray
) -> tuple[float, float]:
    below = field_cdf[field_values < score]
    q = float(below[-1]) if below.size else 0.0
    match = field_pmf[field_values == score]
    r = float(match[0]) if match.size else 0.0
    return q, r


def probability_first(
    score: float,
    entrants: int,
    field_values: np.ndarray,
    field_pmf: np.ndarray,
    field_cdf: np.ndarray,
) -> tuple[float, float]:
    """Exact P(outright first) and P(finish first, fractional tie credit)
    against ``entrants`` i.i.d. field draws from ``(field_values, field_pmf)``.

    Entrants are i.i.d., so P(all N below ``score``) = ``q**N`` (outright) and,
    writing ``total = q + r`` (P(field <= score)) and ``p = r / total`` (the
    tied-conditional-on-not-above share), the number of the N entrants tied
    with us given none exceeds us is Binomial(N, p); shared credit for a
    j-way tie is ``1 / (1 + j)``. Summing gives
    ``total**N * sum_j Binomial.pmf(j; N, p) / (1 + j)`` -- the same
    tie-credit convention `nfl_ats.pool.simulate_pool_finish` reports as
    ``probability_first``, computed here in closed form instead of by
    resampling ``entrants`` explicitly (which does not scale to N=500 once
    the score depends on which specific games are correct, not just how
    many -- see the module docstring).
    """

    q, r = _field_cdf_pmf_at(score, field_values, field_pmf, field_cdf)
    outright = q**entrants
    total = q + r
    if total <= 0.0:
        return outright, outright
    p = r / total
    j = np.arange(entrants + 1)
    tie_pmf = binom.pmf(j, entrants, p)
    shared = float((total**entrants) * np.sum(tie_pmf / (1.0 + j)))
    return outright, shared


def simulate_week(
    week_df: pd.DataFrame,
    *,
    p_favorite: float,
    field_sizes: tuple[int, ...],
    mc_draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """One week's rows: every strategy x every field size.

    ``week_df`` must already be restricted to one (season, week) and carry
    ``calibrated_probability_at_open`` (see :func:`walk_forward_calibration`).
    """

    n = len(week_df)
    game_ids = week_df["game_id"].to_numpy()
    our_correct = week_df["correct_at_open_probability_rule"].to_numpy(dtype=bool)
    k_ours = int(our_correct.sum())

    favorite_side = week_df["favorite_side"].to_numpy()
    home_covered = week_df["home_covered"].to_numpy(dtype=bool)
    favorite_covered = np.where(
        favorite_side == "HOME",
        home_covered,
        np.where(favorite_side == "AWAY", ~home_covered, False),
    )
    opponent_correct_probability = np.where(
        favorite_side == "NONE",
        0.5,
        np.where(favorite_covered, p_favorite, 1.0 - p_favorite),
    )

    # FLAT's own-score distribution: a random permutation of 1..n dotted with
    # our FIXED correctness pattern.
    own_permutations = rng.random((mc_draws, n)).argsort(axis=1) + 1
    flat_own_draws = (own_permutations * our_correct[None, :]).sum(axis=1)
    own_values, own_counts = np.unique(flat_own_draws, return_counts=True)
    own_pmf = own_counts / mc_draws

    # One field entrant's score distribution: independent per-game coin
    # flips at opponent_correct_probability, dotted with an independent
    # random permutation. i.i.d. entrants means this single distribution
    # (not an (entrants, mc_draws) array) is all any field size needs.
    field_hits = rng.random((mc_draws, n)) < opponent_correct_probability[None, :]
    field_permutations = rng.random((mc_draws, n)).argsort(axis=1) + 1
    field_draws = (field_permutations * field_hits).sum(axis=1)
    field_values, field_counts = np.unique(field_draws, return_counts=True)
    field_pmf = field_counts / mc_draws
    field_cdf = np.cumsum(field_pmf)

    rows: list[dict[str, Any]] = []

    deterministic_confidence = {
        "edge_proportional": (week_df["home_cover_probability_at_open"] - 0.5).abs().to_numpy(),
        "calibrated_edge": (week_df["calibrated_probability_at_open"] - 0.5).abs().to_numpy(),
        "market_only": week_df["tue_open_home_spread"].abs().to_numpy(),
        "oracle": our_correct.astype(float),
    }
    for name, confidence in deterministic_confidence.items():
        points = assign_points(confidence, game_ids)
        score = float(np.sum(points * our_correct))
        for entrants in field_sizes:
            outright, first = probability_first(score, entrants, field_values, field_pmf, field_cdf)
            rows.append(
                {
                    "strategy": name,
                    "field_size": entrants,
                    "expected_points": score,
                    "probability_first": first,
                    "probability_outright": outright,
                }
            )

    expected_flat = k_ours * (n + 1) / 2.0
    for entrants in field_sizes:
        total_first = 0.0
        total_outright = 0.0
        for value, weight in zip(own_values, own_pmf, strict=True):
            outright, first = probability_first(
                float(value), entrants, field_values, field_pmf, field_cdf
            )
            total_first += weight * first
            total_outright += weight * outright
        rows.append(
            {
                "strategy": "flat",
                "field_size": entrants,
                "expected_points": expected_flat,
                "probability_first": total_first,
                "probability_outright": total_outright,
            }
        )

    return rows


def run_simulation(
    graded: pd.DataFrame,
    *,
    p_favorite: float,
    field_sizes: tuple[int, ...] = FIELD_SIZES,
    mc_draws: int = MC_DRAWS,
    seed: int = SIM_SEED,
    min_train: int = DEFAULT_MIN_CALIBRATION_GAMES,
) -> pd.DataFrame:
    """The full per-week x per-strategy x per-field-size table.

    One shared ``rng`` is consumed sequentially over weeks in ascending
    (season, week) order, so the entire run is reproducible from ``seed``
    alone.
    """

    calibrated = walk_forward_calibration(graded, min_train=min_train)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, week_df in calibrated.groupby(["season", "week"], sort=True):
        week_df = week_df.reset_index(drop=True)
        n = len(week_df)
        season_value = int(week_df["season"].iloc[0])
        week_value = int(week_df["week"].iloc[0])
        k = int(week_df["correct_at_open_probability_rule"].sum())
        calibration_applied = bool(week_df["calibration_applied"].iloc[0])
        for row in simulate_week(
            week_df,
            p_favorite=p_favorite,
            field_sizes=field_sizes,
            mc_draws=mc_draws,
            rng=rng,
        ):
            row.update(
                {
                    "season": season_value,
                    "week": week_value,
                    "n_games": n,
                    "k_correct": k,
                    "calibration_applied": calibration_applied,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-week comparison: paired difference vs FLAT, week-blocked bootstrap
# ---------------------------------------------------------------------------


def era_scope_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Two halves of the archive's own week list (chronological), used
    because the frozen opener archive spans 2020-2025 only -- it does not
    reach back to 2009-2017, so LEAD-51's per-era requirement falls back to
    its own stated alternative: the two halves of the archive actually used.
    """

    weeks = rows[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    weeks = weeks.reset_index(drop=True)
    midpoint = -(-len(weeks) // 2)  # ceil division: first half gets the extra week if odd
    weeks["era"] = np.where(weeks.index < midpoint, "first_half", "second_half")
    return rows.merge(weeks, on=["season", "week"], how="left")


def compare_strategies(
    rows: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    field_sizes: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Paired candidate-minus-FLAT deltas, week-blocked bootstrapped, for
    every candidate strategy x metric x scope (overall + both archive halves).

    ``field_sizes`` defaults to the distinct values actually present in
    ``rows`` (derived, not the module's ``FIELD_SIZES`` constant) so this
    function works unchanged on a synthetic test table built with different
    field sizes.
    """

    sizes = field_sizes if field_sizes is not None else tuple(sorted(rows["field_size"].unique()))
    labeled = era_scope_labels(rows)

    points_source = labeled.drop_duplicates(["season", "week", "strategy"])
    points_wide = points_source.pivot(
        index=["season", "week", "era"], columns="strategy", values="expected_points"
    ).reset_index()

    first_source = labeled[["season", "week", "era", "strategy", "field_size", "probability_first"]]
    first_wide = first_source.pivot_table(
        index=["season", "week", "era", "field_size"],
        columns="strategy",
        values="probability_first",
    ).reset_index()

    results: list[dict[str, Any]] = []
    for strategy in CANDIDATE_STRATEGIES:
        for scope_name in ("overall", "first_half", "second_half"):
            era_filter = None if scope_name == "overall" else scope_name

            points_scope = (
                points_wide
                if era_filter is None
                else points_wide.loc[points_wide["era"].eq(era_filter)]
            )
            points_frame = points_scope[["season", "week"]].assign(
                diff=points_scope[strategy] - points_scope["flat"]
            )
            points_result = week_blocked_bootstrap(
                points_frame,
                lambda frame: {"diff_mean": float(frame["diff"].mean())},
                samples=samples,
                seed=seed,
                confidence=confidence,
            ).iloc[0]
            results.append(
                {
                    "strategy": strategy,
                    "scope": scope_name,
                    "metric": "expected_points",
                    "field_size": None,
                    "n_weeks": len(points_frame),
                    "estimate": float(points_result["estimate"]),
                    "lower": float(points_result["lower"]),
                    "upper": float(points_result["upper"]),
                    "probability_positive": float(points_result["probability_positive"]),
                }
            )

            for entrants in sizes:
                first_scope = first_wide.loc[first_wide["field_size"].eq(entrants)]
                if era_filter is not None:
                    first_scope = first_scope.loc[first_scope["era"].eq(era_filter)]
                first_frame = first_scope[["season", "week"]].assign(
                    diff=first_scope[strategy] - first_scope["flat"]
                )
                first_result = week_blocked_bootstrap(
                    first_frame,
                    lambda frame: {"diff_mean": float(frame["diff"].mean())},
                    samples=samples,
                    seed=seed,
                    confidence=confidence,
                ).iloc[0]
                results.append(
                    {
                        "strategy": strategy,
                        "scope": scope_name,
                        "metric": "probability_first",
                        "field_size": entrants,
                        "n_weeks": len(first_frame),
                        "estimate": float(first_result["estimate"]),
                        "lower": float(first_result["lower"]),
                        "upper": float(first_result["upper"]),
                        "probability_positive": float(first_result["probability_positive"]),
                    }
                )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    parser.add_argument("--seed", type=int, default=SIM_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    started = time.time()
    graded = load_archive(args.archive)
    p_favorite = measure_favorite_share(graded)
    n_weeks = graded[["season", "week"]].drop_duplicates().shape[0]
    print(f"weeks={n_weeks} graded_games={len(graded)} p_favorite={p_favorite:.4f}")

    rows = run_simulation(
        graded,
        p_favorite=p_favorite,
        field_sizes=FIELD_SIZES,
        mc_draws=args.mc_draws,
        seed=args.seed,
    )
    comparison = compare_strategies(rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed)

    for _, row in comparison.loc[comparison["scope"].eq("overall")].iterrows():
        field = "-" if pd.isna(row["field_size"]) else int(row["field_size"])
        print(
            f"{row['strategy']:>18s} {row['metric']:>18s} field={field!s:>4s} "
            f"delta={row['estimate']:+.5f} 95%[{row['lower']:+.5f},{row['upper']:+.5f}] "
            f"P+={row['probability_positive']:.4f}"
        )

    # Positive-control self-check: oracle must weakly dominate every other
    # strategy's weekly expected_points, every week (it assigns the largest
    # available point values to the fixed set of correct picks, which is the
    # score-maximizing permutation for that week by construction).
    points_by_week = rows.drop_duplicates(["season", "week", "strategy"]).pivot(
        index=["season", "week"], columns="strategy", values="expected_points"
    )
    oracle_dominates = all(
        bool((points_by_week["oracle"] >= points_by_week[other] - 1e-9).all())
        for other in ALL_STRATEGIES
        if other != "oracle"
    )
    print(f"oracle_dominates_every_week={oracle_dominates}")
    if not oracle_dominates:
        raise RuntimeError(
            "Positive-control failure: oracle did not dominate every week; "
            "the simulator is blind and its results are not trustworthy."
        )

    run_dir_name = args.out or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = DEFAULT_OUTPUT_ROOT / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows.to_csv(output_dir / "per_week.csv", index=False)
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)

    configuration = {
        "archive": str(args.archive),
        "min_graded_games": MIN_GRADED_GAMES,
        "field_sizes": list(FIELD_SIZES),
        "mc_draws": args.mc_draws,
        "sim_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "min_calibration_games": DEFAULT_MIN_CALIBRATION_GAMES,
        "p_favorite_measured": p_favorite,
        "n_weeks": n_weeks,
        "n_graded_games": len(graded),
        "strategies": list(ALL_STRATEGIES),
    }
    metadata = {
        "command": "confidence-allocation-sim",
        "predeclaration": "docs/confidence_allocation_sim.md",
        "oracle_dominates_every_week": oracle_dominates,
        "comparison_overall": comparison.loc[comparison["scope"].eq("overall")].to_dict(
            orient="records"
        ),
        "elapsed_seconds": time.time() - started,
        "provenance": artifact_provenance(configuration, args.archive, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="confidence-allocation-sim",
        metrics={"comparison": comparison.to_dict(orient="records")},
        notes=(
            "LEAD-51 confidence-point allocation simulation; "
            "docs/confidence_allocation_sim.md. NOT recorded to the "
            "weak-signal registry -- expected points / P(first) are not "
            "commensurable with any nfl_ats.weak_signals.EFFECT_UNITS entry."
        ),
        source="docs/confidence_allocation_sim.md",
        project_root=REPO_ROOT,
    )
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

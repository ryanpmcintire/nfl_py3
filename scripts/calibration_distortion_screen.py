"""Does the residual/calibration step distort effects measured the way this project measures them?

Runs the gating experiment for ``docs/revisit_list.md`` defect **D1**. The
recorded evidence for D1 is a LEVEL measured on a plant that overwrote the
target (``purged_cv.inject_synthetic_signal``): planted +1.3 accuracy points
came back -0.7, planted +3.0 came back +1.14, while a sign-only ablation --
same purged/embargoed training set, same ridge fit, no residual-location
read -- recovered both closely (``docs/purged_cv.md``, positive control).
Almost every verdict in this project is instead a PAIRED DELTA between two
arms scored on the same games and the same folds, so this script measures the
distortion in that estimator too, and on plants that do not destroy the real
target.

Four stages, all CFB (rotation-registry rule 8: CFB costs no NFL window;
``registry/`` is never read or written here):

* ``overwrite`` -- the existing positive control, re-run as a magnitude sweep
  and scored BOTH as a level and as a paired delta against a same-folds
  baseline arm.
* ``additive`` -- the same magnitudes planted ADDITIVELY on the real
  ``ats_margin`` (``calibration_distortion.plant_additive_effect``), through an
  orthogonal ``{-1,+1}`` carrier and through several REAL, collinear feature
  columns. This is the bridge that separates "the calibration step distorts
  real effects" from "the plant was orthogonal to everything".
* ``real`` -- real feature-contract and ridge-penalty contrasts, no planting,
  each scored under both estimators on identical folds, then regressed against
  each other.
* ``mechanism`` -- sweeps ``distribution_fraction``, design width and
  ``n_blocks`` to say whether any distortion found is fixable or only
  detectable.

What a result here does NOT license: this script does not re-classify any
registry entry, does not touch ``artifacts/`` or ``registry/``, does not move
a pick or a feature profile, and does not measure NFL. Reader-only candidate
families (``residual_location``, ``ecdf_smoothing``) are deliberately absent
from the ``real`` stage -- they hold the mean model fixed, so the sign-only
estimator is identical in both of their arms by construction and cannot
cross-check them at all.

    ./.tools/uv.exe run --no-sync python scripts/calibration_distortion_screen.py --stage all
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nfl_ats.calibration_distortion import (
    additive_plant_gamma,
    full_pipeline_home_pick,
    implied_pick_threshold,
    orthogonal_carrier,
    plant_additive_effect,
    sign_only_home_pick,
    standardized_carrier,
)
from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import (
    CFB_CONTEXT_FEATURES,
    CFB_EXPERIENCE_FEATURES,
    CFB_MARKET_FEATURES,
    CFB_MODEL_FEATURE_COLUMNS,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact
from nfl_ats.purged_cv import (
    DEFAULT_EMBARGO_WEEKS,
    DEFAULT_PURGE_WEEKS,
    PurgedFold,
    assign_week_order,
    inject_synthetic_signal,
    purged_embargoed_folds,
)

DATA_PATH = Path("data/processed/cfb_game_features.parquet")
DEFAULT_OUTPUT = Path("artifacts/calibration_distortion")

BASE_COLUMNS: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS
THIN_COLUMNS: tuple[str, ...] = (
    *CFB_MARKET_FEATURES,
    *CFB_CONTEXT_FEATURES,
    *CFB_EXPERIENCE_FEATURES,
)
STATE_COLUMNS: tuple[str, ...] = tuple(c for c in BASE_COLUMNS if c not in THIN_COLUMNS)

#: n_blocks=40 is the setting the recorded positive control used
#: (``scripts/purged_validate.py``), so the overwrite stage reproduces it
#: rather than measuring a different configuration.
DEFAULT_N_BLOCKS = 40

#: Planted magnitudes, expressed as the population forced-pick accuracy of the
#: carrier ("+1.3 points" == 0.513). Spans the scale of every effect this
#: project tracks, from below its resolution to well above it.
MAGNITUDES: tuple[float, ...] = (0.505, 0.510, 0.513, 0.520, 0.530)

#: Real carrier columns for the collinear plant, chosen to span the R^2 of a
#: column against the rest of the standardised design (measured on the full
#: 12,500-game table: 0.026 / 0.230 / 0.431 / 0.755 / 0.868). The team-state
#: block is excluded on purpose: every one of its columns sits above R^2 0.997
#: (``diff_x`` is ``home_x - away_x`` by construction), so dropping one from
#: the baseline arm removes no information and the contrast is degenerate.
REAL_CARRIERS: tuple[str, ...] = (
    "rest_diff",
    "week_cos",
    "conference_game",
    "week_sin",
    "home_team_games",
)

PLANT_COLUMN = "planted_carrier"


# ---------------------------------------------------------------------------
# Loading and folds
# ---------------------------------------------------------------------------


def load_completed() -> pd.DataFrame:
    """Completed CFB games with a ``week_order``, sorted the way folds index them."""

    frame = pd.read_parquet(DATA_PATH)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    return (
        assign_week_order(completed).sort_values(["week_order", "game_id"]).reset_index(drop=True)
    )


def build_folds(completed: pd.DataFrame, *, n_blocks: int) -> list[PurgedFold]:
    return purged_embargoed_folds(
        completed,
        n_blocks=n_blocks,
        purge_weeks=DEFAULT_PURGE_WEEKS,
        embargo_weeks=DEFAULT_EMBARGO_WEEKS,
    )


# ---------------------------------------------------------------------------
# One arm = one feature contract + one ridge configuration, scored on every fold
# ---------------------------------------------------------------------------


def score_arm(
    completed: pd.DataFrame,
    folds: list[PurgedFold],
    feature_columns: tuple[str, ...],
    *,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    distribution_fraction: float = CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """Fit the frozen CFB residual recipe on each purged fold; return both picks per game.

    ``full_home`` is the production forced pick and ``sign_home`` the
    sign-only ablation, from the SAME fitted model -- the two differ only by
    the residual sample's location, recorded per fold as ``threshold``.
    """

    parts: list[pd.DataFrame] = []
    for fold in folds:
        train = completed.iloc[fold.train_index]
        test = completed.iloc[fold.test_index]
        if len(train) < min_train_games or test.empty:
            continue
        model = fit_cfb_residual_model(
            train,
            ridge_alpha=ridge_alpha,
            distribution_fraction=distribution_fraction,
            feature_columns=feature_columns,
        )
        if model.estimator is None:  # pragma: no cover - fit_cfb_residual_model always sets it
            raise RuntimeError("fitted CFB residual model has no estimator")
        predicted_residual = np.asarray(
            model.estimator.predict(test.loc[:, list(feature_columns)]), dtype=float
        )
        parts.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "week": test["week"].to_numpy(),
                    "row_position": fold.test_index,
                    "home_cover": pd.to_numeric(test["home_cover"], errors="coerce").to_numpy(),
                    "predicted_residual": predicted_residual,
                    "full_home": full_pipeline_home_pick(model.residuals, predicted_residual),
                    "sign_home": sign_only_home_pick(predicted_residual),
                    "path_id": fold.path_id,
                    "threshold": implied_pick_threshold(model.residuals),
                    "residual_mean": float(np.mean(model.residuals)),
                    "n_residuals": int(model.residuals.size),
                }
            )
        )
    if not parts:
        raise ValueError("no fold produced a scored arm")
    scored = pd.concat(parts, ignore_index=True).sort_values("game_id").reset_index(drop=True)
    return scored


def _correct(scored: pd.DataFrame, pick_column: str) -> np.ndarray:
    picks = scored[pick_column].to_numpy(dtype=bool).astype(float)
    return np.asarray(picks == scored["home_cover"].to_numpy(dtype=float), dtype=float)


def arm_accuracy(scored: pd.DataFrame, pick_column: str) -> float:
    mask = scored["home_cover"].notna().to_numpy()
    return float(_correct(scored, pick_column)[mask].mean())


def paired_delta(
    baseline: pd.DataFrame, candidate: pd.DataFrame, pick_column: str
) -> tuple[float, float]:
    """Paired accuracy delta (points) and disagreement rate ``f`` between two arms.

    Mirrors ``experiments._paired_row_improvements``' accuracy metric: the mean
    over games of ``candidate_correct - baseline_correct``, on identical games.
    """

    if not baseline["game_id"].equals(candidate["game_id"]):
        raise ValueError("paired arms must cover the identical games in the identical order")
    mask = baseline["home_cover"].notna().to_numpy()
    delta = (_correct(candidate, pick_column) - _correct(baseline, pick_column))[mask]
    flips = (
        baseline[pick_column].to_numpy(dtype=bool) != candidate[pick_column].to_numpy(dtype=bool)
    )[mask]
    return 100.0 * float(delta.mean()), float(flips.mean())


def oracle_delta(
    baseline: pd.DataFrame,
    carrier: np.ndarray,
    gamma: float,
    pick_column: str,
) -> float:
    """Best achievable paired delta if the planted coefficient were known exactly.

    Takes the baseline arm's own predicted residual and adds ``gamma *
    carrier`` -- the planted term -- keeping the baseline arm's own residual
    location as the threshold for the ``full_home`` framing and zero for
    ``sign_home``. This is the delta a perfect estimator would produce IN THE
    SAME FRAME the estimator is being judged in, so a ratio against it isolates
    estimation/calibration loss from the frame's own properties. Exact only
    when the carrier is orthogonal to the baseline design; for a collinear
    carrier the baseline arm partially reconstructs the planted term, which
    inflates this oracle for BOTH estimators equally.
    """

    predicted = baseline["predicted_residual"].to_numpy(dtype=float)
    boosted = predicted + gamma * carrier[baseline["row_position"].to_numpy()]
    threshold = (
        baseline["threshold"].to_numpy(dtype=float)
        if pick_column == "full_home"
        else np.zeros(len(baseline))
    )
    mask = baseline["home_cover"].notna().to_numpy()
    actual = baseline["home_cover"].to_numpy(dtype=float)
    base_correct = np.asarray((predicted > threshold).astype(float) == actual, dtype=float)
    boosted_correct = np.asarray((boosted > threshold).astype(float) == actual, dtype=float)
    return 100.0 * float((boosted_correct[mask] - base_correct[mask]).mean())


def fold_diagnostics(scored: pd.DataFrame) -> dict[str, float]:
    """Per-fold location size relative to the spread of predicted residuals."""

    per_fold = scored.groupby("path_id").agg(
        threshold=("threshold", "first"),
        pr_sd=("predicted_residual", "std"),
    )
    ratio = per_fold["threshold"].abs() / per_fold["pr_sd"]
    return {
        "mean_abs_threshold": float(per_fold["threshold"].abs().mean()),
        "sd_threshold": float(per_fold["threshold"].std()),
        "mean_pr_sd": float(per_fold["pr_sd"].mean()),
        "mean_abs_threshold_over_pr_sd": float(ratio.mean()),
    }


# ---------------------------------------------------------------------------
# Stage 1: the recorded positive control, as a level AND as a paired delta
# ---------------------------------------------------------------------------


def stage_overwrite(
    completed: pd.DataFrame,
    folds: list[PurgedFold],
    *,
    replicates: int,
    magnitudes: tuple[float, ...] = MAGNITUDES,
    feature_columns: tuple[str, ...] = BASE_COLUMNS,
    label: str = "full_design",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        for replicate in range(replicates):
            seed = 700_000 + 1_000 * round(magnitude * 1000) + replicate
            injected = inject_synthetic_signal(
                completed, target_accuracy=magnitude, seed=seed, column=PLANT_COLUMN
            )
            population = float(
                (
                    np.sign(injected[PLANT_COLUMN].to_numpy())
                    == np.sign(injected["ats_margin"].to_numpy())
                ).mean()
            )
            candidate = score_arm(injected, folds, (*feature_columns, PLANT_COLUMN))
            baseline = score_arm(injected, folds, feature_columns)
            row: dict[str, Any] = {
                "design": label,
                "magnitude": magnitude,
                "replicate": replicate,
                "population_accuracy": 100.0 * population,
                "truth_level_points": 100.0 * (population - 0.5),
            }
            for pick in ("full_home", "sign_home"):
                key = "full" if pick == "full_home" else "sign"
                row[f"level_{key}_candidate"] = 100.0 * arm_accuracy(candidate, pick)
                row[f"level_{key}_baseline"] = 100.0 * arm_accuracy(baseline, pick)
                delta, flips = paired_delta(baseline, candidate, pick)
                row[f"delta_{key}"] = delta
                row[f"flip_rate_{key}"] = flips
            row.update({f"cand_{k}": v for k, v in fold_diagnostics(candidate).items()})
            rows.append(row)
            print(
                f"  overwrite[{label}] mag={magnitude:.3f} rep={replicate} "
                f"truth={row['truth_level_points']:+.2f} "
                f"level_full={row['level_full_candidate'] - 50.0:+.2f} "
                f"level_sign={row['level_sign_candidate'] - 50.0:+.2f} "
                f"delta_full={row['delta_full']:+.2f} delta_sign={row['delta_sign']:+.2f}",
                flush=True,
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 2: additive plants that leave the real target in place
# ---------------------------------------------------------------------------


def stage_additive(
    completed: pd.DataFrame,
    folds: list[PurgedFold],
    *,
    replicates: int,
    magnitudes: tuple[float, ...] = MAGNITUDES,
) -> pd.DataFrame:
    noise_std = float(pd.to_numeric(completed["ats_margin"], errors="coerce").std())
    specs: list[tuple[str, str | None, int]] = [
        ("orthogonal", None, replicate) for replicate in range(replicates)
    ]
    specs += [(f"real:{column}", column, 0) for column in REAL_CARRIERS]

    rows: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        gamma = additive_plant_gamma(magnitude, noise_std)
        for carrier_label, column, replicate in specs:
            if column is None:
                carrier = orthogonal_carrier(
                    len(completed), seed=800_000 + 1_000 * round(magnitude * 1000) + replicate
                )
                planted = plant_additive_effect(
                    completed, carrier=carrier, gamma=gamma, column=PLANT_COLUMN
                )
                candidate_columns = (*BASE_COLUMNS, PLANT_COLUMN)
                baseline_columns = BASE_COLUMNS
            else:
                carrier = standardized_carrier(completed, column)
                planted = plant_additive_effect(completed, carrier=carrier, gamma=gamma)
                candidate_columns = BASE_COLUMNS
                baseline_columns = tuple(c for c in BASE_COLUMNS if c != column)
            candidate = score_arm(planted, folds, candidate_columns)
            baseline = score_arm(planted, folds, baseline_columns)
            row: dict[str, Any] = {
                "carrier": carrier_label,
                "carrier_kind": "orthogonal" if column is None else "collinear_real",
                "magnitude": magnitude,
                "replicate": replicate,
                "gamma": gamma,
            }
            for pick in ("full_home", "sign_home"):
                key = "full" if pick == "full_home" else "sign"
                delta, flips = paired_delta(baseline, candidate, pick)
                row[f"delta_{key}"] = delta
                row[f"flip_rate_{key}"] = flips
                row[f"oracle_{key}"] = oracle_delta(baseline, carrier, gamma, pick)
                row[f"level_{key}_baseline"] = 100.0 * arm_accuracy(baseline, pick)
                row[f"level_{key}_candidate"] = 100.0 * arm_accuracy(candidate, pick)
            row.update({f"cand_{k}": v for k, v in fold_diagnostics(candidate).items()})
            rows.append(row)
            print(
                f"  additive[{carrier_label}] mag={magnitude:.3f} rep={replicate} "
                f"delta_full={row['delta_full']:+.2f} (oracle {row['oracle_full']:+.2f}) "
                f"delta_sign={row['delta_sign']:+.2f} (oracle {row['oracle_sign']:+.2f})",
                flush=True,
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3: real contrasts, both estimators, identical folds
# ---------------------------------------------------------------------------


def real_contrast_specs() -> dict[str, dict[str, Any]]:
    """Mean-model contrasts only.

    A contrast that changes only how a FIXED residual sample is read (recency
    weighting, location shrinkage, ECDF smoothing) has an identical sign-only
    arm on both sides by construction, so it carries exactly zero information
    about whether the two estimators agree. Those families are excluded rather
    than run and reported as "agreeing".
    """

    off = tuple(c for c in STATE_COLUMNS if "_off_" in c)
    dfn = tuple(c for c in STATE_COLUMNS if "_def_" in c)
    diff = tuple(c for c in STATE_COLUMNS if c.startswith("diff_"))
    epa = tuple(c for c in STATE_COLUMNS if c.endswith("epa_per_play"))
    return {
        "thin_no_team_state": {"columns": THIN_COLUMNS},
        "drop_diff_block": {"columns": tuple(c for c in BASE_COLUMNS if c not in diff)},
        "diff_only_state": {"columns": (*THIN_COLUMNS, *diff)},
        "drop_offense_state": {"columns": tuple(c for c in BASE_COLUMNS if c not in off)},
        "drop_defense_state": {"columns": tuple(c for c in BASE_COLUMNS if c not in dfn)},
        "drop_epa_metrics": {"columns": tuple(c for c in BASE_COLUMNS if c not in epa)},
        "drop_context": {
            "columns": tuple(c for c in BASE_COLUMNS if c not in CFB_CONTEXT_FEATURES)
        },
        "drop_experience": {
            "columns": tuple(c for c in BASE_COLUMNS if c not in CFB_EXPERIENCE_FEATURES)
        },
        "drop_total_line": {"columns": tuple(c for c in BASE_COLUMNS if c != "total_line")},
        # Three deliberately LARGE contrasts. Without them every contrast in
        # this list sits inside the evaluator's own resolution, the regression
        # of one estimator on the other has no signal to fit, and a sign
        # disagreement between two readings of zero would be misread as
        # distortion. These give the regression a range to work over.
        "drop_spread_line": {"columns": tuple(c for c in BASE_COLUMNS if c != "spread_line")},
        "market_columns_only": {"columns": CFB_MARKET_FEATURES},
        "alpha_1e6": {"columns": BASE_COLUMNS, "ridge_alpha": 1_000_000.0},
        "alpha_0p1": {"columns": BASE_COLUMNS, "ridge_alpha": 0.1},
        "alpha_1": {"columns": BASE_COLUMNS, "ridge_alpha": 1.0},
        "alpha_100": {"columns": BASE_COLUMNS, "ridge_alpha": 100.0},
        "alpha_1000": {"columns": BASE_COLUMNS, "ridge_alpha": 1000.0},
        "plus_noise_column": {"columns": (*BASE_COLUMNS, PLANT_COLUMN), "noise_column": True},
    }


def _as_paired_predictions(scored: dict[str, pd.DataFrame], pick_column: str) -> pd.DataFrame:
    """Feed real picks through the project's own paired bootstrap.

    ``home_cover_probability`` is written as a hard 1.0/0.0 indicator, which
    reproduces ``experiments._paired_row_improvements``' accuracy metric
    exactly (it thresholds at 0.5) and makes its Brier/log-loss rows
    meaningless -- those rows are dropped by the caller, never reported.
    """

    frames = []
    for name, arm in scored.items():
        frames.append(
            pd.DataFrame(
                {
                    "feature_set": name,
                    "game_id": arm["game_id"].to_numpy(),
                    "season": arm["season"].to_numpy(),
                    "week": arm["week"].to_numpy(),
                    "home_cover": arm["home_cover"].to_numpy(),
                    "home_cover_probability": arm[pick_column].to_numpy(dtype=bool).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def stage_real(
    completed: pd.DataFrame,
    folds: list[PurgedFold],
    *,
    samples: int,
    seed: int = 20260818,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = completed.copy()
    frame[PLANT_COLUMN] = np.random.default_rng(seed).normal(0.0, 1.0, size=len(frame))

    scored: dict[str, pd.DataFrame] = {"base": score_arm(frame, folds, BASE_COLUMNS)}
    for name, spec in real_contrast_specs().items():
        scored[name] = score_arm(
            frame,
            folds,
            spec["columns"],
            ridge_alpha=float(spec.get("ridge_alpha", CFB_BENCHMARK_RIDGE_ALPHA)),
        )
        print(f"  scored real contrast {name}", flush=True)

    rows: list[dict[str, Any]] = []
    for pick in ("full_home", "sign_home"):
        key = "full" if pick == "full_home" else "sign"
        comparisons = paired_feature_comparisons(
            _as_paired_predictions(scored, pick),
            baseline_feature_set="base",
            samples=samples,
            block="week",
            seed=seed,
        )
        accuracy = comparisons.loc[comparisons["metric"].eq("accuracy_improvement")]
        for _, record in accuracy.iterrows():
            rows.append(
                {
                    "contrast": str(record["candidate_feature_set"]),
                    "estimator": key,
                    # paired_feature_comparisons works in accuracy fractions;
                    # this project reports accuracy POINTS everywhere else.
                    "estimate": 100.0 * float(record["estimate"]),
                    "lower": 100.0 * float(record["lower"]),
                    "upper": 100.0 * float(record["upper"]),
                    "probability_positive": float(record["probability_positive"]),
                    "paired_games": int(record["paired_games"]),
                    # Diagnostics: this arm's own level and its own
                    # full-minus-sign gap, so a delta gap can be attributed to
                    # the candidate arm's location rather than guessed at.
                    "arm_level_full": 100.0
                    * arm_accuracy(scored[str(record["candidate_feature_set"])], "full_home"),
                    "arm_level_sign": 100.0
                    * arm_accuracy(scored[str(record["candidate_feature_set"])], "sign_home"),
                    "arm_flip_rate": float(
                        (
                            scored[str(record["candidate_feature_set"])]["full_home"].to_numpy(
                                dtype=bool
                            )
                            != scored[str(record["candidate_feature_set"])]["sign_home"].to_numpy(
                                dtype=bool
                            )
                        ).mean()
                    ),
                }
            )
    # How much the residual location is WORTH on real data, on this instrument:
    # the same games scored with and without it, week-blocked. The NFL analogue
    # is docs/residual_location.md's +2.12 points for the production reader over
    # sign(predicted_market_residual); this is the CFB purged-CV reading of the
    # identical question, and it is the context any "the calibration step is a
    # defect" claim has to survive.
    location_frame = pd.concat(
        [
            _as_paired_predictions({"base_full": scored["base"]}, "full_home"),
            _as_paired_predictions({"base_sign": scored["base"]}, "sign_home"),
        ],
        ignore_index=True,
    )
    location = paired_feature_comparisons(
        location_frame,
        baseline_feature_set="base_sign",
        samples=samples,
        block="week",
        seed=seed,
    )
    location_row = location.loc[location["metric"].eq("accuracy_improvement")].iloc[0]
    base_arm = scored["base"]
    location_value = {
        "estimate_points": 100.0 * float(location_row["estimate"]),
        "lower": 100.0 * float(location_row["lower"]),
        "upper": 100.0 * float(location_row["upper"]),
        "probability_positive": float(location_row["probability_positive"]),
        "paired_games": int(location_row["paired_games"]),
        "level_full": 100.0 * arm_accuracy(base_arm, "full_home"),
        "level_sign": 100.0 * arm_accuracy(base_arm, "sign_home"),
        "flip_rate": float(
            (
                base_arm["full_home"].to_numpy(dtype=bool)
                != base_arm["sign_home"].to_numpy(dtype=bool)
            ).mean()
        ),
        **fold_diagnostics(base_arm),
    }

    table = pd.DataFrame(rows)

    wide = table.pivot(index="contrast", columns="estimator", values="estimate")
    x = wide["full"].to_numpy(dtype=float)
    y = wide["sign"].to_numpy(dtype=float)
    slope_intercept = np.polyfit(x, y, 1)
    through_origin = float(np.dot(x, y) / np.dot(x, x))
    residual = y - (slope_intercept[0] * x + slope_intercept[1])
    r_squared = 1.0 - float(np.var(residual) / np.var(y))
    regression = {
        "n_contrasts": len(x),
        "slope": float(slope_intercept[0]),
        "intercept": float(slope_intercept[1]),
        "slope_through_origin": through_origin,
        "r_squared": r_squared,
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "sign_disagreements": int(np.count_nonzero(np.sign(x) != np.sign(y))),
        "mean_abs_gap_points": float(np.mean(np.abs(y - x))),
        "max_abs_gap_points": float(np.max(np.abs(y - x))),
        "contrasts": list(wide.index.astype(str)),
        "full_points": [float(v) for v in x],
        "sign_points": [float(v) for v in y],
        "location_value": location_value,
    }
    return table, regression


# ---------------------------------------------------------------------------
# Stage 4: mechanism
# ---------------------------------------------------------------------------


def stage_mechanism(
    completed: pd.DataFrame,
    *,
    replicates: int,
    magnitude: float = 0.513,
) -> pd.DataFrame:
    noise_std = float(pd.to_numeric(completed["ats_margin"], errors="coerce").std())
    gamma = additive_plant_gamma(magnitude, noise_std)
    fold_cache: dict[int, list[PurgedFold]] = {}
    rows: list[dict[str, Any]] = []

    conditions: list[dict[str, Any]] = []
    for fraction in (0.10, 0.20, 0.35, 0.45):
        conditions.append({"knob": "distribution_fraction", "value": fraction, "n_blocks": 40})
    for blocks in (10, 20, 40, 80):
        conditions.append({"knob": "n_blocks", "value": blocks, "n_blocks": blocks})
    for width in ("thin", "full"):
        conditions.append({"knob": "design_width", "value": width, "n_blocks": 40})

    for construction in ("overwrite", "additive"):
        for condition in conditions:
            n_blocks = int(condition["n_blocks"])
            if n_blocks not in fold_cache:
                fold_cache[n_blocks] = build_folds(completed, n_blocks=n_blocks)
            folds = fold_cache[n_blocks]
            fraction = (
                float(condition["value"])
                if condition["knob"] == "distribution_fraction"
                else CFB_BENCHMARK_DISTRIBUTION_FRACTION
            )
            columns = (
                THIN_COLUMNS
                if condition["knob"] == "design_width" and condition["value"] == "thin"
                else BASE_COLUMNS
            )
            for replicate in range(replicates):
                # crc32, not hash(): str hashing is salted per process, which
                # would make this stage's seeds irreproducible across runs.
                seed = 900_000 + 37 * replicate + zlib.crc32(str(condition).encode()) % 1000
                if construction == "overwrite":
                    planted = inject_synthetic_signal(
                        completed, target_accuracy=magnitude, seed=seed, column=PLANT_COLUMN
                    )
                    truth = 100.0 * (
                        float(
                            (
                                np.sign(planted[PLANT_COLUMN].to_numpy())
                                == np.sign(planted["ats_margin"].to_numpy())
                            ).mean()
                        )
                        - 0.5
                    )
                    carrier = planted[PLANT_COLUMN].to_numpy(dtype=float)
                else:
                    carrier = orthogonal_carrier(len(completed), seed=seed)
                    planted = plant_additive_effect(
                        completed, carrier=carrier, gamma=gamma, column=PLANT_COLUMN
                    )
                    truth = float("nan")
                candidate = score_arm(
                    planted, folds, (*columns, PLANT_COLUMN), distribution_fraction=fraction
                )
                baseline = score_arm(planted, folds, columns, distribution_fraction=fraction)
                row: dict[str, Any] = {
                    "construction": construction,
                    "knob": condition["knob"],
                    "value": str(condition["value"]),
                    "replicate": replicate,
                    "truth_points": truth,
                }
                for pick in ("full_home", "sign_home"):
                    key = "full" if pick == "full_home" else "sign"
                    delta, flips = paired_delta(baseline, candidate, pick)
                    row[f"delta_{key}"] = delta
                    row[f"flip_rate_{key}"] = flips
                    row[f"oracle_{key}"] = oracle_delta(baseline, carrier, gamma, pick)
                    row[f"level_{key}"] = 100.0 * arm_accuracy(candidate, pick)
                row.update({f"cand_{k}": v for k, v in fold_diagnostics(candidate).items()})
                rows.append(row)
            print(
                f"  mechanism[{construction}] {condition['knob']}={condition['value']} done",
                flush=True,
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    """Mean, 95% interval and ``probability_positive`` across replicates.

    The interval and the probability come from the spread ACROSS replicates
    (independent plant draws on the same corpus), not from a within-run block
    bootstrap: the estimand here is "what does this estimator recover from a
    fresh plant of this size on this corpus", so the plant draw is the
    replication unit. ``probability_positive`` is the Student-t tail
    ``P(mean > 0)`` at ``n - 1`` degrees of freedom -- reported per
    ``AGENTS.md``'s binding rule that continuous evidence, never "contains
    zero", is what may be stated.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    blank = {
        "n": 0,
        "mean": float("nan"),
        "lower": float("nan"),
        "upper": float("nan"),
        "probability_positive": float("nan"),
    }
    if n == 0:
        return blank
    mean = float(values.mean())
    if n == 1:
        return {**blank, "n": 1, "mean": mean, "lower": mean, "upper": mean}
    se = float(values.std(ddof=1) / np.sqrt(n))
    if se <= 0.0:
        probability = 1.0 if mean > 0 else (0.0 if mean < 0 else 0.5)
        return {
            "n": n,
            "mean": mean,
            "lower": mean,
            "upper": mean,
            "probability_positive": probability,
        }
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
        "probability_positive": float(stats.t.cdf(mean / se, n - 1)),
    }


def summarize_overwrite(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (design, magnitude), group in table.groupby(["design", "magnitude"], sort=True):
        truth = float(group["truth_level_points"].mean())
        framings = (
            ("level_full", group["level_full_candidate"] - 50.0, truth),
            ("level_sign", group["level_sign_candidate"] - 50.0, truth),
            ("delta_full", group["delta_full"], truth),
            ("delta_sign", group["delta_sign"], truth),
            # Paired WITHIN a replicate: both estimators read the identical
            # fitted model, so their difference isolates the calibration step
            # and cancels the plant draw's own variance. Truth is 0 -- a
            # calibration step that neither helps nor hurts the measurement.
            (
                "gap_level_full_minus_sign",
                group["level_full_candidate"] - group["level_sign_candidate"],
                0.0,
            ),
            ("gap_delta_full_minus_sign", group["delta_full"] - group["delta_sign"], 0.0),
        )
        for framing, measured, reference in framings:
            summary = _mean_ci(measured.to_numpy())
            rows.append(
                {
                    "design": design,
                    "magnitude": magnitude,
                    "framing": framing,
                    "truth_points": reference,
                    "recovered_points": summary["mean"],
                    "recovered_lower": summary["lower"],
                    "recovered_upper": summary["upper"],
                    "probability_positive": summary["probability_positive"],
                    "attenuation_ratio": summary["mean"] / reference if reference else float("nan"),
                    "ratio_lower": summary["lower"] / reference if reference else float("nan"),
                    "ratio_upper": summary["upper"] / reference if reference else float("nan"),
                    "replicates": summary["n"],
                }
            )
    return pd.DataFrame(rows)


def _through_origin_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of ``y ~ b * x`` with its standard error.

    Used instead of the mean of per-run ratios ``y / x``: several conditions
    have an oracle delta near zero, and a ratio with a near-zero denominator
    has no usable mean. The regression pools the whole magnitude sweep, where
    the large conditions carry the information.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope * x
    se = float(np.sqrt((residual**2).sum() / (len(x) - 1) / (x**2).sum()))
    return slope, se


def additive_recovery_slopes(table: pd.DataFrame) -> pd.DataFrame:
    """How much of the within-frame oracle delta each estimator actually recovers."""

    rows: list[dict[str, Any]] = []
    for kind, group in table.groupby("carrier_kind", sort=True):
        for key in ("full", "sign"):
            slope, se = _through_origin_slope(
                group[f"oracle_{key}"].to_numpy(), group[f"delta_{key}"].to_numpy()
            )
            rows.append(
                {
                    "carrier_kind": kind,
                    "estimator": key,
                    "recovery_slope": slope,
                    "slope_se": se,
                    "slope_lower": slope - 1.96 * se,
                    "slope_upper": slope + 1.96 * se,
                    "runs": len(group),
                }
            )
        gap = _mean_ci((group["delta_full"] - group["delta_sign"]).to_numpy())
        rows.append(
            {
                "carrier_kind": kind,
                "estimator": "gap_full_minus_sign",
                "recovery_slope": gap["mean"],
                "slope_se": float("nan"),
                "slope_lower": gap["lower"],
                "slope_upper": gap["upper"],
                "runs": gap["n"],
            }
        )
    # Does one estimator's delta predict the other's? Slope 1 means no
    # attenuation from the calibration step; sign disagreements are split by
    # effect size, because two readings of a null must disagree about half the
    # time whatever the estimator does.
    x = table["delta_full"].to_numpy(dtype=float)
    y = table["delta_sign"].to_numpy(dtype=float)
    fit = np.polyfit(x, y, 1)
    resolvable = (np.abs(x) > 0.5) | (np.abs(y) > 0.5)
    rows.append(
        {
            "carrier_kind": "ALL",
            "estimator": "sign_on_full_regression",
            "recovery_slope": float(fit[0]),
            "slope_se": float("nan"),
            "slope_lower": float(fit[1]),
            "slope_upper": float(np.corrcoef(x, y)[0, 1]),
            "runs": len(x),
        }
    )
    rows.append(
        {
            "carrier_kind": "ALL",
            "estimator": "sign_disagreements_above_0p5",
            "recovery_slope": float(
                np.count_nonzero(np.sign(x[resolvable]) != np.sign(y[resolvable]))
            ),
            "slope_se": float("nan"),
            "slope_lower": float(np.count_nonzero(resolvable)),
            "slope_upper": float("nan"),
            "runs": len(x),
        }
    )
    rows.append(
        {
            "carrier_kind": "ALL",
            "estimator": "sign_disagreements_below_0p5",
            "recovery_slope": float(
                np.count_nonzero(np.sign(x[~resolvable]) != np.sign(y[~resolvable]))
            ),
            "slope_se": float("nan"),
            "slope_lower": float(np.count_nonzero(~resolvable)),
            "slope_upper": float("nan"),
            "runs": len(x),
        }
    )
    return pd.DataFrame(rows)


def summarize_additive(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (kind, magnitude), group in table.groupby(["carrier_kind", "magnitude"], sort=True):
        for key in ("full", "sign"):
            measured = _mean_ci(group[f"delta_{key}"].to_numpy())
            oracle = _mean_ci(group[f"oracle_{key}"].to_numpy())
            ratios = _mean_ci(
                (group[f"delta_{key}"] / group[f"oracle_{key}"].replace(0.0, np.nan)).to_numpy()
            )
            rows.append(
                {
                    "carrier_kind": kind,
                    "magnitude": magnitude,
                    "estimator": key,
                    "delta_points": measured["mean"],
                    "delta_lower": measured["lower"],
                    "delta_upper": measured["upper"],
                    "probability_positive": measured["probability_positive"],
                    "oracle_points": oracle["mean"],
                    "recovery_ratio": ratios["mean"],
                    "recovery_lower": ratios["lower"],
                    "recovery_upper": ratios["upper"],
                    "replicates": measured["n"],
                }
            )
        gap = _mean_ci((group["delta_full"] - group["delta_sign"]).to_numpy())
        rows.append(
            {
                "carrier_kind": kind,
                "magnitude": magnitude,
                "estimator": "gap_full_minus_sign",
                "delta_points": gap["mean"],
                "delta_lower": gap["lower"],
                "delta_upper": gap["upper"],
                "probability_positive": gap["probability_positive"],
                "oracle_points": float((group["oracle_full"] - group["oracle_sign"]).mean()),
                "recovery_ratio": float("nan"),
                "recovery_lower": float("nan"),
                "recovery_upper": float("nan"),
                "replicates": gap["n"],
            }
        )
    return pd.DataFrame(rows)


def resummarize(directory: Path) -> None:
    """Re-derive every summary table from an archived run's raw CSVs, no refitting."""

    overwrite_path = directory / "overwrite_raw.csv"
    if overwrite_path.exists():
        overview = summarize_overwrite(pd.read_csv(overwrite_path))
        overview.to_csv(directory / "overwrite_summary.csv", index=False)
        stamp_sidecar(directory / "overwrite_summary.csv")  # ENG-38
        print("=== overwrite ===")
        print(overview.to_string(index=False), flush=True)
    additive_path = directory / "additive_raw.csv"
    if additive_path.exists():
        raw = pd.read_csv(additive_path)
        overview = summarize_additive(raw)
        overview.to_csv(directory / "additive_summary.csv", index=False)
        stamp_sidecar(directory / "additive_summary.csv")  # ENG-38
        slopes = additive_recovery_slopes(raw)
        slopes.to_csv(directory / "additive_recovery_slopes.csv", index=False)
        stamp_sidecar(directory / "additive_recovery_slopes.csv")  # ENG-38
        print("=== additive ===")
        print(overview.to_string(index=False), flush=True)
        print("=== additive recovery slopes ===")
        print(slopes.to_string(index=False), flush=True)
    real_path = directory / "real_contrasts.csv"
    if real_path.exists():
        raw = pd.read_csv(real_path)
        wide = raw.pivot(index="contrast", columns="estimator", values="estimate")
        x = wide["full"].to_numpy(dtype=float)
        y = wide["sign"].to_numpy(dtype=float)
        fit = np.polyfit(x, y, 1)
        resolvable = np.abs(x) > 0.5
        print("=== real contrasts ===")
        print(
            json.dumps(
                {
                    "n_contrasts": len(x),
                    "slope": float(fit[0]),
                    "intercept": float(fit[1]),
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "sign_disagreements": int(np.count_nonzero(np.sign(x) != np.sign(y))),
                    "contrasts_above_0p5": int(np.count_nonzero(resolvable)),
                    "sign_disagreements_above_0p5": int(
                        np.count_nonzero(np.sign(x[resolvable]) != np.sign(y[resolvable]))
                    ),
                    "contrasts_below_0p5": int(np.count_nonzero(~resolvable)),
                    "sign_disagreements_below_0p5": int(
                        np.count_nonzero(np.sign(x[~resolvable]) != np.sign(y[~resolvable]))
                    ),
                    "mean_abs_gap_points": float(np.mean(np.abs(y - x))),
                    "max_abs_gap_points": float(np.max(np.abs(y - x))),
                },
                indent=2,
            ),
            flush=True,
        )

    mechanism_path = directory / "mechanism_raw.csv"
    if mechanism_path.exists():
        table = pd.read_csv(mechanism_path)
        table["gap_full_minus_sign"] = table["delta_full"] - table["delta_sign"]
        grouped = (
            table.groupby(["construction", "knob", "value"], sort=True)[
                [
                    "delta_full",
                    "delta_sign",
                    "gap_full_minus_sign",
                    "oracle_full",
                    "oracle_sign",
                    "cand_mean_abs_threshold",
                    "cand_mean_pr_sd",
                    "cand_mean_abs_threshold_over_pr_sd",
                ]
            ]
            .mean()
            .reset_index()
        )
        grouped.to_csv(directory / "mechanism_summary.csv", index=False)
        stamp_sidecar(directory / "mechanism_summary.csv")  # ENG-38
        print("=== mechanism ===")
        print(grouped.to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "overwrite", "additive", "real", "mechanism"),
    )
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--mechanism-replicates", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=DEFAULT_N_BLOCKS)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resummarize",
        type=Path,
        default=None,
        help=(
            "Re-derive the summary tables from an existing run directory's raw CSVs "
            "without refitting anything. Every summary in the doc is reproducible this "
            "way from the archived raw rows."
        ),
    )
    args = parser.parse_args()

    if args.resummarize is not None:
        resummarize(args.resummarize)
        return

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output / stamp
    output.mkdir(parents=True, exist_ok=True)
    print(f"writing to {output}", flush=True)

    completed = load_completed()
    folds = build_folds(completed, n_blocks=args.n_blocks)
    print(f"{len(completed)} completed CFB games, {len(folds)} purged folds", flush=True)

    summary: dict[str, Any] = {
        "stamp": stamp,
        "games": len(completed),
        "folds": len(folds),
        "n_blocks": int(args.n_blocks),
        "replicates": int(args.replicates),
        "purge_weeks": int(DEFAULT_PURGE_WEEKS),
        "embargo_weeks": int(DEFAULT_EMBARGO_WEEKS),
    }

    if args.stage in ("all", "overwrite"):
        t0 = time.time()
        full = stage_overwrite(completed, folds, replicates=args.replicates)
        thin = stage_overwrite(
            completed,
            folds,
            replicates=args.replicates,
            magnitudes=(0.513, 0.530),
            feature_columns=THIN_COLUMNS,
            label="thin_design",
        )
        table = pd.concat([full, thin], ignore_index=True)
        table.to_csv(output / "overwrite_raw.csv", index=False)
        stamp_sidecar(output / "overwrite_raw.csv")  # ENG-38
        overview = summarize_overwrite(table)
        overview.to_csv(output / "overwrite_summary.csv", index=False)
        stamp_sidecar(output / "overwrite_summary.csv")  # ENG-38
        summary["overwrite_seconds"] = time.time() - t0
        summary["overwrite"] = overview.to_dict(orient="records")
        print(overview.to_string(index=False), flush=True)

    if args.stage in ("all", "additive"):
        t0 = time.time()
        table = stage_additive(completed, folds, replicates=args.replicates)
        table.to_csv(output / "additive_raw.csv", index=False)
        stamp_sidecar(output / "additive_raw.csv")  # ENG-38
        overview = summarize_additive(table)
        overview.to_csv(output / "additive_summary.csv", index=False)
        stamp_sidecar(output / "additive_summary.csv")  # ENG-38
        slopes = additive_recovery_slopes(table)
        slopes.to_csv(output / "additive_recovery_slopes.csv", index=False)
        stamp_sidecar(output / "additive_recovery_slopes.csv")  # ENG-38
        summary["additive_seconds"] = time.time() - t0
        summary["additive"] = overview.to_dict(orient="records")
        summary["additive_recovery_slopes"] = slopes.to_dict(orient="records")
        print(overview.to_string(index=False), flush=True)
        print(slopes.to_string(index=False), flush=True)

    if args.stage in ("all", "real"):
        t0 = time.time()
        table, regression = stage_real(completed, folds, samples=args.bootstrap_samples)
        table.to_csv(output / "real_contrasts.csv", index=False)
        stamp_sidecar(output / "real_contrasts.csv")  # ENG-38
        summary["real_seconds"] = time.time() - t0
        summary["real_contrasts"] = table.to_dict(orient="records")
        summary["real_regression"] = regression
        print(table.to_string(index=False), flush=True)
        print(json.dumps(regression, indent=2), flush=True)

    if args.stage in ("all", "mechanism"):
        t0 = time.time()
        table = stage_mechanism(completed, replicates=args.mechanism_replicates)
        table.to_csv(output / "mechanism_raw.csv", index=False)
        stamp_sidecar(output / "mechanism_raw.csv")  # ENG-38
        grouped = (
            table.groupby(["construction", "knob", "value"], sort=True)[
                [
                    "delta_full",
                    "delta_sign",
                    "oracle_full",
                    "oracle_sign",
                    "cand_mean_abs_threshold",
                    "cand_mean_pr_sd",
                    "cand_mean_abs_threshold_over_pr_sd",
                ]
            ]
            .mean()
            .reset_index()
        )
        grouped.to_csv(output / "mechanism_summary.csv", index=False)
        stamp_sidecar(output / "mechanism_summary.csv")  # ENG-38
        summary["mechanism_seconds"] = time.time() - t0
        summary["mechanism"] = grouped.to_dict(orient="records")
        print(grouped.to_string(index=False), flush=True)

    write_stamped_artifact(summary, output / "summary.json")  # ENG-38
    print(f"\nwrote {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from opener_error_transfer_unit3 import FEATURES as CFB_FEATURES  # noqa: E402
from opener_error_transfer_unit3 import design as cfb_design  # noqa: E402
from opener_error_transfer_unit3 import (  # noqa: E402
    diff_report,
    fit_logit_beta,
    games_record,
    load_cfb_population,
    load_extended_nfl_population,
    loso_logit,
    paired_accuracy_effect,
    prob_metrics,
    standardize,
)
from opener_error_transfer_unit4 import (  # noqa: E402
    BASE_FEATURES,
    attach_cfb_transfer_logit,
    load_line_move_population,
)

from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
)

SEED = 20260923
SAMPLES = 4000
LM_DRAWS = 2000
RELIABILITY_REUSED_FROM_UNIT2 = 0.14132263925353555
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def pooled_design(
    frame: pd.DataFrame,
    feature_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
    is_nfl: np.ndarray,
) -> np.ndarray:
    main = cfb_design(frame, feature_cols, means, stds)
    is_nfl_col = np.asarray(is_nfl, dtype=float).reshape(-1, 1)
    interact = is_nfl_col * main[:, 1:]
    return np.hstack([main, is_nfl_col, interact])


def pooled_transfer_predict(
    nfl: pd.DataFrame, cfb_all: pd.DataFrame, feature_cols: list[str], look_tag: str
) -> tuple[pd.Series, dict[str, Any], dict[str, Any]]:
    pred_p = pd.Series(np.nan, index=nfl.index, dtype=float)
    per_fold_meta: dict[str, Any] = {}
    fold_betas: dict[str, Any] = {}
    coef_names = [
        "intercept",
        *feature_cols,
        "is_nfl",
        *[f"is_nfl_x_{c}" for c in feature_cols],
    ]
    for held in sorted(int(v) for v in nfl["season"].unique()):
        cfb_train = cfb_all.loc[cfb_all["season"].lt(held)]
        nfl_train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if cfb_train.empty or nfl_train.empty or test.empty:
            continue
        pooled = pd.concat(
            [
                cfb_train[[*feature_cols, "home_covered"]].assign(is_nfl=0.0),
                nfl_train[[*feature_cols, "home_covered"]].assign(is_nfl=1.0),
            ],
            ignore_index=True,
        )
        means, stds = standardize(pooled, feature_cols)
        x_train = pooled_design(pooled, feature_cols, means, stds, pooled["is_nfl"].to_numpy())
        beta = fit_logit_beta(x_train, pooled["home_covered"].to_numpy())
        x_test = pooled_design(test, feature_cols, means, stds, np.ones(len(test)))
        z = np.clip(x_test @ beta, -35.0, 35.0)
        pred_p.loc[test.index] = 1.0 / (1.0 + np.exp(-z))
        fold_betas[str(held)] = {
            name: float(val) for name, val in zip(coef_names, beta, strict=True)
        }
        per_fold_meta[str(held)] = {
            "cfb_train_seasons": sorted(int(s) for s in cfb_train["season"].unique()),
            "nfl_train_seasons": sorted(int(s) for s in nfl_train["season"].unique()),
            "cfb_train_games": len(cfb_train),
            "nfl_train_games": len(nfl_train),
        }
    record_look(look_tag)
    return pred_p, per_fold_meta, fold_betas


def logit_of(p: pd.Series) -> pd.Series:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def base_four_term_loso(nfl: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(np.nan, index=nfl.index, dtype=float)
    fold_betas: dict[str, Any] = {}
    for held in sorted(int(v) for v in nfl["season"].unique()):
        train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds), train["home_covered"].to_numpy(), FIT_RIDGE
        )
        out.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
        fold_betas[str(held)] = {
            "intercept": float(fold_beta[0]),
            **{name: float(fold_beta[idx + 1]) for idx, name in enumerate(FIT_FEATURES)},
        }
    record_look("loso_base_four_term_accuracy")
    return out, fold_betas


def accuracy_arm() -> dict[str, Any]:
    nfl, nfl_provenance, graded_meta = load_extended_nfl_population()
    cfb_all = load_cfb_population()

    cfb_p, cfb_meta = attach_cfb_transfer_logit(nfl, cfb_all)
    record_look("cfb_transfer_accuracy_population")
    nfl["cfb_transfer_logit"] = logit_of(cfb_p)

    pooled_p, pooled_meta, pooled_betas = pooled_transfer_predict(
        nfl, cfb_all, CFB_FEATURES, "pooled_fit_accuracy_population"
    )
    nfl["pooled_transfer_logit"] = logit_of(pooled_p)
    nfl = nfl.dropna(subset=["cfb_transfer_logit", "pooled_transfer_logit"]).reset_index(drop=True)

    nfl["base_oos_p"], base_fold_betas = base_four_term_loso(nfl)
    nfl["plus_pooled_oos_p"], plus_pooled_betas = loso_logit(
        nfl, [*FIT_FEATURES, "pooled_transfer_logit"], "home_covered", "plus_pooled_accuracy"
    )
    record_look("loso_plus_pooled_accuracy")
    nfl["plus_cfb_oos_p"], plus_cfb_betas = loso_logit(
        nfl, [*FIT_FEATURES, "cfb_transfer_logit"], "home_covered", "plus_cfb_accuracy"
    )
    record_look("loso_plus_cfb_accuracy")
    nfl = nfl.dropna(subset=["base_oos_p", "plus_pooled_oos_p", "plus_cfb_oos_p"]).reset_index(
        drop=True
    )

    for tag in ("base", "plus_pooled", "plus_cfb"):
        pick = nfl[f"{tag}_oos_p"].ge(0.5)
        nfl[f"{tag}_pick"] = pick
        nfl[f"{tag}_correct"] = pick.astype(float).eq(nfl["home_covered"]).astype(float)

    nfl_2020_2025 = nfl.loc[nfl["season"].ge(2020)].reset_index(drop=True)

    cell_all = {
        "diff": diff_report(
            nfl, "plus_pooled_pick", "plus_pooled_correct", "base_pick", "base_correct"
        ),
        "effect": paired_accuracy_effect(nfl, "plus_pooled_correct", "base_correct"),
        "n_games": len(nfl),
        "n_seasons": int(nfl["season"].nunique()),
    }
    record_look("pooled_vs_base_accuracy_all_graded")
    cell_subset = {
        "diff": diff_report(
            nfl_2020_2025, "plus_pooled_pick", "plus_pooled_correct", "base_pick", "base_correct"
        ),
        "effect": paired_accuracy_effect(nfl_2020_2025, "plus_pooled_correct", "base_correct"),
        "n_games": len(nfl_2020_2025),
        "n_seasons": int(nfl_2020_2025["season"].nunique()),
    }
    record_look("pooled_vs_base_accuracy_2020_2025_subset")
    cell_vs_cfb = {
        "diff": diff_report(
            nfl, "plus_pooled_pick", "plus_pooled_correct", "plus_cfb_pick", "plus_cfb_correct"
        ),
        "effect": paired_accuracy_effect(nfl, "plus_pooled_correct", "plus_cfb_correct"),
        "n_games": len(nfl),
    }
    record_look("pooled_vs_cfb_accuracy_all_graded_context")

    records = {
        "base": games_record(nfl["base_correct"]),
        "plus_pooled": games_record(nfl["plus_pooled_correct"]),
        "plus_cfb": games_record(nfl["plus_cfb_correct"]),
    }

    return {
        "graded_meta": graded_meta,
        "n_games_all_graded": len(nfl),
        "n_games_2020_2025_subset": len(nfl_2020_2025),
        "records": records,
        "pooled_vs_base_accuracy_all_graded": cell_all,
        "pooled_vs_base_accuracy_2020_2025_subset": cell_subset,
        "pooled_vs_cfb_accuracy_all_graded_context": cell_vs_cfb,
        "base_oos_metrics": prob_metrics(
            nfl["base_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()
        ),
        "plus_pooled_oos_metrics": prob_metrics(
            nfl["plus_pooled_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()
        ),
        "plus_cfb_oos_metrics": prob_metrics(
            nfl["plus_cfb_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()
        ),
        "pooled_per_fold_meta": pooled_meta,
        "pooled_loso_fold_betas": pooled_betas,
        "cfb_per_fold_meta": cfb_meta,
        "plus_pooled_loso_fold_betas": plus_pooled_betas,
        "plus_cfb_loso_fold_betas": plus_cfb_betas,
        "base_four_term_loso_fold_betas": base_fold_betas,
        "nfl_provenance": nfl_provenance,
    }


def line_move_arm() -> dict[str, Any]:
    nfl, nfl_provenance, graded_meta = load_line_move_population()
    cfb_all = load_cfb_population()

    cfb_p, cfb_meta = attach_cfb_transfer_logit(nfl, cfb_all)
    record_look("cfb_transfer_linemove_population")
    nfl["cfb_transfer_logit"] = logit_of(cfb_p)

    pooled_p, pooled_meta, pooled_betas = pooled_transfer_predict(
        nfl, cfb_all, CFB_FEATURES, "pooled_fit_linemove_population"
    )
    nfl["pooled_transfer_logit"] = logit_of(pooled_p)
    nfl = nfl.dropna(subset=["cfb_transfer_logit", "pooled_transfer_logit"]).reset_index(drop=True)

    nfl["base_oos_p"], base_fold_betas = loso_logit(
        nfl, list(BASE_FEATURES), "home_covered", "base_tuesday_linemove"
    )
    record_look("loso_base_tuesday_linemove")
    nfl["plus_pooled_oos_p"], plus_pooled_betas = loso_logit(
        nfl, [*BASE_FEATURES, "pooled_transfer_logit"], "home_covered", "plus_pooled_linemove"
    )
    record_look("loso_plus_pooled_linemove")
    nfl["plus_cfb_oos_p"], plus_cfb_betas = loso_logit(
        nfl, [*BASE_FEATURES, "cfb_transfer_logit"], "home_covered", "plus_cfb_linemove"
    )
    record_look("loso_plus_cfb_linemove")
    nfl = nfl.dropna(subset=["base_oos_p", "plus_pooled_oos_p", "plus_cfb_oos_p"]).reset_index(
        drop=True
    )

    for tag in ("base", "plus_pooled", "plus_cfb"):
        pick = nfl[f"{tag}_oos_p"].ge(0.5)
        nfl[f"{tag}_pick"] = pick
        nfl[f"lm_{tag}"] = np.where(pick, 1.0, -1.0) * nfl["open_move"]
        nfl[f"{tag}_correct"] = pick.astype(float).eq(nfl["home_covered"]).astype(float)

    nfl["diff_line_move_pooled_vs_base"] = nfl["lm_plus_pooled"] - nfl["lm_base"]
    nfl["diff_line_move_pooled_vs_cfb"] = nfl["lm_plus_pooled"] - nfl["lm_plus_cfb"]
    nfl["diff_accuracy_pooled_vs_base"] = nfl["plus_pooled_correct"] - nfl["base_correct"]

    nfl_2020_2025 = nfl.loc[nfl["season"].ge(2020)].reset_index(drop=True)

    cell_all = cell_stats(
        "pooled_vs_base_linemove_all_graded", nfl, "diff_line_move_pooled_vs_base", SEED, LM_DRAWS
    )
    record_look("pooled_vs_base_linemove_all_graded")
    cell_subset = cell_stats(
        "pooled_vs_base_linemove_2020_2025_subset",
        nfl_2020_2025,
        "diff_line_move_pooled_vs_base",
        SEED,
        LM_DRAWS,
    )
    record_look("pooled_vs_base_linemove_2020_2025_subset")
    cell_vs_cfb = cell_stats(
        "pooled_vs_cfb_linemove_all_graded_context",
        nfl,
        "diff_line_move_pooled_vs_cfb",
        SEED,
        LM_DRAWS,
    )
    record_look("pooled_vs_cfb_linemove_all_graded_context")
    accuracy_companion = cell_stats(
        "pooled_vs_base_accuracy_companion_linemove_population",
        nfl,
        "diff_accuracy_pooled_vs_base",
        SEED + 500,
        LM_DRAWS,
    )
    record_look("pooled_vs_base_accuracy_companion_linemove_population")

    records = {
        "base_all_graded": games_record(nfl["base_correct"]),
        "plus_pooled_all_graded": games_record(nfl["plus_pooled_correct"]),
        "plus_cfb_all_graded": games_record(nfl["plus_cfb_correct"]),
        "base_2020_2025": games_record(nfl_2020_2025["base_correct"]),
        "plus_pooled_2020_2025": games_record(nfl_2020_2025["plus_pooled_correct"]),
    }

    return {
        "graded_meta": graded_meta,
        "n_games_all_graded": len(nfl),
        "n_games_2020_2025_subset": len(nfl_2020_2025),
        "records": records,
        "pooled_vs_base_linemove_all_graded": cell_all,
        "pooled_vs_base_linemove_2020_2025_subset": cell_subset,
        "pooled_vs_cfb_linemove_all_graded_context": cell_vs_cfb,
        "accuracy_companion_pooled_vs_base": accuracy_companion,
        "pooled_per_fold_meta": pooled_meta,
        "pooled_loso_fold_betas": pooled_betas,
        "cfb_per_fold_meta": cfb_meta,
        "base_loso_fold_betas": base_fold_betas,
        "plus_pooled_loso_fold_betas": plus_pooled_betas,
        "plus_cfb_loso_fold_betas": plus_cfb_betas,
        "nfl_provenance": nfl_provenance,
    }


def main() -> None:
    accuracy = accuracy_arm()
    line_move = line_move_arm()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/opener_error_transfer_unit5" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "unit": 5,
        "features": CFB_FEATURES,
        "ridge": FIT_RIDGE,
        "seed": SEED,
        "accuracy_samples": SAMPLES,
        "line_move_draws": LM_DRAWS,
        "reliability_reused_from_unit2": RELIABILITY_REUSED_FROM_UNIT2,
        "accuracy_arm": accuracy,
        "line_move_arm": line_move,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"artifact_dir={out_dir}")
    print("accuracy_graded_meta", json.dumps(accuracy["graded_meta"]))
    print("accuracy_records", json.dumps(accuracy["records"]))
    print(
        "pooled_vs_base_accuracy_all_graded",
        json.dumps(accuracy["pooled_vs_base_accuracy_all_graded"]["effect"]),
    )
    print(
        "pooled_vs_base_accuracy_2020_2025_subset",
        json.dumps(accuracy["pooled_vs_base_accuracy_2020_2025_subset"]["effect"]),
    )
    print(
        "pooled_vs_cfb_accuracy_all_graded_context",
        json.dumps(accuracy["pooled_vs_cfb_accuracy_all_graded_context"]["effect"]),
    )
    print("line_move_records", json.dumps(line_move["records"]))
    print(
        "pooled_vs_base_linemove_all_graded",
        json.dumps(
            {
                k: line_move["pooled_vs_base_linemove_all_graded"][k]
                for k in (
                    "mean_points",
                    "season_block_interval_low",
                    "season_block_interval_high",
                    "season_block_probability_positive",
                )
            }
        ),
    )
    print(
        "pooled_vs_base_linemove_2020_2025_subset",
        json.dumps(
            {
                k: line_move["pooled_vs_base_linemove_2020_2025_subset"][k]
                for k in (
                    "mean_points",
                    "season_block_interval_low",
                    "season_block_interval_high",
                    "season_block_probability_positive",
                )
            }
        ),
    )
    print(
        "pooled_vs_cfb_linemove_all_graded_context",
        json.dumps(
            {
                k: line_move["pooled_vs_cfb_linemove_all_graded_context"][k]
                for k in (
                    "mean_points",
                    "season_block_interval_low",
                    "season_block_interval_high",
                    "season_block_probability_positive",
                )
            }
        ),
    )
    print("look_count", len(LOOKS))


if __name__ == "__main__":
    main()

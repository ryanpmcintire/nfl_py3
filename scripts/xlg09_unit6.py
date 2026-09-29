from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

import opener_error_transfer_unit5 as u5  # noqa: E402
from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from opener_error_transfer_unit3 import (  # noqa: E402
    FEATURES,
    design,
    fit_logit_beta,
    games_record,
    load_cfb_population,
    load_extended_nfl_population,
    paired_accuracy_effect,
    predict_logit,
    standardize,
)
from opener_error_transfer_unit4 import BASE_FEATURES, load_line_move_population  # noqa: E402

from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
)

LOOKS: list[str] = []


def pooled_logit_for(
    cfb_all: pd.DataFrame, nfl: pd.DataFrame, target_season: int, exclude: set[int]
) -> pd.Series:
    cfb_train = cfb_all.loc[cfb_all["season"].lt(target_season)]
    nfl_train = nfl.loc[~nfl["season"].isin(exclude)]
    test = nfl.loc[nfl["season"].eq(target_season)]
    pooled = pd.concat(
        [
            cfb_train[[*FEATURES, "home_covered"]].assign(is_nfl=0.0),
            nfl_train[[*FEATURES, "home_covered"]].assign(is_nfl=1.0),
        ],
        ignore_index=True,
    )
    means, stds = standardize(pooled, FEATURES)
    x_train = u5.pooled_design(pooled, FEATURES, means, stds, pooled["is_nfl"].to_numpy())
    beta = fit_logit_beta(x_train, pooled["home_covered"].to_numpy())
    x_test = u5.pooled_design(test, FEATURES, means, stds, np.ones(len(test)))
    z = np.clip(x_test @ beta, -35.0, 35.0)
    return u5.logit_of(pd.Series(1.0 / (1.0 + np.exp(-z)), index=test.index))


def nested_column(cfb_all: pd.DataFrame, nfl: pd.DataFrame, held: int) -> pd.Series:
    col = pd.Series(np.nan, index=nfl.index, dtype=float)
    for s in sorted(int(v) for v in nfl["season"].unique()):
        if s == held:
            exclude = {held}
        else:
            exclude = {s, held}
        col.loc[nfl["season"].eq(s)] = pooled_logit_for(cfb_all, nfl, s, exclude)
    return col


def nested_loso(
    nfl: pd.DataFrame, cfb_all: pd.DataFrame, base_cols: list[str]
) -> tuple[pd.Series, dict[str, float]]:
    out = pd.Series(np.nan, index=nfl.index, dtype=float)
    betas: dict[str, float] = {}
    cols = [*base_cols, "nested_pooled_logit"]
    for held in sorted(int(v) for v in nfl["season"].unique()):
        work = nfl.copy()
        work["nested_pooled_logit"] = nested_column(cfb_all, nfl, held)
        train = work.loc[work["season"].ne(held)]
        test = work.loc[work["season"].eq(held)]
        means, stds = standardize(train, cols)
        beta = fit_logit_beta(
            design(train, cols, means, stds), train["home_covered"].astype(float).to_numpy()
        )
        out.loc[test.index] = predict_logit(test, cols, beta, means, stds)
        betas[str(held)] = float(beta[-1])
    return out, betas


def base_p_accuracy(nfl: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=nfl.index, dtype=float)
    for held in sorted(int(v) for v in nfl["season"].unique()):
        train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        m, s = _standardisers(train)
        b = _fit_logit(_design(train, m, s), train["home_covered"].to_numpy(), FIT_RIDGE)
        out.loc[test.index] = _predict(test, b, m, s)
    return out


def base_p_linemove(nfl: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=nfl.index, dtype=float)
    cols = list(BASE_FEATURES)
    for held in sorted(int(v) for v in nfl["season"].unique()):
        train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        m, s = standardize(train, cols)
        b = fit_logit_beta(design(train, cols, m, s), train["home_covered"].astype(float).to_numpy())
        out.loc[test.index] = predict_logit(test, cols, b, m, s)
    return out


def summarize_beta(betas: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(betas.values()))
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "n_negative": int((vals < 0).sum()),
        "n_folds": int(len(vals)),
    }


def main() -> None:
    cfb_all = load_cfb_population()

    nfl, _prov, _meta = load_extended_nfl_population()
    keep = nfl.dropna(subset=list(FIT_FEATURES)).reset_index(drop=True)
    nfl = keep
    p_base = base_p_accuracy(nfl)
    p_nested, acc_betas = nested_loso(nfl, cfb_all, list(FIT_FEATURES))
    nfl["base_pick"] = p_base.ge(0.5)
    nfl["nested_pick"] = p_nested.ge(0.5)
    nfl["base_correct"] = nfl["base_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["nested_correct"] = nfl["nested_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    sub = nfl.loc[nfl["season"].ge(2020)].reset_index(drop=True)
    acc_all = paired_accuracy_effect(nfl, "nested_correct", "base_correct")
    acc_sub = paired_accuracy_effect(sub, "nested_correct", "base_correct")

    lm, _prov2, _meta2 = load_line_move_population()
    p_lb = base_p_linemove(lm)
    p_ln, lm_betas = nested_loso(lm, cfb_all, list(BASE_FEATURES))
    lm["diff_lm"] = (
        np.where(p_ln.ge(0.5), 1.0, -1.0) - np.where(p_lb.ge(0.5), 1.0, -1.0)
    ) * lm["open_move"]
    lm_sub = lm.loc[lm["season"].ge(2020)].reset_index(drop=True)
    keys = (
        "mean_points",
        "season_block_interval_low",
        "season_block_interval_high",
        "season_block_probability_positive",
    )
    lm_all_cell = cell_stats("nested_vs_base_linemove_all", lm, "diff_lm", u5.SEED, u5.LM_DRAWS)
    lm_sub_cell = cell_stats("nested_vs_base_linemove_sub", lm_sub, "diff_lm", u5.SEED, u5.LM_DRAWS)

    result = {
        "accuracy_all": acc_all,
        "accuracy_2020_2025": acc_sub,
        "n_accuracy": len(nfl),
        "n_accuracy_subset": len(sub),
        "records": {
            "base": games_record(nfl["base_correct"]),
            "nested": games_record(nfl["nested_correct"]),
        },
        "accuracy_beta": summarize_beta(acc_betas),
        "linemove_all": {k: lm_all_cell[k] for k in keys},
        "linemove_2020_2025": {k: lm_sub_cell[k] for k in keys},
        "n_linemove": len(lm),
        "linemove_beta": summarize_beta(lm_betas),
        "accuracy_betas_by_fold": acc_betas,
        "look_count": 4,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/xlg09_unit6" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"artifact_dir={out_dir}")
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()

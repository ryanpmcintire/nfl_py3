import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3
from lead98_price_skew import BASE, fit, loso, metrics, predict, reliability
from mod57_p2o_how_wrong_is_the_line import design, fit_coef, team_index
from mod57_u3_deadline_fair_line import block_bootstrap, decisive

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.modeling import regular_season_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "base_label_noise"
SERVED = ROOT / "artifacts" / "pick_probability" / "20261111T011859Z" / "per_game.parquet"
SEASONS = tuple(range(2020, 2026))
PROFILE = "weak_stack"
BASE_ALPHA = 10.0
PROB_METHOD = "gaussian_median"
DRAWS = 10000
SEED = 20261116
FAMILY = "base_label_noise_hindsight"


def hindsight_labels(features):
    reg = regular_season_rows(features)
    reg = reg.loc[reg["result"].notna() & reg["spread_line"].notna()].copy()
    reg = reg.sort_values(["season", "game_id"]).reset_index(drop=True)
    index = team_index(reg)
    pred = np.full(len(reg), np.nan)
    for _, ids in reg.groupby("season").indices.items():
        x = design(reg.iloc[ids], index)
        y = reg["result"].to_numpy(float)[ids]
        for i in range(len(ids)):
            keep = np.arange(len(ids)) != i
            pred[ids[i]] = x[i] @ fit_coef(x[keep], y[keep])
    reg["hindsight_margin"] = pred
    reg["label_h"] = pred - reg["spread_line"].to_numpy(float)
    return reg[["game_id", "season", "week", "result", "spread_line", "ats_margin", "hindsight_margin", "label_h"]]


def label_diagnostics(lab):
    d = lab.loc[lab["season"].lt(SEASONS[0]) & lab["label_h"].notna() & lab["ats_margin"].notna()]
    x = d["label_h"].to_numpy(float)
    y = d["ats_margin"].to_numpy(float)
    slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    return {
        "games": len(d),
        "seasons": [int(d["season"].min()), int(d["season"].max())],
        "sd_real_ats_margin": float(y.std(ddof=1)),
        "sd_label_h": float(x.std(ddof=1)),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "slope_real_on_label": slope,
    }


def eval_season(features, lab, season):
    path = OUT / f"h_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ext = features.copy()
    swap = lab.loc[lab["season"].lt(season) & lab["label_h"].notna()].set_index("game_id")["label_h"]
    mask = ext["game_id"].isin(swap.index) & ext["ats_margin"].notna() & ext["season"].lt(season)
    ext.loc[mask, "ats_margin"] = ext.loc[mask, "game_id"].map(swap).to_numpy()
    unchanged = ext.loc[ext["season"].ge(season), "ats_margin"].fillna(-999.0).equals(
        features.loc[features["season"].ge(season), "ats_margin"].fillna(-999.0)
    )
    if not unchanged:
        raise SystemExit("test-season labels changed")
    cfg = {
        "feature_profile": PROFILE,
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": PROB_METHOD,
        "calibration_method": "none",
        "ridge_alpha": BASE_ALPHA,
        "base_label_noise_variant": f"hindsight-{season}",
    }
    f = ext.loc[ext["season"].le(season)]
    scored = clv.opener_pick_evaluation(
        u3.MARKET_ROOT, f, active_model_config=cfg, min_train_games=DEFAULT_MIN_TRAIN_GAMES
    )
    scored = scored.loc[scored["season"].eq(season)].copy()
    out = scored[["game_id", "season", "week", "home_cover_probability_at_open", "residual_at_open", "margin_vs_open"]].copy()
    out["swapped_training_rows"] = int(mask.sum())
    out.to_parquet(path, index=False)
    print("season", season, len(out), int(mask.sum()), flush=True)
    return out


def season_block(frame, p_new, p_old, rng):
    y = frame["home_covered"].to_numpy(float)
    seasons = frame["season"].to_numpy()
    uniq = np.unique(seasons)
    pn = np.clip(p_new, 1e-9, 1 - 1e-9)
    po = np.clip(p_old, 1e-9, 1 - 1e-9)
    parts = {
        "accuracy_points": 100.0 * (((pn >= 0.5) == (y == 1)).astype(float) - ((po >= 0.5) == (y == 1)).astype(float)),
        "log_loss": -(y * np.log(pn) + (1 - y) * np.log1p(-pn)) + (y * np.log(po) + (1 - y) * np.log1p(-po)),
        "brier": (pn - y) ** 2 - (po - y) ** 2,
    }
    n = np.array([np.sum(seasons == s) for s in uniq], float)
    draws = rng.integers(0, len(uniq), size=(DRAWS, len(uniq)))
    out = {}
    for name, v in parts.items():
        sums = np.array([v[seasons == s].sum() for s in uniq])
        d = sums[draws].sum(axis=1) / n[draws].sum(axis=1)
        better = d > 0 if name == "accuracy_points" else d < 0
        out[name] = {
            "delta_mean": float(v.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "probability_positive": float(better.mean()),
            "per_season": {str(s): float(v[seasons == s].mean()) for s in uniq},
        }
    return out


def slope_with_interval(x, y, seasons, rng):
    uniq = np.unique(seasons)
    idx = [np.flatnonzero(seasons == s) for s in uniq]

    def slope(rows):
        return float(np.cov(x[rows], y[rows])[0, 1] / np.var(x[rows], ddof=1))

    boots = []
    for draw in rng.integers(0, len(uniq), size=(2000, len(uniq))):
        rows = np.concatenate([idx[k] for k in draw])
        boots.append(slope(rows))
    return {"slope": slope(np.arange(len(x))), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    features = pd.read_parquet(u3.FEATURES)
    lab = hindsight_labels(features)
    lab.to_parquet(OUT / "labels.parquet", index=False)
    diag = label_diagnostics(lab)
    print(json.dumps(diag), flush=True)
    new = pd.concat([eval_season(features, lab, s) for s in SEASONS])
    served = pd.read_parquet(SERVED).reset_index(drop=True)
    g = served.merge(
        new.rename(columns={"home_cover_probability_at_open": "p_h", "residual_at_open": "res_h", "margin_vs_open": "mvo_h"})[
            ["game_id", "p_h", "res_h", "mvo_h"]
        ],
        on="game_id",
        how="left",
    )
    if g["p_h"].isna().any():
        raise SystemExit(f"unmatched served games {int(g['p_h'].isna().sum())}")
    y = g["home_covered"].to_numpy(float)
    base_oos, base_folds = loso(g, BASE)
    repro = float(np.max(np.abs(base_oos - g["out_of_season_home_probability"].to_numpy(float))))
    gh = g.copy()
    gh["model_logit"] = logit(g["p_h"].clip(1e-6, 1 - 1e-6))
    h_oos, h_folds = loso(gh, BASE)
    b_in = predict(g, BASE, *fit(g, BASE))
    h_in = predict(gh, BASE, *fit(gh, BASE))
    raw_base = 1.0 / (1.0 + np.exp(-g["model_logit"].to_numpy(float)))
    raw_h = g["p_h"].to_numpy(float)
    seasons = g["season"].to_numpy()
    mvo = g["margin_vs_open"].to_numpy(float)
    report = {
        "declared": "docs/lanes/base-label-noise-2026-11-16.md",
        "label_diagnostics_pre_2020": diag,
        "served_reproduction_max_abs": repro,
        "games": len(g),
        "served_stage": {
            "base": {"oos": metrics(base_oos, y), "in_sample": metrics(b_in, y)},
            "hindsight": {"oos": metrics(h_oos, y), "in_sample": metrics(h_in, y)},
            "season_block": season_block(g, h_oos, base_oos, rng),
            "week_block": block_bootstrap(g, h_oos, base_oos, rng),
            "changed_picks": decisive(h_oos, base_oos, y),
            "reliability": {"base": reliability(base_oos, y), "hindsight": reliability(h_oos, y)},
            "folds": {"base": base_folds, "hindsight": h_folds},
        },
        "model_alone": {
            "base": metrics(raw_base, y),
            "hindsight": metrics(raw_h, y),
            "season_block": season_block(g, raw_h, raw_base, rng),
            "changed_picks": decisive(raw_h, raw_base, y),
        },
        "residual_calibration_slope_on_real_opener_margin": {
            "base": slope_with_interval(g["residual_at_open"].to_numpy(float), mvo, seasons, rng),
            "hindsight": slope_with_interval(g["res_h"].to_numpy(float), mvo, seasons, rng),
            "margin_vs_open_match_max_abs": float(np.nanmax(np.abs(g["mvo_h"].to_numpy(float) - mvo))),
            "corr_residuals": float(np.corrcoef(g["residual_at_open"], g["res_h"])[0, 1]),
            "sd_residual": {"base": float(g["residual_at_open"].std()), "hindsight": float(g["res_h"].std())},
        },
    }
    g.assign(p_served_base=base_oos, p_served_h=h_oos)[
        ["game_id", "season", "week", "home_covered", "model_logit", "p_h", "res_h", "p_served_base", "p_served_h"]
    ].to_parquet(OUT / "per_game.parquet", index=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("served_reproduction_max_abs", "games")}), flush=True)
    print(json.dumps(report["served_stage"]["season_block"], indent=1), flush=True)


if __name__ == "__main__":
    main()

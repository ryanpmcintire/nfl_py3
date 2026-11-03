import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod35_apparent_temp as m

COLD_GAP = 25.0
CONTROLS = ("nrm_home", "week", "wk_x_nrm_home", "cold_flag")


def offset_fit(xtr, ytr, ztr):
    k = xtr.shape[1]

    def f(w):
        s = np.clip(ztr + w[0] + xtr @ w[1:], -30, 30)
        return np.sum(np.logaddexp(0, s) - ytr * s)

    return minimize(f, np.zeros(k + 1), method="BFGS").x


def loso_cols(d, cols, seasons, offset):
    p = np.full(len(d), np.nan)
    bx = []
    y = d.y.to_numpy()
    for s in seasons:
        tr = (d.season != s).to_numpy()
        te = ~tr
        X = d[cols].to_numpy(float) if cols else np.zeros((len(d), 0))
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Xs = (X - mu) / sd if cols else X
        if offset:
            w = offset_fit(Xs[tr], y[tr], d.z0.to_numpy()[tr])
            p[te] = 1 / (1 + np.exp(-(d.z0.to_numpy()[te] + w[0] + Xs[te] @ w[1:])))
            bx.append(w[1:].tolist())
        else:
            F = np.column_stack([d.z0.to_numpy(), Xs])
            mm = m.fit(F[tr], y[tr])
            p[te] = mm.predict_proba(F[te])[:, 1]
            bx.append(mm.coef_[0][1:].tolist())
    return p, np.array(bx)


def cell(d, cols, xcol, seasons, offset, rng, base_cols=None):
    base_cols = base_cols or []
    pb, _ = loso_cols(d, base_cols, seasons, offset)
    pf, bx = loso_cols(d, [*base_cols, xcol], seasons, offset)
    y = d.y.to_numpy()
    mb, mf = m.metrics(y, pb), m.metrics(y, pf)
    b = m.wk_boot(d, pb, pf, rng)
    flip = (pb > 0.5) != (pf > 0.5)
    j = len(base_cols)
    return {
        "n": len(d),
        "base": mb,
        "with_term": mf,
        "acc_gain": b["acc_gain"],
        "ll_gain": b["ll_gain"],
        "decisive_term_vs_base": [
            int(((pf > 0.5) == (y == 1))[flip].sum()),
            int(((pb > 0.5) == (y == 1))[flip].sum()),
            int(flip.sum()),
        ],
        "coef_x_mean": float(bx[:, j].mean()),
        "coef_x_sd": float(bx[:, j].std(ddof=1)),
        "coef_x_pos_folds": int((bx[:, j] > 0).sum()),
    }


def main():
    rng = np.random.default_rng(36)
    fc = pd.read_parquet(m.FC)
    fc = fc.loc[fc.fetch_status == "ok"].copy()
    fc["fc_at"] = m.apparent(fc.forecast_temp_f.to_numpy(), fc.forecast_wind_mph.to_numpy())
    rec = fc.loc[
        fc.roof.isin(m.OUTDOOR) & fc.actual_temp_f.notna() & fc.actual_wind_mph.notna()
    ].copy()
    rec["home"] = rec.game_id.str.split("_").str[3]
    rec["app"] = m.apparent(rec.actual_temp_f.to_numpy(), rec.actual_wind_mph.to_numpy())
    rec = rec[["home", "season", "week", "app"]]
    chosen = m.select_half_widths(rec)
    base = m.build_base().merge(fc[["game_id", "roof", "fc_at"]], on="game_id", how="left")
    base["nrm_away"] = np.nan
    base["nrm_home"] = np.nan
    for s in range(m.FIRST, m.LAST + 1):
        k = chosen[s][0]
        mk = (base.season == s).to_numpy()
        for col, tcol in (("nrm_away", "away"), ("nrm_home", "home")):
            keys = list(
                zip(base.loc[mk, tcol], base.loc[mk, "season"], base.loc[mk, "week"], strict=True)
            )
            base.loc[mk, col] = m.normals(rec, keys, k)
    base["gap"] = base.nrm_away - base.fc_at
    base["x"] = base.gap.abs()
    base["cold_flag"] = (base.gap >= COLD_GAP).astype(float)
    base["wk_x_nrm_home"] = base.week * base.nrm_home
    ok = base.x.notna() & base.nrm_home.notna()
    po = base.loc[ok & base.roof.isin(m.OUTDOOR)].reset_index(drop=True)
    pi = base.loc[ok & base.roof.isin(m.INDOOR)].reset_index(drop=True)
    seasons = sorted(set(po.season) & set(pi.season))
    po = po.loc[po.season.isin(seasons)].reset_index(drop=True)
    pi = pi.loc[pi.season.isin(seasons)].reset_index(drop=True)
    res = {"n_outdoor": len(po), "n_dome": len(pi)}
    srv = pd.read_parquet(m.SERVED)[
        ["game_id", "correct_at_open_probability_rule", "correct_at_open"]
    ]
    s25 = po.loc[po.season >= 2020].merge(srv, on="game_id", how="left")
    yy = s25.y.to_numpy()
    res["served_vs_subset"] = {
        "n_2020_25": len(s25),
        "served_correct_at_open_rule": float(
            s25.correct_at_open_probability_rule.astype(float).mean()
        ),
        "served_correct_at_open": float(s25.correct_at_open.astype(float).mean()),
        "p0_acc_this_subset_2020_25": float(((s25.p0 > 0.5) == (yy == 1)).mean()),
        "p0_acc_2011_19_subset": float(
            ((po.loc[po.season < 2020].p0 > 0.5) == (po.loc[po.season < 2020].y == 1)).mean()
        ),
        "p0_acc_all_subset": m.metrics(po.y.to_numpy(), po.p0.to_numpy())["acc"],
        "intercept_only": float(max(po.y.mean(), 1 - po.y.mean())),
    }
    res["cold_flag"] = {
        "n_flag_outdoor": int(po.cold_flag.sum()),
        "flag_sides": "unserved flip overlay, not in the fitted served probability",
    }
    cells = {}
    for off in (False, True):
        tag = "offset_raw_served_p0" if off else "recal_base"
        cells[f"{tag}/x_abs"] = cell(po, [], "x", seasons, off, rng)
        cells[f"{tag}/x_abs+controls"] = cell(
            po, [], "x", seasons, off, rng, base_cols=list(CONTROLS)
        )
    po["gap_pos"] = po.gap.clip(lower=0)
    po["gap_neg"] = (-po.gap).clip(lower=0)
    cells["recal_base/signed_gap+controls"] = cell(
        po, [], "gap", seasons, False, rng, base_cols=list(CONTROLS)
    )
    cells["recal_base/gap_pos(visitor warmer-climate)+gap_neg+controls"] = cell(
        po, [], "gap_pos", seasons, False, rng, base_cols=[*list(CONTROLS), "gap_neg"]
    )
    cells["recal_base/gap_neg(visitor colder-climate)+gap_pos+controls"] = cell(
        po, [], "gap_neg", seasons, False, rng, base_cols=[*list(CONTROLS), "gap_pos"]
    )
    cells["dome/x_abs"] = cell(pi, [], "x", seasons, False, rng)
    cells["dome/x_abs+controls(home climate,week)"] = cell(
        pi, [], "x", seasons, False, rng, base_cols=["nrm_home", "week", "wk_x_nrm_home"]
    )
    res["cells"] = cells
    pi["xs"] = (pi.x - pi.x.mean()) / pi.x.std()
    res["dome_corr"] = {
        "corr_x_week": float(pi.x.corr(pi.week.astype(float))),
        "corr_x_y": float(pi.x.corr(pi.y.astype(float))),
        "corr_x_fc_at": float(pi.x.corr(pi.fc_at)),
        "corr_x_nrm_away": float(pi.x.corr(pi.nrm_away)),
        "n_home_teams": int(pi.home.nunique()),
        "home_team_eta2_of_x": float(
            1
            - ((pi.x - pi.groupby("home").x.transform("mean")) ** 2).sum()
            / ((pi.x - pi.x.mean()) ** 2).sum()
        ),
        "x_by_home": pi.groupby("home")
        .x.agg(["mean", "size"])
        .round(2)
        .sort_values("mean")
        .to_dict("index"),
        "x_mean_by_week_bucket": {
            str(k): float(v)
            for k, v in pi.assign(wb=pd.cut(pi.week, [0, 4, 8, 12, 16, 22]))
            .groupby("wb", observed=True)
            .x.mean()
            .items()
        },
    }
    out = m.OUT / "unit2.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    s = json.loads(json.dumps(res, default=str))
    s["dome_corr"].pop("x_by_home")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    with threadpool_limits(1):
        main()

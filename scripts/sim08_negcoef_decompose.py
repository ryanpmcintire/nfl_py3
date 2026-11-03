import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim08_evaluate as ev

base = ev.base
OUT = ev.RUN_ROOT / "main" / "negcoef_decompose.json"


def logit_fit(X, y):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e10, max_iter=2000).fit(X, y)
    return m, m.predict_proba(X)[:, 1]


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ols(X, t):
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, t, rcond=None)
    r = t - A @ b
    return b, 1 - r.var() / t.var(), r


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    from nfl_ats.schedule_flag_features import default_opener_lines

    frame = base.load_games()
    lines = default_opener_lines(frame[["game_id", "season", "week"]])
    frame = frame.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    cov, _push, loss = base.baseline_three_way(frame)
    served = base.safe_logit(cov / np.clip(cov + loss, base.EPS, None))
    m3 = ev.market_hist_three_way(frame)
    market = base.safe_logit(m3[0] / np.clip(m3[0] + m3[2], base.EPS, None))
    S = ev.load_arm("main", "primary", frame)
    line = frame["line"].to_numpy(dtype=float)
    c, _, l_ = ev.sim_counts(S["margins"], line)
    sim = base.safe_logit((c + 0.5) / (c + l_ + 1.0))
    cols = {"served": served, "sim": sim, "market": market}
    rows = {}
    for s_ in base.GRADED_SEASONS:
        tr = (seasons != s_) & nonpush
        te = (seasons == s_) & nonpush
        r = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        r["corr_sim_served_train"] = corr(sim[tr], served[tr])
        r["corr_sim_served_test"] = corr(sim[te], served[te])
        r["corr_sim_market_train"] = corr(sim[tr], market[tr])
        r["corr_served_market_train"] = corr(served[tr], market[tr])
        r["sd_train_sim_served_market"] = [float(np.std(v[tr])) for v in (sim, served, market)]

        def coefs(names, mask=tr):
            m, _ = logit_fit(np.column_stack([cols[n] for n in names])[mask], y[mask])
            return [float(v) for v in m.coef_[0]]

        r["solo_sim_coef"] = coefs(["sim"])[0]
        r["solo_served_coef"] = coefs(["served"])[0]
        r["solo_market_coef"] = coefs(["market"])[0]
        r["blend_served_sim"] = coefs(["served", "sim"])
        r["blend_served_market"] = coefs(["served", "market"])
        r["blend_served_sim_market"] = coefs(["served", "sim", "market"])
        r["blend_sim_market"] = coefs(["sim", "market"])
        ms, _ = logit_fit(served[tr][:, None], y[tr])
        res_y = y[tr] - ms.predict_proba(served[tr][:, None])[:, 1]
        _, _, sim_res = ols(served[tr][:, None], sim[tr])
        r["partial_corr_sim_y_given_served_train"] = corr(sim_res, res_y)
        res_te = y[te] - ms.predict_proba(served[te][:, None])[:, 1]
        b_te, _, _ = ols(served[tr][:, None], sim[tr])
        sim_res_te = sim[te] - (b_te[0] + b_te[1] * served[te])
        r["partial_corr_sim_y_given_served_test"] = corr(sim_res_te, res_te)
        b, r2, _ = ols(np.column_stack([served[tr], market[tr]]), sim[tr])
        r["sim_on_served_market"] = {
            "intercept": float(b[0]),
            "served": float(b[1]),
            "market": float(b[2]),
            "r2": float(r2),
        }
        _, r2s, _ = ols(served[tr][:, None], sim[tr])
        r["sim_on_served_r2"] = float(r2s)
        _, _, sim_res2 = ols(np.column_stack([served[tr], market[tr]]), sim[tr])
        r["partial_corr_simresid2_y_train"] = corr(sim_res2, res_y)
        lls = {}
        for name, names in (
            ("served", ["served"]),
            ("served_sim", ["served", "sim"]),
            ("served_market", ["served", "market"]),
            ("served_sim_market", ["served", "sim", "market"]),
        ):
            XA = np.column_stack([cols[n] for n in names])
            m, _ = logit_fit(XA[tr], y[tr])
            p = m.predict_proba(XA[te])[:, 1]
            lls[name] = float(ll(p, y[te]).mean())
        r["test_logloss"] = lls
        rows[int(s_)] = r
    pooled = {}
    n_all = sum(rows[s_]["n_test"] for s_ in rows)
    for name in ("served", "served_sim", "served_market", "served_sim_market"):
        tot = sum(rows[s_]["test_logloss"][name] * rows[s_]["n_test"] for s_ in rows)
        pooled[name] = float(tot / n_all)
    allm = nonpush
    XA = np.column_stack([np.ones(allm.sum()), sim[allm]])
    mm, _ = logit_fit(sim[allm][:, None], y[allm])
    pr = mm.predict_proba(sim[allm][:, None])[:, 1]
    A = np.column_stack([np.ones(allm.sum()), sim[allm]])
    H = (A * (pr * (1 - pr))[:, None]).T @ A
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    pooled["solo_sim_pooled"] = {"coef": float(mm.coef_[0][0]), "wald_se": se, "n": int(allm.sum())}
    print(pooled["solo_sim_pooled"], XA.shape)
    out = {"folds": rows, "pooled_test_logloss_weighted": pooled}
    OUT.write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    flat = pd.DataFrame(
        {
            s_: {
                "corr_sim_served": r["corr_sim_served_test"],
                "corr_sim_served_tr": r["corr_sim_served_train"],
                "corr_sim_market_tr": r["corr_sim_market_train"],
                "corr_served_market_tr": r["corr_served_market_train"],
                "solo_sim": r["solo_sim_coef"],
                "solo_served": r["solo_served_coef"],
                "solo_market": r["solo_market_coef"],
                "bl_served": r["blend_served_sim"][0],
                "bl_sim": r["blend_served_sim"][1],
                "pcorr_tr": r["partial_corr_sim_y_given_served_train"],
                "pcorr_te": r["partial_corr_sim_y_given_served_test"],
                "pcorr_res2_tr": r["partial_corr_simresid2_y_train"],
                "sim~srv R2": r["sim_on_served_r2"],
                "sim~srv+mkt R2": r["sim_on_served_market"]["r2"],
                "sim~srv b": r["sim_on_served_market"]["served"],
                "sim~mkt b": r["sim_on_served_market"]["market"],
                "sm_served": r["blend_served_sim_market"][0],
                "sm_sim": r["blend_served_sim_market"][1],
                "sm_market": r["blend_served_sim_market"][2],
                "sk_served": r["blend_served_market"][0],
                "sk_market": r["blend_served_market"][1],
                "ll_served": r["test_logloss"]["served"],
                "ll_s_sim": r["test_logloss"]["served_sim"],
                "ll_s_mkt": r["test_logloss"]["served_market"],
                "ll_s_sim_mkt": r["test_logloss"]["served_sim_market"],
            }
            for s_, r in rows.items()
        }
    )
    print(flat.round(4).to_string())
    print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()

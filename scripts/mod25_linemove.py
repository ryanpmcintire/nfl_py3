import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod24_u1 as u1
import mod25_pipeline as mp

OUT = mp.OUT / "linemove"
ARMS = ("ridge_nomkt", "hgb_nomkt")
DRAWS = 20000
MARKET_PAT = ("spread", "total_line", "close", "moneyline", "handle")


def verify():
    full, nomkt = mp.v2_cols()
    bad = [c for c in nomkt if any(k in c for k in MARKET_PAT)]
    syn = pd.read_parquet(mp.SYN / "features.parquet")
    const = [c for c in nomkt if c in syn.columns and float(syn[c].std(ddof=0)) < 1e-9]
    return {"nomkt_cols": len(nomkt), "market_named_cols_in_nomkt": bad, "excluded": [c for c in full if c not in nomkt], "n_constant_in_synthetic_zero_weight": len(const), "constant_cols": const}


def aligned_base_p():
    feats = pd.read_parquet(REPO / "data" / "processed" / "game_features_weak_stack.parquet")
    pop = u1.proxy_population(feats)
    frame = u1.regular_season_rows(feats)
    pb = pd.read_parquet(mp.OUT / "proxy_base.parquet")
    rows = []
    for (s, w), g in pop.groupby(["season", "week"], sort=True):
        g = g.loc[(g["result"] - g["open_line"]).ne(0)]
        ids = frame.loc[frame["game_id"].isin(set(g["game_id"])), "game_id"].astype(str).tolist()
        ps = pb.loc[pb["season"].eq(s) & pb["week"].eq(w)]
        if len(ps) == 0:
            continue
        assert len(ps) == len(ids), (s, w)
        rows.append(pd.DataFrame({"game_id": ids, "p_base": ps["p"].to_numpy(), "pick_home": ps["pick_home"].to_numpy(), "correct": ps["correct"].to_numpy()}))
    a = pd.concat(rows, ignore_index=True)
    chk = a.merge(pop[["game_id", "result", "open_line"]], on="game_id")
    m = (chk["result"] - chk["open_line"]).to_numpy()
    ok = m != 0
    truth = np.where(chk["pick_home"].to_numpy(), m > 0, m < 0)
    agree = float((truth[ok] == chk["correct"].to_numpy()[ok].astype(bool)).mean())
    return a, agree


def build():
    d = pd.read_parquet(mp.OUT / "real_distilled.parquet")[["game_id", "season", *ARMS]]
    pg = pd.read_parquet(mp.OPENER / "per_game.parquet")
    pg = pg.loc[pg["season"].between(2020, 2025), ["game_id", "tue_open_home_spread", "close_home_spread", "home_cover_probability_at_open"]]
    a = d.merge(pg, on="game_id")
    a["open"] = a["tue_open_home_spread"]
    a["p_base"] = a["home_cover_probability_at_open"]
    a["era"] = "2020_2025"
    sbr = pd.read_parquet(REPO / "data" / "processed" / "sbr_odds.parquet").dropna(subset=["game_id", "open_home_spread", "close_home_spread"]).drop_duplicates("game_id")
    b = d.loc[d["season"].between(2011, 2019)].merge(sbr[["game_id", "open_home_spread", "close_home_spread"]], on="game_id")
    b["open"] = b["open_home_spread"]
    bp, agree = aligned_base_p()
    b = b.merge(bp[["game_id", "p_base"]], on="game_id", how="left")
    b["era"] = "2011_2019"
    cols = ["game_id", "season", "era", "open", "close_home_spread", "p_base", *ARMS]
    x = pd.concat([a[cols], b[cols]], ignore_index=True)
    x["move"] = x["close_home_spread"] - x["open"]
    for arm in ARMS:
        x["q_" + arm] = x[arm] - x["open"]
    pc = x["p_base"].clip(0.02, 0.98)
    x["ctrl"] = np.log(pc / (1 - pc))
    return x, agree


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def loso(df, qcol, use_ctrl, pooled):
    rows = []
    for s in sorted(df["season"].unique()):
        tr = df.loc[df["season"].ne(s)]
        te = df.loc[df["season"].eq(s)]
        feats = [qcol] + (["ctrl"] if use_ctrl else [])
        Xtr = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        if pooled:
            Xtr = np.hstack([Xtr, (tr["era"] == "2020_2025").to_numpy(float)[:, None]])
            Xte = np.hstack([Xte, (te["era"] == "2020_2025").to_numpy(float)[:, None]])
        beta = ols(Xtr, tr["move"].to_numpy(float))
        p = beta[0] + Xte @ beta[1:]
        y = te["move"].to_numpy(float)
        rows.append({"season": int(s), "n": len(te), "b_q": float(beta[1]), "b_ctrl": float(beta[2]) if use_ctrl else None, "corr": float(np.corrcoef(p, y)[0, 1]), "mae_model": float(np.abs(y - p).mean()), "mae_zero": float(np.abs(y).mean()), "d_abs": np.abs(y) - np.abs(y - p)})
    return rows


def boot(per_season_vals, rng):
    k = len(per_season_vals)
    cat = [np.asarray(v, float) for v in per_season_vals]
    stat = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, k, k)
        stat[i] = np.concatenate([cat[j] for j in idx]).mean()
    return {"mean": float(np.concatenate(cat).mean()), "lo": float(np.percentile(stat, 2.5)), "hi": float(np.percentile(stat, 97.5)), "p_pos": float((stat > 0).mean())}


def summarize(rows, rng):
    tbl = [{k: v for k, v in r.items() if k != "d_abs"} for r in rows]
    bs = [r["b_q"] for r in rows]
    return {"per_season": tbl, "slope_positive_seasons": int(sum(b > 0 for b in bs)), "n_seasons": len(bs), "mean_b_q": float(np.mean(bs)), "mean_heldout_corr": float(np.mean([r["corr"] for r in rows])), "mae_gain_vs_zero": boot([r["d_abs"] for r in rows], rng), "n": int(sum(r["n"] for r in rows))}


def direction(df, qcol, rng):
    d = df.loc[df[qcol].ne(0) & df["move"].notna()].copy()
    d["tw"] = np.sign(d[qcol].to_numpy()) * d["move"].to_numpy(float)
    ss = [g["tw"].to_numpy() for _, g in d.groupby("season")]
    moved = d.loc[d["move"].ne(0)]
    return {"n": len(d), "mean_signed_move_toward_q": boot(ss, rng), "share_toward_q_all": float((d["tw"] > 0).mean()), "share_away_all": float((d["tw"] < 0).mean()), "share_zero": float((d["tw"] == 0).mean()), "share_toward_q_among_moved": float((moved["tw"] > 0).mean()), "positive_seasons": int(sum(s.mean() > 0 for s in ss)), "n_seasons": len(ss)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261002)
    x, agree = build()
    res = {"verify": verify(), "base_p_alignment_pick_agreement": agree}
    res["pop"] = {e: int(g["move"].notna().sum()) for e, g in x.groupby("era")}
    res["pop_with_base_p"] = {e: int(g["ctrl"].notna().sum()) for e, g in x.groupby("era")}
    res["move_sd"] = {e: float(g["move"].std()) for e, g in x.groupby("era")}
    looks = 0
    for arm in ARMS:
        q = "q_" + arm
        res[arm] = {}
        for label, sub, pooled in (("2011_2019", x.loc[x["era"].eq("2011_2019")], False), ("2020_2025", x.loc[x["era"].eq("2020_2025")], False), ("pooled", x, True)):
            sub = sub.dropna(subset=["move", q])
            m = sub.dropna(subset=["ctrl"])
            res[arm][label] = {
                "A_q_full": summarize(loso(sub, q, False, pooled), rng),
                "A_q_matched": summarize(loso(m, q, False, pooled), rng),
                "B_q_plus_base_edge_matched": summarize(loso(m, q, True, pooled), rng),
                "direction_full": direction(sub, q, rng),
            }
            looks += 4
    res["looks"] = looks
    (OUT / "linemove.json").write_text(json.dumps(res, indent=1, default=float))
    x.drop(columns=["close_home_spread"]).to_parquet(OUT / "linemove_frame.parquet")
    for arm in ARMS:
        for label in ("2011_2019", "2020_2025", "pooled"):
            r = res[arm][label]
            a = r["A_q_full"]
            b = r["B_q_plus_base_edge_matched"]
            am = r["A_q_matched"]
            dr = r["direction_full"]
            g = a["mae_gain_vs_zero"]
            print(arm, label, "A n", a["n"], "b+", a["slope_positive_seasons"], "/", a["n_seasons"], "meanb %.3f corr %.3f" % (a["mean_b_q"], a["mean_heldout_corr"]), "MAEgain %.4f [%.4f,%.4f] P+ %.3f" % (g["mean"], g["lo"], g["hi"], g["p_pos"]))
            gb = b["mae_gain_vs_zero"]
            print("   matched A n", am["n"], "b+", am["slope_positive_seasons"], "corr %.3f gain %.4f P+ %.3f" % (am["mean_heldout_corr"], am["mae_gain_vs_zero"]["mean"], am["mae_gain_vs_zero"]["p_pos"]), "| B b+", b["slope_positive_seasons"], "corr %.3f gain %.4f [%.4f,%.4f] P+ %.3f" % (b["mean_heldout_corr"], gb["mean"], gb["lo"], gb["hi"], gb["p_pos"]))
            m_ = dr["mean_signed_move_toward_q"]
            print("   dir n", dr["n"], "toward-q mean %.4f [%.4f,%.4f] P+ %.3f" % (m_["mean"], m_["lo"], m_["hi"], m_["p_pos"]), "share toward %.3f away %.3f zero %.3f; among moved %.3f; seasons+ %d/%d" % (dr["share_toward_q_all"], dr["share_away_all"], dr["share_zero"], dr["share_toward_q_among_moved"], dr["positive_seasons"], dr["n_seasons"]))
    print(json.dumps({k: res[k] for k in ("verify", "base_p_alignment_pick_agreement", "pop", "pop_with_base_p", "move_sd", "looks")}, indent=1)[:3500])


if __name__ == "__main__":
    main()

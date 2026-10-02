import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
R2 = REPO / "artifacts" / "mod25e3" / "r2gap"
OUT = REPO / "artifacts" / "mod25e3" / "gain"
CLASSES = ["garb", "lead4", "aftertov", "rz_td", "rz_fg", "rz_other", "nz_td", "nz_fg", "nz_other"]


def real_drives(seasons):
    import mod25d_variance as dv
    import mod25_generator as gen

    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    p = dv.sim.load_reg_seasons(tuple(seasons))
    p = p[p["posteam"].notna() & p["epa"].notna() & p["fixed_drive"].notna() & (p["qtr"] <= 4) & p["play_type"].isin(["run", "pass"])].copy()
    p = p.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    h = (p["posteam"] == p["home_team"]).to_numpy()
    sdh = np.where(h, p["score_differential"], -p["score_differential"])
    wp = np.where(h, p["wp"], 1 - p["wp"])
    p = p.assign(offhome=h.astype(int), sdh=sdh, wph=wp)
    gk = p.groupby(["game_id", "fixed_drive"], sort=False)
    D = pd.DataFrame({
        "game_id": gk["game_id"].first(), "offhome": gk["offhome"].first(), "sd0": gk["sdh"].first(), "q0": gk["qtr"].first(), "gsr0": gk["game_seconds_remaining"].first(),
        "wph": gk["wph"].first(), "ylmin": gk["yardline_100"].min(), "res": gk["fixed_drive_result"].first(), "season": gk["season"].first(),
        "epa": gk["epa"].sum(), "n": gk.size(),
    }).reset_index(drop=True)
    D["tov"] = D["res"].isin(["Turnover"]).astype(int)
    D["oc"] = np.where(D["res"] == "Touchdown", "td", np.where(D["res"] == "Field goal", "fg", "other"))
    D["cl"] = D["season"].astype(str)
    rg = gen.real_games(tuple(seasons))
    G = rg[rg["game_id"].isin(D["game_id"])].copy()
    G["final"] = G["home_score"] - G["away_score"]
    G["cl"] = G["season"].astype(str)
    return D, G.reset_index(drop=True)


def sim_drives():
    f = pd.read_parquet(R2 / "feat_plays.parquet")
    f["k"] = f.groupby("game_id").cumcount()
    fr = []
    for fp in sorted((R2 / "feat_sim").glob("play_*_*.parquet")):
        _, w, s = fp.stem.split("_")
        w, s = int(w), int(s)
        if s < 2:
            continue
        d = pd.read_parquet(fp)
        d = d[d["qtr"] <= 4].copy()
        d["game_id"] = [f"w{w:04d}s{s:02d}g{int(g):03d}" for g in d["g"]]
        d["w"], d["s"] = w, s
        d["kept"] = d["code"].isin([0, 1])
        d["k"] = np.where(d["kept"], d.groupby("game_id")["kept"].cumsum() - 1, -1)
        fr.append(d)
    P = pd.concat(fr, ignore_index=True)
    P = P.merge(f[["game_id", "k", "epa", "interception", "fumble_lost"]], on=["game_id", "k"], how="left")
    print("sim epa matched on kept plays", float(P.loc[P["k"] >= 0, "epa"].notna().mean()), flush=True)
    P["tovp"] = ((P["interception"].fillna(0) == 1) | (P["fumble_lost"].fillna(0) == 1)).astype(int)
    P["sdh"] = np.where(P["offhome"] == 1, P["sd"], -P["sd"])
    prev = P.groupby("game_id")["offhome"].shift(1)
    pq = P.groupby("game_id")["qtr"].shift(1)
    newd = prev.isna() | (prev != P["offhome"]) | ((pq <= 2) & (P["qtr"] >= 3))
    P["drv"] = newd.cumsum()
    gk = P.groupby("drv", sort=False)
    D = pd.DataFrame({
        "game_id": gk["game_id"].first(), "offhome": gk["offhome"].first(), "sd0": gk["sdh"].first(), "q0": gk["qtr"].first(), "gsr0": gk["gsr"].first(),
        "ylmin": gk["yl"].min(), "po": gk["po"].sum(), "lastcode": gk["code"].last(), "tov": gk["tovp"].last(),
        "epa": gk["epa"].sum(), "n": gk.size(),
    }).reset_index(drop=True)
    D["oc"] = np.where(D["po"] >= 6, "td", np.where((D["po"] == 3) & (D["lastcode"] == 3), "fg", "other"))
    gm = pd.read_parquet(R2 / "feat_games.parquet")
    G = gm[gm["game_id"].isin(D["game_id"])].copy()
    G["final"] = G["home_score"] - G["away_score"]
    G["cl"] = G["game_id"].str[:9]
    D["cl"] = D["game_id"].str[:9]
    return D, G.reset_index(drop=True)


def wp_model(D, G):
    from sklearn.linear_model import LogisticRegression

    def feats(x):
        t = np.sqrt(np.maximum(x["gsr0"].to_numpy(float), 0) / 60.0 + 1.0)
        return np.column_stack([x["sd0"], x["sd0"] / t, x["sd0"] / (t * t)])

    fin = D["game_id"].map(G.set_index("game_id")["final"]).to_numpy(float)
    ok = fin != 0
    m = LogisticRegression(C=1e3, max_iter=2000).fit(feats(D)[ok], (fin[ok] > 0).astype(int))
    return lambda x: m.predict_proba(feats(x))[:, 1]


def finish(D, G, fitted_wp):
    D = D.sort_values(["game_id"], kind="stable").reset_index(drop=True)
    nxt = D.groupby("game_id")["sd0"].shift(-1)
    fin = D["game_id"].map(G.set_index("game_id")["final"])
    D["dm"] = np.where(nxt.isna(), fin, nxt) - D["sd0"]
    D["wpf"] = fitted_wp(D)
    garb = (D["wpf"] < 0.05) | (D["wpf"] > 0.95)
    off_lead = np.where(D["offhome"] == 1, D["sd0"] > 0, D["sd0"] < 0)
    prevtov = D.groupby("game_id")["tov"].shift(1).fillna(0) == 1
    rz = D["ylmin"] <= 20
    D["cls"] = np.where(garb, "garb", np.where((D["q0"] == 4) & off_lead, "lead4", np.where(prevtov, "aftertov", np.where(rz, "rz_" + D["oc"], "nz_" + D["oc"]))))
    D["epah"] = np.where(D["offhome"] == 1, D["epa"], -D["epa"])
    return D


def table(D, G):
    h = D.assign(side=np.where(D["offhome"] == 1, "h", "a"))
    pv = h.pivot_table(index="game_id", columns="side", values=["epa", "n"], aggfunc="sum")
    e = (pv["epa"]["h"] - pv["epa"]["a"]).rename("e")
    comp = D.pivot_table(index="game_id", columns="cls", values="dm", aggfunc="sum").reindex(columns=CLASSES).fillna(0.0)
    ecomp = D.pivot_table(index="game_id", columns="cls", values="epah", aggfunc="sum").reindex(columns=CLASSES).fillna(0.0).add_prefix("epa_")
    T = G.set_index("game_id")[["home_team", "away_team", "final", "cl"]].join(e).join(comp).join(ecomp)
    T["hteam"] = T["cl"] + "_" + T["home_team"]
    T["ateam"] = T["cl"] + "_" + T["away_team"]
    rows = pd.concat([pd.DataFrame({"t": T["hteam"], "n": T["e"]}), pd.DataFrame({"t": T["ateam"], "n": -T["e"]})])
    S = rows.groupby("t")["n"].mean()
    T["p"] = T["hteam"].map(S) - T["ateam"].map(S)
    T["tr"] = T["e"] - T["p"]
    T["sumc"] = T[CLASSES].sum(axis=1)
    return T.dropna(subset=["e"]).reset_index()


def gains(T):
    X = np.column_stack([np.ones(len(T)), T["p"], T["tr"]])
    Y = T[["final"] + CLASSES].to_numpy(float)
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def boot(T, rng, n=400):
    cl = T["cl"].to_numpy()
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    out = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        out.append(gains(T.iloc[np.concatenate([idx[c] for c in pick])]))
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    Dr, Gr = real_drives(range(2018, 2026))
    Ds, Gs = sim_drives()
    wpf = wp_model(Dr, Gr)
    Dr = finish(Dr, Gr, wpf)
    Ds = finish(Ds, Gs, wpf)
    Tr, Ts = table(Dr, Gr), table(Ds, Gs)
    print("games", len(Tr), len(Ts), "margin check max abs", float((Tr["sumc"] - Tr["final"]).abs().max()), float((Ts["sumc"] - Ts["final"]).abs().max()), flush=True)
    br, bs = gains(Tr), gains(Ts)
    Br, Bs = boot(Tr, rng), boot(Ts, rng)
    names = ["final"] + CLASSES
    res = {}
    for j, nm in enumerate(names):
        row = {}
        for k, lab in ((1, "persist"), (2, "transient")):
            d = Bs[:, k, j] - Br[:, k, j]
            row[lab] = {"real": float(br[k, j]), "sim": float(bs[k, j]), "diff": float(bs[k, j] - br[k, j]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "probability_positive": float((d > 0).mean())}
        res[nm] = row
    rr = Br[:, 1, 0] / Br[:, 2, 0]
    rs = Bs[:, 1, 0] / Bs[:, 2, 0]
    res["ratio"] = {"real": float(br[1, 0] / br[2, 0]), "sim": float(bs[1, 0] / bs[2, 0]), "diff_ci": [float(np.percentile(rs - rr, 2.5)), float(np.percentile(rs - rr, 97.5))], "probability_positive": float(((rs - rr) > 0).mean())}
    res["var"] = {k + "_" + n: float(T[k].var()) for k in ("e", "p", "tr") for n, T in (("real", Tr), ("sim", Ts))}
    ec = [c for c in Tr if c.startswith("epa_")]
    res["epa_share_var"] = {c: [float(Tr[c].var() / Tr[ec].sum(axis=1).var()), float(Ts[c].var() / Ts[ec].sum(axis=1).var())] for c in ec}
    res["class_margin_var"] = {c: [float(Tr[c].var()), float(Ts[c].var())] for c in CLASSES}
    res["class_share_drives"] = {c: [float((Dr["cls"] == c).mean()), float((Ds["cls"] == c).mean())] for c in CLASSES}
    res["drives_per_game"] = [float(len(Dr) / len(Tr)), float(len(Ds) / len(Ts))]
    (OUT / "gain.json").write_text(json.dumps(res, indent=1))
    for nm in names:
        r = res[nm]
        a, b = r["persist"], r["transient"]
        print(f"{nm:9s} P real {a['real']:7.2f} sim {a['sim']:7.2f} d {a['diff']:6.2f} [{a['ci'][0]:6.2f},{a['ci'][1]:6.2f}] pp {a['probability_positive']:.2f} | T real {b['real']:7.2f} sim {b['sim']:7.2f} d {b['diff']:6.2f} [{b['ci'][0]:6.2f},{b['ci'][1]:6.2f}] pp {b['probability_positive']:.2f}")
    print(json.dumps({k: res[k] for k in ("ratio", "var", "epa_share_var", "class_margin_var", "class_share_drives", "drives_per_game")}, indent=1))


if __name__ == "__main__":
    main()

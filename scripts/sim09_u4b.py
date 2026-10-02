import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_u3a as u3  # noqa: E402
import sim09_u3d as u3d  # noqa: E402

CRZK = REPO / "artifacts" / "sim09" / "u3d" / "play_crzk"


def phase(q, gsr):
    q = np.asarray(q)
    g = np.asarray(gsr)
    hu = ((q == 2) & (g <= 1920) & (g > 1800)) | ((q == 4) & (g <= 120))
    lateq4 = (q == 4) & (g <= 480) & ~hu
    return np.where(hu, "endhalf_game_2min", np.where(lateq4, "q4_last8", "other"))


def nhtab(nm, P):
    n = P["src_no_huddle"].fillna(0) == 1
    for pl, c in (("run", 0), ("pass", 1)):
        m = P["code"] == c
        print(nm, pl, "NH share", round(float(n[m].mean()), 3), "ypp NH", round(float(P.loc[m & n, "yards"].mean()), 3), "non", round(float(P.loc[m & ~n, "yards"].mean()), 3))
    for dn in (1, 2, 3, 4):
        m = P["down"] == dn
        print(nm, "down", dn, "NH share", round(float(n[m].mean()), 3), "ypp NH", round(float(P.loc[m & n, "yards"].mean()), 3), "non", round(float(P.loc[m & ~n, "yards"].mean()), 3))
    bins = [(-99, -9), (-8, -1), (0, 0), (1, 8), (9, 99)]
    for lo, hi in bins:
        m = P["sd"].between(lo, hi)
        print(nm, "margin", lo, hi, "NH share", round(float(n[m].mean()), 3), "ypp NH", round(float(P.loc[m & n, "yards"].mean()), 3), "non", round(float(P.loc[m & ~n, "yards"].mean()), 3))


def main():
    R = u3d.real_frame()
    u3.bud.OUT = CRZK
    S = u3.load_sim_frames(2)
    idx = S["idx"].to_numpy(np.int64)
    ok = S["code"].isin([0, 1]).to_numpy()
    ry = R["yards"].to_numpy()[idx]
    print("pool align yards match", float(np.mean(np.isclose(ry[ok], S["yards"].to_numpy()[ok], equal_nan=True))))
    for c in ["down", "dist", "yl", "sd", "gsr", "qtr", "code"]:
        S["r_" + c] = R[c].to_numpy()[idx]
    print("N real plays", len(R), "N sim plays", len(S))

    print("\n== (1) no-huddle ==")
    for nm, P in (("real", R), ("sim", S)):
        k = P["code"].isin([0, 1]) & P["yards"].notna()
        nh = P["src_no_huddle"].fillna(0) == 1
        print(nm, "NH share", round(float(nh[k].mean()), 4), "ypp NH", round(float(P.loc[k & nh, "yards"].mean()), 3), "ypp non", round(float(P.loc[k & ~nh, "yards"].mean()), 3), "diff", round(float(P.loc[k & nh, "yards"].mean() - P.loc[k & ~nh, "yards"].mean()), 3))
        ph = phase(P["qtr"], P["gsr"])
        for p in ["endhalf_game_2min", "q4_last8", "other"]:
            m = k & (ph == p)
            print("  state-phase", p, "n", int(m.sum()), "NH share", round(float(nh[m].mean()), 3), "ypp NH", round(float(P.loc[m & nh, "yards"].mean()), 3), "ypp non", round(float(P.loc[m & ~nh, "yards"].mean()), 3))
    k = S["code"].isin([0, 1])
    nh = S["src_no_huddle"].fillna(0) == 1
    sp = phase(S["qtr"], S["gsr"])
    rp = phase(S["r_qtr"], S["r_gsr"])
    print("sim NH draws: rows=sim state phase, cols=source row phase")
    print(pd.crosstab(sp[k & nh], rp[k & nh]))
    print("sim non-NH draws")
    print(pd.crosstab(sp[k & ~nh], rp[k & ~nh]))
    for nm, m in (("NH drawn from source late/hurry", rp != "other"), ("NH drawn from source other", rp == "other")):
        mm = k & nh & m
        print(nm, "n", int(mm.sum()), "ypp", round(float(S.loc[mm, "yards"].mean()), 3))
    nhtab("real", R[R["code"].isin([0, 1])])
    nhtab("sim", S[k])
    Sk = S[k]
    nhk = nh[k]
    print("sim NH: source margin vs sim margin mean abs", round(float((Sk.loc[nhk, "r_sd"] - Sk.loc[nhk, "sd"]).abs().mean()), 2), "source down==sim down", round(float((Sk.loc[nhk, "r_down"] == Sk.loc[nhk, "down"]).mean()), 3))
    print("sim NH: source dist mean", round(float(Sk.loc[nhk, "r_dist"].mean()), 2), "sim dist mean", round(float(Sk.loc[nhk, "dist"].mean()), 2))
    Rk = R[R["code"].isin([0, 1])]
    rn = Rk["src_no_huddle"].fillna(0) == 1
    print("real NH dist mean", round(float(Rk.loc[rn, "dist"].mean()), 2), "non-NH", round(float(Rk.loc[~rn, "dist"].mean()), 2), "sim non-NH dist", round(float(Sk.loc[~nhk, "dist"].mean()), 2))
    print("real NH pass share", round(float((Rk.loc[rn, "code"] == 1).mean()), 3), "sim", round(float((Sk.loc[nhk, "code"] == 1).mean()), 3))

    print("\n== (2) 4th/3rd down ==")
    db = [0, 1, 2, 3, 4, 6, 10, 100]
    keep = {}
    for nm, P in (("real", R), ("sim", S)):
        P = P.sort_values("g", kind="stable").reset_index(drop=True)
        nd = P.groupby("g")["down"].shift(-1)
        P["conv"] = ((P["yards"] >= P["dist"]) | ((nd == 1) & ~P["flip"])).astype(float)
        P["db"] = pd.cut(P["dist"], db)
        keep[nm] = P
        for dn in (4, 3):
            d = P[P["down"] == dn]
            go = d["code"].isin([0, 1])
            att = d[go]
            print(nm, "down", dn, "n", len(d), "go share", round(float(go.mean()), 4), "conv", round(float(att["conv"].mean()), 4))
            rows = []
            for b, x in d.groupby("db", observed=True):
                g = x["code"].isin([0, 1])
                rows.append((str(b), len(x), round(len(x) / len(d), 4), round(float(g.mean()), 4), round(float(x.loc[g, "conv"].mean()), 4), int(g.sum()), round(float((x.loc[g, "code"] == 1).mean()), 3)))
            print(pd.DataFrame(rows, columns=["dist", "n", "mix", "go", "conv", "gon", "pass"]).to_string(index=False))
            if dn == 4:
                print(nm, "4th conv pass", round(float(att[att["code"] == 1]["conv"].mean()), 4), "run", round(float(att[att["code"] == 0]["conv"].mean()), 4))
    RR, SS = keep["real"], keep["sim"]
    r4 = RR[(RR["down"] == 4) & RR["code"].isin([0, 1])]
    s4 = SS[(SS["down"] == 4) & SS["code"].isin([0, 1])]
    mixr = r4.groupby("db", observed=True).size() / len(r4)
    mixs = s4.groupby("db", observed=True).size() / len(s4)
    cr = r4.groupby("db", observed=True)["conv"].mean()
    cs = s4.groupby("db", observed=True)["conv"].mean()
    print("4th go decomposition sim-real conv", round(float(s4["conv"].mean() - r4["conv"].mean()), 4), "mix effect", round(float(((mixs - mixr) * cr).sum()), 4), "within-bucket", round(float((mixs * (cs - cr)).sum()), 4))
    s4b = S[(S["down"] == 4) & S["code"].isin([0, 1])]
    print("source down of sim 4th go", s4b["r_down"].value_counts(normalize=True).round(3).to_dict())
    print("mean dist sim", round(float(s4b["dist"].mean()), 3), "source dist", round(float(s4b["r_dist"].mean()), 3), "real go dist", round(float(r4["dist"].mean()), 3))
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 100)):
        m = s4b["dist"].between(lo, hi)
        print("sim 4th dist", lo, hi, "n", int(m.sum()), "source mean dist", round(float(s4b.loc[m, "r_dist"].mean()), 3), "mean yards", round(float(s4b.loc[m, "yards"].mean()), 3), "real mean yards", round(float(r4.loc[r4["dist"].between(lo, hi), "yards"].mean()), 3), "frac source dist>=sim", round(float((s4b.loc[m, "r_dist"] >= s4b.loc[m, "dist"]).mean()), 3))
    s4c = s4b.copy()
    s4c["yd_ge_src"] = s4c["yards"] >= s4c["r_dist"]
    print("sim 4th conv if judged vs source dist", round(float(s4c["yd_ge_src"].mean()), 4), "judged vs sim dist (yards only)", round(float((s4c["yards"] >= s4c["dist"]).mean()), 4), "engine conv", round(float(SS[(SS["down"] == 4) & SS["code"].isin([0, 1])]["conv"].mean()), 4))

    print("\n== (3) Q2 final-2-minute window ==")
    for nm, P in (("real", R), ("sim", S)):
        D, Rr = dv.drive_table(P)
        ng = P["g"].nunique()
        Rw = Rr[(Rr["qtr"] == 2) & (Rr["gsr"] <= 1920) & (Rr["gsr"] > 1800)]
        dsn = Rw.groupby("poss").size()
        print(nm, "games", ng, "drives with snap in window/game", round(len(dsn) / ng, 4), "snaps/game", round(len(Rw) / ng, 3), "sec per snap", round(float(Rw["el"].mean()), 2))
        Dq = D[(D["q0"] == 2) & (D["gsr0"] <= 1920) & (D["gsr0"] > 1800)]
        print("  drives starting in window/game", round(len(Dq) / ng, 4), "plays/drive", round(float(Dq["n"].mean()), 3), "secs/drive", round(float(Dq["secs"].mean()), 1), "oc mix", Dq["oc"].value_counts(normalize=True).round(3).to_dict())
        gl = Rw.groupby("g")["poss"].nunique()
        cnt = gl.reindex(np.unique(P["g"].to_numpy())).fillna(0)
        print("  window drive-count per game dist", cnt.value_counts(normalize=True).sort_index().round(3).to_dict())
        D2 = D[D["q0"] == 2].copy()
        D2["prev_oc"] = D2.groupby("g")["oc"].shift(1)
        w = D2[(D2["gsr0"] <= 1920) & (D2["gsr0"] > 1800)]
        print("  prev drive outcome for drives starting in window", w["prev_oc"].value_counts(normalize=True).round(3).to_dict())
        last = D2.groupby("g").tail(1)
        print("  last Q2 drive start gsr mean", round(float(last["gsr0"].mean()), 1), "sd", round(float(last["gsr0"].std()), 1), "share start<=1920", round(float((last["gsr0"] <= 1920).mean()), 3), "n drives q2 per game", round(float(len(D2) / ng), 3))
        Rq = Rr[Rr["qtr"] == 2]
        e = Rq[(Rq["gsr"] > 1920) & (Rq["gsr"] <= 2100)]
        print("  sec/snap 1920-2100", round(float(e["el"].mean()), 2), "q2 mean sec/snap", round(float(Rq["el"].mean()), 2), "q2 snaps/game", round(len(Rq) / ng, 3))
        lq = Rq.groupby("g").tail(1)
        print("  last q2 play code mix", lq["code"].value_counts(normalize=True).round(3).to_dict(), "gsr mean", round(float(lq["gsr"].mean()), 1))
        ps = Rq.groupby("g")["el"].sum()
        print("  q2 elapsed per game", round(float(ps.mean()), 1))
        print("  window sec/snap by code", Rw.groupby("code")["el"].agg(["size", "mean"]).round(2).to_dict())


def main2():
    R = u3d.real_frame()
    u3.bud.OUT = CRZK
    S = u3.load_sim_frames(2)
    db = [0, 1, 2, 3, 4, 6, 10, 100]
    ylb = [0, 10, 20, 30, 40, 50, 100]
    for nm, P in (("real", R), ("sim", S)):
        P = P.sort_values("g", kind="stable").reset_index(drop=True)
        nd = P.groupby("g")["down"].shift(-1)
        P["yconv"] = (P["yards"] >= P["dist"]).astype(float)
        P["pconv"] = ((nd == 1) & ~P["flip"] & (P["yards"] < P["dist"])).astype(float)
        P["db"] = pd.cut(P["dist"], db)
        a = P[(P["down"] == 4) & P["code"].isin([0, 1])]
        print(nm, "4th go: yards-only conv", round(float(a["yconv"].mean()), 4), "extra nextdown1 no flip", round(float(a["pconv"].mean()), 4))
        print(a.groupby("db", observed=True)[["yconv", "pconv"]].mean().round(4).T.to_string())
        print(nm, "4th go yards<=0 share", round(float((a["yards"] <= 0).mean()), 4), "mean yards", round(float(a["yards"].mean()), 3))
        P["dt"] = P["gsr"] - P.groupby("g")["gsr"].shift(-1)
        w = P[(P["qtr"] == 2) & (P["gsr"] <= 1980) & (P["gsr"] > 1800) & P["dt"].between(0, 200)]
        print(nm, "window gsr gap to next snap by code and flip")
        print(w.assign(kind=np.where(w["flip"], "flip", "same")).groupby(["code", "kind"])["dt"].agg(["size", "mean"]).round(1).to_string())
        q = P[(P["qtr"] == 2) & P["dt"].between(0, 200)]
        print(nm, "q2 gsr-gap per game", round(float(q.groupby("g")["dt"].sum().mean()), 1), "gap/snap", round(float(q["dt"].mean()), 2), "el/snap", round(float(P.loc[P["qtr"] == 2, "el"].mean()), 2))
        ng = P["g"].nunique()
        for lab, m in (("q2 window", (P["qtr"] == 2) & (P["gsr"] <= 1920) & (P["gsr"] > 1800)), ("q4 last2min", (P["qtr"] == 4) & (P["gsr"] <= 120)), ("rest", ~(((P["qtr"] == 2) & (P["gsr"] <= 1920) & (P["gsr"] > 1800)) | ((P["qtr"] == 4) & (P["gsr"] <= 120))))):
            d = P[m & (P["down"] == 4)]
            d = d.assign(yb=pd.cut(d["yl"], ylb))
            t = d.groupby("yb", observed=True)["code"].agg(n="size", fg=lambda x: (x == 3).mean(), punt=lambda x: (x == 2).mean(), go=lambda x: x.isin([0, 1]).mean())
            print(nm, lab, "4th-down plays per game", round(len(d) / ng, 3), "FG attempts per game", round(float((d["code"] == 3).sum() / ng), 4))
            print(t.round(3).to_string())
        w1 = P[(P["qtr"] == 2) & (P["gsr"] <= 1920) & (P["gsr"] > 1800)]
        print(nm, "window plays by down", w1["down"].value_counts(normalize=True).sort_index().round(3).to_dict(), "mean yl", round(float(w1["yl"].mean()), 1), "mean sd", round(float(w1["sd"].mean()), 2))


def main3():
    R = u3d.real_frame()
    u3.bud.OUT = CRZK
    S = u3.load_sim_frames(2)
    for nm, P in (("real", R), ("sim", S)):
        P = P.sort_values("g", kind="stable").reset_index(drop=True)
        P = P[P["code"].isin([0, 1])].reset_index(drop=True)
        nh = (P["src_no_huddle"].fillna(0) == 1).astype(float)
        P["nh"] = nh
        P["prior"] = P.groupby("g")["nh"].transform(lambda x: x.shift(1).rolling(10, min_periods=10).mean())
        P["prev"] = P.groupby("g")["nh"].shift(1)
        d = P[P["prior"].notna()]
        print(nm, "P(NH|prev NH)", round(float(d.loc[d["prev"] == 1, "nh"].mean()), 4), "P(NH|prev not)", round(float(d.loc[d["prev"] == 0, "nh"].mean()), 4), "base", round(float(d["nh"].mean()), 4), "corr(prior10 share, own NH)", round(float(np.corrcoef(d["prior"], d["nh"])[0, 1]), 4), "sd of prior10 share", round(float(d["prior"].std()), 4))
        y = d["yards"].to_numpy(float)
        ok = np.isfinite(y)
        for lab, cols in (("prior only", ["prior"]), ("prior + own NH", ["prior", "nh"]), ("prior + own NH + margin bins + qtr", ["prior", "nh", "m1", "m2", "m3", "m4", "q2", "q3", "q4"])):
            d2 = d.assign(m1=(d["sd"] <= -9).astype(float), m2=d["sd"].between(-8, -1).astype(float), m3=d["sd"].between(1, 8).astype(float), m4=(d["sd"] >= 9).astype(float), q2=(d["qtr"] == 2).astype(float), q3=(d["qtr"] == 3).astype(float), q4=(d["qtr"] == 4).astype(float))
            X = np.column_stack([np.ones(len(d2))] + [d2[c].to_numpy(float) for c in cols])[ok]
            b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            print("  ", lab, "coef prior", round(float(b[1]), 3), "own NH", round(float(b[2]), 3) if "nh" in cols else None)
        dd = d.assign(hi=(d["prior"] >= 0.3))
        print("  ypp by prior10 NH share>=.3", round(float(dd.loc[dd["hi"], "yards"].mean()), 3), "n", int(dd["hi"].sum()), "else", round(float(dd.loc[~dd["hi"], "yards"].mean()), 3), "NH share in hi", round(float(dd.loc[dd["hi"], "nh"].mean()), 3))
        dn = dd[dd["nh"] == 0]
        print("  non-NH plays: ypp prior>=.3", round(float(dn.loc[dn["hi"], "yards"].mean()), 3), "else", round(float(dn.loc[~dn["hi"], "yards"].mean()), 3))


if len(sys.argv) > 1 and sys.argv[1] == "2":
    main2()
elif len(sys.argv) > 1 and sys.argv[1] == "3":
    main3()
else:
    main()

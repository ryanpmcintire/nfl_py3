import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_resid as mr
import mod25_generator as gen

E3 = REPO / "artifacts" / "mod25e3"
OUT = E3 / "channels"
NB = 300
NET = ["scrim_net", "scrim_off", "scrim_def", "fp_net", "punt_net", "fg_luck", "fg_exp", "ret_pts", "tov_net", "go4", "pen_yd", "st_epa", "st_ko", "st_punt", "st_fg"]
SIM_ONLY_MISSING = ("pen_yd", "st_epa", "st_ko", "st_punt", "st_fg")


def plays_real(seasons):
    p = mr.load_real(seasons)
    return p[(p["qtr"] <= 4) & p["posteam"].notna() & p["play_type"].isin(["run", "pass", "punt", "field_goal", "kickoff", "extra_point", "no_play"])].copy()


def prep_real(p):
    q = p[p["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    q["off"] = q["posteam"]
    q["kind"] = q["play_type"].map({"run": "scrim", "pass": "scrim", "punt": "punt", "field_goal": "fg"})
    q["epa"] = q["epa"].astype(float)
    q["yl"] = q["yardline_100"].astype(float)
    q["tov"] = (q["interception"].fillna(0) + q["fumble_lost"].fillna(0)).astype(float)
    return q[["game_id", "off", "kind", "epa", "yl", "down", "tov"]].reset_index(drop=True)


def prep_sim(P, g):
    Q = P[P["code"].isin([0, 1, 2, 3])].copy()
    gm = g.set_index("game_id")
    ht = gm["home_team"].reindex(Q["game_id"]).to_numpy()
    at = gm["away_team"].reindex(Q["game_id"]).to_numpy()
    Q["off"] = np.where(Q["offhome"].to_numpy() == 1, ht, at)
    Q["kind"] = np.select([Q["code"] < 2, Q["code"] == 2], ["scrim", "punt"], "fg")
    Q["epa"] = Q["epa"].astype(float).where(Q["code"] < 2)
    Q["yl"] = Q["yl"].astype(float)
    Q["tov"] = (Q["interception"].fillna(0) + Q["fumble_lost"].fillna(0)).astype(float).where(Q["code"] < 2, 0.0)
    return Q[["game_id", "off", "kind", "epa", "yl", "down", "tov"]].reset_index(drop=True)


def team_game(Q, games, fgdf, evdf, pfg):
    Q = Q.copy()
    Q["start"] = (Q["game_id"] != Q["game_id"].shift()) | (Q["off"] != Q["off"].shift())
    nd = (Q["game_id"] == Q["game_id"].shift(-1)) & (Q["off"] != Q["off"].shift(-1))
    pn = np.where((Q["kind"] == "punt") & nd, Q["yl"] - (100.0 - Q["yl"].shift(-1)), np.nan)
    Q["pnet"] = pn - np.nanmean(pn)
    g4 = np.where(Q["down"] == 4, (Q["kind"] == "scrim").astype(float), np.nan)
    Q["g4"] = g4 - np.nanmean(g4)
    sc = Q[Q["kind"] == "scrim"]
    t = pd.DataFrame({"off_epa": sc.groupby(["game_id", "off"])["epa"].sum(), "tov": sc.groupby(["game_id", "off"])["tov"].sum()})
    t["fp"] = -Q[Q["start"]].groupby(["game_id", "off"])["yl"].mean()
    t["punt"] = Q.groupby(["game_id", "off"])["pnet"].sum()
    t["g4"] = Q.groupby(["game_id", "off"])["g4"].sum()
    t = t.fillna(0.0).reset_index()
    ph = pfg(fgdf["dist"])
    fgx = pd.DataFrame({"game_id": fgdf["game_id"], "luck": 3.0 * (fgdf["made"] - ph) * fgdf["sgn"], "exp": 3.0 * ph * fgdf["sgn"]})
    fgh = fgx.groupby("game_id")[["luck", "exp"]].sum()
    ev = evdf[evdf["comp"].str.startswith("ret_")]
    reth = ev.groupby("game_id")["pts"].sum()
    out = []
    for side in ("home", "away"):
        a = games[["game_id", "season", "week"]].copy()
        a["team"] = games[side + "_team"].to_numpy()
        a["opp"] = games["away_team" if side == "home" else "home_team"].to_numpy()
        a["sg"] = 1.0 if side == "home" else -1.0
        a["M"] = a["sg"] * (games["home_score"] - games["away_score"]).to_numpy()
        out.append(a)
    T = pd.concat(out, ignore_index=True)
    own = t.rename(columns={"off": "team"})
    opp = t.rename(columns={"off": "opp"})
    T = T.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left", suffixes=("", "_o")).fillna(0.0)
    T["scrim_off"] = T["off_epa"]
    T["scrim_def"] = -T["off_epa_o"]
    T["scrim_net"] = T["scrim_off"] + T["scrim_def"]
    T["fp_net"] = T["fp"] - T["fp_o"]
    T["punt_net"] = T["punt"] - T["punt_o"]
    T["tov_net"] = T["tov_o"] - T["tov"]
    T["go4"] = T["g4"] - T["g4_o"]
    T["fg_luck"] = T["sg"] * T["game_id"].map(fgh["luck"]).fillna(0.0)
    T["fg_exp"] = T["sg"] * T["game_id"].map(fgh["exp"]).fillna(0.0)
    T["ret_pts"] = T["sg"] * T["game_id"].map(reth).fillna(0.0)
    return T


def real_extra(T, p):
    z = p.copy()
    pen = z[z["penalty_team"].notna()].groupby(["game_id", "penalty_team"])["penalty_yards"].sum().rename("pen").reset_index().rename(columns={"penalty_team": "team"})
    st = z[z["play_type"].isin(["punt", "kickoff", "field_goal", "extra_point"])]
    stg = st.groupby(["game_id", "posteam", "play_type"])["epa"].sum().unstack(fill_value=0.0).reset_index().rename(columns={"posteam": "team"})
    for c in ("punt", "kickoff", "field_goal", "extra_point"):
        if c not in stg.columns:
            stg[c] = 0.0
    stg["st"] = stg[["punt", "kickoff", "field_goal", "extra_point"]].sum(axis=1)
    stg = stg[["game_id", "team", "st", "kickoff", "punt", "field_goal"]].rename(columns={"punt": "stp", "kickoff": "stk", "field_goal": "stf"})
    T = T.merge(pen, on=["game_id", "team"], how="left").merge(stg, on=["game_id", "team"], how="left")
    for c in ("pen", "st", "stp", "stk", "stf"):
        T[c] = T[c].fillna(0.0)
    po = T[["game_id", "team", "pen", "st", "stp", "stk", "stf"]].rename(columns={"team": "opp", "pen": "pen_o", "st": "st_o", "stp": "stp_o", "stk": "stk_o", "stf": "stf_o"})
    T = T.merge(po, on=["game_id", "opp"], how="left")
    T["pen_yd"] = T["pen_o"] - T["pen"]
    T["st_epa"] = T["st"] - T["st_o"]
    T["st_punt"] = T["stp"] - T["stp_o"]
    T["st_ko"] = T["stk"] - T["stk_o"]
    T["st_fg"] = T["stf"] - T["stf_o"]
    return T


def halves(T, chans):
    T = T.sort_values(["season", "team", "week"]).copy()
    T["pos"] = T.groupby(["season", "team"]).cumcount()
    cols = ["M"] + chans
    for c in cols:
        T[c] = T[c] - T.groupby("season")[c].transform("mean")
    od = T[T["pos"] % 2 == 0].groupby(["season", "team"])[cols].mean()
    ev = T[T["pos"] % 2 == 1].groupby(["season", "team"])[cols].mean()
    j = od.index.intersection(ev.index)
    return od.loc[j], ev.loc[j]


def stats(od, ev, ix, chans):
    a = od.to_numpy()[ix]
    b = ev.to_numpy()[ix]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    C = (a.T @ b) / len(a)
    C = (C + C.T) / 2.0
    k = {n: i for i, n in enumerate(od.columns)}
    r = {}
    VM = C[0, 0]
    s = k["scrim_net"]
    vs = C[s, s]
    expl_s = C[0, s] ** 2 / vs
    r["V_M"] = VM
    r["expl_scrim"] = expl_s
    r["persist_resid"] = VM - expl_s
    r["persist_resid_share"] = (VM - expl_s) / VM
    o, d = k["scrim_off"], k["scrim_def"]
    r["rho_off_def"] = C[o, d] / np.sqrt(C[o, o] * C[d, d])
    r["ratio_M_scrim"] = VM / vs
    for c in chans:
        i = k[c]
        rr = np.corrcoef(a[:, i], b[:, i])[0, 1]
        r[c + "|tvar"] = C[i, i]
        r[c + "|rel"] = 2 * rr / (1 + rr)
        r[c + "|corrM"] = C[0, i] / np.sqrt(max(VM * C[i, i], 1e-12))
        if c == "scrim_net":
            continue
        cs = C[i, s]
        vcs = C[i, i] - cs ** 2 / vs
        ccs = C[0, i] - C[0, s] * cs / vs
        r[c + "|incr"] = ccs ** 2 / vcs if vcs > 1e-12 else 0.0
    ix_s = [s] + [k[c] for c in chans if c not in ("scrim_net", "scrim_off", "scrim_def")]
    Cs = C[np.ix_(ix_s, ix_s)] + 1e-9 * np.eye(len(ix_s))
    beta = np.linalg.solve(Cs, C[0, ix_s])
    r["expl_all"] = float(C[0, ix_s] @ beta)
    r["persist_resid_all_share"] = float((VM - r["expl_all"]) / VM)
    return r


def run(T, chans, label, rng):
    od, ev = halves(T, chans)
    n = len(od)
    seasons = np.asarray(od.index.get_level_values("season"))
    ukeys = pd.unique(seasons)
    pos = {u: np.where(seasons == u)[0] for u in ukeys}
    base = stats(od, ev, np.arange(n), chans)
    boots = []
    for _ in range(NB):
        pick = rng.choice(len(ukeys), len(ukeys))
        boots.append(stats(od, ev, np.concatenate([pos[ukeys[i]] for i in pick]), chans))
    B = pd.DataFrame(boots)
    return {"label": label, "n_team_seasons": n, "base": base, "boot_lo": B.quantile(.025).to_dict(), "boot_hi": B.quantile(.975).to_dict(), "boot": B.to_dict("list")}


def load_real_games(seasons):
    rg = gen.real_games(tuple(seasons))
    return rg[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    Psim = mr.load_sim()
    g = pd.read_parquet(E3 / "e5_crzhk_s11" / "sim_games.parquet")
    g = g[((g["season"] % 1000) >= 3) & g["game_id"].isin(Psim["game_id"])].reset_index(drop=True)
    Pp = Psim[Psim["game_id"].isin(g["game_id"])]
    Qs = prep_sim(Pp, g)
    evs, fgs = mr.sim_events(Pp)
    real = {}
    for lab, ss in (("real_2009_17", range(2009, 2018)), ("real_2018_25", range(2018, 2026))):
        p = plays_real(ss)
        Qr = prep_real(p)
        evr, fgr = mr.real_events(p)
        real[lab] = (Qr, p, evr, fgr, ss)
    pfg = mr.fit_fg(real["real_2009_17"][3])
    chans_sim = [c for c in NET if c not in SIM_ONLY_MISSING]
    Ts = team_game(Qs, g[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]], fgs, evs, pfg)
    res["sim"] = run(Ts, chans_sim, "sim_crzhk_s11", rng)
    for lab, (Qr, p, evr, fgr, ss) in real.items():
        gr = load_real_games(ss)
        gr = gr[gr["game_id"].isin(Qr["game_id"].unique())].reset_index(drop=True)
        Tr = real_extra(team_game(Qr, gr, fgr, evr, pfg), p)
        res[lab] = run(Tr, NET, lab, rng)
    (OUT / "channels.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "boot"} for k, v in res.items()}, default=float))
    lines = []
    for k in ("real_2009_17", "real_2018_25", "sim"):
        r = res[k]
        b, lo, hi = r["base"], r["boot_lo"], r["boot_hi"]
        lines.append(f"{k} rho_off_def {b['rho_off_def']:.3f} [{lo['rho_off_def']:.3f},{hi['rho_off_def']:.3f}] M/scrim {b['ratio_M_scrim']:.3f}")
        lines.append(f"{k} n={r['n_team_seasons']} V_M {b['V_M']:.1f} [{lo['V_M']:.1f},{hi['V_M']:.1f}] scrim_expl {b['expl_scrim']:.1f} persist_resid {b['persist_resid']:.1f} [{lo['persist_resid']:.1f},{hi['persist_resid']:.1f}] share {b['persist_resid_share']:.3f} after_all_channels {b['persist_resid_all_share']:.3f}")
        for c in NET:
            if c + "|tvar" in b:
                ic = c + "|incr"
                lines.append(f"  {c:10s} tvar {b[c+'|tvar']:.3f} rel {b[c+'|rel']:.2f} corrM {b[c+'|corrM']:+.2f} incr {b.get(ic, float('nan')):.2f} [{lo.get(ic, float('nan')):.2f},{hi.get(ic, float('nan')):.2f}]")
    for ref in ("real_2009_17",):
        lines.append("sim minus " + ref + " (probability_positive = P(sim > real) over season bootstrap)")
        keys = ["V_M", "rho_off_def", "ratio_M_scrim", "expl_scrim", "persist_resid", "persist_resid_share", "persist_resid_all_share"] + [c + s for c in chans_sim for s in ("|tvar", "|incr", "|rel")]
        for key in keys:
            if key in res["sim"]["boot"] and key in res[ref]["boot"]:
                d = np.array(res["sim"]["boot"][key]) - np.array(res[ref]["boot"][key])
                lines.append(f"  {key:22s} sim {res['sim']['base'][key]:.3f} real {res[ref]['base'][key]:.3f} diff {res['sim']['base'][key]-res[ref]['base'][key]:+.3f} pp {float((d > 0).mean()):.2f}")
    (OUT / "channels.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

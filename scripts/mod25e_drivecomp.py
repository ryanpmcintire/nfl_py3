import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1]
LABEL = os.environ.get("DC_LABEL", "crHpqokgndecsk2mfwt")
SEEDS = tuple(int(x) for x in os.environ.get("DC_SEEDS", "11,12,13").split(","))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "drivecomp"
LIVE = ("run", "pass", "punt", "field_goal")
KN = ["N", "F", "L", "EFF", "CONV"]
KDESC = {"N": "drive count x mean pts", "F": "start field position", "L": "score-band state", "EFF": "play EPA residual", "CONV": "points minus EPA residual"}
LATER = ["F", "EFF", "CONV"]
MASK = np.triu(np.ones((4, 4)), 1)
BANDS = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
CH = 10
rng = np.random.default_rng(76)
OUT = []
LOOKS = [0]


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def iv(d, pt):
    LOOKS[0] += 1
    d = np.asarray(d)
    return f"{pt:+.3f} [{np.percentile(d, 5):+.3f},{np.percentile(d, 95):+.3f}] pp {float((d > 0).mean()):.2f}"


def epa_table():
    import mod25d_variance as dv
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp, team_ratings=sim.load_team_ratings()).reset_index(drop=True)
    return trans[["game_id", "play_id"]].merge(pbp[["game_id", "play_id", "epa"]].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left").epa.to_numpy(float)


def outcome_code(td, fg, pu):
    return np.where(td, "TD", np.where(fg, "FG", np.where(pu, "P", "O")))


def real_build(ns):
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "epa", "fixed_drive_result"]
    out = []
    for s in ns["POOL"]:
        p = pd.read_parquet(ns["PBP"] / f"season={s}" / "plays.parquet", columns=cols)
        p["season"] = s
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4)]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        p["m"] = np.where(p.posteam == p.home_team, 1.0, -1.0) * p.score_differential
        fin = (gf.home_score - gf.away_score).reindex(p.game_id).to_numpy()
        g = p.game_id.to_numpy()
        last = np.r_[g[1:] != g[:-1], True]
        nxt = np.r_[p.m.to_numpy()[1:], 0.0]
        p["dm"] = np.where(last, fin - p.m.to_numpy(), nxt - p.m.to_numpy())
        p = p[p.dm.notna()]
        ko = (p.play_type == "kickoff").to_numpy()
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        live = p.play_type.isin(LIVE).to_numpy()
        p["ep"] = np.where(live, p.epa.fillna(0.0).to_numpy(), 0.0)
        p["lv"] = live.astype(int)
        g = p.game_id.to_numpy()
        pos = p.posteam.to_numpy()
        kc = p.kocum.to_numpy()
        q = p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        gp = p.groupby("d", sort=True)
        first_pos = gp.posteam.first()
        res = gp.fixed_drive_result.last()
        D = pd.DataFrame({"gk": gp.season.first().astype(str) + "_" + gp.game_id.first(), "season": gp.season.first(), "off": first_pos, "dfn": gp.defteam.first(),
                          "s": np.where(first_pos == gp.home_team.first(), 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yardline_100.first(), "n": gp.size(), "dmsum": gp.dm.sum(),
                          "sd0": gp.score_differential.first(), "e": gp.ep.sum(), "nlive": gp.lv.sum(), "gsr0": gp.game_seconds_remaining.first()})
        D["out"] = outcome_code((res == "Touchdown").to_numpy(), (res == "Field goal").to_numpy(), (res == "Punt").to_numpy())
        D["p"] = D.s * D.dmsum
        D["tm"] = D.season.astype(str) + "_" + D.off
        D["td"] = D.season.astype(str) + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return pd.concat(out, ignore_index=True)


def sim_build(epa_t, late):
    out = []
    cov = {"live": 0, "ok": 0}
    for sdn in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sdn}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            tm = sg[(sg.w == w) & (sg.s == s)].set_index("g")
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "sd", "po", "pdf", "code", "gsr", "idx"])
            d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
            live = d.code.isin([0, 1, 2, 3]).to_numpy()
            ix = d.idx.to_numpy()
            ok = live & np.isfinite(ix)
            cov["live"] += int(live.sum())
            cov["ok"] += int(ok.sum())
            ep = np.zeros(len(d))
            ep[ok] = np.nan_to_num(epa_t[ix[ok].astype(int)])
            d["ep"] = ep
            d["lv"] = live.astype(int)
            g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
            d["d"] = np.cumsum(new)
            d["pp"] = d.po - d.pdf
            gi = d.g.astype(int)
            d["ht"] = tm.home_team.reindex(gi).to_numpy()
            d["at"] = tm.away_team.reindex(gi).to_numpy()
            d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
            d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
            gp = d.groupby("d", sort=True)
            key = f"{sdn}_{w}_{s}"
            own = gp.po.sum()
            D = pd.DataFrame({"gk": ((sdn * 10000 + w) * 100 + s) * 100000 + gp.g.first().astype(np.int64), "season": 0, "off": gp.off.first(), "dfn": gp.dfn.first(),
                              "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "n": gp.size(), "sd0": gp.sd.first(),
                              "p": gp.pp.sum(), "e": gp.ep.sum(), "nlive": gp.lv.sum(), "gsr0": gp.gsr.first()})
            D["out"] = outcome_code((own >= 6).to_numpy(), (own == 3).to_numpy(), (gp.code.last() == 2).to_numpy())
            D["tm"] = key + "_" + D.off
            D["td"] = key + "_" + D.dfn
            out.append(D)
    return pd.concat(out, ignore_index=True), cov


def prep(D, ns):
    D = D.reset_index(drop=True)
    comp = {}
    for nm, x in (("p", D.p), ("e", D.e), ("cv", D.p - D.e)):
        comp[nm] = ns["components"](D.assign(p=x.to_numpy()))[0]
    K = {"N": comp["p"].c_count.to_numpy(), "F": comp["p"].c_startfp.to_numpy(), "L": comp["p"].c_leadstate.to_numpy(), "EFF": comp["e"].c_rest.to_numpy(), "CONV": comp["cv"].c_rest.to_numpy()}
    a = (comp["p"].p - comp["p"].c_strength).to_numpy()
    chk = float(np.abs(a - sum(K.values())).max())
    gi, gn = pd.factorize(D.gk)
    G = len(gn)
    qi = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
    bi = pd.cut(D.sd0, ns["LEAD_EDGES"], labels=False).astype(int).to_numpy()
    idx = gi * 20 + qi * 5 + bi
    s = D.s.to_numpy()

    def gs(v):
        return np.bincount(idx, weights=np.asarray(v, float), minlength=G * 20).reshape(G, 20)

    out = D.out.to_numpy()
    A = {"G": G, "cnt": gs(np.ones(len(D))), "msg": gs(s), "npl": gs(s * D.nlive.to_numpy()), "Sx": {k: gs(K[k]) for k in KN}, "Ms": {k: gs(s * K[k]) for k in KN},
         "Sv": {"conv": gs((D.p - D.e).to_numpy()), "eff": gs(D.e.to_numpy()), "plays": gs(D.nlive.to_numpy()), "TD": gs(out == "TD"), "FG": gs(out == "FG"), "punt": gs(out == "P")},
         "chk": chk, "gi": gi, "bi": bi, "mean_p": float(D.p.mean()), "mean_e": float(D.e.mean())}
    return A


def stats(A, W):
    G, nb = A["G"], len(W)
    tc = W @ A["cnt"]
    ab = {k: ((W @ A["Sx"][k]) / np.maximum(tc, 1e-9)).reshape(nb, 4, 5) for k in KN}
    I = {k: A["Ms"][k].reshape(G, 4, 5).sum(2) for k in KN}
    msg = A["msg"].reshape(G, 4, 5)
    IE = {k: I[k][None] - np.einsum("grd,brd->bgr", msg, ab[k]) for k in KN}
    sw = W.sum(1)

    def cov(X, Y):
        Xb = np.broadcast_to(X, (nb, G, 4))
        Yb = np.broadcast_to(Y, (nb, G, 4))
        mx = np.einsum("bg,bgq->bq", W, Xb) / sw[:, None]
        my = np.einsum("bg,bgq->bq", W, Yb) / sw[:, None]
        c = np.matmul((Xb * W[:, :, None]).transpose(0, 2, 1), Yb) / sw[:, None, None] - mx[:, :, None] * my[:, None, :]
        return (c * MASK).sum((1, 2))

    TOT = {(j, k): cov(I[j], I[k]) for j in KN for k in KN}
    EM = {(j, k): cov(I[j], IE[k]) for j in KN for k in LATER}
    r = {}
    for k in KN:
        r["tot_" + k] = sum(TOT[j, k] + TOT[k, j] for j in KN)
    r["tot"] = 2 * sum(TOT.values())
    r["E"] = 2 * sum(EM.values())
    r["S"] = r["tot"] - r["E"]
    for k in LATER:
        r["Elate_" + k] = 2 * sum(EM[j, k] for j in KN)
    for j in KN:
        r["Eearly_" + j] = 2 * sum(EM[j, k] for k in LATER)
        for k in LATER:
            r[f"Emat_{j}_{k}"] = 2 * EM[j, k]
    IT = sum(I[k] for k in KN)
    Ic = msg.sum(2)
    Ip = A["npl"].reshape(G, 4, 5).sum(2)
    r["dir_count"] = 2 * cov(Ic, Ic)
    r["dir_plays"] = 2 * cov(Ip, Ip)
    r["lead_to_count"] = 2 * cov(IT, Ic)
    r["lead_to_plays"] = 2 * cov(IT, Ip)
    return r


def run_stats(A):
    G = A["G"]
    chunks = [stats(A, np.ones((1, G)))]
    done = 0
    while done < NB:
        nb = min(CH, NB - done)
        W = rng.multinomial(G, np.ones(G) / G, size=nb).astype(float)
        chunks.append(stats(A, W))
        done += nb
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def report_cov(R, S):
    def line(nm, lab):
        r, s = R[nm], S[nm]
        return f"  {lab}: real {r[0]:+.2f} | sim {s[0]:+.2f} | sim-real {iv(s[1:] - r[1:], s[0] - r[0])}"

    say("## A. cross-quarter covariance (2x cov summed over the 6 quarter pairs, home view, strength-adjusted drive points), components of a = N + F + L + EFF + CONV")
    say("   N drive count x mean pts per drive; F start-field-position expectation; L score-band state mean; EFF residual of summed play EPA; CONV residual of (points - summed play EPA)")
    say("   tot_k = cov(I_k early, I_total later) + cov(I_total early, I_k later), summed over k gives total")
    say(line("tot", "total (matches catchup 'sum total')"))
    say(line("S", "S state response"))
    say(line("E", "E within-band residual"))
    say("  attribution of total by component")
    for k in KN:
        say(line("tot_" + k, f"{k:5s} {KDESC[k]}"))
    say("  attribution of E by LATER-quarter residual component (early side is the full total)")
    for k in LATER:
        say(line("Elate_" + k, f"{k:5s} {KDESC[k]}"))
    say("  attribution of E by EARLY-quarter component")
    for j in KN:
        say(line("Eearly_" + j, f"{j:5s} {KDESC[j]}"))
    say("  E matrix early component x later residual component")
    for j in KN:
        for k in LATER:
            say(line(f"Emat_{j}_{k}", f"early {j:5s} -> later {k:5s}"))
    say("  tempo, direct (signed home-minus-away counts)")
    say(line("dir_count", "cross-quarter cov of signed drive count"))
    say(line("dir_plays", "cross-quarter cov of signed live plays"))
    say(line("lead_to_count", "early adjusted points -> later signed drive count"))
    say(line("lead_to_plays", "early adjusted points -> later signed live plays"))


def behave_table(Ar, As):
    say("## B. drive outcomes by quarter (last quarter of drive) x offence lead band at drive start; real | sim, sim-real, game bootstrap (conv = points minus summed play EPA, eff = summed play EPA per drive)")
    Wr = rng.multinomial(Ar["G"], np.ones(Ar["G"]) / Ar["G"], size=NB).astype(float)
    Ws = rng.multinomial(As["G"], np.ones(As["G"]) / As["G"], size=NB).astype(float)
    names = ["conv", "eff", "plays", "TD", "FG", "punt"]
    for q in range(4):
        say(f"  Q{q + 1}")
        for b in range(5):
            c = q * 5 + b
            row = []
            for nm in names:
                def f(A, W):
                    return (W @ A["Sv"][nm][:, c]) / np.maximum(W @ A["cnt"][:, c], 1e-9)

                pr, ps = f(Ar, np.ones((1, Ar["G"])))[0], f(As, np.ones((1, As["G"])))[0]
                d = f(As, Ws) - f(Ar, Wr)
                row.append(f"{nm} {pr:.3f}|{ps:.3f} {iv(d, ps - pr)}")
            nr = (Ar["cnt"][:, c].sum() / Ar["G"], As["cnt"][:, c].sum() / As["G"])
            say(f"    {BANDS[b]:8s} drives/g {nr[0]:.3f}|{nr[1]:.3f} ; " + " ; ".join(row))


def design(D, ns):
    z = np.clip(D.sd0.to_numpy(float), -21, 21) / 7.0
    g = D.gsr0.to_numpy(float)
    qs = np.clip(np.floor((3600.0 - g) / 900.0), 0, 3).astype(int)
    h = np.clip((g - 1800.0 * (g > 1800.0)) / 1800.0, 0, 1)
    h2 = (qs >= 2).astype(float)
    X = np.column_stack([np.ones(len(D)), qs == 1, qs == 2, qs == 3, z, z * z, z * h, z * h * h, z * h2, z * h * h2]).astype(float)
    cell = qs * 5 + pd.cut(D.sd0, ns["LEAD_EDGES"], labels=False).astype(int).to_numpy()
    return X, cell


YN = ["conv", "eff", "plays", "TD", "FG"]


def yvars(D):
    return {"conv": (D.p - D.e).to_numpy(float), "eff": D.e.to_numpy(float), "plays": D.nlive.to_numpy(float), "TD": (D.out == "TD").to_numpy(float), "FG": (D.out == "FG").to_numpy(float)}


CNAME = ["1", "Q2", "Q3", "Q4", "z", "z2", "z*h", "z*h2", "z*H2", "z*h*H2"]
CONTR = [("H1 start", 0.875, 0.0), ("H1 end", 0.125, 0.0), ("H2 start", 0.875, 1.0), ("H2 end", 0.125, 1.0)]


def contrast_vec(h, H2):
    v = np.zeros(10)
    v[4], v[6], v[7], v[8], v[9] = 2.0, 2.0 * h, 2.0 * h * h, 2.0 * H2, 2.0 * h * H2
    return v


def loso(Dr, ns):
    say("## C. real LOSO (nine held-out seasons) smooth state x time form vs quarter-by-band cell means; per-drive squared error; gain = cell-mean error minus smooth error (positive favours smooth)")
    say("   smooth form columns: " + ", ".join(CNAME) + "; z = clip(lead,+-21)/7, h = fraction of the half left at drive start, H2 = second-half flag")
    X, cell = design(Dr, ns)
    gi, gn = pd.factorize(Dr.gk)
    G = len(gn)
    seas = Dr.season.to_numpy()
    Y = yvars(Dr)
    W = rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)
    for yn in YN:
        y = Y[yn]
        pa, pb = np.zeros(len(y)), np.zeros(len(y))
        for s in np.unique(seas):
            tr, te = seas != s, seas == s
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            pb[te] = X[te] @ beta
            cm = np.bincount(cell[tr], weights=y[tr], minlength=20) / np.maximum(np.bincount(cell[tr], minlength=20), 1)
            pa[te] = cm[cell[te]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        ins_b = float(((y - X @ beta) ** 2).mean())
        cm = np.bincount(cell, weights=y, minlength=20) / np.maximum(np.bincount(cell, minlength=20), 1)
        ins_a = float(((y - cm[cell]) ** 2).mean())
        ea, eb = (y - pa) ** 2, (y - pb) ** 2
        dg = np.bincount(gi, weights=ea - eb, minlength=G)
        cn = np.bincount(gi, minlength=G).astype(float)
        gain = (W @ dg) / (W @ cn)
        pt = float(dg.sum() / cn.sum())
        say(f"  {yn:6s} out-of-season MSE cells {ea.mean():.4f} smooth {eb.mean():.4f}; in-sample cells {ins_a:.4f} smooth {ins_b:.4f}; optimism (out-in) cells {ea.mean() - ins_a:+.4f} smooth {eb.mean() - ins_b:+.4f}; gain {iv(gain, pt)}")


def coef_fit(Dr, Ds, ns):
    say("## D. real vs sim smooth-form coefficients and leader-minus-trailer contrasts (a 2*z contrast: lead +7 versus trail 7, per drive), game bootstrap; coefficients in per-drive units of each y")
    out = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        X, _ = design(D, ns)
        gi, gn = pd.factorize(D.gk)
        G = len(gn)
        XtX = np.stack([np.bincount(gi, weights=X[:, i] * X[:, j], minlength=G) for i in range(10) for j in range(10)], 1)
        Y = yvars(D)
        Xty = {yn: np.stack([np.bincount(gi, weights=X[:, i] * Y[yn], minlength=G) for i in range(10)], 1) for yn in YN}
        W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])
        M = (W @ XtX).reshape(len(W), 10, 10)
        out[nm] = {yn: np.linalg.solve(M, (W @ Xty[yn])[:, :, None])[:, :, 0] for yn in YN}
    for yn in YN:
        say(f"  {yn}")
        br, bs = out["real"][yn], out["sim"][yn]
        for i in (4, 5, 6, 7, 8, 9):
            say(f"    {CNAME[i]:7s} real {br[0, i]:+.4f} sim {bs[0, i]:+.4f} sim-real {iv(bs[1:, i] - br[1:, i], bs[0, i] - br[0, i])}")
        for lab, h, H2 in CONTR:
            v = contrast_vec(h, H2)
            cr, cs = br @ v, bs @ v
            say(f"    leader-trailer {lab:9s} real {cr[0]:+.4f} sim {cs[0]:+.4f} sim-real {iv(cs[1:] - cr[1:], cs[0] - cr[0])}")


def main():
    import mod25e_late as late

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    Dr = real_build(ns)
    epa_t = epa_table()
    Ds, cov = sim_build(epa_t, late)
    say(f"label {LABEL} seeds {SEEDS}; bootstrap {NB}; real drives {len(Dr)} sim drives {len(Ds)}; sim live rows with a source EPA {cov['ok']}/{cov['live']}")
    Ar, As = prep(Dr, ns), prep(Ds, ns)
    say(f"real games {Ar['G']} sim games {As['G']}; decomposition identity max abs error real {Ar['chk']:.2e} sim {As['chk']:.2e}")
    say(f"mean pts/drive real {Ar['mean_p']:+.4f} sim {As['mean_p']:+.4f}; mean summed EPA/drive real {Ar['mean_e']:+.4f} sim {As['mean_e']:+.4f} (sim EPA is the drawn source row's EPA)")
    say("   caveat: sim EPA is the source row's EPA, not a re-evaluation of the sim state; CONV therefore absorbs state drift between drawn EPA and realised points")
    R, S = run_stats(Ar), run_stats(As)
    report_cov(R, S)
    behave_table(Ar, As)
    loso(Dr, ns)
    coef_fit(Dr, Ds, ns)
    say(f"looks counted (interval prints): {LOOKS[0]}")
    (OUTD / "drivecomp.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()

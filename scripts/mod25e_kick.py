import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e3" / "kick"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
SIMD = REPO / "artifacts" / "mod25e3" / "revert_sim" / "crzhc"
POOL = tuple(range(2009, 2018))
LATE = tuple(range(2018, 2026))
LIVE = ("run", "pass", "punt", "field_goal")
KD_EDGES = [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf]
KD_NAMES = ["K trail>8", "K trail1-8", "tied", "K lead1-8", "K lead>8"]
T_EDGES = [-1, 300, 900, 1800, 3600]
T_NAMES = ["last5", "5-15", "Q2-3", "Q1/Q2"]
NBOOT = 300
BURN = int(__import__("os").environ.get("KICK_BURN", 2))


def load_xq():
    src = (REPO / "scripts" / "mod25e_xq.py").read_text().rsplit("\ncommon()", 1)[0].replace('"n": gp.size(),', '"n": gp.size(), "gsr0": gp.game_seconds_remaining.first(),')
    ns = {"__name__": "mod25e_xq_lib", "__file__": str(REPO / "scripts" / "mod25e_xq.py")}
    exec(compile(src, "mod25e_xq_lib", "exec"), ns)
    return ns


def real_kicks(seasons):
    cols = ["game_id", "play_id", "season_type", "qtr", "game_seconds_remaining", "posteam", "defteam", "home_team", "play_type", "yardline_100", "score_differential", "touchdown", "yards_gained", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
    out = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[p.season_type == "REG"].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        live = p.play_type.isin(LIVE).to_numpy()
        n = len(p)
        idx = np.where(live, np.arange(n), n)
        nxt = np.minimum.accumulate(idx[::-1])[::-1]
        nxt = np.r_[nxt[1:], n]
        ok = nxt < n
        nj = np.where(ok, nxt, 0)
        same = ok & (p.game_id.to_numpy()[nj] == p.game_id.to_numpy())
        ko = (p.play_type == "kickoff").to_numpy()
        k = p[ko & same & p.score_differential.notna().to_numpy()].copy()
        j = nj[k.index.to_numpy()]
        k["season"] = s
        k["np"] = p.posteam.to_numpy()[j]
        k["nfp"] = p.yardline_100.to_numpy()[j]
        k["ngsr"] = p.game_seconds_remaining.to_numpy()[j]
        k["nqtr"] = p.qtr.to_numpy()[j]
        k["retained"] = (k.np == k.defteam).astype(int)
        k["kd"] = -k.score_differential
        k["gsr"] = k.game_seconds_remaining
        k["ktos"] = k.defteam_timeouts_remaining
        k["rtos"] = k.posteam_timeouts_remaining
        k["rtd"] = (k.touchdown == 1).astype(int)
        k["kd_next"] = k.groupby("game_id").kd.shift(-1)
        out.append(k[["game_id", "play_id", "season", "qtr", "gsr", "kd", "ktos", "rtos", "retained", "nfp", "ngsr", "rtd", "yards_gained", "kd_next"]])
    K = pd.concat(out, ignore_index=True)
    K = K[(K.qtr <= 4) & ~K.gsr.isin([3600, 1800])].reset_index(drop=True)
    K["pat_extra"] = np.where(K.rtd == 1, (K.kd_next + K.kd - 6).clip(0, 2), np.nan)
    return K


def tag(K):
    K = K.assign(kb=pd.cut(K.kd, KD_EDGES, labels=KD_NAMES).astype(str), tb=pd.cut(K.gsr, T_EDGES, labels=T_NAMES[::-1]).astype(str))
    modal = K.groupby("season").nfp.agg(lambda x: x[K.loc[x.index, "retained"] == 0].mode().iloc[0])
    K["tbk"] = ((K.nfp == modal.reindex(K.season).to_numpy()) & (K.retained == 0) & (K.rtd == 0)).astype(int)
    K["short"] = ((K.retained == 0) & (K.rtd == 0) & (K.nfp <= 55)).astype(int)
    K["att"] = ((K.retained == 1) | (K.short == 1)).astype(int)
    return K


def kick_stats(K, name, L):
    n = len(K)
    L.append(f"== {name}: {n} after-score kickoffs ({K.season.nunique()} seasons), {n / K.game_id.nunique():.2f}/game")
    L.append(f"  kicker retains {K.retained.mean():.4f} return TD {K.rtd.mean():.4f} touchback-spot {K.tbk.mean():.3f} short(<=55, non-ret) {K.short.mean():.4f}")
    nr = K[(K.retained == 0) & (K.rtd == 0)]
    L.append(f"  receiver start fp (non-retained): mean {nr.nfp.mean():.2f} sd {nr.nfp.std():.2f} var {nr.nfp.var():.2f}; retained start fp mean {K[K.retained == 1].nfp.mean():.1f}")
    ctrl = K[(K.kd >= 0) & (K.gsr > 1800)]
    L.append(f"  control (kicker leads/tied, before Q3): attempt-proxy {ctrl.att.mean():.4f} retained {ctrl.retained.mean():.4f}")
    rows = []
    for kb in KD_NAMES:
        for tb in T_NAMES:
            d = K[(K.kb == kb) & (K.tb == tb)]
            if len(d) >= 15:
                rows.append(f"    {kb:11s} {tb:6s} n {len(d):4d} att {d.att.mean():.3f} retain {d.retained.mean():.3f} rec|att {d.retained.sum() / max(d.att.sum(), 1):.2f} rtd {d.rtd.mean():.4f} fp {d[(d.retained == 0) & (d.rtd == 0)].nfp.mean():.1f}")
    L.extend(rows)
    d = K[(K.kb.isin(["K trail>8", "K trail1-8"])) & (K.gsr <= 900)]
    for t in sorted(d.ktos.dropna().unique()):
        e = d[d.ktos == t]
        L.append(f"  trailing kicker, <=15 min, kicker timeouts {int(t)}: n {len(e)} att {e.att.mean():.3f} retain {e.retained.mean():.3f}")
    for t in sorted(d.rtos.dropna().unique()):
        e = d[d.rtos == t]
        L.append(f"  trailing kicker, <=15 min, receiver timeouts {int(t)}: n {len(e)} att {e.att.mean():.3f} retain {e.retained.mean():.3f}")


def season_line(K, L):
    L.append("per season: n retain rtd tbk short fp_mean fp_sd")
    for s, d in K.groupby("season"):
        nr = d[(d.retained == 0) & (d.rtd == 0)]
        L.append(f"  {s} {len(d):4d} {d.retained.mean():.4f} {d.rtd.mean():.4f} {d.tbk.mean():.3f} {d.short.mean():.4f} {nr.nfp.mean():.1f} {nr.nfp.std():.1f}")


def drives_from_sim():
    out = []
    for f in sorted(SIMD.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
        g = d.g.to_numpy()
        oh = d.offhome.to_numpy()
        q = d.qtr.to_numpy()
        sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
        d["d"] = np.cumsum(new)
        d["pp"] = d.po - d.pdf
        d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
        d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
        gp = d.groupby("d", sort=True)
        key = w * 1000 + s
        D = pd.DataFrame({"gk": f"{key}_" + gp.g.first().astype(int).astype(str), "season": key, "off": gp.off.first(), "dfn": gp.dfn.first(),
                          "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "gsr0": gp.gsr.first(), "n": gp.size(),
                          "sd0": gp.sd.first(), "p": gp.pp.sum()})
        D["tm"] = D.season.astype(str) + "_" + D.off.astype(int).astype(str)
        D["td"] = D.season.astype(str) + "_" + D.dfn.astype(int).astype(str)
        out.append(D)
    return pd.concat(out, ignore_index=True)


def drives_from_real(xq, seasons):
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential"]
    D, _ = xq["real_drives"](seasons)
    gs = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "play_type", "score_differential"])
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4) & (p.play_type != "kickoff") & p.play_type.notna()]
        p = p.sort_values(["game_id", "play_id"], kind="stable")
        gs.append(p.groupby("game_id").size())
    return D


def seq_table(D, name, L):
    g = D.gk.to_numpy()
    nxt_same_game = np.r_[g[1:] == g[:-1], False]
    off = D.off.to_numpy()
    nxt_off = np.concatenate([off[1:], off[-1:]])
    retained = nxt_same_game & (nxt_off == off)
    p = D.p.to_numpy()
    nxt_p = np.r_[p[1:], 0.0]
    q4 = (D.ql.to_numpy() == 4)
    trail = D.sd0.to_numpy() < 0
    scored = p > 0
    sel = q4 & trail & scored & nxt_same_game
    ret = retained[sel]
    leader_scores = (nxt_p[sel] > 0) & ~ret
    L.append(f"== {name} Q4 sequence: trailer (trailing at drive start) scores in Q4, then next drive, n {int(sel.sum())}")
    L.append(f"  P(trailer keeps ball: onside-type retention) {ret.mean():.4f} (n {int(ret.sum())})")
    nr = ~ret
    L.append(f"  P(leader scores on next drive | not retained) {leader_scores[nr].mean():.4f} (n {int(nr.sum())})")
    L.append(f"  P(trailer scores again on the retained drive) {(nxt_p[sel][ret] > 0).mean() if ret.any() else float('nan'):.4f}")
    sd0n = D.sd0.to_numpy()[sel] + p[sel]
    close = sd0n <= 0
    L.append(f"  of which trailer still trailing/tied after score: n {int(close.sum())}, retain {ret[close].mean():.4f}, leader scores next {leader_scores[close & nr].mean() if (close & nr).any() else float('nan'):.4f}")
    gsr = D.gsr0.to_numpy()[sel] if "gsr0" in D else None
    if gsr is not None:
        for lo, hi in ((0, 300), (300, 900)):
            m = (gsr >= lo) & (gsr < hi)
            if m.any():
                L.append(f"  gsr at trailer drive start [{lo},{hi}): n {int(m.sum())} retain {ret[m].mean():.4f} leader scores next {leader_scores[m & nr].mean() if (m & nr).any() else float('nan'):.4f}")


def pair_cov(D, xq, name, L, res):
    D, _ = xq["components"](D)
    D = D.assign(sc_prev=False)
    r = D.c_rest.to_numpy()
    s = D.s.to_numpy()
    gk = D.gk.to_numpy()
    q = D.ql.to_numpy()
    p = D.p.to_numpy()
    codes, starts = np.unique(gk, return_index=True)
    order = np.argsort(starts)
    bounds = list(starts[order]) + [len(D)]
    tot = np.zeros(4)
    adj_after = np.zeros(4)
    adj_other = np.zeros(4)
    ng = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        rr, ss, qq, pp = r[a:b], s[a:b], q[a:b], p[a:b]
        nh = int((ss > 0).sum())
        na = int((ss < 0).sum())
        if nh == 0 or na == 0:
            continue
        ng += 1
        M = np.outer(rr, rr) * (np.outer(ss, ss) < 0) / (nh * na)
        M = np.triu(M) + np.tril(M, -1).T * 0
        full = np.outer(rr, rr) / (nh * na) * (np.outer(ss, ss) < 0)
        n = b - a
        ii = np.arange(n - 1)
        opp = ss[ii] != ss[ii + 1]
        adj_val = full[ii, ii + 1] * opp
        scored_prev = pp[ii] > 0
        late = qq[ii] == 4
        both = full.sum() / 2
        tot[0] += both
        tot[1] += adj_val[scored_prev].sum()
        tot[2] += adj_val[~scored_prev].sum()
        tot[3] += adj_val[scored_prev & late].sum()
    res[name] = [float(x / ng) for x in tot]
    L.append(f"{name}: cov(H,A) game-mean residual pair sum {tot[0] / ng:+.4f}; adjacent opposite pairs after a score {tot[1] / ng:+.4f} (Q4 {tot[3] / ng:+.4f}), after a non-score {tot[2] / ng:+.4f}  ({ng} games)")


def cmd_an(a):
    xq = load_xq()
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    res = {}
    for name, seasons in (("pool 2009-17", POOL), ("late 2018-25", LATE)):
        K = tag(real_kicks(seasons))
        kick_stats(K, name, L)
        if name.startswith("late"):
            season_line(K, L)
        K.to_parquet(OUT / f"kicks_{seasons[0]}.parquet")
    Kp = tag(real_kicks(POOL))
    season_line(Kp, L)
    drv = {"pool": xq["real_drives"](POOL)[0], "late": xq["real_drives"](LATE)[0], "sim": drives_from_sim()}
    for k, D in drv.items():
        seq_table(D, k, L)
    sims = drv["sim"]
    ksim = sims[(sims.ql <= 4)]
    for k, D in drv.items():
        pair_cov(D, xq, k, L, res)
    (OUT / "an.json").write_text(json.dumps(res))
    (OUT / "an.txt").write_text("\n".join(L))
    print("\n".join(L))


def install_kick():
    import mod25d_variance as dv

    cfg = dv._G["cfg"]
    mode = int(cfg.get("kick", 1))
    seed = int(cfg.get("seed", 3))
    K = pd.read_parquet(OUT / f"kicks_{POOL[0]}.parquet")
    kd = K.kd.to_numpy(float)
    gs = K.gsr.to_numpy(float)
    sk, st_ = float(kd.std()), float(gs.std())
    ret = K.retained.to_numpy(int)
    nfp = K.nfp.to_numpy(float)
    rtd = K.rtd.to_numpy(int)
    pe = K.pat_extra.dropna().to_numpy(int)
    ok_all = np.ones(len(K), dtype=bool)
    if os.environ.get("OKK") == "1":
        ok_all = ~((gs > 1800.0) & (K.ngsr.to_numpy(float) <= 1800.0))
    ok_nt = ok_all & (rtd == 0)
    state = {"k": None, "rng": None}

    def draw(rng, kdv, tv, mask):
        idx = np.flatnonzero(mask)
        d = ((kd[idx] - kdv) / sk) ** 2 + ((gs[idx] - tv) / st_) ** 2
        m = int(np.sqrt(len(idx)))
        near = idx[np.argpartition(d, m - 1)[:m]]
        return int(near[int(rng.integers(0, len(near)))])

    base = dv._G["pol"]

    after = os.environ.get("KGZ") == "1"

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if after:
            fr = sys._getframe(1)
            while fr is not None and "offense" not in fr.f_locals:
                fr = fr.f_back
            offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
            qtr_l, gsr, possessions = fr.f_locals.get("qtr"), fr.f_locals.get("gsr"), fr.f_locals.get("possessions")  # noqa: F841
            drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        po = float(drawn["points_off"])
        pdf = float(drawn["points_def"])
        if mode == 0 or qtr > 4 or not ((po >= 3 and pdf == 0) or (pdf >= 6 and po == 0)):
            return drawn if after else base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(15485863, seed)
        rng = state["rng"]
        k_off = po > 0
        kdv = score_diff + po if k_off else pdf - score_diff
        t = max(float(clock_val) - float(drawn["clock_elapsed"]), 0.0)
        i = draw(rng, kdv, t, ok_all)
        new = dict(drawn)
        holder_k = bool(ret[i])
        fp = nfp[i]
        if rtd[i]:
            b = int(pe[int(rng.integers(0, len(pe)))]) if len(pe) else 1
            if k_off:
                new["points_def"] = 6.0 + b
            else:
                new["points_off"] = 6.0 + b
            j = draw(rng, 6.0 + b - kdv, t, ok_nt)
            holder_k = not bool(ret[j])
            fp = nfp[j]
        holder_off = holder_k == k_off
        new["flip"] = not holder_off
        new["next_down"] = 1.0
        new["next_distance"] = min(10.0, float(fp))
        new["next_yardline"] = float(fp)
        return new if after else base(down, distance, yardline, score_diff, qtr, clock_val, new)

    dv._G["pol"] = pol


def kick_init_ss(setting):
    import sim09_hk as hk

    hk.hk_init_ss(setting)
    install_kick()


def kick_init_budget(setting):
    import sim09_hk as hk

    hk.hk_init_budget(setting)
    install_kick()


def kick_variant(mode):
    import sim09_hk as hk

    hk.dv.DV["crzhq"] = dict(hk.dv.DV["crzhk"], kick=mode)
    return hk


def cmd_sim(a):
    import os

    import mod25_generator as gen
    import mod25e_budget as bud
    import mod25e_cov as cv
    import sim09_u3a as u3

    hk = kick_variant(a.mode)
    cv.patch_generator()
    bud.OUT = Path(a.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    hk.dv.c25.ensure_policies()
    v = hk.dv.DV["crzhq"]
    cfgj = json.dumps(dict(v, seed=a.seed, name=f"crzhq_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = kick_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, a.worlds, a.seasons, a.workers, a.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    np.savez(bud.OUT / "latents.npz", **{f"{k}_{w}_{s}": v2 for (w, s), (wk, qb) in latents.items() for k, v2 in (("w", wk), ("q", qb))})
    print("done", len(games), el, flush=True)


def cmd_e5(a):
    import mod25e_cov as cv
    import mod25e_scorestate as ss

    kick_variant(a.mode)
    cv.patch_generator()
    ss.s_init = kick_init_ss
    a.variant = "crzhq"
    a.scale = 1.0
    ss.cmd_e5(a)


def sim_kicks(sim_dir):
    out = []
    for f in sorted(Path(sim_dir).glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
        g = d.g.to_numpy()
        po = d.po.to_numpy()
        pdf = d.pdf.to_numpy()
        oh = d.offhome.to_numpy()
        score = ((po >= 3) & (pdf == 0)) | ((pdf >= 6) & (po == 0)) | ((po >= 3) & (pdf >= 6))
        nxt_ok = np.r_[g[1:] == g[:-1], False] & score
        i = np.flatnonzero(nxt_ok)
        both = (po[i] > 0) & (pdf[i] > 0)
        k_off = po[i] > 0
        lead_off = d.sd.to_numpy()[i] + po[i] - pdf[i]
        kd = np.where(k_off, lead_off, -lead_off)
        nxt_oh = oh[i + 1]
        holder_off = nxt_oh == oh[i]
        retained = np.where(k_off, holder_off, ~holder_off).astype(int)
        out.append(pd.DataFrame({"game_id": f"{w * 1000 + s}_" + d.g.to_numpy()[i].astype(int).astype(str), "season": w * 1000 + s, "qtr": d.qtr.to_numpy()[i], "gsr": d.gsr.to_numpy()[i + 1], "kd": kd,
                                 "ktos": np.nan, "rtos": np.nan, "retained": retained, "nfp": d.yl.to_numpy()[i + 1], "rtd": both.astype(int)}))
    return pd.concat(out, ignore_index=True)


def cmd_val(a):
    xq = load_xq()
    L = []
    global SIMD
    SIMD = Path(a.sim_dir)
    K = tag(sim_kicks(SIMD))
    kick_stats(K, f"sim {SIMD.name}", L)
    D = drives_from_sim()
    seq_table(D, f"sim {SIMD.name}", L)
    res = {}
    pair_cov(D, xq, f"sim {SIMD.name}", L, res)
    txt = "\n".join(L)
    (OUT / f"val_{SIMD.name}.txt").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("an")
    s = sub.add_parser("sim")
    s.add_argument("--out-dir", dest="out_dir", required=True)
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=21)
    s.add_argument("--mode", type=int, default=1)
    e = sub.add_parser("e5")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    e.add_argument("--mode", type=int, default=1)
    v = sub.add_parser("val")
    v.add_argument("--sim-dir", dest="sim_dir", required=True)
    a = ap.parse_args()
    {"an": cmd_an, "sim": cmd_sim, "e5": cmd_e5, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    main()

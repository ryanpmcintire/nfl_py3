import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25_generator as gen  # noqa: E402

E3 = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokgndecsmfwtjo2as2"
SEEDS = (11, 12, 13)
BURN = 2
REAL_SEASONS = tuple(range(2011, 2018))
NB = 300
KEYS = (3, 7, 14)
ENTER_T = {"last5": 300.0, "endQ3": 900.0}
BUCKETS = [(0, 0), (1, 3), (4, 7), (8, 10), (11, 14), (15, 21), (22, 99)]
STATES = [("trail>8", -99, -9), ("trail4-8", -8, -4), ("trail1-3", -3, -1), ("tied", 0, 0), ("lead1-3", 1, 3), ("lead4-8", 4, 8), ("lead>8", 9, 99)]
OUT = []
LOOKS = [0]
rng = np.random.default_rng(96)
COLS = ["game_id", "play_id", "season_type", "posteam", "home_team", "away_team", "qtr", "game_seconds_remaining", "score_differential", "posteam_score"]


def say(s=""):
    OUT.append(s)
    print(s)


def real_events():
    gf = gen.real_games(REAL_SEASONS)
    fin = gf.set_index("game_id")
    rows = []
    for s in REAL_SEASONS:
        p = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=COLS)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & p.posteam_score.notna()]
        rows.append(p)
    p = pd.concat(rows, ignore_index=True).sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    ph = (p.posteam == p.home_team).to_numpy()
    oth = (p.posteam_score - p.score_differential).to_numpy()
    p["h"] = np.where(ph, p.posteam_score.to_numpy(), oth)
    p["a"] = np.where(ph, oth, p.posteam_score.to_numpy())
    p["t"] = p.game_seconds_remaining.groupby(p.game_id).ffill()
    g = p.game_id.to_numpy()
    last = np.r_[g[1:] != g[:-1], True]
    fh = fin.home_score.reindex(p.game_id).to_numpy()
    fa = fin.away_score.reindex(p.game_id).to_numpy()
    nh = np.where(last, fh, np.r_[p.h.to_numpy()[1:], 0.0])
    na = np.where(last, fa, np.r_[p.a.to_numpy()[1:], 0.0])
    dh = nh - p.h.to_numpy()
    da = na - p.a.to_numpy()
    ev = []
    for side, d in ((1, dh), (-1, da)):
        m = d > 0
        ev.append(pd.DataFrame({"gk": g[m], "ord": np.nonzero(m)[0], "side": side, "pts": d[m], "t": p.t.to_numpy()[m], "q": p.qtr.to_numpy()[m]}))
    ev = pd.concat(ev, ignore_index=True).sort_values(["ord", "side"], kind="stable").reset_index(drop=True)
    games = pd.DataFrame({"gk": gf.game_id.to_numpy(), "margin": (gf.home_score - gf.away_score).to_numpy(dtype=float)})
    return ev, games


def sim_events():
    ev, games = [], []
    for sd in SEEDS:
        sdir = E3 / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_score", "away_score"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        sg = sg[sg.s >= BURN]
        sg["gk"] = f"{sd}_" + sg.w.astype(str) + "_" + sg.s.astype(str) + "_" + sg.g.astype(str)
        games.append(pd.DataFrame({"gk": sg.gk, "margin": (sg.home_score - sg.away_score).to_numpy(dtype=float)}))
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "gsr", "offhome", "po", "pdf", "qtr"])
            d = d.sort_values("g", kind="stable").reset_index(drop=True)
            gkey = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            for pts, mult in (("po", 1), ("pdf", -1)):
                m = (d[pts] > 0).to_numpy()
                side = np.where(d.offhome.to_numpy()[m] == 1, 1, -1) * mult
                ev.append(pd.DataFrame({"gk": gkey.to_numpy()[m], "ord": np.nonzero(m)[0] + (sd * 100 + w * 10 + s) * 10**7, "side": side, "pts": d[pts].to_numpy()[m], "t": d.gsr.to_numpy()[m], "q": d.qtr.to_numpy()[m]}))
    ev = pd.concat(ev, ignore_index=True).sort_values(["ord", "side"], kind="stable").reset_index(drop=True)
    return ev, pd.concat(games, ignore_index=True)


def classify(ev):
    g, sd, pt = ev.gk.to_numpy(), ev.side.to_numpy(), ev.pts.to_numpy()
    ng = np.r_[g[1:] == g[:-1], False]
    nsd = np.r_[sd[1:], 0]
    npt = np.r_[pt[1:], 0]
    merged = (pt == 6) & ng & (nsd == sd) & ((npt == 1) | (npt == 2))
    drop = np.r_[False, merged[:-1]]
    typ = np.select([pt == 6, pt == 7, pt == 8, pt == 3, pt == 2], ["td6", "td7", "td8", "fg", "saf"], "oth")
    typ = np.where(merged & (npt == 1), "td7", np.where(merged & (npt == 2), "td8", typ))
    pts = np.where(merged, pt + npt, pt)
    out = ev.assign(typ=typ, spts=pts * sd)[~drop].reset_index(drop=True)
    out["istd"] = out.typ.isin(["td6", "td7", "td8"])
    return out


def build(ev, games):
    games = games.reset_index(drop=True)
    idx = pd.Index(games.gk)
    ev = ev[ev.gk.isin(idx)].reset_index(drop=True)
    gi = idx.get_indexer(ev.gk)
    ev["gi"] = gi
    ev["cum"] = ev.groupby("gi").spts.cumsum() - ev.spts
    ev["pre_scorer"] = ev.cum * ev.side
    rec = np.bincount(gi, weights=ev.spts, minlength=len(games))
    return ev, games, rec


def cnt(ev, G, sel, side=None):
    m = sel
    if side is not None:
        m = m & (ev.side.to_numpy() == side)
    return np.bincount(ev.gi.to_numpy()[m], minlength=G).astype(float)


def features(ev, games):
    G = len(games)
    F = {}
    mg = games.margin.to_numpy()
    sg = np.sign(mg)
    am = np.abs(mg)
    typ = ev.typ.to_numpy()
    for tp in ("td6", "td7", "td8", "fg", "saf", "oth"):
        F["n_" + tp] = cnt(ev, G, typ == tp)
        for side, nm in ((1, "h"), (-1, "a")):
            F[f"n_{tp}_{nm}"] = cnt(ev, G, typ == tp, side)
    F["n_td"] = F["n_td6"] + F["n_td7"] + F["n_td8"]
    tdh = F["n_td6_h"] + F["n_td7_h"] + F["n_td8_h"]
    tda = F["n_td6_a"] + F["n_td7_a"] + F["n_td8_a"]
    dTD = np.where(sg >= 0, tdh - tda, tda - tdh)
    dFG = np.where(sg >= 0, F["n_fg_h"] - F["n_fg_a"], F["n_fg_a"] - F["n_fg_h"])
    flags = np.array([("S" if F["n_saf"][i] > 0 else "") + ("2" if F["n_td8"][i] > 0 else "") + ("M" if F["n_td6"][i] > 0 else "") for i in range(G)])
    lastidx = ev.groupby("gi").tail(1)
    li = lastidx.gi.to_numpy()
    lt = np.full(G, "none", dtype=object)
    lr = np.full(G, "none", dtype=object)
    lt[li] = np.where(lastidx.istd.to_numpy(), "td", lastidx.typ.to_numpy())
    lr[li] = np.where(sg[li] == 0, "tiegame", np.where(lastidx.side.to_numpy() == sg[li], "winner", "loser"))
    cells = {}
    for k in KEYS:
        sel = am == k
        F[f"m{k}"] = sel.astype(float)
        for i in np.nonzero(sel)[0]:
            cells.setdefault(f"m{k}|dTD{int(dTD[i]):+d} dFG{int(dFG[i]):+d} [{flags[i]}]", []).append(i)
            cells.setdefault(f"m{k}|last {lt[i]} by {lr[i]}", []).append(i)
    return F, cells, am


def states_matrix(ev, G):
    cols = {}
    pre = ev.pre_scorer.to_numpy()
    fg = (ev.typ == "fg").to_numpy()
    td = ev.istd.to_numpy()
    for nm, lo, hi in STATES:
        m = (pre >= lo) & (pre <= hi)
        cols[f"st {nm} fg"] = cnt(ev, G, m & fg)
        cols[f"st {nm} td"] = cnt(ev, G, m & td)
    return cols


def entering(ev, G, T):
    m = ev.t.to_numpy() > T
    return np.bincount(ev.gi.to_numpy()[m], weights=ev.spts.to_numpy()[m], minlength=G)


def boot_w(G):
    return np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)])


def fmt(a, d=4):
    return f"{a[0]:.{d}f} [{np.quantile(a[1:], .025):.{d}f},{np.quantile(a[1:], .975):.{d}f}]"


def fmtd(a, d=4):
    return f"{a[0]:+.{d}f} [{np.quantile(a[1:], .025):+.{d}f},{np.quantile(a[1:], .975):+.{d}f}]"


def stat_line(label, r, s, d=4):
    diff = s - r
    LOOKS[0] += 1
    return f"{label:44s} real {fmt(r, d)} sim {fmt(s, d)} s-r {fmtd(diff, d)} P(s-r>0) {np.mean(diff[1:] > 0):.2f}"


def main():
    (E3 / "e96").mkdir(parents=True, exist_ok=True)
    evr, gr = real_events()
    evs, gs = sim_events()
    evr, gr, recr = build(classify(evr), gr)
    evs, gs, recs = build(classify(evs), gs)
    say(f"E96 final-margin mass 3/7/14 decomposition; label {LABEL} sim seeds {SEEDS} burn {BURN}; real REG {REAL_SEASONS[0]}-{REAL_SEASONS[-1]}; game bootstrap {NB}; final margins include overtime in both")
    for nm, ev, g, rec in (("real", evr, gr, recr), ("sim", evs, gs, recs)):
        ok = np.mean(np.abs(rec - g.margin.to_numpy()) < 0.5)
        am = np.abs(g.margin)
        say(f"{nm}: games {len(g)} events {len(ev)} share of games where event sum equals final margin {ok:.4f} mass3 {np.mean(am == 3):.4f} mass7 {np.mean(am == 7):.4f} mass14 {np.mean(am == 14):.4f}")
    Fr, cr, amr = features(evr, gr)
    Fs, cs, ams = features(evs, gs)
    Gr, Gs = len(gr), len(gs)
    Wr, Ws = boot_w(Gr), boot_w(Gs)

    def rate(name):
        return (Wr @ Fr[name]) / Gr, (Ws @ Fs[name]) / Gs

    def ratio(a, b):
        return (Wr @ Fr[a]) / (Wr @ Fr[b]), (Ws @ Fs[a]) / (Ws @ Fs[b])

    say("")
    say("1a. per-game composition rates (all games)")
    for nm in ("n_td", "n_fg", "n_saf", "n_td6", "n_td7", "n_td8", "n_oth"):
        r, s = rate(nm)
        say(stat_line(nm + " per game", r, s))
    r, s = (Wr @ Fr["n_fg"]) / (Wr @ (Fr["n_fg"] + Fr["n_td"])), (Ws @ Fs["n_fg"]) / (Ws @ (Fs["n_fg"] + Fs["n_td"]))
    say(stat_line("FG share of TD+FG", r, s))
    for nm, a, b in (("PAT made per TD", "n_td7", "n_td"), ("2pt made per TD", "n_td8", "n_td"), ("no point after per TD", "n_td6", "n_td"), ("TD:FG ratio", "n_td", "n_fg")):
        r, s = ratio(a, b)
        say(stat_line(nm, r, s))
    say("")
    say("1b. FG share of TD+FG by score state (scorer's margin before the score) and scores per game by state")
    Mr, Ms = states_matrix(evr, Gr), states_matrix(evs, Gs)
    for nm, _, _ in STATES:
        fr, tr = Wr @ Mr[f"st {nm} fg"], Wr @ Mr[f"st {nm} td"]
        fs, ts = Ws @ Ms[f"st {nm} fg"], Ws @ Ms[f"st {nm} td"]
        say(stat_line(f"FG share when {nm} (n real {int(Mr[f'st {nm} fg'].sum() + Mr[f'st {nm} td'].sum())} sim {int(Ms[f'st {nm} fg'].sum() + Ms[f'st {nm} td'].sum())})", fr / (fr + tr), fs / (fs + ts)))
    for nm, _, _ in STATES:
        r = (Wr @ (Mr[f"st {nm} fg"] + Mr[f"st {nm} td"])) / Gr
        s = (Ws @ (Ms[f"st {nm} fg"] + Ms[f"st {nm} td"])) / Gs
        say(stat_line(f"scores per game when {nm}", r, s))
    say("")
    say("1c. mass and composition cells of final |margin| in 3, 7, 14 (rate per game); composition cell = winner-minus-loser TD count, FG count, flags S safety 2 two-point M no point after; last score = type of the game's last score and who scored it")
    for k in KEYS:
        r, s = rate(f"m{k}")
        say(stat_line(f"mass {k}", r, s))
        lab = sorted({c for c in list(cr) + list(cs) if c.startswith(f"m{k}|")})
        res = []
        for c in lab:
            ir = np.zeros(Gr)
            ir[cr.get(c, [])] = 1
            is_ = np.zeros(Gs)
            is_[cs.get(c, [])] = 1
            rr, ss = (Wr @ ir) / Gr, (Ws @ is_) / Gs
            res.append((ss[0] - rr[0], c, rr, ss))
        for grp in ("dTD", "last"):
            part = sorted([x for x in res if x[1].split("|")[1].startswith(grp)], key=lambda x: x[0])
            say(f" mass {k} by {'composition cell' if grp == 'dTD' else 'last score'} ({len(part)} cells), most negative s-r first")
            for d, c, rr, ss in part:
                say(stat_line("  " + c.split("|")[1], rr, ss))
    say("")
    say("2. P(final |margin| = k | |margin| entering the window), real vs sim, with shift/rate decomposition of the mass gap")
    for wn, T in ENTER_T.items():
        er, es = np.abs(entering(evr, Gr, T)), np.abs(entering(evs, Gs, T))
        say(f"window {wn} (entering margin from scores made before {T:.0f} seconds remaining)")
        for k in KEYS:
            shift = np.zeros(NB + 1)
            rate_t = np.zeros(NB + 1)
            say(f" k={k}")
            for lo, hi in BUCKETS:
                br, bs = ((er >= lo) & (er <= hi)).astype(float), ((es >= lo) & (es <= hi)).astype(float)
                nr, ns_ = Wr @ br, Ws @ bs
                hr_, hs_ = Wr @ (br * (amr == k)), Ws @ (bs * (ams == k))
                with np.errstate(invalid="ignore", divide="ignore"):
                    pr, ps = np.nan_to_num(hr_ / nr), np.nan_to_num(hs_ / ns_)
                shr_, shs_ = nr / Gr, ns_ / Gs
                shift += (shs_ - shr_) * (pr + ps) / 2
                rate_t += (shr_ + shs_) / 2 * (ps - pr)
                say(stat_line(f"  [{lo},{hi}] share", shr_, shs_))
                say(stat_line(f"  [{lo},{hi}] P(final={k} | bucket)", pr, ps))
            say(f"  decomposition of mass gap sim-real: shift in entering mix {fmtd(shift)} P(>0) {np.mean(shift[1:] > 0):.2f}; conditional rate {fmtd(rate_t)} P(>0) {np.mean(rate_t[1:] > 0):.2f}")
    say("")
    say("3. overtime: regulation-end margin from scores made in quarters 1-4")
    rr_, rs_ = [], []
    for ev, G in ((evr, Gr), (evs, Gs)):
        m = ev.q.to_numpy() <= 4
        rr_.append(np.abs(np.bincount(ev.gi.to_numpy()[m], weights=ev.spts.to_numpy()[m], minlength=G)))
    ar, as_ = rr_
    tie_r, tie_s = (ar == 0).astype(float), (as_ == 0).astype(float)
    r, s = (Wr @ tie_r) / Gr, (Ws @ tie_s) / Gs
    say(stat_line("P(tied at end of regulation)", r, s))
    for k in KEYS:
        r, s = (Wr @ (tie_r * (amr == k))) / (Wr @ tie_r), (Ws @ (tie_s * (ams == k))) / (Ws @ tie_s)
        say(stat_line(f"P(final={k} | tied after regulation)", r, s))
    nr_, ns_ = (1 - tie_r), (1 - tie_s)
    for k in KEYS:
        r, s = (Wr @ (nr_ * (amr == k))) / Gr, (Ws @ (ns_ * (ams == k))) / Gs
        say(stat_line(f"mass {k} among games decided in regulation (per all games)", r, s))
        r, s = (Wr @ (tie_r * (amr == k))) / Gr, (Ws @ (tie_s * (ams == k))) / Gs
        say(stat_line(f"mass {k} from overtime games (per all games)", r, s))
    say("")
    say(f"looks {LOOKS[0]} (every stat line printed, one family, all cells shown not the best)")
    (E3 / "e96" / "e96.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()

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
SUDDEN_SIM_S = 2
REAL_SEASONS = tuple(range(2011, 2018))
SUDDEN_REAL_LAST = 2011
NB = 300
OUT = []
LOOKS = [0]
rng = np.random.default_rng(97)
RES = ["TD", "FG", "FGmiss", "punt", "turnover", "downs", "other"]
REAL_MAP = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "FGmiss", "Punt": "punt", "Turnover": "turnover", "Turnover on downs": "downs"}
YL_BINS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 65), (66, 99)]
COLS = ["game_id", "season_type", "qtr", "play_id", "posteam", "down", "ydstogo", "yardline_100", "play_type", "fixed_drive", "fixed_drive_result", "score_differential", "game_seconds_remaining"]


def say(s=""):
    OUT.append(s)
    print(s)


def real_ot():
    rows = []
    for s in REAL_SEASONS:
        p = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=COLS)
        p = p[(p.season_type == "REG") & (p.qtr == 5)].copy()
        p["season"] = s
        rows.append(p)
    p = pd.concat(rows, ignore_index=True).sort_values(["game_id", "play_id"], kind="stable")
    p = p[p.play_type.isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"]) & p.down.notna()].copy()
    p["dk"] = p.game_id + "_" + p.fixed_drive.astype(int).astype(str)
    p["didx"] = p.groupby("game_id").fixed_drive.rank(method="dense").astype(int) - 1
    last = p.groupby("dk").tail(1).set_index("dk")
    first = p.groupby("dk").head(1)
    npl = p[p.play_type.isin(["run", "pass"])].groupby("dk").size()
    d = pd.DataFrame({"gk": first.game_id.to_numpy(), "idx": first.didx.to_numpy(), "sd": first.score_differential.fillna(0).to_numpy(), "yl": first.yardline_100.to_numpy(), "season": first.season.to_numpy()}, index=first.dk.to_numpy())
    d["res"] = last.fixed_drive_result.map(REAL_MAP).fillna("other").reindex(d.index).to_numpy()
    d["plays"] = npl.reindex(d.index).fillna(0).to_numpy()
    d["mod"] = d.season > SUDDEN_REAL_LAST
    p["isfg"] = p.play_type == "field_goal"
    p["made"] = p.isfg & (p.dk.map(d.res) == "FG") & (p.groupby("dk").cumcount(ascending=False) == 0)
    p["yl"] = p.yardline_100
    p["sd"] = p.score_differential.fillna(0)
    p["idx"] = p.didx
    p["mod"] = p.season > SUDDEN_REAL_LAST
    p["gk"] = p.game_id
    p["ptype"] = np.where(p.isfg, "fg", np.where(p.play_type == "punt", "punt", "go"))
    return d.reset_index(drop=True), p.reset_index(drop=True)


def sim_ot():
    pls = []
    for sd in SEEDS:
        sdir = E3 / f"e5_{LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "down", "yl", "sd", "qtr", "code", "po", "pdf", "flip", "idx"]).rename(columns={"idx": "pool"})
            d = d[(d.qtr == 5) & d.code.isin([0, 1, 2, 3, 4, 5])]
            if not len(d):
                continue
            d = d.sort_values("g", kind="stable").reset_index(drop=True)
            d["gk"] = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            d["s"] = s
            pls.append(d)
    p = pd.concat(pls, ignore_index=True)
    gk = p.gk.to_numpy()
    end = ((p.flip > 0) | (p.po > 0) | (p.pdf > 0)).to_numpy()
    newg = np.r_[True, gk[1:] != gk[:-1]]
    start = newg | np.r_[True, end[:-1]]
    p["dn"] = np.cumsum(start)
    p["idx"] = p.groupby("gk").dn.rank(method="dense").astype(int) - 1
    p["dk"] = p.dn
    p["isfg"] = p.code == 3
    p["made"] = p.isfg & (p.po == 3)
    p["ptype"] = np.where(p.isfg, "fg", np.where(p.code == 2, "punt", "go"))
    p["mod"] = p.s > SUDDEN_SIM_S
    first = p.groupby("dk").head(1)
    lr = p.groupby("dk").tail(1).set_index("dk")
    isdef = (lr.pdf >= 6) | (lr.po == 2) | (lr.pdf == 2)
    res = np.select(
        [lr.po >= 6, (lr.po == 3) & (lr.code == 3), (lr.po == 0) & (lr.code == 3) & (lr.flip > 0), (lr.code == 2) & (lr.flip > 0), isdef, (lr.flip > 0) & (lr.down == 4) & lr.code.isin([0, 1]), lr.flip > 0],
        ["TD", "FG", "FGmiss", "punt", "other", "downs", "turnover"],
        "other",
    )
    npl = p[p.code.isin([0, 1])].groupby("dk").size()
    d = pd.DataFrame({"gk": first.gk.to_numpy(), "idx": first.idx.to_numpy(), "sd": first.sd.to_numpy(), "yl": first.yl.to_numpy(), "mod": first["mod"].to_numpy()}, index=first.dk.to_numpy())
    d["res"] = pd.Series(res, index=lr.index).reindex(d.index).to_numpy()
    d["plays"] = npl.reindex(d.index).fillna(0).to_numpy()
    return d.reset_index(drop=True), p.reset_index(drop=True)


def state_of(sd):
    return np.where(sd > 0, "lead", np.where(sd < 0, "trail", "tied"))


def idxb(i):
    return np.minimum(i, 2)


def pergame(gi, G, m):
    return np.bincount(gi[m], minlength=G).astype(float)


class Side:
    def __init__(self, d, p, mod_only):
        if mod_only:
            d = d[d["mod"]].reset_index(drop=True)
            p = p[p["mod"]].reset_index(drop=True)
        self.d, self.p = d, p
        self.games = pd.Index(sorted(set(d.gk)))
        self.G = len(self.games)
        self.gi = self.games.get_indexer(d.gk)
        self.pgi = self.games.get_indexer(p.gk)
        self.W = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=NB).astype(float)
        self.state = state_of(d.sd.to_numpy())
        self.ib = idxb(d.idx.to_numpy())
        self.res = d.res.to_numpy()
        self.yl = p.yl.to_numpy()
        self.dn = p.down.to_numpy()
        self.fg = p.isfg.to_numpy()
        self.made = p.made.to_numpy()
        self.pib = idxb(p.idx.to_numpy())
        self.psd = state_of(p.sd.to_numpy())
        self.ptype = p.ptype.to_numpy()

    def rate(self, num, den):
        n = pergame(self.gi, self.G, num)
        k = pergame(self.gi, self.G, den)
        return (self.W @ n) / np.maximum(self.W @ k, 1e-9), n.sum() / max(k.sum(), 1), k.sum()

    def prate(self, num, den):
        n = pergame(self.pgi, self.G, num)
        k = pergame(self.pgi, self.G, den)
        return (self.W @ n) / np.maximum(self.W @ k, 1e-9), n.sum() / max(k.sum(), 1), k.sum()

    def mean(self, v, m):
        n = pergame(self.gi, self.G, m)
        t = np.bincount(self.gi[m], weights=v[m], minlength=self.G)
        return (self.W @ t) / np.maximum(self.W @ n, 1e-9), t.sum() / max(n.sum(), 1), n.sum()


def cmp_line(label, a, b):
    ra, pa, na = a
    rb, pb, nb = b
    diff = rb - ra
    LOOKS[0] += 1
    return f"{label:46s} real {pa:.3f} [{np.percentile(ra, 2.5):.3f},{np.percentile(ra, 97.5):.3f}] n{int(na):5d} sim {pb:.3f} [{np.percentile(rb, 2.5):.3f},{np.percentile(rb, 97.5):.3f}] n{int(nb):6d} s-r {pb - pa:+.3f} [{np.percentile(diff, 2.5):+.3f},{np.percentile(diff, 97.5):+.3f}] P(s>r) {np.mean(diff > 0):.2f}"


def ending_table(R, S):
    say("")
    say("Game endings in OT games by the last drive of the game (type and OT drive index); per OT game")

    def ends(X):
        last = X.d.groupby("gk").tail(1)
        return last, X.games.get_indexer(last.gk)

    lr, gir = ends(R)
    ls, gis = ends(S)
    for kind in ("TD", "FG", "other"):
        for ixb in (0, 1, 2):
            def sel(L, X, gi):
                is_kind = L.res.to_numpy() == kind if kind != "other" else ~L.res.isin(["TD", "FG"]).to_numpy()
                m = is_kind & (idxb(L.idx.to_numpy()) == ixb)
                n = pergame(gi, X.G, m)
                return (X.W @ n) / X.G, n.sum() / X.G, X.G

            say(cmp_line(f"P(game ends {kind} on OT drive {'2+' if ixb == 2 else ixb})", sel(lr, R, gir), sel(ls, S, gis)))


def drive_cells(R, S):
    say("")
    say("Drive outcome by OT drive index (0,1,2+) and score state at drive start (offense view)")
    for ib in (0, 1, 2):
        for st in ("tied", "trail", "lead"):
            mr = (R.ib == ib) & (R.state == st)
            ms = (S.ib == ib) & (S.state == st)
            if mr.sum() + ms.sum() == 0:
                continue
            say(f"-- idx {'2+' if ib == 2 else ib} state {st}: drives real {int(mr.sum())} sim {int(ms.sum())}")
            for r in RES:
                a = R.rate((R.res == r) & mr, mr)
                b = S.rate((S.res == r) & ms, ms)
                if a[1] == 0 and b[1] == 0:
                    continue
                say("   " + cmp_line(f"P({r})", a, b))
            for nm, col in (("start yardline", "yl"), ("plays run+pass", "plays")):
                say("   " + cmp_line(f"mean {nm}", R.mean(R.d[col].to_numpy(), mr), S.mean(S.d[col].to_numpy(), ms)))


def fg_fourth(R, S):
    say("")
    say("FG attempt share of OT scrimmage plays at downs 1-3 inside yl<=40, by drive index")
    for dn in (1, 2, 3):
        for ib in (0, 1, 2):
            dr = (R.dn == dn) & (R.yl <= 40) & (R.pib == ib)
            ds = (S.dn == dn) & (S.yl <= 40) & (S.pib == ib)
            if dr.sum() + ds.sum() == 0:
                continue
            say(cmp_line(f"P(FG | down {dn}, yl<=40, idx {'2+' if ib == 2 else ib})", R.prate(R.fg & dr, dr), S.prate(S.fg & ds, ds)))
    say("")
    say("4th down choice by yardline bin (all OT, down 4), FG made given attempt")
    for lo, hi in YL_BINS:
        dr = (R.dn == 4) & (R.yl >= lo) & (R.yl <= hi)
        ds = (S.dn == 4) & (S.yl >= lo) & (S.yl <= hi)
        say(f"-- yl {lo}-{hi}: n real {int(dr.sum())} sim {int(ds.sum())}")
        for ch in ("go", "fg", "punt"):
            say("   " + cmp_line(f"P({ch})", R.prate((R.ptype == ch) & dr, dr), S.prate((S.ptype == ch) & ds, ds)))
        ar, as_ = dr & R.fg, ds & S.fg
        if ar.sum() + as_.sum():
            say("   " + cmp_line("P(made | FG attempt)", R.prate(R.made & ar, ar), S.prate(S.made & as_, as_)))
    say("")
    say("4th down in OT by drive index and state, yl<=40")
    for ib in (0, 1, 2):
        for st in ("tied", "trail", "lead"):
            dr = (R.dn == 4) & (R.yl <= 40) & (R.pib == ib) & (R.psd == st)
            ds = (S.dn == 4) & (S.yl <= 40) & (S.pib == ib) & (S.psd == st)
            if dr.sum() + ds.sum() < 4:
                continue
            say(f"-- idx {'2+' if ib == 2 else ib} state {st}: n real {int(dr.sum())} sim {int(ds.sum())}")
            for ch in ("fg", "go"):
                say("   " + cmp_line(f"P({ch})", R.prate((R.ptype == ch) & dr, dr), S.prate((S.ptype == ch) & ds, ds)))


def pool_trace(ps):
    import sim04_engine as sim

    pbp = sim.load_reg_seasons(tuple(range(2009, 2020)))
    tr = sim.build_transition_frame(pbp).reset_index(drop=True)
    q = ps[ps["mod"]].reset_index(drop=True)
    row = tr.loc[q.pool.astype(int).to_numpy()].reset_index(drop=True)
    ok = (row.down_i.to_numpy() == np.where(q.down.to_numpy() > 3, 4, q.down.to_numpy())).mean()
    say("")
    say("=" * 100)
    say(f"POOL TRACE: sim OT plays (modified-rule years) {len(q)}; pool rows {len(tr)} (2009-2019 REG), OT pool rows {int((tr.phase == 4).sum())}, phase-3 (Q4 last 5 min) pool rows {int((tr.phase == 3).sum())}; down match check {ok:.3f}")
    ph = row.phase.to_numpy()
    say(f"drawn pool row phase: OT {np.mean(ph == 4):.3f}, Q4 last 5 min {np.mean(ph == 3):.3f}, other {np.mean(~np.isin(ph, (3, 4))):.3f}")
    code = row.play_type_code.to_numpy()
    for dn in (1, 2, 3):
        m = (q.down.to_numpy() == dn) & (q.yl.to_numpy() <= 40)
        for nm, sel in (("OT-row draws", ph == 4), ("Q4-row draws", ph == 3)):
            mm = m & sel
            say(f"down {dn} yl<=40 {nm}: n {int(mm.sum())} FG share {np.mean(code[mm] == 3) if mm.sum() else float('nan'):.3f}")
    ot = tr[tr.phase == 4]
    gy = ot.game_id.str[:4].astype(int).to_numpy()
    for lab, sel in (("pool OT rows 2009-2011 sudden death", gy <= 2011), ("pool OT rows 2012-2019 modified", gy >= 2012)):
        o = ot[sel]
        say(f"{lab}: n {len(o)}")
        for dn in (1, 2, 3):
            m = (o.down_i.to_numpy() == dn) & (o.fp_raw.to_numpy() <= 40) & np.isin(o.play_type_code.to_numpy(), (0, 1, 2, 3))
            say(f"   down {dn} fp<=40 n {int(m.sum())} FG share {np.mean(o.play_type_code.to_numpy()[m] == 3):.3f}")
    t3 = tr[(tr.phase == 3) & (tr.sc_raw == 0)]
    for dn in (1, 2, 3):
        m = (t3.down_i.to_numpy() == dn) & (t3.fp_raw.to_numpy() <= 40) & np.isin(t3.play_type_code.to_numpy(), (0, 1, 2, 3))
        say(f"pool Q4-last-5-min tied rows down {dn} fp<=40 n {int(m.sum())} FG share {np.mean(t3.play_type_code.to_numpy()[m] == 3):.3f}")


def fit_hazard(pr):
    from sklearn.linear_model import LogisticRegression

    d = pr[pr["mod"] & (pr.idx >= 1) & pr.down.isin([1, 2, 3]) & (pr.yl <= 45)].copy()
    d["season"] = d.game_id.str[:4].astype(int)
    d["y"] = d.isfg.astype(int)
    X = np.column_stack([d.yl.to_numpy(), (d.down == 2).to_numpy(), (d.down == 3).to_numpy()]).astype(float)
    y = d.y.to_numpy()
    say("")
    say("=" * 100)
    say(f"SUDDEN-DEATH FG HAZARD FIT: real OT drives idx>=1, modified rules 2012-17, downs 1-3, yl<=45: plays {len(d)} FG attempts {int(y.sum())}; one family of 3 models; leave-one-season-out")
    ll = {"logit": [], "const": [], "down": []}
    tot = {k: 0.0 for k in ll}
    br = {k: 0.0 for k in ll}
    for s in sorted(d.season.unique()):
        te = d.season.to_numpy() == s
        trn = ~te
        m1 = LogisticRegression(C=1e6, max_iter=1000).fit(X[trn], y[trn])
        p1 = m1.predict_proba(X[te])[:, 1]
        p0 = np.full(te.sum(), y[trn].mean())
        pd_ = np.zeros(te.sum())
        for k in (1, 2, 3):
            mk_tr = trn & (d.down.to_numpy() == k)
            mk_te = (d.down.to_numpy()[te] == k)
            pd_[mk_te] = y[mk_tr].mean() if mk_tr.sum() else y[trn].mean()
        for nm, pp in (("logit", p1), ("const", p0), ("down", pd_)):
            pp = np.clip(pp, 1e-6, 1 - 1e-6)
            tot[nm] += -np.sum(y[te] * np.log(pp) + (1 - y[te]) * np.log(1 - pp))
            br[nm] += np.sum((pp - y[te]) ** 2)
        say(f"  held-out {s}: n {int(te.sum())} FG {int(y[te].sum())} coef yl {m1.coef_[0][0]:+.4f} d2 {m1.coef_[0][1]:+.3f} d3 {m1.coef_[0][2]:+.3f} icpt {m1.intercept_[0]:+.3f}")
    for nm in tot:
        say(f"  LOSO log loss {nm}: {tot[nm] / len(d):.4f} brier {br[nm] / len(d):.4f}")
    full = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    pin = full.predict_proba(X)[:, 1]
    say(f"  in-sample log loss logit {-np.mean(y * np.log(pin) + (1 - y) * np.log(1 - pin)):.4f}; full-fit coef yl {full.coef_[0][0]:+.4f} d2 {full.coef_[0][1]:+.3f} d3 {full.coef_[0][2]:+.3f} icpt {full.intercept_[0]:+.3f}")
    for dn in (1, 2, 3):
        for yl in (10, 20, 30, 40):
            say(f"    fitted P(FG | down {dn}, yl {yl}) = {full.predict_proba(np.array([[yl, dn == 2, dn == 3]], float))[0, 1]:.3f}")


def run(dr, pr, ds, ps, mod_only):
    R, S = Side(dr, pr, mod_only), Side(ds, ps, mod_only)
    say("")
    say("=" * 100)
    label = "modified sudden death only (real 2012-17, sim years 2012-16)" if mod_only else "all (real 2011-17, sim years 2011-16)"
    say(f"RULE SET {label}; OT games real {R.G} sim {S.G}; drives real {len(R.d)} sim {len(S.d)}")
    drive_cells(R, S)
    ending_table(R, S)
    fg_fourth(R, S)


def main():
    (E3 / "e97").mkdir(parents=True, exist_ok=True)
    say(f"E97 OT drive table; sim {LABEL} seeds {SEEDS} burn {BURN} (OTY years 2009+s); real REG {REAL_SEASONS[0]}-{REAL_SEASONS[-1]}; game bootstrap {NB}")
    dr, pr = real_ot()
    ds, ps = sim_ot()
    run(dr, pr, ds, ps, True)
    run(dr, pr, ds, ps, False)
    pool_trace(ps)
    fit_hazard(pr)
    say("")
    say(f"looks counted {LOOKS[0]} (one family: every printed cell); P(s>r) is probability_positive of sim minus real")
    (E3 / "e97" / "e97.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25_generator as gen  # noqa: E402

E3 = REPO / "artifacts" / "mod25e3"
OUTD = E3 / "e99"
LABEL = "crHpqokgndecsmfwtjo2as2ypw2"
SEEDS = (11, 12, 13)
BURN = 2
REAL_SEASONS = tuple(range(2011, 2018))
NB = 300
WIN = 300
BANDS = ((0, 30), (30, 120), (120, 300))
RANGE_YL = 40
OUT = []
LOOKS = [0]
rng = np.random.default_rng(99)
RES = ["TD", "FG", "FGmiss", "punt", "turnover", "downs", "other"]
REAL_MAP = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "FGmiss", "Punt": "punt", "Turnover": "turnover", "Turnover on downs": "downs"}
COLS = ["game_id", "season_type", "qtr", "play_id", "posteam", "down", "ydstogo", "yardline_100", "play_type", "fixed_drive", "fixed_drive_result", "score_differential", "game_seconds_remaining"]


def say(s=""):
    OUT.append(s)
    print(s)


def real_tables():
    pl = []
    for s in REAL_SEASONS:
        p = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=COLS)
        p = p[(p.season_type == "REG") & (p.qtr <= 4)].copy()
        pl.append(p)
    p = pd.concat(pl, ignore_index=True).sort_values(["game_id", "play_id"], kind="stable")
    p = p[p.play_type.isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"]) & p.down.notna() & p.game_seconds_remaining.notna()].copy()
    p["dk"] = p.game_id + "_" + p.fixed_drive.astype(int).astype(str)
    p["res"] = p.groupby("dk").fixed_drive_result.transform("last").map(REAL_MAP).fillna("other")
    out = pd.DataFrame({"gk": p.game_id.to_numpy(), "dk": p.dk.to_numpy(), "t": p.game_seconds_remaining.to_numpy(float), "q": p.qtr.to_numpy(), "down": p.down.to_numpy(float), "yl": p.yardline_100.to_numpy(float), "sd": p.score_differential.fillna(0).to_numpy(float), "ptype": p.play_type.map({"run": "go", "pass": "go", "punt": "punt", "field_goal": "fg", "qb_kneel": "kneel", "qb_spike": "spike"}).to_numpy(), "res": p.res.to_numpy()})
    gf = gen.real_games(REAL_SEASONS)
    games = pd.DataFrame({"gk": gf.game_id.to_numpy(), "margin": (gf.home_score - gf.away_score).to_numpy(float)})
    games = games[games.gk.isin(set(out.gk))].reset_index(drop=True)
    return out, games


def sim_tables():
    pls, games = [], []
    for sd in SEEDS:
        sdir = E3 / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_score", "away_score"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        sg = sg[sg.s >= BURN]
        games.append(pd.DataFrame({"gk": f"{sd}_" + sg.w.astype(str) + "_" + sg.s.astype(str) + "_" + sg.g.astype(str), "margin": (sg.home_score - sg.away_score).to_numpy(float)}))
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "down", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip"])
            d = d[d.code.isin([0, 1, 2, 3, 4, 5])].sort_values("g", kind="stable").reset_index(drop=True)
            d["gk"] = f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str)
            end = ((d.flip > 0) | (d.po > 0) | (d.pdf > 0)).to_numpy()
            gk = d.gk.to_numpy()
            start = np.r_[True, gk[1:] != gk[:-1]] | np.r_[True, end[:-1]]
            d["dk"] = f"{sd}_{w}_{s}_" + pd.Series(np.cumsum(start), index=d.index).astype(str)
            lr = d.groupby("dk").tail(1).set_index("dk")
            isdef = (lr.pdf >= 6) | (lr.po == 2) | (lr.pdf == 2)
            res = np.select([lr.po >= 6, (lr.po == 3) & (lr.code == 3), (lr.po == 0) & (lr.code == 3) & (lr.flip > 0), (lr.code == 2) & (lr.flip > 0), isdef, (lr.flip > 0) & (lr.down == 4) & lr.code.isin([0, 1]), lr.flip > 0], ["TD", "FG", "FGmiss", "punt", "other", "downs", "turnover"], "other")
            q4 = d.qtr.to_numpy(); lastg = pd.Series(np.r_[(gk[1:] != gk[:-1]) | (q4[1:] > 4), True] & (q4 <= 4), index=d.index)
            gend = lastg[d.groupby("dk").tail(1).index].to_numpy()
            res = np.where(gend & (res == "turnover"), "other", res)
            d["res"] = pd.Series(res, index=lr.index).reindex(d.dk).to_numpy()
            d = d[d.qtr <= 4]
            pls.append(pd.DataFrame({"gk": d.gk.to_numpy(), "dk": d.dk.to_numpy(), "t": d.gsr.to_numpy(float), "q": d.qtr.to_numpy(), "down": d.down.to_numpy(float), "yl": d.yl.to_numpy(float), "sd": d.sd.to_numpy(float), "ptype": pd.Series(d.code.to_numpy()).map({0.0: "go", 1.0: "go", 2.0: "punt", 3.0: "fg", 4.0: "kneel", 5.0: "spike"}).to_numpy(), "res": d.res.to_numpy()}))
    return pd.concat(pls, ignore_index=True), pd.concat(games, ignore_index=True)


class Side:
    def __init__(self, p, games):
        self.games = pd.Index(games.gk)
        self.margin = games.margin.to_numpy()
        self.G = len(self.games)
        self.W = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=NB).astype(float)
        p = p[p.gk.isin(self.games)].reset_index(drop=True)
        p["gi"] = self.games.get_indexer(p.gk)
        w = p[(p.q == 4) & (p.t <= WIN)].reset_index(drop=True)
        first = w.groupby("gk").head(1)
        self.entry = pd.Series(np.nan, index=np.arange(self.G))
        self.entry.loc[first.gi.to_numpy()] = first.sd.to_numpy()
        w["prev"] = w.groupby("dk").ptype.shift(1)
        self.w = w
        drv = w.groupby("dk")
        d = drv.head(1)[["gi", "dk", "t", "down", "yl", "sd"]].reset_index(drop=True)
        d["res"] = drv.res.last().reindex(d.dk).to_numpy()
        d["tend"] = drv.t.last().reindex(d.dk).to_numpy()
        d["minyl"] = drv.yl.min().reindex(d.dk).to_numpy()
        d["fgatt"] = drv.ptype.apply(lambda s: (s == "fg").any()).reindex(d.dk).to_numpy()
        d["kn"] = drv.ptype.apply(lambda s: s.isin(["kneel", "spike"]).any()).reindex(d.dk).to_numpy()
        self.d = d

    def rate(self, gi, num, den):
        n = np.bincount(gi[num], minlength=self.G).astype(float)
        k = np.bincount(gi[den], minlength=self.G).astype(float)
        return (self.W @ n) / np.maximum(self.W @ k, 1e-9), n.sum() / max(k.sum(), 1), k.sum()

    def mean(self, gi, v, m):
        n = np.bincount(gi[m], minlength=self.G).astype(float)
        t = np.bincount(gi[m], weights=v[m], minlength=self.G)
        return (self.W @ t) / np.maximum(self.W @ n, 1e-9), t.sum() / max(n.sum(), 1), n.sum()


def line(label, a, b):
    ra, pa, na = a
    rb, pb, nb = b
    df = rb - ra
    LOOKS[0] += 1
    return f"{label:44s} real {pa:.3f} [{np.percentile(ra, 2.5):.3f},{np.percentile(ra, 97.5):.3f}] n{int(na):5d} sim {pb:.3f} [{np.percentile(rb, 2.5):.3f},{np.percentile(rb, 97.5):.3f}] n{int(nb):6d} s-r {pb - pa:+.3f} [{np.percentile(df, 2.5):+.3f},{np.percentile(df, 97.5):+.3f}] P(s>r) {np.mean(df > 0):.2f}"


def game_level(R, S):
    say("A. Games by score state entering the last 5 min of Q4 (first play with gsr<=300), final margin incl overtime")
    gr, gs = np.arange(R.G), np.arange(S.G)
    for nm, f in (("tied", lambda e: e == 0), ("|sd|1-2", lambda e: (e != 0) & (np.abs(e) <= 2)), ("|sd|3-8", lambda e: (np.abs(e) >= 3) & (np.abs(e) <= 8))):
        mr = f(R.entry.to_numpy())
        ms = f(S.entry.to_numpy())
        say(f"-- state {nm}: games real {int(mr.sum())} sim {int(ms.sum())}")
        for k, fn in (("P(|m|=3)", lambda m: np.abs(m) == 3), ("P(|m|=0 tie)", lambda m: m == 0), ("P(|m|=1-2)", lambda m: (np.abs(m) >= 1) & (np.abs(m) <= 2)), ("P(|m|=6-7)", lambda m: (np.abs(m) >= 6) & (np.abs(m) <= 7)), ("P(|m|>=8)", lambda m: np.abs(m) >= 8)):
            a = R.rate(gr, mr & fn(R.margin), mr)
            b = S.rate(gs, ms & fn(S.margin), ms)
            say("   " + line(k, a, b))


def drive_level(R, S):
    rg, sg = R.d.gi.to_numpy(), S.d.gi.to_numpy()
    for nm, lo, hi in (("tied", 0, 0), ("trail 1-2", -2, -1), ("lead 1-2", 1, 2)):
        for tmax in (WIN, 120):
            say("")
            say(f"B. Drives starting gsr<={tmax}, offense {nm} (drive result is the whole drive)")
            mr, ms = [(X.d.sd.to_numpy() >= lo) & (X.d.sd.to_numpy() <= hi) & (X.d.t.to_numpy() <= tmax) for X in (R, S)]
            say(f"-- drives real {int(mr.sum())} sim {int(ms.sum())}")
            for r in RES:
                a = R.rate(rg, mr & (R.d.res.to_numpy() == r), mr)
                b = S.rate(sg, ms & (S.d.res.to_numpy() == r), ms)
                say("   " + line(f"P({r})", a, b))
            a = R.mean(rg, R.d.tend.to_numpy(), mr & (R.d.res.to_numpy() == "FG"))
            b = S.mean(sg, S.d.tend.to_numpy(), ms & (S.d.res.to_numpy() == "FG"))
            say("   " + line("mean gsr of FG-making play", a, b))
            for k, col in (("P(FG attempt on drive)", "fgatt"), ("P(kneel/spike on drive)", "kn")):
                a = R.rate(rg, mr & R.d[col].to_numpy(bool), mr)
                b = S.rate(sg, ms & S.d[col].to_numpy(bool), ms)
                say("   " + line(k, a, b))
            rr = mr & (R.d.minyl.to_numpy() <= RANGE_YL)
            rs = ms & (S.d.minyl.to_numpy() <= RANGE_YL)
            say(f"   -- drives reaching yl<={RANGE_YL}: real {int(rr.sum())} sim {int(rs.sum())}")
            for r in RES:
                a = R.rate(rg, rr & (R.d.res.to_numpy() == r), rr)
                b = S.rate(sg, rs & (S.d.res.to_numpy() == r), rs)
                say("      " + line(f"P({r}|reach)", a, b))


def play_level(R, S):
    say("")
    say(f"C. Play choice inside yl<={RANGE_YL}, Q4 gsr<=300, offense tied or trailing 1-2 (a FG ties or wins); P(play type) by down and gsr band")
    for X in (R, S):
        w = X.w
        X.pm = (w.yl.to_numpy() <= RANGE_YL) & (w.sd.to_numpy() <= 0) & (w.sd.to_numpy() >= -2) & (w.down.to_numpy() <= 4)
        X.wg = w.gi.to_numpy()
        X.wt = w.t.to_numpy()
        X.wp = w.ptype.to_numpy()
    for lo, hi in BANDS:
        for dn in (1, 2, 3, 4):
            mr = R.pm & (R.w.down.to_numpy() == dn) & (R.wt >= lo) & (R.wt < hi)
            ms = S.pm & (S.w.down.to_numpy() == dn) & (S.wt >= lo) & (S.wt < hi)
            kinds = ("fg", "go", "punt", "kneel", "spike") if dn == 4 else ("fg", "kneel", "spike")
            for k in kinds:
                a = R.rate(R.wg, mr & (R.wp == k), mr)
                b = S.rate(S.wg, ms & (S.wp == k), ms)
                if a[1] == 0 and b[1] == 0:
                    continue
                say("   " + line(f"gsr[{lo},{hi}) down {dn} P({k})", a, b))
    say("")
    say("D. FG attempts in that state: seconds left at the kick")
    for X in (R, S):
        X.fgm = X.pm & (X.wp == "fg")
        X.pk = X.w.prev.isin(["kneel", "spike"]).to_numpy()
    for thr in (5, 10, 30, 60):
        a = R.rate(R.wg, R.fgm & (R.wt <= thr), R.fgm)
        b = S.rate(S.wg, S.fgm & (S.wt <= thr), S.fgm)
        say("   " + line(f"P(kick with gsr<={thr})", a, b))
    for dn in (1, 2, 3, 4):
        a = R.rate(R.wg, R.fgm & (R.w.down.to_numpy() == dn), R.fgm)
        b = S.rate(S.wg, S.fgm & (S.w.down.to_numpy() == dn), S.fgm)
        say("   " + line(f"P(FG kick on down {dn})", a, b))
    a = R.rate(R.wg, R.fgm & R.pk, R.fgm)
    b = S.rate(S.wg, S.fgm & S.pk, S.fgm)
    say("   " + line("P(prev play kneel/spike | FG kick)", a, b))
    for nm, X in (("real", R), ("sim", S)):
        say(f"   {nm} FG-kick gsr quantiles 10/25/50/75/90: {np.percentile(X.wt[X.fgm], [10, 25, 50, 75, 90]).round(0).tolist()}")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    say(f"E99 late-regulation FG mechanism; sim {LABEL} seeds {SEEDS} burn {BURN}; real REG {REAL_SEASONS[0]}-{REAL_SEASONS[-1]}; game bootstrap {NB}; intervals 2.5-97.5; P(s>r) is probability sim exceeds real")
    rp, rg = real_tables()
    sp, sg = sim_tables()
    R, S = Side(rp, rg), Side(sp, sg)
    say(f"games real {R.G} sim {S.G}; window plays real {len(R.w)} sim {len(S.w)}")
    game_level(R, S)
    drive_level(R, S)
    play_level(R, S)
    say("")
    say(f"looks counted: {LOOKS[0]} (one family: late-regulation FG mechanism diagnostics; none is a decision test)")
    (OUTD / "e99.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()

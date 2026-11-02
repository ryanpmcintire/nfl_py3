import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sim07"
PBP = sorted(glob.glob(str(ROOT / "data/pbp/raw/*")))[-1]
PART = ROOT / "data/players/participation/raw/20260813T131635Z"
CACHE = OUT / "plays.parquet"
SEASONS = range(2016, 2026)
ERAS = ((2016, 2020), (2021, 2025))
COVER = 0.95
BOOT = 200
BOOT_ERA = 100
ITER = 3
SEED = 20261102
BINS = 5


def counts(s):
    if not isinstance(s, str):
        return None
    return {p: int(n) for n, p in re.findall(r"(\d+) (\w+)", s)}


def off_label(s):
    c = counts(s)
    if c is None or ("RB" not in c and "TE" not in c and "WR" not in c):
        return None
    return f"{c.get('RB', 0)}{c.get('TE', 0)}"


def db_count(s):
    c = counts(s)
    if c is None:
        return np.nan
    return float(sum(v for k, v in c.items() if k in ("CB", "S", "FS", "SS", "DB")))


def build():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    cols = [
        "game_id",
        "play_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "down",
        "play_type",
        "epa",
        "qtr",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
        "qb_kneel",
        "qb_spike",
        "score_differential",
    ]
    fr = []
    for s in SEASONS:
        x = pd.read_parquet(f"{PBP}/season={s}/plays.parquet", columns=cols)
        fr.append(x[x.season_type == "REG"])
    d = pd.concat(fr, ignore_index=True)
    d = d[
        d.posteam.notna()
        & d.defteam.notna()
        & d.play_type.isin(["pass", "run"])
        & (d.qb_kneel != 1)
        & (d.qb_spike != 1)
        & (d.qtr <= 4)
        & d.down.notna()
        & d.score_differential.notna()
        & d.epa.notna()
    ].copy()
    pf = []
    for s in SEASONS:
        pf.append(
            pd.read_parquet(
                PART / f"season={s}" / "participation.parquet",
                columns=["game_id", "play_id", "offense_personnel", "defense_personnel"],
            )
        )
    pt = pd.concat(pf, ignore_index=True)
    pt["play_id"] = pt.play_id.astype(d.play_id.dtype)
    pt = pt.drop_duplicates(["game_id", "play_id"])
    d = d.merge(pt, on=["game_id", "play_id"], how="left")
    d["off"] = d.offense_personnel.map(off_label)
    d["ndb"] = d.defense_personnel.map(db_count)
    d = d.drop(
        columns=["offense_personnel", "defense_personnel", "season_type", "qb_kneel", "qb_spike"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE)
    return d


def prepare(d):
    cov = d.assign(has=d.off.notna() & d.ndb.notna()).groupby("season").has.agg(["mean", "size"])
    d = d[d.off.notna() & d.ndb.notna()].copy()
    keep = []
    cum = 0.0
    for k, v in d.off.value_counts(normalize=True).items():
        if cum >= COVER:
            break
        keep.append(k)
        cum += v
    d["og"] = np.where(d.off.isin(keep), d.off, "other")
    dbs = d.ndb.astype(int)
    share = dbs.value_counts(normalize=True)
    med = dbs.median()
    lo_keep = [k for k in share.index if share[k] >= 1 - COVER and k < med]
    hi_keep = [k for k in share.index if share[k] >= 1 - COVER and k >= med]
    dbs = dbs.clip(lower=min(lo_keep), upper=max(hi_keep))
    d["dp"] = dbs.astype(str)
    return d, cov, keep


def design(d):
    parts = []
    for c, q in (
        ("down", None),
        ("ydstogo", BINS),
        ("yardline_100", BINS),
        ("score_differential", BINS),
        ("game_seconds_remaining", BINS),
    ):
        v = d[c].astype(str) if q is None else pd.qcut(d[c], q, duplicates="drop").astype(str)
        parts.append(pd.get_dummies(v, prefix=c, drop_first=True, dtype=np.float32).values)
    og = pd.get_dummies(d.og, prefix="o", drop_first=True, dtype=np.float32).values
    dp = pd.get_dummies(d.dp, prefix="d", drop_first=True, dtype=np.float32).values
    return np.hstack([np.ones((len(d), 1), np.float32)] + parts + [og, dp]).astype(np.float32)


def tau2_moment(m, w, se2):
    tot = w.sum()
    mbar = (m * w).sum() / tot
    num = (w * (m - mbar) ** 2).sum() - ((1 - w / tot) * w * se2).sum()
    den = tot - (w**2).sum() / tot
    return max(0.0, num / den)


class Fitter:
    def __init__(self, d):
        self.X = design(d)
        self.y = d.epa.values.astype(np.float64)
        ot = (d.posteam + d.season.astype(str)).astype("category")
        dt = (d.defteam + d.season.astype(str)).astype("category")
        ck = (d.og + "|" + d.dp).astype("category")
        self.o = ot.cat.codes.values.astype(np.int64)
        self.d = dt.cat.codes.values.astype(np.int64)
        self.c = ck.cat.codes.values.astype(np.int64)
        self.n_o, self.n_d, self.n_c = (
            len(ot.cat.categories),
            len(dt.cat.categories),
            len(ck.cat.categories),
        )
        self.cells = list(ck.cat.categories)
        self.buf = np.empty_like(self.X)
        n = len(self.y)
        self.adj = np.empty(n, np.float32)
        self.fx = np.empty(n, np.float32)
        self.r = np.empty(n)
        self.t1 = np.empty(n)
        self.t2 = np.empty(n)
        self.wf = np.empty(n, np.float32)
        self.games, self.gidx = np.unique(d.game_id.values, return_inverse=True)
        self.wg = np.empty(len(self.games))
        self.to = np.zeros(self.n_o)
        self.td = np.zeros(self.n_d)

    def boot(self, rng):
        self.wg[:] = rng.poisson(1.0, len(self.games))
        return self.wg[self.gidx]

    def shrunk(self, r, w, ids, k):
        sw = np.bincount(ids, w, k)
        np.multiply(w, r, out=self.t1)
        sm = np.bincount(ids, self.t1, k)
        pos = sw > 0
        m = np.zeros(k)
        m[pos] = sm[pos] / sw[pos]
        np.take(m, ids, out=self.t1, mode="clip")
        np.subtract(r, self.t1, out=self.t1)
        np.square(self.t1, out=self.t1)
        np.multiply(self.t1, w, out=self.t1)
        sig2 = self.t1.sum() / max(w.sum() - pos.sum(), 1.0)
        t2 = tau2_moment(m[pos], sw[pos], np.full(pos.sum(), sig2) / sw[pos])
        eff = np.zeros(k)
        if t2 > 0:
            lam = sig2 / t2
            eff[pos] = sm[pos] / (sw[pos] + lam)
        return eff, t2

    def fit(self, w):
        self.to[:] = 0.0
        self.td[:] = 0.0
        self.wf[:] = w
        np.multiply(self.X, self.wf[:, None], out=self.buf)
        xtx = (self.buf.T @ self.X).astype(np.float64) + 1e-6 * np.eye(self.X.shape[1])
        for _ in range(ITER):
            np.take(self.to, self.o, out=self.t1, mode="clip")
            np.take(self.td, self.d, out=self.t2, mode="clip")
            np.add(self.t1, self.t2, out=self.t1)
            np.subtract(self.y, self.t1, out=self.t1)
            self.adj[:] = self.t1
            beta = np.linalg.solve(xtx, (self.buf.T @ self.adj).astype(np.float64))
            np.matmul(self.X, beta.astype(np.float32), out=self.fx)
            np.subtract(self.y, self.fx, out=self.r)
            np.take(self.td, self.d, out=self.t2, mode="clip")
            np.subtract(self.r, self.t2, out=self.t2)
            self.to, t2o = self.shrunk(self.t2, w, self.o, self.n_o)
            np.take(self.to, self.o, out=self.t2, mode="clip")
            np.subtract(self.r, self.t2, out=self.t2)
            self.td, t2d = self.shrunk(self.t2, w, self.d, self.n_d)
        np.take(self.to, self.o, out=self.t1, mode="clip")
        np.take(self.td, self.d, out=self.t2, mode="clip")
        np.add(self.t1, self.t2, out=self.t1)
        np.subtract(self.r, self.t1, out=self.t1)
        sw = np.bincount(self.c, w, self.n_c)
        np.multiply(w, self.t1, out=self.t2)
        sm = np.bincount(self.c, self.t2, self.n_c)
        cell = np.zeros(self.n_c)
        cell[sw > 0] = sm[sw > 0] / sw[sw > 0]
        return cell, sw, t2o, t2d


def wcorr(a, b, w):
    wn = w / w.sum()
    ma, mb = (wn * a).sum(), (wn * b).sum()
    cv = (wn * (a - ma) * (b - mb)).sum()
    va, vb = (wn * (a - ma) ** 2).sum(), (wn * (b - mb) ** 2).sum()
    return float(cv / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan, float(cv)


def split_half(d, rng, boots):
    fo = Fitter(d[d.week % 2 == 1].reset_index(drop=True))
    fe = Fitter(d[d.week % 2 == 0].reset_index(drop=True))
    cells = sorted(set(fo.cells) | set(fe.cells))
    io = np.array([cells.index(c) for c in fo.cells])
    ie = np.array([cells.index(c) for c in fe.cells])
    nc = len(cells)
    est = np.zeros((2, nc))
    n = np.zeros((2, nc))
    c1, s1, *_ = fo.fit(np.ones(len(fo.y)))
    c2, s2, *_ = fe.fit(np.ones(len(fe.y)))
    est[0, io], n[0, io] = c1, s1
    est[1, ie], n[1, ie] = c2, s2
    ok = (n[0] > 0) & (n[1] > 0)
    bs = np.zeros((2, boots, nc))
    for b in range(boots):
        c1, *_ = fo.fit(fo.boot(rng))
        c2, *_ = fe.fit(fe.boot(rng))
        bs[0, b, io] = c1
        bs[1, b, ie] = c2
    se = bs.std(axis=1, ddof=1)
    w = np.where(ok, 1.0 / (se[0] ** 2 + se[1] ** 2 + 1e-300), 0.0)
    r0 = wcorr(est[0][ok], est[1][ok], w[ok])
    rb = np.array([wcorr(bs[0, b][ok], bs[1, b][ok], w[ok]) for b in range(boots)])
    return cells, est, n, r0, rb, bs


def full_fit(d, rng, boots):
    f = Fitter(d)
    cell, sw, t2o, t2d = f.fit(np.ones(len(f.y)))
    bs = np.zeros((boots, f.n_c))
    for b in range(boots):
        bs[b] = f.fit(f.boot(rng))[0]
    se = bs.std(axis=0, ddof=1)
    m = sw > 0
    tau2 = tau2_moment(cell[m], sw[m], se[m] ** 2)
    shr = np.where(se > 0, cell * tau2 / (tau2 + se**2), 0.0)
    q = (1 - COVER) / 2
    tab = pd.DataFrame(
        dict(
            cell=f.cells,
            n=sw,
            est=cell,
            se=se,
            lo=np.quantile(bs, q, axis=0),
            hi=np.quantile(bs, 1 - q, axis=0),
            p_pos=(bs > 0).mean(axis=0),
            shrunk=shr,
        )
    )
    mean_se2 = float((se[m] ** 2 * sw[m]).sum() / sw[m].sum())
    return tab, dict(
        tau2_cell=tau2,
        tau_cell=float(np.sqrt(tau2)),
        mean_se2=mean_se2,
        reliability_cell=float(tau2 / (tau2 + mean_se2)),
        tau2_off_team_season=float(t2o),
        tau2_def_team_season=float(t2d),
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    d, cov, keep = prepare(build())
    cov.to_csv(OUT / "coverage.csv")
    print(cov.to_string())
    print("off groups", keep)
    print(d.og.value_counts().to_string())
    print(d.dp.value_counts().to_string(), flush=True)
    if mode == "dist":
        return
    boots = int(sys.argv[2]) if len(sys.argv) > 2 else BOOT
    boots_era = int(sys.argv[3]) if len(sys.argv) > 3 else BOOT_ERA
    rng = np.random.default_rng(SEED)
    sets = {
        "all": d,
        "pass": d[d.play_type == "pass"].reset_index(drop=True),
        "run": d[d.play_type == "run"].reset_index(drop=True),
    }
    summ = {}
    for name, dd in sets.items():
        cells, est, n, r0, rb, bs = split_half(dd, rng, boots)
        rr = rb[np.isfinite(rb[:, 0]), 0]
        cc = rb[:, 1]
        s = dict(
            plays=len(dd),
            cells=len(cells),
            r_split_w=r0[0],
            r_lo=float(np.quantile(rr, 0.025)),
            r_hi=float(np.quantile(rr, 0.975)),
            p_pos=float((rr > 0).mean()),
            cov_split_w=r0[1],
            cov_lo=float(np.quantile(cc, 0.025)),
            cov_hi=float(np.quantile(cc, 0.975)),
            cov_p_pos=float((cc > 0).mean()),
        )
        pd.DataFrame(
            dict(
                cell=cells,
                odd=est[0],
                even=est[1],
                n_odd=n[0],
                n_even=n[1],
                odd_se=bs[0].std(0, ddof=1),
                even_se=bs[1].std(0, ddof=1),
            )
        ).to_csv(OUT / f"halves_{name}.csv", index=False)
        tab, vc = full_fit(dd, rng, boots)
        s.update(vc)
        tab.to_csv(OUT / f"cells_{name}.csv", index=False)
        ers = []
        for lo_, hi_ in ERAS:
            t, _ = full_fit(
                dd[(dd.season >= lo_) & (dd.season <= hi_)].reset_index(drop=True), rng, boots_era
            )
            ers.append(t.rename(columns=lambda c, y=lo_: c if c == "cell" else f"{c}_{y}"))
        m = ers[0].merge(ers[1], on="cell")
        m.to_csv(OUT / f"era_{name}.csv", index=False)
        a, b = f"{ERAS[0][0]}", f"{ERAS[1][0]}"
        ok = ((m["n_" + a] > 0) & (m["n_" + b] > 0)).values
        mm = m[ok]
        ew = 1.0 / (mm["se_" + a].values ** 2 + mm["se_" + b].values ** 2 + 1e-300)
        s["era_corr_w"], s["era_cov_w"] = wcorr(mm["est_" + a].values, mm["est_" + b].values, ew)
        s["era_mean_abs_w"] = [
            float((ew * np.abs(mm["est_" + y])).sum() / ew.sum()) for y in (a, b)
        ]
        summ[name] = s
        print(name, s, flush=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

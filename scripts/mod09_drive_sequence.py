import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "mod09"
os.environ.setdefault("SIM10_SEASONS", "2018-2025")
os.environ.setdefault("SIM10_OUT", str(REPO / "artifacts" / "sim06" / "sim10"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sim10_drive as s10  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
SEASONS = tuple(range(2018, 2026))
EVAL = tuple(range(2020, 2026))
SEQ_K = (8, 16, 32)
SEQ_HIDDEN = ((4,), (16,))
SEQ_L2 = (1.0, 10.0)
SUM_K = (8, 16, 32, 64, 0)
SUM_ALPHA = (1.0, 10.0, 100.0, 1000.0)
INNER_BLOCKS = 3
NBOOT = 4000
BINS = 5
KINDS = ("seq", "sum")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def build_drives():
    seg = pd.read_parquet(s10.OUT / "segments.parquet")
    po, pd_ = s10.segment_points(seg)
    keep = (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    seg = seg[keep].reset_index(drop=True)
    seg["po"], seg["pd"] = po[keep], pd_[keep]
    rg = pd.read_parquet(s10.OUT / "real_games.parquet")
    rg = rg.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    seg = seg.merge(rg[["game_id", "home_team", "away_team"]], on="game_id")
    seg = seg.sort_values(["game_id", "row"]).reset_index(drop=True)
    return seg, rg


def team_histories(seg, rg):
    hm = seg["home"].to_numpy() == 1
    off = np.where(hm, seg["home_team"], seg["away_team"])
    dfn = np.where(hm, seg["away_team"], seg["home_team"])
    net = (seg["po"] - seg["pd"]).to_numpy(float)
    yl = seg["yl"].to_numpy(float)
    sd = seg["sd"].to_numpy(float)
    gid = seg["game_id"].to_numpy()
    bounds = {}
    st = 0
    for i in range(1, len(gid) + 1):
        if i == len(gid) or gid[i] != gid[st]:
            bounds[gid[st]] = (st, i)
            st = i
    per = {}
    for r in rg.itertuples():
        if r.game_id not in bounds:
            continue
        a, b = bounds[r.game_id]
        for team in (r.home_team, r.away_team):
            mo = off[a:b] == team
            md = dfn[a:b] == team
            per.setdefault(team, []).append(
                (
                    r.game_id,
                    r.season,
                    r.week,
                    pd.Timestamp(r.gameday),
                    np.c_[net[a:b][mo], yl[a:b][mo], sd[a:b][mo]],
                    np.c_[net[a:b][md], yl[a:b][md], sd[a:b][md]],
                )
            )
    return per


def team_game_table(per):
    rows = {}
    for team, games in per.items():
        for i, g in enumerate(games):
            prior = games[:i]
            for p in prior:
                assert (p[1], p[2]) < (g[1], g[2]) and p[3] < g[3], (
                    "history not strictly before the game"
                )
            rows[(g[0], team)] = (len(prior), [p[4] for p in prior], [p[5] for p in prior])
    return rows


def window(chunks, K):
    arr = np.full((K, 3), np.nan)
    if not chunks:
        return arr
    cat = np.vstack(chunks)[-K:]
    arr[K - len(cat) :] = cat
    return arr


def features(rows, game_ids, home, away, K, kind):
    n = len(game_ids)
    X = np.empty((n, 1 + 2 * 2 * K * 3)) if kind == "seq" else np.empty((n, 1 + 2 * 2))
    for j, (g, h, a) in enumerate(zip(game_ids, home, away, strict=True)):
        parts = []
        for team in (h, a):
            _, offs, defs = rows[(g, team)]
            for ch in (offs, defs):
                if kind == "seq":
                    parts.append(window(ch, K).ravel())
                elif ch:
                    cat = np.vstack(ch)
                    cat = cat[-K:] if K else cat
                    parts.append(np.array([cat[:, 0].mean()]))
                else:
                    parts.append(np.array([np.nan]))
        X[j] = np.concatenate([[rows[(g, h)][0]], *parts])
    return X


def target_table(seg):
    sign = np.where(seg["home"].to_numpy() == 1, 1.0, -1.0)
    d = pd.DataFrame(
        {"game_id": seg["game_id"], "t": sign * (seg["po"] - seg["pd"]).to_numpy(float)}
    )
    return d.groupby("game_id")["t"].sum()


def make_model(spec):
    if spec["kind"] == "seq":
        reg = MLPRegressor(
            hidden_layer_sizes=spec["hidden"], alpha=spec["l2"], max_iter=400, random_state=0
        )
    else:
        reg = Ridge(alpha=spec["alpha"])
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), reg)


def specs(kind):
    if kind == "seq":
        return [
            dict(kind="seq", K=k, hidden=h, l2=a) for k in SEQ_K for h in SEQ_HIDDEN for a in SEQ_L2
        ]
    return [dict(kind="sum", K=k, alpha=a) for k in SUM_K for a in SUM_ALPHA]


def spec_name(s):
    if s["kind"] == "seq":
        return f"seq_K{s['K']}_h{'x'.join(map(str, s['hidden']))}_l2{s['l2']}"
    return f"sum_K{s['K']}_a{s['alpha']}"


class Feat:
    def __init__(self, rows, gdf):
        self.rows, self.g = rows, gdf
        self.cache = {}

    def get(self, spec):
        key = (spec["kind"], spec["K"])
        if key not in self.cache:
            self.cache[key] = features(
                self.rows,
                self.g.game_id.to_numpy(),
                self.g.home_team.to_numpy(),
                self.g.away_team.to_numpy(),
                spec["K"],
                spec["kind"],
            )
        return self.cache[key]


def fit_predict(F, spec, y, tr_idx, te_idx):
    X = F.get(spec)
    return make_model(spec).fit(X[tr_idx], y[tr_idx]).predict(X[te_idx])


def select(F, y, season, usable, seasons_train, kind):
    blocks = np.array_split(np.array(sorted(seasons_train)), INNER_BLOCKS)
    best, bs = None, np.inf
    for sp in specs(kind):
        sse, n = 0.0, 0
        for b in blocks:
            te = np.flatnonzero(np.isin(season, b) & usable)
            tr = np.flatnonzero(np.isin(season, seasons_train) & ~np.isin(season, b) & usable)
            p = fit_predict(F, sp, y, tr, te)
            sse += float(((p - y[te]) ** 2).sum())
            n += len(te)
        if sse / n < bs:
            bs, best = sse / n, sp
    return best, bs


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lr_fit(X, y):
    m = LogisticRegression(penalty=None, max_iter=1000).fit(X, y)
    return np.r_[m.intercept_, m.coef_.ravel()]


def lr_pred(b, X):
    return 1 / (1 + np.exp(-(b[0] + X @ b[1:])))


def llv(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_boot(v, blocks, rng):
    ub, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(ub))
    c = np.bincount(inv, minlength=len(ub)).astype(float)
    draws = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=NBOOT)
    m = (draws @ s) / (draws @ c)
    return dict(
        mean=float(v.mean()),
        low=float(np.percentile(m, 2.5)),
        high=float(np.percentile(m, 97.5)),
        probability_positive=float((m > 0).mean()),
        se=float(m.std()),
        blocks=len(ub),
    )


def reliability(p, y):
    q = np.quantile(p, np.linspace(0, 1, BINS + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, BINS - 1)
    return [
        dict(
            bin=int(i),
            n=int((idx == i).sum()),
            mean_p=float(p[idx == i].mean()),
            hit=float(y[idx == i].mean()),
        )
        for i in range(BINS)
        if (idx == i).any()
    ]


def load_eval(gdf):
    pg = pd.read_parquet(
        sorted((REPO / "artifacts" / "pick_probability").glob("*/per_game.parquet"))[-1]
    )
    pg = pg[(pg.game_type == "REG") & pg.season.between(EVAL[0], EVAL[-1])].copy()
    pg["game_id"] = pg.game_id.astype(str)
    pg = pg[pg.game_id.isin(set(gdf.game_id))]
    pg = pg[pg.out_of_season_tuesday_home_probability.notna() & pg.tue_open_home_spread.notna()]
    return pg.reset_index(drop=True)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    seg, rg = build_drives()
    rows = team_game_table(team_histories(seg, rg))
    tgt = target_table(seg)
    gdf = rg[rg.game_id.isin(tgt.index)][
        ["game_id", "season", "week", "home_team", "away_team"]
    ].reset_index(drop=True)
    y_drive = tgt.reindex(gdf.game_id).to_numpy(float)
    season = gdf.season.to_numpy(int)
    n_h = np.array([rows[(g, h)][0] for g, h in zip(gdf.game_id, gdf.home_team, strict=True)])
    n_a = np.array([rows[(g, a)][0] for g, a in zip(gdf.game_id, gdf.away_team, strict=True)])
    usable = (n_h > 0) & (n_a > 0)
    F = Feat(rows, gdf)
    log("games", len(gdf), "usable", int(usable.sum()))

    pg = load_eval(gdf)
    gi = {g: i for i, g in enumerate(gdf.game_id)}
    pos = np.array([gi[g] for g in pg.game_id])
    L = pg.tue_open_home_spread.to_numpy(float)
    ps = pg.out_of_season_tuesday_home_probability.to_numpy(float)
    yc = (pg.result.to_numpy(float) > L).astype(float)
    dec = pg.result.to_numpy(float) != L
    pseason = pg.season.to_numpy(int)
    log("eval games", len(pg), "decisive", int(dec.sum()))

    chosen = {k: {} for k in KINDS}
    zfold = {k: {} for k in KINDS}
    drive_mse = {k: [] for k in KINDS}
    for s in EVAL:
        trs = [t for t in SEASONS if t != s]
        for k in KINDS:
            sp, cv = select(F, y_drive, season, usable, trs, k)
            chosen[k][s] = dict(
                spec={a: (list(b) if isinstance(b, tuple) else b) for a, b in sp.items()},
                inner_mse=cv,
            )
            log("outer", s, k, spec_name(sp), "inner mse", round(cv, 3))
            tr = np.flatnonzero(np.isin(season, trs) & usable)
            te = np.flatnonzero((season == s) & usable)
            p_te = fit_predict(F, sp, y_drive, tr, te)
            drive_mse[k].append(
                dict(
                    season=s,
                    mse=float(((p_te - y_drive[te]) ** 2).mean()),
                    base=float(((y_drive[tr].mean() - y_drive[te]) ** 2).mean()),
                    n=len(te),
                )
            )
            zmap = np.full(len(gdf), np.nan)
            zmap[te] = p_te
            for t in EVAL:
                if t == s:
                    continue
                tr2 = np.flatnonzero(
                    np.isin(season, [u for u in SEASONS if u not in (s, t)]) & usable
                )
                te2 = np.flatnonzero((season == t) & usable)
                zmap[te2] = fit_predict(F, sp, y_drive, tr2, te2)
            zfold[k][s] = zmap

    ok = dec.copy()
    for k in KINDS:
        ok &= ~np.isnan(zfold[k][EVAL[0]][pos])
    lp = logit(ps)
    pr = {
        "served": ps,
        "base": np.full(len(pg), np.nan),
        "seq": np.full(len(pg), np.nan),
        "sum": np.full(len(pg), np.nan),
    }
    coefs = {n: [] for n in ("base", "seq", "sum")}
    insamp = {n: [] for n in ("base", "seq", "sum")}
    zheld = {k: np.full(len(pg), np.nan) for k in KINDS}
    for s in EVAL:
        tr = ok & (pseason != s)
        te = ok & (pseason == s)
        for name in ("base", "seq", "sum"):
            X = lp[:, None] if name == "base" else np.c_[lp, zfold[name][s][pos] - L]
            b = lr_fit(X[tr], yc[tr])
            coefs[name].append(dict(season=s, b=[float(v) for v in b]))
            pr[name][te] = lr_pred(b, X[te])
            insamp[name].append(float(llv(yc[tr], lr_pred(b, X[tr])).mean()))
            if name != "base":
                zheld[name][te] = X[te, 1]

    rng = np.random.default_rng(20261102)
    blocks = pg.season.to_numpy(int) * 100 + pg.week.to_numpy(int)
    y = yc[ok]
    bl = blocks[ok]
    sea = pseason[ok]
    res = {}
    for name in ("served", "base", "seq", "sum"):
        p = pr[name][ok]
        res[name] = dict(
            log_loss=float(llv(y, p).mean()),
            brier=float(((p - y) ** 2).mean()),
            accuracy=float(((p > 0.5) == (y == 1)).mean() * 100),
            reliability=reliability(p, y),
        )

    def comp(a, b):
        pa, pb = pr[a][ok], pr[b][ok]
        d = llv(y, pb) - llv(y, pa)
        ca, cb = ((pa > 0.5) == (y == 1)), ((pb > 0.5) == (y == 1))
        flips = (pa > 0.5) != (pb > 0.5)
        return dict(
            log_loss_improvement=block_boot(d, bl, rng),
            accuracy_points=block_boot((ca.astype(float) - cb.astype(float)) * 100, bl, rng),
            brier_improvement=block_boot((pb - y) ** 2 - (pa - y) ** 2, bl, rng),
            flips=int(flips.sum()),
            flip_record_a_right_wrong=[int(ca[flips].sum()), int((~ca[flips]).sum())],
            ll_improvement_by_season={int(s): float(d[sea == s].mean()) for s in EVAL},
        )

    for a, b in (("seq", "base"), ("sum", "base"), ("seq", "sum"), ("seq", "served")):
        res[f"{a}_vs_{b}"] = comp(a, b)
    res["coefs"] = coefs
    res["in_sample_log_loss_mean"] = {k: float(np.mean(v)) for k, v in insamp.items()}
    res["gap_in_minus_out_ll_gain_vs_base"] = {
        k: float(
            (np.mean(insamp["base"]) - np.mean(insamp[k]))
            - (res["base"]["log_loss"] - res[k]["log_loss"])
        )
        for k in KINDS
    }
    res["corr_z_seq_sum"] = float(np.corrcoef(zheld["seq"][ok], zheld["sum"][ok])[0, 1])
    res["z_sd"] = {k: float(np.std(zheld[k][ok])) for k in KINDS}
    out = dict(
        chosen=chosen,
        drive_mse=drive_mse,
        drive_mse_pooled={
            k: dict(
                model=float(np.average([d["mse"] for d in v], weights=[d["n"] for d in v])),
                mean_only=float(np.average([d["base"] for d in v], weights=[d["n"] for d in v])),
            )
            for k, v in drive_mse.items()
        },
        n_eval=len(pg),
        n_decisive_scored=int(ok.sum()),
        results=res,
    )
    (ART / "results.json").write_text(json.dumps(out, indent=1, default=float))
    log("done")
    for n in ("served", "base", "seq", "sum"):
        print(n, {m: round(res[n][m], 5) for m in ("log_loss", "brier", "accuracy")})
    for k in ("seq_vs_base", "sum_vs_base", "seq_vs_sum", "seq_vs_served"):
        r = res[k]
        print(
            k,
            {
                m: tuple(round(r[m][q], 4) for q in ("mean", "low", "high", "probability_positive"))
                for m in ("log_loss_improvement", "accuracy_points", "brier_improvement")
            },
            r["flips"],
            r["flip_record_a_right_wrong"],
        )
    print(
        "gap",
        res["gap_in_minus_out_ll_gain_vs_base"],
        "drive mse",
        out["drive_mse_pooled"],
        "corr",
        res["corr_z_seq_sum"],
        res["z_sd"],
    )


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
OUT = REPO / "artifacts" / "sim28"
SEASONS = range(2009, 2026)
COLS = [
    "game_id",
    "play_id",
    "season_type",
    "qtr",
    "down",
    "posteam",
    "fixed_drive",
    "fixed_drive_result",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]
WINDOW = 180.0
TWO_MIN = 120.0
CLASSES = ["td", "fg", "punt", "turnover", "expiry"]
MAP = {
    "Touchdown": 0,
    "Field goal": 1,
    "Punt": 2,
    "Turnover": 3,
    "Turnover on downs": 3,
    "Missed field goal": 3,
    "Opp touchdown": 3,
    "Safety": 3,
    "End of half": 4,
}
PTS = np.array([7.0, 3.0, 0.0, 0.0, 0.0])
NBOOT = 1000
SEED = 2801
FOLDS = 10
FP_EDGES = (40.0, 70.0)
FP_NAMES = ["yl<=40", "40<yl<=70", "yl>70"]


def drives():
    fr = []
    for s in SEASONS:
        f = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS)
        fr.append(f[f["season_type"] == "REG"].assign(season=s))
    p = pd.concat(fr, ignore_index=True)
    p = p[p["qtr"].isin([1, 2, 3, 4])]
    p = p.sort_values(["game_id", "play_id"], kind="stable")
    snap = p[p["down"].notna() & p["posteam"].notna() & p["fixed_drive"].notna()]
    d = snap.groupby(["game_id", "fixed_drive"], sort=False).head(1).copy()
    d["half"] = np.where(d["qtr"] <= 2, 1, 2)
    gsr = d["game_seconds_remaining"].to_numpy(float)
    d["secs"] = np.where(d["half"] == 1, gsr - 1800.0, gsr)
    d = d.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
    nxt_same = (d["game_id"].to_numpy()[1:] == d["game_id"].to_numpy()[:-1]) & (
        d["half"].to_numpy()[1:] == d["half"].to_numpy()[:-1]
    )
    nxt = np.r_[np.where(nxt_same, d["secs"].to_numpy()[1:], np.nan), np.nan]
    d["el"] = np.where(np.isnan(nxt), d["secs"], d["secs"] - nxt)
    d["cls"] = d["fixed_drive_result"].map(MAP)
    d = d[
        (d["secs"] > 0)
        & (d["secs"] <= WINDOW)
        & d["cls"].notna()
        & d["posteam_timeouts_remaining"].notna()
        & d["defteam_timeouts_remaining"].notna()
        & d["score_differential"].notna()
        & d["yardline_100"].notna()
    ].copy()
    d["cls"] = d["cls"].astype(int)
    d["to"] = d["posteam_timeouts_remaining"].clip(0, 3).astype(int)
    d["td"] = d["defteam_timeouts_remaining"].clip(0, 3).astype(int)
    d["pts"] = PTS[d["cls"].to_numpy()]
    d["fp"] = np.searchsorted(FP_EDGES, d["yardline_100"].to_numpy(float), side="left")
    d["trail"] = (d["score_differential"] < 0).astype(int)
    d["band"] = (d["secs"] > TWO_MIN).astype(int)
    return d.reset_index(drop=True)


def cluster_sums(cell, gidx, V, ncell, ng):
    A = np.zeros((ng, ncell, V.shape[1]))
    ok = cell >= 0
    np.add.at(A, (gidx[ok], cell[ok]), V[ok])
    return A


def table(d, cell, ncell, labels, rng, gidx, ng, W):
    V = np.c_[
        np.eye(5)[d["cls"].to_numpy()],
        d["pts"].to_numpy(),
        d["el"].to_numpy(),
        np.ones(len(d)),
    ]
    A = cluster_sums(cell, gidx, V, ncell, ng)
    tot = A.sum(axis=0)
    B = np.einsum("bg,gck->bck", W, A)
    rows = []
    names = [*CLASSES, "points", "seconds"]
    for c in range(ncell):
        n = tot[c, 7]
        row = {"cell": labels[c], "n": int(n)}
        for k, nm in enumerate(names):
            if n == 0:
                row[nm] = None
                continue
            est = tot[c, k] / n
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = B[:, c, k] / B[:, c, 7]
            bs = bs[np.isfinite(bs)]
            row[nm] = [float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        rows.append(row)
    return rows


def base_features(d):
    sd = d["score_differential"].to_numpy(float)
    yl = d["yardline_100"].to_numpy(float)
    return np.c_[
        np.log1p(d["secs"].to_numpy(float)),
        yl,
        yl * yl,
        np.clip(sd, -14.0, 14.0),
        (sd < 0).astype(float),
        (d["half"].to_numpy() == 2).astype(float),
    ]


def oof_ll(X, y, groups):
    ll = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1e3, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pr = np.zeros((len(te), 5))
        pr[:, m.classes_] = m.predict_proba(sc.transform(X[te]))
        pr = np.clip(pr, 1e-9, None)
        ll[te] = np.log(pr[np.arange(len(te)), y[te]] / pr[np.arange(len(te))].sum(axis=1))
    return ll


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = drives()
    gid, gidx = np.unique(d["game_id"].to_numpy(), return_inverse=True)
    ng = len(gid)
    W = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=NBOOT).astype(float)
    res = {"n_drives": len(d), "n_games": ng, "seasons": [2009, 2025]}
    res["by_to_counts"] = d["to"].value_counts().sort_index().to_dict()

    cell1 = (d["half"].to_numpy() - 1) * 4 + d["to"].to_numpy()
    lab1 = [f"half{h}_to{t}" for h in (1, 2) for t in range(4)]
    res["T1"] = table(d, cell1, 8, lab1, rng, gidx, ng, W)
    pooled = table(d, d["to"].to_numpy(), 4, [f"to{t}" for t in range(4)], rng, gidx, ng, W)
    res["T1_pooled_halves"] = pooled

    cell2 = ((d["to"].to_numpy() * 3 + d["fp"].to_numpy()) * 2 + d["trail"].to_numpy()) * 2 + d[
        "band"
    ].to_numpy()
    lab2 = [
        f"to{t}_{FP_NAMES[f]}_{'trail' if tr else 'level_or_lead'}_{'120-180' if b else '0-120'}"
        for t in range(4)
        for f in range(3)
        for tr in (0, 1)
        for b in (0, 1)
    ]
    res["T2"] = table(d, cell2, 48, lab2, rng, gidx, ng, W)

    X0 = base_features(d)
    y = d["cls"].to_numpy()
    to = d["to"].to_numpy()
    tdm = d["td"].to_numpy()
    X1 = np.c_[X0, np.eye(4)[to][:, 1:], tdm]
    ll0 = oof_ll(X0, y, gidx)
    ll1 = oof_ll(X1, y, gidx)
    dll = ll1 - ll0
    A = np.zeros(ng)
    np.add.at(A, gidx, dll)
    bs = (W @ A) / len(d)
    res["T3_loglik"] = {
        "base_ll_per_drive": float(ll0.mean()),
        "with_timeouts_ll_per_drive": float(ll1.mean()),
        "delta_per_drive": float(dll.mean()),
        "interval": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "probability_positive": float((bs > 0).mean()),
        "folds": FOLDS,
    }

    Xp = np.c_[np.ones(len(d)), (X0 - X0.mean(axis=0)) / X0.std(axis=0), to.astype(float)]
    yp = d["pts"].to_numpy()
    beta = np.linalg.lstsq(Xp, yp, rcond=None)[0]
    rows_by_game = [np.flatnonzero(gidx == g) for g in range(ng)]
    cb = []
    for _ in range(NBOOT):
        pick = rng.integers(0, ng, ng)
        idx = np.concatenate([rows_by_game[g] for g in pick])
        cb.append(np.linalg.lstsq(Xp[idx], yp[idx], rcond=None)[0][-1])
    cb = np.array(cb)
    res["T3_points_per_offence_timeout"] = {
        "coef": float(beta[-1]),
        "interval": [float(np.quantile(cb, 0.025)), float(np.quantile(cb, 0.975))],
        "probability_positive": float((cb > 0).mean()),
    }
    trail = d["trail"].to_numpy() == 1
    for nm, m in (("trailing", trail), ("level_or_lead", ~trail)):
        Xm, ym = Xp[m], yp[m]
        bm = np.linalg.lstsq(Xm, ym, rcond=None)[0][-1]
        rg = [rows_by_game[g][m[rows_by_game[g]]] for g in range(ng)]
        bb = []
        for _ in range(NBOOT):
            pick = rng.integers(0, ng, ng)
            idx = np.concatenate([rg[g] for g in pick])
            bb.append(np.linalg.lstsq(Xp[idx], yp[idx], rcond=None)[0][-1])
        bb = np.array(bb)
        res[f"T3_points_per_timeout_{nm}"] = {
            "coef": float(bm),
            "interval": [float(np.quantile(bb, 0.025)), float(np.quantile(bb, 0.975))],
            "probability_positive": float((bb > 0).mean()),
            "n": int(m.sum()),
        }

    h = json.loads((REPO / "artifacts" / "sim10" / "bandwidths_week.json").read_text())["h_reg"]
    res["engine_timeout_bandwidths"] = {
        f"regime{r}": {"too": h[str(r)]["h"][3], "tod": h[str(r)]["h"][4]} for r in (1, 4)
    }
    seg = pd.read_parquet(REPO / "artifacts" / "sim10" / "segments.parquet")
    late = seg[(seg["secs"] <= WINDOW) & (seg["secs"] > 0) & (seg["bad"] == 0) & (seg["half"] < 3)]
    res["pool_late_drives"] = {
        "n": len(late),
        "by_offence_timeouts": late["too"].value_counts().sort_index().to_dict(),
    }
    (OUT / "timeouts.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("T2",)}, indent=1, default=str))


if __name__ == "__main__":
    main()

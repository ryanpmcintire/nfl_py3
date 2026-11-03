import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "artifacts" / "sim08_market_anchored" / "main"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
GF = REPO / "data" / "processed" / "game_features.parquet"
SEASONS = range(2020, 2026)
COLS = [
    "game_id",
    "play_id",
    "season_type",
    "home_team",
    "posteam",
    "qtr",
    "fixed_drive",
    "game_seconds_remaining",
    "posteam_score",
    "score_differential",
]
KEYS = ("game_id", "line", "total", "x_diff", "x_env", "margins", "totals", "ot")


def load_sim():
    parts = []
    for s in SEASONS:
        z = np.load(RUN / f"primary_{s}.npz", allow_pickle=True)
        parts.append({k: z[k] for k in KEYS})
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def load_real(ids):
    g = pd.read_parquet(GF, columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    rows = []
    for s in SEASONS:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS)
        p = p[(p["season_type"] == "REG") & p["game_id"].isin(ids)]
        p = p.sort_values(["game_id", "play_id"], kind="stable")
        for gid, d in p.groupby("game_id", sort=False):
            ok = (
                d["posteam"].notna() & d["posteam_score"].notna() & d["score_differential"].notna()
            ).to_numpy()
            ph = (d["posteam"] == d["home_team"]).to_numpy()
            ps = d["posteam_score"].to_numpy(float)
            sd = d["score_differential"].to_numpy(float)
            hs = np.where(ph, ps, ps - sd)[ok]
            as_ = np.where(ph, ps - sd, ps)[ok]
            gsr = d["game_seconds_remaining"].to_numpy(float)[ok]
            qtr = d["qtr"].to_numpy()[ok]
            h_fin, a_fin = g.loc[gid, "home_score"], g.loc[gid, "away_score"]
            ev = []
            ph_, pa_ = 0.0, 0.0
            for i in range(len(hs)):
                if hs[i] != ph_ or as_[i] != pa_:
                    ev.append((hs[i] - ph_, as_[i] - pa_, ph_ - pa_, gsr[i], qtr[i]))
                    ph_, pa_ = hs[i], as_[i]
            if h_fin != ph_ or a_fin != pa_:
                ev.append((h_fin - ph_, a_fin - pa_, ph_ - pa_, 0.0, 4))
            rows.append(
                {
                    "game_id": gid,
                    "h": h_fin,
                    "a": a_fin,
                    "ot": bool((d["qtr"] == 5).any()),
                    "drives": d["fixed_drive"].nunique(),
                    "ev": ev,
                }
            )
    return pd.DataFrame(rows).set_index("game_id")


def last_score_paths(real):
    out = {}
    for _, r in real.iterrows():
        m = r["h"] - r["a"]
        if abs(m) != 3 or not r["ev"]:
            continue
        dh, da, pre, _, _ = r["ev"][-1]
        win_home = m > 0
        scorer_is_winner = (dh > 0) == win_home
        k = dh if dh > 0 else da
        pre_w = pre if win_home else -pre
        key = (
            "OT" if r["ot"] else "REG",
            "winner" if scorer_is_winner else "loser",
            int(k),
            int(pre_w),
        )
        out[key] = out.get(key, 0) + 1
    return out


def slope(x, y):
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN / "shape_decompose.json"))
    a = ap.parse_args()
    S = load_sim()
    ids = set(S["game_id"].tolist())
    real = load_real(ids).loc[S["game_id"]]
    rm = (real["h"] - real["a"]).to_numpy(float)
    rt = (real["h"] + real["a"]).to_numpy(float)
    rot = real["ot"].to_numpy(bool)
    sm, st, so = S["margins"].astype(float), S["totals"].astype(float), S["ot"]
    res = {"n_games": len(rm), "draws": int(sm.shape[1])}
    res["lattice_sim"] = {k: float(np.mean(np.abs(sm) == k)) for k in range(21)}
    res["lattice_real"] = {k: float(np.mean(np.abs(rm) == k)) for k in range(21)}
    res["ot_rate"] = {"sim": float(so.mean()), "real": float(rot.mean())}
    m3s = np.abs(sm) == 3
    res["mass3_by_ot"] = {
        "sim_ot": float((m3s & so).mean()),
        "sim_reg": float((m3s & ~so).mean()),
        "real_ot": float(((np.abs(rm) == 3) & rot).mean()),
        "real_reg": float(((np.abs(rm) == 3) & ~rot).mean()),
    }
    tot_t = S["total"]
    ok = np.isfinite(tot_t)
    edges = [38, 41, 44, 47, 50]
    tb = np.digitize(tot_t, edges)
    res["mass3_by_total_bin"] = {}
    for b in range(len(edges) + 1):
        k = ok & (tb == b)
        res["mass3_by_total_bin"][str(b)] = {
            "n": int(k.sum()),
            "open_total": float(tot_t[k].mean()),
            "sim_total": float(st[k].mean()),
            "real_total": float(rt[k].mean()),
            "sim_m3": float(m3s[k].mean()),
            "real_m3": float((np.abs(rm[k]) == 3).mean()),
        }
    al = np.abs(S["line"])
    lb = np.digitize(al, [3, 6.5, 10])
    res["mass3_by_abs_line_bin"] = {}
    for b in range(4):
        k = lb == b
        res["mass3_by_abs_line_bin"][str(b)] = {
            "n": int(k.sum()),
            "sim": float(m3s[k].mean()),
            "real": float((np.abs(rm[k]) == 3).mean()),
        }
    sh, sa = (st + sm) / 2, (st - sm) / 2
    res["winner_score_mean"] = {
        "sim": float(np.maximum(sh, sa).mean()),
        "real": float(np.maximum(real["h"], real["a"]).mean()),
    }
    res["loser_score_mean"] = {
        "sim": float(np.minimum(sh, sa).mean()),
        "real": float(np.minimum(real["h"], real["a"]).mean()),
    }
    res["real_last_score_paths_margin3"] = {
        "|".join(map(str, k)): v
        for k, v in sorted(last_score_paths(real).items(), key=lambda kv: -kv[1])
    }
    dr = real["drives"].to_numpy(float)
    res["real_drives_per_game"] = float(dr.mean())
    res["real_points_per_drive"] = float(rt.sum() / dr.sum())
    res["slopes"] = {
        "sim_mean_total_on_open_total": slope(tot_t[ok], st.mean(1)[ok]),
        "real_total_on_open_total": slope(tot_t[ok], rt[ok]),
        "real_drives_on_open_total": slope(tot_t[ok], dr[ok]),
        "real_ppd_on_open_total": slope(tot_t[ok], (rt / dr)[ok]),
        "sim_mean_total_on_xenv": slope(S["x_env"], st.mean(1)),
        "open_total_on_xenv": slope(S["x_env"][ok], tot_t[ok]),
        "real_total_on_xenv": slope(S["x_env"], rt),
    }
    q = np.quantile(tot_t[ok], [0.2, 0.4, 0.6, 0.8])
    qb = np.digitize(tot_t, q)
    res["total_quintiles"] = []
    for b in range(5):
        k = ok & (qb == b)
        res["total_quintiles"].append(
            {
                "n": int(k.sum()),
                "open_total": float(tot_t[k].mean()),
                "sim_total": float(st[k].mean()),
                "real_total": float(rt[k].mean()),
                "real_drives": float(dr[k].mean()),
                "real_ppd": float(rt[k].sum() / dr[k].sum()),
            }
        )
    ab = np.digitize(al, np.quantile(al, [0.25, 0.5, 0.75]))
    res["gap_bins"] = []
    for b in range(4):
        k = ab == b
        res["gap_bins"].append(
            {
                "n": int(k.sum()),
                "abs_line": float(al[k].mean()),
                "open_total": float(np.nanmean(tot_t[k])),
                "sim_total": float(st[k].mean()),
                "real_total": float(rt[k].mean()),
                "real_drives": float(dr[k].mean()),
                "real_ppd": float(rt[k].sum() / dr[k].sum()),
                "sim_abs_margin": float(np.abs(sm[k]).mean()),
                "real_abs_margin": float(np.abs(rm[k]).mean()),
            }
        )
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25_generator as gen

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e3" / "revert"
PART = REPO / "data" / "players" / "participation" / "raw" / "20260813T131635Z"
SEASONS = range(2016, 2026)
NBINS = 10
NBOOT = 400
SEED = 0


def long_players(p, col, side):
    q = p[["game_id", "play_id", "possession_team", col]].copy()
    q[col] = q[col].str.split(";")
    q = q.explode(col).rename(columns={col: "pid"})
    q["side"] = side
    return q[q.pid.notna() & (q.pid != "")]


def season_frame(s):
    pb = pd.read_parquet(gen.PBP_DIR / f"season={s}" / "plays.parquet", columns=["game_id", "play_id", "week", "posteam", "defteam", "epa", "wp", "qtr", "game_seconds_remaining", "score_differential", "play_type", "qb_kneel", "qb_spike"])
    pb = pb[(pb.qb_kneel != 1) & (pb.qb_spike != 1)]
    pb = pb[pb.play_type.isin(["pass", "run"]) & pb.epa.notna() & pb.wp.notna() & pb.posteam.notna()]
    pa = pd.read_parquet(glob.glob(str(PART / f"season={s}" / "*.parquet"))[0])
    pa = pa[pa.n_offense > 0]
    pa = pa.merge(pb[["game_id", "play_id", "week", "posteam", "defteam"]], on=["game_id", "play_id"])
    o = long_players(pa, "offense_players", "o")
    o["team"] = o.possession_team
    d = long_players(pa, "defense_players", "d")
    d = d.merge(pa[["game_id", "play_id", "defteam"]], on=["game_id", "play_id"])
    d["team"] = d.defteam
    lp = pd.concat([o, d], ignore_index=True)[["game_id", "play_id", "side", "team", "pid"]]
    lp = lp.merge(pa[["game_id", "week"]].drop_duplicates(), on="game_id")
    cnt = lp.groupby(["side", "team", "pid", "week"]).size().rename("c").reset_index().sort_values("week")
    cnt["cum"] = cnt.groupby(["side", "team", "pid"]).c.cumsum() - cnt.c
    tg = pa.assign(o=pa.posteam, d=pa.defteam)
    tp = pd.concat([tg.groupby(["posteam", "week"]).size().rename("n").reset_index().rename(columns={"posteam": "team"}).assign(side="o"), tg.groupby(["defteam", "week"]).size().rename("n").reset_index().rename(columns={"defteam": "team"}).assign(side="d")]).sort_values("week")
    tp["tcum"] = tp.groupby(["side", "team"]).n.cumsum() - tp.n
    lp = lp.merge(cnt[["side", "team", "pid", "week", "cum"]], on=["side", "team", "pid", "week"]).merge(tp[["side", "team", "week", "tcum"]], on=["side", "team", "week"])
    lp = lp[lp.tcum > 0]
    lp["sh"] = lp.cum / lp.tcum
    g = lp.groupby(["game_id", "play_id", "side"]).sh.mean().unstack("side").rename(columns={"o": "sh_o", "d": "sh_d"}).reset_index()
    out = pb.merge(g, on=["game_id", "play_id"])
    out["season"] = s
    return out


def main():
    df = pd.concat([season_frame(s) for s in SEASONS], ignore_index=True).dropna(subset=["sh_o", "sh_d"])
    lg = df.epa.mean()
    for side, key in [("oe", "posteam"), ("de", "defteam")]:
        grp = df.groupby(["season", key]).epa
        tot, n = grp.transform("sum"), grp.transform("count")
        ge = df.groupby(["season", key, "game_id"]).epa
        gs, gn = ge.transform("sum"), ge.transform("count")
        df[side] = (tot - gs) / (n - gn) - lg
    df["r"] = df.epa - lg - df.oe - df.de
    for c in ["sh_o", "sh_d"]:
        key = "posteam" if c == "sh_o" else "defteam"
        df[c + "_dev"] = df[c] - df.groupby(["season", key])[c].transform("mean")
    df["lead"] = df.score_differential
    df["half2"] = df.qtr.isin([3, 4])
    h = df[df.half2].copy()
    h["bin"] = pd.qcut(h.wp, NBINS, labels=False, duplicates="drop")
    h["cell"] = h.bin.astype(str) + "_" + h.qtr.astype(int).astype(str)
    cells = sorted(h.cell.unique())
    X = np.column_stack([h.sh_o_dev.values, h.sh_d_dev.values] + [(h.cell == c).values.astype(float) for c in cells])
    y = h.r.values
    gid, gi = np.unique(h.game_id.values, return_inverse=True)
    nb = int(h.bin.max()) + 1
    B = np.column_stack([(h.bin.values == b).astype(float) for b in range(nb)])
    rng = np.random.default_rng(SEED)

    def stats(w):
        W = w[gi]
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X, XtW @ y)
        bw = B * W[:, None]
        den = bw.sum(0)
        r_bin = bw.T @ y / den
        so = bw.T @ h.sh_o_dev.values / den
        sd = bw.T @ h.sh_d_dev.values / den
        return np.concatenate([beta[:2], r_bin, so, sd, beta[0] * so, beta[1] * sd, r_bin - beta[0] * so - beta[1] * sd])

    base = stats(np.ones(len(gid)))
    boots = np.array([stats(rng.multinomial(len(gid), np.ones(len(gid)) / len(gid)).astype(float)) for _ in range(NBOOT)])
    names = ["beta_off", "beta_def"] + [f"{k}_{b}" for k in ["r", "sho", "shd", "med_off", "med_def", "unexp"] for b in range(nb)]
    res = {n: dict(est=float(base[i]), lo=float(np.percentile(boots[:, i], 2.5)), hi=float(np.percentile(boots[:, i], 97.5)), pp=float((boots[:, i] > 0).mean())) for i, n in enumerate(names)}
    lines = [f"plays {len(h)} games {len(gid)} lg_epa {lg:.4f}"]
    for k in ["beta_off", "beta_def"]:
        v = res[k]
        lines.append(f"{k} {v['est']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] pp_pos {v['pp']:.2f}")
    wp_edges = h.groupby("bin").wp.agg(["min", "max"])
    lines.append("bin wp_range  r  sh_o_dev sh_d_dev  med_off med_def unexplained (est [lo,hi] pp)")
    for b in range(nb):
        f = lambda k: f"{res[f'{k}_{b}']['est']:+.4f}"
        u = res[f"unexp_{b}"]
        rr = res[f"r_{b}"]
        lines.append(f"{b} {wp_edges.loc[b,'min']:.2f}-{wp_edges.loc[b,'max']:.2f} r {f('r')} [{rr['lo']:+.4f},{rr['hi']:+.4f}] pp {rr['pp']:.2f} sho {f('sho')} shd {f('shd')} med_o {f('med_off')} med_d {f('med_def')} unexp {u['est']:+.4f} [{u['lo']:+.4f},{u['hi']:+.4f}] pp {u['pp']:.2f}")
    (OUT / "revert.txt").write_text("\n".join(lines))
    (OUT / "revert.json").write_text(json.dumps(res))
    print("\n".join(lines))


main()

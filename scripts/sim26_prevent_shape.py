import glob
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

N_BOOT = 300
N_SIGN = 2000
MIN_SPELL_GAMES = 16
EXPL_Q = 0.9
HALF_SECS = 1800.0
rng = np.random.default_rng(2601)
with open("artifacts/sim19/trailing_risk.json") as fh0:
    edges = json.load(fh0)
LE = np.array(edges["deficit_edges"])
TE = np.array(edges["secs_edges"])
NEUTRAL = LE[0]
NT = len(TE) + 1
NL = len(LE)
REL_LEAD = 1
REL_TIME_MAX = 1


def band(df):
    df["tb"] = np.searchsorted(TE, df.gsr.values, side="left")
    lb = np.where(df.lead.abs() <= NEUTRAL, 0, np.searchsorted(LE, df.lead.values, side="left"))
    lb[df.lead.values < -NEUTRAL] = -1
    df["lb"] = lb
    return df[df.lb >= 0].reset_index(drop=True)


d = pd.read_parquet(
    "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "fixed_drive",
        "fixed_drive_result",
        "play_type",
        "yards_gained",
        "qb_kneel",
        "qb_spike",
        "aborted_play",
        "game_seconds_remaining",
        "score_differential",
    ],
)
d = d[d.score_differential.notna() & d.game_seconds_remaining.notna()].copy()
d["lead"] = -d.score_differential
d["gsr"] = d.game_seconds_remaining
pl = d[
    d.play_type.isin(["pass", "run"])
    & (d.qb_kneel != 1)
    & (d.qb_spike != 1)
    & (d.aborted_play != 1)
    & d.yards_gained.notna()
].copy()
thr = float(pl.yards_gained.quantile(EXPL_Q))
pl["ex"] = (pl.yards_gained >= thr).astype(float)
pl["y"] = pl.yards_gained.astype(float)
pl["dsn"] = pl.defteam + "|" + pl.season.astype(str)
pl = band(pl)

dr = (
    d.sort_values(["game_id", "fixed_drive", "gsr"], ascending=[True, True, False])
    .groupby(["game_id", "posteam", "fixed_drive"], as_index=False)
    .agg(
        season=("season", "first"),
        defteam=("defteam", "first"),
        lead=("lead", "first"),
        gsr=("gsr", "first"),
        res=("fixed_drive_result", "first"),
    )
)
dr = dr[dr.res.notna()].copy()
dr["score"] = dr.res.isin(["Touchdown", "Field goal"]).astype(float)
dr["td"] = (dr.res == "Touchdown").astype(float)
dr["dsn"] = dr.defteam + "|" + dr.season.astype(str)
dr = band(dr)

PM = {"ypp": ("y", pl), "expl": ("ex", pl), "score_rate": ("score", dr), "td_rate": ("td", dr)}


def base_resid(df, col, w, grp):
    k = pd.factorize(grp)[0]
    code = k * NT + df.tb.values
    nc = (k.max() + 1) * NT
    nb = df.lb.values == 0
    sb = np.bincount(code[nb], df[col].values[nb] * w[nb], minlength=nc)
    cb = np.bincount(code[nb], w[nb], minlength=nc)
    with np.errstate(all="ignore"):
        base = sb / cb
    return df[col].values - base[code]


def pooled_w(df, col, w):
    resid = base_resid(df, col, w, df.dsn.values)
    ok = np.isfinite(resid) & (w > 0)
    out = {}
    lb, tb = df.lb.values, df.tb.values
    for li in range(1, NL + 1):
        for ti in range(NT):
            m = (lb == li) & (tb == ti) & ok
            ws = w[m].sum()
            out[(li, ti)] = float((resid[m] * w[m]).sum() / ws) if ws > 0 else np.nan
    return out


res = {
    "explosive_threshold_yards": thr,
    "lead_edges": LE.tolist(),
    "secs_edges": TE.tolist(),
    "neutral_abs_diff_max": float(NEUTRAL),
    "states": {},
}
for name, (col, df) in PM.items():
    pt = pooled_w(df, col, np.ones(len(df)))
    gid = pd.factorize(df.game_id)[0]
    ng = gid.max() + 1
    bs = {}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng)[gid].astype(float)
        for kx, v in pooled_w(df, col, w).items():
            bs.setdefault(kx, []).append(v)
    for kx, v in pt.items():
        arr = np.array(bs[kx])
        arr = arr[np.isfinite(arr)]
        res["states"][f"{name}|lead{kx[0]}|t{kx[1]}"] = {
            "n": int(((df.lb == kx[0]) & (df.tb == kx[1])).sum()),
            "diff_vs_neutral": v,
            "ci95": np.percentile(arr, [2.5, 97.5]).tolist(),
            "p_pos": float((arr > 0).mean()),
        }
    print(name, "done", flush=True)

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[["game_id", "gameday"]]
tg = (
    d[["game_id", "season", "week", "defteam"]]
    .drop_duplicates()
    .rename(columns={"defteam": "team"})
    .merge(sch, on="game_id")
)
tg["t"] = (pd.to_datetime(tg.gameday, utc=True) + pd.Timedelta(days=1)).astype(
    "datetime64[ns, UTC]"
)
tg["season"] = tg.season.astype("int64")
ch = pd.read_parquet(sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1])
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
c = ch[ch.role == "DC"].sort_values("effective_observed_at").astype({"season": "int64"})
tg = tg.sort_values("t").reset_index(drop=True)
mm = pd.merge_asof(
    tg[["game_id", "team", "season", "t"]],
    c[["team", "season", "effective_observed_at", "person"]],
    left_on="t",
    right_on="effective_observed_at",
    by=["team", "season"],
    direction="backward",
)
first = c.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
    columns={"person": "fp"}
)
mm = mm.merge(first, on=["team", "season"], how="left")
tg["dc"] = mm.person.fillna(mm.fp).values
tg = tg.dropna(subset=["dc"]).sort_values(["season", "week"]).reset_index(drop=True)
tg["spell"] = tg.dc.astype("string") + "|" + tg.team
tg["par"] = tg.groupby("spell").cumcount() % 2
sz = tg.groupby("spell").size()
tg = tg[tg.spell.isin(sz[sz >= MIN_SPELL_GAMES].index)]
res["n_dc_spells"] = int(tg.spell.nunique())


def half_shift(df, col, ukey):
    j = df.merge(
        tg[["game_id", "team", "spell", "par"]],
        left_on=["game_id", "defteam"],
        right_on=["game_id", "team"],
    )
    j["u"] = j[ukey]
    kc = pd.factorize(j.spell.astype(str) + "|" + j.u.astype(str) + "|" + j.tb.astype(str))[0]
    nb = j.lb.values == 0
    sb = np.bincount(kc[nb], j[col].values[nb], minlength=kc.max() + 1)
    cb = np.bincount(kc[nb], minlength=kc.max() + 1)
    with np.errstate(all="ignore"):
        j["r"] = j[col].values - (sb / cb)[kc]
    sel = (j.lb.values >= REL_LEAD) & (j.tb.values <= REL_TIME_MAX) & np.isfinite(j.r.values)
    return j[sel].groupby(["spell", "u"]).r.mean().reset_index()


def reliability(df, col, mode):
    if mode == "oddeven":
        g = half_shift(df, col, "par")
        pm = g[g.u == 0].merge(g[g.u == 1], on="spell")
    else:
        g = half_shift(df, col, "season")
        nxt = g.copy()
        nxt["u"] = nxt.u - 1
        pm = g.merge(nxt, on=["spell", "u"])
    av, bv = pm.r_x.values, pm.r_y.values
    n = len(av)
    r = float(np.corrcoef(av, bv)[0, 1])
    bs = np.array(
        [np.corrcoef(av[i], bv[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    )
    sf = np.array([np.corrcoef(av * rng.choice([-1.0, 1.0], n), bv)[0, 1] for _ in range(N_SIGN)])
    return {
        "n_pairs": n,
        "r": r,
        "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
        "p_pos": float((bs > 0).mean()),
        "signflip_p_two_sided": float((np.sum(np.abs(sf) >= abs(r)) + 1) / (len(sf) + 1)),
        "mean_shift": float(np.mean((av + bv) / 2)),
    }


res["reliability"] = {}
for name, (col, df) in PM.items():
    for mode in ("oddeven", "t_t1"):
        res["reliability"][f"{name}|{mode}"] = reliability(df, col, mode)
        print(name, mode, res["reliability"][f"{name}|{mode}"], flush=True)

rows = []
for p in sorted(glob.glob("artifacts/sim10/instrument/instr_s11/*.jsonl")):
    with open(p) as fh:
        for ln in fh:
            r = json.loads(ln)
            sc = [0.0, 0.0]
            for hf, pos, _yl, secs, po, pd_, el, _tr in r["dr"]:
                if hf < 3:
                    rows.append(
                        (
                            sc[1 - int(pos)] - sc[int(pos)],
                            secs + (HALF_SECS if hf == 1 else 0.0),
                            po,
                            el,
                        )
                    )
                sc[int(pos)] += po
                sc[1 - int(pos)] += pd_
sim = pd.DataFrame(rows, columns=["lead", "gsr", "po", "el"])
sim = band(sim)
sim["score"] = (sim.po > 0).astype(float)
sim["td"] = (sim.po >= max(sim.po[(sim.po > 0) & (sim.po < 4)].max(), 0) + 1).astype(float)
cmp_ = {}
for col in ("score", "td"):
    for li in range(1, NL + 1):
        for ti in range(NT):
            out = {}
            for nm, df in (("real", dr), ("sim", sim)):
                a = df[(df.lb == li) & (df.tb == ti)][col].values
                b = df[(df.lb == 0) & (df.tb == ti)][col].values
                bt = [
                    a[rng.integers(0, len(a), len(a))].mean()
                    - b[rng.integers(0, len(b), len(b))].mean()
                    for _ in range(N_BOOT)
                ]
                out[nm] = {
                    "diff": float(a.mean() - b.mean()),
                    "ci95": np.percentile(bt, [2.5, 97.5]).tolist(),
                    "n": len(a),
                }
            cmp_[f"{col}|lead{li}|t{ti}"] = out
res["sim_vs_real_drive"] = cmp_
res["looks"] = {"state_cells": 4 * NL * NT, "reliability": 4 * 2, "sim_cells": 2 * NL * NT}
with open("artifacts/sim26/prevent_shape.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res["looks"]))

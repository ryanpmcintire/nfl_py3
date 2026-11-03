import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sim08_terminal"
EARLY = ROOT / "data" / "raw" / "nflverse_pbp_full" / "20261103T131922Z"
LATE = ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
CATS = ["td", "fg", "fg_miss", "turnover", "punt", "kneel", "def_score", "other"]
RESULT_CAT = {
    "Touchdown": "td",
    "Field goal": "fg",
    "Missed field goal": "fg_miss",
    "Turnover": "turnover",
    "Turnover on downs": "turnover",
    "Punt": "punt",
    "Opp touchdown": "def_score",
    "Safety": "def_score",
    "End of half": "other",
}
SCRIMMAGE = {"run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"}
WINDOW_SECS = 300
BIN_SECS = 60
BIN_EDGES = (300, 240, 180, 120, 60, 0)
NBIN = len(BIN_EDGES) - 1
KEY_MARGIN_MAX = 17
OT_MARGIN_MAX = 8
OT_POSS_MAX = 2
DELTA_MAX = 9
DELTA_N = 2 * DELTA_MAX + 1
OT_SECS_OLD = 900
OT_SECS_NEW = 600
OT_NEW_FROM = 2017
RULE_SUDDEN_END = 2011
RULE_BOTH_FROM = 2025
TD_MIN_POINTS = 6
SEASONS = list(range(2009, 2026))
COLS = [
    "game_id",
    "season",
    "season_type",
    "qtr",
    "game_seconds_remaining",
    "fixed_drive",
    "fixed_drive_result",
    "posteam",
    "home_team",
    "posteam_score",
    "posteam_score_post",
    "score_differential",
    "play_type",
    "qb_kneel",
]
PATHS = ["reg_fg_from_tie", "ot_fg", "loser_trims_10", "winner_td_from_minus4", "other"]


def load_season(season):
    if season <= 2017:
        path = EARLY / f"pbp_{season}.parquet"
    else:
        path = LATE / f"season={season}" / "plays.parquet"
    d = pd.read_parquet(path)
    if "season" not in d.columns:
        d["season"] = season
    d = d[d["season_type"] == "REG"]
    return d[[c for c in COLS if c in d.columns]].copy()


def bin_of(secs):
    return np.clip(((WINDOW_SECS - np.asarray(secs)) // BIN_SECS).astype(int), 0, NBIN - 1)


def end_diff(frame, home):
    q = frame.iloc[-1]
    d_post = q["posteam_score_post"] - (q["posteam_score"] - q["score_differential"])
    return float(d_post * (1.0 if q["posteam"] == home else -1.0))


def drives_of(d):
    rows = []
    for gid, g in d.groupby("game_id", sort=False):
        post = g[
            g["posteam"].notna() & g["posteam_score_post"].notna() & g["score_differential"].notna()
        ]
        home = g["home_team"].iloc[0]

        reg_end = end_diff(post[post["qtr"] <= 4], home)
        final = end_diff(post, home)
        sc = g[g["posteam"].notna() & g["play_type"].isin(SCRIMMAGE) & g["fixed_drive"].notna()]
        gb = sc.groupby("fixed_drive", sort=True)
        first = gb.first()
        kneel = gb["qb_kneel"].max().fillna(0).astype(bool)
        drive_last_s = gb["game_seconds_remaining"].min().to_numpy(dtype=float)
        n = len(first)
        if n == 0:
            continue
        sign = np.where(first["posteam"] == home, 1.0, -1.0)
        mh = first["score_differential"].to_numpy() * sign
        secs = first["game_seconds_remaining"].to_numpy(dtype=float)
        qtr = first["qtr"].to_numpy()
        res = first["fixed_drive_result"].to_numpy()
        season = int(first["season"].iloc[0])
        for i in range(n):
            ot = bool(qtr[i] >= 5)
            nxt = i + 1 < n and bool(qtr[i + 1] >= 5) == ot
            if nxt:
                end_mh = mh[i + 1]
                end_s = secs[i + 1]
            else:
                end_mh = final if ot else reg_end
                end_s = drive_last_s[i] if ot else 0.0
            cat = RESULT_CAT.get(res[i], "other")
            if cat == "other" and kneel.iloc[i]:
                cat = "kneel"
            rows.append(
                (
                    gid,
                    season,
                    i,
                    ot,
                    float(secs[i]),
                    float(max(secs[i] - end_s, 0.0)),
                    float(sign[i]),
                    float(mh[i]),
                    float(end_mh),
                    cat,
                    final,
                )
            )
    cols = [
        "game_id",
        "season",
        "k",
        "ot",
        "s",
        "elapsed",
        "pos",
        "mh",
        "end_mh",
        "cat",
        "final_mh",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out["delta"] = (out["end_mh"] - out["mh"]) * out["pos"]
    out["m"] = out["mh"] * out["pos"]
    return out


def all_drives():
    path = OUT / "drives.parquet"
    if path.exists():
        return pd.read_parquet(path)
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat([drives_of(load_season(s)) for s in SEASONS], ignore_index=True)
    out.to_parquet(path)
    return out


def margin_idx(m, lim):
    return np.clip(np.rint(m), -lim, lim).astype(int) + lim


def delta_idx(x):
    return np.clip(np.rint(x), -DELTA_MAX, DELTA_MAX).astype(int) + DELTA_MAX


def fit_table(dr):
    reg = dr[(~dr["ot"]) & (dr["s"] <= WINDOW_SECS)].copy()
    reg["bin"] = bin_of(reg["s"].to_numpy())
    reg["mi"] = margin_idx(reg["m"].to_numpy(), KEY_MARGIN_MAX)
    reg["di"] = delta_idx(reg["delta"].to_numpy())
    nm = 2 * KEY_MARGIN_MAX + 1
    cnt = np.zeros((NBIN, nm, len(CATS)))
    for (b, mi, c), n in reg.groupby(["bin", "mi", "cat"]).size().items():
        cnt[b, mi, CATS.index(c)] = n
    dcnt = np.zeros((nm, len(CATS), DELTA_N))
    for (mi, c, di), n in reg.groupby(["mi", "cat", "di"]).size().items():
        dcnt[mi, CATS.index(c), di] = n
    el = {}
    for (b, c), v in reg.groupby(["bin", "cat"])["elapsed"]:
        el[f"{b}|{c}"] = [float(x) for x in v]
    ot = dr[dr["ot"]].copy()
    ot["pk"] = np.minimum(ot["k"] - ot.groupby("game_id")["k"].transform("min"), OT_POSS_MAX)
    ot["mi"] = margin_idx(ot["m"].to_numpy(), OT_MARGIN_MAX)
    ot["di"] = delta_idx(ot["delta"].to_numpy())
    nmo = 2 * OT_MARGIN_MAX + 1
    ocnt = np.zeros((OT_POSS_MAX + 1, nmo, len(CATS)))
    for (pk, mi, c), n in ot.groupby(["pk", "mi", "cat"]).size().items():
        ocnt[pk, mi, CATS.index(c)] = n
    odc = np.zeros((nmo, len(CATS), DELTA_N))
    for (mi, c, di), n in ot.groupby(["mi", "cat", "di"]).size().items():
        odc[mi, CATS.index(c), di] = n
    oel = {c: [float(x) for x in v] for c, v in ot.groupby("cat")["elapsed"]}
    return {
        "cats": CATS,
        "bin_edges": list(BIN_EDGES),
        "key_margin_max": KEY_MARGIN_MAX,
        "reg_counts": cnt.tolist(),
        "reg_delta_counts": dcnt.tolist(),
        "reg_elapsed": el,
        "ot_margin_max": OT_MARGIN_MAX,
        "ot_counts": ocnt.tolist(),
        "ot_delta_counts": odc.tolist(),
        "ot_elapsed": oel,
        "n_reg_drives": len(reg),
        "n_ot_drives": len(ot),
    }


def cum_probs(cnt):
    nm_axis = cnt.ndim - 2
    c = cnt.copy()
    tot = c.sum(axis=-1)
    for ix in np.argwhere(tot == 0):
        pre = tuple(ix[:nm_axis])
        pool = np.moveaxis(tot, nm_axis, -1)[tuple(ix[:nm_axis])]
        ok = np.nonzero(pool > 0)[0]
        if len(ok) == 0:
            continue
        j = ok[np.argmin(np.abs(ok - ix[nm_axis]))]
        src = (*pre, j)
        c[(*pre, ix[nm_axis])] = cnt[src]
    s = c.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return np.cumsum(c / s, axis=-1)


def pick(cp, u):
    return np.minimum((u[:, None] > cp).sum(axis=1), cp.shape[-1] - 1)


def draw_delta(dcnt, mi, ci, rng):
    out = np.zeros(len(mi))
    for j in range(len(mi)):
        c = dcnt[mi[j], ci[j]]
        if c.sum() == 0:
            c = dcnt[:, ci[j]].sum(axis=0)
        if c.sum() > 0:
            out[j] = rng.choice(DELTA_N, p=c / c.sum()) - DELTA_MAX
    return out


def draw_elapsed(pool, key, rng):
    arr = pool.get(key)
    if arr is None or len(arr) == 0:
        arr = np.concatenate(list(pool.values()))
    return float(arr[rng.integers(0, len(arr))])


class Table:
    def __init__(self, t):
        self.reg_cp = cum_probs(np.array(t["reg_counts"]))
        self.reg_d = np.array(t["reg_delta_counts"])
        self.ot_cp = cum_probs(np.array(t["ot_counts"]))
        self.ot_d = np.array(t["ot_delta_counts"])
        self.el = {k: np.array(v) for k, v in t["reg_elapsed"].items()}
        self.oel = {k: np.array(v) for k, v in t["ot_elapsed"].items()}


def ot_one(tab, season, mh, pos, rec, j, rng):
    s = float(OT_SECS_OLD if season < OT_NEW_FROM else OT_SECS_NEW)
    kk = 0
    while True:
        m = mh[j] * pos[j]
        mi = margin_idx(np.array([m]), OT_MARGIN_MAX)
        ci = pick(tab.ot_cp[min(kk, OT_POSS_MAX), mi[0]][None, :], rng.random(1))
        dl = draw_delta(tab.ot_d, mi, ci, rng)[0]
        el = min(draw_elapsed(tab.oel, CATS[ci[0]], rng), s)
        mh[j] += dl * pos[j]
        s -= el
        if dl != 0:
            rec[j] = (m, dl, True)
        pos[j] = -pos[j]
        kk += 1
        if s <= 0 or dl < 0:
            return
        if season <= RULE_SUDDEN_END:
            if mh[j] != 0:
                return
        elif kk == 1:
            if season < RULE_BOTH_FROM and dl >= TD_MIN_POINTS:
                return
        elif mh[j] != 0:
            return


def simulate_game(tab, season, start, rng, draws):
    mh = np.full(draws, start["mh"])
    pos = np.full(draws, start["pos"])
    s = np.full(draws, start["s"])
    alive = np.ones(draws, bool)
    rec = [(start["pre"], start["pdelta"], False)] * draws
    while alive.any():
        ix = np.nonzero(alive)[0]
        m = mh[ix] * pos[ix]
        b = bin_of(s[ix])
        mi = margin_idx(m, KEY_MARGIN_MAX)
        ci = pick(tab.reg_cp[b, mi], rng.random(len(ix)))
        dl = draw_delta(tab.reg_d, mi, ci, rng)
        el = np.array([draw_elapsed(tab.el, f"{b[j]}|{CATS[ci[j]]}", rng) for j in range(len(ix))])
        el = np.minimum(el, s[ix])
        mh[ix] = mh[ix] + dl * pos[ix]
        s[ix] = s[ix] - el
        for j in np.nonzero(dl != 0)[0]:
            rec[ix[j]] = (m[j], dl[j], False)
        pos[ix] = -pos[ix]
        alive[ix[s[ix] <= 0]] = False
    ot = mh == 0
    for j in np.nonzero(ot)[0]:
        ot_one(tab, season, mh, pos, rec, j, rng)
    return mh, ot, rec


def path_of(pre, delta, ot, final_abs):
    if final_abs != 3 or np.isnan(pre):
        return "other"
    if ot:
        return "ot_fg" if abs(delta) == 3 else "other"
    if pre == 0 and delta == 3:
        return "reg_fg_from_tie"
    if pre == -10 and delta == 7:
        return "loser_trims_10"
    if pre == -4 and delta == 7:
        return "winner_td_from_minus4"
    return "other"


def starts_of(ds):
    out = []
    for _gid, g in ds.groupby("game_id", sort=False):
        reg = g[~g["ot"]]
        w = reg[reg["s"] <= WINDOW_SECS]
        if len(w) == 0:
            continue
        r = w.iloc[0]
        sc = reg[(reg["k"] < r["k"]) & (reg["delta"] != 0)]
        pre, pdelta = (np.nan, 0.0)
        if len(sc):
            pre, pdelta = float(sc.iloc[-1]["m"]), float(sc.iloc[-1]["delta"])
        sc_all = g[g["delta"] != 0]
        rp, rd, ro = (np.nan, 0.0, False)
        if len(sc_all):
            q = sc_all.iloc[-1]
            rp, rd, ro = float(q["m"]), float(q["delta"]), bool(q["ot"])
        out.append(
            {
                "mh": float(r["mh"]),
                "pos": float(r["pos"]),
                "s": float(r["s"]),
                "final": float(r["final_mh"]),
                "ot": bool(g["ot"].any()),
                "pre": pre,
                "pdelta": pdelta,
                "real_path": path_of(rp, rd, ro, abs(float(r["final_mh"]))),
            }
        )
    return out


def cmd_fit(a):
    dr = all_drives()
    for s in SEASONS:
        t = fit_table(dr[dr["season"] != s])
        t["held_out_season"] = s
        (OUT / f"transitions_{s}.json").write_text(json.dumps(t))
        print(s, t["n_reg_drives"], t["n_ot_drives"], flush=True)


def cmd_check(a):
    dr = all_drives()
    rng = np.random.default_rng(a.seed)
    res = {}
    tr = dict.fromkeys(PATHS, 0)
    ts = dict.fromkeys(PATHS, 0.0)
    n3r = 0
    n3s = 0.0
    ng = 0
    otr = [0, 0]
    ots = [0.0, 0.0]
    for s in SEASONS:
        tab = Table(json.loads((OUT / f"transitions_{s}.json").read_text()))
        st = starts_of(dr[dr["season"] == s])
        pr = dict.fromkeys(PATHS, 0)
        ps = dict.fromkeys(PATHS, 0.0)
        r3 = 0
        s3 = 0.0
        for g in st:
            if abs(g["final"]) == 3:
                pr[g["real_path"]] += 1
                r3 += 1
            otr[1] += int(g["ot"])
            otr[0] += int(g["ot"] and g["final"] == 0)
            mh, otg, rec = simulate_game(tab, s, g, rng, a.draws)
            for d in range(a.draws):
                if abs(mh[d]) == 3:
                    s3 += 1.0 / a.draws
                    ps[path_of(rec[d][0], rec[d][1], rec[d][2], 3)] += 1.0 / a.draws
            ots[1] += otg.sum() / a.draws
            ots[0] += (otg & (mh == 0)).sum() / a.draws
        n = len(st)
        res[s] = {
            "games": n,
            "real_mass3": r3 / n,
            "sim_mass3": s3 / n,
            "real_paths": pr,
            "sim_paths": ps,
        }
        print(s, n, round(r3 / n, 4), round(s3 / n, 4), flush=True)
        ng += n
        n3r += r3
        n3s += s3
        for p in PATHS:
            tr[p] += pr[p]
            ts[p] += ps[p]
    res["pooled"] = {
        "games": ng,
        "real_mass3": n3r / ng,
        "sim_mass3": n3s / ng,
        "real_paths": tr,
        "sim_paths": ts,
        "real_ot_games": otr[1],
        "real_ot_tie": otr[0] / max(otr[1], 1),
        "sim_ot_games": ots[1],
        "sim_ot_tie": ots[0] / max(ots[1], 1e-9),
    }
    (OUT / "reproduction.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["pooled"], indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    c = sub.add_parser("check")
    c.add_argument("--draws", type=int, default=100)
    c.add_argument("--seed", type=int, default=20261103)
    a = p.parse_args()
    {"fit": cmd_fit, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()

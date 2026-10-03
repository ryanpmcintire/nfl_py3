import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e3" / "late"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
SIMD = REPO / "artifacts" / "mod25e3" / "revert_sim" / "crzhc"
POOL = tuple(range(2009, 2018))
LATE = tuple(range(2018, 2026))
BURN = int(os.environ.get("KICK_BURN", 2))
NB = 400
LIVE = ("run", "pass", "punt", "field_goal")


def load_xq():
    src = (REPO / "scripts" / "mod25e_xq.py").read_text().rsplit("\ncommon()", 1)[0]
    ns = {"__name__": "mod25e_xq_lib", "__file__": str(REPO / "scripts" / "mod25e_xq.py")}
    exec(compile(src, "mod25e_xq_lib", "exec"), ns)
    return ns


def finish(D):
    g = D.gk.to_numpy()
    same = np.r_[g[1:] == g[:-1], False]
    off = D.off.to_numpy()
    nxt_off = np.r_[off[1:], off[-1:]]
    D["chg"] = same & (nxt_off != off)
    D["ret"] = same & (nxt_off == off)
    for c in ("p", "yl0", "gsr0", "sd0", "ylast"):
        D["n_" + c] = np.r_[D[c].to_numpy()[1:], np.nan]
    D["burn"] = np.where(same, D.gsr0 - D.n_gsr0, D.gsr0)
    p = D.p.to_numpy()
    lp = D.lastpt.to_numpy()
    e = np.full(len(D), "end", dtype=object)
    e[D.chg.to_numpy() | D.ret.to_numpy()] = "other"
    e[(D.tolast.to_numpy() == 1) & D.chg.to_numpy()] = "to"
    e[(D.tolast.to_numpy() == 0) & D.chg.to_numpy() & (D.dlast.to_numpy() == 4)] = "downs"
    e[lp == "field_goal"] = "fgmiss"
    e[lp == "punt"] = "punt"
    e[p < 0] = "defscore"
    e[(p > 0) & (lp == "field_goal")] = "fg"
    e[(p > 0) & (lp != "field_goal")] = "td"
    D["end"] = e
    D["fgm"] = (e == "fg").astype(int)
    return D


def real_play_drives(seasons):
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "first_down", "qb_kneel", "qb_spike", "interception", "fumble_lost", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "yards_gained"]
    out = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p["season"] = s
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4)]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        p["m"] = np.where(p.posteam == p.home_team, 1.0, -1.0) * p.score_differential
        fin = (gf.home_score - gf.away_score).reindex(p.game_id).to_numpy()
        g = p.game_id.to_numpy()
        last = np.r_[g[1:] != g[:-1], True]
        nxt = np.r_[p.m.to_numpy()[1:], 0.0]
        p["dm"] = np.where(last, fin - p.m.to_numpy(), nxt - p.m.to_numpy())
        p = p[p.dm.notna()]
        ko = (p.play_type == "kickoff").to_numpy()
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        g = p.game_id.to_numpy()
        pos = p.posteam.to_numpy()
        kc = p.kocum.to_numpy()
        q = p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        lv = p.play_type.isin(LIVE)
        kn = (p.qb_kneel == 1) | (p.qb_spike == 1)
        p["is_run"] = (lv & (p.play_type == "run") & ~kn).astype(int)
        p["is_pass"] = (lv & (p.play_type == "pass") & ~kn).astype(int)
        p["is_kneel"] = (p.qb_kneel == 1).astype(int)
        p["is_spike"] = (p.qb_spike == 1).astype(int)
        p["fga"] = (p.play_type == "field_goal").astype(int)
        p["fd"] = (p.first_down == 1).astype(int)
        p["to"] = ((p.interception == 1) | (p.fumble_lost == 1)).astype(int)
        p["lyl"] = p.yardline_100
        gl = p[lv].groupby("d", sort=True)
        gp = p.groupby("d", sort=True)
        first_pos = gp.posteam.first()
        D = pd.DataFrame({"gk": gp.season.first().astype(str) + "_" + gp.game_id.first(), "season": gp.season.first(), "off": first_pos, "dfn": gp.defteam.first(),
                          "s": np.where(first_pos == gp.home_team.first(), 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yardline_100.first(), "gsr0": gp.game_seconds_remaining.first(),
                          "n": gp.size(), "dmsum": gp.dm.sum(), "sd0": gp.score_differential.first(), "tos": gp.posteam_timeouts_remaining.first(), "dtos": gp.defteam_timeouts_remaining.first(),
                          "run": gp.is_run.sum(), "pas": gp.is_pass.sum(), "kneel": gp.is_kneel.sum(), "spike": gp.is_spike.sum(), "fga": gp.fga.sum(), "fd": gp.fd.sum(),
                          "tolast": gp.to.max(), "yds": gp.yards_gained.sum()})
        D["lastpt"] = gl.play_type.last().reindex(D.index)
        D["dlast"] = gl.down.last().reindex(D.index)
        D["ylast"] = gl.lyl.last().reindex(D.index)
        D["p"] = D.s * D.dmsum
        D["tm"] = D.season.astype(str) + "_" + D.off
        D["td"] = D.season.astype(str) + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return finish(pd.concat(out, ignore_index=True))


def sim_play_drives():
    out = []
    for f in sorted(SIMD.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
        g = d.g.to_numpy()
        oh = d.offhome.to_numpy()
        q = d.qtr.to_numpy()
        sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
        d["d"] = np.cumsum(new)
        d["pp"] = d.po - d.pdf
        d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
        d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
        c = d.code
        d["is_run"] = (c == 0).astype(int)
        d["is_pass"] = (c == 1).astype(int)
        d["is_kneel"] = (c == 4).astype(int)
        d["is_spike"] = (c == 5).astype(int)
        d["fga"] = (c == 3).astype(int)
        d["fd"] = (d.down == 1).astype(int)
        d["yds"] = d.yards.fillna(0)
        lv = c.isin([0, 1, 2, 3])
        gp = d.groupby("d", sort=True)
        gl = d[lv].groupby("d", sort=True)
        key = w * 1000 + s
        D = pd.DataFrame({"gk": f"{key}_" + gp.g.first().astype(int).astype(str), "season": key, "off": gp.off.first(), "dfn": gp.dfn.first(),
                          "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "gsr0": gp.gsr.first(), "n": gp.size(),
                          "sd0": gp.sd.first(), "p": gp.pp.sum(), "run": gp.is_run.sum(), "pas": gp.is_pass.sum(), "kneel": gp.is_kneel.sum(), "spike": gp.is_spike.sum(),
                          "fga": gp.fga.sum(), "fd": gp.fd.sum() - 1, "yds": gp.yds.sum()})
        D["fd"] = D.fd.clip(lower=0)
        lastc = gl.code.last().reindex(D.index)
        D["lastpt"] = np.select([lastc == 2, lastc == 3], ["punt", "field_goal"], "run")
        D["dlast"] = gl.down.last().reindex(D.index)
        D["ylast"] = gl.yl.last().reindex(D.index)
        flipl = gp.flip.last()
        D["tolast"] = ((flipl == 1) & (lastc.isin([0, 1]))).astype(int)
        D["tos"] = np.nan
        D["dtos"] = np.nan
        D["tm"] = D.season.astype(str) + "_" + D.off.astype(int).astype(str)
        D["td"] = D.season.astype(str) + "_" + D.dfn.astype(int).astype(str)
        out.append(D)
    return finish(pd.concat(out, ignore_index=True))


def lead_rows(D):
    sel = (D.ql.to_numpy() == 4) & (D.sd0.to_numpy() < 0) & (D.p.to_numpy() > 0) & (D.gsr0.to_numpy() < 300) & D.chg.to_numpy()
    return D.iloc[np.flatnonzero(sel) + 1]


def summarize(L, name, R):
    n = len(R)
    L.append(f"== {name}: leader drive after late trailer score, n {n}")
    L.append(f"  P(score) {(R.p > 0).mean():.3f}  P(td) {(R.end == 'td').mean():.3f} P(fg) {(R.end == 'fg').mean():.3f} mean p {R.p.mean():.3f}")
    L.append(f"  start yl0 mean {R.yl0.mean():.1f} median {R.yl0.median():.0f}, start gsr {R.gsr0.mean():.0f}, leader lead at start mean {R.sd0.mean():.2f}, leader now behind/tied {(R.sd0 <= 0).mean():.3f}")
    ends = R.end.value_counts(normalize=True)
    L.append("  end: " + " ".join(f"{k} {v:.3f}" for k, v in ends.items()))
    L.append(f"  plays {R.n.mean():.2f} run share {R.run.sum() / max((R.run + R.pas).sum(), 1):.3f} kneel drives {(R.kneel > 0).mean():.3f} spike {(R.spike > 0).mean():.3f} first downs {R.fd.mean():.2f} yds/drive {R.yds.mean():.1f} yds/play {R.yds.sum() / max(R.n.sum(), 1):.2f}")
    L.append(f"  FG att/drive {R.fga.mean():.3f} makes {R.fgm.mean():.3f} make rate {R.fgm.sum() / max(R.fga.sum(), 1):.3f}; clock burned {R.burn.mean():.0f}s")
    if R.tos.notna().any():
        L.append(f"  leader timeouts left {R.tos.mean():.2f}, trailer timeouts left {R.dtos.mean():.2f}")
    for nm, m in (("leader still ahead", R.sd0 > 0), ("leader now behind/tied", R.sd0 <= 0)):
        r = R[m.to_numpy()]
        if len(r):
            L.append(f"  [{nm}] n {len(r)} P(score) {(r.p > 0).mean():.3f} yl0 {r.yl0.mean():.1f} plays {r.n.mean():.1f} run share {r.run.sum() / max((r.run + r.pas).sum(), 1):.3f} kneel drives {(r.kneel > 0).mean():.3f} fg/drv {r.fga.mean():.3f} burn {r.burn.mean():.0f}")
    bins = [0, 40, 60, 80, 101]
    L.append("  P(score) by start yl0 " + " ".join(f"[{a},{b}) n {int(((R.yl0 >= a) & (R.yl0 < b)).sum())} {(R[(R.yl0 >= a) & (R.yl0 < b)].p > 0).mean():.3f}" for a, b in zip(bins[:-1], bins[1:])))


def standardize(Rs, Rr, L):
    bins = np.array([0, 20, 40, 60, 80, 101])
    br = np.digitize(Rr.yl0, bins) - 1
    bs = np.digitize(Rs.yl0, bins) - 1
    w = np.bincount(br, minlength=len(bins) - 1) / len(Rr)
    rates = np.array([(Rs[bs == b].p > 0).mean() if (bs == b).any() else np.nan for b in range(len(bins) - 1)])
    ok = ~np.isnan(rates)
    L.append(f"  sim P(score) at the real start mix {np.nansum(w[ok] * rates[ok]) / w[ok].sum():.3f} vs sim own {(Rs.p > 0).mean():.3f}; real {(Rr.p > 0).mean():.3f}")


def boot_gap(Rr, Rs, col, rng):
    a = (Rr[col] > 0).to_numpy(float) if col == "p" else Rr[col].to_numpy(float)
    b = (Rs[col] > 0).to_numpy(float) if col == "p" else Rs[col].to_numpy(float)
    g = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(NB)])
    return a.mean() - b.mean(), np.percentile(g, 2.5), np.percentile(g, 97.5), (g > 0).mean()


def handoff(L, name, D, xq):
    D, _ = xq["components"](D)
    r = D.c_rest.to_numpy()
    ok = D.chg.to_numpy() & (D.p.to_numpy() == 0)
    L.append(f"== {name}: next drive after non-score handoff (n {int(ok.sum())})")
    nx = np.r_[r[1:], 0.0]
    nyl = D.n_yl0.to_numpy()
    npts = D.n_p.to_numpy()
    for e in ("punt", "to", "downs", "fgmiss"):
        m = ok & (D.end.to_numpy() == e)
        if not m.any():
            continue
        yl = D.yl0.to_numpy()[m]
        net = yl + nyl[m] - 100
        L.append(f"  {e:6s} n {int(m.sum()):5d} share {m.sum() / ok.sum():.3f} next yl0 {np.nanmean(nyl[m]):.1f} (sd {np.nanstd(nyl[m]):.1f}) next pts {np.nanmean(npts[m]):+.3f} P(next scores) {np.nanmean(npts[m] > 0):.3f} pair resid prod {np.nanmean(r[m] * nx[m]):+.4f}")
        if e == "punt":
            L.append(f"         net {np.nanmean(net):.1f} sd {np.nanstd(net):.1f}; by punter yl0 <50 {np.nanmean(net[yl < 50]):.1f} 50-70 {np.nanmean(net[(yl >= 50) & (yl < 70)]):.1f} 70+ {np.nanmean(net[yl >= 70]) if (yl >= 70).any() else float('nan'):.1f}; touchback share {np.nanmean(nyl[m] == 80):.3f}")
        if e in ("to", "downs"):
            sw = D.ylast.to_numpy()[m] + nyl[m] - 100
            L.append(f"         swing vs spot mean {np.nanmean(sw):.2f} sd {np.nanstd(sw):.2f}; own yl at last play {np.nanmean(D.ylast.to_numpy()[m]):.1f}")
    return D


def state_table(L, name, D):
    q4 = (D.ql.to_numpy() == 4) & (D.gsr0.to_numpy() < 300)
    sd = D.sd0.to_numpy()
    cells = (("trail 1-8", (sd < 0) & (sd >= -8)), ("trail >8", sd < -8), ("tied", sd == 0), ("lead 1-8", (sd > 0) & (sd <= 8)), ("lead >8", sd > 8))
    L.append(f"== {name}: all drives starting in the last 5 min of Q4 by offence score state")
    for nm, m in cells:
        r = D[q4 & m]
        if len(r) < 20:
            continue
        L.append(f"  {nm:9s} n {len(r):5d} P(score) {(r.p > 0).mean():.3f} td {(r.end == 'td').mean():.3f} fg {(r.end == 'fg').mean():.3f} plays {r.n.mean():.2f} fd {r.fd.mean():.2f} yds/play {r.yds.sum() / max(r.n.sum(), 1):.2f} run share {r.run.sum() / max((r.run + r.pas).sum(), 1):.3f} kneel {(r.kneel > 0).mean():.3f} fga {r.fga.mean():.3f} yl0 {r.yl0.mean():.1f} gsr0 {r.gsr0.mean():.0f} burn {r.burn.mean():.0f} punt {(r.end == 'punt').mean():.3f} to {(r.end.isin(['to', 'downs'])).mean():.3f}")
    r = D[(D.ql.to_numpy() <= 3) | (D.gsr0.to_numpy() > 900)]
    L.append(f"  baseline (before last 15 min) n {len(r)} P(score) {(r.p > 0).mean():.3f} mean p {r.p.mean():.3f} punt {(r.end == 'punt').mean():.3f} to+downs {(r.end.isin(['to', 'downs'])).mean():.3f}")
    L.append(f"  all drives P(score) {(D.p > 0).mean():.3f} mean p {D.p.mean():.3f}")
    bins = [0, 40, 60, 80, 101]
    nxt = D.n_p.to_numpy()
    nyl = D.n_yl0.to_numpy()
    for e in ("punt", "to", "downs", "fgmiss"):
        m = D.chg.to_numpy() & (D.p.to_numpy() == 0) & (D.end.to_numpy() == e)
        out = []
        for a, b in zip(bins[:-1], bins[1:]):
            k = m & (nyl >= a) & (nyl < b)
            out.append(f"[{a},{b}) n {int(k.sum())} {np.nanmean(nxt[k] > 0) if k.any() else float('nan'):.3f}")
        L.append(f"  next drive P(score) after {e} by receiver start yl0: " + " ".join(out))
    a_ = D.yl0.to_numpy()
    out = []
    for x, y in zip(bins[:-1], bins[1:]):
        k = (a_ >= x) & (a_ < y)
        out.append(f"[{x},{y}) {(D.p.to_numpy()[k] > 0).mean():.3f}")
    L.append("  any drive P(score) by yl0: " + " ".join(out))


def cmd_state(a):
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    state_table(L, "sim crzhc", sim_play_drives())
    for nm, seas in (("real pool 2009-17", POOL), ("real 2018-25", LATE)):
        state_table(L, nm, real_play_drives(seas))
    (OUT / "state.txt").write_text("\n".join(L))
    print("\n".join(L))


def fourth_plays_real(seasons):
    out = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=["season_type", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike"])
        p = p[(p.season_type == "REG") & (p.down == 4) & p.score_differential.notna() & (p.qtr <= 4) & p.play_type.isin(["run", "pass", "punt", "field_goal"]) & (p.qb_kneel != 1) & (p.qb_spike != 1)]
        out.append(pd.DataFrame({"sd": p.score_differential, "gsr": p.game_seconds_remaining, "qtr": p.qtr, "yl": p.yardline_100, "dist": p.ydstogo, "a": np.select([p.play_type == "punt", p.play_type == "field_goal"], [2, 3], 1)}))
    return pd.concat(out, ignore_index=True)


def fourth_plays_sim():
    out = []
    for f in sorted(SIMD.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[(d.qtr <= 4) & (d.down == 4) & d.code.isin([0, 1, 2, 3])]
        out.append(pd.DataFrame({"sd": d.sd, "gsr": d.gsr, "qtr": d.qtr, "yl": d.yl, "dist": d.dist, "a": np.select([d.code == 2, d.code == 3], [2, 3], 1)}))
    return pd.concat(out, ignore_index=True)


def fourth_table(L, name, F, n_games):
    L.append(f"== {name}: 4th-down actions (go/punt/fg shares) by offence score state, time, yl; per-game 4ths {len(F) / n_games:.2f}")
    sd = F.sd.to_numpy()
    zt = (("last5", (F.qtr == 4) & (F.gsr < 300)), ("5-15", (F.qtr == 4) & (F.gsr >= 300) & (F.gsr < 900)), ("rest", ~((F.qtr == 4) & (F.gsr < 900))))
    zs = (("trail>8", sd < -8), ("trail1-8", (sd < 0) & (sd >= -8)), ("tied", sd == 0), ("lead1-8", (sd > 0) & (sd <= 8)), ("lead>8", sd > 8))
    zy = (("fgrange yl<=37", F.yl <= 37), ("mid 37-55", (F.yl > 37) & (F.yl <= 55)), ("own>55", F.yl > 55))
    for tn, tm in zt:
        for sn, sm in zs:
            for yn, ym in zy:
                r = F[tm & sm & ym]
                if len(r) >= 25:
                    L.append(f"  {tn:5s} {sn:8s} {yn:15s} n {len(r):5d} go {(r.a == 1).mean():.3f} punt {(r.a == 2).mean():.3f} fg {(r.a == 3).mean():.3f} dist {r.dist.mean():.1f}")


def cmd_fourth(a):
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    Fs = fourth_plays_sim()
    ngs = len(list(SIMD.glob("play_*_*.parquet"))) - BURN * len({int(f.stem.split("_")[1]) for f in SIMD.glob("play_*_*.parquet")})
    fourth_table(L, "sim crzhc", Fs, max(ngs, 1) * 272)
    for nm, seas in (("real pool 2009-17", POOL), ("real 2018-25", LATE)):
        fourth_table(L, nm, fourth_plays_real(seas), len(seas) * 272)
    (OUT / "fourth.txt").write_text("\n".join(L))
    print("\n".join(L))


def clock_table(L, name, D):
    q4 = (D.ql.to_numpy() == 4) & (D.gsr0.to_numpy() < 300)
    sd = D.sd0.to_numpy()
    L.append(f"== {name}: late Q4 drives where offence is tied or trailing by <=8, split by drive start time and kneel")
    for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 300)):
        m = q4 & (sd <= 0) & (sd >= -8) & (D.gsr0.to_numpy() >= lo) & (D.gsr0.to_numpy() < hi)
        r = D[m]
        if len(r) < 20:
            continue
        sh = " ".join(f"{k} {(r.end == k).mean():.3f}" for k in ("td", "fg", "fgmiss", "punt", "to", "downs", "end"))
        L.append(f"  gsr0 [{lo},{hi}) n {len(r)} P(score) {(r.p > 0).mean():.3f} fga {r.fga.mean():.3f} kneel {(r.kneel > 0).mean():.3f} plays {r.n.mean():.2f} burn {r.burn.mean():.0f} | {sh}")
    m = q4 & (sd == 0)
    r = D[m & (D.kneel.to_numpy() > 0)]
    k = D[m & (D.kneel.to_numpy() == 0)]
    L.append(f"  tied with kneel: n {len(r)} fga {r.fga.mean():.3f} P(score) {(r.p > 0).mean():.3f} end-share {(r.end == 'end').mean():.3f}; tied no kneel: n {len(k)} fga {k.fga.mean():.3f} P(score) {(k.p > 0).mean():.3f} end-share {(k.end == 'end').mean():.3f}")
    m2 = q4 & (sd >= -3) & (sd <= 0) & (D.gsr0.to_numpy() < 120)
    r = D[m2]
    L.append(f"  trailing 0-3 starting under 2:00: n {len(r)} fga {r.fga.mean():.3f} fgmake {(r.end == 'fg').mean():.3f} td {(r.end == 'td').mean():.3f} end-share {(r.end == 'end').mean():.3f} mean plays {r.n.mean():.2f} yds/play {r.yds.sum() / max(r.n.sum(), 1):.2f} first downs {r.fd.mean():.2f} yl0 {r.yl0.mean():.1f}")


def cmd_clock(a):
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    clock_table(L, "sim crzhc", sim_play_drives())
    for nm, seas in (("real pool 2009-17", POOL), ("real 2018-25", LATE)):
        clock_table(L, nm, real_play_drives(seas))
    (OUT / "clock.txt").write_text("\n".join(L))
    print("\n".join(L))


def fgdown_real(seasons):
    out = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=["season_type", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "qb_kneel", "qb_spike", "game_id"])
        p = p[(p.season_type == "REG") & p.score_differential.notna() & (p.qtr == 4) & (p.game_seconds_remaining < 120) & (p.score_differential <= 0) & (p.score_differential >= -3) & p.play_type.isin(["run", "pass", "punt", "field_goal"]) & (p.yardline_100 <= 45)]
        out.append(pd.DataFrame({"down": p.down, "gsr": p.game_seconds_remaining, "yl": p.yardline_100, "fg": (p.play_type == "field_goal").astype(int), "kn": ((p.qb_kneel == 1) | (p.qb_spike == 1)).astype(int), "game": p.game_id}))
    return pd.concat(out, ignore_index=True)


def fgdown_sim():
    out = []
    for f in sorted(SIMD.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[(d.qtr == 4) & (d.gsr < 120) & (d.sd <= 0) & (d.sd >= -3) & d.code.isin([0, 1, 2, 3, 4, 5]) & (d.yl <= 45)]
        out.append(pd.DataFrame({"down": d.down, "gsr": d.gsr, "yl": d.yl, "fg": (d.code == 3).astype(int), "kn": d.code.isin([4, 5]).astype(int), "game": d.g}))
    return pd.concat(out, ignore_index=True)


def cmd_fgdown(a):
    L = []
    for nm, F in (("sim crzhc", fgdown_sim()), ("real pool", fgdown_real(POOL)), ("real late", fgdown_real(LATE))):
        L.append(f"== {nm}: Q4 <2:00, offence tied/trailing 0-3, yl<=45 snaps; FG rate and kneel/spike rate by down and time (snaps per game-unit n)")
        for lo, hi in ((0, 15), (15, 45), (45, 120)):
            for dn in (1, 2, 3, 4):
                r = F[(F.down == dn) & (F.gsr >= lo) & (F.gsr < hi)]
                if len(r) >= 20:
                    L.append(f"  gsr [{lo},{hi}) down {dn} n {len(r):4d} fg {r.fg.mean():.3f} kneel/spike {r.kn.mean():.3f}")
    (OUT / "fgdown.txt").write_text("\n".join(L))
    print("\n".join(L))


def cmd_an(a):
    xq = load_xq()
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    rng = np.random.default_rng(5)
    S = sim_play_drives()
    Rs = lead_rows(S)
    summarize(L, "sim crzhc", Rs)
    reals = {}
    for nm, seas in (("real pool 2009-17", POOL), ("real 2018-25", LATE)):
        D = real_play_drives(seas)
        Rr = lead_rows(D)
        summarize(L, nm, Rr)
        standardize(Rs, Rr, L)
        for col in ("p", "yl0", "n", "fd", "burn", "fga", "kneel"):
            g = boot_gap(Rr, Rs, col, rng)
            L.append(f"  gap real-sim {col}: {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}] pp {g[3]:.2f}")
        handoff(L, nm, D, xq)
    handoff(L, "sim crzhc", S, xq)
    (OUT / "an.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("an")
    sp.add_parser("state")
    sp.add_parser("fourth")
    sp.add_parser("clock")
    sp.add_parser("fgdown")
    args = ap.parse_args()
    {"an": cmd_an, "state": cmd_state, "fourth": cmd_fourth, "clock": cmd_clock, "fgdown": cmd_fgdown}[args.cmd](args)

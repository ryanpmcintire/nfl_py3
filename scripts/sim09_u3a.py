import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import mod25e_budget as bud  # noqa: E402
import mod25e_clock as clk  # noqa: E402
import sim09_dynamics as dyn  # noqa: E402
import sim09_urgency as urg  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u3a"
SRC = ["qb_dropback", "sack", "interception", "fumble_lost", "shotgun", "no_huddle", "air_yards", "qb_kneel", "qb_spike", "yards_gained"]
RESMAP = {"td": "Touchdown", "fg": "Field goal", "punt": "Punt", "turnover": "Turnover", "downs": "Turnover on downs", "fg_miss": "Missed field goal", "def_score": "Opp touchdown", "end_half": "End of half", "other": "Other"}


def s_play_season(task):
    bud.CAP["rows"].clear()
    res = dv.d_play_season(task)
    plays = res[3]
    sched = task[3]
    frames = []
    for gi, (lg, pl, tot, mar) in enumerate(bud.CAP["rows"]):
        assert len(lg) == len(pl)
        n = len(lg)
        kept = np.isin(lg[:, 6], (0, 1))
        pg = plays[plays["g"] == gi]
        assert kept.sum() == len(pg)
        assert np.array_equal(pg["offhome"], pl[kept, 0].astype(np.int8))
        epa = np.full(n, np.nan)
        yds = np.full(n, np.nan)
        epa[kept] = pg["epa"]
        yds[kept] = pg["yards"]
        week, h, a = sched[gi]
        frames.append(np.column_stack([np.full(n, gi), np.full(n, week), np.full(n, h), np.full(n, a), lg[:, :11], pl[:, 0], lg[:, 11], epa, yds]))
    cols = ["g", "week", "ht", "at", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "el", "offhome", "idx", "epa", "yards"]
    pd.DataFrame(np.concatenate(frames), columns=cols).to_parquet(bud.OUT / f"play_{task[0]}_{task[1]}.parquet")
    return res


def cmd_sim(args):
    import mod25_generator as gen

    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = bud.e_init
    gen.play_season = s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def source_map():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    at = dv.c25.attrs_from(pbp, tr)
    keys = tr[["game_id", "play_id"]].reset_index(drop=True)
    nv = pd.concat([pd.read_parquet(urg.NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "play_type"] + SRC) for s in range(min(dv.TRAIN), max(dv.TRAIN) + 1)], ignore_index=True)
    nv = nv.drop_duplicates(["game_id", "play_id"])
    src = keys.merge(nv, on=["game_id", "play_id"], how="left")
    return src, np.asarray(at["yards"], dtype=float)


def load_sim_frames(minsid):
    src, pool_yards = source_map()
    fs = sorted(bud.OUT.glob("play_*_*.parquet"))
    out = []
    for i, f in enumerate(fs):
        _, w, s = f.stem.split("_")
        if int(s) < minsid:
            continue
        d = pd.read_parquet(f)
        d["w"], d["s"], d["f"] = int(w), int(s), i
        out.append(d)
    P = pd.concat(out, ignore_index=True)
    P["g"] = P["f"].astype(np.int64) * 100000 + P["g"].astype(np.int64)
    idx = P["idx"].to_numpy(np.int64)
    kept = P["code"].isin([0, 1]).to_numpy()
    sx = src.iloc[idx].reset_index(drop=True)
    chk = np.isclose(sx["yards_gained"].to_numpy()[kept], pool_yards[idx[kept]], equal_nan=True)
    print("source rows joined", float(sx["game_id"].notna().mean()), "yards match", float(np.mean(chk)), flush=True)
    for c in SRC:
        P["src_" + c] = sx[c].to_numpy()
    P["flip"] = P["flip"].astype(bool)
    nx = P.groupby("g")["offhome"].shift(-1)
    P["flip"] = (nx.notna() & (nx != P["offhome"])) | P["flip"]
    tov = ((P["src_interception"].fillna(0) == 1) | (P["src_fumble_lost"].fillna(0) == 1)) & kept
    P["tov"] = tov.astype(int)
    return P


def to_pbp(P):
    D, R = dv.drive_table(P)
    R = R.copy()
    R["res"] = R["poss"].map(dict(zip(D["poss"], D["oc"].map(RESMAP))))
    kept = R["code"].isin([0, 1])
    h = (R["offhome"] == 1).to_numpy()
    hn = R["w"].astype(str) + "_" + R["ht"].astype(int).astype(str)
    an = R["w"].astype(str) + "_" + R["at"].astype(int).astype(str)
    d = pd.DataFrame({
        "game_id": R["g"].astype(np.int64).astype(str), "play_id": R.groupby("g").cumcount(), "season": R["w"] * 10 + R["s"], "week": R["week"].astype(int), "season_type": "REG",
        "posteam": np.where(h, hn, an), "defteam": np.where(h, an, hn), "home_team": hn, "away_team": an,
        "qtr": R["qtr"].astype(int), "game_seconds_remaining": R["gsr"], "score_differential": R["sd"],
        "play_type": np.where(R["code"] == 0, "run", np.where(R["code"] == 1, "pass", np.where(R["code"] == 2, "punt", "field_goal"))),
        "air_yards": R["src_air_yards"], "shotgun": R["src_shotgun"], "no_huddle": R["src_no_huddle"], "xpass": np.nan, "epa": R["epa"],
        "yards_gained": R["yards"], "sack": R["src_sack"], "interception": R["src_interception"], "fumble_lost": R["src_fumble_lost"],
        "qb_dropback": R["src_qb_dropback"], "qb_kneel": R["src_qb_kneel"].fillna(0), "qb_spike": R["src_qb_spike"].fillna(0), "drive": R["poss"], "fixed_drive": R["poss"],
        "fixed_drive_result": R["res"], "home_coach": np.nan, "away_coach": np.nan, "down": R["down"], "ydstogo": R["dist"], "yardline_100": R["yl"],
        "success": np.where(kept, (R["epa"] > 0).astype(float), np.nan), "passer_player_id": np.nan,
    })
    d["game_key"] = d["game_id"]
    return d.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def cmp_table(real, sim, keys, vcol="coef", secol="se"):
    m = real.merge(sim, on=keys, suffixes=("_real", "_sim"))
    m["diff"] = m[vcol + "_sim"] - m[vcol + "_real"]
    m["se_diff"] = np.sqrt(m[secol + "_sim"] ** 2 + m[secol + "_real"] ** 2)
    m["lo"] = m["diff"] - 1.96 * m["se_diff"]
    m["hi"] = m["diff"] + 1.96 * m["se_diff"]
    m["prob_pos"] = norm.cdf(m["diff"] / m["se_diff"])
    return m


def run_u1(d, hd):
    urg.OUT = OUT
    urg.OUTCOMES = [o for o in urg.OUTCOMES if o != "pass_oe"]
    u = d[(d.qtr <= 4) & d.posteam.notna() & d.score_differential.notna() & d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    u["t"] = u.game_seconds_remaining.clip(lower=0)
    u = urg.situation(urg.controls(u), hd)
    base, sl, ref = urg.fit_all(u)
    A = REPO / "artifacts" / "sim09"
    c1 = cmp_table(pd.read_csv(A / "response_slopes.csv"), sl, ["outcome", "term"])
    c2 = cmp_table(pd.read_csv(A / "ref_effects.csv"), ref, ["outcome", "ref"], "effect", "se")
    real_dr, dre = urg.drives(base)
    c3 = cmp_table(pd.read_csv(A / "drive_effects.csv"), dre, ["outcome", "ref"], "effect", "se")
    c1.to_csv(OUT / "cmp_slopes.csv", index=False)
    c2.to_csv(OUT / "cmp_refs.csv", index=False)
    c3.to_csv(OUT / "cmp_drive.csv", index=False)
    return c1, c2, c3, real_dr


def run_dyn(d):
    dyn.OUT = OUT
    dyn.nohuddle = lambda p: p
    dyn.by_era = lambda p, fn: [dict(r, era="fit") for r in fn(p)]
    dyn.HELD = (10**9, 10**9 + 1)
    p = dyn.prplays(d)
    t = dyn.drive_table(d)
    b = dyn.b1(p, t)
    e = dyn.d3(p, t)
    A = REPO / "artifacts" / "sim09"
    out = []
    for rr, ss in ((pd.read_csv(A / "dyn_b1.csv"), b), (pd.read_csv(A / "dyn_d3.csv"), e)):
        for era in ("fit", "held"):
            m = cmp_table(rr[rr.era == era].drop(columns="era"), ss.drop(columns="era"), ["term", "outcome"])
            m["real_era"] = era
            out.append(m)
    r = pd.concat(out)
    r.to_csv(OUT / "cmp_dyn.csv", index=False)
    return r


def per_game(num, den, g):
    a = pd.DataFrame({"n": num, "d": den, "g": g}).groupby("g")[["n", "d"]].sum()
    return a["n"].to_numpy(float), a["d"].to_numpy(float)


def game_boot(real, sim, reps, rng):
    nr, dr = per_game(*real)
    ns, ds = per_game(*sim)
    diffs = np.empty(reps)
    for k in range(reps):
        ir = rng.integers(0, len(nr), len(nr))
        js = rng.integers(0, len(ns), len(ns))
        diffs[k] = ns[js].sum() / ds[js].sum() - nr[ir].sum() / dr[ir].sum()
    pr, ps = nr.sum() / dr.sum(), ns.sum() / ds.sum()
    return pr, ps, ps - pr, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs > 0).mean())


def conv_frame(P):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    nd = P.groupby("g")["down"].shift(-1)
    conv = (P["yards"] >= P["dist"]) | ((nd == 1) & ~P["flip"])
    return P.assign(conv=conv.astype(float)), P["code"].isin([0, 1]) & P["down"].between(1, 4)


def run_fd_drive(Psim, reps):
    rng = np.random.default_rng(20261004)
    real = clk.load_real()["train"]
    rows = []
    cr, pr_ = conv_frame(real)
    cs, ps_ = conv_frame(Psim)
    sel = [("down1", lambda c: c["down"] == 1), ("down2", lambda c: c["down"] == 2), ("down3", lambda c: c["down"] == 3), ("down4", lambda c: c["down"] == 4),
           ("3rd&1", lambda c: (c["down"] == 3) & (c["dist"] == 1)), ("3rd 2-3", lambda c: (c["down"] == 3) & c["dist"].between(2, 3)), ("3rd 7+", lambda c: (c["down"] == 3) & (c["dist"] >= 7))]
    for nm, f in sel:
        mr, ms = pr_ & f(cr), ps_ & f(cs)
        rows.append(("first_down_" + nm,) + game_boot((cr.loc[mr, "conv"].to_numpy(), np.ones(mr.sum()), cr.loc[mr, "g"].to_numpy()), (cs.loc[ms, "conv"].to_numpy(), np.ones(ms.sum()), cs.loc[ms, "g"].to_numpy()), reps, rng))
    Dr, _ = dv.drive_table(real)
    Ds, _ = dv.drive_table(Psim)
    for nm, f in (("q3_leader_td", lambda D: (D["q0"] == 3) & (D["sd0"] > 0)), ("q4_leader_td", lambda D: (D["q0"] == 4) & (D["sd0"] > 0)), ("q3_trailer_td", lambda D: (D["q0"] == 3) & (D["sd0"] < 0)), ("q4_trailer_td", lambda D: (D["q0"] == 4) & (D["sd0"] < 0))):
        mr, ms = f(Dr), f(Ds)
        rows.append((nm,) + game_boot(((Dr.loc[mr, "oc"] == "td").to_numpy(float), np.ones(mr.sum()), Dr.loc[mr, "g"].to_numpy()), ((Ds.loc[ms, "oc"] == "td").to_numpy(float), np.ones(ms.sum()), Ds.loc[ms, "g"].to_numpy()), reps, rng))
    ar = Dr.groupby("g").size().to_numpy(float)
    as_ = Ds.groupby("g").size().to_numpy(float)
    dd = np.array([as_[rng.integers(0, len(as_), len(as_))].mean() - ar[rng.integers(0, len(ar), len(ar))].mean() for _ in range(reps)])
    rows.append(("drives_per_game", ar.mean(), as_.mean(), as_.mean() - ar.mean(), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), float((dd > 0).mean())))
    r = pd.DataFrame(rows, columns=["metric", "real", "sim", "diff", "lo", "hi", "prob_pos"])
    r.to_csv(OUT / "cmp_fd_drive.csv", index=False)
    return r


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_sim_frames(args.minsid)
    print("sim plays", len(P), "games", P["g"].nunique(), flush=True)
    d = to_pbp(P)
    gh = d[d.qtr == 3].groupby("game_id").first()
    hd = (gh["score_differential"] * np.where(gh["posteam"] == gh["home_team"], 1, -1)).rename("hd_home")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    c1, c2, c3, real_dr = run_u1(d, hd)
    print(c1[["outcome", "term", "coef_real", "coef_sim", "diff", "lo", "hi", "prob_pos"]].round(4).to_string(index=False))
    print(c2[c2.ref.isin(["q3_lead7", "final2_trail7", "q4start_trail10"])][["outcome", "ref", "effect_real", "effect_sim", "diff", "lo", "hi", "prob_pos"]].round(4).to_string(index=False))
    print(c3[["outcome", "ref", "effect_real", "effect_sim", "diff", "lo", "hi", "prob_pos"]].round(4).to_string(index=False))
    print(run_dyn(d)[["real_era", "term", "outcome", "coef_real", "coef_sim", "diff", "lo", "hi", "prob_pos"]].round(4).to_string(index=False))
    print(run_fd_drive(P, args.boot).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", dest="out_dir", default=str(REPO / "artifacts" / "sim09" / "u3a_play"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crz")
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    a = sub.add_parser("analyze")
    a.add_argument("--minsid", type=int, default=2)
    a.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    os.environ["BUD_OUT"] = str(Path(args.out_dir).resolve())
    bud.OUT = Path(os.environ["BUD_OUT"])
    {"sim": cmd_sim, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
